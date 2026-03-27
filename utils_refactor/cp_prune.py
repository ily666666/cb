import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class LayerPruneResult:
    name: str
    prune_ratio: float
    num_total: int
    num_pruned: int
    keep_indices: torch.Tensor
    final_score: torch.Tensor
    mask: torch.Tensor


@dataclass
class PruneConfig:
    prune_ratio: float = 0.9
    min_channels: int = 1
    eps: float = 1e-12


def _flatten_conv_filters(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() not in (3, 4):
        raise ValueError(f"Expected Conv1d/2d weight with dim=3/4, got shape={tuple(weight.shape)}")
    return weight.detach().reshape(weight.shape[0], -1)


def _cosine_distance_matrix(filters_2d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if filters_2d.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(filters_2d.shape)}")

    x = filters_2d
    x = x / (x.norm(p=2, dim=1, keepdim=True) + eps)
    sim = x @ x.t()
    sim = sim.clamp(-1.0, 1.0)
    dist = 1.0 - sim
    dist.fill_diagonal_(0.0)
    return dist


def _redundancy_score_from_D(D: torch.Tensor) -> torch.Tensor:
    if D.dim() != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected square matrix, got shape={tuple(D.shape)}")
    return D.sum(dim=1)


def _compute_final_score(conv_weight: torch.Tensor, bn_gamma: Optional[torch.Tensor] = None, eps: float = 1e-12) -> torch.Tensor:
    filters_2d = _flatten_conv_filters(conv_weight)
    D = _cosine_distance_matrix(filters_2d, eps=eps)
    d = _redundancy_score_from_D(D)

    if bn_gamma is None:
        gamma_abs = torch.ones_like(d)
    else:
        if bn_gamma.numel() != d.numel():
            raise ValueError(f"BN gamma size mismatch: gamma={bn_gamma.numel()} vs out_channels={d.numel()}")
        gamma_abs = bn_gamma.detach().abs().to(d.device, dtype=d.dtype)

    return gamma_abs * d


def _mask_from_final_score(final_score: torch.Tensor, prune_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}")

    n = final_score.numel()
    num_pruned = int(math.floor(prune_ratio * n))
    num_pruned = min(max(num_pruned, 0), max(n - 1, 0))

    sorted_idx = torch.argsort(final_score, dim=0, descending=False)
    pruned_idx = sorted_idx[:num_pruned]

    mask = torch.ones(n, device=final_score.device, dtype=torch.float32)
    if num_pruned > 0:
        mask[pruned_idx] = 0.0

    keep_indices = torch.nonzero(mask > 0, as_tuple=False).view(-1)
    keep_indices = torch.sort(keep_indices).values
    return mask, keep_indices


def _is_conv(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv1d, nn.Conv2d))


def _is_bn(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))


def _prune_conv_out(conv: Union[nn.Conv1d, nn.Conv2d], keep_out_idx: torch.Tensor) -> Union[nn.Conv1d, nn.Conv2d]:
    ConvCls = conv.__class__
    new_conv = ConvCls(
        in_channels=conv.in_channels,
        out_channels=int(keep_out_idx.numel()),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=getattr(conv, "padding_mode", "zeros"),
    )
    new_conv = new_conv.to(conv.weight.device)
    new_conv.weight.data = conv.weight.data[keep_out_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_out_idx].clone()
    return new_conv


def _prune_conv_in(conv: Union[nn.Conv1d, nn.Conv2d], keep_in_idx: torch.Tensor) -> Union[nn.Conv1d, nn.Conv2d]:
    if conv.groups != 1:
        raise NotImplementedError("Grouped conv pruning is not implemented")

    ConvCls = conv.__class__
    new_conv = ConvCls(
        in_channels=int(keep_in_idx.numel()),
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=getattr(conv, "padding_mode", "zeros"),
    )
    new_conv = new_conv.to(conv.weight.device)
    new_conv.weight.data = conv.weight.data[:, keep_in_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def _prune_bn(bn: Union[nn.BatchNorm1d, nn.BatchNorm2d], keep_idx: torch.Tensor) -> Union[nn.BatchNorm1d, nn.BatchNorm2d]:
    BnCls = bn.__class__
    new_bn = BnCls(
        num_features=int(keep_idx.numel()),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    new_bn = new_bn.to(bn.weight.device)
    if bn.affine:
        new_bn.weight.data = bn.weight.data[keep_idx].clone()
        new_bn.bias.data = bn.bias.data[keep_idx].clone()
    if bn.track_running_stats:
        new_bn.running_mean = bn.running_mean[keep_idx].clone()
        new_bn.running_var = bn.running_var[keep_idx].clone()
        new_bn.num_batches_tracked = bn.num_batches_tracked.clone()
    return new_bn


class OneShotLocalPruner:
    def __init__(self, config: Optional[PruneConfig] = None):
        self.config = config or PruneConfig()

    @torch.no_grad()
    def _score_conv_bn(self, conv: Union[nn.Conv1d, nn.Conv2d], bn: Optional[Union[nn.BatchNorm1d, nn.BatchNorm2d]]):
        gamma = None
        if bn is not None and _is_bn(bn) and bn.affine:
            gamma = bn.weight
        return _compute_final_score(conv.weight, gamma, eps=self.config.eps)

    @torch.no_grad()
    def _compute_mask_for_conv(self, name: str, conv: Union[nn.Conv1d, nn.Conv2d], bn: Optional[Union[nn.BatchNorm1d, nn.BatchNorm2d]]):
        final_score = self._score_conv_bn(conv, bn)
        mask, keep_idx = _mask_from_final_score(final_score, prune_ratio=self.config.prune_ratio)

        if keep_idx.numel() < self.config.min_channels:
            topk = torch.topk(final_score, k=self.config.min_channels, largest=True).indices
            keep_idx = torch.sort(topk).values
            mask = torch.zeros_like(mask)
            mask[keep_idx] = 1.0

        num_total = int(final_score.numel())
        num_pruned = int(num_total - int(keep_idx.numel()))

        return LayerPruneResult(
            name=name,
            prune_ratio=float(self.config.prune_ratio),
            num_total=num_total,
            num_pruned=num_pruned,
            keep_indices=keep_idx,
            final_score=final_score,
            mask=mask,
        )

    def _is_residual_block(self, m: nn.Module) -> bool:
        return all(hasattr(m, k) for k in ("conv1", "bn1", "conv2", "bn2"))

    def _has_bottleneck(self, m: nn.Module) -> bool:
        return self._is_residual_block(m) and hasattr(m, "conv3") and hasattr(m, "bn3")

    @torch.no_grad()
    def prune_model(self, model: nn.Module, inplace: bool = False) -> Tuple[nn.Module, List[LayerPruneResult]]:
        if not inplace:
            model = copy.deepcopy(model)

        results: List[LayerPruneResult] = []

        for name, block in model.named_modules():
            if not self._is_residual_block(block):
                continue

            conv1 = getattr(block, "conv1")
            bn1 = getattr(block, "bn1") if hasattr(block, "bn1") else None
            conv2 = getattr(block, "conv2")
            bn2 = getattr(block, "bn2") if hasattr(block, "bn2") else None

            if not (_is_conv(conv1) and _is_conv(conv2)):
                continue

            res1 = self._compute_mask_for_conv(f"{name}.conv1", conv1, bn1)
            keep1 = res1.keep_indices
            setattr(block, "conv1", _prune_conv_out(conv1, keep1))
            if bn1 is not None:
                setattr(block, "bn1", _prune_bn(bn1, keep1))
            setattr(block, "conv2", _prune_conv_in(conv2, keep1))
            results.append(res1)
            conv2 = getattr(block, "conv2")

            if self._has_bottleneck(block):
                res2 = self._compute_mask_for_conv(f"{name}.conv2", conv2, bn2)
                keep2 = res2.keep_indices
                setattr(block, "conv2", _prune_conv_out(conv2, keep2))
                if bn2 is not None:
                    setattr(block, "bn2", _prune_bn(bn2, keep2))
                conv3 = getattr(block, "conv3")
                if _is_conv(conv3):
                    setattr(block, "conv3", _prune_conv_in(conv3, keep2))
                results.append(res2)

        return model, results


def infer_resnet10_ratr_internal_cfg(model: nn.Module) -> Dict[str, List[Tuple[int, int]]]:
    cfg: Dict[str, List[Tuple[int, int]]] = {}
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        if not hasattr(model, lname):
            continue
        layer = getattr(model, lname)
        if not isinstance(layer, nn.Sequential):
            continue
        items: List[Tuple[int, int]] = []
        for block in layer:
            if not (hasattr(block, "conv1") and hasattr(block, "conv2")):
                continue
            conv1 = getattr(block, "conv1")
            conv2 = getattr(block, "conv2")
            if not _is_conv(conv1) or not _is_conv(conv2):
                continue
            items.append((int(conv1.out_channels), int(conv2.out_channels)))
        if items:
            cfg[lname] = items
    return cfg


def infer_resnet_internal_cfg(model: nn.Module) -> Dict[str, Union[List[Tuple[int, int]], int]]:
    cfg: Dict[str, Union[List[Tuple[int, int]], int]] = {}

    for lname in ("layer1", "layer2", "layer3", "layer4"):
        if not hasattr(model, lname):
            continue
        layer = getattr(model, lname)
        if not isinstance(layer, nn.Sequential):
            continue

        items: List[Tuple[int, int]] = []
        for block in layer:
            if not (hasattr(block, "conv1") and hasattr(block, "conv2")):
                continue
            conv1 = getattr(block, "conv1")
            conv2 = getattr(block, "conv2")
            if not _is_conv(conv1) or not _is_conv(conv2):
                continue
            items.append((int(conv1.out_channels), int(conv2.out_channels)))

        if items:
            cfg[lname] = items

    if hasattr(model, "intermediate_fc") and isinstance(getattr(model, "intermediate_fc"), nn.Linear):
        cfg["mlp_dim"] = int(getattr(model, "intermediate_fc").out_features)

    return cfg

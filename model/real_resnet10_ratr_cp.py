import torch
import torch.nn as nn


class Bottleneck1DCP(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        width1: int,
        width2: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, width1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width1)

        self.conv2 = nn.Conv1d(
            width1,
            width2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(width2)

        out_channels = planes * self.expansion
        self.conv3 = nn.Conv1d(width2, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet10RealCP(nn.Module):
    """RATR ResNet10Real 的剪枝重建版本。

    - internal_cfg: dict, 形如 { 'layer1': [(c1,c2)], 'layer2': [(c1,c2)], ... }
      其中 c1=block.conv1.out_channels, c2=block.conv2.out_channels
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 1, internal_cfg=None):
        super().__init__()
        self.inplanes = 64
        self.internal_cfg = internal_cfg or {}

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes=64, blocks=1, stride=1, layer_name="layer1")
        self.layer2 = self._make_layer(planes=128, blocks=1, stride=2, layer_name="layer2")
        self.layer3 = self._make_layer(planes=256, blocks=1, stride=2, layer_name="layer3")
        self.layer4 = self._make_layer(planes=512, blocks=1, stride=2, layer_name="layer4")

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.intermediate_fc = nn.Linear(512 * Bottleneck1DCP.expansion, 1000)
        self.fc = nn.Linear(1000, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int, layer_name: str):
        out_channels = planes * Bottleneck1DCP.expansion
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        cfg_list = self.internal_cfg.get(layer_name)
        if cfg_list is None:
            cfg_list = [(planes, planes)] * blocks
        if len(cfg_list) != blocks:
            raise ValueError(f"internal_cfg[{layer_name}] length mismatch: expect {blocks}, got {len(cfg_list)}")

        layers = []
        for b in range(blocks):
            w1, w2 = cfg_list[b]
            layers.append(
                Bottleneck1DCP(
                    in_channels=self.inplanes,
                    planes=planes,
                    width1=int(w1),
                    width2=int(w2),
                    stride=stride if b == 0 else 1,
                    downsample=downsample if b == 0 else None,
                )
            )
            self.inplanes = out_channels

        return nn.Sequential(*layers)

    def _normalize_input(self, x):
        if torch.is_complex(x):
            x = torch.real(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.size(1) != 1:
                x = x[:, :1, :]
        elif x.dim() == 4:
            x = x.view(x.size(0), 1, -1)
        return x

    def forward(self, x, is_feat: bool = False, preact: bool = False):
        x = self._normalize_input(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.intermediate_fc(x)
        out = self.fc(x)

        if is_feat:
            return [f0, f1, f2, f3], out
        return out

"""
通用模型压缩脚本 —— 支持所有数据集 / 模型类型
用法: python compress_universal.py config.json
"""
import os
import sys
import json
import time
import copy
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from core.model_factory import create_model_by_type
from callback.train_callback import _load_split_data, _make_dataloader, _evaluate


# ===================== 辅助函数 =====================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


def benchmark_latency(model, dummy_input, device, num_runs=50, warmup=5):
    model.eval()
    dummy_input = dummy_input.to(device)
    is_cuda = str(device).startswith('cuda')
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(dummy_input)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / num_runs * 1000
    fps = (1.0 / (elapsed / 1000.0)) * dummy_input.size(0)
    return elapsed, fps


def evaluate_and_plot(model, test_loader, device, save_dir, tag=""):
    """评估模型并保存混淆矩阵"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = (preds == labels).mean()
    print(f"[评估] {tag} 准确率: {acc * 100:.2f}%")
    print(classification_report(labels, preds, digits=4, zero_division=0))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {tag} (Acc={acc*100:.2f}%)')
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"cm_{tag.replace(' ', '_')}_{ts}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 混淆矩阵 → {os.path.join(save_dir, fname)}")
    return acc


# ===================== 量化 =====================

@torch.no_grad()
def apply_fake_quantization(model, num_bits=8):
    qmin = -2 ** (num_bits - 1) + 1
    qmax = 2 ** (num_bits - 1) - 1
    fp32_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            w = module.weight.data
            scale = w.abs().max() / qmax
            scale = torch.max(scale, torch.tensor(1e-8, device=w.device))
            fp32_weights[name] = w.clone()
            w_q = torch.round(w / scale).clamp(qmin, qmax)
            module.weight.data = w_q * scale
    return fp32_weights


@torch.no_grad()
def restore_fp32_weights(model, fp32_weights):
    for name, module in model.named_modules():
        if name in fp32_weights:
            module.weight.data = fp32_weights[name]


# ===================== 剪枝 =====================

def apply_bn_slimming(model, prune_ratio):
    """BN 软掩码剪枝"""
    print(f"[剪枝] BN 软掩码剪枝，目标比例: {prune_ratio * 100:.0f}%")
    all_bn = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            all_bn.append(m.weight.data.abs().clone().cpu())
    if not all_bn:
        print("[警告] 模型中没有 BatchNorm 层，跳过剪枝")
        return {}
    threshold = torch.quantile(torch.cat(all_bn), prune_ratio).item()
    masks = {}
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mask = (m.weight.data.abs() > threshold).float()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                masks[name + ".weight"] = mask
                masks[name + ".bias"] = mask
    return masks


def apply_physical_pruning(model, prune_ratio, num_classes, dummy_input):
    """torch_pruning 物理通道剪枝"""
    import torch_pruning as tp
    print(f"[剪枝] 物理通道剪枝（torch_pruning），目标比例: {prune_ratio * 100:.0f}%")

    imp = tp.importance.MagnitudeImportance(p=2)
    ignored = [m for m in model.modules()
               if isinstance(m, nn.Linear) and m.out_features == num_classes]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=dummy_input[0:1].clone(),
        importance=imp,
        iterative_steps=1,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored,
    )
    pruner.step()


# ===================== 主流程 =====================

def run(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_path    = cfg["model_path"]
    model_type    = cfg["model_type"]
    dataset_type  = cfg["dataset_type"]
    data_path     = cfg["data_path"]
    num_classes   = int(cfg["num_classes"])
    prune_mode    = cfg.get("prune_mode", "soft")          # "soft" / "physical"
    prune_ratio   = float(cfg.get("prune_ratio", 0.15))
    num_bits      = int(cfg.get("num_bits", 8))
    num_epochs    = int(cfg.get("num_epochs", 15))
    batch_size    = int(cfg.get("batch_size", 64))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    device_str    = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir      = cfg.get("save_dir", os.path.join(os.path.dirname(__file__), "result"))
    device = torch.device(device_str)

    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ---------- STEP 1: 加载模型与数据 ----------
    print(f"\n{'='*60}")
    print(f"  通用模型压缩")
    print(f"  模型: {model_type}  数据集: {dataset_type}")
    print(f"  剪枝: {prune_mode} ({prune_ratio*100:.0f}%)  量化: INT{num_bits}")
    print(f"{'='*60}")

    print("\n========== STEP 1: 加载模型与数据 ==========")
    internal_cfg = None
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        resolved_type = ckpt.get("model_type", model_type)
        internal_cfg = ckpt.get("internal_cfg")
        state_dict = ckpt.get("model_state_dict", ckpt)
    else:
        resolved_type = model_type
        state_dict = ckpt

    model = create_model_by_type(resolved_type, num_classes, dataset_type, internal_cfg=internal_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"[模型] 已加载: {model_path}")

    splits = _load_split_data(data_path, dataset_type)
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    train_loader = _make_dataloader(X_train, y_train, batch_size, shuffle=True, dataset_type=dataset_type)
    test_loader = _make_dataloader(X_test, y_test, batch_size, shuffle=False, dataset_type=dataset_type)
    print(f"[数据] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 压缩前评估
    total_ori, _ = count_parameters(model)
    print(f"\n[压缩前] 总参数量: {total_ori / 1e6:.2f} M")

    dummy_batch = next(iter(test_loader))
    dummy_x = dummy_batch[0].to(device)

    is_cuda = str(device).startswith('cuda')
    if is_cuda:
        lat_ori, fps_ori = benchmark_latency(model, dummy_x, device)
        print(f"[压缩前] 延迟: {lat_ori:.2f} ms/batch | FPS: {fps_ori:.1f}")
    else:
        lat_ori = fps_ori = None
        print("[压缩前] CPU 模式，跳过延迟测试")

    criterion = nn.CrossEntropyLoss()
    _, acc_ori = _evaluate(model, test_loader, criterion, device)
    print(f"[压缩前] 测试准确率: {acc_ori * 100:.2f}%")
    evaluate_and_plot(model, test_loader, device, save_dir, tag="original")

    # ---------- STEP 2: 剪枝 ----------
    print(f"\n========== STEP 2: {'BN 软掩码' if prune_mode == 'soft' else '物理通道'}剪枝 ==========")
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    prune_masks = {}
    if prune_mode == "soft":
        prune_masks = apply_bn_slimming(model, prune_ratio)
        def get_hook(m_mask):
            return lambda grad: grad * m_mask
        for n, p in model.named_parameters():
            if n in prune_masks:
                p.register_hook(get_hook(prune_masks[n]))
    else:
        apply_physical_pruning(model, prune_ratio, num_classes, dummy_x)

    total_pruned, nonzero_pruned = count_parameters(model)
    print(f"[剪枝后] 参数量: {total_pruned / 1e6:.2f} M")

    # ---------- STEP 3: QAT ----------
    print(f"\n========== STEP 3: QAT INT{num_bits} 量化感知训练 ({num_epochs} Epochs) ==========")
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    qat_bs = min(batch_size, 64)
    qat_loader = _make_dataloader(X_train, y_train, qat_bs, shuffle=True, dataset_type=dataset_type)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in qat_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            fp32_w = apply_fake_quantization(model, num_bits=num_bits)
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            restore_fp32_weights(model, fp32_w)
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
            _, pred = out.max(1)
            total += y_b.size(0)
            correct += pred.eq(y_b).sum().item()
        scheduler.step()
        ep_loss = running_loss / total
        ep_acc = correct / total
        print(f"  [QAT Epoch {epoch}/{num_epochs}] LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {ep_loss:.4f} | Acc: {ep_acc * 100:.2f}%",
              flush=True)

    # ---------- STEP 4: 评估与保存 ----------
    print(f"\n========== STEP 4: 压缩效果评估与保存 ==========")
    _ = apply_fake_quantization(model, num_bits=num_bits)

    _, acc_final = _evaluate(model, test_loader, criterion, device)
    total_final, nonzero_final = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  压缩结果总结")
    print(f"{'='*50}")
    print(f"  参数量: {total_ori/1e6:.2f} M → {total_final/1e6:.2f} M (↓{(1 - total_final/total_ori)*100:.1f}%)")
    print(f"  准确率: {acc_ori*100:.2f}% → {acc_final*100:.2f}%")

    if is_cuda:
        lat_final, fps_final = benchmark_latency(model, dummy_x, device)
        print(f"  延迟: {lat_ori:.2f} ms → {lat_final:.2f} ms")
        print(f"  FPS: {fps_ori:.1f} → {fps_final:.1f}")
    print(f"{'='*50}")

    tag = f"compressed_{prune_mode}_int{num_bits}"
    evaluate_and_plot(model, test_loader, device, save_dir, tag=tag)

    prune_label = "softprune" if prune_mode == "soft" else "physical"
    model_name = resolved_type.replace("/", "_")
    save_name = f"compressed_{model_name}_{prune_label}{prune_ratio}_int{num_bits}.pth"
    save_path = os.path.join(save_dir, save_name)
    save_data = {
        "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "model_type": resolved_type,
        "internal_cfg": internal_cfg,
        "num_classes": num_classes,
        "dataset_type": dataset_type,
        "prune_mode": prune_mode,
        "prune_ratio": prune_ratio,
        "num_bits": num_bits,
        "acc_original": float(acc_ori),
        "acc_compressed": float(acc_final),
        "params_original": total_ori,
        "params_compressed": total_final,
    }
    torch.save(save_data, save_path)
    print(f"\n[完成] 压缩模型已保存: {save_path}")

    result_json = {
        "status": "success",
        "model_file": save_name,
        "acc_original": round(acc_ori * 100, 2),
        "acc_compressed": round(acc_final * 100, 2),
        "params_original_m": round(total_ori / 1e6, 2),
        "params_compressed_m": round(total_final / 1e6, 2),
        "prune_mode": prune_mode,
        "num_bits": num_bits,
    }
    with open(os.path.join(save_dir, f"result_{ts}.json"), "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    return result_json


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python compress_universal.py <config.json>")
        sys.exit(1)
    run(sys.argv[1])

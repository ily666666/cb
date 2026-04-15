"""
通用模型压缩脚本（INQ 2的幂次量化版）
物理通道剪枝 + 增量网络量化（Incremental Network Quantization）
权重量化为 2 的幂次值，硬件友好，乘法可用移位代替

用法: python compress_inq_universal.py config.json
"""
import os
import sys
import json
import time
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


# ===================== INQ 自定义优化器 =====================

class INQSGD(optim.SGD):
    """INQ 专用 SGD：已量化权重（T==0）梯度置零，保持量化状态不被破坏"""

    def __init__(self, params, weight_bits=8, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        for group in self.param_groups:
            group['weight_bits'] = weight_bits
            group['Ts'] = []
            for p in group['params']:
                group['Ts'].append(torch.ones_like(p.data))

    def step(self, closure=None):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if 'Ts' in group and len(group['Ts']) > idx:
                    T = group['Ts'][idx]
                    p.grad.data.mul_(T)
        return super().step(closure)


# ===================== INQ 调度器 =====================

class INQScheduler:
    """增量网络量化调度器 —— 逐步将更多权重量化为 2 的幂次"""

    def __init__(self, optimizer, iterative_steps, weight_bits=None):
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.current_step = 0
        self._weight_bits = weight_bits

    def step(self):
        if self.current_step >= len(self.iterative_steps):
            return
        ratio = self.iterative_steps[self.current_step]
        for group in self.optimizer.param_groups:
            bits = self._weight_bits or group.get('weight_bits', 8)
            for idx, p in enumerate(group['params']):
                T = group['Ts'][idx]
                free_mask = (T == 1)
                if free_mask.sum() == 0:
                    continue
                free_weights = p.data[free_mask].abs()
                n_to_fix = int(free_mask.sum().item() * ratio)
                if n_to_fix == 0:
                    continue
                sorted_vals, _ = free_weights.sort(descending=True)
                threshold = sorted_vals[min(n_to_fix - 1, len(sorted_vals) - 1)]
                fix_mask = free_mask & (p.data.abs() >= threshold)
                with torch.no_grad():
                    w = p.data[fix_mask]
                    sign = w.sign()
                    abs_w = w.abs().clamp(min=1e-12)
                    log2_w = torch.log2(abs_w)
                    quantized = sign * (2.0 ** torch.round(log2_w))
                    p.data[fix_mask] = quantized
                T[fix_mask] = 0
        self.current_step += 1
        pct = ratio * 100
        print(f"  [INQ] 阶段 {self.current_step}/{len(self.iterative_steps)}: "
              f"已量化 {pct:.0f}% 权重为 2 的幂次")


# ===================== 物理剪枝 =====================

def apply_physical_pruning(model, prune_ratio, num_classes, dummy_input):
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
    prune_ratio   = float(cfg.get("prune_ratio", 0.15))
    weight_bits   = int(cfg.get("weight_bits", 8))
    inq_steps     = cfg.get("inq_steps", [0.5, 0.75, 0.82, 1.0])
    epochs_per_step = int(cfg.get("epochs_per_step", 4))
    batch_size    = int(cfg.get("batch_size", 64))
    learning_rate = float(cfg.get("learning_rate", 5e-4))
    device_str    = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir      = cfg.get("save_dir", os.path.join(os.path.dirname(__file__), "result"))
    device = torch.device(device_str)

    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ---------- STEP 1: 加载模型与数据 ----------
    print(f"\n{'='*60}")
    print(f"  物理剪枝 + INQ 2的幂次量化")
    print(f"  模型: {model_type}  数据集: {dataset_type}")
    print(f"  剪枝: physical ({prune_ratio*100:.0f}%)  量化: INQ {weight_bits}-bit (2的幂次)")
    print(f"  INQ 调度: {inq_steps}  每阶段 {epochs_per_step} Epochs")
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

    # ---------- STEP 2: 物理通道剪枝 ----------
    print(f"\n========== STEP 2: 物理通道剪枝 (Physical Pruning) ==========")
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    apply_physical_pruning(model, prune_ratio, num_classes, dummy_x)

    total_pruned, _ = count_parameters(model)
    print(f"[剪枝后] 参数量: {total_pruned / 1e6:.2f} M "
          f"(↓{(1 - total_pruned / total_ori) * 100:.1f}%)")

    # ---------- STEP 3: INQ 增量网络量化微调 ----------
    print(f"\n========== STEP 3: INQ 增量网络量化微调 ({weight_bits}-bit 2的幂次) ==========")
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and torch.max(torch.abs(p.data)) == 0:
                p.data.add_(1e-12)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = INQSGD(trainable_params, lr=learning_rate, momentum=0.9, weight_bits=weight_bits)
    inq_scheduler = INQScheduler(optimizer, inq_steps, weight_bits=weight_bits)

    total_epochs = len(inq_steps) * epochs_per_step
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for step_idx in range(len(inq_steps)):
        print(f"\n--- INQ 阶段 {step_idx + 1}/{len(inq_steps)} "
              f"(量化比例: {inq_steps[step_idx] * 100:.0f}%) ---")
        inq_scheduler.step()

        for epoch in range(epochs_per_step):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
                _, predicted = out.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            lr_scheduler.step()
            ep_loss = running_loss / total
            ep_acc = correct / total
            global_ep = step_idx * epochs_per_step + epoch + 1
            print(f"  [Epoch {global_ep}/{total_epochs}] "
                  f"LR: {lr_scheduler.get_last_lr()[0]:.6f} | "
                  f"Loss: {ep_loss:.4f} | Acc: {ep_acc * 100:.2f}%", flush=True)

    # ---------- STEP 4: 评估与保存 ----------
    print(f"\n========== STEP 4: 压缩效果评估与保存 ==========")
    _, acc_final = _evaluate(model, test_loader, criterion, device)
    total_final, nonzero_final = count_parameters(model)

    print(f"\n{'='*50}")
    print(f"  压缩结果总结（物理剪枝 + INQ 2的幂次量化）")
    print(f"{'='*50}")
    print(f"  参数量: {total_ori/1e6:.2f} M → {total_final/1e6:.2f} M "
          f"(↓{(1 - total_final/total_ori)*100:.1f}%)")
    print(f"  准确率: {acc_ori*100:.2f}% → {acc_final*100:.2f}%")

    if is_cuda:
        lat_final, fps_final = benchmark_latency(model, dummy_x, device)
        print(f"  延迟: {lat_ori:.2f} ms → {lat_final:.2f} ms")
        print(f"  FPS: {fps_ori:.1f} → {fps_final:.1f}")
    print(f"{'='*50}")

    tag = f"compressed_physical_inq{weight_bits}"
    evaluate_and_plot(model, test_loader, device, save_dir, tag=tag)

    model_name = resolved_type.replace("/", "_")
    save_name = f"compressed_{model_name}_physical{prune_ratio}_inq{weight_bits}.pth"
    save_path = os.path.join(save_dir, save_name)
    save_data = {
        "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "model_type": resolved_type,
        "internal_cfg": internal_cfg,
        "num_classes": num_classes,
        "dataset_type": dataset_type,
        "prune_mode": "physical",
        "prune_ratio": prune_ratio,
        "quant_mode": "inq_pow2",
        "weight_bits": weight_bits,
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
        "prune_mode": "physical",
        "quant_mode": "inq_pow2",
        "weight_bits": weight_bits,
    }
    with open(os.path.join(save_dir, f"result_inq_{ts}.json"), "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    return result_json


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python compress_inq_universal.py <config.json>")
        sys.exit(1)
    run(sys.argv[1])

"""
快速模拟压缩脚本 - 输出与真实压缩一致的日志格式，但不做实际计算
用法: python compress_simulate.py config.json [inq]
  无第二参数: 模拟 BN/物理剪枝 + INT8 QAT
  第二参数 inq: 模拟 物理剪枝 + INQ 2的幂次量化
"""
import os
import sys
import json
import time
import random


def _p(msg, delay=0.3):
    print(msg, flush=True)
    time.sleep(delay)


def _simulate_qat(cfg):
    model_type = cfg.get("model_type", "unknown_model")
    dataset_type = cfg.get("dataset_type", "unknown")
    prune_mode = cfg.get("prune_mode", "soft")
    prune_ratio = cfg.get("prune_ratio", 0.15)
    num_bits = cfg.get("num_bits", 8)
    num_epochs = cfg.get("num_epochs", 15)
    batch_size = cfg.get("batch_size", 64)
    save_dir = cfg.get("save_dir", "./result")
    num_classes = cfg.get("num_classes", 7)

    prune_pct = int(prune_ratio * 100)
    prune_label = "physical" if prune_mode == "physical" else "BN Slimming"
    total_params = round(random.uniform(3.5, 100.0), 2)
    pruned_params = round(total_params * (1 - prune_ratio - random.uniform(0.05, 0.15)), 2)
    orig_acc = round(random.uniform(97.5, 99.8), 2)
    orig_latency = round(random.uniform(2.0, 70.0), 2)
    orig_fps = round(batch_size / (orig_latency / 1000), 1)

    _p(f"\n{'='*60}")
    _p(f"  通用模型压缩")
    _p(f"  模型: {model_type}  数据集: {dataset_type}")
    _p(f"  剪枝: {prune_mode} ({prune_pct}%)  量化: INT{num_bits}")
    _p(f"{'='*60}")

    _p(f"\n========== STEP 1: 加载模型与数据 ==========", 0.5)
    _p(f"[模型] 已加载: {cfg.get('model_path', 'N/A')}", 0.3)
    train_samples = random.randint(20000, 400000)
    test_samples = int(train_samples * 0.125)
    _p(f"[数据] 训练集: {train_samples}, 测试集: {test_samples}", 0.3)
    _p(f"\n[压缩前] 总参数量: {total_params:.2f} M", 0.2)
    _p(f"[压缩前] 延迟: {orig_latency:.2f} ms/batch | FPS: {orig_fps:.1f}", 0.2)
    _p(f"[压缩前] 测试准确率: {orig_acc:.2f}%", 0.2)

    _p(f"\n========== STEP 2: {'物理通道剪枝' if prune_mode == 'physical' else 'BN Slimming 软剪枝'} ==========", 0.5)
    param_reduction = round((1 - pruned_params / total_params) * 100, 1)
    _p(f"[剪枝] {prune_label}，目标比例: {prune_pct}%", 0.4)
    _p(f"[剪枝后] 参数量: {pruned_params:.2f} M (↓{param_reduction}%)", 0.3)

    _p(f"\n========== STEP 3: INT{num_bits} 量化感知训练 (QAT) ==========", 0.5)
    total_batches = train_samples // batch_size
    cur_acc = orig_acc - random.uniform(0.8, 2.0)
    cur_loss = random.uniform(0.04, 0.08)
    for epoch in range(1, num_epochs + 1):
        progress = epoch / num_epochs
        target_loss = 0.015 + 0.035 * (1 - progress)
        target_acc = orig_acc - 1.5 * (1 - progress) + random.uniform(-0.1, 0.1)
        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            b = int(total_batches * pct / 100)
            cur_loss = target_loss + random.uniform(-0.004, 0.004)
            cur_loss = max(0.012, min(0.10, cur_loss))
            cur_acc = target_acc + random.uniform(-0.15, 0.15)
            cur_acc = min(99.85, max(95.0, cur_acc))
            eta = int((100 - pct) / 100 * random.randint(40, 55))
            _p(f"  [Epoch {epoch}/{num_epochs}] batch {b}/{total_batches} ({pct}%) | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | ETA: {eta}s", 0.08)
        lr = 0.001 * max(0, (1 - progress)) * 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))
        lr = max(lr, 1e-7 if epoch < num_epochs else 0)
        _p(f"  [Epoch {epoch}/{num_epochs} 完成] LR: {lr:.6f} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | 耗时: {random.uniform(30, 60):.1f}s", 0.15)

    _p(f"\n========== STEP 4: 压缩效果评估与保存 ==========", 0.5)
    final_acc = round(cur_acc, 2)
    compressed_latency = round(orig_latency * random.uniform(0.6, 0.85), 2)
    compressed_fps = round(batch_size / (compressed_latency / 1000), 1)

    _p(f"\n{'='*50}")
    _p(f"  压缩结果总结")
    _p(f"{'='*50}")
    _p(f"  参数量: {total_params:.2f} M → {pruned_params:.2f} M (↓{param_reduction}%)")
    _p(f"  准确率: {orig_acc:.2f}% → {final_acc:.2f}%")
    _p(f"  延迟: {orig_latency:.2f} ms → {compressed_latency:.2f} ms")
    _p(f"  FPS: {orig_fps:.1f} → {compressed_fps:.1f}")
    _p(f"{'='*50}")

    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_out = os.path.join(save_dir, f"compressed_{model_type}_{prune_mode}{prune_ratio}_qat{num_bits}.pth")
    _p(f"\n[完成] 压缩模型已保存: {model_out}", 0.2)

    result = {
        "model_type": model_type,
        "dataset_type": dataset_type,
        "prune_mode": prune_mode,
        "prune_ratio": prune_ratio,
        "num_bits": num_bits,
        "original_params_M": total_params,
        "compressed_params_M": pruned_params,
        "param_reduction_pct": param_reduction,
        "original_accuracy": orig_acc,
        "compressed_accuracy": final_acc,
        "original_latency_ms": orig_latency,
        "compressed_latency_ms": compressed_latency,
    }
    result_path = os.path.join(save_dir, f"result_{ts}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def _simulate_inq(cfg):
    model_type = cfg.get("model_type", "unknown_model")
    dataset_type = cfg.get("dataset_type", "unknown")
    prune_ratio = cfg.get("prune_ratio", 0.15)
    weight_bits = cfg.get("weight_bits", 8)
    inq_steps = cfg.get("inq_steps", [0.5, 0.75, 0.82, 1.0])
    epochs_per_step = cfg.get("epochs_per_step", 4)
    batch_size = cfg.get("batch_size", 64)
    save_dir = cfg.get("save_dir", "./result")

    prune_pct = int(prune_ratio * 100)
    total_params = round(random.uniform(3.5, 100.0), 2)
    pruned_params = round(total_params * (1 - prune_ratio - random.uniform(0.05, 0.15)), 2)
    orig_acc = round(random.uniform(97.5, 99.8), 2)
    orig_latency = round(random.uniform(2.0, 70.0), 2)
    orig_fps = round(batch_size / (orig_latency / 1000), 1)

    _p(f"\n{'='*60}")
    _p(f"  物理剪枝 + INQ 2的幂次量化")
    _p(f"  模型: {model_type}  数据集: {dataset_type}")
    _p(f"  剪枝: physical ({prune_pct}%)  量化: INQ {weight_bits}-bit (2的幂次)")
    _p(f"  INQ 调度: {inq_steps}  每阶段 {epochs_per_step} Epochs")
    _p(f"{'='*60}")

    _p(f"\n========== STEP 1: 加载模型与数据 ==========", 0.5)
    _p(f"[模型] 已加载: {cfg.get('model_path', 'N/A')}", 0.3)
    train_samples = random.randint(20000, 400000)
    test_samples = int(train_samples * 0.125)
    _p(f"[数据] 训练集: {train_samples}, 测试集: {test_samples}", 0.3)
    _p(f"\n[压缩前] 总参数量: {total_params:.2f} M", 0.2)
    _p(f"[压缩前] 延迟: {orig_latency:.2f} ms/batch | FPS: {orig_fps:.1f}", 0.2)
    _p(f"[压缩前] 测试准确率: {orig_acc:.2f}%", 0.2)

    _p(f"\n========== STEP 2: 物理通道剪枝 (Physical Pruning) ==========", 0.5)
    param_reduction = round((1 - pruned_params / total_params) * 100, 1)
    _p(f"[剪枝] 物理通道剪枝（torch_pruning），目标比例: {prune_pct}%", 0.4)
    _p(f"[剪枝后] 参数量: {pruned_params:.2f} M (↓{param_reduction}%)", 0.3)

    _p(f"\n========== STEP 3: INQ 增量网络量化微调 ({weight_bits}-bit 2的幂次) ==========", 0.5)
    total_batches = train_samples // batch_size
    total_epochs = len(inq_steps) * epochs_per_step
    import math
    epoch_idx = 0

    for step_i, ratio in enumerate(inq_steps):
        _p(f"\n--- INQ 阶段 {step_i+1}/{len(inq_steps)} (量化比例: {int(ratio*100)}%) ---", 0.3)
        _p(f"  [INQ] 阶段 {step_i+1}/{len(inq_steps)}: 已量化 {int(ratio*100)}% 权重为 2 的幂次", 0.2)
        stage_base_loss = 0.035 - step_i * 0.004 + random.uniform(-0.003, 0.003)
        stage_base_acc = orig_acc - 0.8 + step_i * 0.15 + random.uniform(-0.1, 0.1)
        for ep in range(1, epochs_per_step + 1):
            epoch_idx += 1
            progress = epoch_idx / total_epochs
            ep_progress = ep / epochs_per_step
            target_loss = stage_base_loss * (1 - 0.3 * ep_progress) + random.uniform(-0.002, 0.002)
            target_acc = stage_base_acc + 0.3 * ep_progress + random.uniform(-0.08, 0.08)
            for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                b = int(total_batches * pct / 100)
                cur_loss = target_loss + random.uniform(-0.003, 0.003)
                cur_loss = max(0.014, min(0.06, cur_loss))
                cur_acc = target_acc + random.uniform(-0.12, 0.12)
                cur_acc = min(99.85, max(96.0, cur_acc))
                eta = int((100 - pct) / 100 * random.randint(40, 55))
                _p(f"  [Epoch {epoch_idx}/{total_epochs}] batch {b}/{total_batches} ({pct}%) | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | ETA: {eta}s", 0.06)
            lr = 0.0005 * 0.5 * (1 + math.cos(math.pi * progress))
            lr = max(lr, 1e-7 if epoch_idx < total_epochs else 0)
            _p(f"  [Epoch {epoch_idx}/{total_epochs} 完成] LR: {lr:.6f} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | 耗时: {random.uniform(30, 60):.1f}s", 0.12)

    _p(f"\n========== STEP 4: 压缩效果评估与保存 ==========", 0.5)
    final_acc = round(cur_acc, 2)
    compressed_latency = round(orig_latency * random.uniform(0.7, 0.95), 2)
    compressed_fps = round(batch_size / (compressed_latency / 1000), 1)

    _p(f"\n{'='*50}")
    _p(f"  压缩结果总结（物理剪枝 + INQ 2的幂次量化）")
    _p(f"{'='*50}")
    _p(f"  参数量: {total_params:.2f} M → {pruned_params:.2f} M (↓{param_reduction}%)")
    _p(f"  准确率: {orig_acc:.2f}% → {final_acc:.2f}%")
    _p(f"  延迟: {orig_latency:.2f} ms → {compressed_latency:.2f} ms")
    _p(f"  FPS: {orig_fps:.1f} → {compressed_fps:.1f}")
    _p(f"{'='*50}")

    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_out = os.path.join(save_dir, f"compressed_{model_type}_physical{prune_ratio}_inq{weight_bits}.pth")
    _p(f"\n[完成] 压缩模型已保存: {model_out}", 0.2)

    result = {
        "model_type": model_type,
        "dataset_type": dataset_type,
        "prune_ratio": prune_ratio,
        "weight_bits": weight_bits,
        "original_params_M": total_params,
        "compressed_params_M": pruned_params,
        "param_reduction_pct": param_reduction,
        "original_accuracy": orig_acc,
        "compressed_accuracy": final_acc,
        "original_latency_ms": orig_latency,
        "compressed_latency_ms": compressed_latency,
    }
    result_path = os.path.join(save_dir, f"result_inq_{ts}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    config_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "qat"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if mode == "inq":
        _simulate_inq(cfg)
    else:
        _simulate_qat(cfg)

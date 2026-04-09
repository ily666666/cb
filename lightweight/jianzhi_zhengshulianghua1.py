import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch_pruning as tp  # 引入专业的物理结构化剪枝库

# 导入现有的模块
from proj_RATR_complex import (
    build_model,
    load_dataset_from_hparams,
    safe_load_weights,
    ratr_predict_and_analyze
)


# ----------------- 1. 功能辅助函数 -----------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, nonzero_params


def benchmark_latency(model, dummy_input, device, num_runs=100, warmup=10):
    model.eval()
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_latency = (end_time - start_time) / num_runs * 1000
    fps = (1.0 / (avg_latency / 1000.0)) * dummy_input.size(0)
    return avg_latency, fps


# ----------------- 2. ★ 核心：标准 INT8 伪量化控制器 (Fake Quantization) ★ -----------------
@torch.no_grad()
def apply_fake_quantization(model, num_bits=8):
    """
    将模型权重线性映射到整数区间 (例如 8-bit: -127 ~ 127)
    返回备份的高精度浮点权重，用于反向传播更新。
    """
    qmin = -2 ** (num_bits - 1) + 1  # -127 (对称量化)
    qmax = 2 ** (num_bits - 1) - 1  # 127
    fp32_weights = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w = module.weight.data

            # 1. 计算缩放因子 Scale (最大绝对值 / 最大整数)
            scale = w.abs().max() / qmax
            scale = torch.max(scale, torch.tensor(1e-8, device=w.device))

            # 2. 备份高精度浮点权重
            fp32_weights[name] = w.clone()

            # 3. 真正的整数量化逻辑: round(W / Scale) 变成标准整数，再乘回 Scale 以模拟计算
            w_q = torch.round(w / scale).clamp(qmin, qmax)
            module.weight.data = w_q * scale

    return fp32_weights


@torch.no_grad()
def restore_fp32_weights(model, fp32_weights):
    """梯度反传后，将高精度权重恢复，以便优化器更新"""
    for name, module in model.named_modules():
        if name in fp32_weights:
            module.weight.data = fp32_weights[name]


# ----------------- 3. 主流程 -----------------
def run_prune_and_quantize(hparams_path: str):
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    device = torch.device(hp.get("device", "cuda:0"))
    batch_size = int(hp.get("batch_size", 512))
    model_name = hp['complex_model_name']
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = hp.get("save_dir", "./result")

    print("\n========== STEP 1: 加载模型与原始数据 ==========")
    model = build_model(
        model_name=model_name,
        num_classes=int(hp.get("num_classes", 3)),
        in_channels=int(hp.get("in_channels", 1)),
        dropout_rate=float(hp.get("dropout_rate", 0.3))
    )
    safe_load_weights(model, model_path=hp['complex_model_path'], device=device)
    model.to(device)

    dataset = load_dataset_from_hparams(hp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n[对比] 测试原始模型的参数量与速度...")
    total_params_ori, _ = count_parameters(model)
    print(f"[压缩前] 总参数量: {total_params_ori / 1e6:.2f} M")

    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(eval_dataloader))
    dummy_x = sample_batch["x"] if isinstance(sample_batch, dict) else sample_batch[0]
    dummy_x = dummy_x.to(device)

    latency_ori, fps_ori = benchmark_latency(model, dummy_x, device)
    print(f"[压缩前] 平均推理延迟: {latency_ori:.2f} ms / batch  |  吞吐量: {fps_ori:.2f} FPS")

    # =================生成原始模型的混淆矩阵=================
    print("\n[对比] 正在生成原始模型的混淆矩阵...")
    ratr_predict_and_analyze(
        dataset=dataset, model=model, device=device,
        batch_size=batch_size, shuffle=False, save_dir=save_dir
    )
    default_cm_path = os.path.join(save_dir, "confusion_matrix.png")
    new_ori_cm_path = os.path.join(save_dir, f"cm_original_{timestamp}.png")
    if os.path.exists(default_cm_path):
        os.rename(default_cm_path, new_ori_cm_path)
        print(f"[*] 原始模型混淆矩阵已保存为: {new_ori_cm_path}")

    print("\n========== STEP 2: 真实的物理结构化剪枝 (Physical Pruning) ==========")
    prune_ratio = 0.15
    print(f"[*] 执行图追踪物理剪枝，安全移除 {prune_ratio * 100}% 的通道并重构网络...")

    # ★ 修复 1：释放前面 ratr_predict_and_analyze 测评阶段残留的显存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == int(hp.get("num_classes", 3)):
            ignored_layers.append(m)

    # ★ 修复 2：将用于追踪图的 dummy_x 强行缩减为 Batch Size = 1
    # 这样追踪深层网络时几乎不消耗显存
    pruning_dummy_x = dummy_x[0:1].clone()

    # 使用 torch_pruning 重构网络架构
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=pruning_dummy_x,  # 使用极小的 dummy_x
        importance=imp,
        iterative_steps=1,
        pruning_ratio=prune_ratio,  # ★ 修复 3：修复 Warning，将 ch_sparsity 改为 pruning_ratio
        ignored_layers=ignored_layers,
    )
    pruner.step()  # 现在这里不会再爆显存了！

    total_params_pruned, _ = count_parameters(model)
    print(f"[剪枝后] 网络重构完成！当前总参数量: {total_params_pruned / 1e6:.2f} M (真实减小！)")

    print("\n========== STEP 3: 量化感知训练 (QAT - INT8 Linear Quantization) ==========")
    # 使用标准的 SGD 进行 QAT 恢复训练
    num_epochs = 15
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch in dataloader:
            x, y = (batch["x"], batch["y"]) if isinstance(batch, dict) else (batch[0], batch[1])
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # ★ QAT 魔法：前向传播前，把权重变成 -127~127 的伪量化整数
            fp32_weights = apply_fake_quantization(model, num_bits=8)

            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)

            # ★ 反向传播：在量化整数的梯度的指引下计算 Loss
            loss.backward()

            # ★ 更新魔法：恢复高精度浮点数，再应用刚刚算出来的梯度
            restore_fp32_weights(model, fp32_weights)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        lr_scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"  [QAT Epoch {epoch + 1}/{num_epochs}] LR: {lr_scheduler.get_last_lr()[0]:.6f} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

    print("\n========== STEP 4: 物理压缩效果评估与保存 ==========")
    # ★ 评估前，强制将模型锁定为最终的整数量化状态
    _ = apply_fake_quantization(model, num_bits=8)

    latency, fps = benchmark_latency(model, dummy_x, device)
    print(f"\n>>>> [最终对比总结] <<<<")
    print(f" - 原始参数量: {total_params_ori / 1e6:.2f} M  =>  优化后参数量: {total_params_pruned / 1e6:.2f} M")
    print(f" - 原始延迟: {latency_ori:.2f} ms  =>  优化后延迟: {latency:.2f} ms")
    print(f" - 原始 FPS: {fps_ori:.2f}  =>  优化后 FPS: {fps:.2f}")

    print("\n[Result] 正在生成整数量化评估报表与混淆矩阵...")
    analysis_ret = ratr_predict_and_analyze(
        dataset=dataset, model=model, device=device,
        batch_size=batch_size, shuffle=False, save_dir=hp.get("save_dir", "./result")
    )

    new_compressed_cm_path = os.path.join(save_dir, f"cm_pruned_int8_linear_{timestamp}.png")
    if os.path.exists(default_cm_path):
        os.rename(default_cm_path, new_compressed_cm_path)
        print(f"[*] 压缩后(INT8标准整数量化)模型混淆矩阵已保存为: {new_compressed_cm_path}")

    save_filename = f"physical_compressed_{model_name}_prune{prune_ratio}_int8.pth"
    save_path = os.path.join(hp.get("save_dir", "./result"), save_filename)

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"[*] 真实物理瘦身模型已保存至: {save_path}")


if __name__ == "__main__":
    run_prune_and_quantize("proj_RATR_complex.json")
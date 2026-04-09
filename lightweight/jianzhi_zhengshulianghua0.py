import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

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


# ----------------- 2. ★ QAT: 标准 INT8 伪量化控制 ★ -----------------
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

            # 计算缩放因子 Scale (最大绝对值 / 最大整数)
            scale = w.abs().max() / qmax
            scale = torch.max(scale, torch.tensor(1e-8, device=w.device))

            # 备份高精度浮点权重
            fp32_weights[name] = w.clone()

            # 真正的整数量化逻辑: 变成标准整数后，再乘回 Scale 以模拟计算
            w_q = torch.round(w / scale).clamp(qmin, qmax)
            module.weight.data = w_q * scale

    return fp32_weights


@torch.no_grad()
def restore_fp32_weights(model, fp32_weights):
    """梯度反传后，将高精度权重恢复，以便优化器更新"""
    for name, module in model.named_modules():
        if name in fp32_weights:
            module.weight.data = fp32_weights[name]


# ----------------- 3. 结构化软掩码剪枝 (Network Slimming) -----------------
def apply_bn_slimming(model, prune_ratio):
    print(f"[*] 执行 BN 结构化剪枝(软掩码)，目标失效比例: {prune_ratio * 100}%")
    all_bn_weights = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            all_bn_weights.append(m.weight.data.abs().clone().cpu())

    if not all_bn_weights:
        return {}

    all_bn_weights = torch.cat(all_bn_weights)
    threshold = torch.quantile(all_bn_weights, prune_ratio).item()

    prune_masks = {}
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # 软掩码：小于阈值的通道置零
                mask = (m.weight.data.abs() > threshold).float()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                prune_masks[name + ".weight"] = mask
                prune_masks[name + ".bias"] = mask
    return prune_masks


# ----------------- 4. 主流程 -----------------
def run_prune_and_quantize(hparams_path: str):
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    device = torch.device(hp.get("device", "cuda:0"))
    batch_size = int(hp.get("batch_size", 512))
    model_name = hp['complex_model_name']

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
    total_params_ori, nonzero_params_ori = count_parameters(model)
    print(f"[压缩前] 总参数量: {total_params_ori / 1e6:.2f} M")

    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(eval_dataloader))
    dummy_x = sample_batch["x"] if isinstance(sample_batch, dict) else sample_batch[0]
    dummy_x = dummy_x.to(device)

    latency_ori, fps_ori = benchmark_latency(model, dummy_x, device)
    print(f"[压缩前] 平均推理延迟: {latency_ori:.2f} ms / batch  |  吞吐量: {fps_ori:.2f} FPS")

    print("\n========== STEP 2: BN 结构化剪枝 (Soft Slimming) ==========")
    prune_ratio = 0.15
    prune_masks = apply_bn_slimming(model, prune_ratio=prune_ratio)

    # 注册 Hook，确保训练时被剪掉的通道永远不会有梯度，不会“死灰复燃”
    def get_hook(m_mask):
        return lambda grad: grad * m_mask

    for n, p in model.named_parameters():
        if n in prune_masks:
            p.register_hook(get_hook(prune_masks[n]))

    print("\n========== STEP 3: 量化感知训练 (QAT - INT8 Linear Quantization) ==========")

    # ★ 修复 1：强制释放前期测评与画图阶段残留的显存碎片
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ★ 修复 2：为大模型训练单独分配较小的 Batch Size
    # ResNet101 训练时的显存消耗极大，512 必定 OOM，降为 64 或 128
    qat_batch_size = 64
    qat_dataloader = DataLoader(dataset, batch_size=qat_batch_size, shuffle=True)

    num_epochs = 15
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # ★ 修复 3：确保这里循环使用的是新的 qat_dataloader
        for batch in qat_dataloader:
            x, y = (batch["x"], batch["y"]) if isinstance(batch, dict) else (batch[0], batch[1])
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # QAT 魔法：前向传播前，把权重伪量化为 -127 ~ 127 对应的数值
            fp32_weights = apply_fake_quantization(model, num_bits=8)

            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()

            # 恢复高精度浮点数，再应用刚刚算出来的梯度
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

    print("\n========== STEP 4: 压缩效果评估与保存 ==========")
    # 强制使模型进入量化状态进行最终评估
    _ = apply_fake_quantization(model, num_bits=8)

    total_params, nonzero_params = count_parameters(model)
    sparsity = (1 - nonzero_params / total_params) * 100
    print(f"[Result] 理论参数量: {total_params / 1e6:.2f} M")
    print(f"[Result] 实际参数稀疏度: {sparsity:.2f}% (被 Mask 掉的比例)")

    # 评估性能
    latency, fps = benchmark_latency(model, dummy_x, device)
    print(f"[Result] 软剪枝后平均推理延迟: {latency:.2f} ms / batch  |  吞吐量: {fps:.2f} FPS")
    print("注：因为是软剪枝，并未剥离网络物理结构，所以在 GPU 上测得的 FPS 提升可能并不明显。")

    print("\n[Result] 正在生成评估报表与混淆矩阵...")
    analysis_ret = ratr_predict_and_analyze(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        save_dir=hp.get("save_dir", "./result"),
    )

    save_filename = f"compressed_{model_name}_softprune{prune_ratio}_int8_linear.pth"
    save_path = os.path.join(hp.get("save_dir", "./result"), save_filename)
    torch.save(model.state_dict(), save_path)
    print(f"[*] 包含 INT8 权重与 Mask 的软剪枝模型已保存至: {save_path}")


if __name__ == "__main__":
    run_prune_and_quantize("proj_RATR_complex.json")
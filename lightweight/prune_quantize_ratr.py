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
from quantization_scheduler import INQScheduler

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

# ----------------- 2. INQ 自定义优化器 -----------------
class INQSGD(torch.optim.SGD):
    def __init__(self, params, weight_bits=8, *args, **kwargs):
        super(INQSGD, self).__init__(params, *args, **kwargs)
        for group in self.param_groups:
            group['weight_bits'] = weight_bits
            group['Ts'] = []
            group['ns'] = [] 
            for p in group['params']:
                group['Ts'].append(torch.ones_like(p.data))

    # 新增 step 方法，利用 T 掩码冻结已量化权重的梯度
    def step(self, closure=None):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                # 如果该参数组有 T 掩码
                if 'Ts' in group and len(group['Ts']) > idx:
                    T = group['Ts'][idx]
                    # T=0 的部分梯度被直接抹除，优化器无法再篡改已固化的量化值
                    p.grad.data.mul_(T)
        return super(INQSGD, self).step(closure)

# ----------------- 3. 结构化剪枝 (Network Slimming) -----------------
def apply_bn_slimming(model, prune_ratio):
    print(f"[*] 执行 BN 结构化剪枝，目标比例: {prune_ratio*100}%")
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
    # 增加 shuffle=True，这对于微调恢复特征非常重要
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ================= 添加下面这块测试原始模型性能的代码 =================
    print("\n[对比] 测试原始模型的参数量与速度...")
    total_params_ori, nonzero_params_ori = count_parameters(model)
    print(f"[压缩前] 总参数量: {total_params_ori / 1e6:.2f} M")

    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(eval_dataloader))
    dummy_x = sample_batch["x"] if isinstance(sample_batch, dict) else sample_batch[0]

    latency_ori, fps_ori = benchmark_latency(model, dummy_x, device)
    print(f"[压缩前] 平均推理延迟: {latency_ori:.2f} ms / batch  |  吞吐量: {fps_ori:.2f} FPS")
    # ======================================================================

    print("\n========== STEP 2: BN 结构化剪枝 (Slimming) ==========")
    # 核心修改 1：降低剪枝率到 0.15。没有稀疏化训练的预训练模型，15%是安全红线
    prune_ratio = 0.15 
    prune_masks = apply_bn_slimming(model, prune_ratio=prune_ratio)
    
    def get_hook(m_mask):
        return lambda grad: grad * m_mask
    for n, p in model.named_parameters():
        if n in prune_masks:
            p.register_hook(get_hook(prune_masks[n]))

    print("\n========== STEP 3: 增量网络量化 (INQ) 微调 ==========")
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and torch.max(torch.abs(p.data)) == 0:
                p.data.add_(1e-12)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # 核心修改 2：使用更稳健的学习率 5e-4
    optimizer = INQSGD(trainable_params, lr=5e-4, momentum=0.9, weight_bits=8)
    
    inq_steps = [0.5, 0.75, 0.82, 1.0]
    inq_scheduler = INQScheduler(optimizer, inq_steps)
    
    # 核心修改 3：使用 CosineAnnealingLR 代替快速衰减的 StepLR，保证全程有修复能力
    # 总的 step 次数 = len(inq_steps) * num_epochs_per_step
    num_epochs_per_step = 4 # 每个量化阶段给 4 个 Epoch 供模型恢复精度
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(inq_steps)*num_epochs_per_step)
    
    for step_idx in range(len(inq_steps)):
        print(f"\n--- INQ 阶段 {step_idx+1}/{len(inq_steps)} (量化比例: {inq_steps[step_idx]*100}%) ---")
        inq_scheduler.step() # 划分量化组
        
        for epoch in range(num_epochs_per_step):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in dataloader:
                x, y = (batch["x"], batch["y"]) if isinstance(batch, dict) else (batch[0], batch[1])
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * x.size(0)
                _, predicted = out.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            
            lr_scheduler.step() # 正确的顺序：optimizer.step() 之后
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"  [Epoch {epoch+1}/{num_epochs_per_step}] LR: {lr_scheduler.get_last_lr()[0]:.6f} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")

    print("\n========== STEP 4: 压缩效果评估与保存 ==========")
    total_params, nonzero_params = count_parameters(model)
    sparsity = (1 - nonzero_params / total_params) * 100
    print(f"[Result] 总参数量: {total_params / 1e6:.2f} M")
    print(f"[Result] 实际稀疏度: {sparsity:.2f}%")

    # 评估需要关闭 shuffle
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(eval_dataloader))
    dummy_x = sample_batch["x"] if isinstance(sample_batch, dict) else sample_batch[0]
    latency, fps = benchmark_latency(model, dummy_x, device)
    print(f"[Result] 平均推理延迟: {latency:.2f} ms / batch  |  吞吐量: {fps:.2f} FPS")

    print("\n[Result] 正在生成评估报表与混淆矩阵...")
    # 保存绘图逻辑
    analysis_ret = ratr_predict_and_analyze(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        save_dir=hp.get("save_dir", "./result"),
        # output_dir=hp.get("output_dir", "./output")
    )

    save_filename = f"compressed_{model_name}_prune{prune_ratio}_int8.pth"
    save_path = os.path.join(hp.get("save_dir", "./result"), save_filename)
    torch.save(model.state_dict(), save_path)
    print(f"[*] 模型已保存至: {save_path}")

if __name__ == "__main__":
    run_prune_and_quantize("proj_RATR_complex.json")
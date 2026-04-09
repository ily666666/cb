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


# ----------------- 2. 优化后的 INQ 自定义优化器 -----------------
class INQSGD(torch.optim.SGD):
    def __init__(self, params, weight_bits=8, *args, **kwargs):
        super(INQSGD, self).__init__(params, *args, **kwargs)
        for group in self.param_groups:
            group['weight_bits'] = weight_bits
            group['Ts'] = []
            group['ns'] = []
            for p in group['params']:
                group['Ts'].append(torch.ones_like(p.data))

    def step(self, closure=None):
        # 在反向传播后，强制将已被量化(T==0)的权重的梯度置零，防止优化器撤销它的量化状态
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if 'Ts' in group and len(group['Ts']) > idx:
                    T = group['Ts'][idx]
                    p.grad.data.mul_(T)
        return super(INQSGD, self).step(closure)


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

    # 如果有双卡，启动并行加速（如果是物理剪枝，建议先单卡剪完再包裹DataParallel）
    # use_multi_gpu = torch.cuda.device_count() > 1

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
        dataset=dataset,
        model=model,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        save_dir=save_dir
    )
    # 重命名第一张图 (压缩前)
    default_cm_path = os.path.join(save_dir, "confusion_matrix.png")
    new_ori_cm_path = os.path.join(save_dir, f"cm_original_{timestamp}.png")
    if os.path.exists(default_cm_path):
        os.rename(default_cm_path, new_ori_cm_path)
        print(f"[*] 原始模型混淆矩阵已保存为: {new_ori_cm_path}")
    # =====================================================



    print("\n========== STEP 2: 真实的物理结构化剪枝 (Physical Pruning) ==========")
    prune_ratio = 0.15
    print(f"[*] 执行图追踪物理剪枝，安全移除 {prune_ratio * 100}% 的通道并重构网络...")

    # 建立重要性评估器（L2 范数）
    imp = tp.importance.MagnitudeImportance(p=2)
    # 忽略最后的全连接分类层，不参与剪枝
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == int(hp.get("num_classes", 3)):
            ignored_layers.append(m)

    # 使用 torch_pruning 重构网络架构
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=dummy_x,
        importance=imp,
        iterative_steps=1,
        ch_sparsity=prune_ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()  # 这里发生真实的物理切除！

    total_params_pruned, _ = count_parameters(model)
    print(f"[剪枝后] 网络重构完成！当前总参数量: {total_params_pruned / 1e6:.2f} M (真实减小！)")

   # # 剪枝完之后再套用多卡并行，防止底层名冲突
   # if use_multi_gpu:
   #     print(f"[Hardware] 开启 DataParallel 双卡加速！")
   #     model = nn.DataParallel(model)

    print("\n========== STEP 3: 增量网络量化 (INQ) 微调 ==========")
    # 把权重极小的值抹平
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and torch.max(torch.abs(p.data)) == 0:
                p.data.add_(1e-12)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = INQSGD(trainable_params, lr=5e-4, momentum=0.9, weight_bits=8)

    inq_steps = [0.5, 0.75, 0.82, 1.0]
    inq_scheduler = INQScheduler(optimizer, inq_steps)

    num_epochs_per_step = 4
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(inq_steps) * num_epochs_per_step)

    for step_idx in range(len(inq_steps)):
        print(f"\n--- INQ 阶段 {step_idx + 1}/{len(inq_steps)} (量化比例: {inq_steps[step_idx] * 100}%) ---")
        inq_scheduler.step()

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

            lr_scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(
                f"  [Epoch {epoch + 1}/{num_epochs_per_step}] LR: {lr_scheduler.get_last_lr()[0]:.6f} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

    print("\n========== STEP 4: 物理压缩效果评估与保存 ==========")

    latency, fps = benchmark_latency(model, dummy_x, device)
    print(f"\n>>>> [最终对比总结] <<<<")
    print(f" - 原始参数量: {total_params_ori / 1e6:.2f} M  =>  优化后参数量: {total_params_pruned / 1e6:.2f} M")
    print(f" - 原始延迟: {latency_ori:.2f} ms  =>  优化后延迟: {latency:.2f} ms")
    print(f" - 原始 FPS: {fps_ori:.2f}  =>  优化后 FPS: {fps:.2f}")

    print("\n[Result] 正在生成评估报表与混淆矩阵...")
    analysis_ret = ratr_predict_and_analyze(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        save_dir=hp.get("save_dir", "./result")
    )

    # =================重命名压缩后的混淆矩阵=================
    new_compressed_cm_path = os.path.join(save_dir, f"cm_pruned_int8_{timestamp}.png")
    if os.path.exists(default_cm_path):
        os.rename(default_cm_path, new_compressed_cm_path)
        print(f"[*] 压缩后模型混淆矩阵已保存为: {new_compressed_cm_path}")
    # ===================================================


    save_filename = f"physical_compressed_{model_name}_prune{prune_ratio}_int8.pth"
    save_path = os.path.join(hp.get("save_dir", "./result"), save_filename)

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"[*] 真实物理瘦身模型已保存至: {save_path}")


if __name__ == "__main__":
    run_prune_and_quantize("proj_RATR_complex.json")
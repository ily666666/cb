import os
import sys
import argparse
import numpy as np
import time
import torch
from easydict import EasyDict
import torch.nn as nn
from thop import profile, clever_format
from torchvision import datasets, transforms
from slimming.models.resnet_ext import Bottleneck

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
import utils.Helper as Helper
import utils.Datasets as Datasets
from slimming.models.resnet_ext import resnet50_cifar100
from utils.cutmix import cutmix_data


# ---------- 参数 ----------
def get_params():
    parser = argparse.ArgumentParser(description='ResNet50蒸馏后剪枝 (CIFAR100)')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--percent', type=float, default=0.30,
                        help='全局剪枝比例（默认 30%）')
    parser.add_argument('--model', default='./distill_logs/distilled_resnet50.pth.tar')
    parser.add_argument('--save', default='./pruned_resnet50_logs')
    parser.add_argument('--arch', default='resnet50_cifar100')
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--finetune-full', action='store_true')
    parser.add_argument('--init-lr', type=float, default=0.01)
    parser.add_argument('--use-cutmix', action='store_true')
    parser.add_argument('--cutmix-beta', type=float, default=1.0)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    os.makedirs(args.save, exist_ok=True)
    return args


# ---------- 数据 ----------
def get_dataloader(args):
    if args.dataset == 'cifar100':
        mean = [0.50705882, 0.48666667, 0.44078431]
        std = [0.26745098, 0.25568627, 0.27607843]
    else:
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.CIFAR100(root=args.data_path, train=True,
                                      download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=args.data_path, train=False,
                                     download=True, transform=test_transform)

    kwargs = dict(num_workers=4, pin_memory=True) if args.cuda else dict(num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)
    return train_loader, test_loader


# ---------- 测试 ----------
def test(model, test_loader, args):
    model.eval()
    correct = total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    print(f'测试精度: {correct}/{total} ({acc:.2f}%)')
    return acc


# ---------- 阈值 ----------
def get_bn_threshold(args, model):
    bn_weights = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'fc' not in name:
            bn_weights.append(m.weight.data.abs().clone())
    if not bn_weights:
        raise ValueError("未找到BN层，无法进行剪枝")
    all_weights = torch.cat(bn_weights)
    sorted_weights, _ = torch.sort(all_weights)
    threshold_idx = int(len(sorted_weights) * args.percent)
    threshold = sorted_weights[threshold_idx] if threshold_idx > 0 else sorted_weights[0]
    print(f"剪枝阈值: {threshold.item()}")
    return threshold


# ---------- 剪枝配置 ----------
def generate_pruned_config(model, threshold, args):
    default_cfg = {
        'conv1': 64,
        'layer1': [[64, 64, 256]] * 3,
        'layer2': [[128, 128, 512]] * 4,
        'layer3': [[256, 256, 1024]] * 6,
        'layer4': [[512, 512, 2048]] * 3,
        'fc': 100
    }
    new_cfg = EasyDict(default_cfg.copy())
    masks = []
    layer_counts = {'layer1': 0, 'layer2': 0, 'layer3': 0, 'layer4': 0}
    prev_out_channels = None

    # conv1 / bn1
    if hasattr(model, 'bn1') and isinstance(model.bn1, nn.BatchNorm2d):
        bn_weight = model.bn1.weight.data.abs()
        keep_idx = torch.nonzero(bn_weight > threshold).squeeze()
        if keep_idx.ndim == 0:
            keep_idx = keep_idx.unsqueeze(0)
        keep = max(len(keep_idx), default_cfg['conv1'] // 2)  # ≥50%
        if len(keep_idx) > keep:
            _, keep_idx = torch.topk(bn_weight, keep)
        mask = torch.zeros_like(bn_weight)
        mask[keep_idx] = 1.0
        prev_out_channels = int(mask.sum())
        new_cfg['conv1'] = prev_out_channels
        masks.append(('bn1', mask))
        print(f"conv1/bn1: 保留 {prev_out_channels}/{len(bn_weight)} 通道")

    # Bottleneck 块
    for name, module in model.named_modules():
        if isinstance(module, Bottleneck):
            layer_name = next((k for k in layer_counts if k in name), None)
            if not layer_name:
                continue
            block_idx = layer_counts[layer_name]
            if block_idx >= len(default_cfg[layer_name]):
                layer_counts[layer_name] = 0
                continue

            def prune_bn(bn, max_c):
                bn_w = bn.weight.data.abs()
                keep_idx = torch.nonzero(bn_w > threshold).squeeze()
                if keep_idx.ndim == 0:
                    keep_idx = keep_idx.unsqueeze(0)
                keep = max(len(keep_idx), max_c // 2)  # ≥50%
                if len(keep_idx) > keep:
                    _, keep_idx = torch.topk(bn_w, keep)
                mask = torch.zeros_like(bn_w)
                mask[keep_idx] = 1.0
                return mask, int(mask.sum())

            # bn1
            bn1_mask, c1 = prune_bn(module.bn1, default_cfg[layer_name][block_idx][0])
            if prev_out_channels is not None:
                c1 = min(c1, prev_out_channels)
                bn1_mask[list(range(c1, len(bn1_mask)))] = 0
            new_cfg[layer_name][block_idx][0] = c1
            masks.append((f"{layer_name}.{block_idx}.bn1", bn1_mask))

            # bn2
            bn2_mask, c2 = prune_bn(module.bn2, default_cfg[layer_name][block_idx][1])
            new_cfg[layer_name][block_idx][1] = c2
            masks.append((f"{layer_name}.{block_idx}.bn2", bn2_mask))

            # bn3
            bn3_mask, c3 = prune_bn(module.bn3, default_cfg[layer_name][block_idx][2])
            new_cfg[layer_name][block_idx][2] = c3
            masks.append((f"{layer_name}.{block_idx}.bn3", bn3_mask))

            # downsample
            if hasattr(module, 'downsample') and module.downsample is not None:
                down_bn = module.downsample[1]
                if isinstance(down_bn, nn.BatchNorm2d):
                    # 强制对齐主路径输出通道数
                    out_channels = new_cfg[layer_name][block_idx][2]
                    ds_mask = torch.zeros_like(down_bn.weight.data)
                    ds_mask[:out_channels] = 1.0
                    masks.append((f"{layer_name}.{block_idx}.downsample.1", ds_mask))
                    print(f"{layer_name}.{block_idx}.downsample: 保留 {out_channels} 通道")

            print(f"{layer_name}.{block_idx}: "
                  f"bn1={c1}/{default_cfg[layer_name][block_idx][0]} "
                  f"bn2={c2}/{default_cfg[layer_name][block_idx][1]} "
                  f"bn3={c3}/{default_cfg[layer_name][block_idx][2]}")

            prev_out_channels = c3
            layer_counts[layer_name] += 1

    return new_cfg, masks


# ---------- 生成模型 ----------
def generate_pruned_model(new_cfg, args, original_model):
    device = next(original_model.parameters()).device
    pruned_model = resnet50_cifar100(cfg=new_cfg).to(device)
    last_out_channels = new_cfg['layer4'][-1][2]
    pruned_model.fc = nn.Linear(last_out_channels, new_cfg['fc']).to(device)
    return pruned_model


# ---------- 权重复制 ----------
def copy_origin_weights(args, original_model, pruned_model, cfg, masks):
    device = next(original_model.parameters()).device
    pruned_model = pruned_model.to(device)
    mask_dict = {name: mask.to(device) for name, mask in masks}

    def copy_conv_bn(src_conv, src_bn, dst_conv, dst_bn, out_mask, in_mask=None):
        out_idx = torch.nonzero(out_mask).squeeze()
        if out_idx.ndim == 0:
            out_idx = out_idx.unsqueeze(0)
        if in_mask is None:
            w = src_conv.weight.data[out_idx]
        else:
            in_idx = torch.nonzero(in_mask).squeeze()
            if in_idx.ndim == 0:
                in_idx = in_idx.unsqueeze(0)
            w = src_conv.weight.data[out_idx][:, in_idx]
        dst_conv.weight.data = w.clone()
        dst_bn.weight.data = src_bn.weight.data[out_idx].clone()
        dst_bn.bias.data = src_bn.bias.data[out_idx].clone()
        dst_bn.running_mean = src_bn.running_mean[out_idx].clone()
        dst_bn.running_var = src_bn.running_var[out_idx].clone()

    # conv1/bn1
    if 'bn1' in mask_dict:
        copy_conv_bn(original_model.conv1, original_model.bn1,
                     pruned_model.conv1, pruned_model.bn1,
                     mask_dict['bn1'])

    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    prev_layer_name = None
    for layer_name in layer_names:
        orig_layer = getattr(original_model, layer_name)
        prun_layer = getattr(pruned_model, layer_name)
        for block_idx, (orig_block, prun_block) in enumerate(zip(orig_layer, prun_layer)):
            if layer_name == 'layer1' and block_idx == 0:
                prev_mask = mask_dict['bn1']
            else:
                if block_idx > 0:
                    prev_bn3_id = f"{layer_name}.{block_idx - 1}.bn3"
                else:
                    prev_bn3_id = f"{prev_layer_name}.{len(getattr(cfg, prev_layer_name)) - 1}.bn3"
                prev_mask = mask_dict[prev_bn3_id]

            bn1_id = f"{layer_name}.{block_idx}.bn1"
            bn2_id = f"{layer_name}.{block_idx}.bn2"
            bn3_id = f"{layer_name}.{block_idx}.bn3"

            copy_conv_bn(orig_block.conv1, orig_block.bn1,
                         prun_block.conv1, prun_block.bn1,
                         mask_dict[bn1_id], prev_mask)

            copy_conv_bn(orig_block.conv2, orig_block.bn2,
                         prun_block.conv2, prun_block.bn2,
                         mask_dict[bn2_id], mask_dict[bn1_id])

            copy_conv_bn(orig_block.conv3, orig_block.bn3,
                         prun_block.conv3, prun_block.bn3,
                         mask_dict[bn3_id], mask_dict[bn2_id])

            ds_bn_id = f"{layer_name}.{block_idx}.downsample.1"
            if hasattr(orig_block, 'downsample') and orig_block.downsample is not None and \
                    ds_bn_id in mask_dict:
                copy_conv_bn(orig_block.downsample[0], orig_block.downsample[1],
                             prun_block.downsample[0], prun_block.downsample[1],
                             mask_dict[ds_bn_id], prev_mask)

        prev_layer_name = layer_name

    return pruned_model


# ---------- 微调 ----------
def finetune(args, model, train_loader, test_loader, epochs=30):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)
    if not args.finetune_full:
        for name, p in model.named_parameters():
            if not name.startswith('layer3') and not name.startswith('layer4') and 'fc' not in name:
                p.requires_grad = False
        print("🔒 已冻结 layer1/layer2/conv1，仅微调 layer3+layer4+fc")
    else:
        print("🔓 微调整个网络")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.init_lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    patience, counter = 5, 0
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            if args.use_cutmix:
                data, target_a, target_b, lam = cutmix_data(
                    data, target, alpha=args.cutmix_beta, device=device)
            optimizer.zero_grad()
            output = model(data)
            loss = (lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)) \
                if args.use_cutmix else criterion(output, target)
            if torch.isnan(loss):
                print("❗ NaN loss，停止训练")
                return best_acc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        acc = test(model, test_loader, args)
        print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}, acc={acc:.2f}%, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if epoch_loss + 1e-4 < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * 0.3, 1e-5)
                for g in optimizer.param_groups:
                    g['lr'] = new_lr
                print(f"🔻 lr 动态降至 {new_lr:.6f}")
                counter = 0

        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(args.save, f'best_finetune_p{int(args.percent*100)}{best_acc:.2f}%.pth.tar'))

    return best_acc


# ---------- 主流程 ----------
def pruning(args, model):
    threshold = get_bn_threshold(args, model)
    new_cfg, masks = generate_pruned_config(model, threshold, args)
    device = next(model.parameters()).device
    pruned_model = resnet50_cifar100(cfg=new_cfg).to(device)
    return pruned_model, new_cfg, masks


def evaluate(args, original_model, pruned_model, cfg, masks):
    train_loader, test_loader = get_dataloader(args)
    device = next(original_model.parameters()).device
    input_tensor = torch.randn(1, 3, 32, 32).to(device)

    original_flops, original_params = profile(original_model, inputs=(input_tensor,))
    original_flops_str, original_params_str = clever_format(
        [original_flops, original_params], "%.3f")
    print("\n原始模型评估:")
    original_acc = test(original_model, test_loader, args)

    try:
        pruned_flops, pruned_params = profile(pruned_model, inputs=(input_tensor,))
        pruned_flops_str, pruned_params_str = clever_format(
            [pruned_flops, pruned_params], "%.3f")
    except AttributeError:
        pruned_flops_str = pruned_params_str = "N/A"

    print("\n剪枝模型初始评估:")
    initial_acc = test(pruned_model, test_loader, args)

    print(f"\n开始剪枝后微调 ({args.finetune_epochs}个epoch):")
    finetuned_acc = finetune(args, pruned_model, train_loader,
                             test_loader, args.finetune_epochs)

    save_path = os.path.join(args.save, f"pruned_resnet50_p{int(args.percent*100)}_acc{finetuned_acc:.2f}%.pth.tar")
    torch.save({'state_dict': pruned_model.state_dict(),
                'cfg': cfg,
                'masks': masks,
                'accuracy': finetuned_acc}, save_path)

    print("\n===== 剪枝结果 =====")
    print(f"原始模型: FLOPs={original_flops_str}, 参数量={original_params_str}, 精度={original_acc:.2f}%")
    print(f"剪枝模型: FLOPs={pruned_flops_str}, 参数量={pruned_params_str}, 微调后精度={finetuned_acc:.2f}%")
    print(f"模型已保存至: {save_path}")

        # ===== 新增：对微调后的剪枝模型测速 =====
    BATCH = 128
    NUM_REPEAT = 200
    device = next(pruned_model.parameters()).device
    dummy = torch.randn(BATCH, 3, 32, 32).to(device)   # CIFAR-100 尺寸
    latency_ms = benchmark(pruned_model, dummy, num_repeat=NUM_REPEAT)
    fps = BATCH / (latency_ms / 1000.0)
    print(f"\n[Speed] After fine-tuning: {latency_ms:.2f} ms / batch  |  {fps:.0f} FPS")

@torch.no_grad()
def benchmark(model, x, num_repeat=200, warmup=20):
    """返回单次前向的毫秒耗时"""
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_repeat):
        _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / num_repeat * 1000.0   # ms


def main():
    args = get_params()
    train_loader, test_loader = get_dataloader(args)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = resnet50_cifar100().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("成功加载原始模型")

    print("剪枝前模型精度：")
    test(model, test_loader, args)

    pruned_model, new_cfg, masks = pruning(args, model)
    pruned_model = copy_origin_weights(args, model, pruned_model, new_cfg, masks)

    # for m in pruned_model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.kaiming_normal_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    evaluate(args, model, pruned_model, new_cfg, masks)



if __name__ == "__main__":
    main()
import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from datetime import datetime

# 模型定义（假设已存在于slimming.models.resnet_ext中）
from slimming.models.resnet_ext import resnet101_cifar100, resnet50_cifar100

# 设置随机种子确保可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(dataset):
    """获取数据集专用的数据增强策略"""
    if dataset == "cifar10":
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == "cifar100":
        mean = [0.50705882, 0.48666667, 0.44078431]
        std = [0.26745098, 0.25568627, 0.27607843]
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, test_transform

class DistillationLoss(nn.Module):
    """蒸馏损失函数，结合软标签、硬标签和特征匹配损失"""
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, student_feats, teacher_logits, teacher_feats, targets, alpha=0.5, beta=0.3):
        # 软标签损失（KL散度）
        soft_loss = self.kl_loss(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature **2)
        
        # 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, targets)
        
        # 多层特征匹配损失
        feat_loss = 0.0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            s_feat_pooled = nn.functional.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
            feat_loss += self.mse_loss(s_feat_pooled, t_feat)
        feat_loss /= len(student_feats)
        
        # 组合损失
        return (alpha * soft_loss) + ((1 - alpha - beta) * hard_loss) + (beta * feat_loss)

def get_features(model, x, is_teacher=True):
    """获取模型多层特征用于蒸馏"""
    features = []
    
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    features.append(x)  # 第一层特征
    
    x = model.layer1(x)
    features.append(x)  # layer1特征
    
    x = model.layer2(x)
    features.append(x)  # layer2特征
    
    x = model.layer3(x)
    features.append(x)  # layer3特征
    
    x = model.layer4(x)
    features.append(x)  # layer4特征
    
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    logits = model.fc(x)
    
    return logits, features

def init_student_from_teacher(student, teacher):
    """从教师模型初始化学生模型可迁移的参数"""
    student_dict = student.state_dict()
    teacher_dict = teacher.state_dict()
    
    transferable = {k: v for k, v in teacher_dict.items() 
                   if k in student_dict and v.shape == student_dict[k].shape}
    
    student_dict.update(transferable)
    student.load_state_dict(student_dict)
    print(f"从教师模型迁移 {len(transferable)}/{len(student_dict)} 个参数")
    return student

def train_epoch(args, teacher, student, train_loader, criterion, optimizer, epoch, writer, device):
    """蒸馏训练一个epoch"""
    student.train()
    teacher.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    batch_time = 0.0
    data_time = 0.0
    
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time += time.time() - end
        
        data, target = data.to(device), target.to(device)
        
        # 教师模型前向传播（不跟踪梯度）
        with torch.no_grad():
            teacher_logits, teacher_feats = get_features(teacher, data, is_teacher=True)
        
        # 学生模型前向传播
        student_logits, student_feats = get_features(student, data, is_teacher=False)
        
        # 计算蒸馏损失
        loss = criterion(
            student_logits, student_feats,
            teacher_logits, teacher_feats,
            target,
            alpha=args.alpha,
            beta=args.beta
        )
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 计时
        batch_time += time.time() - end
        end = time.time()
        
        # 日志输出
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {current_loss:.6f}\tAcc: {current_acc:.2f}%')
            
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/batch_acc', 100. * predicted.eq(target).sum().item() / target.size(0), global_step)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total
    print(f'Epoch {epoch} 训练结果: 平均损失: {avg_loss:.6f}, 准确率: {avg_acc:.2f}%')
    
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    writer.add_scalar('train/epoch_acc', avg_acc, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    return avg_loss, avg_acc

def test(model, test_loader, criterion, device):
    """测试模型性能"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            logits, _ = get_features(model, data)
            loss = criterion(logits, target)
            
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / total
    print(f'测试集结果: 平均损失: {avg_loss:.6f}, 准确率: {correct}/{total} ({avg_acc:.2f}%)')
    
    return avg_loss, avg_acc

def get_dataloader(args, dataset, data_path, batch_size, test_batch_size, num_workers=4):
    """获取训练和测试数据加载器"""
    train_transform, test_transform = get_transforms(dataset)
    
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_path, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=data_path, train=False, transform=test_transform, download=True
        )
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_path, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR100(
            root=data_path, train=False, transform=test_transform, download=True
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='ResNet知识蒸馏训练')
    parser.add_argument('--dataset', type=str, default='cifar100',
                      help='数据集类型 (cifar10/cifar100)')
    parser.add_argument('--data-path', type=str, default='./data',
                      help='数据集路径')
    parser.add_argument('--teacher-pretrained', type=str, required=True,
                      help='教师模型预训练权重路径')
    parser.add_argument('--epochs', type=int, default=120,
                      help='蒸馏训练轮次')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='训练批次大小')
    parser.add_argument('--test-batch-size', type=int, default=128,
                      help='测试批次大小')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='动量')
    parser.add_argument('--temperature', type=float, default=4.0,
                      help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.6,
                      help='软标签损失权重')
    parser.add_argument('--beta', type=float, default=0.2,
                      help='特征匹配损失权重')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='不使用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--log-interval', type=int, default=100,
                      help='日志打印间隔')
    parser.add_argument('--save', type=str, default='./distill_logs',
                      help='模型保存路径')
    parser.add_argument('--tensorboard', type=str, default='./tensorboard_distill',
                      help='TensorBoard日志路径')
    parser.add_argument('--early-stopping', type=int, default=20,
                      help='早停轮次')
    parser.add_argument('--resume', type=str, default='',
                      help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化设备
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save, exist_ok=True)
    log_dir = os.path.join(args.tensorboard, f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 数据加载
    print(f"加载{args.dataset}数据集...")
    train_loader, test_loader = get_dataloader(
        args, args.dataset, args.data_path, 
        args.batch_size, args.test_batch_size
    )
    
    # 初始化教师模型并测试精度
    print("初始化教师模型...")
    teacher = resnet101_cifar100().to(device)
    teacher_checkpoint = torch.load(args.teacher_pretrained, map_location=device)
    # 处理不同格式的检查点
    if 'state_dict' in teacher_checkpoint:
        teacher.load_state_dict(teacher_checkpoint['state_dict'])
    else:
        teacher.load_state_dict(teacher_checkpoint)
    teacher.eval()
    
    # 测试教师模型精度
    print("测试教师模型精度...")
    teacher_criterion = nn.CrossEntropyLoss()
    teacher_loss, teacher_acc = test(teacher, test_loader, teacher_criterion, device)
    print(f"教师模型测试精度: {teacher_acc:.2f}%")
    
    # 初始化学生模型
    print("初始化学生模型...")
    student = resnet50_cifar100().to(device)
    student = init_student_from_teacher(student, teacher)
    
    # 定义损失函数和优化器
    criterion = DistillationLoss(temperature=args.temperature)
    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )
    
    # 恢复训练
    start_epoch = 0
    best_student_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        student.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_student_acc = checkpoint['best_acc']
        print(f"恢复完成，从第 {start_epoch} 轮开始训练，最佳学生精度: {best_student_acc:.2f}%")
    
    # 早停计数器
    early_stopping_counter = 0
    
    # 开始训练
    print("开始蒸馏训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(args, teacher, student, train_loader, criterion, optimizer, epoch, writer, device)
        
        # 在测试集上验证
        student_loss, student_acc = test(student, test_loader, teacher_criterion, device)
        
        # 记录到TensorBoard
        writer.add_scalar('test/loss', student_loss, epoch)
        writer.add_scalar('test/acc', student_acc, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if student_acc > best_student_acc:
            best_student_acc = student_acc
            early_stopping_counter = 0  # 重置早停计数器
            
            # 计算并输出精度差
            acc_diff = teacher_acc - best_student_acc
            print(f"教师模型精度: {teacher_acc:.2f}%，最佳学生模型精度: {best_student_acc:.2f}%")
            print(f"精度差异: {acc_diff:.2f}%")
            
            # 保存文件名包含学生模型精度
            save_path = os.path.join(args.save, f"best_student_resnet50_acc{best_student_acc:.2f}.pth.tar")
            torch.save({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_acc': best_student_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_path)
            print(f"保存最佳模型至: {save_path}")
        else:
            early_stopping_counter += 1
            print(f"早停计数器: {early_stopping_counter}/{args.early_stopping}")
            if early_stopping_counter >= args.early_stopping:
                print(f"早停触发: 连续 {args.early_stopping} 轮未提升精度")
                break
        
        # 每50个epoch保存一次检查点
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(args.save, f"checkpoint_student_resnet50_epoch{epoch}_acc{student_acc:.2f}.pth.tar")
            torch.save({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_acc': best_student_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"保存检查点至: {checkpoint_path}")
    
    # 训练结束
    final_path = os.path.join(args.save, f"final_student_resnet50_acc{best_student_acc:.2f}.pth.tar")
    torch.save({
        'epoch': args.epochs,
        'state_dict': student.state_dict(),
        'best_acc': best_student_acc,
    }, final_path)
    print(f"\n训练完成! 最佳学生测试精度: {best_student_acc:.2f}%")
    print(f"教师模型测试精度: {teacher_acc:.2f}%")
    print(f"最终精度差异: {teacher_acc - best_student_acc:.2f}%")
    print(f"最终模型保存至: {final_path}")
    
    writer.close()

if __name__ == '__main__':
    main()
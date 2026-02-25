#!/usr/bin/env python3
"""
边侧联邦学习脚本

功能：
1. 连接云侧服务器
2. 下载全局模型
3. 本地训练
4. 上传本地模型
5. 重复指定轮次

使用示例：
    python run/edge/run_edge_federated.py --edge_id 1 --cloud_host localhost --data_path E:/3dataset --dataset_type link11 --num_rounds 10
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import f1_score

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed.project import create_model_by_type
from run.network_utils import NetworkEdge
from torch.utils.data import DataLoader
import pickle


def load_edge_data(edge_id, dataset_type, batch_size, data_path, add_noise=False, noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
    """加载边侧数据"""
    data_splits_dir = f'{data_path}/{dataset_type}'
    presplit_edge_file = f'{data_splits_dir}/edge_{edge_id}_data.pkl'
    
    if not os.path.exists(presplit_edge_file):
        raise FileNotFoundError(f"边侧 {edge_id} 数据文件不存在: {presplit_edge_file}")
    
    with open(presplit_edge_file, 'rb') as f:
        data = pickle.load(f)
    
    if 'train' in data:
        from utils.readdata_presplit import PresplitDataset
        
        train_subset = PresplitDataset(presplit_edge_file, split='train',
                                      add_noise=add_noise, noise_type=noise_type,
                                      noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        val_subset = PresplitDataset(presplit_edge_file, split='val',
                                    add_noise=add_noise, noise_type=noise_type,
                                    noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        test_subset = PresplitDataset(presplit_edge_file, split='test',
                                     add_noise=add_noise, noise_type=noise_type,
                                     noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        
        print(f"✅ 预划分数据加载完成")
        print(f"  训练样本: {len(train_subset)}")
        print(f"  验证样本: {len(val_subset)}")
        print(f"  测试样本: {len(test_subset)}")
    else:
        raise ValueError("仅支持预划分数据格式")
    
    use_drop_last = (dataset_type in ['radioml', 'radar', 'rml2016', 'link11'])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=use_drop_last)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=use_drop_last)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader, test_loader


def train_local_model(model, train_loader, global_model_state, device, config):
    """本地训练模型"""
    model.load_state_dict(global_model_state)
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                             momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    total_loss = 0
    correct = 0
    total = 0
    
    for epoch in range(config['local_epochs']):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if config.get('grad_clip', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('grad_clip_value', 1.0))
            
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_total += targets.size(0)
            epoch_correct += predicted.eq(targets).sum().item()
        
        total_loss = epoch_loss / len(train_loader)
        total = epoch_total
        correct = epoch_correct
        
        print(f"  Epoch {epoch+1}/{config['local_epochs']}: Loss={total_loss:.4f}, Acc={100.*correct/total:.2f}%")
    
    train_acc = 100. * correct / total
    return model.state_dict(), total_loss, train_acc


def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    test_acc = 100. * correct / total if total > 0 else 0.0
    test_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) if len(all_targets) > 0 else 0.0
    
    return test_loss, test_acc, test_f1


def main():
    parser = argparse.ArgumentParser(description='边侧联邦学习')
    
    # 边侧参数
    parser.add_argument('--edge_id', type=int, required=True, help='边侧ID')
    parser.add_argument('--cloud_host', type=str, required=True, help='云侧主机地址')
    parser.add_argument('--cloud_port', type=int, default=9999, help='云侧端口')
    
    # 数据参数
    parser.add_argument('--dataset_type', type=str, default='link11', help='数据集类型')
    parser.add_argument('--data_path', type=str, required=True, help='数据集根目录路径')
    parser.add_argument('--num_classes', type=int, default=7, help='类别数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    # 模型参数
    parser.add_argument('--edge_model', type=str, default=None,
                        help='边侧模型类型（默认：real_resnet20_{dataset_type}）')
    parser.add_argument('--kd_model_path', type=str, default=None,
                        help='知识蒸馏模型路径（默认：run/edge/pth/{dataset_type}/edge_{edge_id}_kd_model.pth）')
    
    # 训练参数
    parser.add_argument('--num_rounds', type=int, default=5, help='联邦学习轮次')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器')
    
    # 噪声参数
    parser.add_argument('--add_noise', action='store_true', help='是否添加噪声')
    parser.add_argument('--noise_type', type=str, default='awgn', help='噪声类型')
    parser.add_argument('--noise_snr_db', type=float, default=15, help='噪声SNR(dB)')
    
    # 网络参数
    parser.add_argument('--rate_limit', type=float, default=10.0, help='网络速率限制（MB/s，默认10）')
    
    args = parser.parse_args()
    
    # 设置默认值
    if args.edge_model is None:
        args.edge_model = f'real_resnet20_{args.dataset_type}'
    
    if args.kd_model_path is None:
        args.kd_model_path = f'run/edge/pth/{args.dataset_type}/edge_{args.edge_id}_kd_model.pth'
    
    print(f"\n{'='*70}")
    print(f"[边侧 {args.edge_id}] 联邦学习启动")
    print(f"{'='*70}")
    print(f"云侧: {args.cloud_host}:{args.cloud_port}")
    print(f"数据集: {args.dataset_type}")
    print(f"模型: {args.edge_model}")
    print(f"训练轮次: {args.num_rounds}")
    print(f"本地训练轮数: {args.local_epochs}")
    if args.rate_limit:
        print(f"网络限速: {args.rate_limit} MB/s")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 创建网络边侧
    network_edge = NetworkEdge(args.edge_id, args.cloud_host, args.cloud_port, rate_limit_mbps=args.rate_limit)
    
    # 2. 等待云侧准备好
    network_edge.wait_for_cloud_ready(max_wait=300, check_interval=5)
    
    # 3. 加载数据
    print(f"[边侧 {args.edge_id}] 加载异构数据...")
    train_loader, val_loader, test_loader, global_test_loader = load_edge_data(
        args.edge_id, args.dataset_type, args.batch_size, args.data_path,
        args.add_noise, args.noise_type, args.noise_snr_db
    )
    
    # 4. 创建模型
    print(f"[边侧 {args.edge_id}] 创建学生模型...")
    model = create_model_by_type(args.edge_model, args.num_classes, args.dataset_type)
    
    # 5. 加载知识蒸馏模型（如果存在）
    if os.path.exists(args.kd_model_path):
        print(f"[边侧 {args.edge_id}] 加载知识蒸馏模型: {args.kd_model_path}")
        checkpoint = torch.load(args.kd_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[边侧 {args.edge_id}] 知识蒸馏模型加载完成")
    else:
        print(f"[边侧 {args.edge_id}] 警告: 未找到知识蒸馏模型，使用随机初始化")
    
    # 6. 训练配置
    config = {
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'local_epochs': args.local_epochs,
        'grad_clip': True,
        'grad_clip_value': 1.0,
    }
    
    print(f"\n{'='*70}")
    print(f"[联邦学习] 开始训练")
    print(f"{'='*70}")
    
    # 初始化最终指标变量
    local_acc = 0.0
    global_acc = 0.0
    local_f1 = 0.0
    global_f1 = 0.0
    
    # 7. 联邦学习训练循环
    for round_num in range(args.num_rounds):
        print(f"\n{'='*70}")
        print(f"[边侧 {args.edge_id}] 轮次 {round_num + 1}/{args.num_rounds}")
        print(f"{'='*70}")
        
        try:
            # 检查云侧是否准备好
            print(f"\n[等待] 检查云侧是否准备好轮次 {round_num + 1}...")
            network_edge.check_round_ready(round_num)
            
            # 下载全局模型
            print(f"\n[阶段1] 下载全局模型...")
            global_model_state, download_time, download_size = network_edge.download_model(round_num)
            
            # 等待云侧通知所有边侧下载完成
            print(f"[边侧 {args.edge_id}] 等待云侧通知所有边侧下载完成...")
            network_edge.wait_for_phase_ready('upload', max_wait=300)
            
            # 本地训练
            print(f"\n[阶段2] 本地训练...")
            print(f"[边侧 {args.edge_id}] 训练样本数: {len(train_loader.dataset)}")
            
            local_model_state, train_loss, train_acc = train_local_model(
                model, train_loader, global_model_state, device, config
            )
            
            print(f"[边侧 {args.edge_id}] 训练完成: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            
            # 测试评估
            print(f"\n[阶段3] 测试评估...")
            
            local_loss, local_acc, local_f1 = test_model(model, test_loader, device)
            print(f"[边侧 {args.edge_id}] 本地测试集: Loss={local_loss:.4f}, Acc={local_acc:.2f}%, F1={local_f1:.4f}")
            
            global_loss, global_acc, global_f1 = test_model(model, global_test_loader, device)
            print(f"[边侧 {args.edge_id}] 全局测试集: Loss={global_loss:.4f}, Acc={global_acc:.2f}%, F1={global_f1:.4f}")
            
            # 上传本地模型
            print(f"\n[阶段4] 上传本地模型...")
            upload_time, upload_size = network_edge.upload_model(
                local_model_state, num_samples=len(train_loader.dataset)
            )
            
            print(f"\n[边侧 {args.edge_id}] 轮次 {round_num + 1} 完成")
            print(f"  - 下载: {download_size:.2f}MB, {download_time:.2f}s")
            print(f"  - 训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  - 本地测试: Loss={local_loss:.4f}, Acc={local_acc:.2f}%, F1={local_f1:.4f}")
            print(f"  - 全局测试: Loss={global_loss:.4f}, Acc={global_acc:.2f}%, F1={global_f1:.4f}")
            print(f"  - 上传: {upload_size:.2f}MB, {upload_time:.2f}s")
            
        except Exception as e:
            print(f"[边侧 {args.edge_id}] 错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 8. 保存最终联邦学习模型
    print(f"\n{'='*70}")
    print(f"[边侧 {args.edge_id}] 保存最终模型")
    print(f"{'='*70}")
    
    save_dir = f'run/edge/pth/{args.dataset_type}'
    os.makedirs(save_dir, exist_ok=True)
    final_model_path = f'{save_dir}/edge_{args.edge_id}_federated_model.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'edge_id': args.edge_id,
        'model_architecture': args.edge_model,
        'num_rounds': args.num_rounds,
        'final_local_acc': local_acc,
        'final_global_acc': global_acc,
        'final_local_f1': local_f1,
        'final_global_f1': global_f1
    }, final_model_path)
    
    print(f"[边侧 {args.edge_id}] 模型已保存: {final_model_path}")
    print(f"  - 最终本地测试准确率: {local_acc:.2f}%")
    print(f"  - 最终全局测试准确率: {global_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"[边侧 {args.edge_id}] 完成所有训练轮次")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

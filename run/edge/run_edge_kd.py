#!/usr/bin/env python3
"""
边侧轻量化脚本

功能：
1. 从云侧下载教师模型
2. 加载边侧异构数据
3. 从教师模型蒸馏学生模型
4. 保存蒸馏后的学生模型

使用示例：
    python run/edge/run_edge_kd.py --edge_id 1 --cloud_host 192.168.1.100 --dataset_type link11 --kd_epochs 20
"""
import sys
import os
import argparse
import socket
import struct
import time
import torch
import pickle

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed.project import create_model_by_type
from helper.distill_helper import DistillationHelper
from torch.utils.data import DataLoader


def send_data(conn, data, rate_limit_mbps=None):
    """发送数据（带速率限制）
    
    改进的速率限制算法：
    - 使用更大的chunk_size (1MB) 减少循环开销
    - 累积计时，避免频繁的time.time()调用
    - 只在必要时sleep，提高精度
    """
    serialized = pickle.dumps(data)
    size = len(serialized)
    
    # 发送数据大小
    conn.sendall(struct.pack('Q', size))
    
    # 如果启用限速，分块发送
    if rate_limit_mbps:
        chunk_size = 1024 * 1024  # 1MB per chunk (原来是64KB)
        rate_limit_bps = rate_limit_mbps * 1024 * 1024  # 转换为 bytes/s
        
        sent = 0
        start_time = time.time()  # 记录整体开始时间
        
        while sent < size:
            chunk = serialized[sent:sent + chunk_size]
            conn.sendall(chunk)
            sent += len(chunk)
            
            # 每发送完一个chunk，检查是否需要限速
            elapsed = time.time() - start_time
            expected_time = sent / rate_limit_bps
            
            # 如果发送太快，sleep到预期时间
            if elapsed < expected_time:
                sleep_time = expected_time - elapsed
                # 只有当sleep时间大于10ms时才sleep，避免精度问题
                if sleep_time > 0.01:
                    time.sleep(sleep_time)
    else:
        # 不限速，直接发送
        conn.sendall(serialized)
    
    return size


def receive_data(conn):
    """接收数据"""
    size_data = _recv_all(conn, 8)
    size = struct.unpack('Q', size_data)[0]
    data = _recv_all(conn, size)
    return pickle.loads(data), size


def _recv_all(conn, size):
    """确保接收完整数据"""
    data = b''
    while len(data) < size:
        packet = conn.recv(min(size - len(data), 65536))
        if not packet:
            raise ConnectionError("连接断开")
        data += packet
    return data


def download_teacher_model(edge_id, cloud_host, cloud_port, save_path):
    """从云侧下载教师模型
    
    Args:
        edge_id: 边侧ID
        cloud_host: 云侧主机地址
        cloud_port: 云侧端口
        save_path: 保存路径
        
    Returns:
        teacher_model_state: 教师模型权重
        teacher_model_architecture: 教师模型架构
    """
    print(f"\n[边侧 {edge_id}] 从云侧下载教师模型...")
    print(f"  云侧地址: {cloud_host}:{cloud_port}")
    
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # 连接云侧
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.settimeout(300)  # 5分钟超时
            conn.connect((cloud_host, cloud_port))
            
            try:
                # 发送下载请求
                print(f"[边侧 {edge_id}] 发送下载请求...")
                send_data(conn, {
                    'type': 'download_teacher',
                    'edge_id': edge_id
                })
                
                # 接收教师模型
                print(f"[边侧 {edge_id}] 接收教师模型...")
                start_time = time.time()
                response, size = receive_data(conn)
                download_time = time.time() - start_time
                
                if response.get('status') == 'success':
                    teacher_model_state = response['model_state']
                    teacher_model_architecture = response['model_architecture']
                    num_classes = response['num_classes']
                    dataset_type = response['dataset_type']
                    
                    print(f"[边侧 {edge_id}] 下载完成: {size/(1024*1024):.2f}MB, 耗时 {download_time:.2f}s")
                    print(f"  模型架构: {teacher_model_architecture}")
                    print(f"  类别数: {num_classes}")
                    
                    # 保存到本地
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': teacher_model_state,
                        'model_architecture': teacher_model_architecture,
                        'num_classes': num_classes,
                        'dataset_type': dataset_type
                    }, save_path)
                    print(f"[边侧 {edge_id}] 教师模型已保存: {save_path}")
                    
                    return teacher_model_state, teacher_model_architecture
                else:
                    error_msg = response.get('message', '未知错误')
                    raise Exception(f"云侧拒绝: {error_msg}")
                
            finally:
                conn.close()
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[边侧 {edge_id}] 下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
                print(f"[边侧 {edge_id}] {retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"下载教师模型失败（已重试{max_retries}次）: {e}")
    
    raise Exception("下载教师模型失败：超过最大重试次数")


def load_edge_data(data_path, dataset_type, batch_size):
    """加载边侧数据
    
    Args:
        data_path: 边侧数据文件路径
        dataset_type: 数据集类型
        batch_size: 批次大小
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"边侧数据文件不存在: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'train' in data:
        from utils.readdata_presplit import PresplitDataset
        
        train_subset = PresplitDataset(data_path, split='train')
        test_subset = PresplitDataset(data_path, split='test')
        
        print(f"✅ 预划分数据加载完成")
        print(f"  训练样本: {len(train_subset)}")
        print(f"  测试样本: {len(test_subset)}")
    else:
        raise ValueError("仅支持预划分数据格式（需要包含 'train' 和 'test' 键）")
    
    use_drop_last = (dataset_type in ['radioml', 'radar', 'rml2016', 'link11'])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=use_drop_last)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='边侧轻量化')
    
    # 边侧参数
    parser.add_argument('--edge_id', type=int, required=True, help='边侧ID')
    
    # 云侧连接参数
    parser.add_argument('--cloud_host', type=str, required=True, help='云侧主机地址')
    parser.add_argument('--cloud_port', type=int, default=9999, help='云侧端口（默认9999）')
    
    # 数据参数
    parser.add_argument('--dataset_type', type=str, default='link11', help='数据集类型')
    parser.add_argument('--num_classes', type=int, default=7, help='类别数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--data_path', type=str, required=True,
                        help='边侧数据文件路径（预划分数据，例如：E:\\3dataset\\link11\\edge_1_data.pkl）')
    
    # 模型参数
    parser.add_argument('--edge_model', type=str, default=None,
                        help='边侧模型类型（默认：real_resnet20_{dataset_type}）')
    parser.add_argument('--teacher_model_path', type=str, default=None,
                        help='教师模型本地缓存路径（默认：run/edge/pth/{dataset_type}/teacher_model.pth）')
    
    # 轻量化参数
    parser.add_argument('--kd_epochs', type=int, default=20, help='蒸馏训练轮数')
    parser.add_argument('--kd_temperature', type=float, default=4.0, help='蒸馏温度')
    parser.add_argument('--kd_alpha', type=float, default=0.9, help='蒸馏损失权重')
    parser.add_argument('--kd_distill', type=str, default='kd', help='蒸馏方法')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录（默认：run/edge/pth/{dataset_type}）')
    
    args = parser.parse_args()
    
    # 设置默认值
    if args.edge_model is None:
        args.edge_model = f'real_resnet20_{args.dataset_type}'
    
    if args.teacher_model_path is None:
        args.teacher_model_path = f'run/edge/pth/{args.dataset_type}/teacher_model.pth'
    
    if args.save_dir is None:
        args.save_dir = f'run/edge/pth/{args.dataset_type}'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"[边侧 {args.edge_id}] 轻量化")
    print(f"{'='*70}")
    print(f"云侧: {args.cloud_host}:{args.cloud_port}")
    print(f"数据集: {args.dataset_type}")
    print(f"学生模型: {args.edge_model}")
    print(f"蒸馏轮数: {args.kd_epochs}")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 加载数据
    print(f"[步骤1] 加载边侧 {args.edge_id} 的数据...")
    print(f"  数据路径: {args.data_path}")
    train_loader, test_loader = load_edge_data(args.data_path, args.dataset_type, args.batch_size)
    
    # 2. 创建学生模型
    print(f"\n[步骤2] 创建学生模型...")
    edge_model = create_model_by_type(args.edge_model, args.num_classes, args.dataset_type)
    
    # 3. 下载或加载教师模型
    print(f"\n[步骤3] 获取教师模型...")
    
    # 检查本地是否已有教师模型
    if os.path.exists(args.teacher_model_path):
        print(f"[边侧 {args.edge_id}] 检测到本地教师模型: {args.teacher_model_path}")
        print(f"[边侧 {args.edge_id}] 跳过下载，直接使用本地模型")
        
        # 添加模块支持
        dataset_modules = ['readdata_rml2016', 'readdata_radar', 'readdata_radioml',
                           'readdata_reii', 'readdata_25', 'readdata_link11']
        
        for module_name in dataset_modules:
            if module_name not in sys.modules:
                try:
                    module = __import__(f'utils.{module_name}', fromlist=[module_name])
                    sys.modules[module_name] = module
                except:
                    pass
        
        try:
            teacher_checkpoint = torch.load(args.teacher_model_path, map_location='cpu')
        except Exception as e:
            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module in dataset_modules:
                        module = f'utils.{module}'
                    return super().find_class(module, name)
            
            with open(args.teacher_model_path, 'rb') as f:
                teacher_checkpoint = CPU_Unpickler(f).load()
        
        if isinstance(teacher_checkpoint, dict) and 'model_state_dict' in teacher_checkpoint:
            teacher_model_state = teacher_checkpoint['model_state_dict']
            teacher_model_architecture = teacher_checkpoint.get('model_architecture', f'complex_resnet50_{args.dataset_type}')
        else:
            teacher_model_state = teacher_checkpoint
            teacher_model_architecture = f'complex_resnet50_{args.dataset_type}'
    else:
        # 从云侧下载教师模型
        print(f"[边侧 {args.edge_id}] 本地无教师模型，从云侧下载...")
        teacher_model_state, teacher_model_architecture = download_teacher_model(
            edge_id=args.edge_id,
            cloud_host=args.cloud_host,
            cloud_port=args.cloud_port,
            save_path=args.teacher_model_path
        )
    
    print(f"教师模型: {teacher_model_architecture}")
    
    # 4. 执行轻量化
    print(f"\n[步骤4] 开始轻量化...")
    kd_config = {
        'kd_distill': args.kd_distill,
        'kd_temperature': args.kd_temperature,
        'kd_alpha': args.kd_alpha,
        'kd_epochs': args.kd_epochs,
        'optimizer': 'adam',
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'lr_scheduler': 'cosine',
        'dataset_type': args.dataset_type,
        'num_classes': args.num_classes,
    }
    
    distilled_state = DistillationHelper.distill_from_teacher(
        edge_model=edge_model,
        teacher_model_state=teacher_model_state,
        teacher_model_architecture=teacher_model_architecture,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        config=kd_config,
        edge_id=args.edge_id - 1,
        create_model_func=create_model_by_type,
        local_test_loader=None
    )
    
    # 5. 保存蒸馏后的模型
    kd_model_path = f'{args.save_dir}/edge_{args.edge_id}_kd_model.pth'
    torch.save({
        'model_state_dict': distilled_state,
        'edge_id': args.edge_id,
        'model_architecture': args.edge_model
    }, kd_model_path)
    
    print(f"\n{'='*70}")
    print(f"[完成] 轻量化完成！")
    print(f"模型已保存: {kd_model_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
云侧联邦学习服务器脚本

功能：
1. 启动联邦学习服务器，监听边侧连接
2. 分发全局模型给边侧
3. 接收边侧更新并聚合
4. 保存最终全局模型

使用示例：
    python run/cloud/run_cloud_federated.py --dataset_type link11 --num_edges 3 --num_rounds 10
"""
import sys
import os
import argparse
import torch
import signal
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed.project import create_model_by_type, ProjectCloud
from run.network_utils import NetworkCloud


def main():
    parser = argparse.ArgumentParser(description='云侧联邦学习服务器')
    
    # 网络参数
    parser.add_argument('--cloud_port', type=int, default=9999, help='云侧端口')
    parser.add_argument('--rate_limit', type=float, default=10.0, help='网络速率限制（MB/s，默认10）')
    parser.add_argument('--download_timeout', type=int, default=120, help='下载阶段超时（秒，默认120）')
    parser.add_argument('--upload_timeout', type=int, default=300, help='上传阶段超时（秒，默认300）')
    
    # 联邦学习参数
    parser.add_argument('--num_edges', type=int, default=2, help='边侧数量')
    parser.add_argument('--num_rounds', type=int, default=5, help='联邦学习轮次')
    
    # 模型参数
    parser.add_argument('--dataset_type', type=str, default='link11', help='数据集类型')
    parser.add_argument('--num_classes', type=int, default=7, help='类别数')
    parser.add_argument('--edge_model', type=str, default=None,
                        help='边侧模型类型（默认：real_resnet20_{dataset_type}）')
    
    # 模型路径
    parser.add_argument('--teacher_model_path', type=str, default=None,
                        help='教师模型路径（默认：run/cloud/pth/{dataset_type}/teacher_model.pth）')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='结果保存目录（默认：run/cloud/results/{dataset_type}）')
    
    args = parser.parse_args()
    
    # 设置默认值
    if args.edge_model is None:
        args.edge_model = f'real_resnet20_{args.dataset_type}'
    
    if args.teacher_model_path is None:
        args.teacher_model_path = f'run/cloud/pth/{args.dataset_type}/teacher_model.pth'
    
    if args.save_dir is None:
        args.save_dir = f'run/cloud/results/{args.dataset_type}'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"[云侧联邦学习] 服务器启动")
    print(f"{'='*70}")
    print(f"数据集: {args.dataset_type}")
    print(f"边侧数量: {args.num_edges}")
    print(f"训练轮次: {args.num_rounds}")
    print(f"边侧模型: {args.edge_model}")
    print(f"端口: {args.cloud_port}")
    if args.rate_limit:
        print(f"网络限速: {args.rate_limit} MB/s")
    print(f"下载超时: {args.download_timeout}秒")
    print(f"上传超时: {args.upload_timeout}秒")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 检查教师模型是否存在
    if not os.path.exists(args.teacher_model_path):
        print(f"错误: 教师模型不存在: {args.teacher_model_path}")
        print("请先运行 run_cloud_pretrain.py 训练教师模型")
        return
    
    print(f"[步骤1] 教师模型路径: {args.teacher_model_path}")
    
    # 2. 创建初始学生模型（用于联邦学习）
    print("\n[步骤2] 创建初始学生模型...")
    initial_client_model = create_model_by_type(args.edge_model, args.num_classes, args.dataset_type)
    global_model_state = initial_client_model.state_dict()
    print(f"学生模型架构: {args.edge_model}")
    
    # 3. 创建 ProjectCloud（用于模型聚合）
    print("\n[步骤3] 初始化联邦学习服务器...")
    config = {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
    }
    
    # 创建一个简单的 ProjectCloud 用于聚合
    class FederatedCloud:
        def __init__(self, global_model_state):
            self.global_model_state = global_model_state
        
        def aggregate_models(self, edge_updates, edge_weights):
            """聚合边侧模型"""
            aggregated_state = {}
            
            for key in self.global_model_state.keys():
                aggregated_state[key] = sum(
                    edge_weights[i] * edge_updates[i][key]
                    for i in range(len(edge_updates))
                )
            
            return aggregated_state
    
    project_cloud = FederatedCloud(global_model_state)
    
    # 4. 启动网络服务器
    print("\n[步骤4] 启动网络服务器...")
    network_cloud = NetworkCloud(project_cloud, port=args.cloud_port, rate_limit_mbps=args.rate_limit)
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print('\n[云侧] 收到退出信号，正在关闭...')
        network_cloud.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 5. 运行联邦学习
    print("\n[步骤5] 开始联邦学习...")
    network_cloud.run(args.num_edges, args.num_rounds, args.download_timeout, args.upload_timeout)
    
    # 6. 保存最终模型
    final_model_path = f'{args.save_dir}/final_global_model.pth'
    torch.save({
        'model_state_dict': project_cloud.global_model_state,
        'model_architecture': args.edge_model,
        'num_classes': args.num_classes,
        'dataset_type': args.dataset_type,
        'num_rounds': args.num_rounds
    }, final_model_path)
    
    print(f"\n{'='*70}")
    print(f"[完成] 联邦学习完成！")
    print(f"最终模型已保存: {final_model_path}")
    print(f"云侧将继续保持在线状态...")
    print(f"按 Ctrl+C 退出")
    print(f"{'='*70}\n")
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[云侧] 收到键盘中断")
        network_cloud.close()
        print("[云侧] 已关闭")


if __name__ == '__main__':
    main()

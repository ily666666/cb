#!/usr/bin/env python3
"""
数据预划分脚本 - 将数据集物理分割成云侧和边侧的独立文件

使用方法：
    python run/prepare_data_splits.py --dataset_type link11 --num_edges 3

功能：
    1. 加载完整数据集
    2. 按照狄利克雷分布或类别重叠方式划分数据
    3. 将云侧和每个边侧的数据保存为独立的pkl文件
    4. 云侧和边侧启动时直接加载对应的pkl文件，无需索引

优势：
    - 云侧启动时无需数据划分，直接加载预划分的数据
    - 边侧只加载自己的数据文件，内存占用最小
    - 数据划分只需执行一次，可重复使用
"""
import sys
import os
import argparse
import pickle
import numpy as np
from collections import defaultdict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 数据集配置（绑定数据集类型、路径和类别数）
DATASET_CONFIG = {
    'link11': {
        'data_path': 'run/data/link11.pkl',
        'num_classes': 7,
        'description': 'Link11 - 7类雷达发射机识别'
    },
    'rml2016': {
        'data_path': 'run/data/rml2016.pkl',
        'num_classes': 6,
        'description': 'RML2016 - 6类调制识别'
    },
    'radar': {
        'data_path': 'run/data/radar.mat',
        'num_classes': 7,
        'description': 'Radar - 7类雷达个体识别'
    }
}


def dirichlet_split_indices(labels, num_edges, alpha, seed=42):
    """使用狄利克雷分布划分数据索引"""
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    
    # 按类别收集样本索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # 初始化边侧索引列表
    edge_indices = [[] for _ in range(num_edges)]
    
    # 对每个类别使用狄利克雷分布分配
    for c, indices in enumerate(class_indices):
        rng.shuffle(indices)
        
        # 使用狄利克雷分布生成每个边侧获得该类别样本的比例
        proportions = rng.dirichlet(np.repeat(alpha, num_edges))
        
        # 按比例分配样本
        proportions = np.cumsum(proportions)
        split_points = (proportions * len(indices)).astype(int)[:-1]
        
        # 分割样本索引
        splits = np.split(indices, split_points)
        
        for edge_id, split in enumerate(splits):
            edge_indices[edge_id].extend(split.tolist())
    
    return edge_indices


def load_and_split_data(data_path, dataset_type, num_edges, num_classes, 
                        partition_method='dirichlet', dirichlet_alpha=0.5, 
                        cloud_ratio=0.3, seed=42):
    """
    加载并划分数据集
    
    Returns:
        cloud_data: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        edge_data_list: [{'train': (X, y), 'val': (X, y), 'test': (X, y)}, ...]
    """
    print(f"\n{'='*70}")
    print(f"数据预划分 - {dataset_type} 数据集")
    print(f"{'='*70}")
    print(f"边侧数量: {num_edges}")
    print(f"划分方法: {partition_method}")
    if partition_method == 'dirichlet':
        print(f"Dirichlet Alpha: {dirichlet_alpha}")
    print(f"云侧数据比例: {cloud_ratio}")
    print(f"{'='*70}\n")
    
    # 1. 加载完整数据集
    print("步骤1: 加载完整数据集...")
    
    if dataset_type == 'link11':
        from utils.readdata_link11 import Link11Dataset
        full_train_dataset = Link11Dataset(pkl_path=data_path, split='train', seed=seed)
        full_val_dataset = Link11Dataset(pkl_path=data_path, split='val', seed=seed)
        full_test_dataset = Link11Dataset(pkl_path=data_path, split='test', seed=seed)
    elif dataset_type == 'rml2016':
        from utils.readdata_rml2016 import RML2016Dataset
        full_train_dataset = RML2016Dataset(pkl_path=data_path, split='train', seed=seed)
        full_val_dataset = RML2016Dataset(pkl_path=data_path, split='val', seed=seed)
        full_test_dataset = RML2016Dataset(pkl_path=data_path, split='test', seed=seed)
    elif dataset_type == 'radar':
        # Radar可以直接使用.mat文件
        from utils.readdata_radar import RadarDataset
        full_train_dataset = RadarDataset(mat_path=data_path, split='train', seed=seed)
        full_val_dataset = RadarDataset(mat_path=data_path, split='val', seed=seed)
        full_test_dataset = RadarDataset(mat_path=data_path, split='test', seed=seed)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # 提取数据和标签
    if dataset_type == 'radar':
        # Radar数据集使用不同的属性名
        train_X = full_train_dataset.X[full_train_dataset.indices]
        train_y = full_train_dataset.Y_adjusted[full_train_dataset.indices]
        val_X = full_val_dataset.X[full_val_dataset.indices]
        val_y = full_val_dataset.Y_adjusted[full_val_dataset.indices]
        test_X = full_test_dataset.X[full_test_dataset.indices]
        test_y = full_test_dataset.Y_adjusted[full_test_dataset.indices]
    else:
        # Link11和RML2016使用标准属性名
        train_X = full_train_dataset.signals
        train_y = full_train_dataset.labels
        val_X = full_val_dataset.signals
        val_y = full_val_dataset.labels
        test_X = full_test_dataset.signals
        test_y = full_test_dataset.labels
    
    print(f"✅ 数据加载完成")
    print(f"   训练集: {len(train_X)} 样本")
    print(f"   验证集: {len(val_X)} 样本")
    print(f"   测试集: {len(test_X)} 样本")
    
    # 2. 分配云侧数据（从每个类别采样指定比例）
    print(f"\n步骤2: 分配云侧数据（{cloud_ratio*100:.0f}%）...")
    
    rng = np.random.RandomState(seed)
    
    cloud_train_indices = []
    cloud_val_indices = []
    cloud_test_indices = []
    remaining_train_indices = []
    remaining_val_indices = []
    remaining_test_indices = []
    
    for class_id in range(num_classes):
        # 训练数据
        class_train_idx = np.where(train_y == class_id)[0]
        rng.shuffle(class_train_idx)
        server_count = max(1, int(len(class_train_idx) * cloud_ratio))
        cloud_train_indices.extend(class_train_idx[:server_count].tolist())
        remaining_train_indices.extend(class_train_idx[server_count:].tolist())
        
        # 验证数据
        class_val_idx = np.where(val_y == class_id)[0]
        rng.shuffle(class_val_idx)
        server_count = max(1, int(len(class_val_idx) * cloud_ratio))
        cloud_val_indices.extend(class_val_idx[:server_count].tolist())
        remaining_val_indices.extend(class_val_idx[server_count:].tolist())
        
        # 测试数据
        class_test_idx = np.where(test_y == class_id)[0]
        rng.shuffle(class_test_idx)
        server_count = max(1, int(len(class_test_idx) * cloud_ratio))
        cloud_test_indices.extend(class_test_idx[:server_count].tolist())
        remaining_test_indices.extend(class_test_idx[server_count:].tolist())
    
    # 云侧数据
    cloud_data = {
        'train': (train_X[cloud_train_indices], train_y[cloud_train_indices]),
        'val': (val_X[cloud_val_indices], val_y[cloud_val_indices]),
        'test': (test_X[cloud_test_indices], test_y[cloud_test_indices])
    }
    
    print(f"✅ 云侧数据分配完成")
    print(f"   训练: {len(cloud_train_indices)} 样本")
    print(f"   验证: {len(cloud_val_indices)} 样本")
    print(f"   测试: {len(cloud_test_indices)} 样本")
    
    # 3. 分配边侧数据
    print(f"\n步骤3: 分配边侧数据（{partition_method}）...")
    
    remaining_train_y = train_y[remaining_train_indices]
    remaining_val_y = val_y[remaining_val_indices]
    remaining_test_y = test_y[remaining_test_indices]
    
    # 使用狄利克雷分布划分训练集
    edge_train_local_indices = dirichlet_split_indices(
        remaining_train_y, num_edges, dirichlet_alpha, seed=seed
    )
    
    # 获取每个边侧在训练集中的比例
    edge_train_proportions = [len(indices) / len(remaining_train_y) for indices in edge_train_local_indices]
    
    # 用相同的比例分配验证集和测试集
    edge_val_local_indices = [[] for _ in range(num_edges)]
    edge_test_local_indices = [[] for _ in range(num_edges)]
    
    # 验证集
    for class_id in range(num_classes):
        class_mask = remaining_val_y == class_id
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) > 0:
            rng.shuffle(class_indices)
            start_idx = 0
            for edge_id in range(num_edges):
                proportion = edge_train_proportions[edge_id]
                end_idx = start_idx + int(len(class_indices) * proportion)
                if edge_id == num_edges - 1:
                    end_idx = len(class_indices)
                edge_val_local_indices[edge_id].extend(class_indices[start_idx:end_idx].tolist())
                start_idx = end_idx
    
    # 测试集
    for class_id in range(num_classes):
        class_mask = remaining_test_y == class_id
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) > 0:
            rng.shuffle(class_indices)
            start_idx = 0
            for edge_id in range(num_edges):
                proportion = edge_train_proportions[edge_id]
                end_idx = start_idx + int(len(class_indices) * proportion)
                if edge_id == num_edges - 1:
                    end_idx = len(class_indices)
                edge_test_local_indices[edge_id].extend(class_indices[start_idx:end_idx].tolist())
                start_idx = end_idx
    
    # 转换为原始数据集索引
    edge_train_indices = [[remaining_train_indices[idx] for idx in local_idx] 
                            for local_idx in edge_train_local_indices]
    edge_val_indices = [[remaining_val_indices[idx] for idx in local_idx] 
                          for local_idx in edge_val_local_indices]
    edge_test_indices = [[remaining_test_indices[idx] for idx in local_idx] 
                           for local_idx in edge_test_local_indices]
    
    # 创建边侧数据列表
    edge_data_list = []
    for i in range(num_edges):
        edge_data = {
            'train': (train_X[edge_train_indices[i]], train_y[edge_train_indices[i]]),
            'val': (val_X[edge_val_indices[i]], val_y[edge_val_indices[i]]),
            'test': (test_X[edge_test_indices[i]], test_y[edge_test_indices[i]])
        }
        edge_data_list.append(edge_data)
        
        # 统计类别分布
        unique, counts = np.unique(train_y[edge_train_indices[i]], return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"   边侧 {i+1}: 训练={len(edge_train_indices[i])}, 验证={len(edge_val_indices[i])}, 测试={len(edge_test_indices[i])}, 类别分布={label_dist}")
    
    print(f"✅ 边侧数据分配完成")
    
    return cloud_data, edge_data_list


def save_split_data(cloud_data, edge_data_list, dataset_type, output_dir='run/data/splits'):
    """保存划分后的数据"""
    print(f"\n步骤4: 保存划分后的数据...")
    
    # 创建输出目录
    split_dir = f'{output_dir}/{dataset_type}'
    os.makedirs(split_dir, exist_ok=True)
    
    # 保存云侧数据
    server_file = f'{split_dir}/cloud_data.pkl'
    with open(server_file, 'wb') as f:
        pickle.dump(cloud_data, f)
    print(f"✅ 云侧数据已保存: {server_file}")
    
    # 保存边侧数据
    for i, edge_data in enumerate(edge_data_list):
        edge_file = f'{split_dir}/edge_{i+1}_data.pkl'
        with open(edge_file, 'wb') as f:
            pickle.dump(edge_data, f)
        print(f"✅ 边侧 {i+1} 数据已保存: {edge_file}")
    
    print(f"\n{'='*70}")
    print(f"数据预划分完成！")
    print(f"输出目录: {split_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='数据预划分脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
数据集配置（自动绑定）:
  link11   : {DATASET_CONFIG['link11']['description']} ({DATASET_CONFIG['link11']['num_classes']}类)
  rml2016  : {DATASET_CONFIG['rml2016']['description']} ({DATASET_CONFIG['rml2016']['num_classes']}类)
  radar    : {DATASET_CONFIG['radar']['description']} ({DATASET_CONFIG['radar']['num_classes']}类)

示例:
  python ker/prepare_data_splits.py --dataset_type link11 --num_edges 3
  python run/prepare_docdata_splits.py --dataset_type radar --num_edges 5 --dirichlet_alpha 0.3
        """
    )
    
    parser.add_argument('--dataset_type', type=str, 
                        default='radar',
                        choices=['link11', 'rml2016', 'radar'], 
                        help='数据集类型（自动设置路径和类别数）')
    parser.add_argument('--data_path', type=str, 
                        default=None,
                        help='数据集路径（可选，不填则根据 dataset_type 自动选择）')
    parser.add_argument('--num_edges', type=int, default=2, 
                        help='边侧数量')
    parser.add_argument('--partition_method', type=str, default='dirichlet', 
                        choices=['dirichlet'], 
                        help='数据划分方法')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, 
                        help='Dirichlet分布参数（越小各边数据差异越大，0.3=强异构）')
    parser.add_argument('--cloud_ratio', type=float, default=0.8, 
                        help='云侧数据比例（默认0.8，云侧80%%用于预训练+蒸馏，边侧20%%用于联邦学习）')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--output_dir', type=str, default='dataset/splits', 
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 从配置中获取数据集信息
    dataset_config = DATASET_CONFIG[args.dataset_type]
    num_classes = dataset_config['num_classes']
    
    # 智能路径选择：如果用户没有指定 data_path，根据 dataset_type 自动选择
    if args.data_path is None:
        data_path = dataset_config['data_path']
    else:
        data_path = args.data_path
    
    # 打印配置
    print(f"\n{'='*70}")
    print(f"数据预划分配置")
    print(f"{'='*70}")
    print(f"  数据集类型: {args.dataset_type}")
    print(f"  数据集描述: {dataset_config['description']}")
    print(f"  数据路径: {data_path}")
    print(f"  类别数量: {num_classes} (自动设置)")
    print(f"  边侧数量: {args.num_edges}")
    print(f"  划分方法: {args.partition_method}")
    print(f"  Dirichlet Alpha: {args.dirichlet_alpha}")
    print(f"  云侧比例: {args.cloud_ratio}")
    print(f"  随机种子: {args.seed}")
    print(f"  输出目录: {args.output_dir}")
    print(f"{'='*70}")
    
    # 加载并划分数据
    cloud_data, edge_data_list = load_and_split_data(
        data_path=data_path,
        dataset_type=args.dataset_type,
        num_edges=args.num_edges,
        num_classes=num_classes,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        cloud_ratio=args.cloud_ratio,
        seed=args.seed
    )
    
    # 保存划分后的数据
    save_split_data(cloud_data, edge_data_list, args.dataset_type, args.output_dir)


if __name__ == '__main__':
    main()

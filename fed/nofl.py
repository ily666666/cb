"""
No Federated Learning (NoFL) Mode
独立边侧训练模式 - 用于与联邦学习对比

特点:
1. 数据分割与联邦学习相同（30%云侧数据不用，70%边侧数据按策略分割）
2. 每个边侧独立训练，无全局聚合
3. 无知识蒸馏
4. 保存各边侧的独立模型和结果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import copy
import csv
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from trainer import Trainer


def create_model_by_type(model_name, num_classes, dataset_type='ads'):
    """
    根据数据集类型创建对应的模型
    
    Returns:
        对应的模型实例
    """
    if dataset_type == 'radioml':
        if model_name == 'complex_resnet50_radioml':
            from model.complex_resnet50_radioml import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_radioml import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'reii':
        if model_name == 'complex_resnet50_reii':
            from model.complex_resnet50_reii import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_reii import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'radar':
        if model_name == 'complex_resnet50_radar':
            from model.complex_resnet50_radar import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_radar_with_attention':
            from model.complex_resnet50_radar_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_radar_with_attention_1000':
            from model.complex_resnet50_radar_with_attention_1000 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_radar import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'rml2016':
        if model_name == 'complex_resnet50_rml2016':
            from model.complex_resnet50_rml2016 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_rml2016 import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'link11':
        if model_name == 'complex_resnet50_link11':
            from model.complex_resnet50_link11 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_link11 import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    else:
        # ADS数据集专用模型
        if model_name == 'real_resnet20_ads':
            from model.real_resnet20_ads import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.complex_resnet50_ads import CombinedModel
            return CombinedModel(num_classes=num_classes)


def assign_class_subsets_project(num_classes, num_clients, hetero_level):
    """
    基于异构水平的类别重叠划分（与project模式完全相同）
    """
    assert 0.0 <= hetero_level <= 1.0
    rng = np.random.RandomState(42)

    r_float = 1.0 + (num_clients - 1) * (1.0 - hetero_level)
    r_floor = int(np.floor(r_float))
    r_frac = r_float - r_floor

    client_subsets = [set() for _ in range(num_clients)]
    client_load = np.zeros(num_clients, dtype=int)

    classes = list(range(num_classes))
    rng.shuffle(classes)

    for c in classes:
        r_c = r_floor + (1 if rng.rand() < r_frac else 0)
        r_c = max(1, min(num_clients, r_c))

        candidates = list(range(num_clients))
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: (client_load[i], rng.rand()))
        chosen = candidates[:r_c]
        for cid in chosen:
            client_subsets[cid].add(c)
            client_load[cid] += 1

    empty_clients = [i for i, s in enumerate(client_subsets) if len(s) == 0]
    if empty_clients:
        class_freq = {c: 0 for c in range(num_classes)}
        for s in client_subsets:
            for c in s:
                class_freq[c] += 1
        rare_classes = sorted(class_freq.keys(), key=lambda x: (class_freq[x], rng.rand()))
        rc_idx = 0
        for cid in empty_clients:
            while rc_idx < len(rare_classes):
                c = rare_classes[rc_idx]
                rc_idx += 1
                if c not in client_subsets[cid]:
                    client_subsets[cid].add(c)
                    client_load[cid] += 1
                    break
            if len(client_subsets[cid]) == 0:
                c = rng.randint(0, num_classes)
                client_subsets[cid].add(c)
                client_load[cid] += 1

    return client_subsets


def dirichlet_split_indices_project(labels, num_clients, alpha, seed):
    """
    使用狄利克雷分布划分数据索引（Project模式）
    
    Args:
        labels: 所有样本的标签数组
        num_clients: 边侧数量
        alpha: 狄利克雷分布参数
        seed: 随机种子
        
    Returns:
        client_indices: list of lists, 每个边侧的样本索引
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    
    # 按类别收集样本索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # 初始化边侧索引列表
    client_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别使用狄利克雷分布分配
    for c, indices in enumerate(class_indices):
        rng.shuffle(indices)
        
        # 使用狄利克雷分布生成每个边侧获得该类别样本的比例
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        
        # 按比例分配样本
        proportions = np.cumsum(proportions)
        split_points = (proportions * len(indices)).astype(int)[:-1]
        
        # 分割样本索引
        splits = np.split(indices, split_points)
        
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    return client_indices


def split_data_for_nofl(data_path, batch_size, num_workers, num_classes, num_clients, 
                        partition_method, hetero_level, dirichlet_alpha, server_ratio=0.3, 
                        dataset_type='ads', snr_filter=None, seed=42,
                        add_noise=False, noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
    """
    为NoFL模式划分数据集 - 完全复制project.py的split_data_for_project逻辑
    返回边侧数据加载器和全局测试加载器（不返回云侧加载器）
    """
    from torch.utils.data import Subset, DataLoader
    import numpy as np
    from utils.readdata_25 import subDataset
    from utils.readdata_radioml import RadioMLDataset
    from collections import defaultdict

    # 1. 根据数据集类型加载数据
    if dataset_type == 'radioml':
        full_train_dataset = RadioMLDataset(datapath=data_path, split='train', transform=None, snr_filter=snr_filter)
        full_val_dataset = RadioMLDataset(datapath=data_path, split='valid', transform=None, snr_filter=snr_filter)
        full_test_dataset = RadioMLDataset(datapath=data_path, split='test', transform=None, snr_filter=snr_filter)
    elif dataset_type == 'reii':
        from utils.readdata_reii import REIIDataset
        full_train_dataset = REIIDataset(datapath=data_path, split='train', transform=None)
        full_val_dataset = REIIDataset(datapath=data_path, split='valid', transform=None)
        full_test_dataset = REIIDataset(datapath=data_path, split='test', transform=None)
    elif dataset_type == 'radar':
        from utils.readdata_radar import RadarDataset
        full_train_dataset = RadarDataset(mat_path=data_path, split='train', seed=42)
        full_val_dataset = RadarDataset(mat_path=data_path, split='val', seed=42)
        full_test_dataset = RadarDataset(mat_path=data_path, split='test', seed=42)
    elif dataset_type == 'rml2016':
        from utils.readdata_rml2016 import RML2016Dataset
        # 训练、验证、测试均可按需加入噪声
        full_train_dataset = RML2016Dataset(
            pkl_path=data_path, split='train', seed=42,
            add_noise=add_noise, noise_type=noise_type,
            noise_snr_db=noise_snr_db, noise_factor=noise_factor
        )
        full_val_dataset = RML2016Dataset(
            pkl_path=data_path, split='val', seed=42,
            add_noise=add_noise, noise_type=noise_type,
            noise_snr_db=noise_snr_db, noise_factor=noise_factor
        )
        full_test_dataset = RML2016Dataset(
            pkl_path=data_path, split='test', seed=42,
            add_noise=add_noise, noise_type=noise_type,
            noise_snr_db=noise_snr_db, noise_factor=noise_factor
        )
    elif dataset_type == 'link11':
        from utils.readdata_link11 import Link11Dataset
        # 训练、验证、测试均可按需加入噪声
        full_train_dataset = Link11Dataset(
            pkl_path=data_path, split='train', seed=42,
            add_noise=add_noise, noise_type=noise_type,
            noise_snr_db=noise_snr_db, noise_factor=noise_factor
        )
        full_val_dataset = Link11Dataset(
            pkl_path=data_path, split='val', seed=42,
            add_noise=add_noise, noise_type=noise_type,
            noise_snr_db=noise_snr_db, noise_factor=noise_factor
        )
        full_test_dataset = Link11Dataset(
            pkl_path=data_path, split='test', seed=42,
            add_noise=add_noise, noise_type=noise_type,
            noise_snr_db=noise_snr_db, noise_factor=noise_factor
        )
    else:
        full_train_dataset = subDataset(datapath=data_path, split='train', transform=None, allowed_classes=None)
        full_val_dataset = subDataset(datapath=data_path, split='valid', transform=None, allowed_classes=None)
        full_test_dataset = subDataset(datapath=data_path, split='test', transform=None, allowed_classes=None)

    rng = np.random.RandomState(seed)

    # 2. 获取所有标签
    def get_all_labels(dataset):
        """获取数据集的所有标签"""
        if hasattr(dataset, 'file_path_label'):
            return np.array([label for _, label in dataset.file_path_label])
        elif hasattr(dataset, 'samples'):
            return np.array([s['label'] for s in dataset.samples])
        elif hasattr(dataset, 'sample_meta'):
            return np.array([m['label'] for m in dataset.sample_meta])
        else:
            return np.array([dataset[i][1] for i in range(len(dataset))])

    train_labels = get_all_labels(full_train_dataset)
    val_labels = get_all_labels(full_val_dataset)
    test_labels = get_all_labels(full_test_dataset)

    # 3. 分配云侧数据（从每个类别采样指定比例）
    print("\n=== 云侧数据分配（分层采样）===")
    server_train_indices = []
    server_val_indices = []
    server_test_indices = []
    remaining_train_indices = []
    remaining_val_indices = []
    remaining_test_indices = []
    remaining_train_labels = []
    remaining_val_labels = []
    remaining_test_labels = []

    for class_id in range(num_classes):
        # 训练数据
        class_train_idx = np.where(train_labels == class_id)[0]
        rng.shuffle(class_train_idx)
        server_count = max(1, int(len(class_train_idx) * server_ratio))
        server_train_indices.extend(class_train_idx[:server_count].tolist())
        remaining_train_indices.extend(class_train_idx[server_count:].tolist())
        remaining_train_labels.extend([class_id] * (len(class_train_idx) - server_count))
        
        # 验证数据
        class_val_idx = np.where(val_labels == class_id)[0]
        rng.shuffle(class_val_idx)
        server_count = max(1, int(len(class_val_idx) * server_ratio))
        server_val_indices.extend(class_val_idx[:server_count].tolist())
        remaining_val_indices.extend(class_val_idx[server_count:].tolist())
        remaining_val_labels.extend([class_id] * (len(class_val_idx) - server_count))
        
        # 测试数据
        class_test_idx = np.where(test_labels == class_id)[0]
        rng.shuffle(class_test_idx)
        server_count = max(1, int(len(class_test_idx) * server_ratio))
        server_test_indices.extend(class_test_idx[:server_count].tolist())
        remaining_test_indices.extend(class_test_idx[server_count:].tolist())
        remaining_test_labels.extend([class_id] * (len(class_test_idx) - server_count))
        
        print(f"  类别 {class_id}: 训练 {len(class_train_idx)} (云侧 {int(len(class_train_idx)*server_ratio)}, 边侧 {len(class_train_idx)-int(len(class_train_idx)*server_ratio)})")

    remaining_train_labels = np.array(remaining_train_labels)
    remaining_val_labels = np.array(remaining_val_labels)
    remaining_test_labels = np.array(remaining_test_labels)

    print(f"云侧数据 - 训练: {len(server_train_indices)}, 验证: {len(server_val_indices)}")

    # 4. 边侧数据分配
    if partition_method == 'dirichlet':
        # 狄利克雷分布分配 - 改进版本
        # 步骤：用Dirichlet生成比例，然后分别应用到train/val/test
        print(f"\n=== 边侧数据分配（狄利克雷分布 alpha={dirichlet_alpha}）===")
        
        # 第一步：用Dirichlet生成分配比例（基于训练集标签）
        # 这样保证train/val/test的分配比例一致
        client_train_local_indices = dirichlet_split_indices_project(
            remaining_train_labels, num_clients, dirichlet_alpha, seed=seed
        )
        
        # 获取每个边侧在训练集中的比例
        client_train_proportions = [len(indices) / len(remaining_train_labels) for indices in client_train_local_indices]
        
        # 第二步：用相同的比例分配验证集和测试集
        client_val_local_indices = [[] for _ in range(num_clients)]
        client_test_local_indices = [[] for _ in range(num_clients)]
        
        # 对验证集按照相同的比例分配
        val_indices_array = np.array(remaining_val_indices)
        val_labels_array = np.array(remaining_val_labels)
        
        for class_id in range(num_classes):
            class_mask = val_labels_array == class_id
            class_indices = val_indices_array[class_mask]
            
            if len(class_indices) > 0:
                rng.shuffle(class_indices)
                
                # 按照train中的比例分配
                start_idx = 0
                for client_id in range(num_clients):
                    proportion = client_train_proportions[client_id]
                    end_idx = start_idx + int(len(class_indices) * proportion)
                    if client_id == num_clients - 1:  # 最后一个边侧获得剩余的
                        end_idx = len(class_indices)
                    
                    client_val_local_indices[client_id].extend(class_indices[start_idx:end_idx].tolist())
                    start_idx = end_idx
        
        # 对测试集按照相同的比例分配
        test_indices_array = np.array(remaining_test_indices)
        test_labels_array = np.array(remaining_test_labels)
        
        for class_id in range(num_classes):
            class_mask = test_labels_array == class_id
            class_indices = test_indices_array[class_mask]
            
            if len(class_indices) > 0:
                rng.shuffle(class_indices)
                
                # 按照train中的比例分配
                start_idx = 0
                for client_id in range(num_clients):
                    proportion = client_train_proportions[client_id]
                    end_idx = start_idx + int(len(class_indices) * proportion)
                    if client_id == num_clients - 1:  # 最后一个边侧获得剩余的
                        end_idx = len(class_indices)
                    
                    client_test_local_indices[client_id].extend(class_indices[start_idx:end_idx].tolist())
                    start_idx = end_idx
        
        # 转换为原始数据集索引
        client_train_indices = [[remaining_train_indices[idx] for idx in local_idx] 
                                for local_idx in client_train_local_indices]
        # val和test的local_indices已经是原始索引，直接使用
        client_val_indices = client_val_local_indices
        client_test_indices = client_test_local_indices
        
        # 统计每个边侧的类别分布（基于train集）
        for i in range(num_clients):
            client_train_labels = remaining_train_labels[client_train_local_indices[i]]
            unique, counts = np.unique(client_train_labels, return_counts=True)
            label_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"边侧 {i+1} - 训练: {len(client_train_indices[i])} 样本, 验证: {len(client_val_indices[i])} 样本, 测试: {len(client_test_indices[i])} 样本, 类别分布: {label_dist}")
    
    else:  # class_overlap方法
        # 类别重叠分配（原始方法）
        print(f"\n=== 边侧数据分配（类别重叠 hetero_level={hetero_level}）===")
        
        # 分配边侧类别子集
        client_class_subsets = assign_class_subsets_project(
            num_classes=num_classes,
            num_clients=num_clients,
            hetero_level=hetero_level
        )
        for i, subset in enumerate(client_class_subsets):
            print(f"边侧 {i+1} 分配的类别: {sorted(list(subset))}")
        
        # 收集剩余样本的类别索引
        remaining_train_class_samples = defaultdict(list)
        remaining_val_class_samples = defaultdict(list)
        remaining_test_class_samples = defaultdict(list)
        
        for idx, label in zip(remaining_train_indices, remaining_train_labels):
            remaining_train_class_samples[int(label)].append(idx)
        for idx, label in zip(remaining_val_indices, remaining_val_labels):
            remaining_val_class_samples[int(label)].append(idx)
        for idx, label in zip(remaining_test_indices, remaining_test_labels):
            remaining_test_class_samples[int(label)].append(idx)
        
        # 统计哪些边侧需要每个类别
        class_to_clients = defaultdict(list)
        for client_id, subset in enumerate(client_class_subsets):
            for class_id in subset:
                class_to_clients[class_id].append(client_id)
        
        # 为每个边侧分配样本索引
        client_train_indices = [[] for _ in range(num_clients)]
        client_val_indices = [[] for _ in range(num_clients)]
        client_test_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            clients_for_class = class_to_clients[class_id]
            
            if len(clients_for_class) > 0:
                # 训练数据分配
                if class_id in remaining_train_class_samples:
                    train_samples = remaining_train_class_samples[class_id]
                    if len(train_samples) > 0:
                        samples_per_client = len(train_samples) // len(clients_for_class)
                        remainder = len(train_samples) % len(clients_for_class)
                        
                        start_idx = 0
                        for i, client_id in enumerate(clients_for_class):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + samples_per_client + extra
                            client_train_indices[client_id].extend(train_samples[start_idx:end_idx])
                            start_idx = end_idx
                
                # 验证数据分配
                if class_id in remaining_val_class_samples:
                    val_samples = remaining_val_class_samples[class_id]
                    if len(val_samples) > 0:
                        samples_per_client = len(val_samples) // len(clients_for_class)
                        remainder = len(val_samples) % len(clients_for_class)
                        
                        start_idx = 0
                        for i, client_id in enumerate(clients_for_class):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + samples_per_client + extra
                            client_val_indices[client_id].extend(val_samples[start_idx:end_idx])
                            start_idx = end_idx
                
                # 测试数据分配
                if class_id in remaining_test_class_samples:
                    test_samples = remaining_test_class_samples[class_id]
                    if len(test_samples) > 0:
                        samples_per_client = len(test_samples) // len(clients_for_class)
                        remainder = len(test_samples) % len(clients_for_class)
                        
                        start_idx = 0
                        for i, client_id in enumerate(clients_for_class):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + samples_per_client + extra
                            client_test_indices[client_id].extend(test_samples[start_idx:end_idx])
                            start_idx = end_idx
        
        # 统计每个边侧的样本数
        for i in range(num_clients):
            print(f"边侧 {i+1} - 训练: {len(client_train_indices[i])} 样本, 验证: {len(client_val_indices[i])} 样本, 测试: {len(client_test_indices[i])} 样本")

    # 创建边侧数据加载器
    client_train_loaders = []
    client_val_loaders = []
    client_test_loaders = []
    
    # RadioML 数据集使用 drop_last=True 避免 batch_size=1 导致 BatchNorm 报错
    use_drop_last = (dataset_type == 'radioml' or dataset_type == 'radar' or dataset_type == 'rml2016')
    
    # 创建边侧数据加载器
    for i in range(num_clients):
        if len(client_train_indices[i]) > 0:
            client_train_subset = Subset(full_train_dataset, client_train_indices[i])
            client_train_loader = DataLoader(
                client_train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=use_drop_last
            )
        else:
            # 创建空的数据加载器
            client_train_loader = DataLoader([], batch_size=batch_size)
        
        if len(client_val_indices[i]) > 0:
            client_val_subset = Subset(full_val_dataset, client_val_indices[i])
            client_val_loader = DataLoader(
                client_val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=use_drop_last
            )
        else:
            client_val_loader = DataLoader([], batch_size=batch_size)
        
        if len(client_test_indices[i]) > 0:
            client_test_subset = Subset(full_test_dataset, client_test_indices[i])
            client_test_loader = DataLoader(
                client_test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False  # 测试集不drop_last
            )
        else:
            client_test_loader = DataLoader([], batch_size=batch_size)
        
        client_train_loaders.append(client_train_loader)
        client_val_loaders.append(client_val_loader)
        client_test_loaders.append(client_test_loader)
        
        print(f"边侧 {i+1} 样本分配 - 训练: {len(client_train_indices[i])}, 验证: {len(client_val_indices[i])}, 测试: {len(client_test_indices[i])}")

    # 创建全局测试集（完整100%测试集，用于公平对比）
    global_test_loader = DataLoader(
        full_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 测试集保留所有样本
    )

    # 验证无重叠
    total_client_train = sum(len(indices) for indices in client_train_indices)
    total_client_val = sum(len(indices) for indices in client_val_indices)
    total_client_test = sum(len(indices) for indices in client_test_indices)

    print(f"\n=== 数据划分总结（无样本重叠）===")
    print(f"云侧: {len(server_train_indices)} 训练 + {len(server_val_indices)} 验证 + {len(server_test_indices)} 测试")
    print(f"所有边侧: {total_client_train} 训练 + {total_client_val} 验证 + {total_client_test} 测试")
    print(f"全局测试集: {len(global_test_loader.dataset)} 样本（完整100%测试集，用于公平对比）")
    print(f"训练数据验证: 云侧({len(server_train_indices)}) + 边侧({total_client_train}) = {len(server_train_indices) + total_client_train} (原始: {len(full_train_dataset)})")
    print(f"验证数据验证: 云侧({len(server_val_indices)}) + 边侧({total_client_val}) = {len(server_val_indices) + total_client_val} (原始: {len(full_val_dataset)})")
    print(f"测试数据验证: 云侧({len(server_test_indices)}) + 边侧({total_client_test}) = {len(server_test_indices) + total_client_test} (原始: {len(full_test_dataset)})")

    return client_train_loaders, client_val_loaders, client_test_loaders, global_test_loader


class NoFLTrainer:
    """
    NoFL (No Federated Learning) 训练器
    每个边侧独立训练，无全局聚合
    """

    def __init__(self, client_models, client_train_loaders, client_val_loaders, 
                 client_test_loaders, global_test_loader, device, config):
        """
        初始化NoFL训练器
        
        Args:
            client_models: 边侧模型列表
            client_train_loaders: 边侧训练数据加载器列表
            client_val_loaders: 边侧验证数据加载器列表
            client_test_loaders: 边侧测试数据加载器列表
            global_test_loader: 全局测试数据加载器
            device: 设备
            config: 配置字典
        """
        self.client_models = client_models
        self.client_train_loaders = client_train_loaders
        self.client_val_loaders = client_val_loaders
        self.client_test_loaders = client_test_loaders
        self.global_test_loader = global_test_loader
        self.device = device
        self.config = config
        self.num_clients = len(client_models)

        # 为每个边侧创建训练器
        self.trainers = []
        for client_id in range(self.num_clients):
            trainer_config = copy.deepcopy(config)
            trainer_config['save_dir'] = os.path.join(
                config['save_dir'],
                f"client_{client_id+1:03d}"
            )
            os.makedirs(trainer_config['save_dir'], exist_ok=True)

            trainer = Trainer(
                model=client_models[client_id],
                train_loader=client_train_loaders[client_id],
                val_loader=client_val_loaders[client_id],
                test_loader=client_test_loaders[client_id],
                device=device,
                config=trainer_config
            )
            self.trainers.append(trainer)

    def train_nofl(self):
        """
        执行NoFL训练 - 每个边侧独立训练
        
        Returns:
            training_history: 训练历史字典
        """
        print("\n=== 开始NoFL训练 ===")
        print(f"边侧数量: {self.num_clients}")
        print(f"本地训练轮数: {self.config['epochs']}")

        start_time = time.time()
        training_history = {
            'client_results': [],
            'global_test_results': [],
            'training_time': 0
        }

        # 每个边侧独立训练
        for client_id in range(self.num_clients):
            print(f"\n{'='*60}")
            print(f"边侧 {client_id+1}/{self.num_clients} 训练")
            print(f"{'='*60}")

            # 训练该边侧
            train_losses, train_accs, val_losses, val_accs = self.trainers[client_id].train()

            # 测试该边侧
            test_loss, test_acc, test_f1, conf_matrix = self.trainers[client_id].test()

            print(f"\n边侧 {client_id+1} 最终结果:")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.2f}%")
            print(f"  Test F1 Score: {test_f1:.4f}")

            # 保存边侧结果
            client_result = {
                'client_id': client_id + 1,
                'test_loss': float(test_loss),
                'test_accuracy': float(test_acc),
                'test_f1': float(test_f1),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            training_history['client_results'].append(client_result)

            # 在全局测试集上评估
            self.trainers[client_id].model.eval()
            global_test_loss = 0
            global_correct = 0
            global_total = 0
            global_preds = []
            global_targets = []

            with torch.no_grad():
                for data, targets in tqdm(self.global_test_loader, desc=f"Global Test (Client {client_id+1})", leave=False):
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.trainers[client_id].model(data)
                    loss = self.trainers[client_id].criterion(outputs, targets)

                    global_test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    global_total += targets.size(0)
                    global_correct += predicted.eq(targets).sum().item()

                    global_preds.extend(predicted.cpu().numpy())
                    global_targets.extend(targets.cpu().numpy())

            global_test_loss = global_test_loss / len(self.global_test_loader)
            global_test_acc = 100. * global_correct / global_total
            global_test_f1 = f1_score(global_targets, global_preds, average='macro')

            print(f"\n边侧 {client_id+1} 在全局测试集上的结果:")
            print(f"  Global Test Loss: {global_test_loss:.4f}")
            print(f"  Global Test Accuracy: {global_test_acc:.2f}%")
            print(f"  Global Test F1 Score: {global_test_f1:.4f}")

            global_result = {
                'client_id': client_id + 1,
                'global_test_loss': float(global_test_loss),
                'global_test_accuracy': float(global_test_acc),
                'global_test_f1': float(global_test_f1)
            }
            training_history['global_test_results'].append(global_result)

        # 计算总训练时间
        total_time = time.time() - start_time
        training_history['training_time'] = total_time

        # 打印汇总统计
        print(f"\n{'='*60}")
        print("=== NoFL 训练完成 ===")
        print(f"{'='*60}")
        print(f"总训练时间: {total_time/60:.2f} 分钟")

        # 计算平均准确率
        local_accs = [r['test_accuracy'] for r in training_history['client_results']]
        global_accs = [r['global_test_accuracy'] for r in training_history['global_test_results']]

        print(f"\n本地测试集平均准确率: {np.mean(local_accs):.2f}% (±{np.std(local_accs):.2f}%)")
        print(f"全局测试集平均准确率: {np.mean(global_accs):.2f}% (±{np.std(global_accs):.2f}%)")

        print("\n各边侧本地测试结果:")
        for result in training_history['client_results']:
            print(f"  边侧 {result['client_id']}: Acc={result['test_accuracy']:.2f}%, F1={result['test_f1']:.4f}")

        print("\n各边侧全局测试结果:")
        for result in training_history['global_test_results']:
            print(f"  边侧 {result['client_id']}: Acc={result['global_test_accuracy']:.2f}%, F1={result['global_test_f1']:.4f}")

        return training_history

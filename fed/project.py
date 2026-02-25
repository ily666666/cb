import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import copy
import csv
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

# 只使用model文件夹中的模型
from helper.kd_utils import (
    calculate_adaptive_alpha,
    compute_teacher_loss_mean,
    create_kd_criterion,
    get_linear_schedule_k
)
from helper.distill_helper import DistillationHelper
from distiller_zoo import DistillKL
from .fedaware_algorithm import (
    Cloud_MomentumGradientCache,
    FeedbackSampler,
    FedAvgSerialEdgeTrainer
)


def create_model_by_type(model_name, num_classes, dataset_type='ads'):
    """
    根据数据集类型创建对应的模型
    只使用model文件夹中实际存在的模型
    

    Returns:
        对应的模型实例
    """


    if dataset_type == 'radioml':
        # RadioML数据集使用对应的复数ResNet50模型
        if model_name == 'complex_resnet50_radioml':
            from model.complex_resnet50_radioml import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else :
            from model.real_resnet20_radioml import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'reii':
        # REII数据集使用对应的复数ResNet50模型
        if model_name == 'complex_resnet50_reii':
            from model.complex_resnet50_reii import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_reii import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'radar':
        # 雷达数据集使用对应的复数ResNet50模型
        if model_name == 'complex_resnet50_radar':
            from model.complex_resnet50_radar import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_radar_with_attention':
            from model.complex_resnet50_radar_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_radar_with_attention_1000':
            from model.complex_resnet50_radar_with_attention_1000 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'real_resnet20_radar_h':
            from model.real_resnet20_radar_h import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.real_resnet20_radar import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'rml2016':
        # RML2016数据集使用对应的复数ResNet50模型
        if model_name == 'complex_resnet50_rml2016':
            from model.complex_resnet50_rml2016 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_rml2016_with_attention':
            from model.complex_resnet50_rml2016_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'real_resnet11_rml2016':
            from model.real_resnet11_rml2016 import ResNet11Real
            return ResNet11Real(num_classes=num_classes)
        elif model_name == 'real_resnet20_rml2016_h':
            from model.real_resnet20_rml2016_h import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.real_resnet20_rml2016 import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'link11':
        # Link11数据集使用对应的复数ResNet50模型
        if model_name == 'complex_resnet50_link11':
            from model.complex_resnet50_link11 import ComplexResNet50Link11
            return ComplexResNet50Link11(num_classes=num_classes)
        elif model_name == 'complex_resnet50_link11_with_attention':
            from model.complex_resnet50_link11_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_tiny_link11':
            # Tiny 复值模型（用于快速验证）
            from model.complex_tiny_link11 import ComplexTinyLink11
            return ComplexTinyLink11(num_classes=num_classes)
        elif model_name == 'real_tiny_link11':
            # Tiny 实值模型（用于快速验证）
            from model.real_tiny_link11 import RealTinyLink11
            return RealTinyLink11(num_classes=num_classes)
        elif model_name == 'real_resnet9_link11':
            from model.real_resnet9_link11 import ResNet9Real
            return ResNet9Real(num_classes=num_classes)
        elif model_name == 'real_resnet20_link11_h':
            from model.real_resnet20_link11_h import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
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



class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        计算知识蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出  
            targets: 真实标签
        """
        # 标准分类损失
        ce_loss = self.ce_loss(student_outputs, targets)
        
        # 知识蒸馏损失
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        kd_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # 总损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return total_loss, ce_loss, kd_loss


def assign_class_subsets_project(num_classes, num_edges, hetero_level):
    """
    基于异构水平的类别重叠划分（原始方法）
    """
    assert 0.0 <= hetero_level <= 1.0
    rng = np.random.RandomState(42)

    r_float = 1.0 + (num_edges - 1) * (1.0 - hetero_level)
    r_floor = int(np.floor(r_float))
    r_frac = r_float - r_floor

    edge_subsets = [set() for _ in range(num_edges)]
    edge_load = np.zeros(num_edges, dtype=int)

    classes = list(range(num_classes))
    rng.shuffle(classes)

    for c in classes:
        r_c = r_floor + (1 if rng.rand() < r_frac else 0)
        r_c = max(1, min(num_edges, r_c))

        candidates = list(range(num_edges))
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: (edge_load[i], rng.rand()))
        chosen = candidates[:r_c]
        for cid in chosen:
            edge_subsets[cid].add(c)
            edge_load[cid] += 1

    empty_edges = [i for i, s in enumerate(edge_subsets) if len(s) == 0]
    if empty_edges:
        class_freq = {c: 0 for c in range(num_classes)}
        for s in edge_subsets:
            for c in s:
                class_freq[c] += 1
        rare_classes = sorted(class_freq.keys(), key=lambda x: (class_freq[x], rng.rand()))
        rc_idx = 0
        for cid in empty_edges:
            while rc_idx < len(rare_classes):
                c = rare_classes[rc_idx]
                rc_idx += 1
                if c not in edge_subsets[cid]:
                    edge_subsets[cid].add(c)
                    edge_load[cid] += 1
                    break
            if len(edge_subsets[cid]) == 0:
                c = rng.randint(0, num_classes)
                edge_subsets[cid].add(c)
                edge_load[cid] += 1

    return edge_subsets


def dirichlet_split_indices_project(labels, num_edges, alpha, seed=42):
    """
    使用狄利克雷分布划分数据索引（Project模式）
    
    Args:
        labels: 所有样本的标签数组
        num_edges: 边侧数量
        alpha: 狄利克雷分布参数
        seed: 随机种子
        
    Returns:
        edge_indices: list of lists, 每个边侧的样本索引
    """
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


def split_data_for_project(data_path, batch_size, num_workers, num_classes, num_edges, partition_method, hetero_level, dirichlet_alpha, cloud_ratio=0.3, dataset_type='ads', snr_filter=None, seed=42, add_noise=False, noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
    """
    为project模式划分数据集 - 使用狄利克雷分布
    
    Args:
        data_path: 数据集路径
        batch_size: 批次大小
        num_workers: 工作进程数
        num_classes: 类别数量
        num_edges: 边侧数量
        dirichlet_alpha: 狄利克雷分布参数
        cloud_ratio: 云侧数据比例
        dataset_type: 数据集类型 ('ads', 'radioml', 或 'reii')
        snr_filter: SNR 过滤 (仅 radioml)
        seed: 随机种子 (默认: 42)
        
    Returns:
        cloud_train_loader: 云侧训练数据加载器（包含所有类别）
        cloud_val_loader: 云侧验证数据加载器（包含所有类别）
        cloud_test_loader: 云侧测试数据加载器（包含所有类别）
        edge_train_loaders: 边侧训练数据加载器列表（狄利克雷划分）
        edge_val_loaders: 边侧验证数据加载器列表（狄利克雷划分）
        edge_test_loaders: 边侧测试数据加载器列表（狄利克雷划分）
        global_test_loader: 全局测试数据加载器（完整100%测试集）
    """
    from utils.dataloader import get_dataloaders
    from torch.utils.data import Subset, DataLoader
    import numpy as np
    import os
    from collections import defaultdict
    from utils.readdata_25 import subDataset
    from utils.readdata_radioml import RadioMLDataset
    
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
        # 训练、验证和测试数据都添加噪声
        full_train_dataset = RML2016Dataset(pkl_path=data_path, split='train', seed=42,
                                            add_noise=add_noise, noise_type=noise_type,
                                            noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        full_val_dataset = RML2016Dataset(pkl_path=data_path, split='val', seed=42,
                                          add_noise=add_noise, noise_type=noise_type,
                                          noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        full_test_dataset = RML2016Dataset(pkl_path=data_path, split='test', seed=42,
                                           add_noise=add_noise, noise_type=noise_type,
                                           noise_snr_db=noise_snr_db, noise_factor=noise_factor)
    elif dataset_type == 'link11':
        from utils.readdata_link11 import Link11Dataset
        # 训练、验证和测试数据都添加噪声
        full_train_dataset = Link11Dataset(pkl_path=data_path, split='train', seed=42,
                                           add_noise=add_noise, noise_type=noise_type,
                                           noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        full_val_dataset = Link11Dataset(pkl_path=data_path, split='val', seed=42,
                                         add_noise=add_noise, noise_type=noise_type,
                                         noise_snr_db=noise_snr_db, noise_factor=noise_factor)
        full_test_dataset = Link11Dataset(pkl_path=data_path, split='test', seed=42,
                                          add_noise=add_noise, noise_type=noise_type,
                                          noise_snr_db=noise_snr_db, noise_factor=noise_factor)
    else:
        full_train_dataset = subDataset(datapath=data_path, split='train', transform=None, allowed_classes=None)
        full_val_dataset = subDataset(datapath=data_path, split='valid', transform=None, allowed_classes=None)
        full_test_dataset = subDataset(datapath=data_path, split='test', transform=None, allowed_classes=None)
    
    rng = np.random.RandomState(seed)
    
    # 2. 获取所有标签（用于后续分配）
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
    cloud_train_indices = []
    cloud_val_indices = []
    cloud_test_indices = []
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
        cloud_count = max(1, int(len(class_train_idx) * cloud_ratio))
        cloud_train_indices.extend(class_train_idx[:cloud_count].tolist())
        remaining_train_indices.extend(class_train_idx[cloud_count:].tolist())
        remaining_train_labels.extend([class_id] * (len(class_train_idx) - cloud_count))
        
        # 验证数据
        class_val_idx = np.where(val_labels == class_id)[0]
        rng.shuffle(class_val_idx)
        cloud_count = max(1, int(len(class_val_idx) * cloud_ratio))
        cloud_val_indices.extend(class_val_idx[:cloud_count].tolist())
        remaining_val_indices.extend(class_val_idx[cloud_count:].tolist())
        remaining_val_labels.extend([class_id] * (len(class_val_idx) - cloud_count))
        
        # 测试数据
        class_test_idx = np.where(test_labels == class_id)[0]
        rng.shuffle(class_test_idx)
        cloud_count = max(1, int(len(class_test_idx) * cloud_ratio))
        cloud_test_indices.extend(class_test_idx[:cloud_count].tolist())
        remaining_test_indices.extend(class_test_idx[cloud_count:].tolist())
        remaining_test_labels.extend([class_id] * (len(class_test_idx) - cloud_count))
        
        print(f"  类别 {class_id}: 训练 {len(class_train_idx)} (云侧 {int(len(class_train_idx)*cloud_ratio)}, 边侧 {len(class_train_idx)-int(len(class_train_idx)*cloud_ratio)})")
    
    remaining_train_labels = np.array(remaining_train_labels)
    remaining_val_labels = np.array(remaining_val_labels)
    remaining_test_labels = np.array(remaining_test_labels)
    
    # Windows系统上pickle-based数据集强制使用num_workers=0避免序列化错误
    import platform
    pickle_based_datasets = ['link11', 'rml2016', 'radar', 'radioml']
    if platform.system() == 'Windows' and dataset_type in pickle_based_datasets:
        if num_workers > 0:
            print(f"⚠️ Windows系统检测到pickle数据集({dataset_type})，强制设置 num_workers=0 (原值:{num_workers}) 避免序列化错误")
            num_workers = 0
    
    # 创建云侧数据加载器
    cloud_train_subset = Subset(full_train_dataset, cloud_train_indices)
    cloud_val_subset = Subset(full_val_dataset, cloud_val_indices)
    cloud_test_subset = Subset(full_test_dataset, cloud_test_indices)
    
    # RadioML 数据集使用 drop_last=True 避免 batch_size=1 导致 BatchNorm 报错
    use_drop_last = (dataset_type == 'radioml' or dataset_type == 'radar' or dataset_type == 'rml2016' or dataset_type == 'link11')
    
    cloud_train_loader = DataLoader(
        cloud_train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=use_drop_last
    )
    
    cloud_val_loader = DataLoader(
        cloud_val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=use_drop_last
    )
    
    cloud_test_loader = DataLoader(
        cloud_test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 测试集不drop_last
    )
    
    print(f"云侧数据 - 训练: {len(cloud_train_subset)}, 验证: {len(cloud_val_subset)}, 测试: {len(cloud_test_subset)}")
    print(f"云侧包含所有{num_classes}个类别")
    
    # 4. 边侧数据分配
    if partition_method == 'dirichlet':
        # 狄利克雷分布分配 - 改进版本
        # 步骤：用Dirichlet生成比例，然后分别应用到train/val/test
        print(f"\n=== 边侧数据分配（狄利克雷分布 alpha={dirichlet_alpha}）===")
        
        # 第一步：用Dirichlet生成分配比例（基于训练集标签）
        # 这样保证train/val/test的分配比例一致
        edge_train_local_indices = dirichlet_split_indices_project(
            remaining_train_labels, num_edges, dirichlet_alpha, seed=seed
        )
        
        # 获取每个边侧在训练集中的比例
        edge_train_proportions = [len(indices) / len(remaining_train_labels) for indices in edge_train_local_indices]
        
        # 第二步：用相同的比例分配验证集和测试集
        edge_val_local_indices = [[] for _ in range(num_edges)]
        edge_test_local_indices = [[] for _ in range(num_edges)]
        
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
                for edge_id in range(num_edges):
                    proportion = edge_train_proportions[edge_id]
                    end_idx = start_idx + int(len(class_indices) * proportion)
                    if edge_id == num_edges - 1:  # 最后一个边侧获得剩余的
                        end_idx = len(class_indices)
                    
                    edge_val_local_indices[edge_id].extend(class_indices[start_idx:end_idx].tolist())
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
                for edge_id in range(num_edges):
                    proportion = edge_train_proportions[edge_id]
                    end_idx = start_idx + int(len(class_indices) * proportion)
                    if edge_id == num_edges - 1:  # 最后一个边侧获得剩余的
                        end_idx = len(class_indices)
                    
                    edge_test_local_indices[edge_id].extend(class_indices[start_idx:end_idx].tolist())
                    start_idx = end_idx
        
        # 转换为原始数据集索引
        edge_train_indices = [[remaining_train_indices[idx] for idx in local_idx] 
                                for local_idx in edge_train_local_indices]
        # val和test的local_indices已经是原始索引，直接使用
        edge_val_indices = edge_val_local_indices
        edge_test_indices = edge_test_local_indices
        
        # 统计每个边侧的类别分布（基于train集）
        for i in range(num_edges):
            edge_train_labels = remaining_train_labels[edge_train_local_indices[i]]
            unique, counts = np.unique(edge_train_labels, return_counts=True)
            label_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"边侧 {i+1} - 训练: {len(edge_train_indices[i])} 样本, 验证: {len(edge_val_indices[i])} 样本, 测试: {len(edge_test_indices[i])} 样本, 类别分布: {label_dist}")
    
    else:  # class_overlap方法
        # 类别重叠分配（原始方法）
        print(f"\n=== 边侧数据分配（类别重叠 hetero_level={hetero_level}）===")
        
        # 分配边侧类别子集
        edge_class_subsets = assign_class_subsets_project(
            num_classes=num_classes,
            num_edges=num_edges,
            hetero_level=hetero_level
        )
        for i, subset in enumerate(edge_class_subsets):
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
        class_to_edges = defaultdict(list)
        for edge_id, subset in enumerate(edge_class_subsets):
            for class_id in subset:
                class_to_edges[class_id].append(edge_id)
        
        # 为每个边侧分配样本索引
        edge_train_indices = [[] for _ in range(num_edges)]
        edge_val_indices = [[] for _ in range(num_edges)]
        edge_test_indices = [[] for _ in range(num_edges)]
        
        for class_id in range(num_classes):
            edges_for_class = class_to_edges[class_id]
            
            if len(edges_for_class) > 0:
                # 训练数据分配
                if class_id in remaining_train_class_samples:
                    train_samples = remaining_train_class_samples[class_id]
                    if len(train_samples) > 0:
                        samples_per_edge = len(train_samples) // len(edges_for_class)
                        remainder = len(train_samples) % len(edges_for_class)
                        
                        start_idx = 0
                        for i, edge_id in enumerate(edges_for_class):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + samples_per_edge + extra
                            edge_train_indices[edge_id].extend(train_samples[start_idx:end_idx])
                            start_idx = end_idx
                
                # 验证数据分配
                if class_id in remaining_val_class_samples:
                    val_samples = remaining_val_class_samples[class_id]
                    if len(val_samples) > 0:
                        samples_per_edge = len(val_samples) // len(edges_for_class)
                        remainder = len(val_samples) % len(edges_for_class)
                        
                        start_idx = 0
                        for i, edge_id in enumerate(edges_for_class):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + samples_per_edge + extra
                            edge_val_indices[edge_id].extend(val_samples[start_idx:end_idx])
                            start_idx = end_idx
                
                # 测试数据分配
                if class_id in remaining_test_class_samples:
                    test_samples = remaining_test_class_samples[class_id]
                    if len(test_samples) > 0:
                        samples_per_edge = len(test_samples) // len(edges_for_class)
                        remainder = len(test_samples) % len(edges_for_class)
                        
                        start_idx = 0
                        for i, edge_id in enumerate(edges_for_class):
                            extra = 1 if i < remainder else 0
                            end_idx = start_idx + samples_per_edge + extra
                            edge_test_indices[edge_id].extend(test_samples[start_idx:end_idx])
                            start_idx = end_idx
        
        # 统计每个边侧的样本数
        for i in range(num_edges):
            print(f"边侧 {i+1} - 训练: {len(edge_train_indices[i])} 样本, 验证: {len(edge_val_indices[i])} 样本, 测试: {len(edge_test_indices[i])} 样本")
    
    # 创建边侧数据加载器
    edge_train_loaders = []
    edge_val_loaders = []
    edge_test_loaders = []
    
    # 创建边侧数据加载器
    for i in range(num_edges):
        if len(edge_train_indices[i]) > 0:
            edge_train_subset = Subset(full_train_dataset, edge_train_indices[i])
            edge_train_loader = DataLoader(
                edge_train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=use_drop_last
            )
        else:
            # 创建空的数据加载器
            edge_train_loader = DataLoader([], batch_size=batch_size)
        
        if len(edge_val_indices[i]) > 0:
            edge_val_subset = Subset(full_val_dataset, edge_val_indices[i])
            edge_val_loader = DataLoader(
                edge_val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=use_drop_last
            )
        else:
            edge_val_loader = DataLoader([], batch_size=batch_size)
        
        if len(edge_test_indices[i]) > 0:
            edge_test_subset = Subset(full_test_dataset, edge_test_indices[i])
            edge_test_loader = DataLoader(
                edge_test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False  # 测试集不drop_last
            )
        else:
            edge_test_loader = DataLoader([], batch_size=batch_size)
        
        edge_train_loaders.append(edge_train_loader)
        edge_val_loaders.append(edge_val_loader)
        edge_test_loaders.append(edge_test_loader)
        
        print(f"边侧 {i+1} 样本分配 - 训练: {len(edge_train_indices[i])}, 验证: {len(edge_val_indices[i])}, 测试: {len(edge_test_indices[i])}")
    
    # 6. 创建全局测试集（完整100%测试集，用于公平对比）
    global_test_loader = DataLoader(
        full_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 测试集保留所有样本
    )
    
    # 验证无重叠
    total_edge_train = sum(len(indices) for indices in edge_train_indices)
    total_edge_val = sum(len(indices) for indices in edge_val_indices)
    total_edge_test = sum(len(indices) for indices in edge_test_indices)
    
    print(f"\n=== 数据划分总结（无样本重叠）===")
    print(f"云侧: {len(cloud_train_indices)} 训练 + {len(cloud_val_indices)} 验证 + {len(cloud_test_indices)} 测试")
    print(f"所有边侧: {total_edge_train} 训练 + {total_edge_val} 验证 + {total_edge_test} 测试")
    print(f"全局测试集: {len(global_test_loader.dataset)} 样本（完整100%测试集，用于公平对比）")
    print(f"训练数据验证: 云侧({len(cloud_train_indices)}) + 边侧({total_edge_train}) = {len(cloud_train_indices) + total_edge_train} (原始: {len(full_train_dataset)})")
    print(f"验证数据验证: 云侧({len(cloud_val_indices)}) + 边侧({total_edge_val}) = {len(cloud_val_indices) + total_edge_val} (原始: {len(full_val_dataset)})")
    print(f"测试数据验证: 云侧({len(cloud_test_indices)}) + 边侧({total_edge_test}) = {len(cloud_test_indices) + total_edge_test} (原始: {len(full_test_dataset)})")
    
    return cloud_train_loader, cloud_val_loader, cloud_test_loader, edge_train_loaders, edge_val_loaders, edge_test_loaders, global_test_loader


class ProjectCloud:
    """Project模式的云侧"""
    
    def __init__(self, cloud_model, train_loader, val_loader, test_loader, device, config, save_dir, local_test_loader=None):
        """
        初始化Project云侧
        
        Args:
            cloud_model: 云侧模型（如ResNet50）
            train_loader: 云侧训练数据
            val_loader: 云侧验证数据
            test_loader: 全局测试数据
            device: 设备
            config: 配置参数
            save_dir: 保存目录
            local_test_loader: 云侧本地测试数据（可选）
        """
        self.cloud_model = cloud_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # 全局测试集（100%）
        self.local_test_loader = local_test_loader  # 云侧本地测试集（30%）
        self.device = device
        self.config = config
        self.save_dir = save_dir
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 设置优化器
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                cloud_model.parameters(), 
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                cloud_model.parameters(),
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
        
        # 设置学习率调度器
        if config['lr_scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['lr_step_size'],
                gamma=config['lr_gamma']
            )
        elif config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['cloud_epochs'],
                eta_min=config['lr_min']
            )
        else:
            self.scheduler = None
            
        # 初始化全局模型状态（用于聚合）
        self.global_model_state = None
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_f1': []
        }
        
        # 联邦学习历史
        self.federated_history = {
            'round': [],
            'global_test_loss': [],
            'global_test_acc': [],
            'global_test_f1': []
        }
        
        # 创建日志文件
        self.cloud_log_file = os.path.join(save_dir, 'cloud_pretrain_log.csv')
        self.cloud_summary_file = os.path.join(save_dir, 'cloud_pretrain_summary.txt')
        self._init_cloud_log()
    
    def _init_cloud_log(self):
        """初始化云侧训练日志文件"""
        with open(self.cloud_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 
                           'Global_Test_Loss', 'Global_Test_Acc', 'Global_Test_F1', 'Global_Eval_Time_Sec',
                           'Local_Test_Loss', 'Local_Test_Acc', 'Local_Test_F1', 'Local_Eval_Time_Sec',
                           'Learning_Rate', 'Epoch_Time_Sec', 'Is_Best'])
    
    def _log_cloud_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc,
                         global_test_loss, global_test_acc, global_test_f1, global_eval_time,
                         local_test_loss, local_test_acc, local_test_f1, local_eval_time,
                         lr, epoch_time, is_best):
        """记录云侧单个epoch的训练结果到CSV"""
        with open(self.cloud_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{train_loss:.6f}',
                f'{train_acc:.4f}',
                f'{val_loss:.6f}',
                f'{val_acc:.4f}',
                f'{global_test_loss:.6f}',
                f'{global_test_acc:.4f}',
                f'{global_test_f1:.6f}',
                f'{global_eval_time:.2f}' if global_eval_time is not None else 'N/A',
                f'{local_test_loss:.6f}' if local_test_loss is not None else 'N/A',
                f'{local_test_acc:.4f}' if local_test_acc is not None else 'N/A',
                f'{local_test_f1:.6f}' if local_test_f1 is not None else 'N/A',
                f'{local_eval_time:.2f}' if local_eval_time is not None else 'N/A',
                f'{lr:.8f}',
                f'{epoch_time:.2f}',
                'Yes' if is_best else 'No'
            ])
    
    def _save_cloud_summary(self, total_epochs, best_epoch, best_val_acc, best_test_acc, 
                           final_test_loss, final_test_acc, final_test_f1, total_time):
        """保存云侧预训练总结到TXT"""
        with open(self.cloud_summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("云侧预训练总结\n")
            f.write("="*80 + "\n\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总训练轮数: {total_epochs}\n")
            f.write(f"总耗时: {total_time/60:.2f} 分钟 ({total_time:.2f} 秒)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("最佳模型 (基于验证集)\n")
            f.write("-"*80 + "\n")
            f.write(f"最佳Epoch: {best_epoch}\n")
            f.write(f"验证准确率: {best_val_acc:.4f}%\n")
            f.write(f"对应测试准确率: {best_test_acc:.4f}%\n\n")
            
            f.write("-"*80 + "\n")
            f.write("最终测试结果\n")
            f.write("-"*80 + "\n")
            f.write(f"Test Loss: {final_test_loss:.6f}\n")
            f.write(f"Test Accuracy: {final_test_acc:.4f}%\n")
            f.write(f"Test F1 Score: {final_test_f1:.6f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("配置参数\n")
            f.write("-"*80 + "\n")
            for key, value in self.config.items():
                if key not in ['train_loader', 'val_loader', 'test_loader', 'pretrained_cloud_model']:
                    f.write(f"{key}: {value}\n")
            f.write("="*80 + "\n")
    
    def pretrain_cloud(self):
        """云侧预训练到收敛"""
        # 检查是否有预训练模型可以直接加载或继续训练
        pretrained_path = self.config.get('pretrained_cloud_model', '')
        resume_training = self.config.get('resume_cloud_training', False)
        
        start_epoch = 0
        best_val_acc = 0
        best_test_acc = 0
        best_epoch = 0
        early_stop_counter = 0
        
        if pretrained_path and os.path.exists(pretrained_path):
            if resume_training:
                # 从预训练模型继续训练
                print(f"发现预训练云侧模型: {pretrained_path}")
                print("从预训练模型继续训练...")
                
                # 加载checkpoint
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                
                # 加载模型
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    print(f"加载模型状态字典，包含 {len(model_state)} 个参数")
                else:
                    model_state = checkpoint
                    print(f"加载模型状态字典，包含 {len(model_state)} 个参数")
                
                self.cloud_model.load_state_dict(model_state)
                self.cloud_model = self.cloud_model.to(self.device)
                
                # 尝试加载训练状态（epoch, optimizer, scheduler等）
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"从epoch {start_epoch} 继续训练")
                
                if 'best_val_acc' in checkpoint:
                    best_val_acc = checkpoint['best_val_acc']
                    best_test_acc = checkpoint.get('test_acc', 0)
                    best_epoch = checkpoint.get('best_epoch', 0)
                    print(f"恢复最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch})")
                
                # 加载优化器状态（如果存在）
                if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("已恢复优化器状态")
                    except Exception as e:
                        print(f"警告: 无法加载优化器状态 ({e})，将使用新的优化器")
                
                # 加载调度器状态（如果存在）
                if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("已恢复学习率调度器状态")
                    except Exception as e:
                        print(f"警告: 无法加载调度器状态 ({e})，将使用新的调度器")
                
                # 恢复训练历史（如果存在）
                if 'train_history' in checkpoint:
                    self.train_history = checkpoint['train_history']
                    print(f"已恢复训练历史（{len(self.train_history['epoch'])} 个epoch）")
                
                # 在全局测试集上评估当前模型
                test_loss, test_acc, test_f1, _, eval_time = self.evaluate_on_test()
                print(f"当前模型性能 - Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}, 评估耗时: {eval_time:.2f}秒")
                
            else:
                # 跳过训练，直接加载模型
                print(f"发现预训练云侧模型: {pretrained_path}")
                print("跳过云侧预训练，直接加载预训练模型...")
                
                # 加载预训练模型
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    print(f"加载模型状态字典，包含 {len(model_state)} 个参数")
                else:
                    # 如果直接保存的是state_dict
                    model_state = checkpoint
                    print(f"加载模型状态字典，包含 {len(model_state)} 个参数")
                
                self.cloud_model.load_state_dict(model_state)
                self.cloud_model = self.cloud_model.to(self.device)
                
                # 在全局测试集上评估预训练模型
                test_loss, test_acc, test_f1, _, eval_time = self.evaluate_on_test()
                print(f"预训练云侧模型性能 - Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}, 评估耗时: {eval_time:.2f}秒")
                
                return self.cloud_model.state_dict()
        
        # 如果没有预训练模型或需要继续训练，进行云侧预训练
        if not (pretrained_path and os.path.exists(pretrained_path) and not resume_training):
            print("开始云侧预训练...")
        
        self.cloud_model = self.cloud_model.to(self.device)
        patience = self.config.get('patience', 10)
        
        pretrain_start_time = time.time()
        total_epochs = self.config['cloud_epochs']
        
        for epoch in range(start_epoch, total_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self._validate()
            
            # 在全局测试集上测试
            global_test_loss, global_test_acc, global_test_f1, _, global_eval_time = self.evaluate_on_test()
            
            # 在云侧本地测试集上测试（如果有）
            local_test_loss, local_test_acc, local_test_f1, local_eval_time = None, None, None, None
            if self.local_test_loader is not None:
                local_test_loss, local_test_acc, local_test_f1, _, local_eval_time = self.evaluate_on_local_test()
            
            # 更新历史记录
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['test_loss'].append(global_test_loss)
            self.train_history['test_acc'].append(global_test_acc)
            self.train_history['test_f1'].append(global_test_f1)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # 早停检查
            is_best = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = global_test_acc
                best_epoch = epoch + 1
                early_stop_counter = 0
                is_best = True
                self._save_cloud_model(epoch, is_best=True, 
                                      best_val_acc=best_val_acc, 
                                      best_test_acc=best_test_acc, 
                                      best_epoch_num=best_epoch)
            else:
                early_stop_counter += 1
                # 即使不是最佳，也保存最新checkpoint（用于恢复训练）
                self._save_cloud_model(epoch, is_best=False,
                                      best_val_acc=best_val_acc,
                                      best_test_acc=best_test_acc,
                                      best_epoch_num=best_epoch)
            
            # 记录到CSV文件
            self._log_cloud_epoch(
                epoch + 1, train_loss, train_acc, val_loss, val_acc,
                global_test_loss, global_test_acc, global_test_f1, global_eval_time,
                local_test_loss, local_test_acc, local_test_f1, local_eval_time,
                current_lr, epoch_time, is_best
            )
            
            # 打印训练信息
            print_msg = (f"Cloud Epoch {epoch+1}/{self.config['cloud_epochs']} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                        f"Global Test Loss: {global_test_loss:.4f}, Acc: {global_test_acc:.2f}%, F1: {global_test_f1:.4f}")
            if local_test_loss is not None:
                print_msg += f", Local Test Loss: {local_test_loss:.4f}, Acc: {local_test_acc:.2f}%, F1: {local_test_f1:.4f}"
            print(print_msg)
                
            if early_stop_counter >= patience:
                print(f"云侧预训练早停，在epoch {epoch + 1}")
                break
        
        pretrain_total_time = time.time() - pretrain_start_time
        
        # 加载最佳模型
        self._load_best_cloud_model()
        
        print(f"云侧预训练完成，最佳验证准确率: {best_val_acc:.2f}%")
        
        # 最终测试
        test_loss, test_acc, test_f1, conf_matrix, eval_time = self.evaluate_on_test()
        print(f"云侧预训练最终测试结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}, 评估耗时: {eval_time:.2f}秒")
        
        # 保存总结到TXT
        self._save_cloud_summary(
            epoch + 1, best_epoch, best_val_acc, best_test_acc,
            test_loss, test_acc, test_f1, pretrain_total_time
        )
        
        # 保存最终预训练模型供下次直接使用或继续训练
        # 只保存超参数配置，不保存 DataLoader 等对象
        config_to_save = {k: v for k, v in self.config.items() 
                         if k not in ['train_loader', 'val_loader', 'test_loader', 'pretrained_cloud_model', 'resume_cloud_training']}
        
        final_pretrained_path = os.path.join(self.save_dir, 'pretrained_cloud_model.pth')
        final_checkpoint = {
            'model_state_dict': self.cloud_model.state_dict(),
            'epoch': epoch + 1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch,
            'config': config_to_save,
            'train_history': self.train_history
        }
        
        # 保存优化器状态（如果存在）
        if hasattr(self, 'optimizer'):
            final_checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # 保存调度器状态（如果存在）
        if self.scheduler is not None:
            final_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(final_checkpoint, final_pretrained_path)
        print(f"云侧预训练模型已保存: {final_pretrained_path}")
        print(f"下次可使用 --pretrained_cloud_model {final_pretrained_path} 跳过预训练")
        print(f"或使用 --pretrained_cloud_model {final_pretrained_path} --resume_cloud_training 继续训练")
        
        return self.cloud_model.state_dict()
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.cloud_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Cloud Epoch {epoch+1}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.cloud_model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', False):
                nn.utils.clip_grad_norm_(self.cloud_model.parameters(), self.config['grad_clip_value'])
                
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def _validate(self):
        """验证"""
        self.cloud_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.cloud_model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def evaluate_on_test(self):
        """在全局测试集上评估"""
        eval_start_time = time.time()
        
        self.cloud_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.cloud_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time
    
    def evaluate_on_local_test(self):
        """在云侧本地测试集上评估"""
        if self.local_test_loader is None:
            return None, None, None, None, None
        
        eval_start_time = time.time()
        
        self.cloud_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.local_test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.cloud_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        test_loss = test_loss / len(self.local_test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time
    
    def aggregate_models(self, edge_models, edge_weights=None):
        """聚合边侧模型"""
        if edge_weights is None:
            edge_weights = [1.0 / len(edge_models)] * len(edge_models)
        
        # 归一化权重
        total_weight = sum(edge_weights)
        edge_weights = [w / total_weight for w in edge_weights]
        
        # 确保所有边侧模型都在 CPU 上（避免设备不匹配）
        edge_models_cpu = []
        for state in edge_models:
            state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
            edge_models_cpu.append(state_cpu)
        
        # 初始化聚合后的模型状态
        aggregated_state = {}
        param_keys = edge_models_cpu[0].keys()
        
        # 加权聚合每个参数
        for key in param_keys:
            if key.endswith('num_batches_tracked'):
                aggregated_state[key] = edge_models_cpu[0][key].clone()
                continue
                
            tensor_template = edge_models_cpu[0][key]
            if tensor_template.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
                aggregated_state[key] = tensor_template.clone()
                continue
                
            # 浮点参数聚合
            aggregated = torch.zeros_like(tensor_template)
            for i, state in enumerate(edge_models_cpu):
                aggregated += state[key] * edge_weights[i]
            aggregated_state[key] = aggregated
        
        # 更新全局模型状态
        self.global_model_state = aggregated_state
        
        return self.global_model_state
    
    def evaluate_global_model(self, global_model):
        """评估全局模型"""
        eval_start_time = time.time()
        
        global_model = global_model.to(self.device)
        global_model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = global_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        if self.device.type == 'cuda':
            global_model = global_model.cpu()
            torch.cuda.empty_cache()
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time
    
    def update_federated_history(self, round_num, test_loss, test_acc, test_f1):
        """更新联邦学习历史"""
        self.federated_history['round'].append(round_num)
        self.federated_history['global_test_loss'].append(test_loss)
        self.federated_history['global_test_acc'].append(test_acc)
        self.federated_history['global_test_f1'].append(test_f1)
    
    def _save_cloud_model(self, epoch, is_best=False, best_val_acc=0, best_test_acc=0, best_epoch_num=0):
        """保存云侧模型（包含优化器和调度器状态，用于恢复训练）"""
        # 只保存超参数配置，不保存 DataLoader 等对象
        config_to_save = {k: v for k, v in self.config.items() 
                         if k not in ['train_loader', 'val_loader', 'test_loader', 'pretrained_cloud_model', 'resume_cloud_training']}
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.cloud_model.state_dict(),
            'config': config_to_save,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'best_epoch': best_epoch_num,
            'train_history': self.train_history
        }
        
        # 保存优化器状态（如果存在）
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # 保存调度器状态（如果存在）
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_cloud_model.pth'))
        
        # 同时保存最新的checkpoint（用于恢复训练）
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_cloud_model.pth'))
    
    def _load_best_cloud_model(self):
        """加载最佳云侧模型"""
        path = os.path.join(self.save_dir, 'best_cloud_model.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False)
            self.cloud_model.load_state_dict(checkpoint['model_state_dict'])


class ProjectFedAvgEdge:
    """Project模式的FedAvg边侧（无正则化）"""
    
    def __init__(self, edge_id, edge_model, train_loader, val_loader, test_loader, device, config, local_test_loader=None):
        """
        初始化Project FedAvg边侧
        
        Args:
            edge_id: 边侧ID
            edge_model: 边侧模型（如ResNet18）
            train_loader: 边侧训练数据
            val_loader: 边侧验证数据
            test_loader: 全局测试数据
            device: 设备
            config: 配置参数
            local_test_loader: 边侧本地测试数据（可选）
        """
        self.edge_id = edge_id
        self.edge_model = edge_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # 全局测试集（100%）
        self.local_test_loader = local_test_loader  # 边侧本地测试集
        self.device = device
        self.config = config
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 教师模型（用于知识蒸馏）
        self.teacher_model = None
    
    def distill_from_teacher(self, teacher_model_state, teacher_model_architecture):
        """使用教师模型进行知识蒸馏（调用通用蒸馏辅助类）"""
        # 临时降低FedProx正则化强度以减少蒸馏训练损失
        original_prox_mu = self.prox_mu
        original_head_reg_lambda = self.head_reg_lambda
        
        # 为蒸馏训练阶段降低正则化强度
        if self.prox_mu > 0:
            self.prox_mu = min(self.prox_mu * 0.1, 0.0001)  # 降低10倍或最多到0.0001
            print(f"蒸馏训练：临时降低prox_mu从{original_prox_mu}到{self.prox_mu}")
        
        if self.head_reg_lambda > 0:
            self.head_reg_lambda = min(self.head_reg_lambda * 0.1, 0.0001)  # 降低10倍或最多到0.0001
            print(f"蒸馏训练：临时降低head_reg_lambda从{original_head_reg_lambda}到{self.head_reg_lambda}")
        
        try:
            # 调用DistillationHelper进行蒸馏
            new_state = DistillationHelper.distill_from_teacher(
                edge_model=self.edge_model,
                teacher_model_state=teacher_model_state,
                teacher_model_architecture=teacher_model_architecture,
                train_loader=self.train_loader,
                test_loader=self.test_loader,  # 全局测试集
                device=self.device,
                config=self.config,
                edge_id=self.edge_id,
                create_model_func=create_model_by_type,
                local_test_loader=self.local_test_loader  # 边侧本地测试集
            )
            
            # 更新边侧模型状态
            self.edge_model.load_state_dict(new_state)
            
            return new_state
        finally:
            # 恢复原始正则化强度
            self.prox_mu = original_prox_mu
            self.head_reg_lambda = original_head_reg_lambda
    
    def train_local(self, global_model_state):
        """标准FedAvg本地训练（无正则化）"""
        # 加载全局模型参数
        self.edge_model.load_state_dict(global_model_state)
        self.edge_model = self.edge_model.to(self.device)
        
        # 设置优化器
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.edge_model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.edge_model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay']
            )
        
        # 本地训练
        self.edge_model.train()
        
        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            pbar = tqdm(
                self.train_loader,
                desc=f"Edge {self.edge_id+1} Local Epoch {epoch+1}/{self.config['local_epochs']}",
                leave=False
            )
            
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.edge_model(data)
                loss = self.criterion(outputs, targets)  # 标准交叉熵损失，无正则化
                
                loss.backward()
                
                # 启用梯度裁剪并添加数值稳定性检查
                if self.config.get('grad_clip', False):
                    # 计算梯度范数
                    total_norm = nn.utils.clip_grad_norm_(self.edge_model.parameters(), 
                                                       self.config['grad_clip_value'])
                    # 检查梯度异常
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        print(f"警告：Edge {self.edge_id+1} 梯度范数异常，跳过该batch")
                        optimizer.zero_grad()
                        continue
                    elif total_norm > 50.0:
                        print(f"警告：Edge {self.edge_id+1} 梯度范数过大({total_norm:.2f})，进行裁剪")
                else:
                    # 默认启用梯度裁剪，阈值为1.0
                    total_norm = nn.utils.clip_grad_norm_(self.edge_model.parameters(), 1.0)
                    if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 50.0:
                        print(f"警告：Edge {self.edge_id+1} 梯度范数异常({total_norm:.2f})，跳过该batch")
                        optimizer.zero_grad()
                        continue
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                    'acc': f'{100. * epoch_correct / epoch_total:.2f}%'
                })
            
            train_loss = epoch_loss / len(self.train_loader)
            train_acc = 100. * epoch_correct / epoch_total
            
            print(f"Edge {self.edge_id+1} Local Epoch {epoch+1} - "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 在全局测试集上测试
        local_model_state = copy.deepcopy(self.edge_model.state_dict())
        global_test_loss, global_test_acc, global_test_f1, _, global_eval_time = self.test_on_global(local_model_state)
        
        # 在边侧本地测试集上测试（如果有）
        local_test_loss, local_test_acc, local_test_f1, local_eval_time = None, None, None, None
        if self.local_test_loader is not None:
            local_test_loss, local_test_acc, local_test_f1, _, local_eval_time = self.test_on_local(local_model_state)
        
        # 打印测试结果
        print_msg = (f"Edge {self.edge_id+1} 本地训练完成 - "
                    f"Global Test Loss: {global_test_loss:.4f}, Acc: {global_test_acc:.2f}%, F1: {global_test_f1:.4f}")
        if local_test_loss is not None:
            print_msg += f", Local Test Loss: {local_test_loss:.4f}, Acc: {local_test_acc:.2f}%, F1: {local_test_f1:.4f}"
        else:
            print_msg += ", Local Test: N/A"
        print(print_msg)
        
        # 直接获取模型状态字典，避免设备不匹配
        # 注意：不在这里移动模型到CPU，保持设备一致性
        
        return local_model_state, train_loss, train_acc
    
    def test_on_global(self, model_state):
        """在全局测试集上测试"""
        eval_start_time = time.time()
        
        # 创建测试模型
        test_model = copy.deepcopy(self.edge_model)
        test_model.load_state_dict(model_state)
        test_model = test_model.to(self.device)
        test_model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = test_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        
        # 清理内存
        if self.device.type == 'cuda':
            del test_model
            torch.cuda.empty_cache()
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time


class FedAWAREEdge:
    """FedAWARE算法边侧"""
    
    def __init__(self, edge_id, edge_model, train_loader, val_loader, test_loader, 
                 device, config, local_test_loader=None):
        """
        初始化FedAWARE边侧
        
        Args:
            edge_id: 边侧ID
            edge_model: 边侧模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 计算设备
            config: 边侧配置
            local_test_loader: 本地测试数据加载器
        """
        self.edge_id = edge_id
        self.edge_model = edge_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.local_test_loader = local_test_loader
        
        # 设置损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 设置优化器
        optimizer_name = config.get('optimizer', 'sgd').lower()
        lr = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 5e-4)
        
        if optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.edge_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.edge_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # FedAWARE特定参数 - FedAWARE不使用FedProx正则化
        self.mu = 0.0  # 禁用FedProx正则化
        self.local_epochs = config.get('local_epochs', 5)
        
    def local_train(self, global_model_state):
        """执行FedAWARE本地训练"""
        # 加载全局模型状态
        self.edge_model.load_state_dict(global_model_state)
        
        # 获取初始模型参数
        initial_params = torch.cat([p.data.clone().flatten() for p in self.edge_model.parameters()])
        
        self.edge_model.train()
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            batch_losses = []
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.edge_model(data)
                
                # 计算标准损失
                ce_loss = self.criterion(outputs, targets)
                
                # 添加FedProx近端项损失
                if self.mu > 0:
                    # 计算当前模型参数与初始参数的距离
                    current_params = torch.cat([p.data.flatten() for p in self.edge_model.parameters()])
                    prox_loss = self.mu * torch.norm(current_params - initial_params) ** 2
                    total_loss = ce_loss + prox_loss
                else:
                    total_loss = ce_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                batch_losses.append(total_loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
        
        # 获取训练后的模型参数
        updated_params = torch.cat([p.data.clone().flatten() for p in self.edge_model.parameters()])
        
        # 计算参数变化
        param_diff = updated_params - initial_params
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = self._evaluate_local_accuracy()
        
        return {
            'edge_id': self.edge_id,
            'param_diff': param_diff,
            'avg_loss': avg_loss,
            'train_acc': train_acc,
            'num_samples': len(self.train_loader.dataset)
        }
    
    def _evaluate_local_accuracy(self):
        """评估本地训练准确率"""
        self.edge_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.edge_model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total if total > 0 else 0.0
    
    def test_on_global(self, model_state):
        """在全局测试集上测试模型"""
        # 创建测试模型副本
        test_model = copy.deepcopy(self.edge_model)
        test_model.load_state_dict(model_state)
        test_model = test_model.to(self.device)
        test_model.eval()
        
        eval_start_time = time.time()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = test_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total if total > 0 else 0.0
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # 清理内存
        if self.device.type == 'cuda':
            del test_model
            torch.cuda.empty_cache()
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time
    
    def test_on_local(self, model_state):
        """在本地测试集上测试模型"""
        if self.local_test_loader is None:
            return None, None, None, None, None
        
        # 创建测试模型副本
        test_model = copy.deepcopy(self.edge_model)
        test_model.load_state_dict(model_state)
        test_model = test_model.to(self.device)
        test_model.eval()
        
        eval_start_time = time.time()
        test_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.local_test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = test_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集预测和真实标签用于F1计算
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(self.local_test_loader)
        test_acc = 100. * correct / total if total > 0 else 0.0
        
        # 计算F1分数
        test_f1 = 0.0
        if len(all_targets) > 0 and len(set(all_targets)) > 1:
            from sklearn.metrics import f1_score
            test_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        elif len(all_targets) > 0:
            # 如果只有一个类别，F1分数等于准确率
            test_f1 = 1.0 if correct == total else 0.0
        
        # 清理内存
        if self.device.type == 'cuda':
            del test_model
            torch.cuda.empty_cache()
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, None, eval_time
    
    def train_local(self, global_model_state):
        """执行FedAWARE本地训练（供Project模式调用）"""
        # 加载全局模型状态
        self.edge_model.load_state_dict(global_model_state)
        # 确保模型在正确的设备上
        self.edge_model = self.edge_model.to(self.device)
        
        # 获取初始模型参数
        initial_params = torch.cat([p.data.clone().flatten() for p in self.edge_model.parameters()])
        
        self.edge_model.train()
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            batch_losses = []
            self.edge_model.train()
            
            # 添加进度条
            progress_bar = tqdm(enumerate(self.train_loader), 
                              total=len(self.train_loader), 
                              desc=f"Edge {self.edge_id+1} Epoch {epoch+1}")
            
            for batch_idx, (data, targets) in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.edge_model(data)
                
                # 计算标准损失
                ce_loss = self.criterion(outputs, targets)
                
                # 添加FedProx近端项损失
                if self.mu > 0:
                    # 计算当前模型参数与初始参数的距离
                    current_params = torch.cat([p.data.flatten() for p in self.edge_model.parameters()])
                    prox_loss = self.mu * torch.norm(current_params - initial_params) ** 2
                    total_loss = ce_loss + prox_loss
                else:
                    total_loss = ce_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                batch_losses.append(total_loss.item())
                
                # 计算当前batch的准确率
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    batch_correct = predicted.eq(targets).sum().item()
                    batch_total = targets.size(0)
                    batch_acc = 100. * batch_correct / batch_total if batch_total > 0 else 0.0
                
                # 更新进度条（明确标示这是batch级别的准确率）
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Batch_Acc': f'{batch_acc:.1f}%'
                })
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
            # 计算本轮训练准确率
            self.edge_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in self.train_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.edge_model(data)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total if total > 0 else 0.0
            
            print(f"Edge {self.edge_id+1} Local Epoch {epoch+1} - "
                  f"Loss: {epoch_loss:.4f}, Acc: {train_acc:.2f}%")
            
            self.edge_model.train()
        
        # 获取训练后的模型状态
        local_model_state = copy.deepcopy(self.edge_model.state_dict())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        final_train_acc = train_acc
        
        # 在全局测试集上测试
        global_test_loss, global_test_acc, global_test_f1, _, global_eval_time = self.test_on_global(local_model_state)
        
        # 在边侧本地测试集上测试（如果有）
        local_test_loss, local_test_acc, local_test_f1, local_eval_time = None, None, None, None
        if self.local_test_loader is not None:
            local_test_loss, local_test_acc, local_test_f1, _, local_eval_time = self.test_on_local(local_model_state)
        
        # 打印测试结果
        global_test_loss_str = f"{global_test_loss:.4f}" if global_test_loss is not None else "N/A"
        global_test_acc_str = f"{global_test_acc:.2f}" if global_test_acc is not None else "N/A"
        global_test_f1_str = f"{global_test_f1:.4f}" if global_test_f1 is not None else "N/A"
        
        print_msg = (f"Edge {self.edge_id+1} 本地训练完成 - "
                    f"Global Test Loss: {global_test_loss_str}, Acc: {global_test_acc_str}%, F1: {global_test_f1_str}")
        
        if local_test_loss is not None:
            local_test_loss_str = f"{local_test_loss:.4f}"
            local_test_acc_str = f"{local_test_acc:.2f}"
            local_test_f1_str = f"{local_test_f1:.4f}"
            print_msg += f", Local Test Loss: {local_test_loss_str}, Acc: {local_test_acc_str}%, F1: {local_test_f1_str}"
        
        print(print_msg)
        
        # 清理GPU内存（可选，可以注释掉以保持设备一致性）
        # if self.device.type == 'cuda':
        #     self.edge_model = self.edge_model.cpu()
        #     torch.cuda.empty_cache()
        
        return local_model_state, avg_loss, final_train_acc

    def distill_from_teacher(self, teacher_model_state, teacher_model_architecture):
        """使用教师模型进行知识蒸馏"""
        # 调用DistillationHelper进行蒸馏
        new_state = DistillationHelper.distill_from_teacher(
            edge_model=self.edge_model,
            teacher_model_state=teacher_model_state,
            teacher_model_architecture=teacher_model_architecture,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            device=self.device,
            config=self.config,
            edge_id=self.edge_id,
            create_model_func=create_model_by_type,
            local_test_loader=self.local_test_loader
        )
        
        # 更新边侧模型状态
        self.edge_model.load_state_dict(new_state)
        
        return new_state


class ProjectFedProxEdge:
    """Project模式的FedProx边侧（带正则化）"""
    
    def __init__(self, edge_id, edge_model, train_loader, val_loader, test_loader, device, config, local_test_loader=None):
        """
        初始化Project FedProx边侧
        
        Args:
            edge_id: 边侧ID
            edge_model: 边侧模型（如ResNet18）
            train_loader: 边侧训练数据
            val_loader: 边侧验证数据
            test_loader: 全局测试数据
            device: 设备
            config: 配置参数
            local_test_loader: 边侧本地测试数据（可选）
        """
        self.edge_id = edge_id
        self.edge_model = edge_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # 全局测试集（100%）
        self.local_test_loader = local_test_loader  # 边侧本地测试集
        self.device = device
        self.config = config
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # FedProx参数
        self.prox_mu = config.get('prox_mu', 0.0) or 0.0
        self.head_reg_lambda = config.get('head_reg_lambda', 0.0) or 0.0
        
        # 教师模型（用于知识蒸馏）
        self.teacher_model = None
    
    def distill_from_teacher(self, teacher_model_state, teacher_model_architecture):
        """使用教师模型进行知识蒸馏（调用通用蒸馏辅助类）"""
        # 调用DistillationHelper进行蒸馏
        new_state = DistillationHelper.distill_from_teacher(
            edge_model=self.edge_model,
            teacher_model_state=teacher_model_state,
            teacher_model_architecture=teacher_model_architecture,
            train_loader=self.train_loader,
            test_loader=self.test_loader,  # 全局测试集
            device=self.device,
            config=self.config,
            edge_id=self.edge_id,
            create_model_func=create_model_by_type,
            local_test_loader=self.local_test_loader  # 边侧本地测试集
        )
        
        # 更新边侧模型状态
        self.edge_model.load_state_dict(new_state)
        
        return new_state
    
    def train_local(self, global_model_state):
        """FedProx本地训练（带正则化）"""
        # 加载全局模型参数
        self.edge_model.load_state_dict(global_model_state)
        self.edge_model = self.edge_model.to(self.device)
        
        # 设置优化器
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.edge_model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.edge_model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay']
            )
        
        # 记录全局参数用于FedProx
        temp_model = copy.deepcopy(self.edge_model)
        temp_model.load_state_dict(global_model_state)
        global_params = {name: p.detach().clone() for name, p in temp_model.named_parameters() if p.requires_grad}
        del temp_model
        
        # 推断边侧包含的标签集合
        edge_present_classes = None
        if self.head_reg_lambda > 0.0 and hasattr(self.edge_model, 'fc'):
            try:
                for data_tmp, targets_tmp in self.train_loader:
                    edge_present_classes = set(targets_tmp.view(-1).tolist())
                    break
            except Exception:
                edge_present_classes = None
        
        # 本地训练
        self.edge_model.train()
        
        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            epoch_prox_sum = 0.0
            epoch_head_sum = 0.0
            batch_count = 0
            
            pbar = tqdm(
                self.train_loader,
                desc=f"Edge {self.edge_id+1} Local Epoch {epoch+1}/{self.config['local_epochs']}",
                leave=False
            )
            
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.edge_model(data)
                loss = self.criterion(outputs, targets)
                
                # FedProx近端正则 - 修复设备不匹配问题
                prox_val = 0.0
                if self.prox_mu > 0.0:
                    prox_sum = None
                    total_numel = 0
                    for name, p in self.edge_model.named_parameters():
                        if not p.requires_grad or name not in global_params:
                            continue
                        # 确保global_params[name]在正确的设备上
                        global_param = global_params[name].to(p.device)
                        diff = p - global_param
                        term = (diff * diff).sum()
                        prox_sum = term if prox_sum is None else (prox_sum + term)
                        total_numel += p.numel()
                    if prox_sum is not None and total_numel > 0:
                        prox_mean = prox_sum / float(total_numel)
                        prox_val = prox_mean.detach().item()
                        loss = loss + (self.prox_mu / 2.0) * prox_mean
                
                # 分类头正则 - 修复设备不匹配问题
                head_val = 0.0
                if self.head_reg_lambda > 0.0 and edge_present_classes is not None and hasattr(self.edge_model, 'fc'):
                    fc_weight = self.edge_model.fc.weight
                    fc_bias = getattr(self.edge_model.fc, 'bias', None)
                    all_classes = set(range(fc_weight.size(0)))
                    absent = list(all_classes - set(edge_present_classes))
                    if len(absent) > 0:
                        gw = global_params.get('fc.weight')
                        gb = global_params.get('fc.bias')
                        head_terms = []
                        if gw is not None:
                            # 确保global params在正确设备上
                            global_w = gw.to(fc_weight.device)
                            term_w = (fc_weight[absent] - global_w[absent])
                            head_terms.append((term_w * term_w).mean())
                        if gb is not None and fc_bias is not None:
                            # 确保global params在正确设备上
                            global_b = gb.to(fc_bias.device)
                            term_b = (fc_bias[absent] - global_b[absent])
                            head_terms.append((term_b * term_b).mean())
                        if len(head_terms) > 0:
                            head_term = torch.stack(head_terms).mean()
                            head_val = head_term.detach().item()
                            loss = loss + self.head_reg_lambda * head_term
                
                loss.backward()
                
                # 启用梯度裁剪并添加数值稳定性检查
                if self.config.get('grad_clip', False):
                    # 计算梯度范数
                    total_norm = nn.utils.clip_grad_norm_(self.edge_model.parameters(), 
                                                       self.config['grad_clip_value'])
                    # 检查梯度异常
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        print(f"警告：Edge {self.edge_id+1} 蒸馏训练梯度范数异常，跳过该batch")
                        optimizer.zero_grad()
                        continue
                    elif total_norm > 50.0:
                        print(f"警告：Edge {self.edge_id+1} 蒸馏训练梯度范数过大({total_norm:.2f})，进行裁剪")
                else:
                    # 默认启用梯度裁剪，阈值为1.0
                    total_norm = nn.utils.clip_grad_norm_(self.edge_model.parameters(), 1.0)
                    if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 50.0:
                        print(f"警告：Edge {self.edge_id+1} 蒸馏训练梯度范数异常({total_norm:.2f})，跳过该batch")
                        optimizer.zero_grad()
                        continue
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()
                epoch_prox_sum += float(prox_val)
                epoch_head_sum += float(head_val)
                batch_count += 1
                
                pbar.set_postfix({
                    'loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                    'acc': f'{100. * epoch_correct / epoch_total:.2f}%',
                    'prox': f'{epoch_prox_sum / max(1, batch_count):.2e}',
                    'head': f'{epoch_head_sum / max(1, batch_count):.2e}'
                })
            
            train_loss = epoch_loss / len(self.train_loader)
            train_acc = 100. * epoch_correct / epoch_total
            avg_prox = epoch_prox_sum / max(1, batch_count)
            avg_head = epoch_head_sum / max(1, batch_count)
            
            print(f"Edge {self.edge_id+1} Local Epoch {epoch+1} - "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"prox: {avg_prox:.2e}, head: {avg_head:.2e}")
        
        # 在全局测试集上测试
        local_model_state = copy.deepcopy(self.edge_model.state_dict())
        global_test_loss, global_test_acc, global_test_f1, _, global_eval_time = self.test_on_global(local_model_state)
        
        # 在边侧本地测试集上测试（如果有）
        local_test_loss, local_test_acc, local_test_f1, local_eval_time = None, None, None, None
        if self.local_test_loader is not None:
            local_test_loss, local_test_acc, local_test_f1, _, local_eval_time = self.test_on_local(local_model_state)
        
        # 打印测试结果
        print_msg = (f"Edge {self.edge_id+1} 本地训练完成 - "
                    f"Global Test Loss: {global_test_loss:.4f}, Acc: {global_test_acc:.2f}%, F1: {global_test_f1:.4f}")
        if local_test_loss is not None:
            print_msg += f", Local Test Loss: {local_test_loss:.4f}, Acc: {local_test_acc:.2f}%, F1: {local_test_f1:.4f}"
        print(print_msg)
        
        # 清理GPU内存
        if self.device.type == 'cuda':
            self.edge_model = self.edge_model.cpu()
            torch.cuda.empty_cache()
        
        return local_model_state, train_loss, train_acc
    
    def test_on_global(self, model_state):
        """在全局测试集上测试"""
        eval_start_time = time.time()
        
        # 创建测试模型
        test_model = copy.deepcopy(self.edge_model)
        test_model.load_state_dict(model_state)
        test_model = test_model.to(self.device)
        test_model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = test_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # 清理内存
        if self.device.type == 'cuda':
            del test_model
            torch.cuda.empty_cache()
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time
    
    def test_on_local(self, model_state):
        """在边侧本地测试集上测试"""
        if self.local_test_loader is None:
            return None, None, None, None, None
        
        eval_start_time = time.time()
        
        # 创建测试模型
        test_model = copy.deepcopy(self.edge_model)
        test_model.load_state_dict(model_state)
        test_model = test_model.to(self.device)
        test_model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.local_test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = test_model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(self.local_test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # 清理内存
        if self.device.type == 'cuda':
            del test_model
            torch.cuda.empty_cache()
        
        eval_time = time.time() - eval_start_time
        
        return test_loss, test_acc, test_f1, conf_matrix, eval_time


class ProjectTrainer:
    """Project模式训练器"""
    
    def __init__(self, cloud_model, edge_configs, cloud_config, project_config, save_dir, fed_algorithm='fedprox', kd_models_dir='', force_retrain_kd=False, use_pretrained_kd=True, kd_save_interval=1):
        """
        初始化Project训练器
        
        Args:
            cloud_model: 云侧模型
            edge_configs: 边侧配置列表  
            cloud_config: 云侧配置
            project_config: 项目特定配置
            save_dir: 保存目录
            fed_algorithm: 联邦学习算法 ('fedavg' 或 'fedprox')
            kd_models_dir: 预蒸馏边侧模型目录路径（不指定则自动检查默认目录）
            force_retrain_kd: 强制重新进行知识蒸馏，忽略所有预蒸馏模型
            use_pretrained_kd: 是否尝试加载预蒸馏的边侧模型（默认True）
            kd_save_interval: 知识蒸馏模型保存间隔（每多少轮保存一次，默认1）
        """
        self.cloud_config = cloud_config
        self.edge_configs = edge_configs
        self.project_config = project_config
        self.save_dir = save_dir
        self.fed_algorithm = fed_algorithm
        self.kd_models_dir = kd_models_dir
        self.force_retrain_kd = force_retrain_kd
        self.use_pretrained_kd = use_pretrained_kd
        self.kd_save_interval = kd_save_interval
        
        # 设置设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 创建联邦学习日志文件
        self.federated_log_file = os.path.join(save_dir, f'federated_{fed_algorithm}_log.csv')
        self.edge_log_file = os.path.join(save_dir, f'edge_training_log.csv')
        self.kd_log_file = os.path.join(save_dir, 'knowledge_distillation_log.csv')
        self.final_summary_file = os.path.join(save_dir, f'project_{fed_algorithm}_summary.txt')
        self._init_federated_logs()
        
        # 创建云侧
        self.cloud = ProjectCloud(
            cloud_model=cloud_model,
            train_loader=cloud_config['train_loader'],
            val_loader=cloud_config['val_loader'],
            test_loader=cloud_config['test_loader'],  # 全局测试集
            device=self.device,
            config=cloud_config,
            save_dir=save_dir,
            local_test_loader=cloud_config.get('local_test_loader')  # 云侧本地测试集
        )
        
        # 创建边侧
        self.edges = []
        for i, config in enumerate(edge_configs):
            # 创建边侧模型 - 根据数据集类型选择
            edge_model = create_model_by_type(
                project_config['edge_model'],
                project_config['num_classes'],
                dataset_type=project_config.get('dataset_type', 'ads')
            )
            
            # 过滤边侧配置
            edge_config = {k: v for k, v in config.items() 
                           if k not in ['train_loader', 'val_loader', 'test_loader', 'local_test_loader']}
            
            # 根据算法选择边侧类型
            if fed_algorithm == 'fedavg':
                edge = ProjectFedAvgEdge(
                    edge_id=i,
                    edge_model=edge_model,
                    train_loader=config['train_loader'],
                    val_loader=config['val_loader'],
                    test_loader=cloud_config['test_loader'],  # 全局测试集
                    device=self.device,
                    config=edge_config,
                    local_test_loader=config.get('local_test_loader')  # 边侧本地测试集
                )
            elif fed_algorithm == 'fedprox':
                edge = ProjectFedProxEdge(
                    edge_id=i,
                    edge_model=edge_model,
                    train_loader=config['train_loader'],
                    val_loader=config['val_loader'],
                    test_loader=cloud_config['test_loader'],  # 全局测试集
                    device=self.device,
                    config=edge_config,
                    local_test_loader=config.get('local_test_loader')  # 边侧本地测试集
                )
            elif fed_algorithm == 'fedaware':
                edge = FedAWAREEdge(
                    edge_id=i,
                    edge_model=edge_model,
                    train_loader=config['train_loader'],
                    val_loader=config['val_loader'],
                    test_loader=cloud_config['test_loader'],  # 全局测试集
                    device=self.device,
                    config=edge_config,
                    local_test_loader=config.get('local_test_loader')  # 边侧本地测试集
                )
            else:
                raise ValueError(f"Unsupported federated algorithm: {fed_algorithm}")
            
            self.edges.append(edge)
    
    def _init_federated_logs(self):
        """初始化联邦学习日志文件"""
        # 联邦学习轮次日志
        with open(self.federated_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Global_Test_Loss', 'Global_Test_Acc', 'Global_Test_F1',
                           'Avg_Edge_Train_Loss', 'Avg_Edge_Train_Acc', 'Round_Time_Sec'])
        
        # 边侧训练日志
        with open(self.edge_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Edge_ID', 'Train_Loss', 'Train_Acc',
                           'Global_Test_Loss', 'Global_Test_Acc', 'Global_Test_F1', 'Global_Eval_Time_Sec',
                           'Local_Test_Loss', 'Local_Test_Acc', 'Local_Test_F1', 'Local_Eval_Time_Sec',
                           'Training_Time_Sec'])
        
        # 知识蒸馏日志
        with open(self.kd_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Edge_ID', 'KD_Method', 'Best_Epoch', 'Best_Val_Acc',
                           'Final_Test_Loss', 'Final_Test_Acc', 'Final_Test_F1',
                           'KD_Time_Sec'])
    
    def _log_federated_round(self, round_num, global_test_loss, global_test_acc, global_test_f1,
                            avg_edge_train_loss, avg_edge_train_acc, round_time):
        """记录联邦学习轮次结果"""
        with open(self.federated_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f'{global_test_loss:.6f}',
                f'{global_test_acc:.4f}',
                f'{global_test_f1:.6f}',
                f'{avg_edge_train_loss:.6f}',
                f'{avg_edge_train_acc:.4f}',
                f'{round_time:.2f}'
            ])
    
    def _log_edge_training(self, round_num, edge_id, train_loss, train_acc,
                           global_test_loss, global_test_acc, global_test_f1, global_eval_time,
                           local_test_loss, local_test_acc, local_test_f1, local_eval_time,
                           training_time):
        """记录单个边侧训练结果"""
        with open(self.edge_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                edge_id,
                f'{train_loss:.6f}',
                f'{train_acc:.4f}',
                f'{global_test_loss:.6f}',
                f'{global_test_acc:.4f}',
                f'{global_test_f1:.6f}',
                f'{global_eval_time:.2f}' if global_eval_time is not None else 'N/A',
                f'{local_test_loss:.6f}' if local_test_loss is not None else 'N/A',
                f'{local_test_acc:.4f}' if local_test_acc is not None else 'N/A',
                f'{local_test_f1:.6f}' if local_test_f1 is not None else 'N/A',
                f'{local_eval_time:.2f}' if local_eval_time is not None else 'N/A',
                f'{training_time:.2f}'
            ])
    
    def _log_kd_result(self, edge_id, kd_method, best_epoch, best_val_acc,
                      final_test_loss, final_test_acc, final_test_f1, kd_time):
        """记录知识蒸馏结果"""
        with open(self.kd_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 安全处理可能为字符串、None或数字的参数
            if isinstance(best_epoch, str) or best_epoch is None:
                best_epoch_str = best_epoch if best_epoch is not None else 'N/A'
            else:
                best_epoch_str = f'{best_epoch}'
            
            if isinstance(best_val_acc, str) or best_val_acc is None:
                best_val_acc_str = best_val_acc if best_val_acc is not None else 'N/A'
            else:
                best_val_acc_str = f'{best_val_acc:.4f}'
            
            writer.writerow([
                edge_id,
                kd_method,
                best_epoch_str,
                best_val_acc_str,
                f'{final_test_loss:.6f}',
                f'{final_test_acc:.4f}',
                f'{final_test_f1:.6f}',
                f'{kd_time:.2f}'
            ])
    
    def _save_final_summary(self, total_rounds, best_round, best_acc, best_f1,
                           final_test_loss, final_test_acc, final_test_f1, total_time):
        """保存Project模式训练总结"""
        with open(self.final_summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Project模式训练总结 - {self.fed_algorithm.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"联邦学习算法: {self.fed_algorithm.upper()}\n")
            f.write(f"总训练轮数: {total_rounds}\n")
            f.write(f"总耗时: {total_time/60:.2f} 分钟 ({total_time:.2f} 秒)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("模型配置\n")
            f.write("-"*80 + "\n")
            f.write(f"云侧模型: {self.project_config['cloud_model']}\n")
            f.write(f"边侧模型: {self.project_config['edge_model']}\n")
            f.write(f"边侧数量: {len(self.edges)}\n")
            f.write(f"本地训练轮数: {self.project_config['local_epochs']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("最佳性能 (联邦学习阶段)\n")
            f.write("-"*80 + "\n")
            f.write(f"最佳轮次: {best_round}\n")
            f.write(f"最佳准确率: {best_acc:.4f}%\n")
            f.write(f"最佳F1分数: {best_f1:.6f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("最终测试结果\n")
            f.write("-"*80 + "\n")
            f.write(f"Test Loss: {final_test_loss:.6f}\n")
            f.write(f"Test Accuracy: {final_test_acc:.4f}%\n")
            f.write(f"Test F1 Score: {final_test_f1:.6f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("配置参数\n")
            f.write("-"*80 + "\n")
            for key, value in self.project_config.items():
                if key not in ['train_loader', 'val_loader', 'test_loader']:
                    f.write(f"{key}: {value}\n")
            f.write("="*80 + "\n")
    
    def train_project(self):
        """执行Project模式训练"""
        print("=== 开始Project模式训练 ===")
        print(f"算法: {self.fed_algorithm.upper()}")
        print(f"云侧模型: {self.project_config['cloud_model']}")
        print(f"边侧模型: {self.project_config['edge_model']}")
        print(f"边侧数量: {len(self.edges)}")
        
        # 阶段1: 云侧预训练
        print("\n=== 阶段1: 云侧预训练 ===")
        teacher_model_state = self.cloud.pretrain_cloud()
        
        # 保存预训练历史
        self._save_cloud_history()
        
        # 阶段2: 边侧知识蒸馏
        print("\n=== 阶段2: 边侧知识蒸馏 ===")
        
        # 确定知识蒸馏模型目录
        if self.kd_models_dir:
            kd_models_dir = self.kd_models_dir
            print(f"使用指定的预蒸馏模型目录: {kd_models_dir}")
        else:
            kd_models_dir = os.path.join(self.save_dir, 'kd_trained_models')
            print(f"使用默认预蒸馏模型目录: {kd_models_dir}")
        
        edge_models_after_kd = []
        kd_start_time = time.time()
        
        # 检查是否有预保存的蒸馏模型
        should_check_pretrained = (not self.force_retrain_kd and 
                                 self.use_pretrained_kd and 
                                 os.path.exists(kd_models_dir) and 
                                 len(os.listdir(kd_models_dir)) == len(self.edges))
        
        if should_check_pretrained:
            print("发现预保存的知识蒸馏模型，检查兼容性...")
            compatible_models = True
            expected_pattern = f'edge_\\d{{3}}_kd_model\\.pth$'
            
            for i in range(len(self.edges)):
                expected_file = os.path.join(kd_models_dir, f'edge_{i+1:03d}_kd_model.pth')
                if not os.path.exists(expected_file):
                    compatible_models = False
                    print(f"  ✗ 缺少文件: edge_{i+1:03d}_kd_model.pth")
                    break
                    
                try:
                    # 加载模型文件检查兼容性
                    model_data = torch.load(expected_file, map_location='cpu')
                    saved_config = model_data.get('cloud_model_architecture', '')
                    saved_edge_arch = model_data.get('edge_model_architecture', '')
                    
                    if (saved_config != self.project_config['cloud_model'] or 
                        saved_edge_arch != self.project_config['edge_model']):
                        print(f"  ✗ 边侧 {i+1} 模型架构不兼容")
                        print(f"    预期: 云侧={self.project_config['cloud_model']}, 边侧={self.project_config['edge_model']}")
                        print(f"    实际: 云侧={saved_config}, 边侧={saved_edge_arch}")
                        compatible_models = False
                        break
                        
                    # 加载模型状态
                    edge_models_after_kd.append(model_data['model_state_dict'])
                    print(f"  ✓ 边侧 {i+1} 兼容的蒸馏模型已加载")
                    
                except Exception as e:
                    print(f"  ✗ 边侧 {i+1} 模型文件损坏: {str(e)}")
                    compatible_models = False
                    break
            
            if compatible_models:
                print("✓ 所有边侧蒸馏模型加载成功，跳过知识蒸馏阶段")
                kd_total_time = time.time() - kd_start_time
                print(f"  节省时间: {kd_total_time/60:.2f}分钟")
                
                # 直接进行初始聚合
                edge_sample_counts = [len(c.train_loader.dataset) for c in self.edges]
                total_samples = float(sum(edge_sample_counts))
                edge_weights = [n / total_samples for n in edge_sample_counts]
                
                global_model_state = self.cloud.aggregate_models(edge_models_after_kd, edge_weights)
                
                # 创建全局模型用于评估
                global_model = copy.deepcopy(self.edges[0].edge_model)
                global_model.load_state_dict(global_model_state)
                
                print("直接进入联邦学习训练阶段...")
                
            else:
                print("预保存模型不兼容，重新进行知识蒸馏")
                edge_models_after_kd = []
                kd_start_time = time.time()
        else:
            if self.force_retrain_kd:
                print("强制重新进行知识蒸馏...")
            elif not self.use_pretrained_kd:
                print("禁用预蒸馏模型加载，进行新的一轮知识蒸馏")
            else:
                print("未发现预保存的知识蒸馏模型，开始新的一轮知识蒸馏")
        
        # 如果没有加载预保存的模型，进行知识蒸馏
        if not edge_models_after_kd:
            print("\n开始边侧知识蒸馏...")
            for i, edge in enumerate(self.edges):
                edge_kd_start = time.time()
                
                kd_model_state = edge.distill_from_teacher(
                    teacher_model_state, 
                    self.project_config['cloud_model']  # 传递云侧模型架构
                )
                edge_models_after_kd.append(kd_model_state)
                
                edge_kd_time = time.time() - edge_kd_start
                
                # 测试蒸馏后的模型性能
                test_loss, test_acc, test_f1, _, kd_eval_time = edge.test_on_global(kd_model_state)
                
                # 记录KD结果 (注意：这里简化处理，实际的best_epoch等信息需要从DistillationHelper返回)
                self._log_kd_result(
                    edge_id=i+1,
                    kd_method=self.project_config.get('kd_method', 'KD'),
                    best_epoch='N/A',  # DistillationHelper未返回此信息
                    best_val_acc='N/A',
                    final_test_loss=test_loss,
                    final_test_acc=test_acc,
                    final_test_f1=test_f1,
                    kd_time=edge_kd_time
                )
                
                print(f"边侧 {i+1} KD完成 - Test Acc: {test_acc:.2f}%, 耗时: {edge_kd_time:.2f}秒")
            
            kd_total_time = time.time() - kd_start_time
            print(f"所有边侧知识蒸馏完成，总耗时: {kd_total_time/60:.2f}分钟")
            
            # 保存知识蒸馏后的边侧模型
            os.makedirs(kd_models_dir, exist_ok=True)
            print(f"保存知识蒸馏后的边侧模型到: {kd_models_dir}")
            
            for i, kd_model_state in enumerate(edge_models_after_kd):
                kd_model_path = os.path.join(kd_models_dir, f'edge_{i+1:03d}_kd_model.pth')
                torch.save({
                    'edge_id': i + 1,
                    'model_state_dict': kd_model_state,
                    'kd_method': self.project_config.get('kd_method', 'KD'),
                    'kd_temperature': self.project_config.get('kd_temperature', 4.0),
                    'kd_alpha': self.project_config.get('kd_alpha', 0.5),
                    'cloud_model_architecture': self.project_config['cloud_model'],
                    'edge_model_architecture': self.project_config['edge_model'],
                    'timestamp': time.strftime('%Y%m%d_%H%M%S')
                }, kd_model_path)
                print(f"  ✓ 边侧 {i+1} 知识蒸馏模型已保存")
        
        # 使用蒸馈后的边侧模型进行初始聚合
        edge_sample_counts = [len(c.train_loader.dataset) for c in self.edges]
        total_samples = float(sum(edge_sample_counts))
        edge_weights = [n / total_samples for n in edge_sample_counts]
        
        if edge_models_after_kd:
            global_model_state = self.cloud.aggregate_models(edge_models_after_kd, edge_weights)
        else:
            global_model_state = self.cloud.aggregate_models([self.edges[0].edge_model.state_dict() for _ in self.edges], edge_weights)
        
        # 创建全局模型用于评估
        global_model = copy.deepcopy(self.edges[0].edge_model)
        global_model.load_state_dict(global_model_state)
        
        # 阶段3: 联邦学习训练
        print("\n=== 阶段3: 联邦学习训练 ===")
        print(f"联邦学习轮数: {self.project_config['num_rounds']}")
        print(f"本地训练轮数: {self.project_config['local_epochs']}")
        
        # 创建边侧模型保存目录
        edge_save_dir = os.path.join(self.save_dir, 'edge_models')
        os.makedirs(edge_save_dir, exist_ok=True)
        print(f"边侧模型保存目录: {edge_save_dir}")
        
        federated_start_time = time.time()
        best_round = 0
        best_acc = 0
        best_f1 = 0
        
        for round_num in range(self.project_config['num_rounds']):
            round_start_time = time.time()
            print(f"\n=== 联邦学习轮次 {round_num + 1}/{self.project_config['num_rounds']} ===")
            
            # 边侧本地训练
            edge_models = []
            edge_train_losses = []
            edge_train_accs = []
            
            for i, edge in enumerate(self.edges):
                edge_train_start = time.time()
                print(f"训练边侧 {i + 1}/{len(self.edges)}...")
                
                local_model_state, train_loss, train_acc = edge.train_local(global_model_state)
                
                edge_models.append(local_model_state)
                edge_train_losses.append(train_loss)
                edge_train_accs.append(train_acc)
                
                edge_train_time = time.time() - edge_train_start
                
                # 保存边侧模型（根据配置的间隔）
                if (round_num + 1) % self.project_config.get('edge_save_interval', 5) == 0:
                    # 先获取测试准确率，避免出现未定义变量错误
                    global_test_loss, global_test_acc, global_test_f1, _, edge_global_eval_time = edge.test_on_global(local_model_state)
                    
                    # 在本地测试集上测试（如果有）
                    local_test_loss, local_test_acc, local_test_f1, edge_local_eval_time = None, None, None, None
                    if edge.local_test_loader is not None:
                        local_test_loss, local_test_acc, local_test_f1, _, edge_local_eval_time = edge.test_on_local(local_model_state)
                    
                    edge_save_path = os.path.join(edge_save_dir, f'edge_{i+1:03d}_round_{round_num+1:03d}.pth')
                    torch.save({
                        'round': round_num + 1,
                        'edge_id': i + 1,
                        'model_state_dict': local_model_state,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'global_test_acc': global_test_acc,
                        'local_test_acc': local_test_acc
                    }, edge_save_path)
                    print(f"  → 边侧 {i + 1} 模型已保存: edge_{i+1:03d}_round_{round_num+1:03d}.pth")
                
                # 在全局测试集上测试边侧模型
                global_test_loss, global_test_acc, global_test_f1, _, edge_global_eval_time = edge.test_on_global(local_model_state)
                
                # 在本地测试集上测试（如果有）
                local_test_loss, local_test_acc, local_test_f1, edge_local_eval_time = None, None, None, None
                if edge.local_test_loader is not None:
                    local_test_loss, local_test_acc, local_test_f1, _, edge_local_eval_time = edge.test_on_local(local_model_state)
                
                # 记录边侧训练结果
                self._log_edge_training(
                    round_num + 1, i + 1,
                    train_loss, train_acc,
                    global_test_loss, global_test_acc, global_test_f1, edge_global_eval_time,
                    local_test_loss, local_test_acc, local_test_f1, edge_local_eval_time,
                    edge_train_time
                )
                
                print(f"边侧 {i + 1} 本地训练结果 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                     f"Test Acc: {global_test_acc:.2f}%, 耗时: {edge_train_time:.2f}秒")
            
            # 云侧聚合
            print("聚合边侧模型...")
            global_model_state = self.cloud.aggregate_models(edge_models, edge_weights)
            
            # 更新全局模型
            global_model.load_state_dict(global_model_state)
            
            # 全局评估
            print("评估全局模型...")
            test_loss, test_acc, test_f1, conf_matrix, global_model_eval_time = self.cloud.evaluate_global_model(global_model)
            
            # 计算平均边侧性能
            avg_train_loss = np.mean(edge_train_losses)
            avg_train_acc = np.mean(edge_train_accs)
            
            round_time = time.time() - round_start_time
            
            # 记录联邦学习轮次结果
            self._log_federated_round(
                round_num + 1,
                test_loss, test_acc, test_f1,
                avg_train_loss, avg_train_acc,
                round_time
            )
            
            # 更新云侧历史记录
            self.cloud.update_federated_history(round_num + 1, test_loss, test_acc, test_f1)
            
            # 追踪最佳性能
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
                best_round = round_num + 1
            
            # 打印轮次总结
            print(f"\n{'='*80}")
            print(f"轮次 {round_num + 1} 总结 - 耗时: {round_time:.2f}秒")
            print(f"  平均边侧训练 - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%")
            print(f"  全局模型测试   - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}")
            print(f"  当前最佳       - 轮次: {best_round}, Acc: {best_acc:.2f}%, F1: {best_f1:.4f}")
            print(f"{'='*80}\n")
        
        federated_total_time = time.time() - federated_start_time
        
        # 获取最终测试结果
        final_test_loss = self.cloud.federated_history['global_test_loss'][-1]
        final_test_acc = self.cloud.federated_history['global_test_acc'][-1]
        final_test_f1 = self.cloud.federated_history['global_test_f1'][-1]
        
        # 保存最终总结
        self._save_final_summary(
            self.project_config['num_rounds'],
            best_round, best_acc, best_f1,
            final_test_loss, final_test_acc, final_test_f1,
            federated_total_time
        )
        
        # 保存最终结果
        self._save_final_results(global_model_state)
        
        print(f"\n{'='*80}")
        print(f"Project模式训练完成!")
        print(f"  联邦学习耗时: {federated_total_time/60:.2f} 分钟")
        print(f"  最佳轮次: {best_round}")
        print(f"  最佳准确率: {best_acc:.2f}%")
        print(f"  最终准确率: {final_test_acc:.2f}%")
        print(f"  日志已保存至: {self.save_dir}")
        print(f"{'='*80}\n")
        
        return self.cloud.federated_history
    
    def _save_cloud_history(self):
        """保存云侧预训练历史"""
        import json
        history_path = os.path.join(self.save_dir, 'cloud_pretrain_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.cloud.train_history, f, indent=4)
    
    def _save_final_results(self, global_model_state):
        """保存最终模型和JSON历史"""
        import json
        
        # 只保存超参数配置，不保存 DataLoader 等对象
        config_to_save = {k: v for k, v in self.project_config.items() 
                         if k not in ['train_loader', 'val_loader', 'test_loader']}
        
        # 保存全局模型
        final_save_path = os.path.join(self.save_dir, 'final_global_model.pth')
        torch.save({
            'model_state_dict': global_model_state,
            'config': config_to_save,
            'final_round': self.project_config['num_rounds']
        }, final_save_path)
        print(f"✓ 全局模型已保存: {final_save_path}")
        
        # 保存联邦学习历史
        federated_history_path = os.path.join(self.save_dir, f'project_{self.fed_algorithm}_federated_history.json')
        with open(federated_history_path, 'w') as f:
            json.dump(self.cloud.federated_history, f, indent=4)
        print(f"✓ 联邦学习历史已保存: {federated_history_path}")

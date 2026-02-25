"""
REII 数据集加载模块
用于加载 REII LFM 数据集（MATLAB v7.3 格式）
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import os
from glob import glob


class REIIDataset(Dataset):
    """
    REII LFM 数据集类
    
    数据格式:
    - X: (2, 2000, 33000) - IQ信号数据
    - Y: (1, 33000) - 标签 (1.0, 2.0, 3.0)
    - snr_levels: (11, 1) - SNR级别
    """
    
    def __init__(self, datapath, transform=None, split='train', snr_filter=None, signal_length=2000):
        """
        初始化 REII 数据集
        
        Args:
            datapath: 数据集目录路径 (包含多个.mat文件)
            transform: 数据变换（保持接口一致）
            split: 'train', 'valid', 或 'test'
            snr_filter: SNR 过滤，tuple (min_snr, max_snr) 或 None 表示使用所有
            signal_length: 信号长度，默认2000（如果需要截断或填充）
        """
        self.datapath = datapath
        self.split = split
        self.transform = transform
        self.snr_filter = snr_filter
        self.signal_length = signal_length
        
        # 加载和处理数据
        self._load_and_split_data()
        
    def _load_and_split_data(self):
        """加载所有.mat文件并划分数据集（内存优化版本）"""
        
        print(f"正在加载 REII 数据集: {self.datapath}")
        print(f"   目标 split: {self.split}")
        
        # 获取所有.mat文件
        mat_files = sorted(glob(os.path.join(self.datapath, "*.mat")))
        
        if len(mat_files) == 0:
            raise ValueError(f"在 {self.datapath} 中没有找到.mat文件")
        
        print(f"   发现 {len(mat_files)} 个.mat文件")
        
        # 第一步：收集所有样本的元数据（不加载实际数据）
        all_sample_meta = []
        
        for file_idx, mat_file in enumerate(mat_files):
            with h5py.File(mat_file, 'r') as f:
                # 只读取标签，不读取数据
                Y = np.array(f['Y']).flatten()  # (33000,)
                
                # 转换标签为0-based索引
                Y = (Y - 1.0).astype(np.int64)  # 1.0, 2.0, 3.0 -> 0, 1, 2
                
                # 为每个样本创建元数据
                for i in range(len(Y)):
                    all_sample_meta.append({
                        'file_path': mat_file,
                        'file_idx': file_idx,
                        'sample_idx': i,
                        'label': Y[i]
                    })
        
        print(f"   总样本数: {len(all_sample_meta):,}")
        
        # 第二步：划分数据集索引 (70% train, 15% valid, 15% test)
        random.seed(42)
        random.shuffle(all_sample_meta)
        
        n_total = len(all_sample_meta)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        if self.split == 'train':
            self.sample_meta = all_sample_meta[:n_train]
        elif self.split == 'valid':
            self.sample_meta = all_sample_meta[n_train:n_train + n_val]
        elif self.split == 'test':
            self.sample_meta = all_sample_meta[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # 统计类别分布
        label_counts = {}
        for meta in self.sample_meta:
            label = int(meta['label'])
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"   {self.split} 数据集: {len(self.sample_meta):,} 样本")
        print(f"   类别分布: {label_counts}")
        
        # 设置类别数
        self.num_classes = 3
        
        # 缓存打开的文件句柄（减少重复打开）
        self._file_cache = {}
        
    def __len__(self):
        return len(self.sample_meta)
    
    def __getitem__(self, idx):
        """
        获取单个样本（Lazy Loading）
        返回格式: (复数张量, 标签)
        
        数据格式与项目其他数据集保持一致
        """
        meta = self.sample_meta[idx]
        file_path = meta['file_path']
        sample_idx = meta['sample_idx']
        label = meta['label']
        
        # 从文件中读取单个样本（lazy loading）
        with h5py.File(file_path, 'r') as f:
            # X shape: (2, 2000, 33000)
            # 只读取第sample_idx个样本: X[:, :, sample_idx]
            data = f['X'][:, :, sample_idx]  # (2, 2000)
        
        # 转换为 float32
        data = data.astype(np.float32)
        
        # 转换为 PyTorch 张量
        data_real = torch.from_numpy(data[0])  # (2000,)
        data_imag = torch.from_numpy(data[1])  # (2000,)
        
        # 转换为复数格式
        out = torch.view_as_complex(torch.stack([data_real, data_imag], dim=-1))
        
        return out, label


def get_reii_dataloaders(data_path, batch_size=32, num_workers=4, signal_length=2000):
    """
    创建 REII 数据集的 DataLoaders
    
    Args:
        data_path: 数据集路径
        batch_size: 批次大小
        num_workers: 工作进程数
        signal_length: 信号长度
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    from torch.utils.data import DataLoader
    
    # 创建数据集
    train_dataset = REIIDataset(datapath=data_path, split='train', signal_length=signal_length)
    val_dataset = REIIDataset(datapath=data_path, split='valid', signal_length=signal_length)
    test_dataset = REIIDataset(datapath=data_path, split='test', signal_length=signal_length)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # 避免最后一个batch只有1个样本
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False  # 测试集保留所有样本
    )
    
    num_classes = train_dataset.num_classes
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == '__main__':
    # 测试代码
    data_path = r'E:\BaiduNet_Download\REII'
    
    print("测试 REII 数据集加载...")
    print("="*70)
    
    # 创建数据集
    train_dataset = REIIDataset(datapath=data_path, split='train')
    val_dataset = REIIDataset(datapath=data_path, split='valid')
    test_dataset = REIIDataset(datapath=data_path, split='test')
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset):,}")
    print(f"  验证集: {len(val_dataset):,}")
    print(f"  测试集: {len(test_dataset):,}")
    print(f"  类别数: {train_dataset.num_classes}")
    
    # 测试读取
    print(f"\n测试读取第一个样本:")
    data, label = train_dataset[0]
    print(f"  数据形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")
    print(f"  标签: {label}")
    
    # 测试DataLoader
    print(f"\n测试DataLoader:")
    train_loader, val_loader, test_loader, num_classes = get_reii_dataloaders(
        data_path, batch_size=32, num_workers=0
    )
    
    batch_data, batch_labels = next(iter(train_loader))
    print(f"  Batch数据形状: {batch_data.shape}")
    print(f"  Batch标签形状: {batch_labels.shape}")
    print(f"  Batch标签: {batch_labels[:10]}")
    
    print("\n[SUCCESS] 测试通过！")


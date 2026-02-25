"""
RadioML 数据集加载模块
模仿 readdata_25.py 的结构，用于加载 pkl 格式的 RadioML 数据集
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import random
import time
import os


# Global cache for loaded data to avoid reloading
_RADIOML_DATA_CACHE = {}


class RadioMLDataset(Dataset):
    """
    RadioML 数据集类 (Optimized with caching)
    模仿 subDataset 的结构
    
    数据格式: {(调制类型, SNR): numpy数组(N, 2, 128)}
    """
    
    def __init__(self, datapath, transform, split, snr_filter=None):
        """
        初始化 RadioML 数据集
        
        Args:
            datapath: pkl 文件路径
            transform: 数据变换（保持接口一致，实际可能不用）
            split: 'train', 'valid', 或 'test'
            snr_filter: SNR 过滤，tuple (min_snr, max_snr) 或 None 表示使用所有
        """
        self.datapath = datapath
        self.split = split
        self.transform = transform
        self.snr_filter = snr_filter
        
        # 加载和处理数据 (with caching)
        self._load_and_split_data()
        
    def _load_and_split_data(self):
        """加载 pkl 文件并划分数据集 - 内存优化版本 with caching"""
        
        cache_key = f"{self.datapath}_42"  # seed=42
        
        # Check if data is already in cache
        if cache_key not in _RADIOML_DATA_CACHE:
            load_start_time = time.time()
            print(f"正在加载 RadioML 数据集: {self.datapath}")
            
            pkl_load_start = time.time()
            with open(self.datapath, 'rb') as f:
                raw_data = pickle.load(f)
            pkl_load_time = time.time() - pkl_load_start
            
            # 提取所有调制类型并创建标签映射
            extract_start = time.time()
            modulations = sorted(list(set([key[0] for key in raw_data.keys()])))
            modulation_to_label = {mod: idx for idx, mod in enumerate(modulations)}
            num_classes = len(modulations)
            
            print(f"✅ 数据加载完成 (耗时: {pkl_load_time:.2f}秒)")
            print(f"   调制类型: {modulations}")
            print(f"   类别数: {num_classes}")
            print(f"   正在划分数据集...")
            
            # 收集样本索引（不复制数据）
            sample_indices = []  # 存储 (key, sample_idx) 的列表
            
            for key in raw_data.keys():
                modulation, snr = key
                
                # SNR 过滤
                if self.snr_filter is not None:
                    min_snr, max_snr = self.snr_filter
                    if snr < min_snr or snr > max_snr:
                        continue
                
                num_samples = len(raw_data[key])
                for i in range(num_samples):
                    sample_indices.append((key, i, modulation_to_label[modulation]))
            
            extract_time = time.time() - extract_start
            
            # 划分索引
            split_start = time.time()
            random.seed(42)
            random.shuffle(sample_indices)
            
            n_total = len(sample_indices)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            
            train_indices = sample_indices[:n_train]
            val_indices = sample_indices[n_train:n_train + n_val]
            test_indices = sample_indices[n_train + n_val:]
            split_time = time.time() - split_start
            
            # Cache the data and indices
            _RADIOML_DATA_CACHE[cache_key] = {
                'raw_data': raw_data,
                'modulation_to_label': modulation_to_label,
                'num_classes': num_classes,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            }
            
            total_load_time = time.time() - load_start_time
            print(f"✅ 数据集划分完成并缓存 (耗时: {split_time:.2f}秒)")
            print(f"   训练集: {len(train_indices)} 样本")
            print(f"   验证集: {len(val_indices)} 样本")
            print(f"   测试集: {len(test_indices)} 样本")
            print(f"📊 总加载时间: {total_load_time:.2f}秒 (pkl: {pkl_load_time:.2f}s, 提取: {extract_time:.2f}s, 划分: {split_time:.2f}s)")
        else:
            print(f"✅ 从缓存加载 RadioML 数据集 ({self.split})")
        
        # Get data from cache
        cached_data = _RADIOML_DATA_CACHE[cache_key]
        raw_data = cached_data['raw_data']
        modulation_to_label = cached_data['modulation_to_label']
        self.num_classes = cached_data['num_classes']
        
        # Select split
        if self.split == 'train':
            selected_indices = cached_data['train_indices']
        elif self.split == 'valid':
            selected_indices = cached_data['val_indices']
        elif self.split == 'test':
            selected_indices = cached_data['test_indices']
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # 只提取当前 split 需要的数据
        print(f"   正在提取 {self.split} 数据...")
        self.samples = []
        
        for key, idx, label in selected_indices:
            modulation, snr = key
            data = raw_data[key][idx].copy()  # 复制单个样本
            
            self.samples.append({
                'data': data,
                'label': label,
                'modulation': modulation,
                'snr': snr
            })
        
        # 打印信息
        print(f"   {self.split} 数据集: {len(self.samples):,} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        返回格式与 subDataset 保持一致：(复数张量, 标签)
        """
        sample = self.samples[idx]
        data = sample['data']  # (2, 128)
        label = sample['label']
        
        # 转换为 float32
        data = data.astype(np.float32)
        
        # 转换为 PyTorch 张量
        data_real = torch.from_numpy(data[0])  # (128,)
        data_imag = torch.from_numpy(data[1])  # (128,)
        
        # 转换为复数格式，与 subDataset 的输出格式一致
        out = torch.view_as_complex(torch.stack([data_real, data_imag], dim=-1))
        
        return out, label


if __name__ == '__main__':
    # 测试代码
    pkl_path = r'E:\BaiduNet_Download\augmented_data.pkl'
    
    print("测试 RadioML 数据集加载...")
    print("="*70)
    
    # 创建数据集
    train_dataset = RadioMLDataset(datapath=pkl_path, split='train', transform=None, snr_filter=None)
    val_dataset = RadioMLDataset(datapath=pkl_path, split='valid', transform=None, snr_filter=None)
    test_dataset = RadioMLDataset(datapath=pkl_path, split='test', transform=None, snr_filter=None)
    
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
    
    print("\n✅ 测试通过！")


import h5py
import numpy as np
import torch
import time
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Global cache for loaded data to avoid reloading
_RADAR_DATA_CACHE = {}


class RadarDataset(Dataset):
    """
    Radar Emitter Individual Identification Dataset (Optimized with caching)
    
    Data format:
    - X: (2, 500, num_samples) - IQ signals with 500 time points
    - Y: (1, num_samples) - Class labels (1-7)
    """
    
    def __init__(self, mat_path, split='train', test_size=0.2, val_size=0.1, seed=42, transform=None):
        """
        Args:
            mat_path: Path to the .mat file
            split: 'train', 'val', or 'test'
            test_size: Proportion of test set (default: 0.2)
            val_size: Proportion of validation set from remaining data (default: 0.1)
            seed: Random seed for reproducibility
            transform: Optional transforms to be applied on samples
        """
        self.transform = transform
        self.split = split
        self.mat_path = mat_path
        self.seed = seed
        
        # Load and split data (with caching)
        self._load_data(test_size, val_size)
    
    def _load_data(self, test_size, val_size):
        """Load and split dataset (with global caching to avoid reloading)"""
        cache_key = f"{self.mat_path}_{self.seed}_{test_size}_{val_size}"
        
        # Check if data is already in cache
        if cache_key not in _RADAR_DATA_CACHE:
            load_start_time = time.time()
            print(f"正在加载 Radar 数据集: {self.mat_path}")
            
            # 检查是否有对应的pkl文件（更快）
            pkl_path = self.mat_path.replace('.mat', '.pkl')
            
            if os.path.exists(pkl_path):
                # 使用pickle加载（快速）
                print(f"   发现 .pkl 文件，使用快速加载...")
                pkl_load_start = time.time()
                import pickle
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                X = data['X']
                Y = data['Y']
                mat_load_time = time.time() - pkl_load_start
                print(f"✅ 数据加载完成 (耗时: {mat_load_time:.2f}秒, 使用pkl)")
            else:
                # 使用h5py加载（慢速）
                print(f"   未找到 .pkl 文件，使用 h5py 加载 .mat 文件...")
                print(f"   提示: 运行 convert_radar_to_pkl.py 可以加速后续加载")
                mat_load_start = time.time()
                with h5py.File(self.mat_path, 'r') as mat:
                    X = np.array(mat['X'])  # Shape: (2, 500, num_samples)
                    Y = np.array(mat['Y']).flatten()  # Shape: (num_samples,)
                mat_load_time = time.time() - mat_load_start
                
                # Convert to (num_samples, 2, 500) format
                convert_start = time.time()
                num_samples = X.shape[2]
                X = np.transpose(X, (2, 0, 1))  # (num_samples, 2, 500)
                convert_time = time.time() - convert_start
                print(f"✅ 数据加载完成 (耗时: {mat_load_time:.2f}秒)")
                print(f"   数据转换完成 (耗时: {convert_time:.2f}秒)")
            
            num_samples = X.shape[0]
            print(f"   样本数: {num_samples}")
            print(f"   类别数: {len(np.unique(Y))}")
            print(f"   正在划分数据集...")
            
            # Split data: train/val/test
            split_start = time.time()
            # First split: separate test set
            train_val_indices, test_indices = train_test_split(
                np.arange(num_samples),
                test_size=test_size,
                random_state=self.seed,
                stratify=Y
            )
            
            # Second split: separate val from train
            train_val_Y = Y[train_val_indices]
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size / (1 - test_size),
                random_state=self.seed,
                stratify=train_val_Y
            )
            split_time = time.time() - split_start
            
            # Cache the split data
            _RADAR_DATA_CACHE[cache_key] = {
                'X': X,
                'Y': Y,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            }
            
            total_load_time = time.time() - load_start_time
            print(f"✅ 数据集划分完成并缓存 (耗时: {split_time:.2f}秒)")
            print(f"   训练集: {len(train_indices)} 样本")
            print(f"   验证集: {len(val_indices)} 样本")
            print(f"   测试集: {len(test_indices)} 样本")
            print(f"📊 总加载时间: {total_load_time:.2f}秒")
        else:
            print(f"✅ 从缓存加载 Radar 数据集 ({self.split})")
        
        # Get data from cache
        cached_data = _RADAR_DATA_CACHE[cache_key]
        
        # Select split
        if self.split == 'train':
            self.indices = cached_data['train_indices']
        elif self.split == 'val':
            self.indices = cached_data['val_indices']
        elif self.split == 'test':
            self.indices = cached_data['test_indices']
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Store references to shared data (NOT copies!)
        # 只存储引用，不复制数据，大幅提升速度
        self.X = cached_data['X']  # 共享完整数据
        self.Y = cached_data['Y']  # 共享完整数据
        
        # Convert labels to 0-indexed (from 1-7 to 0-6)
        self.Y_adjusted = self.Y - 1
        
        self.num_classes = len(np.unique(self.Y_adjusted[self.indices]))
        
        print(f"   {self.split} 数据集: {len(self.indices)} 样本, {self.num_classes} 类别")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get actual index from split indices
        actual_idx = self.indices[idx]
        
        # Get IQ data: (2, 500)
        iq_data = self.X[actual_idx]  # (2, 500)
        
        # Convert to complex tensor: (500,)
        I = torch.from_numpy(iq_data[0]).float()
        Q = torch.from_numpy(iq_data[1]).float()
        signal = torch.complex(I, Q)
        
        # Get label (0-indexed)
        label = torch.tensor(self.Y_adjusted[actual_idx], dtype=torch.long)
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label


def get_radar_dataloaders(mat_path, batch_size=64, num_workers=4, seed=42):
    """
    Create train, validation, and test dataloaders for Radar dataset
    
    Args:
        mat_path: Path to the .mat file
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        seed: Random seed
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    train_dataset = RadarDataset(mat_path, split='train', seed=seed)
    val_dataset = RadarDataset(mat_path, split='val', seed=seed)
    test_dataset = RadarDataset(mat_path, split='test', seed=seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    num_classes = train_dataset.num_classes
    
    return train_loader, val_loader, test_loader, num_classes

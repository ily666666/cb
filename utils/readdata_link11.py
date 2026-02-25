"""
Link11 Dataset Reader
Dataset: Link11 - 7 radar emitter types with SNR variation
Signal format: IQ (In-phase and Quadrature) with 1024 time samples
"""
import pickle
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# Global cache for loaded data to avoid reloading
_LINK11_DATA_CACHE = {}


class Link11Dataset(Dataset):
    """
    Link11 Dataset Loader (Optimized with caching)
    
    Structure: {(emitter_type, SNR): signal_array}
    Signal shape: (num_samples, 2, 1024) where 2 is I/Q channels
    """
    
    def __init__(self, pkl_path, split='train', snr_range=None, seed=42,
                 add_noise=False, noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
        """
        Args:
            pkl_path: Path to link11.pkl file
            split: 'train', 'val', or 'test'
            snr_range: Tuple (min_snr, max_snr) to filter by SNR, None for all
            seed: Random seed for reproducibility
            add_noise: Whether to add noise to signals
            noise_type: Type of noise ('awgn' or 'factor')
            noise_snr_db: SNR in dB for AWGN noise
            noise_factor: Noise factor (0.0-1.0) for factor noise
        """
        self.pkl_path = pkl_path
        self.split = split
        self.snr_range = snr_range
        self.seed = seed
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_snr_db = noise_snr_db
        self.noise_factor = noise_factor
        
        # Emitter types mapping
        self.emitter_types = ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']
        self.emitter_to_label = {emitter: idx for idx, emitter in enumerate(self.emitter_types)}
        self.num_classes = len(self.emitter_types)
        
        # Load and prepare data (with caching)
        self._load_data()
    
    def _load_data(self):
        """Load and split dataset (with global caching to avoid reloading)"""
        cache_key = f"{self.pkl_path}_{self.seed}"
        
        # Check if data is already in cache
        if cache_key not in _LINK11_DATA_CACHE:
            load_start_time = time.time()
            print(f"正在加载 Link11 数据集: {self.pkl_path}")
            
            # Load pickle file
            pkl_load_start = time.time()
            with open(self.pkl_path, 'rb') as f:
                raw_data = pickle.load(f)
            pkl_load_time = time.time() - pkl_load_start
            
            print(f"✅ 数据加载完成 (耗时: {pkl_load_time:.2f}秒)")
            print(f"   发射机类型: {self.emitter_types}")
            print(f"   类别数: {self.num_classes}")
            
            # Extract signals and labels (优化版本 - 预分配数组)
            extract_start = time.time()
            
            # 先计算总样本数和过滤后的数据
            filtered_data = []
            total_samples = 0
            for (emitter_type, snr), signal_array in raw_data.items():
                # Filter by SNR range if specified
                if self.snr_range is not None:
                    if snr < self.snr_range[0] or snr > self.snr_range[1]:
                        continue
                label = self.emitter_to_label[emitter_type]
                filtered_data.append((signal_array, label))
                total_samples += len(signal_array)
            
            # 预分配数组（避免多次内存分配）
            first_shape = filtered_data[0][0].shape[1:]  # (2, 1024)
            signals = np.empty((total_samples, *first_shape), dtype=np.float32)
            labels = np.empty(total_samples, dtype=np.int64)
            
            # 填充数据
            idx = 0
            for signal_array, label in filtered_data:
                n = len(signal_array)
                signals[idx:idx+n] = signal_array
                labels[idx:idx+n] = label
                idx += n
            extract_time = time.time() - extract_start
            
            print(f"   数据提取完成 (耗时: {extract_time:.2f}秒)")
            print(f"   正在划分数据集...")
            
            # Stratified split: 80% train, 10% val, 10% test
            split_start = time.time()
            np.random.seed(self.seed)
            
            # Get indices for each class
            class_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                class_indices[label].append(idx)
            
            # Split each class
            train_indices = []
            val_indices = []
            test_indices = []
            
            for label in range(self.num_classes):
                indices = np.array(class_indices[label])
                np.random.shuffle(indices)
                
                n_samples = len(indices)
                n_train = int(0.8 * n_samples)
                n_val = int(0.1 * n_samples)
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
            
            split_time = time.time() - split_start
            
            # Cache the split data
            _LINK11_DATA_CACHE[cache_key] = {
                'signals': signals,
                'labels': labels,
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
            print(f"✅ 从缓存加载 Link11 数据集 ({self.split})")
        
        # Get data from cache
        cached_data = _LINK11_DATA_CACHE[cache_key]
        
        # Select split
        if self.split == 'train':
            indices = cached_data['train_indices']
        elif self.split == 'val':
            indices = cached_data['val_indices']
        else:  # test
            indices = cached_data['test_indices']
        
        # Store references to shared data (not copies!)
        self.signals = cached_data['signals'][indices]
        self.labels = cached_data['labels'][indices]
        
        print(f"   {self.split} 数据集: {len(self.signals)} 样本")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        """
        Returns:
            signal: Complex tensor of shape (1024,)
            label: Integer class label
        """
        signal = self.signals[idx]  # shape: (2, 1024)
        label = self.labels[idx]
        
        # Convert to complex tensor: (2, 1024) -> (1024,) complex
        signal_complex = signal[0] + 1j * signal[1]
        
        # Add noise if enabled
        if self.add_noise:
            signal_complex = self._add_noise(signal_complex)
        
        signal_tensor = torch.from_numpy(signal_complex).to(torch.complex64)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor
    
    def _add_noise(self, signal):
        """
        Add noise to the signal
        
        Args:
            signal: Complex signal array
            
        Returns:
            Noisy signal
        """
        if self.noise_type == 'awgn':
            return self._add_awgn_noise(signal)
        elif self.noise_type == 'factor':
            return self._add_factor_noise(signal)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _add_awgn_noise(self, signal):
        """
        Add Additive White Gaussian Noise (AWGN)
        
        Args:
            signal: Complex signal
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy signal
        """
        # Calculate signal power
        P_signal = np.mean(np.abs(signal) ** 2)
        
        # Calculate noise power from SNR
        P_noise = P_signal / (10 ** (self.noise_snr_db / 10))
        
        # Generate Gaussian noise (real and imaginary parts independently)
        noise_real = np.random.normal(0, np.sqrt(P_noise / 2), signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(P_noise / 2), signal.shape)
        noise = noise_real + 1j * noise_imag
        
        # Add noise to signal
        noisy_signal = signal + noise
        
        # 调试：打印噪声强度
        if not hasattr(self, '_noise_debug_awgn_printed'):
            print(f"[DEBUG AWGN] Signal power: {P_signal:.6f}, Noise power: {P_noise:.6f}, Ratio: {P_signal/P_noise:.2f}")
            self._noise_debug_awgn_printed = True
        
        return noisy_signal
    
    def _add_factor_noise(self, signal):
        """
        Add noise based on noise factor (可以任意值，包括>1.0)
        
        Args:
            signal: Complex signal
            noise_factor: Noise factor, scales the noise amplitude
                         noise_factor=0.1 means noise power = 0.1 * signal power
                         noise_factor=1.0 means noise power = signal power
                         noise_factor=5.0 means noise power = 5 * signal power (非常强的噪声)
            
        Returns:
            Noisy signal
        """
        # Calculate signal power
        P_signal = np.mean(np.abs(signal) ** 2)
        
        # Generate Gaussian noise with standard deviation 1
        noise_real = np.random.normal(0, 1, signal.shape)
        noise_imag = np.random.normal(0, 1, signal.shape)
        noise = noise_real + 1j * noise_imag
        
        # Scale noise by factor relative to signal power
        # noise_factor可以任意值，包括>1.0
        noise_power = P_signal * self.noise_factor
        noise = noise * np.sqrt(noise_power)  # Scale to desired noise power
        
        # Add noise to signal
        noisy_signal = signal + noise
        
        # 计算实际噪声功率（验证）
        actual_noise = noisy_signal - signal
        actual_noise_power = np.mean(np.abs(actual_noise) ** 2)
        noisy_signal_power = np.mean(np.abs(noisy_signal) ** 2)
        
        # 调试：打印噪声强度
        if not hasattr(self, '_noise_debug_factor_printed'):
            print(f"[DEBUG FACTOR] Signal power: {P_signal:.6f}, Expected noise power: {noise_power:.6f}, Actual noise power: {actual_noise_power:.6f}, Noisy signal power: {noisy_signal_power:.6f}, Ratio: {P_signal/noise_power:.2f}")
            self._noise_debug_factor_printed = True
        
        return noisy_signal


def get_link11_dataloaders(pkl_path, batch_size=64, num_workers=0, snr_range=None, seed=42,
                           add_noise=False, noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
    """
    Create train/val/test dataloaders for Link11 dataset
    
    Args:
        pkl_path: Path to link11.pkl file
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        snr_range: Tuple (min_snr, max_snr) to filter by SNR, None for all
        seed: Random seed
        add_noise: Whether to add noise to signals
        noise_type: Type of noise ('awgn' or 'factor')
        noise_snr_db: SNR in dB for AWGN noise
        noise_factor: Noise factor (0.0-1.0) for factor noise
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    train_dataset = Link11Dataset(pkl_path, split='train', snr_range=snr_range, seed=seed,
                                  add_noise=add_noise, noise_type=noise_type, 
                                  noise_snr_db=noise_snr_db, noise_factor=noise_factor)
    val_dataset = Link11Dataset(pkl_path, split='val', snr_range=snr_range, seed=seed,
                                add_noise=add_noise, noise_type=noise_type, 
                                noise_snr_db=noise_snr_db, noise_factor=noise_factor)
    test_dataset = Link11Dataset(pkl_path, split='test', snr_range=snr_range, seed=seed,
                                 add_noise=add_noise, noise_type=noise_type, 
                                 noise_snr_db=noise_snr_db, noise_factor=noise_factor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes


if __name__ == '__main__':
    # Test the data loader
    pkl_path = r"E:\BaiduNet_Download\link11.pkl"
    train_loader, val_loader, test_loader, num_classes = get_link11_dataloaders(
        pkl_path, batch_size=64, num_workers=0
    )
    
    print(f"\nDataLoader Test:")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Num classes: {num_classes}")
    
    # Get a batch
    for signals, labels in train_loader:
        print(f"\nBatch shape:")
        print(f"  Signals: {signals.shape}, dtype: {signals.dtype}")
        print(f"  Labels: {labels.shape}")
        break

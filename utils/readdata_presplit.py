"""
预划分数据加载器 - 直接加载预划分的pkl文件

用于加载由 prepare_data_splits.py 生成的预划分数据
无需索引，直接加载完整的客户端/服务器数据
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class PresplitDataset(Dataset):
    """
    预划分数据集加载器
    
    直接加载预划分的pkl文件，无需索引
    """
    
    def __init__(self, pkl_path, split='train', add_noise=False, 
                 noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
        """
        Args:
            pkl_path: 预划分数据文件路径 (server_data.pkl 或 client_X_data.pkl)
            split: 'train', 'val', or 'test'
            add_noise: 是否添加噪声
            noise_type: 噪声类型
            noise_snr_db: AWGN噪声SNR
            noise_factor: 因子噪声强度
        """
        self.split = split
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_snr_db = noise_snr_db
        self.noise_factor = noise_factor
        
        # 加载预划分数据
        print(f"[Presplit] 加载预划分数据: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取对应split的数据
        self.signals, self.labels = data[split]
        
        print(f"[Presplit] ✅ 加载完成: {len(self.signals)} 样本 ({split})")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        """
        Returns:
            signal: Complex tensor
            label: Integer class label
        """
        signal = self.signals[idx]  # shape: (2, N)
        label = self.labels[idx]
        
        # Convert to complex tensor: (2, N) -> (N,) complex
        signal_complex = signal[0] + 1j * signal[1]
        
        # Add noise if enabled
        if self.add_noise:
            signal_complex = self._add_noise(signal_complex)
        
        signal_tensor = torch.from_numpy(signal_complex).to(torch.complex64)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor
    
    def _add_noise(self, signal):
        """添加噪声"""
        if self.noise_type == 'awgn':
            return self._add_awgn_noise(signal)
        elif self.noise_type == 'factor':
            return self._add_factor_noise(signal)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _add_awgn_noise(self, signal):
        """添加AWGN噪声"""
        P_signal = np.mean(np.abs(signal) ** 2)
        P_noise = P_signal / (10 ** (self.noise_snr_db / 10))
        
        noise_real = np.random.normal(0, np.sqrt(P_noise / 2), signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(P_noise / 2), signal.shape)
        noise = noise_real + 1j * noise_imag
        
        return signal + noise
    
    def _add_factor_noise(self, signal):
        """添加因子噪声"""
        P_signal = np.mean(np.abs(signal) ** 2)
        
        noise_real = np.random.normal(0, 1, signal.shape)
        noise_imag = np.random.normal(0, 1, signal.shape)
        noise = noise_real + 1j * noise_imag
        
        noise_power = P_signal * self.noise_factor
        noise = noise * np.sqrt(noise_power)
        
        return signal + noise

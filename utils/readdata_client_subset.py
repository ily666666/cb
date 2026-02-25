"""
客户端子集数据加载器 - 只加载客户端分配的数据，避免加载完整数据集
用于分布式训练场景，大幅降低客户端内存占用
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class ClientSubsetDataset(Dataset):
    """
    客户端子集数据集 - 只加载指定索引的数据
    
    适用于Link11、RML2016等大型数据集的分布式训练
    避免客户端加载完整数据集，只加载自己需要的部分
    """
    
    def __init__(self, pkl_path, indices, dataset_type='link11', add_noise=False, 
                 noise_type='awgn', noise_snr_db=15, noise_factor=0.1):
        """
        Args:
            pkl_path: 数据集pkl文件路径
            indices: 客户端需要的样本索引列表
            dataset_type: 数据集类型 ('link11', 'rml2016')
            add_noise: 是否添加噪声
            noise_type: 噪声类型 ('awgn' or 'factor')
            noise_snr_db: AWGN噪声的SNR (dB)
            noise_factor: 因子噪声的强度
        """
        self.pkl_path = pkl_path
        self.indices = indices
        self.dataset_type = dataset_type
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_snr_db = noise_snr_db
        self.noise_factor = noise_factor
        
        # 加载原始数据
        print(f"[ClientSubset] 正在加载数据集: {pkl_path}")
        print(f"[ClientSubset] 只加载 {len(indices)} 个样本（而非完整数据集）")
        
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 根据数据集类型提取数据
        if dataset_type == 'link11':
            self._load_link11_subset(raw_data)
        elif dataset_type == 'rml2016':
            self._load_rml2016_subset(raw_data)
        elif dataset_type == 'radar':
            self._load_radar_subset(raw_data)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        print(f"[ClientSubset] ✅ 数据加载完成，共 {len(self.signals)} 个样本")
    
    def _load_link11_subset(self, raw_data):
        """加载Link11数据集的子集"""
        # Link11数据格式: {(emitter_type, SNR): signal_array}
        emitter_types = ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']
        emitter_to_label = {emitter: idx for idx, emitter in enumerate(emitter_types)}
        
        # 先展平所有数据以获取全局索引映射
        all_signals = []
        all_labels = []
        
        for (emitter_type, snr), signal_array in raw_data.items():
            label = emitter_to_label[emitter_type]
            all_signals.append(signal_array)
            all_labels.extend([label] * len(signal_array))
        
        all_signals = np.concatenate(all_signals, axis=0)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        # 只提取客户端需要的索引
        self.signals = all_signals[self.indices].astype(np.float32)
        self.labels = all_labels[self.indices]
        
        self.num_classes = len(emitter_types)
    
    def _load_rml2016_subset(self, raw_data):
        """加载RML2016数据集的子集"""
        # RML2016数据格式: {(modulation_type, SNR): signal_array}
        modulation_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
        mod_to_label = {mod: idx for idx, mod in enumerate(modulation_types)}
        
        # 先展平所有数据以获取全局索引映射
        all_signals = []
        all_labels = []
        
        for (mod_type, snr), signal_array in raw_data.items():
            label = mod_to_label[mod_type]
            all_signals.append(signal_array)
            all_labels.extend([label] * len(signal_array))
        
        all_signals = np.concatenate(all_signals, axis=0)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        # 只提取客户端需要的索引
        self.signals = all_signals[self.indices].astype(np.float32)
        self.labels = all_labels[self.indices]
        
        self.num_classes = len(modulation_types)
    
    def _load_radar_subset(self, raw_data):
        """加载Radar数据集的子集（从pkl文件）"""
        # Radar pkl格式: {'X': array, 'Y': array}
        # X shape: (num_samples, 2, 500)
        # Y shape: (num_samples,) with labels 1-7
        
        X = raw_data['X']  # (num_samples, 2, 500)
        Y = raw_data['Y']  # (num_samples,) labels 1-7
        
        # 只提取客户端需要的索引
        self.signals = X[self.indices].astype(np.float32)  # (client_samples, 2, 500)
        self.labels = (Y[self.indices] - 1).astype(np.int64)  # 转换为0-indexed (0-6)
        
        self.num_classes = 7
    
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
        """添加噪声到信号"""
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

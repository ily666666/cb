#!/usr/bin/env python3
"""
端侧设备 - 实时生成信号并发送给边侧
支持三种数据集：Link11, RML2016, Radar
"""
import sys
import os
import argparse
import time
import numpy as np
import zmq
import pickle
from scipy.signal import lfilter

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============== 物理常数 ==============
c = 3e8  # 光速 (m/s)
k = 1.38e-23  # 玻尔兹曼常数
T = 290  # 绝对温度 (K)
R_earth = 6371e3  # 地球半径 (m)

# 个体相位偏移（确保唯一）
individual_phase_offset = {
    'E-2D_1': 0.2 * np.pi,
    'E-2D_2': 0.5 * np.pi,
    'P-3C_1': 0.8 * np.pi,
    'P-3C_2': 1.1 * np.pi,
    'P-8A_1': 1.4 * np.pi,
    'P-8A_2': 1.7 * np.pi,
    'P-8A_3': 2.3 * np.pi
}

# 飞机参数
aircraft_parameters = {
    'E-2D_1': {
        'type': 'E-2D', 'frequency': 225e6, 'transmit_power': 100,
        'tx_gain': 12, 'rx_gain': 2, 'height': 8000,
        'start_lat': 26.16, 'start_lon': 127.66,
        'end_lat': 24.09, 'end_lon': 122.41,
        'noise_figure': 4, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    },
    'E-2D_2': {
        'type': 'E-2D', 'frequency': 225e6, 'transmit_power': 100,
        'tx_gain': 12, 'rx_gain': 1, 'height': 9000,
        'start_lat': 26.09, 'start_lon': 127.67,
        'end_lat': 25.01, 'end_lon': 122.43,
        'noise_figure': 4, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    },
    'P-3C_1': {
        'type': 'P-3C', 'frequency': 270e6, 'transmit_power': 30,
        'tx_gain': 8, 'rx_gain': 2, 'height': 3000,
        'start_lat': 26.48, 'start_lon': 127.63,
        'end_lat': 25.14, 'end_lon': 122.44,
        'noise_figure': 4, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    },
    'P-3C_2': {
        'type': 'P-3C', 'frequency': 270e6, 'transmit_power': 30,
        'tx_gain': 8, 'rx_gain': 1, 'height': 2500,
        'start_lat': 26.27, 'start_lon': 127.75,
        'end_lat': 25.02, 'end_lon': 122.42,
        'noise_figure': 4, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    },
    'P-8A_1': {
        'type': 'P-8A', 'frequency': 300e6, 'transmit_power': 50,
        'tx_gain': 10, 'rx_gain': 1, 'height': 6000,
        'start_lat': 26.38, 'start_lon': 127.68,
        'end_lat': 25.11, 'end_lon': 122.44,
        'noise_figure': 5, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    },
    'P-8A_2': {
        'type': 'P-8A', 'frequency': 300e6, 'transmit_power': 50,
        'tx_gain': 10, 'rx_gain': 1, 'height': 6200,
        'start_lat': 26.57, 'start_lon': 127.67,
        'end_lat': 25.09, 'end_lon': 122.42,
        'noise_figure': 5, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    },
    'P-8A_3': {
        'type': 'P-8A', 'frequency': 300e6, 'transmit_power': 50,
        'tx_gain': 10, 'rx_gain': 2, 'height': 5800,
        'start_lat': 26.32, 'start_lon': 127.71,
        'end_lat': 25.08, 'end_lon': 122.43,
        'noise_figure': 5, 'bandwidth': 16e3, 'symbol_rate': 1200, 'modulation': '2FSK'
    }
}

# 类别映射
AIRCRAFT_ID_TO_LABEL = {
    'E-2D_1': 0, 'E-2D_2': 1, 'P-3C_1': 2, 'P-3C_2': 3,
    'P-8A_1': 4, 'P-8A_2': 5, 'P-8A_3': 6
}


# ============== 流量控制工具 ==============

class RateLimiter:
    """流量限速器 - 基于滑动窗口的速率控制"""
    
    def __init__(self, rate_limit_mbps=0.0):
        """
        初始化限速器
        
        Args:
            rate_limit_mbps: 速率限制（MB/s），0表示不限速
        """
        self.rate_limit_mbps = rate_limit_mbps
        self.rate_limit_bytes_per_sec = rate_limit_mbps * 1024 * 1024
        self.bytes_sent_in_window = 0
        self.window_start_time = time.time()
        self.enabled = rate_limit_mbps > 0
        
        # 统计信息
        self.total_bytes_sent = 0
        self.start_time = time.time()
    
    def wait_if_needed(self, data_size_bytes):
        """
        根据发送的数据大小，决定是否需要等待
        
        Args:
            data_size_bytes: 本次发送的字节数
        """
        # 更新总统计
        self.total_bytes_sent += data_size_bytes
        
        if not self.enabled:
            return
        
        self.bytes_sent_in_window += data_size_bytes
        
        # 计算当前窗口的实际速率
        elapsed_time = time.time() - self.window_start_time
        
        if elapsed_time > 0:
            current_rate = self.bytes_sent_in_window / elapsed_time  # bytes/s
            
            # 如果超过限速，计算需要等待的时间
            if current_rate > self.rate_limit_bytes_per_sec:
                # 需要等待的时间 = (已发送字节数 / 限速) - 已经过的时间
                required_time = self.bytes_sent_in_window / self.rate_limit_bytes_per_sec
                wait_time = required_time - elapsed_time
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    # 更新 elapsed_time（因为我们等待了）
                    elapsed_time = time.time() - self.window_start_time
        
        # 每秒重置窗口
        if elapsed_time >= 1.0:
            self.bytes_sent_in_window = 0
            self.window_start_time = time.time()
    
    def get_current_rate_mbps(self):
        """获取当前速率（MB/s）"""
        if not self.enabled:
            return 0.0
        
        elapsed_time = time.time() - self.window_start_time
        if elapsed_time > 0.01:  # 至少10ms
            return (self.bytes_sent_in_window / (1024 * 1024)) / elapsed_time
        return 0.0
    
    def get_average_rate_mbps(self):
        """获取平均速率（MB/s）"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return (self.total_bytes_sent / (1024 * 1024)) / elapsed_time
        return 0.0
    
    def get_total_mb_sent(self):
        """获取总发送量（MB）"""
        return self.total_bytes_sent / (1024 * 1024)


# ============== 本地数据加载器 ==============

class LocalDataLoader:
    """
    从本地文件加载数据并通过 ZeroMQ 发送
    
    支持两种模式：
    1. 单文件模式：一次性加载完整数据集（.pkl 或 .mat 文件）
    2. 文件夹模式：逐批次加载文件（文件夹包含多个 .pkl 或 .mat 文件）
    """
    
    def __init__(self, data_path, dataset_type, edge_host='localhost', edge_port=7777, rate_limit_mbps=0.0):
        """
        初始化本地数据加载器
        
        Args:
            data_path: 数据文件或文件夹路径
            dataset_type: 数据集类型 ('link11', 'rml2016', 'radar')
            edge_host: 边侧主机地址
            edge_port: 边侧数据端口
            rate_limit_mbps: 流量限制（MB/s），0表示不限速
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.edge_response_port = 7778  # 默认响应端口，可以后续设置
        
        # 创建限速器
        self.rate_limiter = RateLimiter(rate_limit_mbps)
        
        # ZeroMQ 设置 - 双向连接
        self.context = zmq.Context()
        
        # PUSH socket：发送数据
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.SNDHWM, 10000)
        self.push_socket.setsockopt(zmq.SNDBUF, 64 * 1024 * 1024)
        self.push_socket.setsockopt(zmq.LINGER, 0)
        
        # PULL socket：接收完成标志
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 1000)
        self.pull_socket.setsockopt(zmq.RCVBUF, 16 * 1024 * 1024)
        self.pull_socket.setsockopt(zmq.RCVTIMEO, 600000)  # 60秒超时
        
        # 优化：MessagePack 支持
        try:
            import msgpack
            import msgpack_numpy as m
            m.patch()
            self.use_msgpack = True
            print(f"[端侧-本地数据] 使用 MessagePack 序列化（优化模式）")
        except ImportError:
            self.use_msgpack = False
            print(f"[端侧-本地数据] MessagePack 未安装，使用 Pickle 序列化")
        
        # 检测路径类型
        self.is_folder = os.path.isdir(data_path)
        self.file_format = None
        self.batch_files = None
        self.raw_data = None
        
        if self.is_folder:
            print(f"[端侧-本地数据] 检测到文件夹: {self.data_path}")
            self._setup_batch_files()
        else:
            print(f"[端侧-本地数据] 检测到文件: {self.data_path}")
            # 加载单个文件
            self.raw_data = self._load_data()
    
    def _setup_batch_files(self):
        """设置文件夹批次加载"""
        import glob
        
        # 查找 .pkl 和 .mat 文件
        batch_files_pkl = sorted(glob.glob(os.path.join(self.data_path, '*.pkl')))
        batch_files_mat = sorted(glob.glob(os.path.join(self.data_path, '*.mat')))
        
        if batch_files_pkl:
            self.batch_files = batch_files_pkl
            self.file_format = 'pkl'
        elif batch_files_mat:
            self.batch_files = batch_files_mat
            self.file_format = 'mat'
        else:
            raise ValueError(f"文件夹中没有找到 .pkl 或 .mat 文件: {self.data_path}")
        
        print(f"[端侧-本地数据] 找到 {len(self.batch_files)} 个批次文件 (.{self.file_format})")
    
    def _load_data(self, file_path=None):
        """加载本地数据文件
        
        Args:
            file_path: 文件路径，如果为 None 则使用 self.data_path
        """
        if file_path is None:
            file_path = self.data_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pkl':
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)
        
        elif ext == '.mat':
            # 加载 MATLAB 文件
            try:
                import h5py
                with h5py.File(file_path, 'r') as mat:
                    mat_data = {key: np.array(mat[key]) for key in mat.keys() if not key.startswith('__')}
            except (OSError, ImportError):
                import scipy.io as scio
                mat_data = scio.loadmat(file_path)
            
            # 转换格式
            raw_data = self._convert_mat_to_dict(mat_data)
        
        else:
            raise ValueError(f"不支持的文件格式: {ext}. 支持的格式: .pkl, .mat")
        
        return raw_data
    
    def _convert_mat_to_dict(self, mat_data):
        """将 MATLAB 数据转换为字典格式"""
        if self.dataset_type == 'radar':
            # Radar 数据集格式: 'X_batch' 和 'Y_batch' (批次文件)
            # 或 'X' 和 'Y' (单个文件)
            
            # 尝试批次格式
            if 'X_batch' in mat_data and 'Y_batch' in mat_data:
                X = np.array(mat_data['X_batch'])  # (2, 500, num_samples)
                Y = np.array(mat_data['Y_batch']).flatten()  # (num_samples,)
            elif 'X' in mat_data and 'Y' in mat_data:
                X = np.array(mat_data['X'])  # (2, 500, num_samples)
                Y = np.array(mat_data['Y']).flatten()  # (num_samples,)
            else:
                raise ValueError(f"Radar 数据集需要 'X' 和 'Y' 或 'X_batch' 和 'Y_batch' 字段，但只找到: {list(mat_data.keys())}")
            
            # Convert to (num_samples, 2, 500) format
            num_samples = X.shape[2]
            X = np.transpose(X, (2, 0, 1))  # (num_samples, 2, 500)
            
            # Convert labels to 0-indexed (from 1-7 to 0-6)
            Y_adjusted = Y - 1
            
            # 按类别组织数据
            raw_data = {}
            unique_labels = np.unique(Y_adjusted)
            for label in unique_labels:
                label_idx = np.where(Y_adjusted == label)[0]
                raw_data[int(label)] = X[label_idx]
            
            return raw_data
        
        # 通用转换：假设数据已按类别组织
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raw_data = {}
        for key in data_keys:
            try:
                label = int(key)
                raw_data[label] = mat_data[key]
            except ValueError:
                continue
        
        if not raw_data:
            raise ValueError(f"无法从 MATLAB 文件中提取数据")
        
        return raw_data
    
    def connect(self):
        """连接到边侧（双向连接）"""
        # 连接PUSH socket（发送数据）
        data_addr = f"tcp://{self.edge_host}:{self.edge_port}"
        self.push_socket.connect(data_addr)
        
        # 连接PULL socket（接收完成标志）
        response_addr = f"tcp://{self.edge_host}:{self.edge_response_port}"
        self.pull_socket.connect(response_addr)
        
        print(f"[端侧-本地数据] 已连接到边侧:")
        print(f"  数据端口: {data_addr}")
        print(f"  响应端口: {response_addr}")
    
    def load_and_send(self, batch_size=100, interval=0.1, total_samples=None):
        """
        加载数据并发送给边侧
        
        Args:
            batch_size: 每批次样本数
            interval: 发送间隔（秒）
            total_samples: 总样本数限制（None表示发送所有数据）
        """
        if self.is_folder:
            # 文件夹模式：逐批次加载文件
            self._load_and_send_from_folder(batch_size, interval, total_samples)
        else:
            # 单文件模式：一次性加载所有数据
            self._load_and_send_from_file(batch_size, interval, total_samples)
    
    def _load_and_send_from_file(self, batch_size, interval, total_samples):
        """从单个文件加载并发送数据"""
        # 准备所有数据
        all_signals = []
        all_labels = []
        
        for label, signal_array in self.raw_data.items():
            # 转换标签：如果是字符串，映射成数字
            numeric_label = self._convert_label_to_int(label)
            
            for signal in signal_array:
                all_signals.append(signal)
                all_labels.append(numeric_label)
        
        total_available = len(all_signals)
        
        # 确定要发送的样本数
        if total_samples is None or total_samples > total_available:
            samples_to_send = total_available
        else:
            samples_to_send = total_samples
        
        print(f"[端侧-本地数据] 开始发送数据（单文件模式），批次大小: {batch_size}，间隔: {interval}s")
        print(f"[端侧-本地数据] 总样本数: {samples_to_send}/{total_available}")
        
        sample_count = 0
        batch_count = 0
        
        try:
            while sample_count < samples_to_send:
                # 计算本批次大小
                current_batch_size = min(batch_size, samples_to_send - sample_count)
                
                # 获取本批次数据
                batch_signals = all_signals[sample_count:sample_count + current_batch_size]
                batch_labels = all_labels[sample_count:sample_count + current_batch_size]
                
                # 转换信号格式：如果是 (2, length) 格式，转换成复数
                converted_signals = []
                for signal in batch_signals:
                    if signal.shape[0] == 2:
                        # I/Q 格式：转换成复数
                        signal_complex = signal[0] + 1j * signal[1]
                        converted_signals.append(signal_complex)
                    else:
                        # 已经是复数或其他格式
                        converted_signals.append(signal)
                
                # 打包数据
                batch_data = {
                    'signals': np.stack(converted_signals, axis=0).astype(np.complex64),
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'snrs': np.zeros(current_batch_size, dtype=np.float32),  # 本地数据没有SNR
                    'timestamp': time.time(),
                    'batch_id': batch_count
                }
                
                # 发送数据
                if self.use_msgpack:
                    import msgpack
                    serialized_data = msgpack.packb(batch_data, use_bin_type=True)
                else:
                    serialized_data = pickle.dumps(batch_data)
                
                self.push_socket.send(serialized_data)
                
                # 流量控制
                self.rate_limiter.wait_if_needed(len(serialized_data))
                
                sample_count += current_batch_size
                batch_count += 1
                
                if batch_count % 10 == 0:
                    progress = 100 * sample_count / samples_to_send
                    avg_rate = self.rate_limiter.get_average_rate_mbps()
                    total_mb = self.rate_limiter.get_total_mb_sent()
                    print(f"[端侧-本地数据] 已发送 {batch_count} 批次, {sample_count}/{samples_to_send} 样本 ({progress:.1f}%), "
                          f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print(f"\n[端侧-本地数据] 用户中断，共发送 {batch_count} 批次, {sample_count} 样本")
        
        print(f"[端侧-本地数据] 发送完成，共发送 {batch_count} 批次, {sample_count} 样本")
        
        # 发送结束标志
        self._send_end_signal(sample_count, batch_count)
    
    def _convert_label_to_int(self, label):
        """将标签转换为整数
        
        Args:
            label: 标签，可能是整数、字符串或元组(class_name, snr)
            
        Returns:
            int: 整数标签
        """
        # 处理元组类型（Link11和RML2016数据集格式：(class_name, snr)）
        if isinstance(label, tuple):
            if len(label) >= 2:
                # 提取类别名称（第一个元素），忽略SNR（第二个元素）
                class_name = label[0]
                snr = label[1]
                
                if self.dataset_type == 'link11':
                    # Link11 数据集：飞机ID映射
                    if class_name in AIRCRAFT_ID_TO_LABEL:
                        return AIRCRAFT_ID_TO_LABEL[class_name]
                    else:
                        raise ValueError(f"未知的 Link11 飞机ID: {class_name} (来自元组 {label})")
                
                elif self.dataset_type == 'rml2016':
                    # RML2016 数据集：调制类型映射
                    # 顺序：['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
                    # 标签：   0       1       2       3       4       5
                    modulation_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
                    if class_name in modulation_types:
                        return modulation_types.index(class_name)
                    else:
                        raise ValueError(f"未知的 RML2016 调制类型: {class_name} (来自元组 {label})")
                
                else:
                    raise ValueError(f"数据集 {self.dataset_type} 不支持元组标签格式: {label}")
            else:
                raise ValueError(f"元组标签格式错误（长度不足2）: {label}")
        
        # 如果已经是整数，直接返回
        if isinstance(label, (int, np.integer)):
            return int(label)
        
        # 如果是字符串，根据数据集类型映射
        if isinstance(label, str):
            if self.dataset_type == 'link11':
                # Link11 数据集：飞机ID映射
                if label in AIRCRAFT_ID_TO_LABEL:
                    return AIRCRAFT_ID_TO_LABEL[label]
                else:
                    raise ValueError(f"未知的 Link11 飞机ID: {label}")
            
            elif self.dataset_type == 'rml2016':
                # RML2016 数据集：调制类型映射
                # 顺序：['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
                # 标签：   0       1       2       3       4       5
                modulation_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
                if label in modulation_types:
                    return modulation_types.index(label)
                else:
                    raise ValueError(f"未知的 RML2016 调制类型: {label}")
            
            elif self.dataset_type == 'radar':
                # Radar 数据集：飞机类型映射
                # P-8A: 0-2, P-3C: 3-4, E-2D: 5-6
                aircraft_map = {
                    'P-8A_1': 0, 'P-8A_2': 1, 'P-8A_3': 2,
                    'P-3C_1': 3, 'P-3C_2': 4,
                    'E-2D_1': 5, 'E-2D_2': 6
                }
                if label in aircraft_map:
                    return aircraft_map[label]
                else:
                    raise ValueError(f"未知的 Radar 飞机类型: {label}")
            
            else:
                # 其他数据集：尝试直接转换
                try:
                    return int(label)
                except ValueError:
                    raise ValueError(f"无法将标签 '{label}' 转换为整数")
        
        # 其他类型，尝试转换
        try:
            return int(label)
        except (ValueError, TypeError):
            raise ValueError(f"无法将标签 {label} (类型: {type(label)}) 转换为整数")
    
    def _load_and_send_from_folder(self, batch_size, interval, total_samples):
        """从文件夹逐批次加载并发送数据"""
        print(f"[端侧-本地数据] 开始发送数据（文件夹模式），批次大小: {batch_size}，间隔: {interval}s")
        print(f"[端侧-本地数据] 批次文件数: {len(self.batch_files)}")
        
        sample_count = 0
        batch_count = 0
        
        try:
            for file_idx, batch_file in enumerate(self.batch_files):
                # 检查是否达到总样本数限制
                if total_samples and sample_count >= total_samples:
                    print(f"\n[端侧-本地数据] 已达到总样本数限制: {total_samples}")
                    break
                
                # 加载批次文件
                print(f"[端侧-本地数据] 加载批次文件 {file_idx + 1}/{len(self.batch_files)}: {os.path.basename(batch_file)}")
                batch_raw_data = self._load_data(batch_file)
                
                # 准备批次数据
                batch_signals = []
                batch_labels = []
                batch_snrs = []
                
                # 批次数据格式：{(class, snr): signal_array}
                for key, signal_array in batch_raw_data.items():
                    # key 是元组 (class, snr)
                    if isinstance(key, tuple):
                        class_label = key[0]  # 第一个元素是类别
                        snr_value = key[1] if len(key) > 1 else 0.0  # 第二个元素是SNR
                    else:
                        # 如果不是元组，直接作为标签
                        class_label = key
                        snr_value = 0.0
                    
                    # 转换标签为整数
                    numeric_label = self._convert_label_to_int(class_label)
                    
                    for signal in signal_array:
                        # 检查是否达到总样本数限制
                        if total_samples and sample_count >= total_samples:
                            break
                        
                        batch_signals.append(signal)
                        batch_labels.append(numeric_label)
                        batch_snrs.append(snr_value)
                        
                        # 当达到批次大小时发送
                        if len(batch_signals) >= batch_size:
                            self._send_batch(batch_signals, batch_labels, batch_snrs, batch_count)
                            sample_count += len(batch_signals)
                            batch_count += 1
                            
                            if batch_count % 10 == 0:
                                if total_samples:
                                    progress = 100 * sample_count / total_samples
                                    avg_rate = self.rate_limiter.get_average_rate_mbps()
                                    total_mb = self.rate_limiter.get_total_mb_sent()
                                    print(f"[端侧-本地数据] 已发送 {batch_count} 批次, {sample_count}/{total_samples} 样本 ({progress:.1f}%), "
                                          f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                                else:
                                    avg_rate = self.rate_limiter.get_average_rate_mbps()
                                    total_mb = self.rate_limiter.get_total_mb_sent()
                                    print(f"[端侧-本地数据] 已发送 {batch_count} 批次, {sample_count} 样本, "
                                          f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                            
                            batch_signals = []
                            batch_labels = []
                            batch_snrs = []
                            time.sleep(interval)
                    
                    if total_samples and sample_count >= total_samples:
                        break
                
                # 发送剩余数据
                if batch_signals:
                    self._send_batch(batch_signals, batch_labels, batch_snrs, batch_count)
                    sample_count += len(batch_signals)
                    batch_count += 1
                    time.sleep(interval)
        
        except KeyboardInterrupt:
            print(f"\n[端侧-本地数据] 用户中断，共发送 {batch_count} 批次, {sample_count} 样本")
        
        print(f"[端侧-本地数据] 发送完成，共发送 {batch_count} 批次, {sample_count} 样本")
        
        # 发送结束标志
        self._send_end_signal(sample_count, batch_count)
    
    def _send_batch(self, batch_signals, batch_labels, batch_snrs, batch_count):
        """发送一个批次的数据"""
        # 转换信号格式：如果是 (2, length) 格式，转换成复数
        converted_signals = []
        for signal in batch_signals:
            if signal.shape[0] == 2:
                # I/Q 格式：转换成复数
                signal_complex = signal[0] + 1j * signal[1]
                converted_signals.append(signal_complex)
            else:
                # 已经是复数或其他格式
                converted_signals.append(signal)
        
        batch_data = {
            'signals': np.stack(converted_signals, axis=0).astype(np.complex64),
            'labels': np.array(batch_labels, dtype=np.int64),
            'snrs': np.array(batch_snrs, dtype=np.float32),
            'timestamp': time.time(),
            'batch_id': batch_count
        }
        
        if self.use_msgpack:
            import msgpack
            serialized_data = msgpack.packb(batch_data, use_bin_type=True)
        else:
            serialized_data = pickle.dumps(batch_data)
        
        self.push_socket.send(serialized_data)
        
        # 流量控制
        self.rate_limiter.wait_if_needed(len(serialized_data))
    
    def _send_end_signal(self, sample_count, batch_count):
        """发送结束标志并等待边侧完成"""
        try:
            print(f"[端侧-本地数据] 发送结束标志...")
            end_signal = {
                'type': 'end_transmission',
                'total_samples': sample_count,
                'total_batches': batch_count,
                'timestamp': time.time()
            }
            if self.use_msgpack:
                import msgpack
                self.push_socket.send(msgpack.packb(end_signal, use_bin_type=True))
            else:
                self.push_socket.send(pickle.dumps(end_signal))
            print(f"[端侧-本地数据] 结束标志已发送")
            
            # ✅ 等待边侧返回"推理完成"标志
            print(f"[端侧-本地数据] 等待边侧推理完成...")
            response_data = self.pull_socket.recv()  # 阻塞等待，有600秒超时
            
            # 反序列化
            if self.use_msgpack:
                import msgpack
                response = msgpack.unpackb(response_data, raw=False)
            else:
                response = pickle.loads(response_data)
            
            if response.get('type') == 'inference_complete':
                print(f"[端侧-本地数据] 收到边侧推理完成标志")
            else:
                print(f"[端侧-本地数据] 收到未知响应类型: {response.get('type')}")
                
        except zmq.Again:
            print(f"[端侧-本地数据] 警告：等待边侧响应超时（600秒）")
        except Exception as e:
            print(f"[端侧-本地数据] 发送结束标志或接收响应失败: {e}")
    
    def close(self):
        """关闭连接"""
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()
        print("[端侧-本地数据] 连接已关闭")


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间水平距离（大圆距离），单位：米"""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    return R_earth * 2 * np.arcsin(np.sqrt(a))


class Link11SignalGenerator:
    """Link11 信号生成器"""
    
    def __init__(self, aircraft_id, receiver_lat=25.45, receiver_lon=122.07, receiver_height=2):
        self.aircraft_id = aircraft_id
        params = aircraft_parameters[aircraft_id]
        self.type = params['type']
        self.f = params['frequency']
        self.lambda_ = c / self.f
        self.Pt = params['transmit_power']
        self.Gt = params['tx_gain']
        self.Gr = params['rx_gain']
        self.Ht = params['height']
        self.Hr = receiver_height
        self.noise_figure = params['noise_figure']
        self.B = params['bandwidth']
        self.symbol_rate = params['symbol_rate']
        
        # 接收机位置
        self.receiver_lat = receiver_lat
        self.receiver_lon = receiver_lon
        
        # 航线参数
        self.start_lat = params['start_lat']
        self.start_lon = params['start_lon']
        self.end_lat = params['end_lat']
        self.end_lon = params['end_lon']
        
        # 当前位置（0-1 之间的进度）
        self.position = np.random.random()  # 随机初始位置
        
        # 噪声功率
        noise_power_watts = k * T * self.B
        self.noise_power_dBW = 10 * np.log10(noise_power_watts) + self.noise_figure
        
        # 相位偏移
        self.phase_offset = individual_phase_offset[aircraft_id]
        
        # 唯一标识码
        id_hash = hash(aircraft_id) % (2**16)
        self.identifier = np.unpackbits(np.array([id_hash], dtype=np.uint16).view(np.uint8))
    
    def _get_current_position(self):
        """获取当前位置的经纬度"""
        lat = self.start_lat + (self.end_lat - self.start_lat) * self.position
        lon = self.start_lon + (self.end_lon - self.start_lon) * self.position
        return lat, lon
    
    def _update_position(self, speed=0.0001):
        """更新位置（模拟飞行）"""
        self.position += speed
        if self.position > 1.0:
            self.position = 0.0  # 循环航线
    
    def _calculate_snr(self, lat, lon):
        """计算当前位置的SNR"""
        horizontal_dist = haversine_distance(self.receiver_lat, self.receiver_lon, lat, lon)
        d_line = np.sqrt(horizontal_dist**2 + (self.Ht - self.Hr)**2)
        
        # 自由空间损耗
        fs_loss = 20 * np.log10(4 * np.pi * d_line / self.lambda_)
        
        # 额外损耗
        d_los = np.sqrt(2 * R_earth * self.Ht) + np.sqrt(2 * R_earth * self.Hr)
        if horizontal_dist > d_los:
            excess = horizontal_dist - d_los
            if self.type == 'E-2D':
                extra_loss = 0.3 * (excess / 1000)
            elif self.type == 'P-8A':
                extra_loss = 0.25 * (excess / 1000)
            else:
                extra_loss = 0.07 * (excess / 1000)
        else:
            extra_loss = 0
        
        total_loss = fs_loss + extra_loss
        Pt_dBW = 10 * np.log10(self.Pt)
        received_power = Pt_dBW + self.Gt + self.Gr - total_loss
        return received_power - self.noise_power_dBW
    
    def _generate_frame(self, frame_length=128):
        """生成 Link11 帧"""
        preamble = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        data = np.random.randint(0, 2, frame_length - len(preamble) - len(self.identifier) - 8)
        checksum = np.sum(np.concatenate([self.identifier, data])) % 256
        checksum_bits = np.unpackbits(np.array([checksum], dtype=np.uint8))
        return np.concatenate([preamble, self.identifier, data, checksum_bits])
    
    def generate_signal(self, samples_per_symbol=8):
        """生成一个信号样本"""
        # 更新位置
        self._update_position()
        
        # 获取当前位置和SNR
        lat, lon = self._get_current_position()
        snr_db = self._calculate_snr(lat, lon)
        
        # 生成帧数据
        frame_data = self._generate_frame()
        num_symbols = len(frame_data)
        num_samples = num_symbols * samples_per_symbol
        
        # 2FSK调制
        f0, f1 = 1000, 2000
        fs = samples_per_symbol * self.symbol_rate
        t = np.arange(num_samples) / fs
        
        real_signal = np.zeros(num_samples)
        for i, bit in enumerate(frame_data):
            start_idx = i * samples_per_symbol
            end_idx = start_idx + samples_per_symbol
            freq = f1 if bit == 1 else f0
            real_signal[start_idx:end_idx] = np.cos(2 * np.pi * freq * t[start_idx:end_idx])
        
        # 高斯滤波
        b = np.exp(-(np.arange(-16, 17) ** 2) / (2 * 3 ** 2))
        b /= np.sum(b)
        real_signal = lfilter(b, 1, real_signal)
        
        # 加入相位偏移，转为复信号
        theta = self.phase_offset
        complex_signal = real_signal * np.cos(theta) + 1j * real_signal * np.sin(theta)
        
        # 加噪声
        signal_power = np.mean(np.abs(complex_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        noisy_signal = complex_signal + noise
        
        # 保持复数格式
        return noisy_signal.astype(np.complex64), snr_db


class Link11DeviceSimulator:
    """端侧设备模拟器"""
    
    def __init__(self, edge_host='localhost', edge_data_port=7777, edge_response_port=7778, rate_limit_mbps=0.0):
        self.edge_host = edge_host
        self.edge_data_port = edge_data_port
        self.edge_response_port = edge_response_port
        
        # 创建限速器
        self.rate_limiter = RateLimiter(rate_limit_mbps)
        
        # 创建所有飞机的信号生成器
        self.generators = {
            aircraft_id: Link11SignalGenerator(aircraft_id)
            for aircraft_id in aircraft_parameters.keys()
        }
        
        # ZeroMQ 设置 - 双向连接
        self.context = zmq.Context()
        
        # PUSH socket：发送数据到边侧
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.SNDHWM, 1000)
        self.push_socket.setsockopt(zmq.SNDBUF, 16 * 1024 * 1024)
        self.push_socket.setsockopt(zmq.LINGER, 0)
        
        # PULL socket：接收边侧的完成标志
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 1000)
        self.pull_socket.setsockopt(zmq.RCVBUF, 16 * 1024 * 1024)
        self.pull_socket.setsockopt(zmq.RCVTIMEO, 600000)  # 600秒超时
        
        # 优化：使用 MessagePack 替代 Pickle（方案1）
        try:
            import msgpack
            import msgpack_numpy as m
            m.patch()
            self.use_msgpack = True
            print("[端侧] 使用 MessagePack 序列化（优化模式）")
        except ImportError:
            self.use_msgpack = False
            print("[端侧] MessagePack 未安装，使用 Pickle 序列化")
            print("[端侧] 提示：安装 msgpack-numpy 可提升2-3倍传输速度")
            print("[端侧]   pip install msgpack-numpy")
        
    def connect(self):
        """连接到边侧（双向连接）"""
        # 连接PUSH socket（发送数据）
        data_addr = f"tcp://{self.edge_host}:{self.edge_data_port}"
        self.push_socket.connect(data_addr)
        
        # 连接PULL socket（接收完成标志）
        response_addr = f"tcp://{self.edge_host}:{self.edge_response_port}"
        self.pull_socket.connect(response_addr)
        
        print(f"[端侧] 已连接到边侧:")
        print(f"  数据端口: {data_addr}")
        print(f"  响应端口: {response_addr}")
    
    def generate_and_send(self, batch_size=32, interval=0.1, total_samples=None):
        """生成数据并发送给边侧
        
        Args:
            batch_size: 每批次样本数
            interval: 发送间隔（秒）
            total_samples: 总样本数限制（None表示无限制）
        """
        aircraft_ids = list(self.generators.keys())
        sample_count = 0
        batch_count = 0
        
        if total_samples:
            print(f"[端侧] 开始生成数据，批次大小: {batch_size}，间隔: {interval}s，总样本数: {total_samples}")
        else:
            print(f"[端侧] 开始生成数据，批次大小: {batch_size}，间隔: {interval}s（无限制）")
        
        try:
            while True:
                # 检查是否达到总样本数限制
                if total_samples and sample_count >= total_samples:
                    print(f"\n[端侧] 已达到总样本数限制: {total_samples}")
                    break
                
                # 计算本批次实际大小（最后一批可能不足batch_size）
                if total_samples:
                    current_batch_size = min(batch_size, total_samples - sample_count)
                else:
                    current_batch_size = batch_size
                
                # 生成一个批次的数据
                batch_signals = []
                batch_labels = []
                batch_snrs = []
                
                for _ in range(current_batch_size):
                    # 随机选择一个飞机
                    aircraft_id = np.random.choice(aircraft_ids)
                    generator = self.generators[aircraft_id]
                    
                    # 生成信号
                    signal, snr = generator.generate_signal()
                    label = AIRCRAFT_ID_TO_LABEL[aircraft_id]
                    
                    batch_signals.append(signal)
                    batch_labels.append(label)
                    batch_snrs.append(snr)
                
                # 打包数据
                batch_data = {
                    'signals': np.stack(batch_signals, axis=0),  # (batch_size, signal_length) complex64
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'snrs': np.array(batch_snrs, dtype=np.float32),
                    'timestamp': time.time(),
                    'batch_id': batch_count
                }
                
                # 发送到边侧（使用 MessagePack 或 Pickle）
                if self.use_msgpack:
                    import msgpack
                    serialized_data = msgpack.packb(batch_data, use_bin_type=True)
                else:
                    serialized_data = pickle.dumps(batch_data)
                
                self.push_socket.send(serialized_data)
                
                # 流量控制
                self.rate_limiter.wait_if_needed(len(serialized_data))
                
                # 显式删除大对象，帮助垃圾回收（可选）
                del batch_data, batch_signals, batch_labels, batch_snrs
                
                sample_count += current_batch_size
                batch_count += 1
                
                if batch_count % 10 == 0:
                    avg_rate = self.rate_limiter.get_average_rate_mbps()
                    total_mb = self.rate_limiter.get_total_mb_sent()
                    if total_samples:
                        progress = 100 * sample_count / total_samples
                        print(f"[端侧] 已发送 {batch_count} 批次, {sample_count}/{total_samples} 样本 ({progress:.1f}%), "
                              f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                    else:
                        print(f"[端侧] 已发送 {batch_count} 批次, {sample_count} 样本, "
                              f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                
                # 等待一段时间（模拟实时数据流）
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n[端侧] 用户中断，共发送 {batch_count} 批次, {sample_count} 样本")
        
        print(f"[端侧] 发送完成，共发送 {batch_count} 批次, {sample_count} 样本")
        
        # 发送结束标志（使用与数据相同的序列化方法）
        try:
            print(f"[端侧] 发送结束标志...")
            end_signal = {
                'type': 'end_transmission',
                'total_samples': sample_count,
                'total_batches': batch_count,
                'timestamp': time.time()
            }
            # 使用与数据批次相同的序列化方法
            if self.use_msgpack:
                import msgpack
                self.push_socket.send(msgpack.packb(end_signal, use_bin_type=True))
            else:
                self.push_socket.send(pickle.dumps(end_signal))
            print(f"[端侧] 结束标志已发送")
        except Exception as e:
            print(f"[端侧] 发送结束标志失败: {e}")
        
        # ✅ 等待边侧返回"推理完成"标志
        try:
            print(f"[端侧] 等待边侧推理完成...")
            response_data = self.pull_socket.recv()  # 阻塞等待，有60秒超时
            
            # 反序列化
            if self.use_msgpack:
                import msgpack
                response = msgpack.unpackb(response_data, raw=False)
            else:
                response = pickle.loads(response_data)
            
            if response.get('type') == 'inference_complete':
                print(f"[端侧] 收到边侧推理完成标志")
            else:
                print(f"[端侧] 收到未知响应类型: {response.get('type')}")
                
        except zmq.Again:
            print(f"[端侧] 警告：等待边侧响应超时（600秒）")
        except Exception as e:
            print(f"[端侧] 接收边侧响应失败: {e}")
    
    def close(self):
        """关闭连接"""
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()
        print("[端侧] 连接已关闭")


# ============== RML2016 数据集模拟器 ==============

class RML2016DeviceSimulator:
    """RML2016 数据集设备模拟器"""
    
    def __init__(self, edge_host='localhost', edge_port=7777, edge_response_port=7778, rate_limit_mbps=0.0):
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.edge_response_port = edge_response_port
        
        # 创建限速器
        self.rate_limiter = RateLimiter(rate_limit_mbps)
        
        # 调制类型（必须与数据加载代码保持一致！）
        # 顺序：['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
        # 标签：   0       1       2       3       4       5
        self.modulations = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
        
        # ZeroMQ 设置 - 双向连接
        self.context = zmq.Context()
        
        # PUSH socket：发送数据
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.SNDHWM, 1000)
        self.push_socket.setsockopt(zmq.SNDBUF, 16 * 1024 * 1024)
        self.push_socket.setsockopt(zmq.LINGER, 0)
        
        # PULL socket：接收完成标志
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 1000)
        self.pull_socket.setsockopt(zmq.RCVBUF, 16 * 1024 * 1024)
        self.pull_socket.setsockopt(zmq.RCVTIMEO, 600000)  # 600秒超时
        
        # 优化：使用 MessagePack 替代 Pickle（方案1）
        try:
            import msgpack
            import msgpack_numpy as m
            m.patch()
            self.use_msgpack = True
            print("[端侧-RML2016] 使用 MessagePack 序列化（优化模式）")
        except ImportError:
            self.use_msgpack = False
            print("[端侧-RML2016] MessagePack 未安装，使用 Pickle 序列化")
            print("[端侧-RML2016] 提示：安装 msgpack-numpy 可提升2-3倍传输速度")
            print("[端侧-RML2016]   pip install msgpack-numpy")
    
    def connect(self):
        """连接到边侧（双向连接）"""
        # 连接PUSH socket（发送数据）
        data_addr = f"tcp://{self.edge_host}:{self.edge_port}"
        self.push_socket.connect(data_addr)
        
        # 连接PULL socket（接收完成标志）
        response_addr = f"tcp://{self.edge_host}:{self.edge_response_port}"
        self.pull_socket.connect(response_addr)
        
        print(f"[端侧-RML2016] 已连接到边侧:")
        print(f"  数据端口: {data_addr}")
        print(f"  响应端口: {response_addr}")
    
    def generate_signal(self, modulation, num_samples=600):
        """生成 RML2016 调制信号（简化版本）"""
        # 这里使用简化的信号生成逻辑
        # 实际应用中可以使用更复杂的调制方案
        
        if modulation == 'BPSK':
            symbols = np.random.randint(0, 2, num_samples // 4)
            phases = symbols * np.pi
            signal = np.exp(1j * phases)
            signal = np.repeat(signal, 4)[:num_samples]
        
        elif modulation == 'QPSK':
            symbols = np.random.randint(0, 4, num_samples // 4)
            phases = (2 * np.pi * symbols / 4)
            signal = np.exp(1j * phases)
            signal = np.repeat(signal, 4)[:num_samples]
        
        elif modulation == '8PSK':
            symbols = np.random.randint(0, 8, num_samples // 4)
            phases = (2 * np.pi * symbols / 8)
            signal = np.exp(1j * phases)
            signal = np.repeat(signal, 4)[:num_samples]
        
        elif modulation == '16QAM':
            symbols = np.random.randint(0, 16, num_samples // 4)
            i = 2 * ((symbols // 4) % 4) - 3
            q = 2 * (symbols % 4) - 3
            signal = (i + 1j * q) * 0.4
            signal = np.repeat(signal, 4)[:num_samples]
        
        elif modulation == '64QAM':
            symbols = np.random.randint(0, 64, num_samples // 4)
            i = 2 * ((symbols // 8) % 8) - 7
            q = 2 * (symbols % 8) - 7
            signal = (i + 1j * q) * 0.4
            signal = np.repeat(signal, 4)[:num_samples]
        
        elif modulation == 'GMSK':
            num_bits = num_samples // 8 + 2
            bits = np.random.randint(0, 2, num_bits)
            data = 2 * bits - 1
            t = np.linspace(-3, 3, 17)
            gaussian = np.exp(-(np.pi**2 * 0.3**2 * t**2) / np.log(2))
            gaussian /= np.sum(gaussian)
            shaped_data = np.zeros(num_bits * 8)
            shaped_data[::8] = data
            filtered_data = np.convolve(shaped_data, gaussian, mode='same')
            phase = np.cumsum(filtered_data) * (np.pi / 2)
            signal = np.exp(1j * phase)[:num_samples]
        
        else:
            signal = np.exp(1j * np.random.randn(num_samples))
        
        # 添加噪声
        snr_db = np.random.uniform(-10, 20)
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        return (signal + noise).astype(np.complex64), snr_db
    
    def generate_and_send(self, batch_size=100, interval=0.1, total_samples=None):
        """生成数据并发送给边侧
        
        Args:
            batch_size: 每批次样本数
            interval: 发送间隔（秒）
            total_samples: 总样本数限制（None表示无限制）
        """
        sample_count = 0
        batch_count = 0
        
        if total_samples:
            print(f"[端侧-RML2016] 开始生成 RML2016 数据，批次大小: {batch_size}，间隔: {interval}s，总样本数: {total_samples}")
        else:
            print(f"[端侧-RML2016] 开始生成 RML2016 数据，批次大小: {batch_size}，间隔: {interval}s（无限制）")
        
        try:
            while True:
                # 检查是否达到总样本数限制
                if total_samples and sample_count >= total_samples:
                    print(f"\n[端侧-RML2016] 已达到总样本数限制: {total_samples}")
                    break
                
                # 计算本批次实际大小
                if total_samples:
                    current_batch_size = min(batch_size, total_samples - sample_count)
                else:
                    current_batch_size = batch_size
                
                # 生成一个批次的数据
                batch_signals = []
                batch_labels = []
                batch_snrs = []
                
                for _ in range(current_batch_size):
                    # 随机选择一个调制类型
                    modulation = np.random.choice(self.modulations)
                    
                    # 生成信号
                    signal, snr = self.generate_signal(modulation)
                    label = self.modulations.index(modulation)
                    
                    batch_signals.append(signal)
                    batch_labels.append(label)
                    batch_snrs.append(snr)
                
                # 打包数据
                batch_data = {
                    'signals': np.stack(batch_signals, axis=0),  # (batch_size, signal_length) complex64
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'snrs': np.array(batch_snrs, dtype=np.float32),
                    'timestamp': time.time(),
                    'batch_id': batch_count
                }
                
                # 发送到边侧（使用 MessagePack 或 Pickle）
                if self.use_msgpack:
                    import msgpack
                    serialized_data = msgpack.packb(batch_data, use_bin_type=True)
                else:
                    serialized_data = pickle.dumps(batch_data)
                
                self.push_socket.send(serialized_data)
                
                # 流量控制
                self.rate_limiter.wait_if_needed(len(serialized_data))
                
                # 显式删除大对象
                del batch_data, batch_signals, batch_labels, batch_snrs
                
                sample_count += current_batch_size
                batch_count += 1
                
                if batch_count % 10 == 0:
                    avg_rate = self.rate_limiter.get_average_rate_mbps()
                    total_mb = self.rate_limiter.get_total_mb_sent()
                    if total_samples:
                        progress = 100 * sample_count / total_samples
                        print(f"[端侧-RML2016] 已发送 {batch_count} 批次, {sample_count}/{total_samples} 样本 ({progress:.1f}%), "
                              f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                    else:
                        print(f"[端侧-RML2016] 已发送 {batch_count} 批次, {sample_count} 样本, "
                              f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n[端侧-RML2016] 用户中断，共发送 {batch_count} 批次, {sample_count} 样本")
        
        print(f"[端侧-RML2016] 发送完成，共发送 {batch_count} 批次, {sample_count} 样本")
        
        # 发送结束标志并等待边侧完成
        try:
            print(f"[端侧-RML2016] 发送结束标志...")
            end_signal = {
                'type': 'end_transmission',
                'total_samples': sample_count,
                'total_batches': batch_count,
                'timestamp': time.time()
            }
            if self.use_msgpack:
                import msgpack
                self.push_socket.send(msgpack.packb(end_signal, use_bin_type=True))
            else:
                self.push_socket.send(pickle.dumps(end_signal))
            print(f"[端侧-RML2016] 结束标志已发送")
            
            # ✅ 等待边侧返回"推理完成"标志
            print(f"[端侧-RML2016] 等待边侧推理完成...")
            response_data = self.pull_socket.recv()  # 阻塞等待，有600秒超时
            
            # 反序列化
            if self.use_msgpack:
                import msgpack
                response = msgpack.unpackb(response_data, raw=False)
            else:
                response = pickle.loads(response_data)
            
            if response.get('type') == 'inference_complete':
                print(f"[端侧-RML2016] 收到边侧推理完成标志")
            else:
                print(f"[端侧-RML2016] 收到未知响应类型: {response.get('type')}")
                
        except zmq.Again:
            print(f"[端侧-RML2016] 警告：等待边侧响应超时（600秒）")
        except Exception as e:
            print(f"[端侧-RML2016] 发送结束标志或接收响应失败: {e}")
    
    def close(self):
        """关闭连接"""
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()
        print("[端侧-RML2016] 连接已关闭")


# ============== Radar 数据集模拟器 ==============

class RadarDeviceSimulator:
    """Radar 数据集设备模拟器"""
    
    def __init__(self, edge_host='localhost', edge_port=7777, edge_response_port=7778, rate_limit_mbps=0.0):
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.edge_response_port = edge_response_port
        
        # 创建限速器
        self.rate_limiter = RateLimiter(rate_limit_mbps)
        
        # 飞机类型：P-8A (1-3), P-3C (4-5), E-2D (6-7)
        self.aircraft_types = ['P-8A', 'P-3C', 'E-2D']
        self.aircraft_individuals = {
            'P-8A': [1, 2, 3],
            'P-3C': [1, 2],
            'E-2D': [1, 2]
        }
        
        # ZeroMQ 设置 - 双向连接
        self.context = zmq.Context()
        
        # PUSH socket：发送数据
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.SNDHWM, 1000)
        self.push_socket.setsockopt(zmq.SNDBUF, 16 * 1024 * 1024)
        self.push_socket.setsockopt(zmq.LINGER, 0)
        
        # PULL socket：接收完成标志
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 1000)
        self.pull_socket.setsockopt(zmq.RCVBUF, 16 * 1024 * 1024)
        self.pull_socket.setsockopt(zmq.RCVTIMEO, 600000)  # 600秒超时
        
        # 优化：使用 MessagePack 替代 Pickle（方案1）
        try:
            import msgpack
            import msgpack_numpy as m
            m.patch()
            self.use_msgpack = True
            print("[端侧-Radar] 使用 MessagePack 序列化（优化模式）")
        except ImportError:
            self.use_msgpack = False
            print("[端侧-Radar] MessagePack 未安装，使用 Pickle 序列化")
            print("[端侧-Radar] 提示：安装 msgpack-numpy 可提升2-3倍传输速度")
            print("[端侧-Radar]   pip install msgpack-numpy")
    
    def connect(self):
        """连接到边侧（双向连接）"""
        # 连接PUSH socket（发送数据）
        data_addr = f"tcp://{self.edge_host}:{self.edge_port}"
        self.push_socket.connect(data_addr)
        
        # 连接PULL socket（接收完成标志）
        response_addr = f"tcp://{self.edge_host}:{self.edge_response_port}"
        self.pull_socket.connect(response_addr)
        
        print(f"[端侧-Radar] 已连接到边侧:")
        print(f"  数据端口: {data_addr}")
        print(f"  响应端口: {response_addr}")
    
    def generate_signal(self, aircraft_type, individual_idx, num_samples=500):
        """生成 Radar LFM 信号（简化版本）"""
        # 信号参数
        T = 5e-6  # 脉冲宽度
        fs = 100e6  # 采样率
        pulse_samples = int(fs * T)
        
        # 带宽根据飞机类型
        if aircraft_type == 'P-8A':
            B = 42e6
        elif aircraft_type == 'P-3C':
            B = 30e6
        else:  # E-2D
            B = 20e6
        
        # 生成 LFM 信号
        t_pulse = np.linspace(-T/2, T/2, pulse_samples)
        chirp_slope = B / T
        signal = np.exp(1j * np.pi * chirp_slope * t_pulse**2)
        
        # 添加相位噪声（简化版本）
        phase_noise = 0.1 * np.random.randn(pulse_samples)
        signal = signal * np.exp(1j * phase_noise)
        
        # 添加高斯噪声
        snr_db = np.random.uniform(0, 30)
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(pulse_samples) + 1j * np.random.randn(pulse_samples))
        signal = signal + noise
        
        # 归一化
        signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))
        
        return signal.astype(np.complex64), snr_db
    
    def generate_and_send(self, batch_size=100, interval=0.1, total_samples=None):
        """生成数据并发送给边侧
        
        Args:
            batch_size: 每批次样本数
            interval: 发送间隔（秒）
            total_samples: 总样本数限制（None表示无限制）
        """
        sample_count = 0
        batch_count = 0
        
        if total_samples:
            print(f"[端侧-Radar] 开始生成 Radar 数据，批次大小: {batch_size}，间隔: {interval}s，总样本数: {total_samples}")
        else:
            print(f"[端侧-Radar] 开始生成 Radar 数据，批次大小: {batch_size}，间隔: {interval}s（无限制）")
        
        try:
            while True:
                # 检查是否达到总样本数限制
                if total_samples and sample_count >= total_samples:
                    print(f"\n[端侧-Radar] 已达到总样本数限制: {total_samples}")
                    break
                
                # 计算本批次实际大小
                if total_samples:
                    current_batch_size = min(batch_size, total_samples - sample_count)
                else:
                    current_batch_size = batch_size
                
                # 生成一个批次的数据
                batch_signals = []
                batch_labels = []
                batch_snrs = []
                
                for _ in range(current_batch_size):
                    # 随机选择飞机类型和个体
                    aircraft_type = np.random.choice(self.aircraft_types)
                    individual_idx = np.random.choice(self.aircraft_individuals[aircraft_type])
                    
                    # 生成信号
                    signal, snr = self.generate_signal(aircraft_type, individual_idx)
                    
                    # 计算标签
                    if aircraft_type == 'P-8A':
                        label = individual_idx - 1  # 0, 1, 2
                    elif aircraft_type == 'P-3C':
                        label = 3 + individual_idx - 1  # 3, 4
                    else:  # E-2D
                        label = 5 + individual_idx - 1  # 5, 6
                    
                    batch_signals.append(signal)
                    batch_labels.append(label)
                # 打包数据
                batch_data = {
                    'signals': np.stack(batch_signals, axis=0),  # (batch_size, signal_length) complex64
                    'labels': np.array(batch_labels, dtype=np.int64),
                    'snrs': np.array(batch_snrs, dtype=np.float32),
                    'timestamp': time.time(),
                    'batch_id': batch_count
                }
                
                # 发送到边侧（使用 MessagePack 或 Pickle）
                if self.use_msgpack:
                    import msgpack
                    serialized_data = msgpack.packb(batch_data, use_bin_type=True)
                else:
                    serialized_data = pickle.dumps(batch_data)
                
                self.push_socket.send(serialized_data)
                
                # 流量控制
                self.rate_limiter.wait_if_needed(len(serialized_data))
                
                # 显式删除大对象
                del batch_data, batch_signals, batch_labels, batch_snrs
                
                sample_count += current_batch_size
                batch_count += 1
                
                if batch_count % 10 == 0:
                    avg_rate = self.rate_limiter.get_average_rate_mbps()
                    total_mb = self.rate_limiter.get_total_mb_sent()
                    if total_samples:
                        progress = 100 * sample_count / total_samples
                        print(f"[端侧-Radar] 已发送 {batch_count} 批次, {sample_count}/{total_samples} 样本 ({progress:.1f}%), "
                              f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                    else:
                        print(f"[端侧-Radar] 已发送 {batch_count} 批次, {sample_count} 样本, "
                              f"速率: {avg_rate:.2f}MB/s, 累计: {total_mb:.2f}MB")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n[端侧-Radar] 用户中断，共发送 {batch_count} 批次, {sample_count} 样本")
        
        print(f"[端侧-Radar] 发送完成，共发送 {batch_count} 批次, {sample_count} 样本")
        
        # 发送结束标志并等待边侧完成
        try:
            print(f"[端侧-Radar] 发送结束标志...")
            end_signal = {
                'type': 'end_transmission',
                'total_samples': sample_count,
                'total_batches': batch_count,
                'timestamp': time.time()
            }
            if self.use_msgpack:
                import msgpack
                self.push_socket.send(msgpack.packb(end_signal, use_bin_type=True))
            else:
                self.push_socket.send(pickle.dumps(end_signal))
            print(f"[端侧-Radar] 结束标志已发送")
            
            # ✅ 等待边侧返回"推理完成"标志
            print(f"[端侧-Radar] 等待边侧推理完成...")
            response_data = self.pull_socket.recv()  # 阻塞等待，有600秒超时
            
            # 反序列化
            if self.use_msgpack:
                import msgpack
                response = msgpack.unpackb(response_data, raw=False)
            else:
                response = pickle.loads(response_data)
            
            if response.get('type') == 'inference_complete':
                print(f"[端侧-Radar] 收到边侧推理完成标志")
            else:
                print(f"[端侧-Radar] 收到未知响应类型: {response.get('type')}")
                
        except zmq.Again:
            print(f"[端侧-Radar] 警告：等待边侧响应超时（600秒）")
        except Exception as e:
            print(f"[端侧-Radar] 发送结束标志或接收响应失败: {e}")
    
    def close(self):
        """关闭连接"""
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()
        print("[端侧-Radar] 连接已关闭")


def send_to_cloud_directly(args):
    """
    端侧直接连接云侧进行推理（绕过边侧）
    使用 ZeroMQ PUSH-PULL 模式
    
    支持两种数据来源：
    1. 实时生成（data_source='generate'）
    2. 本地文件加载（data_source='local'）
    """
    print(f"[端侧→云侧] 建立 ZeroMQ 直连模式...")
    print(f"[端侧→云侧] 数据来源: {args.data_source}")
    
    # 根据数据来源准备数据
    if args.data_source == 'generate':
        # 创建信号生成器
        if args.dataset_type == 'link11':
            generators = {
                aircraft_id: Link11SignalGenerator(aircraft_id)
                for aircraft_id in aircraft_parameters.keys()
            }
            aircraft_ids = list(generators.keys())
        elif args.dataset_type == 'rml2016':
            modulations = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
        elif args.dataset_type == 'radar':
            aircraft_types = ['P-8A', 'P-3C', 'E-2D']
            aircraft_individuals = {
                'P-8A': [1, 2, 3],
                'P-3C': [1, 2],
                'E-2D': [1, 2]
            }
        data_loader = None
    
    elif args.data_source == 'local':
        # 加载本地数据
        print(f"[端侧→云侧] 加载本地数据: {args.data_path}")
        loader = LocalDataLoader(args.data_path, args.dataset_type, 'localhost', 0)
        
        # 准备所有数据
        if loader.is_folder:
            # 文件夹模式：逐批次加载
            data_loader = {
                'type': 'folder',
                'loader': loader
            }
        else:
            # 单文件模式：一次性加载
            all_signals = []
            all_labels = []
            
            for key, signal_array in loader.raw_data.items():
                # 处理元组 key
                if isinstance(key, tuple):
                    class_label = key[0]
                else:
                    class_label = key
                
                numeric_label = loader._convert_label_to_int(class_label)
                
                for signal in signal_array:
                    # 转换信号格式
                    if signal.shape[0] == 2:
                        signal_complex = signal[0] + 1j * signal[1]
                        all_signals.append(signal_complex)
                    else:
                        all_signals.append(signal)
                    all_labels.append(numeric_label)
            
            data_loader = {
                'type': 'file',
                'signals': all_signals,
                'labels': all_labels
            }
            print(f"[端侧→云侧] 数据加载完成: {len(all_signals)} 样本")
        
        generators = None
        aircraft_ids = None
        modulations = None
        aircraft_types = None
        aircraft_individuals = None
    
    # 创建 ZeroMQ 连接
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    pull_socket = context.socket(zmq.PULL)
    
    # 设置缓冲区
    push_socket.setsockopt(zmq.SNDHWM, 10000)
    push_socket.setsockopt(zmq.SNDBUF, 64 * 1024 * 1024)
    push_socket.setsockopt(zmq.LINGER, 0)
    
    pull_socket.setsockopt(zmq.RCVHWM, 10000)
    pull_socket.setsockopt(zmq.RCVBUF, 64 * 1024 * 1024)
    pull_socket.setsockopt(zmq.RCVTIMEO, 600000)  # 600秒超时
    
    # MessagePack 支持
    try:
        import msgpack
        import msgpack_numpy as m
        m.patch()
        use_msgpack = True
        print(f"[端侧→云侧] 使用 MessagePack 序列化")
    except ImportError:
        use_msgpack = False
        print(f"[端侧→云侧] 使用 Pickle 序列化")
    
    # 连接到云侧
    try:
        # 端侧 PUSH 到云侧 PULL 端口（发送请求）
        request_addr = f"tcp://{args.cloud_host}:{args.zmq_pull_port}"
        # 端侧 PULL 从云侧 PUSH 端口（接收响应）
        response_addr = f"tcp://{args.cloud_host}:{args.zmq_push_port}"
        
        push_socket.connect(request_addr)
        pull_socket.connect(response_addr)
        
        print(f"[端侧→云侧] 已连接到云侧:")
        print(f"  请求端口: {request_addr}")
        print(f"  响应端口: {response_addr}\n")
        
    except Exception as e:
        print(f"[端侧→云侧] 连接失败: {e}")
        push_socket.close()
        pull_socket.close()
        context.term()
        return
    
    
    # ========== 异步模式：创建多线程流水线 ==========
    import threading
    import queue
    from collections import defaultdict
    
    # 共享数据结构
    upload_queue = queue.Queue(maxsize=100)  # 待上传队列
    result_dict = {}  # {batch_id: {'predictions': [...], 'labels': [...], 'received': bool}}
    result_lock = threading.Lock()
    
    # 统计指标
    stats = {
        'total_samples': 0,
        'total_batches': 0,
        'correct_predictions': 0,
        'inference_times': [],
        'batch_times': [],
        'upload_times': [],
        'download_times': []
    }
    stats_lock = threading.Lock()
    
    # 停止事件和推理完成事件
    stop_event = threading.Event()
    inference_complete_event = threading.Event()  # ✅ 新增：标记云侧推理完成
    
    # 批次ID计数器
    batch_id_counter = [0]
    batch_id_lock = threading.Lock()
    
    def get_next_batch_id():
        with batch_id_lock:
            batch_id = batch_id_counter[0]
            batch_id_counter[0] += 1
            return batch_id
    
    # ========== 线程1：数据发送线程 ==========
    def send_thread_func():
        """持续发送数据到云侧（不等待响应）"""
        print(f"[发送线程] 启动")
        
        sample_count = 0
        batch_count = 0
        
        # 创建限速器
        rate_limiter = RateLimiter(args.rate_limit)
        
        try:
            while not stop_event.is_set():
                # 检查是否达到总样本数限制
                if args.total_samples and sample_count >= args.total_samples:
                    print(f"\n[发送线程] 已达到总样本数限制: {args.total_samples}")
                    break
                
                # 计算本批次实际大小
                if args.total_samples:
                    current_batch_size = min(args.batch_size, args.total_samples - sample_count)
                else:
                    current_batch_size = args.batch_size
                
                batch_start_time = time.time()
                
                # 生成一个批次的数据
                batch_signals = []
                batch_labels = []
                
                if args.data_source == 'generate':
                    # 实时生成数据
                    for _ in range(current_batch_size):
                        if args.dataset_type == 'link11':
                            aircraft_id = np.random.choice(aircraft_ids)
                            generator = generators[aircraft_id]
                            signal, snr = generator.generate_signal()
                            label = AIRCRAFT_ID_TO_LABEL[aircraft_id]
                        
                        elif args.dataset_type == 'rml2016':
                            modulation = np.random.choice(modulations)
                            signal, snr = generate_rml2016_signal(modulation)
                            label = modulations.index(modulation)
                        
                        elif args.dataset_type == 'radar':
                            aircraft_type = np.random.choice(aircraft_types)
                            individual_idx = np.random.choice(aircraft_individuals[aircraft_type])
                            signal, snr = generate_radar_signal(aircraft_type, individual_idx)
                            
                            if aircraft_type == 'P-8A':
                                label = individual_idx - 1
                            elif aircraft_type == 'P-3C':
                                label = 3 + individual_idx - 1
                            else:  # E-2D
                                label = 5 + individual_idx - 1
                        
                        batch_signals.append(signal)
                        batch_labels.append(label)
                
                elif args.data_source == 'local':
                    # 从本地数据加载
                    if data_loader['type'] == 'file':
                        # 单文件模式：从已加载的数据中取
                        start_idx = sample_count
                        end_idx = min(start_idx + current_batch_size, len(data_loader['signals']))
                        
                        if start_idx >= len(data_loader['signals']):
                            print(f"\n[发送线程] 所有数据已发送完毕")
                            break
                        
                        batch_signals = data_loader['signals'][start_idx:end_idx]
                        batch_labels = data_loader['labels'][start_idx:end_idx]
                        current_batch_size = len(batch_signals)
                    
                    elif data_loader['type'] == 'folder':
                        # 文件夹模式：逐批次加载
                        loader = data_loader['loader']
                        
                        # 如果还没有开始加载，初始化
                        if 'file_idx' not in data_loader:
                            data_loader['file_idx'] = 0
                            data_loader['current_batch_data'] = []
                            data_loader['current_batch_labels'] = []
                        
                        # 收集足够的数据组成一个批次
                        while len(data_loader['current_batch_data']) < current_batch_size:
                            # 检查是否还有文件
                            if data_loader['file_idx'] >= len(loader.batch_files):
                                # 所有文件都加载完了
                                if len(data_loader['current_batch_data']) == 0:
                                    print(f"\n[发送线程] 所有数据已发送完毕")
                                    break
                                else:
                                    # 发送剩余数据
                                    current_batch_size = len(data_loader['current_batch_data'])
                                    break
                            
                            # 加载下一个批次文件
                            batch_file = loader.batch_files[data_loader['file_idx']]
                            batch_raw_data = loader._load_data(batch_file)
                            data_loader['file_idx'] += 1
                            
                            # 解析批次数据
                            for key, signal_array in batch_raw_data.items():
                                # key 是元组 (class, snr)
                                if isinstance(key, tuple):
                                    class_label = key[0]
                                else:
                                    class_label = key
                                
                                numeric_label = loader._convert_label_to_int(class_label)
                                
                                for signal in signal_array:
                                    # 转换信号格式
                                    if signal.shape[0] == 2:
                                        signal_complex = signal[0] + 1j * signal[1]
                                        data_loader['current_batch_data'].append(signal_complex)
                                    else:
                                        data_loader['current_batch_data'].append(signal)
                                    data_loader['current_batch_labels'].append(numeric_label)
                        
                        # 检查是否收集到数据
                        if len(data_loader['current_batch_data']) == 0:
                            break
                        
                        # 取出一个批次
                        batch_signals = data_loader['current_batch_data'][:current_batch_size]
                        batch_labels = data_loader['current_batch_labels'][:current_batch_size]
                        
                        # 移除已取出的数据
                        data_loader['current_batch_data'] = data_loader['current_batch_data'][current_batch_size:]
                        data_loader['current_batch_labels'] = data_loader['current_batch_labels'][current_batch_size:]
                        
                        current_batch_size = len(batch_signals)
                
                # 转换为 numpy 数组
                batch_signals_np = np.stack(batch_signals, axis=0)  # (batch_size, signal_length)
                batch_labels_np = np.array(batch_labels, dtype=np.int64)
                
                # 分配批次ID
                batch_id = get_next_batch_id()
                
                # 记录到结果字典（等待接收）
                with result_lock:
                    result_dict[batch_id] = {
                        'labels': batch_labels_np.tolist(),
                        'predictions': None,
                        'received': False,
                        'send_time': time.time()
                    }
                
                # 发送推理请求到云侧（异步，不等待响应）
                upload_start = time.time()
                
                request = {
                    'type': 'cloud_inference',
                    'batch_id': batch_id,
                    'sample_indices': list(range(current_batch_size)),
                    'inputs': batch_signals_np,
                    'labels': batch_labels_np.tolist(),
                    'edge_id': 'device_direct',
                    'edge_send_time': time.time()
                }
                
                # 序列化数据
                if use_msgpack:
                    import msgpack
                    serialized_data = msgpack.packb(request, use_bin_type=True)
                else:
                    serialized_data = pickle.dumps(request)
                
                data_size_bytes = len(serialized_data)
                
                # 使用 ZeroMQ 发送请求（非阻塞）
                push_socket.send(serialized_data)
                
                upload_end = time.time()
                upload_time = (upload_end - upload_start) * 1000  # 毫秒
                
                # 流量控制
                rate_limiter.wait_if_needed(data_size_bytes)
                
                # 更新统计
                sample_count += current_batch_size
                batch_count += 1
                
                with stats_lock:
                    stats['total_samples'] += current_batch_size
                    stats['total_batches'] += 1
                    stats['upload_times'].append(upload_time)
                    stats['total_bytes_sent'] = stats.get('total_bytes_sent', 0) + data_size_bytes
                
                # 每50个批次打印一次
                if batch_count % 50 == 0:
                    with stats_lock:
                        avg_upload = np.mean(stats['upload_times'][-50:]) if stats['upload_times'] else 0
                        total_mb_sent = stats.get('total_bytes_sent', 0) / (1024 * 1024)
                    
                    # 获取传输速率
                    avg_rate = rate_limiter.get_average_rate_mbps()
                    total_mb = rate_limiter.get_total_mb_sent()
                    
                    if args.total_samples:
                        progress = 100 * sample_count / args.total_samples
                        print(f"[发送线程] 批次 {batch_count}: "
                              f"上传 {avg_upload:.2f}ms, "
                              f"速率 {avg_rate:.2f}MB/s, "
                              f"累计 {total_mb:.2f}MB, "
                              f"进度 {sample_count}/{args.total_samples} ({progress:.1f}%)")
                    else:
                        print(f"[发送线程] 批次 {batch_count}: "
                              f"上传 {avg_upload:.2f}ms, "
                              f"速率 {avg_rate:.2f}MB/s, "
                              f"累计 {total_mb:.2f}MB, "
                              f"已发送 {sample_count} 样本")
        
        except Exception as e:
            print(f"[发送线程] 错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print(f"[发送线程] 结束，共发送 {batch_count} 批次，{sample_count} 样本")
    
    # ========== 线程2：结果接收线程 ==========
    def receive_thread_func():
        """持续接收云侧返回的推理结果和完成标志"""
        print(f"[接收线程] 启动")
        
        received_count = 0
        
        try:
            while not stop_event.is_set():
                try:
                    # 使用 ZeroMQ 接收响应（带超时）
                    if pull_socket.poll(100):  # 100ms 超时
                        download_start = time.time()
                        
                        response_data = pull_socket.recv()
                        
                        download_end = time.time()
                        download_time = (download_end - download_start) * 1000  # 毫秒
                        
                        # 反序列化
                        if use_msgpack:
                            import msgpack
                            response = msgpack.unpackb(response_data, raw=False)
                        else:
                            response = pickle.loads(response_data)
                        
                        # ✅ 检查响应类型
                        response_type = response.get('type')
                        
                        if response_type == 'inference_complete':
                            # ✅ 收到推理完成标志
                            print(f"[接收线程] 收到云侧推理完成标志")
                            inference_complete_event.set()
                            break  # 退出接收循环
                        
                        # 处理推理结果
                        batch_id = response.get('batch_id')
                        predictions = response.get('predictions')
                        inference_time = response.get('inference_time', 0.0)
                        
                        if batch_id is not None and predictions is not None:
                            # 存入结果字典
                            with result_lock:
                                if batch_id in result_dict:
                                    result_dict[batch_id]['predictions'] = predictions
                                    result_dict[batch_id]['received'] = True
                                    result_dict[batch_id]['receive_time'] = time.time()
                                    
                                    # 计算准确率
                                    labels = result_dict[batch_id]['labels']
                                    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
                                    
                                    with stats_lock:
                                        stats['correct_predictions'] += correct
                                        stats['inference_times'].append(inference_time)
                                        stats['download_times'].append(download_time)
                            
                            received_count += 1
                            
                            # 每50个批次打印一次
                            if received_count % 50 == 0:
                                with stats_lock:
                                    if stats['total_samples'] > 0:
                                        accuracy = 100.0 * stats['correct_predictions'] / stats['total_samples']
                                    else:
                                        accuracy = 0.0
                                    avg_inference = np.mean(stats['inference_times'][-50:]) if stats['inference_times'] else 0
                                    avg_download = np.mean(stats['download_times'][-50:]) if stats['download_times'] else 0
                                
                                print(f"[接收线程] 已接收 {received_count} 批次: ")
                
                except zmq.Again:
                    # 超时，继续等待
                    continue
                except Exception as e:
                    print(f"[接收线程] 处理响应错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            print(f"[接收线程] 错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print(f"[接收线程] 结束，共接收 {received_count} 批次")
    
    # 启动线程
    send_thread = threading.Thread(target=send_thread_func, name="SendThread")
    receive_thread = threading.Thread(target=receive_thread_func, name="ReceiveThread")
    
    send_thread.start()
    receive_thread.start()
    
    start_time = time.time()
    
    if args.total_samples:
        print(f"[端侧→云侧] 开始异步推理测试，批次大小: {args.batch_size}，总样本数: {args.total_samples}\n")
    else:
        print(f"[端侧→云侧] 开始异步推理测试，批次大小: {args.batch_size}（无限制）\n")
    
    try:
        # 等待发送线程完成
        send_thread.join()
        print(f"\n[主线程] 发送线程已完成")
        
        # 发送结束标志
        print(f"[主线程] 发送结束标志...")
        with stats_lock:
            end_request = {
                'type': 'end_transmission',
                'edge_id': 'device_direct',
                'total_samples': stats['total_samples'],
                'total_batches': stats['total_batches']
            }
        
        # 使用 ZeroMQ 发送结束标志
        if use_msgpack:
            import msgpack
            push_socket.send(msgpack.packb(end_request, use_bin_type=True))
        else:
            push_socket.send(pickle.dumps(end_request))
        
        print(f"[主线程] 结束标志已发送")
        
        # ✅ 等待云侧返回"推理完成"标志
        print(f"[主线程] 等待云侧推理完成...")
        inference_complete_received = inference_complete_event.wait(timeout=600)  # 最多等待10分钟
        
        if inference_complete_received:
            print(f"[主线程] 云侧推理已完成，准备关闭连接")
        else:
            print(f"[主线程] 警告：等待超时（300秒），强制关闭连接")
        
    except KeyboardInterrupt:
        print(f"\n[主线程] 用户中断")
    
    finally:
        # 停止所有线程
        print(f"\n[主线程] 停止所有线程...")
        stop_event.set()
        
        send_thread.join(timeout=5.0)
        receive_thread.join(timeout=5.0)
        
        print(f"[主线程] 所有线程已停止\n")
    
    # ========== 异步模式已完成，跳过原来的同步循环 ==========
    # 原来的同步代码已被异步流水线替代
    
    # 关闭 ZeroMQ 连接
    push_socket.close()
    pull_socket.close()
    context.term()
    print(f"[端侧→云侧] ZeroMQ 连接已关闭")
    
    # 打印最终统计
    total_time = time.time() - start_time
    
    with stats_lock:
        print(f"\n{'='*70}")
        print(f"[端侧→云侧] 异步推理完成")
        print(f"{'='*70}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"总批次数: {stats['total_batches']}")
        print(f"正确预测: {stats['correct_predictions']}")
        if stats['total_samples'] > 0:
            accuracy = 100.0 * stats['correct_predictions'] / stats['total_samples']
            print(f"整体准确率: {accuracy:.2f}%")
        print(f"总耗时: {total_time:.2f}秒")
        
        if stats['inference_times']:
            avg_inference_time = np.mean(stats['inference_times'])
            min_inference_time = np.min(stats['inference_times'])
            max_inference_time = np.max(stats['inference_times'])
            print(f"平均推理时间: {avg_inference_time:.2f}ms")
            print(f"最小推理时间: {min_inference_time:.2f}ms")
            print(f"最大推理时间: {max_inference_time:.2f}ms")
        
        if stats['upload_times']:
            avg_upload_time = np.mean(stats['upload_times'])
            print(f"平均上传时间: {avg_upload_time:.2f}ms")
        
        if stats['download_times']:
            avg_download_time = np.mean(stats['download_times'])
            print(f"平均下载时间: {avg_download_time:.2f}ms")
        
        # 流量统计
        total_bytes_sent = stats.get('total_bytes_sent', 0)
        if total_bytes_sent > 0:
            total_mb_sent = total_bytes_sent / (1024 * 1024)
            avg_rate_mbps = total_mb_sent / total_time if total_time > 0 else 0
            print(f"总发送流量: {total_mb_sent:.2f}MB")
            print(f"平均发送速率: {avg_rate_mbps:.2f}MB/s")
        
        if stats['total_samples'] > 0 and total_time > 0:
            throughput = stats['total_samples'] / total_time
            print(f"吞吐量: {throughput:.2f} 样本/秒")
        
        print(f"{'='*70}\n")


def generate_rml2016_signal(modulation, num_samples=600):
    """生成 RML2016 调制信号（简化版本）"""
    if modulation == 'BPSK':
        symbols = np.random.randint(0, 2, num_samples // 4)
        phases = symbols * np.pi
        signal = np.exp(1j * phases)
        signal = np.repeat(signal, 4)[:num_samples]
    
    elif modulation == 'QPSK':
        symbols = np.random.randint(0, 4, num_samples // 4)
        phases = (2 * np.pi * symbols / 4)
        signal = np.exp(1j * phases)
        signal = np.repeat(signal, 4)[:num_samples]
    
    elif modulation == '8PSK':
        symbols = np.random.randint(0, 8, num_samples // 4)
        phases = (2 * np.pi * symbols / 8)
        signal = np.exp(1j * phases)
        signal = np.repeat(signal, 4)[:num_samples]
    
    elif modulation == '16QAM':
        symbols = np.random.randint(0, 16, num_samples // 4)
        i = 2 * ((symbols // 4) % 4) - 3
        q = 2 * (symbols % 4) - 3
        signal = (i + 1j * q) * 0.4
        signal = np.repeat(signal, 4)[:num_samples]
    
    elif modulation == '64QAM':
        symbols = np.random.randint(0, 64, num_samples // 4)
        i = 2 * ((symbols // 8) % 8) - 7
        q = 2 * (symbols % 8) - 7
        signal = (i + 1j * q) * 0.4
        signal = np.repeat(signal, 4)[:num_samples]
    
    elif modulation == 'GMSK':
        num_bits = num_samples // 8 + 2
        bits = np.random.randint(0, 2, num_bits)
        data = 2 * bits - 1
        t = np.linspace(-3, 3, 17)
        gaussian = np.exp(-(np.pi**2 * 0.3**2 * t**2) / np.log(2))
        gaussian /= np.sum(gaussian)
        shaped_data = np.zeros(num_bits * 8)
        shaped_data[::8] = data
        filtered_data = np.convolve(shaped_data, gaussian, mode='same')
        phase = np.cumsum(filtered_data) * (np.pi / 2)
        signal = np.exp(1j * phase)[:num_samples]
    
    else:
        signal = np.exp(1j * np.random.randn(num_samples))
    
    # 添加噪声
    snr_db = np.random.uniform(-10, 20)
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    
    return (signal + noise).astype(np.complex64), snr_db


def generate_radar_signal(aircraft_type, individual_idx, num_samples=500):
    """生成 Radar LFM 信号（简化版本）"""
    # 信号参数
    T = 5e-6  # 脉冲宽度
    fs = 100e6  # 采样率
    pulse_samples = int(fs * T)
    
    # 带宽根据飞机类型
    if aircraft_type == 'P-8A':
        B = 42e6
    elif aircraft_type == 'P-3C':
        B = 30e6
    else:  # E-2D
        B = 20e6
    
    # 生成 LFM 信号
    t_pulse = np.linspace(-T/2, T/2, pulse_samples)
    chirp_slope = B / T
    signal = np.exp(1j * np.pi * chirp_slope * t_pulse**2)
    
    # 添加相位噪声（简化版本）
    phase_noise = 0.1 * np.random.randn(pulse_samples)
    signal = signal * np.exp(1j * phase_noise)
    
    # 添加高斯噪声
    snr_db = np.random.uniform(0, 30)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(pulse_samples) + 1j * np.random.randn(pulse_samples))
    signal = signal + noise
    
    # 归一化
    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))
    
    return signal.astype(np.complex64), snr_db


def main():
    parser = argparse.ArgumentParser(description='端侧设备 - 实时生成信号或加载本地数据')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['link11', 'rml2016', 'radar'],
                       help='数据集类型')
    parser.add_argument('--data_source', type=str, default='generate', 
                       choices=['generate', 'local'],
                       help='数据来源：generate (实时生成) 或 local (本地文件)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='本地数据文件/文件夹路径（data_source=local 时必需）')
    parser.add_argument('--mode', type=str, default='edge', choices=['edge', 'cloud'],
                       help='运行模式：edge (发送到边侧) 或 cloud (直接发送到云侧)')
    parser.add_argument('--edge_host', type=str, default='localhost', help='边侧主机地址 (edge 模式)')
    parser.add_argument('--cloud_host', type=str, default='localhost', help='云侧主机地址 (cloud 模式)')
    parser.add_argument('--zmq_data_port', type=int, default=7777, help='ZeroMQ 数据端口（edge 模式：边侧数据端口，默认：7777）')
    parser.add_argument('--zmq_response_port', type=int, default=7778, help='ZeroMQ 响应端口（edge 模式：边侧响应端口，默认：7778）')
    parser.add_argument('--zmq_push_port', type=int, default=7777, help='ZeroMQ PUSH 端口（cloud 模式：云侧 PUSH 端口，默认：7777）')
    parser.add_argument('--zmq_pull_port', type=int, default=5555, help='ZeroMQ PULL 端口（cloud 模式：云侧 PULL 端口，默认：5555）')
    parser.add_argument('--batch_size', type=int, default=32, help='每批次样本数')
    parser.add_argument('--interval', type=float, default=0.0, help='固定发送间隔（秒），0表示不使用固定间隔')
    parser.add_argument('--rate_limit', type=float, default=100.0, help='流量限制（MB/s），0表示不限速')
    parser.add_argument('--total_samples', type=int, default=None, 
                       help='总样本数限制（默认：无限制，持续生成）')
    
    args = parser.parse_args()
    # args.batch_size = 32
    # 验证参数
    if args.data_source == 'local' and args.data_path is None:
        parser.error("--data_source=local 时必须指定 --data_path")
    
    print(f"\n{'='*70}")
    print(f"[端侧设备] 启动")
    print(f"{'='*70}")
    print(f"数据集类型: {args.dataset_type}")
    print(f"数据来源: {args.data_source}")
    if args.data_source == 'local':
        print(f"数据路径: {args.data_path}")
    print(f"运行模式: {args.mode}")
    if args.mode == 'edge':
        print(f"边侧地址: {args.edge_host}")
        print(f"数据端口: {args.zmq_data_port}")
        print(f"响应端口: {args.zmq_response_port}")
    else:  # cloud mode
        print(f"云侧地址: {args.cloud_host}")
        print(f"ZeroMQ PULL 端口: {args.zmq_pull_port}")
        print(f"ZeroMQ PUSH 端口: {args.zmq_push_port}")
    print(f"批次大小: {args.batch_size}")
    print(f"发送间隔: {args.interval}s")
    if args.total_samples:
        print(f"总样本数: {args.total_samples}")
    else:
        print(f"总样本数: 无限制（持续生成）")
    print(f"{'='*70}\n")
    
    # 根据模式选择不同的处理方式
    if args.mode == 'edge':
        # 发送到边侧
        
        if args.data_source == 'generate':
            # 模式1：实时生成信号
            if args.dataset_type == 'link11':
                device = Link11DeviceSimulator(args.edge_host, args.zmq_data_port, args.zmq_response_port, args.rate_limit)
            elif args.dataset_type == 'rml2016':
                device = RML2016DeviceSimulator(args.edge_host, args.zmq_data_port, args.zmq_response_port, args.rate_limit)
            elif args.dataset_type == 'radar':
                device = RadarDeviceSimulator(args.edge_host, args.zmq_data_port, args.zmq_response_port, args.rate_limit)
            
            try:
                device.connect()
                device.generate_and_send(args.batch_size, args.interval, args.total_samples)
            except Exception as e:
                print(f"[端侧] 错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                device.close()
        
        else:  # data_source == 'local'
            # 模式2：从本地文件加载数据
            loader = LocalDataLoader(args.data_path, args.dataset_type, args.edge_host, args.zmq_data_port, args.rate_limit)
            loader.edge_response_port = args.zmq_response_port  # 设置响应端口
            
            try:
                loader.connect()
                loader.load_and_send(args.batch_size, args.interval, args.total_samples)
            except Exception as e:
                print(f"[端侧] 错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                loader.close()
    
    else:  # cloud mode
        # 直接发送到云侧进行推理
        send_to_cloud_directly(args)


if __name__ == '__main__':
    main()

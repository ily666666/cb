"""
数据源模块

提供统一的数据源接口，支持从端侧设备实时接收数据或从本地文件加载数据。
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any
import torch
import zmq
import pickle
import time
import numpy as np


class DataSource(ABC):
    """
    数据源抽象基类
    
    定义所有数据源必须实现的统一接口，使推理逻辑与数据来源解耦。
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        返回数据迭代器
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: (signals, labels) 元组
                - signals: 信号张量，shape (batch_size, 2, signal_length)
                - labels: 标签张量，shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据源统计信息
        
        Returns:
            Dict[str, Any]: 包含样本数、批次数等统计信息的字典
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        关闭数据源，释放资源
        
        应在数据处理完成后调用，确保正确清理资源（如网络连接、文件句柄等）。
        """
        pass



class DeviceDataSource(DataSource):
    """
    从端侧设备接收数据的数据源实现
    
    使用 ZeroMQ PULL socket 从端侧设备实时接收批次数据。
    支持超时处理、错误恢复和统计信息跟踪。
    """
    
    def __init__(self, port: int = 7777, response_port: int = 7778, timeout: int = 30000, batch_size: int = 32, max_timeouts: int = None, bind_address: str = "127.0.0.1"):
        """
        初始化设备数据源
        
        Args:
            port: ZeroMQ 数据接收端口，默认 7777
            response_port: ZeroMQ 响应发送端口，默认 7778
            timeout: 接收超时时间（毫秒），默认 30000 (30秒)
            batch_size: 期望的批次大小，默认 32
            max_timeouts: 最大超时次数，超过后停止接收。None 表示无限制
            bind_address: 绑定地址，默认 "127.0.0.1"（本地回环，不需要管理员权限）。使用 "*" 绑定所有接口（需要管理员权限）
        """
        self.port = port
        self.response_port = response_port
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_timeouts = max_timeouts
        self.bind_address = bind_address
        
        # 统计信息
        self.total_batches = 0
        self.total_samples = 0
        self.timeout_count = 0
        self.start_time = None
        
        # ZeroMQ 设置
        self.context = zmq.Context()
        
        # PULL socket：接收数据
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.pull_socket.setsockopt(zmq.LINGER, 0)
        self.pull_socket.setsockopt(zmq.RCVHWM, 10000)
        self.pull_socket.setsockopt(zmq.RCVBUF, 64 * 1024 * 1024)
        
        # PUSH socket：发送完成标志
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.setsockopt(zmq.SNDHWM, 1000)
        self.push_socket.setsockopt(zmq.SNDBUF, 16 * 1024 * 1024)
        self.push_socket.setsockopt(zmq.LINGER, 0)
        
        # 优化：MessagePack 支持（方案1）
        try:
            import msgpack
            import msgpack_numpy as m
            m.patch()
            self.use_msgpack = True
            print("[设备数据源] 使用 MessagePack 反序列化（优化模式）")
        except ImportError:
            self.use_msgpack = False
            print("[设备数据源] MessagePack 未安装，使用 Pickle 反序列化")
            print("[设备数据源] 提示：安装 msgpack-numpy 可提升2-3倍传输速度")
            print("[设备数据源]   pip install msgpack-numpy")
        
        # 绑定端口
        data_address = f"tcp://{bind_address}:{port}"
        response_address = f"tcp://{bind_address}:{response_port}"
        
        try:
            self.pull_socket.bind(data_address)
            self.push_socket.bind(response_address)
            print(f"[设备数据源] 已绑定:")
            print(f"  数据端口: {data_address}")
            print(f"  响应端口: {response_address}")
            print(f"[设备数据源] 等待端侧连接...")
        except zmq.ZMQError as e:
            print(f"[设备数据源] 绑定失败: {e}")
            print(f"[设备数据源] 提示: 如果是权限错误，请检查：")
            print(f"  1. 端口 {port} 或 {response_port} 是否被占用")
            print(f"  2. 防火墙是否阻止了端口绑定")
            print(f"  3. 尝试以管理员身份运行")
            self.close()
            raise
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        接收并迭代数据批次
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: (signals, labels) 元组
        """
        self.start_time = time.time()
        
        try:
            while True:
                try:
                    # 接收数据
                    data = self.pull_socket.recv()
                    
                    # 反序列化数据（MessagePack 或 Pickle）
                    try:
                        if self.use_msgpack:
                            import msgpack
                            batch_data = msgpack.unpackb(data, raw=False)
                        else:
                            batch_data = pickle.loads(data)
                    except (pickle.UnpicklingError, EOFError, Exception) as e:
                        print(f"[设备数据源] 反序列化错误: {e}，跳过该批次")
                        continue
                    
                    # 提取信号和标签
                    try:
                        # 检查是否是结束标志
                        if batch_data.get('type') == 'end_transmission':
                            total_samples = batch_data.get('total_samples', 0)
                            total_batches = batch_data.get('total_batches', 0)
                            print(f"[设备数据源] 收到结束标志: 端侧发送了 {total_samples} 样本, {total_batches} 批次")
                            print(f"[设备数据源] 停止接收数据")
                            break
                        
                        signals = batch_data['signals']
                        labels = batch_data['labels']
                        
                        # 转换为 torch.Tensor
                        if isinstance(signals, np.ndarray):
                            # MessagePack 反序列化的数组是只读的，需要复制
                            if not signals.flags.writeable:
                                signals = signals.copy()
                            # 如果是复数数组，需要特殊处理
                            if np.iscomplexobj(signals):
                                signals = torch.from_numpy(signals)
                            else:
                                signals = torch.from_numpy(signals)
                        elif not isinstance(signals, torch.Tensor):
                            signals = torch.tensor(signals)
                        
                        if isinstance(labels, np.ndarray):
                            if not labels.flags.writeable:
                                labels = labels.copy()
                            labels = torch.from_numpy(labels)
                        elif not isinstance(labels, torch.Tensor):
                            labels = torch.tensor(labels)
                        
                    except (KeyError, TypeError) as e:
                        print(f"[设备数据源] 数据格式错误: {e}，跳过该批次")
                        continue
                    
                    # 更新统计
                    self.total_batches += 1
                    self.total_samples += signals.shape[0]
                    
                    # 定期打印统计
                    if self.total_batches % 10 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.total_samples / elapsed if elapsed > 0 else 0
                        print(f"[设备数据源] 已接收 {self.total_batches} 批次, "
                              f"{self.total_samples} 样本 ({rate:.1f} 样本/秒)")
                    
                    yield signals, labels
                    
                except zmq.Again:
                    # 接收超时
                    self.timeout_count += 1
                    print(f"[设备数据源] 接收超时 ({self.timeout}ms)，等待数据... (超时次数: {self.timeout_count})")
                    
                    # 如果设置了最大超时次数，检查是否超过
                    if self.max_timeouts is not None and self.timeout_count >= self.max_timeouts:
                        print(f"[设备数据源] 达到最大超时次数 ({self.max_timeouts})，停止接收")
                        break
                    
                    continue
                    
                except zmq.ZMQError as e:
                    # ZeroMQ 连接错误
                    print(f"[设备数据源] ZeroMQ 错误: {e}")
                    break
                    
                except Exception as e:
                    # 其他未预期的错误
                    print(f"[设备数据源] 接收错误: {e}")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n[设备数据源] 用户中断")
    
    def send_inference_complete(self) -> None:
        """
        发送推理完成标志给端侧
        
        应在边侧推理完成后调用此方法
        """
        try:
            print(f"[设备数据源] 发送推理完成标志给端侧...")
            complete_signal = {
                'type': 'inference_complete',
                'timestamp': time.time()
            }
            
            if self.use_msgpack:
                import msgpack
                self.push_socket.send(msgpack.packb(complete_signal, use_bin_type=True))
            else:
                self.push_socket.send(pickle.dumps(complete_signal))
            
            print(f"[设备数据源] 推理完成标志已发送")
        except Exception as e:
            print(f"[设备数据源] 发送推理完成标志失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 包含批次数、样本数、接收速率等统计信息
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'total_batches': self.total_batches,
            'total_samples': self.total_samples,
            'elapsed_time': elapsed,
            'samples_per_second': self.total_samples / elapsed if elapsed > 0 else 0,
            'port': self.port,
            'response_port': self.response_port,
            'timeout': self.timeout
        }
    
    def close(self) -> None:
        """
        关闭连接，释放资源
        """
        try:
            self.pull_socket.close()
            self.push_socket.close()
            self.context.term()
            print("[设备数据源] 连接已关闭")
        except Exception as e:
            print(f"[设备数据源] 关闭时出错: {e}")


class DataSourceFactory:
    """
    数据源工厂类
    
    根据配置参数创建相应的数据源实例。
    支持创建 DeviceDataSource 和 LocalDataSource。
    """
    
    @staticmethod
    def create(args) -> DataSource:
        """
        根据参数创建数据源
        
        Args:
            args: 命令行参数对象，应包含以下属性：
                - data_source: 数据源类型 ('device' 或 'local')
                - device_port: 设备端口（device 模式）
                - device_timeout: 设备超时时间（device 模式）
                - data_path: 数据路径（local 模式）
                - dataset_type: 数据集类型（local 模式）
                - batch_size: 批次大小
        
        Returns:
            DataSource: 数据源实例
        
        Raises:
            ValueError: 如果数据源类型无效
        
        Examples:
            >>> # 创建设备数据源
            >>> args.data_source = 'device'
            >>> args.device_port = 5555
            >>> args.device_timeout = 30000
            >>> args.batch_size = 32
            >>> data_source = DataSourceFactory.create(args)
            
            >>> # 创建本地数据源
            >>> args.data_source = 'local'
            >>> args.data_path = '/path/to/data.pkl'
            >>> args.dataset_type = 'rml2016'
            >>> args.batch_size = 32
            >>> data_source = DataSourceFactory.create(args)
        """
        # 获取数据源类型，默认为 'local'
        data_source_type = getattr(args, 'data_source', 'local')
        
        # 确保 data_source_type 是字符串类型
        if not isinstance(data_source_type, str):
            data_source_type = 'local'
        
        if data_source_type == 'device':
            # 创建设备数据源
            # 使用 try-except 来处理属性获取，避免 Mock 对象问题
            try:
                port = args.device_port
                # 确保是整数类型
                if not isinstance(port, int):
                    port = 7777
            except AttributeError:
                port = 7777
            
            try:
                response_port = args.device_response_port
                # 确保是整数类型
                if not isinstance(response_port, int):
                    response_port = 7778
            except AttributeError:
                response_port = 7778
            
            try:
                timeout = args.device_timeout
                # 确保是整数类型
                if not isinstance(timeout, int):
                    timeout = 30000
            except AttributeError:
                timeout = 30000
            
            try:
                batch_size = args.batch_size
                # 确保是整数类型
                if not isinstance(batch_size, int):
                    batch_size = 32
            except AttributeError:
                batch_size = 32
            
            return DeviceDataSource(
                port=port,
                response_port=response_port,
                timeout=timeout,
                batch_size=batch_size
            )
        
        elif data_source_type == 'local':
            # 创建本地数据源
            try:
                data_path = args.data_path
                # 确保是字符串类型
                if not isinstance(data_path, str):
                    data_path = None
            except AttributeError:
                data_path = None
            
            if data_path is None:
                raise ValueError("本地数据源需要指定 data_path 参数")
            
            try:
                dataset_type = args.dataset_type
                if not isinstance(dataset_type, str):
                    dataset_type = 'rml2016'
            except AttributeError:
                dataset_type = 'rml2016'
            
            try:
                batch_size = args.batch_size
                if not isinstance(batch_size, int):
                    batch_size = 32
            except AttributeError:
                batch_size = 32
            
            return LocalDataSource(
                data_path=data_path,
                dataset_type=dataset_type,
                batch_size=batch_size
            )
        
        else:
            # 无效的数据源类型
            raise ValueError(
                f"未知的数据源类型: {data_source_type}. "
                f"支持的类型: 'device', 'local'"
            )


class LocalDataSource(DataSource):
    """
    从本地文件加载数据的数据源实现
    
    支持两种模式：
    1. 单文件模式：一次性加载完整数据集
    2. 文件夹模式：逐批次加载文件
    """
    
    def __init__(self, data_path: str, dataset_type: str, batch_size: int = 32):
        """
        初始化本地数据源
        
        Args:
            data_path: 数据文件或文件夹路径
            dataset_type: 数据集类型 ('rml2016', 'link11', 'radar')
            batch_size: 批次大小，默认 32
            
        Raises:
            FileNotFoundError: 如果数据路径不存在
            ValueError: 如果文件夹中没有找到支持的文件格式
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        
        # 统计信息
        self.total_samples = 0
        self.is_folder = False
        self.file_format = None
        
        # 数据加载器
        self.data_loader = None
        
        # 子任务 3.1: 实现路径类型检测
        self._validate_and_detect_path()
        
        # 根据路径类型设置数据加载器
        if self.is_folder:
            # 子任务 3.3: 实现文件夹批次加载模式
            self._setup_batch_loader()
        else:
            # 子任务 3.2: 实现单文件加载模式
            self._setup_file_loader()
    
    def _validate_and_detect_path(self) -> None:
        """
        子任务 3.1: 验证路径并检测类型
        
        检查路径是否存在，判断是文件还是文件夹
        
        Raises:
            FileNotFoundError: 如果路径不存在
        """
        import os
        
        # 检查路径是否存在
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
        
        # 判断是文件还是文件夹
        self.is_folder = os.path.isdir(self.data_path)
        
        if self.is_folder:
            print(f"[本地数据源] 检测到文件夹: {self.data_path}")
        else:
            print(f"[本地数据源] 检测到文件: {self.data_path}")
    
    def _setup_file_loader(self) -> None:
        """
        子任务 3.2: 设置单文件加载器
        
        加载 pickle 或 MATLAB 文件，使用现有的 LazyDataset，创建 DataLoader
        """
        import pickle
        import os
        from torch.utils.data import DataLoader
        
        print(f"[本地数据源] 加载数据文件: {self.data_path}")
        
        try:
            # 检测文件扩展名
            _, ext = os.path.splitext(self.data_path)
            ext = ext.lower()
            
            if ext == '.pkl':
                # 加载 pickle 文件
                with open(self.data_path, 'rb') as f:
                    raw_data = pickle.load(f)
                self.file_format = 'pkl'
                
            elif ext == '.mat':
                # 加载 MATLAB 文件
                # 尝试使用 h5py (MATLAB v7.3+) 或 scipy.io.loadmat (旧版本)
                try:
                    import h5py
                    with h5py.File(self.data_path, 'r') as mat:
                        # 将 h5py 对象转换为字典
                        mat_data = {key: np.array(mat[key]) for key in mat.keys() if not key.startswith('__')}
                    print(f"[本地数据源] 使用 h5py 加载 MATLAB v7.3+ 文件")
                except (OSError, ImportError):
                    # 如果 h5py 失败，尝试 scipy.io.loadmat
                    import scipy.io as scio
                    mat_data = scio.loadmat(self.data_path)
                    print(f"[本地数据源] 使用 scipy.io 加载 MATLAB 文件")
                
                # 转换 MATLAB 数据格式为 pickle 格式
                raw_data = self._convert_mat_to_dict(mat_data)
                self.file_format = 'mat'
                
            else:
                raise ValueError(f"不支持的文件格式: {ext}. 支持的格式: .pkl, .mat")
            
            # 子任务 3.4: 计算总样本数
            self.total_samples = sum(len(signal_array) for signal_array in raw_data.values())
            print(f"[本地数据源] 数据加载完成: {self.total_samples} 样本")
            
            # 使用现有的 LazyDataset
            from run.edge.run_edge_collaborative import LazyDataset
            dataset = LazyDataset(raw_data, self.dataset_type)
            
            # 创建 DataLoader
            self.data_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=False, 
                num_workers=0, 
                pin_memory=True
            )
            
        except Exception as e:
            raise ValueError(f"加载数据文件失败: {e}")
    
    def _convert_mat_to_dict(self, mat_data: Dict) -> Dict:
        """
        将 MATLAB 数据格式转换为 pickle 字典格式
        
        Args:
            mat_data: scipy.io.loadmat 或 h5py 返回的数据
            
        Returns:
            Dict: 转换后的数据字典，格式与 pickle 文件一致
        """
        # 根据数据集类型进行转换
        if self.dataset_type == 'radar':
            # Radar 数据集格式: 'X' 和 'Y'
            # X: (2, 500, num_samples) - IQ signals
            # Y: (1, num_samples) or (num_samples,) - Class labels (1-7)
            
            if 'X' in mat_data and 'Y' in mat_data:
                X = np.array(mat_data['X'])  # (2, 500, num_samples)
                Y = np.array(mat_data['Y']).flatten()  # (num_samples,)
                
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
                
                print(f"[本地数据源] Radar 数据转换完成: {num_samples} 样本, {len(unique_labels)} 类别")
                return raw_data
            else:
                raise ValueError(f"Radar 数据集需要 'X' 和 'Y' 字段，但只找到: {list(mat_data.keys())}")
        
        # 移除 MATLAB 元数据键
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        # 如果无法识别格式，尝试通用转换
        # 假设数据已经按类别组织
        raw_data = {}
        for key in data_keys:
            try:
                label = int(key)
                raw_data[label] = mat_data[key]
            except ValueError:
                # 如果键不是数字，跳过
                continue
        
        if not raw_data:
            raise ValueError(f"无法从 MATLAB 文件中提取数据。可用的键: {data_keys}")
        
        return raw_data

    
    def _setup_batch_loader(self) -> None:
        """
        子任务 3.3: 设置文件夹批次加载器
        
        查找 .pkl 和 .mat 文件，使用现有的 BatchLoadingDataset，创建 DataLoader
        
        Raises:
            ValueError: 如果文件夹中没有找到支持的文件格式
        """
        import os
        import glob
        from torch.utils.data import DataLoader
        
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
        
        print(f"[本地数据源] 找到 {len(self.batch_files)} 个批次文件 (.{self.file_format})")
        
        # 使用现有的 BatchLoadingDataset
        from run.edge.run_edge_collaborative import BatchLoadingDataset
        dataset = BatchLoadingDataset(self.batch_files, self.dataset_type, self.file_format)
        
        # 创建 DataLoader
        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=0, 
            pin_memory=True
        )
        
        # 子任务 3.4: 对于文件夹模式，我们无法预先知道总样本数
        # 但可以记录文件数量
        self.total_samples = 0  # 将在迭代过程中更新
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        迭代数据批次
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: (signals, labels) 元组
        """
        sample_count = 0
        
        for batch in self.data_loader:
            # 处理不同的 batch 格式
            if len(batch) == 3:
                signals, labels, _ = batch
            elif len(batch) == 2:
                signals, labels = batch
            else:
                continue
            
            sample_count += signals.shape[0]
            yield signals, labels
        
        # 更新总样本数（对于文件夹模式）
        if self.is_folder:
            self.total_samples = sample_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        子任务 3.4: 获取统计信息
        
        Returns:
            Dict[str, Any]: 包含总样本数、数据路径和类型等统计信息
        """
        stats = {
            'total_samples': self.total_samples,
            'is_folder': self.is_folder,
            'data_path': self.data_path,
            'dataset_type': self.dataset_type,
            'file_format': self.file_format
        }
        
        if self.is_folder:
            stats['num_batch_files'] = len(self.batch_files)
        
        return stats
    
    def close(self) -> None:
        """
        关闭数据源
        
        DataLoader 会自动清理资源，无需额外操作
        """
        # DataLoader 会自动清理
        pass

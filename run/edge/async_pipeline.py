#!/usr/bin/env python3
"""
异步流水线核心数据结构和工具类
用于边侧协同推理的异步流水线架构
"""
import threading
import time
import queue
import socket
import pickle
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch


class GlobalTimeTracker:
    """全局时间追踪器 - 记录整个流水线的实际时间消耗"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # 阶段时间记录：记录每个阶段实际工作的时间段
        self.stage_intervals = {
            'device_to_edge_transfer': [],      # 端侧→边侧传输时间段
            'edge_inference': [],               # 边侧推理时间段
            'edge_to_cloud_transfer': [],       # 边侧→云侧传输时间段
            'cloud_inference': [],              # 云侧推理时间段
            'cloud_to_edge_transfer': [],       # 云侧→边侧传输时间段
        }
        
        # 全局开始和结束时间
        self.global_start_time = None
        self.global_end_time = None
    
    def start_global(self):
        """开始全局计时"""
        self.global_start_time = time.time()
    
    def end_global(self):
        """结束全局计时"""
        self.global_end_time = time.time()
    
    def record_interval(self, stage: str, start_time: float, end_time: float):
        """记录某个阶段的时间段"""
        with self.lock:
            self.stage_intervals[stage].append((start_time, end_time))
    
    def get_total_time_by_stage(self):
        """计算每个阶段的总时间（合并重叠的时间段）"""
        result = {}
        
        with self.lock:
            for stage, intervals in self.stage_intervals.items():
                if not intervals:
                    result[stage] = 0.0
                    continue
                
                # 合并重叠的时间段
                sorted_intervals = sorted(intervals, key=lambda x: x[0])
                merged = []
                current_start, current_end = sorted_intervals[0]
                
                for start, end in sorted_intervals[1:]:
                    if start <= current_end:
                        # 重叠，合并
                        current_end = max(current_end, end)
                    else:
                        # 不重叠，保存当前段，开始新段
                        merged.append((current_start, current_end))
                        current_start, current_end = start, end
                
                merged.append((current_start, current_end))
                
                # 计算总时间
                total_time = sum(end - start for start, end in merged)
                result[stage] = total_time
        
        return result
    
    def get_summary(self):
        """获取时间统计摘要"""
        if self.global_start_time is None or self.global_end_time is None:
            return None
        
        total_elapsed = self.global_end_time - self.global_start_time
        stage_times = self.get_total_time_by_stage()
        
        return {
            'total_elapsed_sec': total_elapsed,
            'stage_times_sec': stage_times,
            'stage_percentages': {
                stage: (time_sec / total_elapsed * 100) if total_elapsed > 0 else 0
                for stage, time_sec in stage_times.items()
            }
        }


@dataclass
class UploadItem:
    """上传队列项 - 待发送到云侧的样本"""
    batch_id: int                    # 批次ID
    sample_indices: List[int]        # 样本在批次中的索引
    inputs: torch.Tensor             # 输入数据
    targets: torch.Tensor            # 真实标签
    timestamp: float                 # 创建时间戳
    upload_start_time: float = 0.0   # 上传开始时间（用于计算云侧延迟）
    
    def __post_init__(self):
        """验证数据完整性"""
        if len(self.sample_indices) != self.inputs.shape[0]:
            raise ValueError(f"样本索引数量({len(self.sample_indices)})与输入数据批次大小({self.inputs.shape[0]})不匹配")
        if len(self.sample_indices) != self.targets.shape[0]:
            raise ValueError(f"样本索引数量({len(self.sample_indices)})与标签数量({self.targets.shape[0]})不匹配")


@dataclass
class BatchResult:
    """批次结果 - 存储边侧和云侧的推理结果"""
    batch_id: int                    # 批次ID
    edge_results: Dict[int, int] = field(default_factory=dict)     # {sample_idx: prediction}
    cloud_results: Dict[int, int] = field(default_factory=dict)    # {sample_idx: prediction}
    targets: Dict[int, int] = field(default_factory=dict)          # {sample_idx: label}
    edge_complete: bool = False      # 边侧结果是否完整
    cloud_complete: bool = False     # 云侧结果是否完整
    timestamp: float = field(default_factory=time.time)            # 创建时间戳
    total_samples: int = 0           # 批次总样本数
    edge_latency_ms: float = 0.0     # 边侧推理延迟（毫秒）
    cloud_latency_ms: float = 0.0    # 云侧推理延迟（毫秒）
    
    def is_complete(self, timeout_seconds: float = 30.0) -> bool:
        """
        检查批次是否完成（或超时）
        
        Args:
            timeout_seconds: 超时时间（秒）
            
        Returns:
            True 如果批次完成或超时
        """
        # 检查是否所有结果都已就绪
        # 批次完成的条件：有边侧结果，且云侧结果已标记完成（即使云侧结果为空）
        total_results = len(self.edge_results) + len(self.cloud_results)
        
        # 如果有总样本数，检查是否所有样本都有结果
        if self.total_samples > 0 and total_results >= self.total_samples:
            return True
        
        # 如果云侧标记为完成，则批次完成
        if self.cloud_complete:
            return True
        
        # 检查是否超时（超时后使用边侧结果作为fallback）
        if time.time() - self.timestamp > timeout_seconds:
            return True
        
        return False
    
    def get_merged_results(self) -> Dict[int, int]:
        """
        合并边侧和云侧结果
        
        Returns:
            合并后的预测结果 {sample_idx: prediction}
        """
        merged = {}
        
        # 先添加边侧结果
        merged.update(self.edge_results)
        
        # 云侧结果覆盖边侧结果（云侧优先级更高）
        merged.update(self.cloud_results)
        
        return merged
    
    def calculate_accuracy(self) -> float:
        """
        计算批次准确率
        
        Returns:
            准确率（0.0-1.0）
        """
        merged = self.get_merged_results()
        
        if not merged or not self.targets:
            return 0.0
        
        correct = sum(1 for idx, pred in merged.items() 
                     if idx in self.targets and pred == self.targets[idx])
        
        total = len(self.targets)
        
        return correct / total if total > 0 else 0.0


class ResultCache:
    """线程安全的结果缓存"""
    
    def __init__(self):
        self.cache: Dict[int, BatchResult] = {}
        self.lock = threading.Lock()
    
    def add_edge_result(self, batch_id: int, sample_indices: List[int], 
                       predictions: torch.Tensor, targets: torch.Tensor,
                       total_samples: int, edge_latency_ms: float = 0.0):
        """
        添加边侧结果
        
        Args:
            batch_id: 批次ID
            sample_indices: 样本索引列表
            predictions: 预测结果张量
            targets: 真实标签张量
            total_samples: 批次总样本数
            edge_latency_ms: 边侧推理延迟（毫秒）
        """
        with self.lock:
            if batch_id not in self.cache:
                self.cache[batch_id] = BatchResult(batch_id=batch_id, total_samples=total_samples)
            
            batch = self.cache[batch_id]
            
            # 转换为Python标量
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().tolist()
            if torch.is_tensor(targets):
                targets = targets.cpu().tolist()
            
            # 确保是列表
            if not isinstance(predictions, list):
                predictions = [predictions]
            if not isinstance(targets, list):
                targets = [targets]
            
            # 添加结果
            for idx, pred, target in zip(sample_indices, predictions, targets):
                batch.edge_results[idx] = pred
                batch.targets[idx] = target
            
            # 更新总样本数和延迟
            batch.total_samples = total_samples
            batch.edge_latency_ms = edge_latency_ms
    
    def add_cloud_result(self, batch_id: int, sample_indices: List[int], 
                        predictions: torch.Tensor, cloud_latency_ms: float = 0.0):
        """
        添加云侧结果
        
        Args:
            batch_id: 批次ID
            sample_indices: 样本索引列表
            predictions: 预测结果张量
            cloud_latency_ms: 云侧推理延迟（毫秒）
        """
        with self.lock:
            if batch_id not in self.cache:
                self.cache[batch_id] = BatchResult(batch_id=batch_id)
            
            batch = self.cache[batch_id]
            
            # 转换为Python标量
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().tolist()
            
            # 确保是列表
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            # 添加结果
            for idx, pred in zip(sample_indices, predictions):
                batch.cloud_results[idx] = pred
            
            batch.cloud_complete = True
            batch.cloud_latency_ms = cloud_latency_ms
    
    def is_batch_complete(self, batch_id: int, timeout_seconds: float = 30.0) -> bool:
        """
        检查批次是否完成（或超时）
        
        Args:
            batch_id: 批次ID
            timeout_seconds: 超时时间（秒）
            
        Returns:
            True 如果批次完成或超时
        """
        with self.lock:
            if batch_id not in self.cache:
                return False
            
            return self.cache[batch_id].is_complete(timeout_seconds)
    
    def get_completed_batches(self, timeout_seconds: float = 30.0) -> List[int]:
        """
        获取所有已完成的批次ID
        
        Args:
            timeout_seconds: 超时时间（秒）
            
        Returns:
            已完成的批次ID列表
        """
        with self.lock:
            completed = []
            for batch_id, batch in self.cache.items():
                if batch.is_complete(timeout_seconds):
                    completed.append(batch_id)
            return completed
    
    def get_batch(self, batch_id: int) -> Optional[BatchResult]:
        """
        获取批次结果
        
        Args:
            batch_id: 批次ID
            
        Returns:
            BatchResult 或 None
        """
        with self.lock:
            return self.cache.get(batch_id)
    
    def remove_batch(self, batch_id: int):
        """
        移除批次（清理已处理的批次）
        
        Args:
            batch_id: 批次ID
        """
        with self.lock:
            if batch_id in self.cache:
                del self.cache[batch_id]
    
    def get_cache_size(self) -> int:
        """获取缓存中的批次数量"""
        with self.lock:
            return len(self.cache)


@dataclass
class PerformanceStats:
    """性能统计收集器"""
    total_batches: int = 0               # 总批次数
    total_samples: int = 0               # 总样本数
    edge_samples: int = 0                # 边侧处理样本数
    cloud_samples: int = 0               # 云侧处理样本数
    correct_predictions: int = 0         # 正确预测数
    
    # 延迟统计（毫秒）
    edge_latencies: List[float] = field(default_factory=list)      # 边侧推理延迟
    cloud_latencies: List[float] = field(default_factory=list)     # 云侧推理延迟
    end_to_end_latencies: List[float] = field(default_factory=list) # 端到端延迟
    
    # ========== 新增：详细时间分解统计 ==========
    edge_inference_times: List[float] = field(default_factory=list)     # T1: 边侧推理时间
    edge_decision_times: List[float] = field(default_factory=list)      # T2: 边侧决策时间
    edge_serialize_times: List[float] = field(default_factory=list)     # T3: 边侧序列化时间
    network_upload_times: List[float] = field(default_factory=list)     # T4: 网络上传时间
    cloud_receive_times: List[float] = field(default_factory=list)      # T5: 云侧接收时间
    cloud_queue_wait_times: List[float] = field(default_factory=list)   # T6: 云侧队列等待时间
    cloud_inference_times: List[float] = field(default_factory=list)    # T7: 云侧推理时间
    network_download_times: List[float] = field(default_factory=list)   # T8: 网络下载时间
    
    # 队列统计
    upload_queue_sizes: List[int] = field(default_factory=list)    # 上传队列长度快照
    inference_queue_sizes: List[int] = field(default_factory=list) # 推理队列长度快照（云侧）
    
    # 时间戳
    start_time: float = field(default_factory=time.time)           # 开始时间
    end_time: Optional[float] = None                               # 结束时间
    
    # 线程安全锁
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def update_batch(self, batch_result: BatchResult, edge_latency: float = 0.0, 
                    cloud_latency: float = 0.0, end_to_end_latency: float = 0.0):
        """
        更新批次统计信息
        
        Args:
            batch_result: 批次结果
            edge_latency: 边侧推理延迟（毫秒）
            cloud_latency: 云侧推理延迟（毫秒）
            end_to_end_latency: 端到端延迟（毫秒）
        """
        with self._lock:
            self.total_batches += 1
            
            # 计算样本数
            edge_count = len(batch_result.edge_results)
            cloud_count = len(batch_result.cloud_results)
            
            self.edge_samples += edge_count
            self.cloud_samples += cloud_count
            self.total_samples += batch_result.total_samples
            
            # 计算正确预测数
            merged = batch_result.get_merged_results()
            correct = sum(1 for idx, pred in merged.items() 
                         if idx in batch_result.targets and pred == batch_result.targets[idx])
            self.correct_predictions += correct
            
            # 记录延迟
            if edge_latency > 0:
                self.edge_latencies.append(edge_latency)
            if cloud_latency > 0:
                self.cloud_latencies.append(cloud_latency)
            if end_to_end_latency > 0:
                self.end_to_end_latencies.append(end_to_end_latency)
    
    def record_queue_size(self, upload_queue_size: int = 0, inference_queue_size: int = 0):
        """
        记录队列长度快照
        
        Args:
            upload_queue_size: 上传队列长度
            inference_queue_size: 推理队列长度
        """
        with self._lock:
            if upload_queue_size > 0:
                self.upload_queue_sizes.append(upload_queue_size)
            if inference_queue_size > 0:
                self.inference_queue_sizes.append(inference_queue_size)
    
    def finalize(self):
        """完成统计（记录结束时间）"""
        with self._lock:
            self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Returns:
            统计摘要字典
        """
        with self._lock:
            # 计算吞吐量
            elapsed = (self.end_time or time.time()) - self.start_time
            throughput = self.total_samples / elapsed if elapsed > 0 else 0.0
            
            # 计算准确率
            accuracy = self.correct_predictions / self.total_samples if self.total_samples > 0 else 0.0
            
            # 计算平均延迟
            avg_edge_latency = sum(self.edge_latencies) / len(self.edge_latencies) if self.edge_latencies else 0.0
            avg_cloud_latency = sum(self.cloud_latencies) / len(self.cloud_latencies) if self.cloud_latencies else 0.0
            avg_e2e_latency = sum(self.end_to_end_latencies) / len(self.end_to_end_latencies) if self.end_to_end_latencies else 0.0
            
            # 计算平均队列长度
            avg_upload_queue = sum(self.upload_queue_sizes) / len(self.upload_queue_sizes) if self.upload_queue_sizes else 0.0
            avg_inference_queue = sum(self.inference_queue_sizes) / len(self.inference_queue_sizes) if self.inference_queue_sizes else 0.0
            
            # ========== 新增：计算详细时间分解的平均值 ==========
            def calc_avg(times_list):
                return sum(times_list) / len(times_list) if times_list else 0.0
            
            time_breakdown = {
                'T1_edge_inference': calc_avg(self.edge_inference_times),
                'T2_edge_decision': calc_avg(self.edge_decision_times),
                'T3_edge_serialize': calc_avg(self.edge_serialize_times),
                'T4_network_upload': calc_avg(self.network_upload_times),
                'T5_cloud_receive': calc_avg(self.cloud_receive_times),
                'T6_cloud_queue_wait': calc_avg(self.cloud_queue_wait_times),
                'T7_cloud_inference': calc_avg(self.cloud_inference_times),
                'T8_network_download': calc_avg(self.network_download_times),
            }
            
            return {
                'total_batches': self.total_batches,
                'total_samples': self.total_samples,
                'edge_samples': self.edge_samples,
                'cloud_samples': self.cloud_samples,
                'correct_predictions': self.correct_predictions,
                'accuracy': accuracy,
                'throughput_samples_per_sec': throughput,
                'elapsed_time_sec': elapsed,
                'avg_edge_latency_ms': avg_edge_latency,
                'avg_cloud_latency_ms': avg_cloud_latency,
                'avg_end_to_end_latency_ms': avg_e2e_latency,
                'avg_upload_queue_size': avg_upload_queue,
                'avg_inference_queue_size': avg_inference_queue,
                'cloud_ratio': self.cloud_samples / self.total_samples if self.total_samples > 0 else 0.0,
                'time_breakdown': time_breakdown  # 新增：详细时间分解
            }


class ThreadSafeQueue:
    """线程安全的队列包装器（基于queue.Queue）"""
    
    def __init__(self, maxsize: int = 0):
        """
        初始化队列
        
        Args:
            maxsize: 队列最大大小（0表示无限制）
        """
        self._queue = queue.Queue(maxsize=maxsize)
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None):
        """
        放入队列
        
        Args:
            item: 要放入的项
            block: 是否阻塞
            timeout: 超时时间（秒）
        
        Raises:
            queue.Full: 如果队列满且不阻塞
        """
        self._queue.put(item, block=block, timeout=timeout)
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """
        从队列获取
        
        Args:
            block: 是否阻塞
            timeout: 超时时间（秒）
            
        Returns:
            队列中的项
            
        Raises:
            queue.Empty: 如果队列空且不阻塞
        """
        return self._queue.get(block=block, timeout=timeout)
    
    def task_done(self):
        """标记任务完成"""
        self._queue.task_done()
    
    def qsize(self) -> int:
        """获取队列大小（近似值）"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()
    
    def full(self) -> bool:
        """检查队列是否已满"""
        return self._queue.full()


class InferenceThread(threading.Thread):
    """
    边侧推理线程
    
    持续执行本地模型推理，不等待云侧响应。
    根据置信度阈值决策样本分流：
    - 高置信度样本直接存入ResultCache
    - 低置信度样本构造UploadItem放入upload_queue
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 data_loader: Any,
                 threshold: float,
                 upload_queue: ThreadSafeQueue,
                 result_cache: ResultCache,
                 device: torch.device,
                 stats_collector: Optional[PerformanceStats] = None,
                 stop_event: Optional[threading.Event] = None,
                 backpressure_threshold: float = 0.9,
                 time_tracker: Optional['GlobalTimeTracker'] = None):
        """
        初始化推理线程
        
        Args:
            model: 边侧学生模型
            data_loader: 数据加载器（可迭代对象）
            threshold: 置信度阈值（低于此值的样本发送到云侧）
            upload_queue: 上传队列（线程安全）
            result_cache: 结果缓存（需要锁保护）
            device: 计算设备（CPU/GPU）
            stats_collector: 性能统计收集器（可选）
            stop_event: 停止事件（用于优雅关闭）
            backpressure_threshold: 队列背压阈值（队列使用率超过此值时暂停推理）
        """
        super().__init__(name="InferenceThread")
        
        self.model = model
        self.data_loader = data_loader
        self.threshold = threshold
        self.upload_queue = upload_queue
        self.result_cache = result_cache
        self.device = device
        self.stats_collector = stats_collector
        
        # 停止事件（用于优雅关闭）
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        
        # 背压机制：暂停事件（当队列接近满时暂停推理）
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初始状态：不暂停
        self.backpressure_threshold = backpressure_threshold
        
        # 批次ID计数器（原子操作）
        self._batch_id_counter = 0
        self._batch_id_lock = threading.Lock()
        
        # 统计信息
        self.batches_processed = 0
        self.samples_processed = 0
        self.high_conf_samples = 0
        self.low_conf_samples = 0
        self.backpressure_events = 0  # 背压事件计数
        self.backpressure_total_time = 0.0  # 背压总时间（秒）
        
        # ========== 时间统计收集 ==========
        self.time_stats = {
            'inference_times': [],      # 推理时间
            'decision_times': [],       # 决策时间
            'cache_times': [],          # 缓存存储时间
            'enqueue_times': [],        # 入队时间
            'edge_total_times': []      # 边侧总时间
        }
        self.time_stats_lock = threading.Lock()
        
        # 全局时间追踪器
        self.time_tracker = time_tracker
        
        # 错误处理：错误标志
        self.error_flag = threading.Event()
        self.last_error = None
    
    def _get_next_batch_id(self) -> int:
        """
        获取下一个批次ID（线程安全）
        
        Returns:
            唯一的批次ID
        """
        with self._batch_id_lock:
            batch_id = self._batch_id_counter
            self._batch_id_counter += 1
            return batch_id
    
    def _check_backpressure(self):
        """
        检查队列背压状态
        
        如果上传队列使用率超过阈值，暂停推理（清除pause_event）
        如果队列使用率恢复正常，恢复推理（设置pause_event）
        """
        try:
            # 获取队列当前大小和最大大小
            current_size = self.upload_queue.qsize()
            max_size = self.upload_queue._queue.maxsize
            
            if max_size > 0:  # 只有当队列有大小限制时才检查背压
                usage_ratio = current_size / max_size
                
                if usage_ratio >= self.backpressure_threshold:
                    # 队列使用率超过阈值，触发背压（暂停推理）
                    if self.pause_event.is_set():
                        self.pause_event.clear()
                elif usage_ratio < self.backpressure_threshold * 0.7:
                    # 队列使用率降到阈值的70%以下，解除背压（恢复推理）
                    if not self.pause_event.is_set():
                        self.pause_event.set()
        except Exception as e:
            # 如果检查失败，默认不暂停
            if not self.pause_event.is_set():
                self.pause_event.set()
    
    def run(self):
        """
        推理线程主循环
        
        执行流程：
        1. 从data_loader迭代获取批次
        2. 执行本地模型推理
        3. 计算置信度和预测结果
        4. 根据阈值分流样本
        5. 高置信度样本存入ResultCache
        6. 低置信度样本放入upload_queue
        7. 检查队列背压，必要时暂停推理
        """
        print(f"[{self.name}] 推理线程启动")
        
        try:
            # 主推理循环
            for inputs, targets in self.data_loader:
                # 检查停止信号
                if self.stop_event.is_set():
                    print(f"[{self.name}] 收到停止信号，退出推理循环")
                    break
                
                # 检查队列背压：如果队列接近满，暂停推理
                self._check_backpressure()
                
                # 等待背压解除（如果被暂停）
                if not self.pause_event.is_set():
                    backpressure_start = time.time()
                    print(f"[{self.name}] 队列背压触发，暂停推理...")
                    self.backpressure_events += 1
                    
                    # 等待背压解除或停止信号
                    while not self.pause_event.is_set() and not self.stop_event.is_set():
                        time.sleep(0.1)
                        self._check_backpressure()  # 持续检查队列状态
                    
                    backpressure_time = time.time() - backpressure_start
                    self.backpressure_total_time += backpressure_time
                    print(f"[{self.name}] 背压解除，恢复推理 (暂停时间: {backpressure_time:.2f}秒)")
                
                # 分配唯一的Batch ID
                batch_id = self._get_next_batch_id()
                
                # ========== 时间记录：数据到达边侧 ==========
                data_arrival_time = time.time()
                
                # 移动数据到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.shape[0]
                
                # ========== T1: 边侧推理时间 ==========
                inference_start = time.time()
                
                # 执行本地推理
                with torch.no_grad():
                    outputs = self.model(inputs)
                    
                    # 处理复数输出
                    if torch.is_complex(outputs):
                        outputs = torch.abs(outputs)
                
                inference_end = time.time()
                inference_time = (inference_end - inference_start) * 1000  # 毫秒
                
                # 记录到全局时间追踪器
                if self.time_tracker:
                    self.time_tracker.record_interval('edge_inference', inference_start, inference_end)
                
                # ========== T2: 边侧决策时间 ==========
                decision_start = time.time()
                
                # 计算置信度和预测结果
                import torch.nn.functional as F
                probs = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                # 决策分流：根据置信度阈值分离高/低置信度样本
                high_conf_mask = confidences >= self.threshold
                low_conf_mask = ~high_conf_mask
                
                decision_end = time.time()
                decision_time = (decision_end - decision_start) * 1000  # 毫秒
                
                # 记录到 stats_collector
                if self.stats_collector:
                    self.stats_collector.edge_inference_times.append(inference_time)
                    self.stats_collector.edge_decision_times.append(decision_time)
                
                # 高置信度样本直接存入ResultCache
                if high_conf_mask.any():
                    cache_start = time.time()
                    
                    high_conf_indices = torch.where(high_conf_mask)[0].cpu().tolist()
                    high_conf_preds = predictions[high_conf_mask]
                    high_conf_targets = targets[high_conf_mask]
                    
                    self.result_cache.add_edge_result(
                        batch_id=batch_id,
                        sample_indices=high_conf_indices,
                        predictions=high_conf_preds,
                        targets=high_conf_targets,
                        total_samples=batch_size,
                        edge_latency_ms=inference_time  # 记录边侧推理延迟
                    )
                    
                    cache_end = time.time()
                    cache_time = (cache_end - cache_start) * 1000  # 毫秒
                    
                    self.high_conf_samples += len(high_conf_indices)
                else:
                    cache_time = 0.0
                
                # 低置信度样本构造UploadItem放入upload_queue
                if low_conf_mask.any():
                    enqueue_start = time.time()
                    
                    low_conf_indices = torch.where(low_conf_mask)[0].cpu().tolist()
                    low_conf_inputs = inputs[low_conf_mask]
                    low_conf_targets = targets[low_conf_mask]
                    
                    upload_item = UploadItem(
                        batch_id=batch_id,
                        sample_indices=low_conf_indices,
                        inputs=low_conf_inputs.cpu(),  # 移回CPU以节省GPU内存
                        targets=low_conf_targets.cpu(),
                        timestamp=time.time()
                    )
                    
                    # 放入上传队列（可能阻塞，实现背压机制）
                    self.upload_queue.put(upload_item, block=True)
                    
                    enqueue_end = time.time()
                    enqueue_time = (enqueue_end - enqueue_start) * 1000  # 毫秒
                    
                    self.low_conf_samples += len(low_conf_indices)
                else:
                    enqueue_time = 0.0
                
                # 更新统计信息
                self.batches_processed += 1
                self.samples_processed += batch_size
                
                # 记录队列状态
                if self.stats_collector:
                    self.stats_collector.record_queue_size(
                        upload_queue_size=self.upload_queue.qsize()
                    )
        
        except Exception as e:
            print(f"[{self.name}] 推理线程异常: {e}")
            import traceback
            traceback.print_exc()
            
            # 设置错误标志并记录错误
            self.error_flag.set()
            self.last_error = {
                'exception': e,
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
        
        finally:
            print(f"[{self.name}] 推理线程结束")
            print(f"[{self.name}] 总计处理: {self.batches_processed} 批次, "
                  f"{self.samples_processed} 样本")
            print(f"[{self.name}] 高置信度: {self.high_conf_samples} "
                  f"({100*self.high_conf_samples/self.samples_processed:.1f}%), "
                  f"低置信度: {self.low_conf_samples} "
                  f"({100*self.low_conf_samples/self.samples_processed:.1f}%)")
            
            # ========== 输出时间统计摘要 ==========
            self._print_time_stats()
    
    def stop(self):
        """
        停止推理线程（优雅关闭）
        """
        print(f"[{self.name}] 请求停止推理线程")
        self.stop_event.set()
    
    def _print_time_stats(self):
        """打印时间统计摘要 - 简化版，只显示关键业务指标"""
        import numpy as np
        
        with self.time_stats_lock:
            if not self.time_stats['inference_times']:
                return
            
            # 只显示边侧推理时间（最关键的指标）
            inference_times = np.array(self.time_stats['inference_times'])
            print(f"\n[{self.name}] 边侧推理: 平均 {np.mean(inference_times):.2f}ms (P95: {np.percentile(inference_times, 95):.2f}ms)")



class UploadThread(threading.Thread):
    """
    边侧上传线程
    
    异步上传低置信度样本到云侧。
    从upload_queue获取UploadItem，序列化并发送到云侧。
    处理连接失败和超时，使用边侧预测作为fallback。
    """
    
    def __init__(self,
                 upload_queue: ThreadSafeQueue,
                 network_edge: Any,
                 dataset_type: str,
                 num_classes: int,
                 result_cache: ResultCache,
                 stats_collector: Optional[PerformanceStats] = None,
                 stop_event: Optional[threading.Event] = None,
                 time_tracker: Optional['GlobalTimeTracker'] = None):
        """
        初始化上传线程
        
        Args:
            upload_queue: 上传队列（线程安全）
            network_edge: 网络通信对象（NetworkEdge实例）
            dataset_type: 数据集类型
            num_classes: 类别数
            result_cache: 结果缓存（用于存储fallback结果）
            stats_collector: 性能统计收集器（可选）
            stop_event: 停止事件（用于优雅关闭）
        """
        super().__init__(name="UploadThread")
        
        self.upload_queue = upload_queue
        self.network_edge = network_edge
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.result_cache = result_cache
        self.stats_collector = stats_collector
        
        # 停止事件（用于优雅关闭）
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        
        # 统计信息
        self.items_uploaded = 0
        self.samples_uploaded = 0
        self.connection_failures = 0
        self.timeouts = 0
        self.fallback_count = 0
        
        # ========== 时间统计收集 ==========
        self.time_stats = {
            'queue_wait_times': [],     # 队列等待时间
            'upload_times': []          # 上传时间（序列化+网络）
        }
        self.time_stats_lock = threading.Lock()
        
        # 全局时间追踪器
        self.time_tracker = time_tracker
        
        # 错误处理：指数退避重连
        self.retry_delay = 1.0  # 初始重试延迟（秒）
        self.max_retry_delay = 30.0  # 最大重试延迟（秒）
        self.current_retry_delay = self.retry_delay
        
        # 错误处理：错误标志
        self.error_flag = threading.Event()
        self.last_error = None
    
    def _log(self, level, message):
        """统一的日志输出格式"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.name}] {message}")
    
    def run(self):
        """
        上传线程主循环
        
        执行流程：
        1. 从upload_queue获取UploadItem（带超时）
        2. 序列化数据并发送到云侧
        3. 处理连接失败和超时（fallback到边侧预测）
        4. 记录上传时间和网络统计
        """
        self._log("INFO", "上传线程启动")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # 从队列获取待上传数据（带超时，避免阻塞）
                    upload_item = self.upload_queue.get(timeout=0.1)
                    
                    # ========== 时间记录：从队列取出 ==========
                    dequeue_time = time.time()
                    queue_wait_time = (dequeue_time - upload_item.timestamp) * 1000  # 毫秒
                    
                    # 记录上传开始时间（用于计算云侧延迟）
                    upload_item.upload_start_time = time.time()
                    
                    # ========== 时间记录：序列化开始 ==========
                    serialize_start = time.time()
                    
                    # 尝试上传到云侧
                    success = self._upload_to_cloud(upload_item)
                    
                    # ========== 时间记录：上传结束（包含序列化+网络传输） ==========
                    upload_end = time.time()
                    upload_time = (upload_end - serialize_start) * 1000  # 毫秒
                    
                    if success:
                        self.items_uploaded += 1
                        self.samples_uploaded += len(upload_item.sample_indices)
                        
                        # ========== 收集时间统计数据 ==========
                        with self.time_stats_lock:
                            self.time_stats['queue_wait_times'].append(queue_wait_time)
                            self.time_stats['upload_times'].append(upload_time)
                        
                        # 定期打印进度（改为每50个批次）
                        if self.items_uploaded % 50 == 0:
                            self._log("PROGRESS",
                                     f"已上传 {self.items_uploaded} 批次, {self.samples_uploaded} 样本 | "
                                     f"连接失败: {self.connection_failures}, 超时: {self.timeouts}, Fallback: {self.fallback_count}")
                    else:
                        # 上传失败，使用边侧预测作为fallback
                        self._handle_upload_failure(upload_item)
                    
                    # 标记任务完成
                    self.upload_queue.task_done()
                    
                except queue.Empty:
                    # 队列空，继续等待
                    continue
                    
                except Exception as e:
                    self._log("ERROR", f"上传线程异常: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 设置错误标志并记录错误
                    self.error_flag.set()
                    self.last_error = {
                        'exception': e,
                        'traceback': traceback.format_exc(),
                        'timestamp': time.time()
                    }
        
        except Exception as e:
            self._log("ERROR", f"上传线程严重错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 设置错误标志并记录错误
            self.error_flag.set()
            self.last_error = {
                'exception': e,
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
        
        finally:
            self._log("INFO", "上传线程结束")
            self._log("STATS", f"总计上传: {self.items_uploaded} 批次, {self.samples_uploaded} 样本")
            
            # ========== 输出时间统计摘要 ==========
            self._print_time_stats()
    
    def _print_time_stats(self):
        """打印时间统计摘要 - 简化版，只显示关键业务指标"""
        import numpy as np
        
        with self.time_stats_lock:
            if not self.time_stats['upload_times']:
                return
            
            # 只显示上传时间（最关键的指标）
            upload_times = np.array(self.time_stats['upload_times'])
            print(f"[{self.name}] 数据上传: 平均 {np.mean(upload_times):.2f}ms (P95: {np.percentile(upload_times, 95):.2f}ms)")
    
    def _upload_to_cloud(self, upload_item: UploadItem) -> bool:
        """
        上传数据到云侧（异步模式：只发送请求，不等待结果）
        
        Args:
            upload_item: 待上传的数据项
            
        Returns:
            bool: 上传是否成功
        """
        try:
            # 异步模式：只发送请求到云侧，不等待推理结果
            # 结果将由ReceiveThread异步接收
            success = self._send_async_request(upload_item)
            
            if success:
                # 上传成功，重置重试延迟
                self.current_retry_delay = self.retry_delay
                return True
            else:
                return False
            
        except ConnectionError as e:
            # 连接失败 - 实现指数退避重连机制
            self.connection_failures += 1
            print(f"[{self.name}] 连接失败 (批次 {upload_item.batch_id}): {e}")
            print(f"[{self.name}] 将在 {self.current_retry_delay:.1f}秒后重试...")
            
            # 指数退避：每次失败后延迟时间翻倍，直到达到最大延迟
            time.sleep(self.current_retry_delay)
            self.current_retry_delay = min(self.current_retry_delay * 2, self.max_retry_delay)
            
            return False
            
        except TimeoutError as e:
            # 超时
            self.timeouts += 1
            print(f"[{self.name}] 上传超时 (批次 {upload_item.batch_id}): {e}")
            return False
            
        except Exception as e:
            # 其他错误
            print(f"[{self.name}] 上传错误 (批次 {upload_item.batch_id}): {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _send_async_request(self, upload_item: UploadItem) -> bool:
        """
        发送异步推理请求到云侧（使用 ZeroMQ，不等待任何响应）
        
        Args:
            upload_item: 待上传的数据项
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # ========== T3: 边侧序列化时间 ==========
            serialize_start = time.time()
            
            # 构造请求
            request = {
                'type': 'cloud_inference',
                'batch_id': upload_item.batch_id,
                'sample_indices': upload_item.sample_indices,
                'inputs': upload_item.inputs.numpy() if torch.is_tensor(upload_item.inputs) else upload_item.inputs,
                'labels': upload_item.targets.numpy() if torch.is_tensor(upload_item.targets) else upload_item.targets,  # 转发标签
                'dataset_type': self.dataset_type,
                'num_classes': self.num_classes,
                'edge_id': getattr(self.network_edge, 'edge_id', 0),  # 添加边侧ID
                'edge_send_time': time.time()  # 添加发送时间戳
            }
            
            serialize_end = time.time()
            serialize_time = (serialize_end - serialize_start) * 1000  # 毫秒
            
            # ========== T4: 网络上传时间 ==========
            upload_start = time.time()
            
            # 使用 ZeroMQ 发送（非阻塞）
            self.network_edge.send_request_zmq(request)
            
            upload_end = time.time()
            upload_time = (upload_end - upload_start) * 1000  # 毫秒
            
            # 记录到全局时间追踪器
            if self.time_tracker:
                self.time_tracker.record_interval('edge_to_cloud_transfer', upload_start, upload_end)
            
            # 记录到 stats_collector
            if self.stats_collector:
                self.stats_collector.edge_serialize_times.append(serialize_time)
                self.stats_collector.network_upload_times.append(upload_time)
            
            # 异步模式：不等待任何响应！
            # 云侧会把结果发回来，由ReceiveThread接收
            return True
                
        except Exception as e:
            print(f"[{self.name}] 发送请求失败 (批次 {upload_item.batch_id}): {e}")
            raise
    
    def _handle_upload_failure(self, upload_item: UploadItem):
        """
        处理上传失败：使用边侧预测作为fallback
        
        Args:
            upload_item: 上传失败的数据项
        """
        self.fallback_count += 1
        
        # 标记云侧结果为完成（使用空结果，表示使用边侧预测）
        # 这样MergeThread会检测到批次完成，并使用边侧预测
        self.result_cache.add_cloud_result(
            batch_id=upload_item.batch_id,
            sample_indices=[],  # 空列表表示没有云侧结果
            predictions=torch.tensor([])  # 空张量
        )
        
        print(f"[{self.name}] 批次 {upload_item.batch_id} 使用边侧预测作为fallback")
    
    def stop(self):
        """
        停止上传线程（优雅关闭）
        """
        print(f"[{self.name}] 请求停止上传线程")
        self.stop_event.set()


class ReceiveThread(threading.Thread):
    """
    边侧接收线程
    
    异步接收云侧推理结果。
    从云侧接收推理响应，解析Batch ID和结果，存入ResultCache。
    处理接收超时和连接断开。
    """
    
    def __init__(self,
                 network_edge: Any,
                 result_cache: ResultCache,
                 stats_collector: Optional[PerformanceStats] = None,
                 stop_event: Optional[threading.Event] = None,
                 receive_timeout: float = 0.5,
                 time_tracker: Optional['GlobalTimeTracker'] = None):
        """
        初始化接收线程
        
        Args:
            network_edge: 网络通信对象（NetworkEdge实例，使用 ZeroMQ）
            result_cache: 结果缓存（需要锁保护）
            stats_collector: 性能统计收集器（可选）
            stop_event: 停止事件（用于优雅关闭）
            receive_timeout: 接收超时时间（秒），默认0.5秒
        """
        super().__init__(name="ReceiveThread")
        
        self.network_edge = network_edge
        self.result_cache = result_cache
        self.stats_collector = stats_collector
        
        # 停止事件（用于优雅关闭）
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        
        # 接收超时时间（转换为毫秒）
        self.receive_timeout_ms = int(receive_timeout * 1000)
        
        # 统计信息
        self.responses_received = 0
        self.samples_received = 0
        self.timeouts = 0
        self.parse_errors = 0
        
        # ========== 时间统计收集 ==========
        self.time_stats = {
            'parse_times': [],          # 解析时间
            'cache_times': [],          # 缓存存储时间
            'download_total_times': [], # 下载总时间
            'cloud_inference_times': [] # 云侧推理时间（从响应中获取）
        }
        self.time_stats_lock = threading.Lock()
        
        # 全局时间追踪器
        self.time_tracker = time_tracker
        
        # 错误处理：错误标志
        self.error_flag = threading.Event()
        self.last_error = None
    
    def _log(self, level, message):
        """统一的日志输出格式"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.name}] {message}")
    
    def run(self):
        """
        接收线程主循环
        
        执行流程：
        1. 从云侧接收推理响应（使用 ZeroMQ PULL，带超时）
        2. 解析Batch ID、sample indices、predictions
        3. 将结果存入ResultCache（加锁）
        """
        self._log("INFO", "接收线程启动")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # ========== T8: 网络下载开始 ==========
                    download_start = time.time()
                    
                    # 使用 ZeroMQ 接收响应（带超时）
                    response = self.network_edge.receive_response_zmq(timeout_ms=self.receive_timeout_ms)
                    
                    download_end = time.time()
                    
                    # 记录到全局时间追踪器
                    if response is not None and self.time_tracker:
                        self.time_tracker.record_interval('cloud_to_edge_transfer', download_start, download_end)
                    
                    if response is None:
                        # 超时，继续等待
                        self.timeouts += 1
                        continue
                    
                    parse_start = time.time()
                    
                    # 解析响应
                    batch_id = response.get('batch_id')
                    predictions = response.get('predictions')
                    sample_indices = response.get('sample_indices')
                    cloud_inference_time = response.get('inference_time', 0.0)
                    
                    # ========== 提取云侧时间信息 ==========
                    cloud_receive_time = response.get('cloud_receive_time', 0.0)  # T5
                    cloud_queue_wait_time = response.get('cloud_queue_wait_time', 0.0)  # T6
                    cloud_inference_only_time = response.get('cloud_inference_only_time', 0.0)  # T7
                    
                    # 验证响应数据完整性
                    if batch_id is None or predictions is None or sample_indices is None:
                        self.parse_errors += 1
                        self._log("WARN",
                                 f"响应数据不完整，跳过: batch_id={batch_id}, "
                                 f"predictions={'存在' if predictions is not None else '缺失'}, "
                                 f"sample_indices={'存在' if sample_indices is not None else '缺失'}")
                        continue
                    
                    # 转换为torch张量（如果需要）
                    if not torch.is_tensor(predictions):
                        predictions = torch.tensor(predictions)
                    
                    parse_end = time.time()
                    parse_time = (parse_end - parse_start) * 1000  # 毫秒
                    
                    cache_start = time.time()
                    
                    # 存入ResultCache（线程安全）
                    self.result_cache.add_cloud_result(
                        batch_id=batch_id,
                        sample_indices=sample_indices,
                        predictions=predictions,
                        cloud_latency_ms=cloud_inference_time
                    )
                    
                    cache_end = time.time()
                    cache_time = (cache_end - cache_start) * 1000  # 毫秒
                    
                    # 计算 T8: 网络下载时间
                    download_time = (download_end - download_start) * 1000  # 毫秒
                    
                    # 更新统计信息
                    self.responses_received += 1
                    self.samples_received += len(sample_indices)
                    
                    # ========== 记录详细时间到 stats_collector ==========
                    if self.stats_collector:
                        if cloud_receive_time > 0:
                            self.stats_collector.cloud_receive_times.append(cloud_receive_time)
                        if cloud_queue_wait_time > 0:
                            self.stats_collector.cloud_queue_wait_times.append(cloud_queue_wait_time)
                        if cloud_inference_only_time > 0:
                            self.stats_collector.cloud_inference_times.append(cloud_inference_only_time)
                        if download_time > 0:
                            self.stats_collector.network_download_times.append(download_time)
                    
                    # 定期打印进度（改为每50个响应）
                    if self.responses_received % 50 == 0:
                        self._log("PROGRESS",
                                 f"已接收 {self.responses_received} 个响应, {self.samples_received} 个样本 | "
                                 f"超时: {self.timeouts}, 解析错误: {self.parse_errors}")
                
                except Exception as e:
                    # 其他错误
                    self._log("ERROR", f"接收线程异常: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 设置错误标志并记录错误
                    self.error_flag.set()
                    self.last_error = {
                        'exception': e,
                        'traceback': traceback.format_exc(),
                        'timestamp': time.time()
                    }
                    
                    # 短暂等待后继续
                    time.sleep(0.5)
        
        except Exception as e:
            self._log("ERROR", f"接收线程严重错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 设置错误标志并记录错误
            self.error_flag.set()
            self.last_error = {
                'exception': e,
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
        
        finally:
            self._log("INFO", "接收线程结束")
            self._log("STATS", f"总计接收: {self.responses_received} 个响应, {self.samples_received} 个样本")
            
            # ========== 输出时间统计摘要 ==========
            self._print_time_stats()
    
    def _print_time_stats(self):
        """打印时间统计摘要 - 简化版，只显示关键业务指标"""
        import numpy as np
        
        with self.time_stats_lock:
            if not self.time_stats['cloud_inference_times']:
                return
            
            # 显示云侧推理时间和下载时间（最关键的指标）
            cloud_times = np.array(self.time_stats['cloud_inference_times'])
            print(f"[{self.name}] 云侧推理: 平均 {np.mean(cloud_times):.2f}ms (P95: {np.percentile(cloud_times, 95):.2f}ms)")
            
            if self.time_stats['download_total_times']:
                download_times = np.array(self.time_stats['download_total_times'])
                print(f"[{self.name}] 结果下载: 平均 {np.mean(download_times):.2f}ms (P95: {np.percentile(download_times, 95):.2f}ms)")
    
    def stop(self):
        """
        停止接收线程（优雅关闭）
        """
        print(f"[{self.name}] 请求停止接收线程")
        self.stop_event.set()



class MergeThread(threading.Thread):
    """
    边侧合并线程
    
    合并边侧和云侧结果，计算准确率。
    检查ResultCache找出已完成的批次，合并结果，处理超时批次。
    """
    
    def __init__(self,
                 result_cache: ResultCache,
                 stats_collector: Optional[PerformanceStats] = None,
                 stop_event: Optional[threading.Event] = None,
                 merge_interval: float = 0.01,
                 cloud_timeout: float = 30.0):
        """
        初始化合并线程
        
        Args:
            result_cache: 结果缓存（需要锁保护）
            stats_collector: 性能统计收集器（可选）
            stop_event: 停止事件（用于优雅关闭）
            merge_interval: 合并线程检查间隔（秒），默认0.01秒
            cloud_timeout: 云侧响应超时时间（秒），默认30秒
        """
        super().__init__(name="MergeThread")
        
        self.result_cache = result_cache
        self.stats_collector = stats_collector
        
        # 停止事件（用于优雅关闭）
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        
        # 配置参数
        self.merge_interval = merge_interval  # 检查间隔，避免busy-wait
        self.cloud_timeout = cloud_timeout    # 云侧超时时间
        
        # 统计信息
        self.batches_merged = 0
        self.samples_merged = 0
        self.timeout_batches = 0
        self.total_accuracy = 0.0
        
        # ========== 时间统计收集 ==========
        self.time_stats = {
            'merge_times': [],          # 合并时间
            'accuracy_times': [],       # 准确率计算时间
            'end_to_end_times': []      # 端到端延迟
        }
        self.time_stats_lock = threading.Lock()
        
        # 批次时间戳记录（用于计算端到端延迟）
        self.batch_timestamps: Dict[int, float] = {}
        self._timestamp_lock = threading.Lock()
        
        # 错误处理：错误标志
        self.error_flag = threading.Event()
        self.last_error = None
    
    def _log(self, level, message):
        """统一的日志输出格式"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.name}] {message}")
    
    def run(self):
        """
        合并线程主循环
        
        执行流程：
        1. 检查ResultCache找出已完成的批次
        2. 合并边侧和云侧结果
        3. 处理超时批次（使用边侧预测作为fallback）
        4. 计算批次准确率
        5. 更新PerformanceStats
        6. 清理已处理的批次
        """
        self._log("INFO", "合并线程启动")
        
        try:
            while not self.stop_event.is_set():
                # 查找已完成的批次
                completed_batches = self.result_cache.get_completed_batches(
                    timeout_seconds=self.cloud_timeout
                )
                
                if completed_batches:
                    # 处理每个已完成的批次
                    for batch_id in completed_batches:
                        self._process_completed_batch(batch_id)
                
                # 定期检查，避免busy-wait
                time.sleep(self.merge_interval)
        
        except Exception as e:
            self._log("ERROR", f"合并线程异常: {e}")
            import traceback
            traceback.print_exc()
            
            # 设置错误标志并记录错误
            self.error_flag.set()
            self.last_error = {
                'exception': e,
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
        
        finally:
            self._log("INFO", "合并线程结束")
            self._log("STATS", f"总计合并: {self.batches_merged} 批次, {self.samples_merged} 样本")
            if self.batches_merged > 0:
                avg_accuracy = self.total_accuracy / self.batches_merged
                self._log("STATS", f"平均准确率: {avg_accuracy:.4f}")
            
            # ========== 输出时间统计摘要 ==========
            self._print_time_stats()
    
    def _print_time_stats(self):
        """打印时间统计摘要 - 简化版，只显示关键业务指标"""
        import numpy as np
        
        with self.time_stats_lock:
            if not self.time_stats['end_to_end_times']:
                return
            
            # 只显示端到端延迟（最关键的指标）
            e2e_times = np.array(self.time_stats['end_to_end_times'])
            print(f"[{self.name}] 端到端延迟: 平均 {np.mean(e2e_times):.2f}ms (P95: {np.percentile(e2e_times, 95):.2f}ms)")
    
    def _process_completed_batch(self, batch_id: int):
        """
        处理已完成的批次
        
        Args:
            batch_id: 批次ID
        """
        # ========== 时间记录：合并开始 ==========
        merge_start = time.time()
        
        # 获取批次结果
        batch = self.result_cache.get_batch(batch_id)
        
        if batch is None:
            # 批次不存在（可能已被清理）
            return
        
        # 检查是否超时
        is_timeout = (time.time() - batch.timestamp) > self.cloud_timeout
        
        if is_timeout and not batch.cloud_complete:
            # 超时且云侧结果未完成，使用边侧预测作为fallback
            self.timeout_batches += 1
            print(f"[{self.name}] 批次 {batch_id} 超时，使用边侧预测作为fallback")
        
        # 合并边侧和云侧结果
        merged_results = batch.get_merged_results()
        
        # ========== 时间记录：合并结束 ==========
        merge_end = time.time()
        merge_time = (merge_end - merge_start) * 1000  # 毫秒
        
        # ========== 时间记录：准确率计算开始 ==========
        accuracy_start = time.time()
        
        # 计算批次准确率
        accuracy = batch.calculate_accuracy()
        
        # ========== 时间记录：准确率计算结束 ==========
        accuracy_end = time.time()
        accuracy_time = (accuracy_end - accuracy_start) * 1000  # 毫秒
        
        # ========== 时间记录：端到端延迟 ==========
        end_to_end_latency = (accuracy_end - batch.timestamp) * 1000  # 毫秒
        
        # 更新统计信息
        self.batches_merged += 1
        self.samples_merged += batch.total_samples
        self.total_accuracy += accuracy
        
        # 更新PerformanceStats
        if self.stats_collector:
            self.stats_collector.update_batch(
                batch_result=batch,
                edge_latency=batch.edge_latency_ms,
                cloud_latency=batch.cloud_latency_ms,
                end_to_end_latency=end_to_end_latency
            )
        
        # ========== 收集时间统计数据 ==========
        with self.time_stats_lock:
            self.time_stats['merge_times'].append(merge_time)
            self.time_stats['accuracy_times'].append(accuracy_time)
            self.time_stats['end_to_end_times'].append(end_to_end_latency)
        
        # 清理已处理的批次
        self.result_cache.remove_batch(batch_id)
    
    def record_batch_start(self, batch_id: int, timestamp: float):
        """
        记录批次开始时间（用于计算端到端延迟）
        
        Args:
            batch_id: 批次ID
            timestamp: 开始时间戳
        """
        with self._timestamp_lock:
            self.batch_timestamps[batch_id] = timestamp
    
    def stop(self):
        """
        停止合并线程（优雅关闭）
        """
        print(f"[{self.name}] 请求停止合并线程")
        self.stop_event.set()

#!/usr/bin/env python3
"""
云侧异步流水线核心数据结构和工具类
用于云侧协同推理的异步流水线架构
"""
import threading
import time
import queue
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import numpy as np


@dataclass
class InferenceRequest:
    """推理请求 - 云侧接收的推理请求"""
    batch_id: int                    # 批次ID
    sample_indices: List[int]        # 样本索引
    inputs: np.ndarray               # 输入数据（numpy格式，便于序列化）
    edge_id: int                     # 边侧ID
    timestamp: float                 # 请求时间戳
    labels: Optional[np.ndarray] = None  # 真实标签（可选，用于计算准确率）
    
    # ========== 新增：时间戳字段 ==========
    edge_send_time: float = 0.0      # 边侧发送时间
    cloud_receive_start: float = 0.0 # 云侧接收开始时间
    cloud_receive_end: float = 0.0   # 云侧接收结束时间
    cloud_queue_enter: float = 0.0   # 进入队列时间
    
    def __post_init__(self):
        """验证数据完整性"""
        if len(self.sample_indices) != self.inputs.shape[0]:
            raise ValueError(f"样本索引数量({len(self.sample_indices)})与输入数据批次大小({self.inputs.shape[0]})不匹配")


@dataclass
class InferenceResponse:
    """推理响应 - 云侧返回的推理结果"""
    batch_id: int                    # 批次ID
    sample_indices: List[int]        # 样本索引
    predictions: List[int]           # 预测结果
    edge_id: int                     # 边侧ID
    inference_time: float            # 推理耗时（毫秒）
    
    # ========== 新增：详细时间信息 ==========
    cloud_receive_time: float = 0.0          # T5: 云侧接收时间（毫秒）
    cloud_queue_wait_time: float = 0.0       # T6: 云侧队列等待时间（毫秒）
    cloud_inference_only_time: float = 0.0   # T7: 云侧推理时间（毫秒）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        return {
            'status': 'success',  # 添加状态字段，表示推理成功
            'batch_id': self.batch_id,
            'sample_indices': self.sample_indices,
            'predictions': self.predictions,
            'edge_id': self.edge_id,
            'inference_time': self.inference_time,
            # 新增：详细时间信息
            'cloud_receive_time': self.cloud_receive_time,
            'cloud_queue_wait_time': self.cloud_queue_wait_time,
            'cloud_inference_only_time': self.cloud_inference_only_time,
        }


class InferenceQueue:
    """
    推理队列 - 云侧用于缓冲待处理的推理请求
    
    基于queue.Queue实现，提供线程安全的队列操作。
    """
    
    def __init__(self, maxsize: int = 1000):
        """
        初始化推理队列
        
        Args:
            maxsize: 队列最大大小（0表示无限制，默认1000）
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        
        # 统计信息
        self._total_requests = 0
        self._total_processed = 0
        self._stats_lock = threading.Lock()
    
    def put(self, request: InferenceRequest, block: bool = True, timeout: Optional[float] = None):
        """
        添加推理请求到队列
        
        Args:
            request: InferenceRequest对象
            block: 是否阻塞（如果队列满）
            timeout: 超时时间（秒），None表示无限等待
        
        Raises:
            queue.Full: 如果队列满且不阻塞
        """
        self._queue.put(request, block=block, timeout=timeout)
        
        with self._stats_lock:
            self._total_requests += 1
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> InferenceRequest:
        """
        从队列获取推理请求
        
        Args:
            block: 是否阻塞（如果队列空）
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            InferenceRequest对象
            
        Raises:
            queue.Empty: 如果队列空且不阻塞
        """
        request = self._queue.get(block=block, timeout=timeout)
        
        with self._stats_lock:
            self._total_processed += 1
        
        return request
    
    def task_done(self):
        """标记任务完成"""
        self._queue.task_done()
    
    def join(self):
        """
        阻塞直到队列中的所有任务都被处理完成
        
        每次调用get()后都应该调用task_done()来标记任务完成。
        join()会阻塞直到所有任务都调用了task_done()。
        """
        self._queue.join()
    
    def qsize(self) -> int:
        """
        获取队列大小（近似值）
        
        Returns:
            队列中的请求数量
        """
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """
        检查队列是否为空
        
        Returns:
            True 如果队列为空
        """
        return self._queue.empty()
    
    def full(self) -> bool:
        """
        检查队列是否已满
        
        Returns:
            True 如果队列已满
        """
        return self._queue.full()
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取队列统计信息
        
        Returns:
            统计信息字典
        """
        with self._stats_lock:
            return {
                'total_requests': self._total_requests,
                'total_processed': self._total_processed,
                'current_size': self.qsize(),
                'maxsize': self._maxsize
            }


class ResponseSender:
    """
    响应发送器 - 使用 ZeroMQ PUSH socket 发送推理结果到边侧
    
    使用 ZeroMQ PUSH-PULL 模式，无需管理连接。
    """
    
    def __init__(self, push_port: int = 5556):
        """
        初始化响应发送器
        
        Args:
            push_port: ZeroMQ PUSH socket 端口（默认 5556）
        """
        import zmq
        
        # 创建 ZeroMQ context 和 PUSH socket
        self.zmq_context = zmq.Context()
        self.zmq_push_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_push_socket.bind(f"tcp://*:{push_port}")
        
        print(f"[ResponseSender] ZeroMQ PUSH socket 已绑定到端口 {push_port}")
        
        # 尝试导入 MessagePack
        try:
            import msgpack
            import msgpack_numpy as m
            m.patch()
            self.use_msgpack = True
            print(f"[ResponseSender] 使用 MessagePack 序列化")
        except ImportError:
            self.use_msgpack = False
            print(f"[ResponseSender] 使用 Pickle 序列化")
        
        # 统计信息
        self._total_responses = 0
        self._failed_responses = 0
        self._stats_lock = threading.Lock()
    
    def send(self, response: InferenceResponse, edge_id: int = None) -> bool:
        """
        发送响应（使用 ZeroMQ PUSH）
        
        Args:
            response: InferenceResponse对象
            edge_id: 边侧ID（ZeroMQ 模式下不需要，保留参数以兼容接口）
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 序列化响应
            response_dict = response.to_dict()
            
            if self.use_msgpack:
                import msgpack
                serialized = msgpack.packb(response_dict, use_bin_type=True)
            else:
                import pickle
                serialized = pickle.dumps(response_dict)
            
            # 使用 ZeroMQ 发送（非阻塞）
            self.zmq_push_socket.send(serialized, flags=0)
            
            with self._stats_lock:
                self._total_responses += 1
            
            return True
            
        except Exception as e:
            print(f"[ResponseSender] 发送响应失败: {e}")
            
            with self._stats_lock:
                self._failed_responses += 1
            
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._stats_lock:
            return {
                'total_responses': self._total_responses,
                'failed_responses': self._failed_responses
            }
    
    def close(self):
        """关闭 ZeroMQ socket"""
        try:
            self.zmq_push_socket.close()
            self.zmq_context.term()
            print(f"[ResponseSender] ZeroMQ 连接已关闭")
        except Exception as e:
            print(f"[ResponseSender] 关闭 ZeroMQ 时出错: {e}")


class WorkerThread(threading.Thread):
    """
    云侧工作线程 - 从队列取请求，执行GPU推理
    
    多个WorkerThread并发执行推理，提高GPU利用率。
    """
    
    # 类级别的统计字典（所有线程共享）
    _global_stats = {}
    _global_stats_lock = threading.Lock()
    
    # 添加定期统计输出的时间戳
    _last_stats_print_time = time.time()
    _stats_print_interval = 30  # 每30秒输出一次统计汇总
    
    def __init__(self,
                 worker_id: int,
                 inference_queue: InferenceQueue,
                 model: torch.nn.Module,
                 device: torch.device,
                 response_sender: ResponseSender,
                 stop_event: Optional[threading.Event] = None,
                 batch_timeout: float = 0.05):
        """
        初始化工作线程
        
        Args:
            worker_id: 工作线程ID
            inference_queue: 推理队列
            model: 云侧教师模型
            device: GPU设备
            response_sender: 响应发送器
            stop_event: 停止事件（用于优雅关闭）
            batch_timeout: 批量推理等待超时（秒，仅对边侧数据生效）
        
        Note:
            批量推理策略：
            - 端侧数据 (edge_id='device_direct'): 始终单独处理
            - 边侧数据 (其他edge_id): 自动启用批量推理优化
        """
        super().__init__(name=f"WorkerThread-{worker_id}")
        
        self.worker_id = worker_id
        self.inference_queue = inference_queue
        self.model = model
        self.device = device
        self.response_sender = response_sender
        
        # 停止事件（用于优雅关闭）
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        
        # 批量推理配置（自动根据数据来源决定）
        self.batch_timeout = batch_timeout
        
        # 统计信息
        self.requests_processed = 0
        self.samples_processed = 0
        self.total_inference_time = 0.0
        self.errors = 0
        self.batches_merged = 0
        
        # ========== 时间统计收集 ==========
        self.time_stats = {
            'inference_times': [],      # 推理时间
            'serialize_times': [],      # 序列化时间
            'send_times': [],           # 发送时间
            'cloud_total_times': []     # 云侧总时间
        }
        self.time_stats_lock = threading.Lock()
    
    @classmethod
    def get_global_stats(cls, edge_id):
        """获取指定 edge_id 的全局统计信息"""
        with cls._global_stats_lock:
            return cls._global_stats.get(edge_id, {
                'total_samples': 0,
                'correct_predictions': 0,
                'total_batches': 0,
                'total_inference_time': 0.0,
                'start_time': None,
                'samples_with_labels': 0  # 添加：有标签的样本数
            })
    
    @classmethod
    def update_global_stats(cls, edge_id, correct, total, inference_time, has_labels=True):
        """更新指定 edge_id 的全局统计信息"""
        with cls._global_stats_lock:
            if edge_id not in cls._global_stats:
                cls._global_stats[edge_id] = {
                    'total_samples': 0,
                    'correct_predictions': 0,
                    'total_batches': 0,
                    'total_inference_time': 0.0,
                    'start_time': time.time(),
                    'samples_with_labels': 0
                }
            cls._global_stats[edge_id]['total_samples'] += total
            cls._global_stats[edge_id]['total_batches'] += 1
            cls._global_stats[edge_id]['total_inference_time'] += inference_time
            
            if has_labels:
                cls._global_stats[edge_id]['correct_predictions'] += correct
                cls._global_stats[edge_id]['samples_with_labels'] += total
            
            # 定期输出统计汇总（每30秒）
            current_time = time.time()
            if current_time - cls._last_stats_print_time >= cls._stats_print_interval:
                cls._last_stats_print_time = current_time
                cls._print_periodic_stats()
    
    @classmethod
    def _print_periodic_stats(cls):
        """定期打印统计汇总（内部方法）"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}] [STATS] ========== 推理统计汇总 ==========")
        
        for edge_id, stats in cls._global_stats.items():
            if stats['total_samples'] > 0:
                elapsed_time = time.time() - stats['start_time']
                avg_time_per_sample = stats['total_inference_time'] / stats['total_samples']
                throughput = stats['total_samples'] / elapsed_time if elapsed_time > 0 else 0
                
                print(f"[{timestamp}] [STATS] Edge ID: {edge_id}")
                print(f"[{timestamp}] [STATS]   - 总样本数: {stats['total_samples']}")
                print(f"[{timestamp}] [STATS]   - 总批次: {stats['total_batches']}")
                
                # 只有当有标签的样本数大于0时才显示准确率
                if stats['samples_with_labels'] > 0:
                    accuracy = 100.0 * stats['correct_predictions'] / stats['samples_with_labels']
                    print(f"[{timestamp}] [STATS]   - 准确率: {accuracy:.2f}% ({stats['correct_predictions']}/{stats['samples_with_labels']})")
                
                print(f"[{timestamp}] [STATS]   - 平均推理时间: {avg_time_per_sample*1000:.2f}ms/样本")
                print(f"[{timestamp}] [STATS]   - 吞吐量: {throughput:.2f} 样本/秒")
        
        print(f"[{timestamp}] [STATS] ==========================================\n")
    
    @classmethod
    def print_global_stats(cls, edge_id):
        """打印指定 edge_id 的全局统计信息"""
        with cls._global_stats_lock:
            if edge_id in cls._global_stats:
                stats = cls._global_stats[edge_id]
                if stats['total_samples'] > 0:
                    total_time = time.time() - stats['start_time']
                    avg_time_per_sample = stats['total_inference_time'] / stats['total_samples']
                    throughput = stats['total_samples'] / total_time
                    
                    print(f"\n{'='*70}")
                    print(f"[云侧] 推理统计汇总")
                    print(f"{'='*70}")
                    print(f"总样本数:           {stats['total_samples']}")
                    print(f"总批次数:           {stats['total_batches']}")
                    
                    # 只有当有标签的样本数大于0时才显示准确率
                    if stats['samples_with_labels'] > 0:
                        accuracy = 100.0 * stats['correct_predictions'] / stats['samples_with_labels']
                        print(f"有标签样本数:       {stats['samples_with_ labels']}")
                        print(f"整体准确率:         {accuracy:.2f}% ({stats['correct_predictions']}/{stats['samples_with_labels']})")
                    
                    print(f"总耗时:             {total_time:.2f}秒")
                    print(f"平均推理时间:       {avg_time_per_sample*1000:.2f}ms/样本")
                    print(f"吞吐量:             {throughput:.2f} 样本/秒")
                    print(f"{'='*70}\n")
    
    def _log(self, level, message):
        """统一的日志输出格式"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.name}] {message}")
    
    def run(self):
        """
        工作线程主循环
        
        执行流程：
        1. 从inference_queue获取请求（带超时）
        2. 自动根据edge_id判断处理策略：
           - 端侧数据 (edge_id='device_direct'): 单独处理（实时性优先）
           - 边侧数据 (其他edge_id): 批量处理（吞吐量优先）
        3. 执行GPU推理（单个或批量）
        4. 构造InferenceResponse
        5. 通过ResponseSender发送结果
        6. 处理GPU内存不足等异常
        """
        self._log("INFO", f"工作线程启动，使用设备: {self.device}")
        self._log("INFO", f"批量推理策略: 端侧数据单独处理，边侧数据批量处理 (timeout={self.batch_timeout}s)")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # 从队列获取请求（带超时，避免阻塞）
                    request = self.inference_queue.get(timeout=0.1)
                    
                    # ========== 自动根据数据来源决定处理策略 ==========
                    # 端侧数据 (device_direct): 单独处理（实时性优先）
                    # 边侧数据 (其他edge_id): 批量处理（吞吐量优先）
                    is_device_direct = (request.edge_id == 'device_direct')
                    
                    if is_device_direct:
                        # 端侧数据：单独处理
                        self._process_single_request(request)
                    else:
                        # 边侧数据：尝试批量处理
                        requests = self._collect_batch_requests(request)
                        
                        if len(requests) > 1:
                            self.batches_merged += 1
                            # 批量处理
                            self._process_batch_requests(requests)
                        else:
                            # 单个处理
                            self._process_single_request(request)
                
                except queue.Empty:
                    # 队列空，继续等待
                    continue
                
                except Exception as e:
                    # 其他异常
                    self._log("ERROR", f"工作线程异常: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            self._log("ERROR", f"工作线程严重错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._log("INFO", "工作线程结束")
            self._log("STATS", f"总计处理: {self.requests_processed} 个请求, {self.samples_processed} 个样本")
            self._log("STATS", f"批量合并次数: {self.batches_merged} (仅边侧数据)")
            if self.requests_processed > 0:
                avg_time = self.total_inference_time / self.requests_processed
                self._log("STATS", f"平均推理时间: {avg_time:.2f}ms")
            if self.errors > 0:
                self._log("WARN", f"错误数: {self.errors}")
            
            # ========== 输出时间统计摘要 ==========
            self._print_time_stats()
    
    def _collect_batch_requests(self, first_request: InferenceRequest) -> List[InferenceRequest]:
        """
        收集多个边侧请求进行批量推理
        
        注意：此方法仅用于边侧数据，端侧数据不会调用此方法
        
        Args:
            first_request: 第一个请求（必须是边侧请求）
            
        Returns:
            请求列表
        """
        requests = [first_request]
        start_time = time.time()
        first_edge_id = first_request.edge_id
        
        # 尝试收集更多来自同一边侧的请求（直到超时）
        while time.time() - start_time < self.batch_timeout:
            try:
                # 非阻塞获取
                request = self.inference_queue.get(block=False)
                
                # 只合并来自同一边侧的请求，跳过端侧请求
                if request.edge_id == first_edge_id and request.edge_id != 'device_direct':
                    requests.append(request)
                    
                    # 限制批量大小（避免GPU内存不足）
                    if len(requests) >= 32:
                        break
                else:
                    # 不同来源的请求，放回队列
                    self.inference_queue.put(request, block=False)
                    break
                    
            except queue.Empty:
                # 队列空，等待一小段时间
                time.sleep(0.001)
                continue
            except queue.Full:
                # 队列满，无法放回，只能处理
                self._log("WARN", f"无法将不同来源的请求放回队列，将单独处理")
                self._process_single_request(request)
                break
        
        return requests
    
    def _process_batch_requests(self, requests: List[InferenceRequest]):
        """
        批量处理多个推理请求
        
        Args:
            requests: 请求列表
        """
        # 记录推理开始时间
        inference_start = time.time()
        
        try:
            # 合并所有输入
            all_inputs = []
            request_info = []  # 记录每个请求的信息
            
            for request in requests:
                all_inputs.append(request.inputs)
                request_info.append({
                    'request': request,
                    'start_idx': len(all_inputs) - 1,
                    'num_samples': len(request.sample_indices)
                })
            
            # 合并为单个批次
            batch_inputs = np.concatenate(all_inputs, axis=0)
            
            # 执行批量推理
            predictions = self._execute_inference_batch(batch_inputs)
            
            # 记录推理时间
            inference_time = (time.time() - inference_start) * 1000  # 毫秒
            
            # 拆分结果并发送到对应边侧
            current_idx = 0
            for info in request_info:
                request = info['request']
                num_samples = info['num_samples']
                
                # 提取该请求的预测结果
                request_predictions = predictions[current_idx:current_idx + num_samples]
                current_idx += num_samples
                
                # 构造响应
                response = InferenceResponse(
                    batch_id=request.batch_id,
                    sample_indices=request.sample_indices,
                    predictions=request_predictions,
                    edge_id=request.edge_id,
                    inference_time=inference_time / len(requests)  # 平均时间
                )
                
                # 发送响应
                success = self.response_sender.send(response, request.edge_id)
                
                if success:
                    # 更新统计信息
                    self.requests_processed += 1
                    self.samples_processed += num_samples
                    self.total_inference_time += inference_time / len(requests)
                else:
                    self.errors += 1
                    print(f"[{self.name}] 发送响应失败 (批次 {request.batch_id})")
                
                # 标记任务完成
                self.inference_queue.task_done()
            
            # 定期打印进度（改为每50个请求打印一次，减少日志量）
            if self.requests_processed > 0 and self.requests_processed % 50 == 0:
                avg_time = self.total_inference_time / self.requests_processed
                self._log("PROGRESS", 
                         f"已处理 {self.requests_processed} 个请求, {self.samples_processed} 个样本 | "
                         f"平均推理时间: {avg_time:.2f}ms | 边侧批量合并: {self.batches_merged} 次")
        
        except RuntimeError as e:
            # GPU内存不足或其他运行时错误
            self.errors += len(requests)
            self._log("ERROR", f"批量推理错误: {e}")
            
            # 尝试清理GPU内存
            if 'out of memory' in str(e).lower():
                self._log("WARN", "GPU内存不足，尝试清理...")
                torch.cuda.empty_cache()
            
            # 标记所有任务完成
            for _ in requests:
                try:
                    self.inference_queue.task_done()
                except ValueError:
                    pass  # 忽略重复调用
        
        except Exception as e:
            # 其他错误
            self.errors += len(requests)
            self._log("ERROR", f"批量处理请求错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 标记所有任务完成
            for _ in requests:
                try:
                    self.inference_queue.task_done()
                except ValueError:
                    pass  # 忽略重复调用
    
    def _process_single_request(self, request: InferenceRequest):
        """
        处理单个推理请求
        
        Args:
            request: InferenceRequest对象
        """
        # 判断数据来源
        is_device_direct = (request.edge_id == 'device_direct')
        source_type = "端侧" if is_device_direct else f"边侧-{request.edge_id}"
        
        # ========== T6: 云侧队列等待时间 ==========
        cloud_queue_exit = time.time()
        cloud_queue_wait_time = (cloud_queue_exit - request.cloud_queue_enter) * 1000 if request.cloud_queue_enter > 0 else 0.0
        
        # ========== T5: 云侧接收时间 ==========
        cloud_receive_time = (request.cloud_receive_end - request.cloud_receive_start) * 1000 if request.cloud_receive_start > 0 else 0.0
        
        # ========== T7: 云侧推理时间 ==========
        inference_start = time.time()
        
        # 执行推理
        try:
            predictions = self._execute_inference(request)
            
            inference_end = time.time()
            inference_time = (inference_end - inference_start)  # 秒
            cloud_inference_only_time = inference_time * 1000  # 毫秒
            
            serialize_start = time.time()
            
            # 构造响应
            response = InferenceResponse(
                batch_id=request.batch_id,
                sample_indices=request.sample_indices,
                predictions=predictions,
                edge_id=request.edge_id,
                inference_time=inference_time * 1000,  # 转换为毫秒
                # 新增：详细时间信息
                cloud_receive_time=cloud_receive_time,
                cloud_queue_wait_time=cloud_queue_wait_time,
                cloud_inference_only_time=cloud_inference_only_time
            )
            
            serialize_end = time.time()
            serialize_time = (serialize_end - serialize_start) * 1000  # 毫秒
            
            send_start = time.time()
            
            # 发送响应
            success = self.response_sender.send(response, request.edge_id)
            
            send_end = time.time()
            send_time = (send_end - send_start) * 1000  # 毫秒
            
            if success:
                # 更新统计信息
                self.requests_processed += 1
                self.samples_processed += len(request.sample_indices)
                self.total_inference_time += inference_time * 1000  # 毫秒
                
                # 每100个请求打印一次处理信息
                if self.requests_processed % 100 == 0:
                    self._log("PROGRESS", 
                             f"已处理 {self.requests_processed} 个请求 | "
                             f"{source_type} | 单独处理")
            else:
                self.errors += 1
                self._log("ERROR", f"发送响应失败 (批次 {request.batch_id})")
        
        except RuntimeError as e:
            # GPU内存不足或其他运行时错误
            self.errors += 1
            self._log("ERROR", f"推理错误 (批次 {request.batch_id}): {e}")
            
            # 尝试清理GPU内存
            if 'out of memory' in str(e).lower():
                self._log("WARN", "GPU内存不足，尝试清理...")
                torch.cuda.empty_cache()
        
        except Exception as e:
            # 其他错误
            self.errors += 1
            self._log("ERROR", f"处理请求错误 (批次 {request.batch_id}): {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 标记任务完成
            self.inference_queue.task_done()
    
    def _execute_inference_batch(self, batch_inputs: np.ndarray) -> List[int]:
        """
        执行批量推理
        
        Args:
            batch_inputs: 批量输入数据（numpy数组）
            
        Returns:
            预测结果列表
        """
        # 转换输入数据为张量
        # 检查数据类型：如果已经是复数，直接使用；否则转换
        if np.iscomplexobj(batch_inputs):
            # 数据已经是复数格式 (batch, length)
            inputs = torch.from_numpy(batch_inputs.copy()).cfloat().to(self.device)
        elif len(batch_inputs.shape) == 3 and batch_inputs.shape[1] == 2:
            # I/Q 格式：(batch, 2, length) -> 复数 (batch, length)
            signal_complex = batch_inputs[:, 0, :] + 1j * batch_inputs[:, 1, :]
            inputs = torch.from_numpy(signal_complex).cfloat().to(self.device)
        else:
            # 实数格式
            inputs = torch.from_numpy(batch_inputs.copy()).float().to(self.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # 处理复数输出
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            
            # 获取预测结果
            predictions = torch.argmax(outputs, dim=1)
        
        # 转换为Python列表
        return predictions.cpu().tolist()
    
    def _execute_inference(self, request: InferenceRequest) -> List[int]:
        """
        执行推理
        
        Args:
            request: InferenceRequest对象
            
        Returns:
            预测结果列表
        """
        # 记录推理开始时间
        inference_start = time.time()
        
        # 转换输入数据为张量
        # 检查数据类型：如果已经是复数，直接使用；否则转换
        if np.iscomplexobj(request.inputs):
            # 数据已经是复数格式 (batch, length)
            inputs = torch.from_numpy(request.inputs.copy()).cfloat().to(self.device)
        elif len(request.inputs.shape) == 3 and request.inputs.shape[1] == 2:
            # I/Q 格式：(batch, 2, length) -> 复数 (batch, length)
            signal_complex = request.inputs[:, 0, :] + 1j * request.inputs[:, 1, :]
            inputs = torch.from_numpy(signal_complex).cfloat().to(self.device)
        else:
            # 实数格式
            inputs = torch.from_numpy(request.inputs.copy()).float().to(self.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # 处理复数输出
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            
            # 获取预测结果
            predictions = torch.argmax(outputs, dim=1)
        
        # 计算推理时间
        inference_time = time.time() - inference_start
        
        # 获取样本数量
        total_samples = len(request.sample_indices)
        
        # 如果有标签，计算准确率
        if request.labels is not None:
            labels_tensor = torch.from_numpy(request.labels.copy()).to(self.device)
            correct = (predictions == labels_tensor).sum().item()
            
            # 更新全局统计（有标签）
            self.update_global_stats(request.edge_id, correct, total_samples, inference_time, has_labels=True)
            
            # 获取当前统计
            stats = self.get_global_stats(request.edge_id)
            
            # 每20个批次打印一次准确率
            if stats['total_batches'] % 20 == 0:
                batch_accuracy = 100.0 * correct / total_samples
                overall_accuracy = 100.0 * stats['correct_predictions'] / stats['samples_with_labels']
                self._log("ACCURACY",
                         f"edge_id={request.edge_id}, batch={stats['total_batches']}: "
                         f"batch_acc={batch_accuracy:.2f}% ({correct}/{total_samples}), "
                         f"overall_acc={overall_accuracy:.2f}% ({stats['correct_predictions']}/{stats['samples_with_labels']})")
        else:
            # 更新全局统计（无标签）
            self.update_global_stats(request.edge_id, 0, total_samples, inference_time, has_labels=False)
        
        # 转换为Python列表
        return predictions.cpu().tolist()
    
    def stop(self):
        """
        停止工作线程（优雅关闭）
        """
        print(f"[{self.name}] 请求停止工作线程")
        self.stop_event.set()
    
    def _print_time_stats(self):
        """打印时间统计摘要"""
        import numpy as np
        
        with self.time_stats_lock:
            if not self.time_stats['inference_times']:
                return
            
            print(f"\n{'='*80}")
            print(f"[{self.name}] 云侧时间统计摘要")
            print(f"{'='*80}")
            
            for key, times in self.time_stats.items():
                if times:
                    times_array = np.array(times)
                    stage_name = key.replace('_', ' ').title().replace('Times', '')
                    
                    print(f"\n{stage_name}:")
                    print(f"  平均值: {np.mean(times_array):.2f}ms")
                    print(f"  中位数: {np.median(times_array):.2f}ms")
                    print(f"  最小值: {np.min(times_array):.2f}ms")
                    print(f"  最大值: {np.max(times_array):.2f}ms")
                    print(f"  P90:    {np.percentile(times_array, 90):.2f}ms")
                    print(f"  P95:    {np.percentile(times_array, 95):.2f}ms")
                    print(f"  P99:    {np.percentile(times_array, 99):.2f}ms")
            
            print(f"\n{'='*80}\n")

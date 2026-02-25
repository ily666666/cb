#!/usr/bin/env python3
"""
时间性能分析器 - 严格追踪云边端协同推理的时间消耗

追踪的关键时间点：
1. 端侧 -> 边侧：数据传输时间
2. 边侧推理：本地模型推理时间
3. 边侧 -> 云侧：上传时间（网络传输 + 序列化）
4. 云侧推理：云端模型推理时间
5. 云侧 -> 边侧：下载时间（网络传输 + 反序列化）
6. 边侧合并：结果合并时间
7. 端到端延迟：从样本到达边侧到最终结果的总时间
"""
import time
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


@dataclass
class TimePoint:
    """单个时间点记录"""
    timestamp: float  # 绝对时间戳（秒）
    event: str        # 事件名称
    batch_id: int     # 批次ID
    sample_count: int = 0  # 样本数量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class BatchTimeProfile:
    """单个批次的完整时间剖析"""
    batch_id: int
    sample_count: int
    
    # ========== 边侧时间点 ==========
    # 1. 数据到达边侧
    edge_data_arrival: Optional[float] = None
    
    # 2. 边侧推理开始
    edge_inference_start: Optional[float] = None
    
    # 3. 边侧推理结束
    edge_inference_end: Optional[float] = None
    
    # 4. 决策分流完成（高置信度直接缓存，低置信度入队）
    edge_decision_complete: Optional[float] = None
    
    # ========== 上传阶段时间点 ==========
    # 5. 低置信度样本入队时间
    upload_queue_enqueue: Optional[float] = None
    
    # 6. 从上传队列取出时间
    upload_queue_dequeue: Optional[float] = None
    
    # 7. 序列化开始
    upload_serialize_start: Optional[float] = None
    
    # 8. 序列化结束
    upload_serialize_end: Optional[float] = None
    
    # 9. 网络发送开始
    upload_network_start: Optional[float] = None
    
    # 10. 网络发送结束
    upload_network_end: Optional[float] = None
    
    # ========== 云侧时间点 ==========
    # 11. 云侧接收到请求
    cloud_request_received: Optional[float] = None
    
    # 12. 请求入推理队列
    cloud_queue_enqueue: Optional[float] = None
    
    # 13. 从推理队列取出
    cloud_queue_dequeue: Optional[float] = None
    
    # 14. 云侧推理开始
    cloud_inference_start: Optional[float] = None
    
    # 15. 云侧推理结束
    cloud_inference_end: Optional[float] = None
    
    # 16. 响应序列化开始
    cloud_serialize_start: Optional[float] = None
    
    # 17. 响应序列化结束
    cloud_serialize_end: Optional[float] = None
    
    # 18. 响应发送开始
    cloud_response_send_start: Optional[float] = None
    
    # 19. 响应发送结束
    cloud_response_send_end: Optional[float] = None
    
    # ========== 下载阶段时间点 ==========
    # 20. 边侧接收到响应
    edge_response_received: Optional[float] = None
    
    # 21. 反序列化开始
    edge_deserialize_start: Optional[float] = None
    
    # 22. 反序列化结束
    edge_deserialize_end: Optional[float] = None
    
    # 23. 结果存入缓存
    edge_cache_stored: Optional[float] = None
    
    # ========== 合并阶段时间点 ==========
    # 24. 合并线程检测到批次完成
    merge_detected: Optional[float] = None
    
    # 25. 合并开始
    merge_start: Optional[float] = None
    
    # 26. 合并结束
    merge_end: Optional[float] = None
    
    # 27. 准确率计算完成
    accuracy_calculated: Optional[float] = None
    
    # ========== 元数据 ==========
    edge_sample_count: int = 0  # 边侧处理的样本数
    cloud_sample_count: int = 0  # 云侧处理的样本数
    confidence_threshold: float = 0.0  # 置信度阈值
    
    def calculate_durations(self) -> Dict[str, float]:
        """
        计算各阶段的持续时间（毫秒）
        
        Returns:
            各阶段时间字典
        """
        durations = {}
        
        # 1. 边侧推理时间
        if self.edge_inference_start and self.edge_inference_end:
            durations['edge_inference_ms'] = (self.edge_inference_end - self.edge_inference_start) * 1000
        
        # 2. 边侧决策时间
        if self.edge_inference_end and self.edge_decision_complete:
            durations['edge_decision_ms'] = (self.edge_decision_complete - self.edge_inference_end) * 1000
        
        # 3. 上传队列等待时间
        if self.upload_queue_enqueue and self.upload_queue_dequeue:
            durations['upload_queue_wait_ms'] = (self.upload_queue_dequeue - self.upload_queue_enqueue) * 1000
        
        # 4. 上传序列化时间
        if self.upload_serialize_start and self.upload_serialize_end:
            durations['upload_serialize_ms'] = (self.upload_serialize_end - self.upload_serialize_start) * 1000
        
        # 5. 上传网络传输时间
        if self.upload_network_start and self.upload_network_end:
            durations['upload_network_ms'] = (self.upload_network_end - self.upload_network_start) * 1000
        
        # 6. 云侧队列等待时间
        if self.cloud_queue_enqueue and self.cloud_queue_dequeue:
            durations['cloud_queue_wait_ms'] = (self.cloud_queue_dequeue - self.cloud_queue_enqueue) * 1000
        
        # 7. 云侧推理时间
        if self.cloud_inference_start and self.cloud_inference_end:
            durations['cloud_inference_ms'] = (self.cloud_inference_end - self.cloud_inference_start) * 1000
        
        # 8. 云侧序列化时间
        if self.cloud_serialize_start and self.cloud_serialize_end:
            durations['cloud_serialize_ms'] = (self.cloud_serialize_end - self.cloud_serialize_start) * 1000
        
        # 9. 云侧响应发送时间
        if self.cloud_response_send_start and self.cloud_response_send_end:
            durations['cloud_response_send_ms'] = (self.cloud_response_send_end - self.cloud_response_send_start) * 1000
        
        # 10. 下载网络传输时间（从云侧发送到边侧接收）
        if self.cloud_response_send_start and self.edge_response_received:
            durations['download_network_ms'] = (self.edge_response_received - self.cloud_response_send_start) * 1000
        
        # 11. 边侧反序列化时间
        if self.edge_deserialize_start and self.edge_deserialize_end:
            durations['edge_deserialize_ms'] = (self.edge_deserialize_end - self.edge_deserialize_start) * 1000
        
        # 12. 结果缓存存储时间
        if self.edge_deserialize_end and self.edge_cache_stored:
            durations['edge_cache_store_ms'] = (self.edge_cache_stored - self.edge_deserialize_end) * 1000
        
        # 13. 合并处理时间
        if self.merge_start and self.merge_end:
            durations['merge_processing_ms'] = (self.merge_end - self.merge_start) * 1000
        
        # 14. 准确率计算时间
        if self.merge_end and self.accuracy_calculated:
            durations['accuracy_calculation_ms'] = (self.accuracy_calculated - self.merge_end) * 1000
        
        # ========== 综合时间 ==========
        # 15. 边侧总时间（数据到达 -> 决策完成）
        if self.edge_data_arrival and self.edge_decision_complete:
            durations['edge_total_ms'] = (self.edge_decision_complete - self.edge_data_arrival) * 1000
        
        # 16. 云侧总时间（请求接收 -> 响应发送）
        if self.cloud_request_received and self.cloud_response_send_end:
            durations['cloud_total_ms'] = (self.cloud_response_send_end - self.cloud_request_received) * 1000
        
        # 17. 上传总时间（入队 -> 网络发送完成）
        if self.upload_queue_enqueue and self.upload_network_end:
            durations['upload_total_ms'] = (self.upload_network_end - self.upload_queue_enqueue) * 1000
        
        # 18. 下载总时间（云侧发送 -> 边侧缓存）
        if self.cloud_response_send_start and self.edge_cache_stored:
            durations['download_total_ms'] = (self.edge_cache_stored - self.cloud_response_send_start) * 1000
        
        # 19. 端到端延迟（数据到达 -> 准确率计算完成）
        if self.edge_data_arrival and self.accuracy_calculated:
            durations['end_to_end_ms'] = (self.accuracy_calculated - self.edge_data_arrival) * 1000
        
        # 20. 云侧往返时间（上传开始 -> 下载完成）
        if self.upload_queue_enqueue and self.edge_cache_stored:
            durations['cloud_round_trip_ms'] = (self.edge_cache_stored - self.upload_queue_enqueue) * 1000
        
        return durations
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取批次时间剖析摘要
        
        Returns:
            摘要字典
        """
        durations = self.calculate_durations()
        
        return {
            'batch_id': self.batch_id,
            'sample_count': self.sample_count,
            'edge_sample_count': self.edge_sample_count,
            'cloud_sample_count': self.cloud_sample_count,
            'durations': durations
        }


class TimeProfiler:
    """
    时间性能分析器
    
    线程安全的时间追踪系统，用于记录云边端协同推理的每个环节
    """
    
    def __init__(self, enabled: bool = True):
        """
        初始化时间分析器
        
        Args:
            enabled: 是否启用时间追踪（默认True）
        """
        self.enabled = enabled
        self.batch_profiles: Dict[int, BatchTimeProfile] = {}
        self.time_points: List[TimePoint] = []
        self.lock = threading.Lock()
        
        # 全局统计
        self.global_start_time: Optional[float] = None
        self.global_end_time: Optional[float] = None
    
    def start(self):
        """开始全局计时"""
        if not self.enabled:
            return
        
        with self.lock:
            self.global_start_time = time.time()
    
    def stop(self):
        """停止全局计时"""
        if not self.enabled:
            return
        
        with self.lock:
            self.global_end_time = time.time()
    
    def record_event(self, event: str, batch_id: int, sample_count: int = 0, **metadata):
        """
        记录一个时间事件
        
        Args:
            event: 事件名称
            batch_id: 批次ID
            sample_count: 样本数量
            **metadata: 额外元数据
        """
        if not self.enabled:
            return
        
        timestamp = time.time()
        
        with self.lock:
            # 记录时间点
            time_point = TimePoint(
                timestamp=timestamp,
                event=event,
                batch_id=batch_id,
                sample_count=sample_count,
                metadata=metadata
            )
            self.time_points.append(time_point)
            
            # 更新批次剖析
            if batch_id not in self.batch_profiles:
                self.batch_profiles[batch_id] = BatchTimeProfile(
                    batch_id=batch_id,
                    sample_count=sample_count
                )
            
            profile = self.batch_profiles[batch_id]
            
            # 根据事件类型更新对应的时间点
            event_mapping = {
                # 边侧事件
                'edge_data_arrival': 'edge_data_arrival',
                'edge_inference_start': 'edge_inference_start',
                'edge_inference_end': 'edge_inference_end',
                'edge_decision_complete': 'edge_decision_complete',
                
                # 上传事件
                'upload_queue_enqueue': 'upload_queue_enqueue',
                'upload_queue_dequeue': 'upload_queue_dequeue',
                'upload_serialize_start': 'upload_serialize_start',
                'upload_serialize_end': 'upload_serialize_end',
                'upload_network_start': 'upload_network_start',
                'upload_network_end': 'upload_network_end',
                
                # 云侧事件
                'cloud_request_received': 'cloud_request_received',
                'cloud_queue_enqueue': 'cloud_queue_enqueue',
                'cloud_queue_dequeue': 'cloud_queue_dequeue',
                'cloud_inference_start': 'cloud_inference_start',
                'cloud_inference_end': 'cloud_inference_end',
                'cloud_serialize_start': 'cloud_serialize_start',
                'cloud_serialize_end': 'cloud_serialize_end',
                'cloud_response_send_start': 'cloud_response_send_start',
                'cloud_response_send_end': 'cloud_response_send_end',
                
                # 下载事件
                'edge_response_received': 'edge_response_received',
                'edge_deserialize_start': 'edge_deserialize_start',
                'edge_deserialize_end': 'edge_deserialize_end',
                'edge_cache_stored': 'edge_cache_stored',
                
                # 合并事件
                'merge_detected': 'merge_detected',
                'merge_start': 'merge_start',
                'merge_end': 'merge_end',
                'accuracy_calculated': 'accuracy_calculated',
            }
            
            if event in event_mapping:
                setattr(profile, event_mapping[event], timestamp)
            
            # 更新元数据
            if 'edge_sample_count' in metadata:
                profile.edge_sample_count = metadata['edge_sample_count']
            if 'cloud_sample_count' in metadata:
                profile.cloud_sample_count = metadata['cloud_sample_count']
            if 'confidence_threshold' in metadata:
                profile.confidence_threshold = metadata['confidence_threshold']
    
    def get_batch_profile(self, batch_id: int) -> Optional[BatchTimeProfile]:
        """
        获取批次时间剖析
        
        Args:
            batch_id: 批次ID
            
        Returns:
            BatchTimeProfile 或 None
        """
        with self.lock:
            return self.batch_profiles.get(batch_id)
    
    def get_all_batch_profiles(self) -> List[BatchTimeProfile]:
        """获取所有批次时间剖析"""
        with self.lock:
            return list(self.batch_profiles.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            if not self.batch_profiles:
                return {}
            
            # 收集所有批次的持续时间
            all_durations = defaultdict(list)
            
            for profile in self.batch_profiles.values():
                durations = profile.calculate_durations()
                for key, value in durations.items():
                    if value is not None and value >= 0:  # 过滤无效值
                        all_durations[key].append(value)
            
            # 计算统计量
            statistics = {}
            
            for key, values in all_durations.items():
                if values:
                    statistics[key] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'p50': float(np.percentile(values, 50)),
                        'p90': float(np.percentile(values, 90)),
                        'p95': float(np.percentile(values, 95)),
                        'p99': float(np.percentile(values, 99)),
                        'count': len(values)
                    }
            
            # 全局统计
            global_stats = {
                'total_batches': len(self.batch_profiles),
                'total_time_points': len(self.time_points),
            }
            
            if self.global_start_time and self.global_end_time:
                global_stats['total_elapsed_sec'] = self.global_end_time - self.global_start_time
            
            return {
                'global': global_stats,
                'durations': statistics
            }
    
    def print_summary(self):
        """打印时间剖析摘要"""
        if not self.enabled:
            print("[TimeProfiler] 时间追踪未启用")
            return
        
        stats = self.get_statistics()
        
        if not stats:
            print("[TimeProfiler] 没有收集到时间数据")
            return
        
        print(f"\n{'='*80}")
        print(f"时间性能分析报告")
        print(f"{'='*80}")
        
        # 全局统计
        if 'global' in stats:
            global_stats = stats['global']
            print(f"\n全局统计:")
            print(f"  总批次数:           {global_stats.get('total_batches', 0)}")
            print(f"  总时间点数:         {global_stats.get('total_time_points', 0)}")
            if 'total_elapsed_sec' in global_stats:
                print(f"  总耗时:             {global_stats['total_elapsed_sec']:.2f}秒")
        
        # 各阶段时间统计
        if 'durations' in stats:
            durations = stats['durations']
            
            print(f"\n{'='*80}")
            print(f"各阶段时间统计 (毫秒)")
            print(f"{'='*80}")
            print(f"{'阶段':<30} {'平均值':<10} {'中位数':<10} {'P90':<10} {'P99':<10} {'最小值':<10} {'最大值':<10}")
            print(f"{'-'*80}")
            
            # 按类别组织输出
            categories = {
                '边侧推理': ['edge_inference_ms', 'edge_decision_ms', 'edge_total_ms'],
                '上传阶段': ['upload_queue_wait_ms', 'upload_serialize_ms', 'upload_network_ms', 'upload_total_ms'],
                '云侧推理': ['cloud_queue_wait_ms', 'cloud_inference_ms', 'cloud_serialize_ms', 
                           'cloud_response_send_ms', 'cloud_total_ms'],
                '下载阶段': ['download_network_ms', 'edge_deserialize_ms', 'edge_cache_store_ms', 'download_total_ms'],
                '合并阶段': ['merge_processing_ms', 'accuracy_calculation_ms'],
                '综合指标': ['end_to_end_ms', 'cloud_round_trip_ms']
            }
            
            for category, keys in categories.items():
                print(f"\n{category}:")
                for key in keys:
                    if key in durations:
                        d = durations[key]
                        # 格式化阶段名称
                        stage_name = key.replace('_ms', '').replace('_', ' ').title()
                        print(f"  {stage_name:<28} {d['mean']:<10.2f} {d['median']:<10.2f} "
                              f"{d['p90']:<10.2f} {d['p99']:<10.2f} {d['min']:<10.2f} {d['max']:<10.2f}")
        
        print(f"\n{'='*80}\n")
    
    def save_to_json(self, filepath: str):
        """
        保存时间剖析数据到JSON文件
        
        Args:
            filepath: 输出文件路径
        """
        if not self.enabled:
            print("[TimeProfiler] 时间追踪未启用，无法保存")
            return
        
        with self.lock:
            data = {
                'global': {
                    'start_time': self.global_start_time,
                    'end_time': self.global_end_time,
                    'total_batches': len(self.batch_profiles),
                    'total_time_points': len(self.time_points)
                },
                'batch_profiles': [profile.get_summary() for profile in self.batch_profiles.values()],
                'statistics': self.get_statistics()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"[TimeProfiler] 时间剖析数据已保存到: {filepath}")


# 全局单例
_global_profiler: Optional[TimeProfiler] = None


def get_profiler() -> TimeProfiler:
    """获取全局时间分析器单例"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = TimeProfiler(enabled=True)
    return _global_profiler


def init_profiler(enabled: bool = True) -> TimeProfiler:
    """
    初始化全局时间分析器
    
    Args:
        enabled: 是否启用时间追踪
        
    Returns:
        TimeProfiler实例
    """
    global _global_profiler
    _global_profiler = TimeProfiler(enabled=enabled)
    return _global_profiler

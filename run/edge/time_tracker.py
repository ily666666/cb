#!/usr/bin/env python3
"""
端到端时间追踪器
用于详细追踪云边协同推理中的各个时间段
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TimeBreakdown:
    """
    单个样本/批次的时间分解
    
    时间构成：
    T_total = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8
    
    T1: 边侧推理时间（Edge Inference）
    T2: 边侧决策时间（Edge Decision）  
    T3: 边侧序列化时间（Edge Serialization）
    T4: 网络上传时间（Network Upload）
    T5: 云侧接收+反序列化时间（Cloud Receive & Deserialization）
    T6: 云侧队列等待时间（Cloud Queue Wait）
    T7: 云侧推理时间（Cloud Inference）
    T8: 网络下载时间（Network Download）
    """
    batch_id: int
    
    # 时间戳
    data_arrival_time: float = 0.0          # 数据到达边侧时间
    edge_inference_start: float = 0.0       # 边侧推理开始
    edge_inference_end: float = 0.0         # 边侧推理结束
    edge_decision_start: float = 0.0        # 边侧决策开始
    edge_decision_end: float = 0.0          # 边侧决策结束
    edge_serialize_start: float = 0.0       # 边侧序列化开始
    edge_serialize_end: float = 0.0         # 边侧序列化结束
    network_upload_start: float = 0.0       # 网络上传开始
    network_upload_end: float = 0.0         # 网络上传结束
    cloud_receive_start: float = 0.0        # 云侧接收开始
    cloud_receive_end: float = 0.0          # 云侧接收结束
    cloud_queue_enter: float = 0.0          # 进入云侧队列
    cloud_queue_exit: float = 0.0           # 离开云侧队列
    cloud_inference_start: float = 0.0      # 云侧推理开始
    cloud_inference_end: float = 0.0        # 云侧推理结束
    cloud_serialize_start: float = 0.0      # 云侧序列化开始
    cloud_serialize_end: float = 0.0        # 云侧序列化结束
    network_download_start: float = 0.0     # 网络下载开始
    network_download_end: float = 0.0       # 网络下载结束
    result_merge_time: float = 0.0          # 结果合并时间
    
    # 样本信息
    num_samples: int = 0
    is_high_confidence: bool = False        # 是否高置信度（不需要云侧）
    is_timeout: bool = False                # 是否超时
    
    def get_time_breakdown_ms(self) -> Dict[str, float]:
        """
        获取时间分解（毫秒）
        
        Returns:
            时间分解字典
        """
        breakdown = {}
        
        # T1: 边侧推理时间
        if self.edge_inference_end > 0 and self.edge_inference_start > 0:
            breakdown['T1_edge_inference'] = (self.edge_inference_end - self.edge_inference_start) * 1000
        
        # T2: 边侧决策时间
        if self.edge_decision_end > 0 and self.edge_decision_start > 0:
            breakdown['T2_edge_decision'] = (self.edge_decision_end - self.edge_decision_start) * 1000
        
        # T3: 边侧序列化时间
        if self.edge_serialize_end > 0 and self.edge_serialize_start > 0:
            breakdown['T3_edge_serialization'] = (self.edge_serialize_end - self.edge_serialize_start) * 1000
        
        # T4: 网络上传时间
        if self.network_upload_end > 0 and self.network_upload_start > 0:
            breakdown['T4_network_upload'] = (self.network_upload_end - self.network_upload_start) * 1000
        
        # T5: 云侧接收+反序列化时间
        if self.cloud_receive_end > 0 and self.cloud_receive_start > 0:
            breakdown['T5_cloud_receive'] = (self.cloud_receive_end - self.cloud_receive_start) * 1000
        
        # T6: 云侧队列等待时间
        if self.cloud_queue_exit > 0 and self.cloud_queue_enter > 0:
            breakdown['T6_cloud_queue_wait'] = (self.cloud_queue_exit - self.cloud_queue_enter) * 1000
        
        # T7: 云侧推理时间
        if self.cloud_inference_end > 0 and self.cloud_inference_start > 0:
            breakdown['T7_cloud_inference'] = (self.cloud_inference_end - self.cloud_inference_start) * 1000
        
        # T8: 网络下载时间（包含云侧序列化）
        if self.network_download_end > 0 and self.cloud_serialize_start > 0:
            breakdown['T8_network_download'] = (self.network_download_end - self.cloud_serialize_start) * 1000
        
        # 总时间
        if self.network_download_end > 0 and self.data_arrival_time > 0:
            breakdown['T_total_end_to_end'] = (self.network_download_end - self.data_arrival_time) * 1000
        elif self.edge_decision_end > 0 and self.data_arrival_time > 0:
            # 高置信度样本，只有边侧时间
            breakdown['T_total_end_to_end'] = (self.edge_decision_end - self.data_arrival_time) * 1000
        
        return breakdown
    
    def is_complete(self) -> bool:
        """检查时间记录是否完整"""
        if self.is_high_confidence:
            # 高置信度样本只需要边侧时间
            return (self.data_arrival_time > 0 and 
                   self.edge_inference_end > 0 and 
                   self.edge_decision_end > 0)
        else:
            # 低置信度样本需要完整的云边时间
            return (self.data_arrival_time > 0 and 
                   self.edge_inference_end > 0 and 
                   self.network_download_end > 0)


class TimeTracker:
    """
    时间追踪器 - 线程安全
    
    用于追踪每个批次的详细时间信息
    """
    
    def __init__(self):
        self.breakdowns: Dict[int, TimeBreakdown] = {}
        self.lock = threading.Lock()
        
        # 统计信息
        self.completed_breakdowns: List[TimeBreakdown] = []
        self.high_conf_count = 0
        self.low_conf_count = 0
        self.timeout_count = 0
    
    def create_breakdown(self, batch_id: int, num_samples: int = 0) -> TimeBreakdown:
        """
        创建新的时间分解记录
        
        Args:
            batch_id: 批次ID
            num_samples: 样本数量
            
        Returns:
            TimeBreakdown对象
        """
        with self.lock:
            breakdown = TimeBreakdown(
                batch_id=batch_id,
                num_samples=num_samples,
                data_arrival_time=time.time()
            )
            self.breakdowns[batch_id] = breakdown
            return breakdown
    
    def get_breakdown(self, batch_id: int) -> Optional[TimeBreakdown]:
        """
        获取时间分解记录
        
        Args:
            batch_id: 批次ID
            
        Returns:
            TimeBreakdown对象或None
        """
        with self.lock:
            return self.breakdowns.get(batch_id)
    
    def update_edge_inference(self, batch_id: int, start_time: float, end_time: float):
        """更新边侧推理时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].edge_inference_start = start_time
                self.breakdowns[batch_id].edge_inference_end = end_time
    
    def update_edge_decision(self, batch_id: int, start_time: float, end_time: float):
        """更新边侧决策时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].edge_decision_start = start_time
                self.breakdowns[batch_id].edge_decision_end = end_time
    
    def update_edge_serialization(self, batch_id: int, start_time: float, end_time: float):
        """更新边侧序列化时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].edge_serialize_start = start_time
                self.breakdowns[batch_id].edge_serialize_end = end_time
    
    def update_network_upload(self, batch_id: int, start_time: float, end_time: float):
        """更新网络上传时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].network_upload_start = start_time
                self.breakdowns[batch_id].network_upload_end = end_time
    
    def update_cloud_receive(self, batch_id: int, start_time: float, end_time: float):
        """更新云侧接收时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].cloud_receive_start = start_time
                self.breakdowns[batch_id].cloud_receive_end = end_time
    
    def update_cloud_queue(self, batch_id: int, enter_time: float, exit_time: float):
        """更新云侧队列等待时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].cloud_queue_enter = enter_time
                self.breakdowns[batch_id].cloud_queue_exit = exit_time
    
    def update_cloud_inference(self, batch_id: int, start_time: float, end_time: float):
        """更新云侧推理时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].cloud_inference_start = start_time
                self.breakdowns[batch_id].cloud_inference_end = end_time
    
    def update_cloud_serialization(self, batch_id: int, start_time: float, end_time: float):
        """更新云侧序列化时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].cloud_serialize_start = start_time
                self.breakdowns[batch_id].cloud_serialize_end = end_time
    
    def update_network_download(self, batch_id: int, start_time: float, end_time: float):
        """更新网络下载时间"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].network_download_start = start_time
                self.breakdowns[batch_id].network_download_end = end_time
    
    def mark_high_confidence(self, batch_id: int):
        """标记为高置信度样本（不需要云侧）"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].is_high_confidence = True
                self.high_conf_count += 1
    
    def mark_low_confidence(self, batch_id: int):
        """标记为低置信度样本（需要云侧）"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].is_high_confidence = False
                self.low_conf_count += 1
    
    def mark_timeout(self, batch_id: int):
        """标记为超时"""
        with self.lock:
            if batch_id in self.breakdowns:
                self.breakdowns[batch_id].is_timeout = True
                self.timeout_count += 1
    
    def finalize_breakdown(self, batch_id: int):
        """
        完成时间分解记录并移动到已完成列表
        
        Args:
            batch_id: 批次ID
        """
        with self.lock:
            if batch_id in self.breakdowns:
                breakdown = self.breakdowns.pop(batch_id)
                if breakdown.is_complete():
                    self.completed_breakdowns.append(breakdown)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典，包含各个时间段的平均值、中位数、P95等
        """
        with self.lock:
            if not self.completed_breakdowns:
                return {}
            
            # 收集所有时间数据
            time_data = {
                'T1_edge_inference': [],
                'T2_edge_decision': [],
                'T3_edge_serialization': [],
                'T4_network_upload': [],
                'T5_cloud_receive': [],
                'T6_cloud_queue_wait': [],
                'T7_cloud_inference': [],
                'T8_network_download': [],
                'T_total_end_to_end': []
            }
            
            for breakdown in self.completed_breakdowns:
                times = breakdown.get_time_breakdown_ms()
                for key, value in times.items():
                    if key in time_data:
                        time_data[key].append(value)
            
            # 计算统计量
            stats = {}
            for key, values in time_data.items():
                if values:
                    values_array = np.array(values)
                    stats[key] = {
                        'count': len(values),
                        'mean': float(np.mean(values_array)),
                        'median': float(np.median(values_array)),
                        'std': float(np.std(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'p50': float(np.percentile(values_array, 50)),
                        'p90': float(np.percentile(values_array, 90)),
                        'p95': float(np.percentile(values_array, 95)),
                        'p99': float(np.percentile(values_array, 99))
                    }
            
            # 添加样本统计
            stats['summary'] = {
                'total_batches': len(self.completed_breakdowns),
                'high_confidence_batches': self.high_conf_count,
                'low_confidence_batches': self.low_conf_count,
                'timeout_batches': self.timeout_count
            }
            
            return stats
    
    def print_statistics(self):
        """打印详细的时间统计信息"""
        stats = self.get_statistics()
        
        if not stats:
            print("\n[TimeTracker] 没有完整的时间统计数据")
            return
        
        print(f"\n{'='*100}")
        print(f"{'端到端时间分解统计':^100}")
        print(f"{'='*100}")
        
        # 打印摘要
        summary = stats.get('summary', {})
        print(f"\n【统计摘要】")
        print(f"  总批次数:           {summary.get('total_batches', 0)}")
        print(f"  高置信度批次:       {summary.get('high_confidence_batches', 0)}")
        print(f"  低置信度批次:       {summary.get('low_confidence_batches', 0)}")
        print(f"  超时批次:           {summary.get('timeout_batches', 0)}")
        
        # 定义时间段名称映射
        time_names = {
            'T1_edge_inference': 'T1: 边侧推理',
            'T2_edge_decision': 'T2: 边侧决策',
            'T3_edge_serialization': 'T3: 边侧序列化',
            'T4_network_upload': 'T4: 网络上传',
            'T5_cloud_receive': 'T5: 云侧接收',
            'T6_cloud_queue_wait': 'T6: 云侧队列等待',
            'T7_cloud_inference': 'T7: 云侧推理',
            'T8_network_download': 'T8: 网络下载',
            'T_total_end_to_end': '总时间: 端到端'
        }
        
        # 打印详细时间统计
        print(f"\n{'='*100}")
        print(f"{'时间段':<25} {'样本数':<10} {'平均值':<12} {'中位数':<12} {'P95':<12} {'P99':<12}")
        print(f"{'-'*100}")
        
        for key in ['T1_edge_inference', 'T2_edge_decision', 'T3_edge_serialization', 
                    'T4_network_upload', 'T5_cloud_receive', 'T6_cloud_queue_wait',
                    'T7_cloud_inference', 'T8_network_download', 'T_total_end_to_end']:
            if key in stats:
                s = stats[key]
                name = time_names.get(key, key)
                print(f"{name:<25} {s['count']:<10} {s['mean']:<12.2f} {s['median']:<12.2f} "
                      f"{s['p95']:<12.2f} {s['p99']:<12.2f}")
        
        print(f"{'='*100}")
        
        # 打印时间占比分析（仅针对低置信度样本）
        if 'T_total_end_to_end' in stats and stats['T_total_end_to_end']['count'] > 0:
            print(f"\n【时间占比分析】（基于平均值）")
            total_time = stats['T_total_end_to_end']['mean']
            
            components = [
                ('T1_edge_inference', 'T1: 边侧推理'),
                ('T2_edge_decision', 'T2: 边侧决策'),
                ('T3_edge_serialization', 'T3: 边侧序列化'),
                ('T4_network_upload', 'T4: 网络上传'),
                ('T5_cloud_receive', 'T5: 云侧接收'),
                ('T6_cloud_queue_wait', 'T6: 云侧队列等待'),
                ('T7_cloud_inference', 'T7: 云侧推理'),
                ('T8_network_download', 'T8: 网络下载')
            ]
            
            for key, name in components:
                if key in stats:
                    time_val = stats[key]['mean']
                    percentage = (time_val / total_time * 100) if total_time > 0 else 0
                    bar_length = int(percentage / 2)  # 每2%一个字符
                    bar = '█' * bar_length
                    print(f"  {name:<25} {time_val:>8.2f}ms ({percentage:>5.1f}%) {bar}")
            
            print(f"  {'-'*80}")
            print(f"  {'总计':<25} {total_time:>8.2f}ms (100.0%)")
        
        print(f"\n{'='*100}\n")

#!/usr/bin/env python3
"""
边侧协同推理
功能：使用学生模型进行边侧推理，低置信度样本发送到云侧进行深度推理
"""
import os
import sys
import argparse
import time
import pickle
import threading
import torch
import torch.nn.functional as F
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fed.project import create_model_by_type
from run.network_utils import NetworkEdge
from run.edge.data_sources import DataSourceFactory
from run.edge.async_pipeline import (
    ThreadSafeQueue, ResultCache, PerformanceStats,
    InferenceThread, UploadThread, ReceiveThread, MergeThread,
    GlobalTimeTracker
)


class BatchLoadingDataset(torch.utils.data.IterableDataset):
    """边加载边推理的数据集 - 从文件夹中逐个加载批次文件"""
    def __init__(self, batch_files, dataset_type, file_format='pkl'):
        self.batch_files = batch_files
        self.dataset_type = dataset_type
        self.file_format = file_format
        
        # 标签映射（仅用于 pkl 格式的字典数据）
        if dataset_type == 'rml2016':
            # 修复：使用与readdata_rml2016.py相同的顺序
            mod_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
            self.label_map = {mod: idx for idx, mod in enumerate(mod_types)}
        elif dataset_type == 'link11':
            emitter_types = ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']
            self.label_map = {emitter: idx for idx, emitter in enumerate(emitter_types)}
        elif dataset_type == 'radar':
            self.label_map = None
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    def __iter__(self):
        """逐个加载批次文件并生成样本"""
        sample_count = 0
        
        if self.file_format == 'pkl':
            # PKL 格式：字典格式 {(class, snr): signal_array}
            for batch_idx, batch_file in enumerate(self.batch_files):
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                
                # 调试：打印第一个批次的信息
                if batch_idx == 0:
                    print(f"[调试] 第一个批次文件: {os.path.basename(batch_file)}")
                    print(f"[调试] 批次数据类型: {type(batch_data)}")
                
                # 遍历批次中的所有样本
                for key, signal_array in batch_data.items():
                    for signal in signal_array:
                        # 转换为复数（I/Q格式）
                        if signal.shape[0] == 2:
                            signal_complex = signal[0] + 1j * signal[1]
                            signal_tensor = torch.from_numpy(signal_complex).cfloat()
                        else:
                            signal_tensor = torch.from_numpy(signal).float()
                        
                        # 获取标签
                        if self.dataset_type == 'rml2016':
                            label = self.label_map[key[0]]
                        elif self.dataset_type == 'link11':
                            label = self.label_map[key[0]]
                        
                        sample_count += 1
                        yield signal_tensor, label
                
                del batch_data
        
        elif self.file_format == 'mat':
            # MAT 格式：radar 数据 X_batch: (2, 500, N), Y_batch: (1, N)
            import h5py
            
            for batch_idx, batch_file in enumerate(self.batch_files):
                with h5py.File(batch_file, 'r') as f:
                    X_batch = np.array(f['X_batch'])
                    Y_batch = np.array(f['Y_batch']).flatten()
                
                if batch_idx == 0:
                    print(f"[调试] X_batch 形状: {X_batch.shape}")
                    print(f"[调试] Y_batch 形状: {Y_batch.shape}")
                
                num_samples = X_batch.shape[2]
                for i in range(num_samples):
                    signal = X_batch[:, :, i]
                    signal_complex = signal[0] + 1j * signal[1]
                    signal_tensor = torch.from_numpy(signal_complex).cfloat()
                    label = int(Y_batch[i]) - 1
                    
                    sample_count += 1
                    yield signal_tensor, label
                
                del X_batch, Y_batch


class LazyDataset(torch.utils.data.Dataset):
    """惰性加载数据集 - 数据已在内存中"""
    def __init__(self, raw_data, dataset_type):
        self.dataset_type = dataset_type
        self.raw_data = raw_data
        
        # 标签映射
        if dataset_type == 'rml2016':
            # 修复：使用与readdata_rml2016.py相同的顺序
            mod_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
            self.label_map = {mod: idx for idx, mod in enumerate(mod_types)}
        elif dataset_type == 'link11':
            emitter_types = ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']
            self.label_map = {emitter: idx for idx, emitter in enumerate(emitter_types)}
        elif dataset_type == 'radar':
            # Radar 数据集: 7个类别 (0-6)
            # 数据已经是数字标签,不需要映射
            self.label_map = {i: i for i in range(7)}
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        self._index_map = None
        self._total_samples = None
    
    def _build_index(self):
        """延迟构建索引映射"""
        if self._index_map is None:
            self._index_map = []
            for key, signal_array in self.raw_data.items():
                for local_idx in range(len(signal_array)):
                    self._index_map.append((key, local_idx))
            self._total_samples = len(self._index_map)
    
    def __len__(self):
        if self._total_samples is None:
            self._total_samples = sum(len(signal_array) for signal_array in self.raw_data.values())
        return self._total_samples
    
    def __getitem__(self, idx):
        if self._index_map is None:
            self._build_index()
        
        key, local_idx = self._index_map[idx]
        signal = self.raw_data[key][local_idx]
        
        # 转换为复数（I/Q格式）
        if signal.shape[0] == 2:
            signal_complex = signal[0] + 1j * signal[1]
            signal_tensor = torch.from_numpy(signal_complex).cfloat()
        else:
            signal_tensor = torch.from_numpy(signal).float()
        
        # 获取标签
        if self.dataset_type == 'rml2016':
            label = self.label_map[key[0]]
        elif self.dataset_type == 'link11':
            label = self.label_map[key[0]]
        elif self.dataset_type == 'radar':
            # Radar 数据集: key 就是标签 (0-6)
            label = key
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
        
        return signal_tensor, label


def run_async_collaborative_edge(args):
    """运行异步流水线云边协同推理边侧"""
    print(f"\n{'='*70}")
    print(f"[边侧] 异步流水线云边协同推理启动")
    print(f"{'='*70}")
    print(f"云侧: {args.cloud_host}")
    print(f"  ZeroMQ PULL 端口: {args.zmq_pull_port}")
    print(f"  ZeroMQ PUSH 端口: {args.zmq_push_port}")
    print(f"数据集: {args.dataset_type}")
    print(f"边侧模型类型: {args.edge_model if args.edge_model else f'real_resnet20_{args.dataset_type}'}")
    print(f"边侧模型路径: {args.edge_model_path}")
    print(f"置信度阈值: {args.thresholds}")
    print(f"上传队列大小: {args.upload_queue_size}")
    print(f"云侧超时: {args.cloud_timeout}秒")
    print(f"合并间隔: {args.merge_interval}秒")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 检查边侧模型路径
    if not args.edge_model_path or not os.path.exists(args.edge_model_path):
        print(f"错误: 边侧模型路径无效: {args.edge_model_path}")
        print("请使用 --edge_model_path 指定训练好的学生模型路径")
        return
    
    # 添加模块别名支持
    dataset_modules = [
        'readdata_rml2016', 'readdata_radar', 'readdata_radioml',
        'readdata_reii', 'readdata_25', 'readdata_link11'
    ]
    
    for module_name in dataset_modules:
        if module_name not in sys.modules:
            try:
                module = __import__(f'utils.{module_name}', fromlist=[module_name])
                sys.modules[module_name] = module
            except Exception:
                pass
    
    # 加载边侧模型（学生模型）
    print("[边侧] 加载学生模型...")
    if args.edge_model:
        edge_model_type = args.edge_model
    else:
        edge_model_type = f'real_resnet20_{args.dataset_type}'
    
    print(f"[边侧] 模型类型: {edge_model_type}")
    edge_model = create_model_by_type(edge_model_type, args.num_classes, args.dataset_type)
    
    try:
        checkpoint = torch.load(args.edge_model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型出错: {e}")
        
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in dataset_modules:
                    module = f'utils.{module}'
                return super().find_class(module, name)
        
        with open(args.edge_model_path, 'rb') as f:
            checkpoint = CPU_Unpickler(f).load()
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        edge_model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        edge_model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        edge_model.load_state_dict(checkpoint['model'])
    else:
        edge_model.load_state_dict(checkpoint)
    
    edge_model = edge_model.to(device)
    edge_model.eval()
    print("[边侧] 学生模型加载完成\n")
    
    # 使用 DataSourceFactory 创建数据源
    print("[边侧] 准备推理数据...")
    
    # 验证参数
    if args.data_source == 'local' and not args.data_path:
        print("错误: local 模式必须指定 --data_path 参数")
        return
    
    try:
        # 使用工厂创建数据源
        data_source = DataSourceFactory.create(args)
        print(f"[边侧] 数据源创建成功: {args.data_source} 模式\n")
    except Exception as e:
        print(f"错误: 创建数据源失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 解析阈值列表
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    print(f"[边侧] 将测试以下置信度阈值: {thresholds}\n")
    
    # 创建网络边侧
    network_edge = NetworkEdge(args.edge_id, args.cloud_host, args.zmq_pull_port, rate_limit_mbps=args.rate_limit)
    
    # 初始化 ZeroMQ 连接（异步模式）
    print(f"[边侧] 初始化 ZeroMQ 连接...")
    network_edge._init_zmq(push_port=args.zmq_pull_port, pull_port=args.zmq_push_port)
    
    # 对每个阈值进行评估
    all_results = []
    
    try:
        for threshold in thresholds:
            start_time = time.time()
            print(f"\n{'='*70}")
            print(f"[边侧] 异步模式评估阈值 T = {threshold:.2f}")
            print(f"{'='*70}")
            
            # 创建共享数据结构
            upload_queue = ThreadSafeQueue(maxsize=args.upload_queue_size)
            result_cache = ResultCache()
            stats_collector = PerformanceStats()
            
            # 创建全局时间追踪器
            time_tracker = GlobalTimeTracker()
            time_tracker.start_global()
            
            # 创建停止事件（用于优雅关闭）
            stop_event = threading.Event()
            
            # 初始化并启动4个线程
            inference_thread = InferenceThread(
                model=edge_model,
                data_loader=data_source,
                threshold=threshold,
                upload_queue=upload_queue,
                result_cache=result_cache,
                device=device,
                stats_collector=stats_collector,
                stop_event=stop_event,
                time_tracker=time_tracker
            )
            
            upload_thread = UploadThread(
                upload_queue=upload_queue,
                network_edge=network_edge,
                dataset_type=args.dataset_type,
                num_classes=args.num_classes,
                result_cache=result_cache,
                stats_collector=stats_collector,
                stop_event=stop_event,
                time_tracker=time_tracker
            )
            
            receive_thread = ReceiveThread(
                network_edge=network_edge,
                result_cache=result_cache,
                stats_collector=stats_collector,
                stop_event=stop_event,
                receive_timeout=0.5,
                time_tracker=time_tracker
            )
            
            merge_thread = MergeThread(
                result_cache=result_cache,
                stats_collector=stats_collector,
                stop_event=stop_event,
                merge_interval=args.merge_interval,
                cloud_timeout=args.cloud_timeout
            )
            
            # 启动所有线程
            print(f"[边侧] 启动异步流水线线程...")
            inference_thread.start()
            upload_thread.start()
            receive_thread.start()
            merge_thread.start()
            
            print(f"[边侧] 所有线程已启动，开始异步推理...\n")
            
            # 主线程监控和协调逻辑
            try:
                # 等待推理线程完成（数据处理完毕）
                inference_thread.join()
                print(f"\n[边侧] 推理线程已完成")
                
                # 等待上传队列清空
                print(f"[边侧] 等待上传队列清空...")
                while not upload_queue.empty():
                    time.sleep(0.1)
                print(f"[边侧] 上传队列已清空")
                
                # 等待所有批次合并完成
                print(f"[边侧] 等待结果合并完成...")
                max_wait_time = args.cloud_timeout + 5  # 额外等待5秒
                wait_start = time.time()
                while result_cache.get_cache_size() > 0:
                    if time.time() - wait_start > max_wait_time:
                        print(f"[边侧] 等待超时，强制结束")
                        break
                    time.sleep(0.1)
                print(f"[边侧] 结果合并完成")
                
            except KeyboardInterrupt:
                print(f"\n[边侧] 收到中断信号，开始优雅关闭...")
            
            finally:
                # 优雅关闭所有线程
                print(f"\n[边侧] 停止所有线程...")
                stop_event.set()
                end_time = time.time()
                
                # 等待所有线程完成（带超时）
                threads = [inference_thread, upload_thread, receive_thread, merge_thread]
                for thread in threads:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        print(f"[边侧] 警告: {thread.name} 未能在超时时间内停止")
                
                print(f"[边侧] 所有线程已停止\n")
                print(f"耗时{end_time-start_time}")
            # 完成统计
            stats_collector.finalize()
            
            # 结束全局计时
            time_tracker.end_global()
            
            # 获取统计摘要
            summary = stats_collector.get_summary()
            
            # # 获取时间分解
            # time_summary = time_tracker.get_summary()
            
            # 构造结果
            results = {
                'threshold': threshold,
                'overall_accuracy': summary['accuracy'],
                'edge_accuracy': 0.0,  # 异步模式不单独统计边侧准确率
                'cloud_accuracy': 0.0,  # 异步模式不单独统计云侧准确率
                'edge_total': summary['edge_samples'],
                'cloud_total': summary['cloud_samples'],
                'cloud_ratio': summary['cloud_ratio'],
                'avg_edge_time_ms': summary['avg_edge_latency_ms'],
                'avg_cloud_time_ms': summary['avg_cloud_latency_ms'],
                'total_samples': summary['total_samples'],
                'total_time_sec': summary['elapsed_time_sec'],
                'throughput': summary['throughput_samples_per_sec'],
                'avg_end_to_end_latency_ms': summary['avg_end_to_end_latency_ms'],
                'avg_upload_queue_size': summary['avg_upload_queue_size']
            }
            
            all_results.append(results)
            
            # # ========== 统一的性能报告 ==========
            # print(f"\n{'='*70}")
            # print(f"云边协同推理性能报告 (阈值 T = {threshold:.2f})")
            # print(f"{'='*70}")
            # print(f"\n【关键性能指标】")
            # print(f"  边侧推理:         平均 {summary['avg_edge_latency_ms']:.2f}ms")
            # print(f"  云侧推理:         平均 {summary['avg_cloud_latency_ms']:.2f}ms")
            # print(f"  端到端延迟:       平均 {summary['avg_end_to_end_latency_ms']:.2f}ms")
            # print(f"\n【样本分布】")
            # print(f"  总样本数:         {summary['total_samples']}")
            # print(f"  边侧处理:         {summary['edge_samples']} ({100*(1-summary['cloud_ratio']):.2f}%)")
            # print(f"  云侧处理:         {summary['cloud_samples']} ({summary['cloud_ratio']*100:.2f}%)")
            # print(f"\n【准确率与吞吐量】")
            # print(f"  整体准确率:       {summary['accuracy']*100:.2f}% ({summary['correct_predictions']}/{summary['total_samples']})")
            # print(f"  总耗时:           {summary['elapsed_time_sec']:.2f}秒")
            # print(f"  吞吐量:           {summary['throughput_samples_per_sec']:.2f} 样本/秒")
            
            # # ========== 新增：详细时间分解 ==========
            # if 'time_breakdown' in summary:
            #     breakdown = summary['time_breakdown']
                
            #     # 计算总耗时（秒）
            #     cloud_samples = summary['cloud_samples']
            #     total_samples = summary['total_samples']
            #     actual_elapsed = summary['elapsed_time_sec']
                
            #     print(f"\n【详细时间分解 - 单样本平均】（仅低置信度样本）")
            #     print(f"  T1: 边侧推理      {breakdown['T1_edge_inference']:.2f}ms")
            #     print(f"  T2: 边侧决策      {breakdown['T2_edge_decision']:.2f}ms")
            #     print(f"  T3: 边侧序列化    {breakdown['T3_edge_serialize']:.2f}ms")
            #     print(f"  T4: 网络上传      {breakdown['T4_network_upload']:.2f}ms")
            #     print(f"  T5: 云侧接收      {breakdown['T5_cloud_receive']:.2f}ms")
            #     print(f"  T6: 云侧队列等待  {breakdown['T6_cloud_queue_wait']:.2f}ms")
            #     print(f"  T7: 云侧推理      {breakdown['T7_cloud_inference']:.2f}ms")
            #     print(f"  T8: 网络下载      {breakdown['T8_network_download']:.2f}ms")
                
            #     # ========== 关键洞察：实际时间构成分析 ==========
            #     print(f"\n【性能瓶颈分析 - 实际总耗时 {actual_elapsed:.2f}秒】")
            #     print(f"")
                
            #     # 在异步流水线中，我们需要分析实际的时间构成
            #     # 不是把单样本时间乘以样本数（那是串行的算法）
                
            #     # 1. 边侧推理：所有样本都要经过边侧
            #     edge_per_sample_ms = breakdown['T1_edge_inference'] + breakdown['T2_edge_decision']
                
            #     # 2. 云侧处理：只有低置信度样本需要云侧
            #     cloud_per_sample_ms = (breakdown['T3_edge_serialize'] + breakdown['T4_network_upload'] + 
            #                           breakdown['T5_cloud_receive'] + breakdown['T6_cloud_queue_wait'] + 
            #                           breakdown['T7_cloud_inference'] + breakdown['T8_network_download'])
                
            #     # 计算理论吞吐量
            #     edge_throughput = 1000 / edge_per_sample_ms if edge_per_sample_ms > 0 else 0  # 样本/秒
            #     cloud_throughput = 1000 / cloud_per_sample_ms if cloud_per_sample_ms > 0 else 0  # 样本/秒
                
            #     print(f"  流水线性能：")
            #     print(f"    - 边侧单样本耗时：{edge_per_sample_ms:.2f}ms")
            #     print(f"    - 边侧理论吞吐量：{edge_throughput:.0f} 样本/秒")
            #     print(f"    - 云侧单样本耗时：{cloud_per_sample_ms:.2f}ms")
            #     print(f"    - 云侧理论吞吐量：{cloud_throughput:.0f} 样本/秒")
            #     print(f"")
                
            #     # 实际性能
            #     print(f"  实际性能：")
            #     print(f"    - 实际总耗时：{actual_elapsed:.2f}秒")
            #     print(f"    - 实际吞吐量：{summary['throughput_samples_per_sec']:.0f} 样本/秒")
            #     print(f"    - 总样本数：{total_samples}")
            #     print(f"    - 云侧样本数：{cloud_samples} ({summary['cloud_ratio']*100:.2f}%)")
            #     print(f"")
                
            #     # 瓶颈判断：比较实际吞吐量和理论吞吐量
            #     actual_throughput = summary['throughput_samples_per_sec']
            #     throughput_ratio = actual_throughput / edge_throughput if edge_throughput > 0 else 0
                
            #     print(f"  瓶颈识别：")
            #     if throughput_ratio < 0.5:
            #         print(f"    ⚠️  边侧推理是主要瓶颈")
            #         print(f"        实际吞吐量仅为边侧理论吞吐量的 {throughput_ratio*100:.1f}%")
            #         print(f"        优化建议：使用更快的边侧模型、增加批处理大小、或使用GPU加速")
            #     elif cloud_samples > 0 and summary['cloud_ratio'] > 0.1:
            #         print(f"    ⚠️  云侧调用影响性能")
            #         print(f"        {cloud_samples} 个样本需要云侧处理 ({summary['cloud_ratio']*100:.2f}%)")
            #         print(f"        云侧单样本耗时：{cloud_per_sample_ms:.2f}ms（是边侧的 {cloud_per_sample_ms/edge_per_sample_ms:.1f}x）")
            #         print(f"        优化建议：提高置信度阈值减少云侧调用，或优化云侧推理速度")
            #     else:
            #         print(f"    ✓  流水线并行效果良好")
            #         print(f"        实际吞吐量达到边侧理论吞吐量的 {throughput_ratio*100:.1f}%")
                
            #     print(f"")
            #     print(f"  云侧处理各阶段时间占比（单样本平均）：")
            #     if cloud_samples > 0:
            #         cloud_stages = [
            #             ('T3: 边侧序列化', breakdown['T3_edge_serialize']),
            #             ('T4: 网络上传', breakdown['T4_network_upload']),
            #             ('T5: 云侧接收', breakdown['T5_cloud_receive']),
            #             ('T6: 云侧队列等待', breakdown['T6_cloud_queue_wait']),
            #             ('T7: 云侧推理', breakdown['T7_cloud_inference']),
            #             ('T8: 网络下载', breakdown['T8_network_download']),
            #         ]
                    
            #         for name, time_ms in cloud_stages:
            #             percentage = (time_ms / cloud_per_sample_ms) * 100 if cloud_per_sample_ms > 0 else 0
            #             bar_length = int(percentage / 2)
            #             bar = '█' * bar_length
            #             print(f"    {name:<18} {time_ms:>7.2f}ms  {percentage:>5.1f}%  {bar}")
            #     else:
            #         print(f"    （无云侧样本）")
                
            #     # ========== 新增：实际时间分解（墙上时钟时间） ==========
            #     if time_summary:
            #         print(f"")
            #         print(f"{'='*70}")
            #         print(f"【实际时间分解 - 墙上时钟时间】")
            #         print(f"{'='*70}")
            #         print(f"")
            #         print(f"  总耗时：{time_summary['total_elapsed_sec']:.2f}秒")
            #         print(f"")
            #         print(f"  各阶段实际工作时间：")
                    
            #         stage_names = {
            #             'edge_inference': '边侧推理',
            #             'edge_to_cloud_transfer': '边侧→云侧传输',
            #             'cloud_inference': '云侧推理',
            #             'cloud_to_edge_transfer': '云侧→边侧传输',
            #         }
                    
            #         stage_times = time_summary['stage_times_sec']
            #         stage_percentages = time_summary['stage_percentages']
                    
            #         for stage_key, stage_name in stage_names.items():
            #             time_sec = stage_times.get(stage_key, 0.0)
            #             percentage = stage_percentages.get(stage_key, 0.0)
            #             bar_length = int(percentage / 2)
            #             bar = '█' * bar_length
            #             print(f"    {stage_name:<18} {time_sec:>7.2f}秒  {percentage:>5.1f}%  {bar}")
                    
            #         print(f"")
            #         print(f"  说明：")
            #         print(f"    - 各阶段时间可能重叠（异步并行执行）")
            #         print(f"    - 百分比表示该阶段占总时间的比例")
            #         print(f"    - 边侧推理通常占主导地位（处理所有样本）")
            #         print(f"    - 云侧相关时间仅针对低置信度样本")
            #         print(f"{'='*70}")
            
            # print(f"{'='*70}\n")
        
        # 保存结果
        if args.save_results:
            import json
            os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
            with open(args.save_results, 'w', encoding='utf-8') as f:
                json.dump({
                    'edge_id': args.edge_id,
                    'mode': 'async',
                    'threshold_results': all_results,
                    'config': {
                        'dataset_type': args.dataset_type,
                        'num_classes': args.num_classes,
                        'edge_model_path': args.edge_model_path,
                        'thresholds': thresholds,
                        'upload_queue_size': args.upload_queue_size,
                        'cloud_timeout': args.cloud_timeout,
                        'merge_interval': args.merge_interval
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"\n[边侧] 结果已保存到: {args.save_results}")
        
        # 显示数据源统计信息
        # print(f"\n{'='*70}")
        # print(f"[边侧] 数据源统计信息")
        # print(f"{'='*70}")
        # data_stats = data_source.get_stats()
        # for key, value in data_stats.items():
        #     if key == 'samples_per_second':
        #         print(f"吞吐量:             {value:.2f} 样本/秒")
        #     elif key == 'elapsed_time':
        #         print(f"总耗时:             {value:.2f}秒")
        #     elif key == 'total_batches':
        #         print(f"总批次数:           {value}")
        #     elif key == 'total_samples':
        #         print(f"总样本数:           {value}")
        #     elif key == 'port':
        #         print(f"数据端口:           {value}")
        #     elif key == 'response_port':
        #         print(f"响应端口:           {value}")
        #     elif key == 'timeout':
        #         print(f"超时设置:           {value}ms")
        #     else:
        #         print(f"{key}: {value}")
        # print(f"{'='*70}\n")
        
        # ✅ 发送推理完成标志给端侧（仅 device 模式）
        if args.data_source == 'device' and hasattr(data_source, 'send_inference_complete'):
            data_source.send_inference_complete()
        
        # 打印总结
        print(f"\n{'='*70}")
        print(f"[边侧] 异步模式评估总结")
        print(f"{'='*70}")
        print(f"{'阈值':<8} {'整体准确率':<12} {'云侧调用率':<12} {'吞吐量(样本/秒)':<18} {'端到端延迟(ms)':<18}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['threshold']:<8.2f} {r['overall_accuracy']*100:<12.2f} {r['cloud_ratio']:<12.2%} "
                  f"{r['throughput']:<18.2f} {r['avg_end_to_end_latency_ms']:<18.2f}")
        
        if all_results:
            best_result = max(all_results, key=lambda x: x['overall_accuracy'] - 10 * x['cloud_ratio'])
            print(f"\n最佳阈值（平衡点）: T = {best_result['threshold']:.2f}")
            print(f"  整体准确率:       {best_result['overall_accuracy']*100:.2f}%")
            print(f"  云侧调用率:       {best_result['cloud_ratio']:.2%}")
            print(f"  吞吐量:           {best_result['throughput']:.2f} 样本/秒")
        
        print(f"{'='*70}\n")
        
    finally:
        # 确保数据源资源被正确清理
        print("\n[边侧] 清理资源...")
        data_source.close()
        
        # 关闭 ZeroMQ 连接（异步模式）
        if 'network_edge' in locals():
            network_edge.close_zmq()


def main():
    parser = argparse.ArgumentParser(description='云边协同推理 - 边侧')

    # 基本参数
    parser.add_argument('--edge_id', type=int, default=0,
                       help='边侧设备ID (默认: 0)')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['rml2016', 'radar', 'link11'],
                       help='数据集类型')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='分类类别数')
    
    # 模型参数
    parser.add_argument('--edge_model_path', type=str, required=True,
                       help='边侧学生模型路径')
    parser.add_argument('--edge_model', type=str, default=None,
                       help='边侧模型类型（可选，如 real_resnet20_link11_h）。'
                            '如果不指定，默认使用 real_resnet20_{dataset_type}')
    
    # 数据源参数
    parser.add_argument('--data_source', type=str, default='local',
                       choices=['device', 'local'],
                       help='数据源类型: device (从端侧设备接收) 或 local (从本地文件加载) (默认: local)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='测试数据路径（文件或文件夹），local 模式必需')
    parser.add_argument('--device_port', type=int, default=7777,
                       help='设备数据接收端口 (device 模式使用，默认: 7777)')
    parser.add_argument('--device_response_port', type=int, default=7778,
                       help='设备响应发送端口 (device 模式使用，默认: 7778)')
    parser.add_argument('--device_timeout', type=int, default=300000,
                       help='设备数据接收超时时间（毫秒）(device 模式使用，默认: 300000)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小 (默认: 128)')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='限制处理的批次数（用于快速测试）')
    
    # 网络参数
    parser.add_argument('--cloud_host', type=str, default='localhost',
                       help='云侧主机地址 (默认: localhost)')
    parser.add_argument('--zmq_pull_port', type=int, default=5555,
                       help='云侧 ZeroMQ PULL 端口（云侧接收请求，默认：5555）')
    parser.add_argument('--zmq_push_port', type=int, default=5556,
                       help='云侧 ZeroMQ PUSH 端口（云侧发送响应，默认：5556）')
    parser.add_argument('--rate_limit', type=float, default=10.0,
                       help='网络速率限制（MB/s，默认10）')
    
    # 协同推理参数
    parser.add_argument('--thresholds', type=str, default='0.5,0.6,0.7,0.8,0.9',
                       help='置信度阈值列表，逗号分隔 (默认: 0.5,0.6,0.7,0.8,0.9)')
    
    # 异步模式参数
    parser.add_argument('--upload_queue_size', type=int, default=100,
                       help='上传队列大小（默认: 100）')
    parser.add_argument('--cloud_timeout', type=float, default=30.0,
                       help='云侧响应超时时间（秒）（默认: 30.0）')
    parser.add_argument('--merge_interval', type=float, default=0.01,
                       help='合并线程检查间隔（秒）（默认: 0.01）')
    
    # 输出参数
    parser.add_argument('--save_results', type=str, default=None,
                       help='保存结果的JSON文件路径')
    
    args = parser.parse_args()
    
    # 只使用异步流水线模式
    print("[边侧] 使用异步流水线模式")
    run_async_collaborative_edge(args)


if __name__ == '__main__':
    main()

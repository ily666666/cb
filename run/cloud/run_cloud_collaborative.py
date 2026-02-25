#!/usr/bin/env python3
"""
云侧协同推理服务器（多线程版本）
功能：接收边侧设备的深度推理请求，使用教师模型进行推理并返回结果
支持多个边侧同时连接和推理
"""
import os
import sys
import socket
import pickle
import struct
import time
import argparse
import torch
import numpy as np
import threading
import queue

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fed.project import create_model_by_type


def handle_zmq_requests(zmq_pull_socket, cloud_model, device, args, request_counter, counter_lock, 
                       inference_queue=None, stop_event=None, response_sender=None):
    """
    处理 ZeroMQ 推理请求（在独立线程中运行）
    
    使用 ZeroMQ PULL socket 接收边侧的推理请求。
    
    Args:
        zmq_pull_socket: ZeroMQ PULL socket
        cloud_model: 云侧模型
        device: 计算设备
        args: 命令行参数
        request_counter: 全局请求计数器（列表，用于线程安全）
        counter_lock: 计数器锁
        inference_queue: 推理队列（异步模式）
        stop_event: 停止事件
    """
    def _log(level, message):
        """统一的日志输出格式"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [ZMQ-Handler] {message}")
    
    _log("INFO", "ZeroMQ 请求处理线程启动")
    
    # 判断是否为异步模式
    async_mode = (inference_queue is not None)
    if async_mode:
        _log("INFO", "异步模式已启用")
    
    # 尝试导入 MessagePack
    try:
        import msgpack
        import msgpack_numpy as m
        m.patch()
        use_msgpack = True
        _log("INFO", "使用 MessagePack 反序列化")
    except ImportError:
        use_msgpack = False
        _log("INFO", "使用 Pickle 反序列化")
    
    try:
        while not (stop_event and stop_event.is_set()):
            try:
                # 使用 ZeroMQ 接收请求（带超时）
                if zmq_pull_socket.poll(100):  # 100ms 超时
                    # ========== T5: 云侧接收开始 ==========
                    cloud_receive_start = time.time()
                    
                    data = zmq_pull_socket.recv()
                    
                    # 反序列化请求（支持 MessagePack 和 Pickle）
                    try:
                        if use_msgpack:
                            request = msgpack.unpackb(data, raw=False)
                        else:
                            request = pickle.loads(data)
                    except Exception as e:
                        # 如果 MessagePack 失败，尝试 Pickle
                        print(f"[云侧] MessagePack 反序列化失败，尝试 Pickle: {e}")
                        request = pickle.loads(data)
                    
                    # ========== T5: 云侧接收结束 ==========
                    cloud_receive_end = time.time()
                    
                    # 更新全局请求计数
                    with counter_lock:
                        request_counter[0] += 1
                        current_count = request_counter[0]
                    
                    # 每100个请求打印一次信息（减少日志量）
                    if current_count % 100 == 0:
                        _log("PROGRESS", f"已接收 {current_count} 个推理请求")
                    
                    request_type = request.get('type', 'unknown')
                    
                    if request_type == 'cloud_inference':
                        inputs = request.get('inputs')
                        labels = request.get('labels', None)  # 获取真实标签（如果有）
                        batch_id = request.get('batch_id', None)
                        sample_indices = request.get('sample_indices', None)
                        dataset_type = request.get('dataset_type', 'unknown')
                        edge_id = request.get('edge_id', 'unknown')  # 获取 edge_id
                        edge_send_time = request.get('edge_send_time', 0.0)  # 获取边侧发送时间
                        batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs)
                        
                        if inputs is None:
                            _log("ERROR", "缺少输入数据")
                            continue
                        
                        # 异步模式：将请求放入队列
                        if async_mode:
                            from async_cloud_pipeline import InferenceRequest
                            
                            # 如果没有batch_id，使用请求计数作为batch_id
                            if batch_id is None:
                                batch_id = current_count
                            
                            # 如果没有sample_indices，使用默认索引
                            if sample_indices is None:
                                sample_indices = list(range(batch_size))
                            
                            # 转换为numpy数组（如果还不是）
                            if isinstance(inputs, torch.Tensor):
                                inputs_np = inputs.cpu().numpy()
                            elif not isinstance(inputs, np.ndarray):
                                inputs_np = np.array(inputs)
                            else:
                                inputs_np = inputs
                            
                            # 转换标签为numpy数组（如果有）
                            labels_np = None
                            if labels is not None:
                                if isinstance(labels, torch.Tensor):
                                    labels_np = labels.cpu().numpy()
                                elif not isinstance(labels, np.ndarray):
                                    labels_np = np.array(labels)
                                else:
                                    labels_np = labels
                            
                            # ========== T6: 进入队列时间 ==========
                            cloud_queue_enter = time.time()
                            
                            inference_request = InferenceRequest(
                                batch_id=batch_id,
                                sample_indices=sample_indices,
                                inputs=inputs_np,
                                edge_id=edge_id,
                                timestamp=time.time(),
                                labels=labels_np,
                                # 新增：时间戳
                                edge_send_time=edge_send_time,
                                cloud_receive_start=cloud_receive_start,
                                cloud_receive_end=cloud_receive_end,
                                cloud_queue_enter=cloud_queue_enter
                            )
                            
                            # 放入推理队列
                            try:
                                inference_queue.put(inference_request, block=True, timeout=5.0)
                            except queue.Full:
                                _log("WARN", f"推理队列已满，丢弃请求 (batch_id={batch_id})")
                        
                        # 同步模式：立即执行推理（注意：ZeroMQ 模式下不推荐同步模式）
                        else:
                            _log("WARN", "ZeroMQ 模式下不支持同步推理，请启用异步模式")
                    
                    elif request_type == 'end_transmission':
                        # 处理结束标志
                        edge_id = request.get('edge_id', 'unknown')
                        total_samples = request.get('total_samples', 0)
                        total_batches = request.get('total_batches', 0)
                        
                        _log("INFO", 
                             f"收到结束标志: edge_id={edge_id}, "
                             f"总样本数={total_samples}, 总批次数={total_batches}")
                        
                        # 输出推理统计结果（如果是端侧直连）
                        if edge_id == 'device_direct':
                            from async_cloud_pipeline import WorkerThread
                            
                            # 等待队列中的所有任务被处理完成
                            if inference_queue is not None:
                                _log("INFO", "等待所有推理任务完成...")
                                inference_queue.join()  # 阻塞直到所有任务完成
                                _log("INFO", "所有推理任务已完成")
                            
                            # 打印统计信息
                            WorkerThread.print_global_stats(edge_id)
                            
                            # 清理显存
                            torch.cuda.empty_cache()
                            
                            # ✅ 发送"推理完成"标志给端侧
                            if response_sender is not None:
                                try:
                                    completion_signal = {
                                        'type': 'inference_complete',
                                        'edge_id': 'device_direct',
                                        'message': '所有推理任务已完成',
                                        'timestamp': time.time(),
                                        'total_samples': total_samples,
                                        'total_batches': total_batches
                                    }
                                    
                                    # 序列化并发送
                                    if use_msgpack:
                                        import msgpack
                                        serialized = msgpack.packb(completion_signal, use_bin_type=True)
                                    else:
                                        serialized = pickle.dumps(completion_signal)
                                    
                                    response_sender.zmq_push_socket.send(serialized)
                                    _log("INFO", "已发送推理完成标志给端侧")
                                except Exception as e:
                                    _log("ERROR", f"发送推理完成标志失败: {e}")
                        
                        # ZeroMQ异步模式：已发送确认响应
                    
                    else:
                        _log("WARN", f"未知请求类型: {request_type}")
            
            except Exception as e:
                _log("ERROR", f"处理请求错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    except Exception as e:
        _log("ERROR", f"ZeroMQ 请求处理线程严重错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        _log("INFO", "ZeroMQ 请求处理线程结束")
    if async_mode:
        print(f"[云侧] [线程-{client_id}] 异步模式已启用")
    
    # 连接级别的统计（用于端侧直连模式）
    connection_stats = {
        'total_samples': 0,
        'correct_predictions': 0,
        'total_batches': 0,
        'edge_id': None,
        'connection_type': None  # 'request' 或 'response'
    }
    
    try:
        # 为连接设置socket选项
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        conn.settimeout(600)
        
        # Windows keepalive
        if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
            conn.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60000, 30000))
        
        # 异步模式：注册连接到ResponseSender（在收到第一个请求后）
        edge_id_registered = False
        
        # 持久连接：循环处理多个请求
        while True:
            try:
                # 接收请求
                def recv_all(conn, size):
                    """确保接收完整的size字节数据"""
                    data = b''
                    while len(data) < size:
                        packet = conn.recv(min(size - len(data), 65536))
                        if not packet:
                            raise ConnectionError(f"连接断开，已接收 {len(data)}/{size} 字节")
                        data += packet
                    return data
                
                # 接收数据大小（8字节）
                size_data = recv_all(conn, 8)
                size = struct.unpack('Q', size_data)[0]
                
                # 接收完整数据
                data = recv_all(conn, size)
                
                # 反序列化请求
                request = pickle.loads(data)
                
                # 检查是否是连接类型握手消息（双连接架构）
                if 'connection_type' in request:
                    connection_type = request.get('connection_type')
                    edge_id = request.get('edge_id', 'unknown')
                    
                    connection_stats['connection_type'] = connection_type
                    connection_stats['edge_id'] = edge_id
                    
                    print(f"[云侧] [线程-{client_id}] 收到连接类型握手: type={connection_type}, edge_id={edge_id}")
                    
                    # 忽略连接类型握手，使用单连接全双工模式
                    # 继续接收下一个请求
                    continue
                
                # 更新全局请求计数
                with counter_lock:
                    request_counter[0] += 1
                    current_count = request_counter[0]
                
                # 每10个请求打印一次信息
                if current_count % 10 == 0:
                    print(f"[云侧] 已处理 {current_count} 个推理请求（来自所有边侧）")
                
                request_type = request.get('type', 'unknown')
                edge_id = request.get('edge_id', 'unknown')  # 获取边侧ID或端侧标识
                
                # 保存 edge_id 到连接统计
                if connection_stats['edge_id'] is None:
                    connection_stats['edge_id'] = edge_id
                
                # 异步模式：注册连接到ResponseSender（使用edge_id）
                if async_mode and not edge_id_registered and edge_id != 'device_direct':
                    # 将edge_id转换为整数（如果可能）
                    try:
                        edge_id_int = int(edge_id) if isinstance(edge_id, str) and edge_id.isdigit() else hash(edge_id) % 10000
                    except:
                        edge_id_int = hash(edge_id) % 10000
                    
                    response_sender.register_connection(edge_id_int, conn)
                    edge_id_registered = True
                    print(f"[云侧] [线程-{client_id}] 已注册边侧 {edge_id} (ID={edge_id_int}) 到ResponseSender")
                
                if request_type == 'cloud_inference':
                    inputs = request.get('inputs')
                    labels = request.get('labels', None)  # 获取真实标签（如果有）
                    batch_id = request.get('batch_id', None)  # 获取批次ID（异步模式）
                    sample_indices = request.get('sample_indices', None)  # 获取样本索引（异步模式）
                    batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs)
                    
                    # 打印推理信息（区分端侧和边侧）
                    if edge_id == 'device_direct':
                        source_label = "端侧"
                    else:
                        source_label = f"边侧-{edge_id}"
                    print(f"[云侧] [{source_label}] [线程-{client_id}] 收到推理请求 #{current_count}: batch_size={batch_size}")
                    
                    if inputs is None:
                        response = {'status': 'error', 'message': '缺少输入数据'}
                    else:
                        # 异步模式：将请求放入队列，立即返回确认
                        if async_mode:
                            # 构造InferenceRequest
                            from async_cloud_pipeline import InferenceRequest
                            
                            # 如果没有batch_id，使用请求计数作为batch_id
                            if batch_id is None:
                                batch_id = current_count
                            
                            # 如果没有sample_indices，使用默认索引
                            if sample_indices is None:
                                sample_indices = list(range(batch_size))
                            
                            # 转换为numpy数组（如果还不是）
                            if isinstance(inputs, torch.Tensor):
                                inputs_np = inputs.cpu().numpy()
                            elif not isinstance(inputs, np.ndarray):
                                inputs_np = np.array(inputs)
                            else:
                                inputs_np = inputs
                            
                            inference_request = InferenceRequest(
                                batch_id=batch_id,
                                sample_indices=sample_indices,
                                inputs=inputs_np,
                                edge_id=edge_id,
                                timestamp=time.time()
                            )
                            
                            # 放入推理队列
                            try:
                                inference_queue.put(inference_request, block=True, timeout=5.0)
                                
                                # 立即返回确认（不等待推理完成）
                                response = {
                                    'status': 'accepted',
                                    'message': '请求已接受，正在处理',
                                    'batch_id': batch_id
                                }
                                
                                print(f"[云侧] [{source_label}] [线程-{client_id}] 请求已放入队列 (batch_id={batch_id})")
                                
                            except queue.Full:
                                # 队列满，返回错误
                                response = {
                                    'status': 'error',
                                    'message': '推理队列已满，请稍后重试'
                                }
                                print(f"[云侧] [{source_label}] [线程-{client_id}] 推理队列已满")
                        
                        # 同步模式：立即执行推理并返回结果
                        else:
                            # 云侧推理
                            with torch.no_grad():
                                if isinstance(inputs, np.ndarray):
                                    inputs_tensor = torch.from_numpy(inputs).to(device)
                                else:
                                    inputs_tensor = torch.tensor(inputs, device=device)
                                
                                outputs = cloud_model(inputs_tensor)
                                
                                if torch.is_complex(outputs):
                                    outputs = torch.abs(outputs)
                                
                                predictions = torch.argmax(outputs, dim=1)
                            
                            # 如果有标签，计算准确率（端侧直连模式）
                            if labels is not None and edge_id == 'device_direct':
                                labels_tensor = torch.tensor(labels, device=device)
                                correct = (predictions == labels_tensor).sum().item()
                                accuracy = 100.0 * correct / len(labels)
                                
                                # 累积统计
                                connection_stats['total_samples'] += len(labels)
                                connection_stats['correct_predictions'] += correct
                                connection_stats['total_batches'] += 1
                                
                                # 打印推理结果和准确率
                                print(f"[云侧] [{source_label}] [线程-{client_id}] 推理完成 #{current_count}: "
                                      f"batch_size={len(predictions)}, 准确率={accuracy:.2f}% ({correct}/{len(labels)})")
                            else:
                                # 边侧请求，只打印推理完成
                                print(f"[云侧] [{source_label}] [线程-{client_id}] 推理完成 #{current_count}: 返回 {len(predictions)} 个预测结果")
                            
                            response = {
                                'status': 'success',
                                'predictions': predictions.cpu().tolist()
                            }
                    
                    # 发送响应（带限速）
                    serialized = pickle.dumps(response)
                    response_size = len(serialized)
                    
                    conn.sendall(struct.pack('Q', response_size))
                    
                    # 分块发送（带限速）
                    if args.rate_limit:
                        chunk_size = 1024 * 1024  # 1MB per chunk
                        rate_limit_bps = args.rate_limit * 1024 * 1024
                        
                        sent = 0
                        start_time = time.time()
                        
                        while sent < response_size:
                            chunk = serialized[sent:sent + chunk_size]
                            conn.sendall(chunk)
                            sent += len(chunk)
                            
                            # 累积计时限速
                            elapsed = time.time() - start_time
                            expected_time = sent / rate_limit_bps
                            
                            if elapsed < expected_time:
                                sleep_time = expected_time - elapsed
                                if sleep_time > 0.01:
                                    time.sleep(sleep_time)
                    else:
                        # 不限速，分块发送
                        sent = 0
                        chunk_size = 65536
                        while sent < response_size:
                            chunk = serialized[sent:sent + chunk_size]
                            conn.sendall(chunk)
                            sent += len(chunk)
                            
                            if sent % (chunk_size * 10) == 0:
                                time.sleep(0.001)
                
                elif request_type == 'end_transmission':
                    # 处理结束标志
                    total_samples = request.get('total_samples', 0)
                    total_batches = request.get('total_batches', 0)
                    
                    if edge_id == 'device_direct':
                        source_label = "端侧"
                    else:
                        source_label = f"边侧-{edge_id}"
                    
                    print(f"[云侧] [{source_label}] [线程-{client_id}] 收到结束标志: "
                          f"总样本数={total_samples}, 总批次数={total_batches}")
                    
                    # 如果是端侧直连，输出整体推理统计
                    if edge_id == 'device_direct' and connection_stats['total_samples'] > 0:
                        overall_accuracy = 100.0 * connection_stats['correct_predictions'] / connection_stats['total_samples']
                        print(f"\n{'='*70}")
                        print(f"[云侧] [{source_label}] 推理统计汇总")
                        print(f"{'='*70}")
                        print(f"总样本数: {connection_stats['total_samples']}")
                        print(f"总批次数: {connection_stats['total_batches']}")
                        print(f"正确预测: {connection_stats['correct_predictions']}")
                        print(f"整体准确率: {overall_accuracy:.2f}%")
                        print(f"{'='*70}\n")
                    
                    response = {
                        'status': 'success',
                        'message': '已确认接收完成'
                    }
                    serialized = pickle.dumps(response)
                    conn.sendall(struct.pack('Q', len(serialized)))
                    conn.sendall(serialized)
                    
                    # 结束标志后退出循环
                    break
                
                else:
                    response = {'status': 'error', 'message': f'未知请求类型: {request_type}'}
                    serialized = pickle.dumps(response)
                    conn.sendall(struct.pack('Q', len(serialized)))
                    
                    # 带限速发送
                    if args.rate_limit:
                        chunk_size = 1024 * 1024
                        rate_limit_bps = args.rate_limit * 1024 * 1024
                        sent = 0
                        start_time = time.time()
                        
                        while sent < len(serialized):
                            chunk = serialized[sent:sent + chunk_size]
                            conn.sendall(chunk)
                            sent += len(chunk)
                            
                            elapsed = time.time() - start_time
                            expected_time = sent / rate_limit_bps
                            if elapsed < expected_time:
                                sleep_time = expected_time - elapsed
                                if sleep_time > 0.01:
                                    time.sleep(sleep_time)
                    else:
                        conn.sendall(serialized)
            
            except (ConnectionError, OSError, struct.error) as e:
                print(f"[云侧] [线程-{client_id}] 连接错误: {e}")
                break
            except Exception as e:
                print(f"[云侧] [线程-{client_id}] 处理请求错误: {e}")
                import traceback
                traceback.print_exc()
                break
    
    finally:
        # 异步模式：注销连接
        if async_mode and edge_id_registered:
            edge_id = connection_stats.get('edge_id')
            if edge_id and edge_id != 'device_direct':
                try:
                    edge_id_int = int(edge_id) if isinstance(edge_id, str) and edge_id.isdigit() else hash(edge_id) % 10000
                except:
                    edge_id_int = hash(edge_id) % 10000
                
                response_sender.unregister_connection(edge_id_int)
                print(f"[云侧] [线程-{client_id}] 已注销边侧 {edge_id} (ID={edge_id_int})")
        
        try:
            conn.close()
            print(f"[云侧] [线程-{client_id}] 连接已关闭")
        except:
            pass


def run_collaborative_cloud(args):
    """运行云边协同推理云侧"""
    print(f"\n{'='*70}")
    print(f"[云侧] 云边协同推理服务器启动")
    print(f"{'='*70}")
    print(f"数据集: {args.dataset_type}")
    print(f"类别数: {args.num_classes}")
    print(f"云侧模型类型: {args.cloud_model if args.cloud_model else f'complex_resnet50_{args.dataset_type}'}")
    print(f"云侧模型路径: {args.cloud_model_path}")
    print(f"监听端口: {args.cloud_port}")
    if args.rate_limit:
        print(f"网络限速: {args.rate_limit} MB/s")
    
    # 异步模式配置
    async_mode = getattr(args, 'async_mode', False)
    if async_mode:
        print(f"异步模式: 已启用")
        print(f"推理队列大小: {getattr(args, 'inference_queue_size', 1000)}")
        print(f"工作线程数: {getattr(args, 'num_workers', 4)}")
        print(f"批量推理策略: 端侧数据单独处理，边侧数据自动批量处理")
        print(f"批量推理超时: {getattr(args, 'batch_timeout', 0.05)}秒")
    else:
        print(f"异步模式: 未启用（同步模式）")
    
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 检查云侧模型路径
    if not args.cloud_model_path or not os.path.exists(args.cloud_model_path):
        print(f"错误: 云侧模型路径无效: {args.cloud_model_path}")
        print("请使用 --cloud_model_path 指定训练好的教师模型路径")
        return
    
    # 添加模块别名支持（兼容旧的 checkpoint）
    dataset_modules = [
        'readdata_rml2016',
        'readdata_radar', 
        'readdata_radioml',
        'readdata_reii',
        'readdata_25',
        'readdata_link11'
    ]
    
    for module_name in dataset_modules:
        if module_name not in sys.modules:
            try:
                module = __import__(f'utils.{module_name}', fromlist=[module_name])
                sys.modules[module_name] = module
            except Exception:
                pass
    
    # 加载云侧模型（教师模型）
    print("[云侧] 加载教师模型...")
    # 如果指定了 cloud_model，使用指定的模型类型；否则使用默认的 complex_resnet50
    if args.cloud_model:
        cloud_model_type = args.cloud_model
    else:
        cloud_model_type = f'complex_resnet50_{args.dataset_type}'
    
    print(f"[云侧] 模型类型: {cloud_model_type}")
    cloud_model = create_model_by_type(cloud_model_type, args.num_classes, args.dataset_type)
    
    try:
        checkpoint = torch.load(args.cloud_model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型出错: {e}")
        print("尝试使用自定义 Unpickler...")
        
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in dataset_modules:
                    module = f'utils.{module}'
                return super().find_class(module, name)
        
        with open(args.cloud_model_path, 'rb') as f:
            checkpoint = CPU_Unpickler(f).load()
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        cloud_model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        cloud_model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        cloud_model.load_state_dict(checkpoint['model'])
    else:
        cloud_model.load_state_dict(checkpoint)
    
    cloud_model = cloud_model.to(device)
    cloud_model.eval()
    print("[云侧] 教师模型加载完成\n")
    
    # 初始化异步流水线组件（如果启用异步模式）
    inference_queue = None
    response_sender = None
    worker_threads = []
    stop_event = None
    
    if async_mode:
        from async_cloud_pipeline import InferenceQueue, ResponseSender, WorkerThread
        
        # 创建推理队列
        queue_size = getattr(args, 'inference_queue_size', 1000)
        inference_queue = InferenceQueue(maxsize=queue_size)
        print(f"[云侧] 推理队列已创建 (maxsize={queue_size})")
        
        # 创建响应发送器（ZeroMQ PUSH socket，端口 5556）
        push_port = getattr(args, 'zmq_push_port', 5556)
        response_sender = ResponseSender(push_port=push_port)
        print(f"[云侧] 响应发送器已创建 (ZeroMQ PUSH 端口: {push_port})")
        
        # 创建停止事件
        stop_event = threading.Event()
        
        # 创建工作线程池
        num_workers = getattr(args, 'num_workers', 4)
        batch_timeout = getattr(args, 'batch_timeout', 0.05)
        
        print(f"[云侧] 创建 {num_workers} 个工作线程...")
        # print(f"[云侧] 批量推理策略: 端侧数据单独处理，边侧数据自动批量处理 (timeout={batch_timeout}s)")
        
        for i in range(num_workers):
            worker = WorkerThread(
                worker_id=i,
                inference_queue=inference_queue,
                model=cloud_model,
                device=device,
                response_sender=response_sender,
                stop_event=stop_event,
                batch_timeout=batch_timeout
            )
            worker.start()
            worker_threads.append(worker)
            print(f"[云侧] WorkerThread-{i} 已启动")
        
        print(f"[云侧] 工作线程池已创建 ({num_workers} 个线程)\n")
        
        # 创建线程健康检查器
        def health_check_worker():
            """监控工作线程健康状态，必要时重启"""
            print("[云侧] 线程健康检查器已启动")
            
            while not stop_event.is_set():
                time.sleep(10)  # 每10秒检查一次
                
                if stop_event.is_set():
                    break
                
                # 检查每个工作线程
                for i, worker in enumerate(worker_threads):
                    if not worker.is_alive():
                        print(f"[云侧] 警告: WorkerThread-{i} 已停止，尝试重启...")
                        
                        # 创建新的工作线程
                        new_worker = WorkerThread(
                            worker_id=i,
                            inference_queue=inference_queue,
                            model=cloud_model,
                            device=device,
                            response_sender=response_sender,
                            stop_event=stop_event
                        )
                        new_worker.start()
                        worker_threads[i] = new_worker
                        
                        print(f"[云侧] WorkerThread-{i} 已重启")
            
            print("[云侧] 线程健康检查器已停止")
        
        # 启动健康检查线程
        health_check_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_check_thread.start()
    
    # 创建 ZeroMQ PULL socket 接收请求（端口 5555）
    import zmq
    
    pull_port = getattr(args, 'zmq_pull_port', 5555)
    zmq_context = zmq.Context()
    zmq_pull_socket = zmq_context.socket(zmq.PULL)
    zmq_pull_socket.bind(f"tcp://*:{pull_port}")
    
    print(f"[云侧] ZeroMQ PULL socket 已绑定到端口 {pull_port}")
    print(f"[云侧] 等待边侧请求...\n")
    
    # 全局请求计数器（使用列表以便在线程间共享）
    request_counter = [0]
    counter_lock = threading.Lock()
    
    # 创建 ZeroMQ 请求处理线程
    zmq_thread = threading.Thread(
        target=handle_zmq_requests,
        args=(zmq_pull_socket, cloud_model, device, args, request_counter, counter_lock),
        kwargs={'inference_queue': inference_queue, 'stop_event': stop_event, 'response_sender': response_sender},
        daemon=True
    )
    zmq_thread.start()
    print(f"[云侧] ZeroMQ 请求处理线程已启动\n")
    
    # 持续运行，直到收到中断信号
    try:
        while True:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n[云侧] 收到键盘中断")
        
        # 停止工作线程（如果有）
        if stop_event is not None:
            print(f"[云侧] 停止 {len(worker_threads)} 个工作线程...")
            stop_event.set()
            
            for worker in worker_threads:
                worker.join(timeout=5)
            
            print(f"[云侧] 所有工作线程已停止")
            
            # 关闭响应发送器
            if response_sender is not None:
                response_sender.close()
        
        # 关闭 ZeroMQ
        try:
            zmq_pull_socket.close()
            zmq_context.term()
            print("[云侧] ZeroMQ 已关闭")
        except:
            pass
        
        print("[云侧] 服务器已关闭")


def main():
    parser = argparse.ArgumentParser(description='云边协同推理 - 云侧服务器')
    
    # 基本参数
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['rml2016', 'radar', 'link11'],
                       help='数据集类型')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='分类类别数')
    
    # 模型参数
    parser.add_argument('--cloud_model_path', type=str, required=True,
                       help='云侧教师模型路径')
    parser.add_argument('--cloud_model', type=str, default=None,
                       help='云侧模型类型（可选，如 complex_resnet50_link11_with_attention）。'
                            '如果不指定，默认使用 complex_resnet50_{dataset_type}')
    
    # 网络参数
    parser.add_argument('--cloud_port', type=int, default=9999,
                       help='云侧监听端口 (默认: 9999，ZeroMQ 模式下不使用)')
    parser.add_argument('--rate_limit', type=float, default=10.0,
                       help='网络速率限制（MB/s，默认10，ZeroMQ 模式下不使用）')
    
    # ZeroMQ 参数
    parser.add_argument('--zmq_pull_port', type=int, default=5555,
                       help='ZeroMQ PULL socket 端口（接收请求，默认：5555）')
    parser.add_argument('--zmq_push_port', type=int, default=5556,
                       help='ZeroMQ PUSH socket 端口（发送响应，默认：5556）')
    
    # 异步模式参数（默认启用）
    parser.add_argument('--inference_queue_size', type=int, default=1000,
                       help='推理队列大小（默认：1000）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='工作线程数（默认：4）')
    parser.add_argument('--batch_timeout', type=float, default=0.05,
                       help='边侧数据批量推理等待超时（秒，默认：0.05）。端侧数据始终单独处理')
    
    args = parser.parse_args()
    
    # 默认启用异步模式
    args.async_mode = True
    
    run_collaborative_cloud(args)


if __name__ == '__main__':
    main()

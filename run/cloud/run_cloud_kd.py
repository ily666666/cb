#!/usr/bin/env python3
"""
云侧轻量化服务器 - 负责向边侧分发教师模型
"""
import sys
import os
import argparse
import socket
import pickle
import struct
import time
import torch

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed.project import create_model_by_type


def send_data(conn, data, rate_limit_mbps=None):
    """发送数据（带速率限制）
    
    改进的速率限制算法：
    - 使用更大的chunk_size (1MB) 减少循环开销
    - 累积计时，避免频繁的time.time()调用
    - 只在必要时sleep，提高精度
    - 分离序列化时间统计
    """
    # 序列化数据（这一步可能很慢，单独计时）
    serialize_start = time.time()
    serialized = pickle.dumps(data)
    serialize_time = time.time() - serialize_start
    size = len(serialized)
    
    print(f"[调试] 序列化耗时: {serialize_time:.2f}s, 数据大小: {size/(1024*1024):.2f}MB")
    
    # 发送数据大小
    conn.sendall(struct.pack('Q', size))
    
    # 开始计时传输时间
    transfer_start = time.time()
    
    # 如果启用限速，分块发送
    if rate_limit_mbps:
        chunk_size = 1024 * 1024  # 1MB per chunk
        rate_limit_bps = rate_limit_mbps * 1024 * 1024  # 转换为 bytes/s
        
        sent = 0
        chunk_start_time = time.time()  # 记录分块发送开始时间
        
        while sent < size:
            chunk = serialized[sent:sent + chunk_size]
            conn.sendall(chunk)
            sent += len(chunk)
            
            # 每发送完一个chunk，检查是否需要限速
            elapsed = time.time() - chunk_start_time
            expected_time = sent / rate_limit_bps
            
            # 如果发送太快，sleep到预期时间
            if elapsed < expected_time:
                sleep_time = expected_time - elapsed
                # 只有当sleep时间大于10ms时才sleep，避免精度问题
                if sleep_time > 0.01:
                    time.sleep(sleep_time)
    else:
        # 不限速，直接发送
        conn.sendall(serialized)
    
    transfer_time = time.time() - transfer_start
    actual_rate = size / (1024 * 1024) / transfer_time if transfer_time > 0 else 0
    print(f"[调试] 传输耗时: {transfer_time:.2f}s, 实际速率: {actual_rate:.2f}MB/s")
    
    return size


def receive_data(conn):
    """接收数据"""
    size_data = _recv_all(conn, 8)
    size = struct.unpack('Q', size_data)[0]
    data = _recv_all(conn, size)
    return pickle.loads(data), size


def _recv_all(conn, size):
    """确保接收完整数据"""
    data = b''
    while len(data) < size:
        packet = conn.recv(min(size - len(data), 65536))
        if not packet:
            raise ConnectionError("连接断开")
        data += packet
    return data


def main():
    parser = argparse.ArgumentParser(description='云侧轻量化服务器 - 向边侧分发教师模型')
    
    # 基本参数
    parser.add_argument('--dataset_type', type=str, required=True, help='数据集类型')
    parser.add_argument('--num_classes', type=int, required=True, help='类别数')
    parser.add_argument('--cloud_model', type=str, required=True, help='教师模型类型')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='教师模型路径')
    parser.add_argument('--cloud_port', type=int, default=9999, help='云侧端口')
    parser.add_argument('--num_edges', type=int, required=True, help='边侧数量')
    parser.add_argument('--rate_limit', type=float, default=10.0, help='网络速率限制（MB/s，默认10）')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"[云侧] 启动")
    print(f"{'='*70}")
    print(f"数据集: {args.dataset_type}")
    print(f"类别数: {args.num_classes}")
    print(f"教师模型: {args.cloud_model}")
    print(f"模型路径: {args.teacher_model_path}")
    print(f"监听端口: {args.cloud_port}")
    print(f"等待边侧数: {args.num_edges}")
    print(f"网络限速: {args.rate_limit} MB/s")
    print(f"{'='*70}\n")
    
    # 检查教师模型是否存在
    if not os.path.exists(args.teacher_model_path):
        print(f"错误: 教师模型文件不存在: {args.teacher_model_path}")
        print(f"请先运行云侧预训练脚本生成教师模型")
        return
    
    # 添加所有数据加载器模块到 sys.modules
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
            except Exception as e:
                pass
    
    # 加载教师模型
    print("[云侧] 加载教师模型...")
    try:
        checkpoint = torch.load(args.teacher_model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型出错: {e}")
        print("尝试使用自定义 Unpickler...")
        
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in dataset_modules:
                    module = f'utils.{module}'
                return super().find_class(module, name)
        
        with open(args.teacher_model_path, 'rb') as f:
            checkpoint = CPU_Unpickler(f).load()
    
    # 提取教师模型权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        teacher_model_state = checkpoint['model_state_dict']
        teacher_model_architecture = checkpoint.get('model_architecture', args.cloud_model)
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        teacher_model_state = checkpoint['state_dict']
        teacher_model_architecture = args.cloud_model
    else:
        teacher_model_state = checkpoint
        teacher_model_architecture = args.cloud_model
    
    print(f"[云侧] 教师模型加载完成")
    print(f"  模型架构: {teacher_model_architecture}")
    print(f"  参数数量: {sum(p.numel() for p in teacher_model_state.values())}")
    
    # 创建socket服务器
    cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cloud_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    cloud_socket.bind(('0.0.0.0', args.cloud_port))
    cloud_socket.listen(args.num_edges * 2)
    
    print(f"\n[云侧] 监听端口 {args.cloud_port}，等待边侧连接...")
    
    # 跟踪已下载的边侧
    downloaded_edges = set()
    
    # 服务循环
    while len(downloaded_edges) < args.num_edges:
        try:
            conn, addr = cloud_socket.accept()
            conn.settimeout(300)  # 5分钟超时
            
            print(f"\n[云侧] 边侧连接: {addr}")
            
            try:
                # 接收请求
                request, _ = receive_data(conn)
                request_type = request.get('type', 'unknown')
                edge_id = request.get('edge_id', -1)
                
                if request_type == 'download_teacher':
                    # 检查是否已经下载过
                    if edge_id in downloaded_edges:
                        print(f"[云侧] 警告: 边侧 {edge_id} 重复下载请求")
                        send_data(conn, {
                            'status': 'error',
                            'message': '已经下载过教师模型'
                        }, rate_limit_mbps=args.rate_limit)
                        conn.close()
                        continue
                    
                    print(f"[云侧] 发送教师模型到边侧 {edge_id}...")
                    start_time = time.time()
                    
                    # 发送教师模型（带限速）
                    size = send_data(conn, {
                        'status': 'success',
                        'model_state': teacher_model_state,
                        'model_architecture': teacher_model_architecture,
                        'num_classes': args.num_classes,
                        'dataset_type': args.dataset_type
                    }, rate_limit_mbps=args.rate_limit)
                    
                    send_time = time.time() - start_time
                    actual_rate = size / (1024 * 1024) / send_time if send_time > 0 else 0
                    print(f"[云侧] 发送完成: {size/(1024*1024):.2f}MB, 耗时 {send_time:.2f}s, 实际速率 {actual_rate:.2f}MB/s")
                    
                    downloaded_edges.add(edge_id)
                    print(f"[云侧] 已发送给 {len(downloaded_edges)}/{args.num_edges} 个边侧")
                    
                else:
                    print(f"[云侧] 警告: 收到非法请求类型 '{request_type}'")
                    send_data(conn, {
                        'status': 'error',
                        'message': f'不支持的请求类型: {request_type}'
                    }, rate_limit_mbps=args.rate_limit)
                
            except Exception as e:
                print(f"[云侧] 处理请求错误: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                conn.close()
        
        except Exception as e:
            print(f"[云侧] 连接错误: {e}")
    
    print(f"\n{'='*70}")
    print(f"[云侧] 所有 {args.num_edges} 个边侧已下载教师模型")
    print(f"[云侧] 服务器关闭")
    print(f"{'='*70}\n")
    
    cloud_socket.close()


if __name__ == '__main__':
    main()

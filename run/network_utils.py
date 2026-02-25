#!/usr/bin/env python3
"""
网络通信工具模块 - 提供云侧和边侧的网络通信类
"""
import socket
import pickle
import struct
import time
import threading


class NetworkCloud:
    """网络云侧 - 处理模型分发和聚合"""
    
    def __init__(self, project_cloud, host='0.0.0.0', port=9999, rate_limit_mbps=None):
        self.project_cloud = project_cloud
        self.host = host
        self.port = port
        self.cloud_socket = None
        self.current_round = 0  # 当前轮次
        self.current_phase = 'kd_sync'  # 当前阶段：kd_sync, download, upload, aggregating
        self.round_complete = False  # 当前轮次是否完成
        self.kd_phase_complete = False  # KD阶段是否完成
        self.rate_limit_mbps = rate_limit_mbps  # 速率限制（MB/s），None 表示不限速
        
    def send_data(self, conn, data):
        """发送数据（带速率限制）
        
        改进的速率限制算法：
        - 使用更大的chunk_size (1MB) 减少循环开销
        - 累积计时，避免频繁的time.time()调用
        - 只在必要时sleep，提高精度
        """
        serialized = pickle.dumps(data)
        size = len(serialized)
        
        # 发送数据大小
        conn.sendall(struct.pack('Q', size))
        
        # 如果启用限速，分块发送
        if self.rate_limit_mbps:
            chunk_size = 1024 * 1024  # 1MB per chunk (原来是64KB)
            rate_limit_bps = self.rate_limit_mbps * 1024 * 1024  # 转换为 bytes/s
            
            sent = 0
            start_time = time.time()  # 记录整体开始时间
            
            while sent < size:
                chunk = serialized[sent:sent + chunk_size]
                conn.sendall(chunk)
                sent += len(chunk)
                
                # 每发送完一个chunk，检查是否需要限速
                elapsed = time.time() - start_time
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
        
        return size
        
    def receive_data(self, conn):
        """接收数据"""
        size_data = self._recv_all(conn, 8)
        size = struct.unpack('Q', size_data)[0]
        data = self._recv_all(conn, size)
        return pickle.loads(data), size
    
    def _recv_all(self, conn, size):
        """确保接收完整数据"""
        data = b''
        while len(data) < size:
            packet = conn.recv(min(size - len(data), 65536))
            if not packet:
                raise ConnectionError("连接断开")
            data += packet
        return data
    
    def handle_edge_request(self, conn, addr):
        """处理边侧请求"""
        try:
            request, size = self.receive_data(conn)
            request_type = request.get('type', 'unknown')
            edge_id = request.get('edge_id', -1)
            
            # 处理状态检查请求（联邦学习轮次同步）
            if request_type == 'check_ready':
                requested_round = request.get('round', -1)
                if requested_round == self.current_round and self.current_phase == 'download':
                    self.send_data(conn, {'status': 'ready'})
                else:
                    self.send_data(conn, {'status': 'wait', 'message': f'当前轮次{self.current_round}, 阶段{self.current_phase}'})
                return None, None
            
            # 处理阶段检查请求（等待所有边侧下载完成）
            if request_type == 'check_phase':
                expected_phase = request.get('expected_phase', '')
                if self.current_phase == expected_phase:
                    self.send_data(conn, {'status': 'ready', 'current_phase': self.current_phase})
                else:
                    self.send_data(conn, {'status': 'wait', 'current_phase': self.current_phase, 'message': f'当前阶段{self.current_phase}, 期望{expected_phase}'})
                return None, None
            
            # 其他请求类型正常返回
            return request, size
            
        except Exception as e:
            print(f"[云侧] 处理请求错误: {e}")
            return None, None
    
    def close(self):
        """关闭云侧"""
        if self.cloud_socket:
            print("\n[云侧] 正在关闭网络云侧...")
            self.cloud_socket.close()
            self.cloud_socket = None
            print("[云侧] 网络云侧已关闭")
    
    def run(self, num_edges, num_rounds, download_timeout=120, upload_timeout=300):
        """运行网络云侧 - 保持在线状态
        
        Args:
            num_edges: 期望的边侧数量
            num_rounds: 联邦学习轮次
            download_timeout: 下载阶段超时（秒），默认120秒
            upload_timeout: 上传阶段超时（秒），默认300秒
        """
        cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cloud_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cloud_socket.settimeout(None)  # 云侧不设置超时
        cloud_socket.bind((self.host, self.port))
        cloud_socket.listen(num_edges * 4)  # 增加队列长度
        
        print(f"\n{'='*70}")
        print(f"[云侧] 网络云侧启动: {self.host}:{self.port}")
        print(f"[云侧] 期望 {num_edges} 个边侧连接（支持动态调整）")
        print(f"[云侧] 下载超时: {download_timeout}秒, 上传超时: {upload_timeout}秒")
        print(f"[云侧] 云侧将保持在线状态")
        print(f"{'='*70}\n")
        
        # 保存 socket 以便后续使用
        self.cloud_socket = cloud_socket
        
        # 标记已准备好联邦学习
        self.kd_phase_complete = True
        self.current_phase = 'download'
        
        print(f"\n{'='*70}")
        print(f"[云侧] 准备就绪，等待边侧连接开始联邦学习")
        print(f"{'='*70}\n")
        
        # ========== 联邦学习阶段 ==========
        for round_num in range(num_rounds):
            self.current_round = round_num
            self.current_phase = 'download'
            
            print(f"\n{'='*70}")
            print(f"[云侧] 联邦学习轮次 {round_num + 1}/{num_rounds}")
            print(f"{'='*70}")
            
            # 阶段1: 发送全局模型给所有边侧（带超时）
            print(f"\n[阶段1] 发送全局模型（期望 {num_edges} 个边侧，超时 {download_timeout}秒）")
            download_edges = set()  # 跟踪已下载的边侧
            download_start_time = time.time()
            
            # 设置socket为非阻塞模式，以便实现超时
            cloud_socket.settimeout(5)  # 5秒检查一次
            
            while len(download_edges) < num_edges:
                # 检查是否超时
                elapsed = time.time() - download_start_time
                if elapsed > download_timeout:
                    print(f"\n[云侧] ⚠️  下载阶段超时（{download_timeout}秒）")
                    print(f"[云侧] 已连接 {len(download_edges)} 个边侧，将使用这些边侧继续训练")
                    if len(download_edges) == 0:
                        print(f"[云侧] ❌ 错误：没有任何边侧连接，无法继续")
                        return
                    break
                
                try:
                    conn, addr = cloud_socket.accept()
                    conn.settimeout(60)  # 单个连接60秒超时
                    
                    try:
                        request, size = self.handle_edge_request(conn, addr)
                        
                        # 如果是状态检查请求，已经在handle_edge_request中处理
                        if request is None:
                            conn.close()
                            continue
                        
                        print(f"[云侧] 边侧连接: {addr}")
                        
                        # 检查请求类型
                        request_type = request.get('type', 'unknown')
                        edge_id = request.get('edge_id', -1)
                        
                        if request_type != 'download':
                            print(f"[云侧] 警告: 收到非下载请求 (type={request_type})，拒绝")
                            self.send_data(conn, {'status': 'error', 'message': '当前阶段只接受下载请求'})
                            conn.close()
                            continue
                        
                        # 检查边侧是否已经下载过
                        if edge_id in download_edges:
                            print(f"[云侧] 警告: 边侧 {edge_id} 重复下载请求")
                            self.send_data(conn, {'status': 'error', 'message': '已经下载过了'})
                            conn.close()
                            continue
                        
                        print(f"[云侧] 发送模型到边侧 {edge_id}")
                        start_time = time.time()
                        
                        # 获取全局模型状态
                        global_model_state = self.project_cloud.global_model_state
                        
                        size = self.send_data(conn, {
                            'status': 'success',
                            'model_state': global_model_state,
                            'round': round_num
                        })
                        
                        send_time = time.time() - start_time
                        print(f"[云侧] 发送完成: {size/(1024*1024):.2f}MB, 耗时 {send_time:.2f}s")
                        download_edges.add(edge_id)
                        print(f"[云侧] 已发送给 {len(download_edges)}/{num_edges} 个边侧")
                        
                    except Exception as e:
                        print(f"[云侧] 发送错误: {e}")
                    finally:
                        conn.close()
                        
                except socket.timeout:
                    # 超时，继续等待
                    continue
                except Exception as e:
                    print(f"[云侧] Accept错误: {e}")
                    continue
            
            # 更新实际参与的边侧数量
            actual_num_edges = len(download_edges)
            print(f"\n[云侧] 下载阶段完成，实际参与边侧数: {actual_num_edges}")
            print(f"[云侧] 等待这 {actual_num_edges} 个边侧上传模型...")
            self.current_phase = 'upload'
            
            # 阶段2: 接收边侧更新（只等待已下载的边侧）
            print(f"\n[阶段2] 接收 {actual_num_edges} 个边侧更新（超时 {upload_timeout}秒）")
            edge_updates = []
            edge_weights = []
            received_edge_ids = set()
            upload_start_time = time.time()
            
            while len(received_edge_ids) < actual_num_edges:
                # 检查是否超时
                elapsed = time.time() - upload_start_time
                if elapsed > upload_timeout:
                    print(f"\n[云侧] ⚠️  上传阶段超时（{upload_timeout}秒）")
                    print(f"[云侧] 已收到 {len(received_edge_ids)}/{actual_num_edges} 个边侧更新")
                    if len(received_edge_ids) == 0:
                        print(f"[云侧] ❌ 错误：没有收到任何边侧更新，跳过本轮")
                        break
                    print(f"[云侧] 将使用已收到的 {len(received_edge_ids)} 个更新进行聚合")
                    break
                
                try:
                    conn, addr = cloud_socket.accept()
                    conn.settimeout(60)  # 单个连接60秒超时
                    
                    try:
                        request, size = self.handle_edge_request(conn, addr)
                        
                        # 如果是状态检查请求，已经在handle_edge_request中处理
                        if request is None:
                            conn.close()
                            continue
                        
                        print(f"[云侧] 边侧连接: {addr}")
                        print(f"[云侧] 接收边侧更新...")
                        start_time = time.time()
                        
                        recv_time = time.time() - start_time
                        
                        # 检查请求类型
                        request_type = request.get('type', 'unknown')
                        edge_id = request.get('edge_id', -1)
                        
                        if request_type != 'upload':
                            print(f"[云侧] 警告: 收到非上传请求 (type={request_type})，拒绝")
                            self.send_data(conn, {'status': 'error', 'message': '当前阶段只接受上传请求'})
                            conn.close()
                            continue
                        
                        # 检查是否是参与下载的边侧
                        if edge_id not in download_edges:
                            print(f"[云侧] 警告: 边侧 {edge_id} 未参与下载，拒绝上传")
                            self.send_data(conn, {'status': 'error', 'message': '未参与本轮下载'})
                            conn.close()
                            continue
                        
                        # 检查是否重复上传
                        if edge_id in received_edge_ids:
                            print(f"[云侧] 警告: 边侧 {edge_id} 重复上传")
                            self.send_data(conn, {'status': 'error', 'message': '已经上传过了'})
                            conn.close()
                            continue
                        
                        model_state = request.get('model_state')
                        num_samples = request.get('num_samples', 1)
                        
                        if model_state is None:
                            print(f"[云侧] 错误: 边侧 {edge_id} 上传请求缺少 model_state")
                            self.send_data(conn, {'status': 'error', 'message': '缺少模型数据'})
                            conn.close()
                            continue
                        
                        edge_updates.append(model_state)
                        edge_weights.append(num_samples)
                        received_edge_ids.add(edge_id)
                        
                        print(f"[云侧] 接收完成: {size/(1024*1024):.2f}MB, 耗时 {recv_time:.2f}s")
                        print(f"[云侧] 边侧 {edge_id} 训练样本数: {num_samples}")
                        print(f"[云侧] 已收到 {len(received_edge_ids)}/{actual_num_edges} 个边侧更新")
                        
                        # 发送确认
                        self.send_data(conn, {'status': 'success'})
                        
                    except Exception as e:
                        print(f"[云侧] 接收错误: {e}")
                    finally:
                        conn.close()
                        
                except socket.timeout:
                    # 超时，继续等待
                    continue
                except Exception as e:
                    print(f"[云侧] Accept错误: {e}")
                    continue
            
            # 聚合模型
            self.current_phase = 'aggregating'
            if len(edge_updates) > 0:
                print(f"\n[云侧] 聚合 {len(edge_updates)} 个边侧模型...")
                
                # 归一化权重
                total_weight = sum(edge_weights)
                normalized_weights = [w / total_weight for w in edge_weights]
                
                print(f"[云侧] 边侧权重: {normalized_weights}")
                
                self.project_cloud.global_model_state = self.project_cloud.aggregate_models(
                    edge_updates, 
                    edge_weights=normalized_weights
                )
                print(f"[云侧] 聚合完成")
                print(f"[云侧] 轮次 {round_num + 1} 完成，准备下一轮...")
            else:
                print(f"\n[云侧] ⚠️  警告：本轮没有收到任何边侧更新，跳过聚合")
        
        print(f"\n{'='*70}")
        print(f"[云侧] 所有 {num_rounds} 轮联邦学习完成！")
        print(f"[云侧] 云侧保持在线，等待新的连接...")
        print(f"{'='*70}\n")


class NetworkEdge:
    """网络边侧 - 处理模型下载和上传"""
    
    def __init__(self, edge_id, cloud_host, cloud_port, rate_limit_mbps=None):
        self.edge_id = edge_id
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.rate_limit_mbps = rate_limit_mbps  # 速率限制（MB/s），None 表示不限速
        
        # 单连接全双工模式：一个连接同时用于发送和接收
        self.cloud_inference_conn = None
        self.connection_failures = 0  # 连接失败计数
        self.last_connection_time = 0  # 上次连接时间
        
        # ZeroMQ 支持（用于异步协同推理）
        self.zmq_context = None
        self.zmq_push_socket = None  # 发送请求
        self.zmq_pull_socket = None  # 接收响应
        self.zmq_initialized = False
        
    def _ensure_cloud_connection(self):
        """确保云侧推理连接存在且有效"""
        import time
        
        # 如果连接不存在或已断开，建立新连接
        if self.cloud_inference_conn is None:
            try:
                print(f"[边侧 {self.edge_id}] 建立到云侧的持久连接...")
                
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(60)
                
                # 设置socket选项（超大缓冲区）
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
                
                # Windows keepalive 设置
                if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
                    conn.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60000, 30000))
                
                conn.connect((self.cloud_host, self.cloud_port))
                
                self.cloud_inference_conn = conn
                self.last_connection_time = time.time()
                self.connection_failures = 0
                
                print(f"[边侧 {self.edge_id}] 持久连接已建立")
                
            except Exception as e:
                self.cloud_inference_conn = None
                raise Exception(f"建立云侧连接失败: {e}")
        
        return self.cloud_inference_conn
        
    def send_data(self, conn, data):
        """发送数据（带速率限制）
        
        改进的速率限制算法：
        - 使用更大的chunk_size (1MB) 减少循环开销
        - 累积计时，避免频繁的time.time()调用
        - 只在必要时sleep，提高精度
        """
        serialized = pickle.dumps(data)
        size = len(serialized)
        
        # 发送数据大小
        conn.sendall(struct.pack('Q', size))
        
        # 如果启用限速，分块发送
        if self.rate_limit_mbps:
            chunk_size = 1024 * 1024  # 1MB per chunk (原来是64KB)
            rate_limit_bps = self.rate_limit_mbps * 1024 * 1024  # 转换为 bytes/s
            
            sent = 0
            start_time = time.time()  # 记录整体开始时间
            
            while sent < size:
                chunk = serialized[sent:sent + chunk_size]
                conn.sendall(chunk)
                sent += len(chunk)
                
                # 每发送完一个chunk，检查是否需要限速
                elapsed = time.time() - start_time
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
        
        return size
        
    def receive_data(self, conn):
        """接收数据"""
        size_data = self._recv_all(conn, 8)
        size = struct.unpack('Q', size_data)[0]
        data = self._recv_all(conn, size)
        return pickle.loads(data), size
    
    def _recv_all(self, conn, size):
        """确保接收完整数据"""
        data = b''
        while len(data) < size:
            packet = conn.recv(min(size - len(data), 65536))
            if not packet:
                raise ConnectionError("连接断开")
            data += packet
        return data
    
    def wait_for_cloud_ready(self, max_wait=300, check_interval=5):
        """智能轮询检测云侧是否准备好"""
        print(f"[边侧 {self.edge_id}] 等待云侧准备数据和预训练教师模型...")
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < max_wait:
            attempt += 1
            try:
                test_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_conn.settimeout(2)
                test_conn.connect((self.cloud_host, self.cloud_port))
                test_conn.close()
                
                elapsed = int(time.time() - start_time)
                print(f"[边侧 {self.edge_id}] ✓ 云侧已准备好！（等待了 {elapsed} 秒）")
                return True
                
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                elapsed = int(time.time() - start_time)
                if attempt % 6 == 1:
                    print(f"[边侧 {self.edge_id}] 云侧未就绪，已等待 {elapsed} 秒...")
                time.sleep(check_interval)
        
        elapsed = int(time.time() - start_time)
        raise Exception(f"等待云侧超时（等待了 {elapsed} 秒）")
    
    def check_round_ready(self, round_num):
        """检查云侧是否准备好下一轮"""
        # 第一轮不需要检查，直接开始
        if round_num == 0:
            print(f"[边侧 {self.edge_id}] 第一轮，直接开始")
            return True
        
        max_retries = 60
        retry_interval = 5
        
        for i in range(max_retries):
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5)
                conn.connect((self.cloud_host, self.cloud_port))
                
                try:
                    self.send_data(conn, {
                        'type': 'check_ready',
                        'edge_id': self.edge_id,
                        'round': round_num
                    })
                    
                    response, _ = self.receive_data(conn)
                    
                    if response.get('status') == 'ready':
                        return True
                    elif response.get('status') == 'wait':
                        if i % 10 == 0:
                            print(f"[边侧 {self.edge_id}] 等待云侧准备下一轮... ({i+1}/{max_retries})")
                        time.sleep(retry_interval)
                    else:
                        time.sleep(retry_interval)
                        
                finally:
                    conn.close()
                    
            except Exception as e:
                if i % 10 == 0:
                    print(f"[边侧 {self.edge_id}] 等待云侧... ({i+1}/{max_retries})")
                time.sleep(retry_interval)
        
        raise Exception(f"等待云侧超时（{max_retries}秒）")
    
    def wait_for_phase_ready(self, phase, max_wait=300):
        """等待云侧进入指定阶段
        
        Args:
            phase: 期望的阶段 ('download' 或 'upload')
            max_wait: 最大等待时间（秒）
        """
        print(f"[边侧 {self.edge_id}] 等待云侧进入 {phase} 阶段...")
        start_time = time.time()
        check_interval = 2  # 每2秒检查一次
        
        while time.time() - start_time < max_wait:
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5)
                conn.connect((self.cloud_host, self.cloud_port))
                
                try:
                    self.send_data(conn, {
                        'type': 'check_phase',
                        'edge_id': self.edge_id,
                        'expected_phase': phase
                    })
                    
                    response, _ = self.receive_data(conn)
                    
                    if response.get('status') == 'ready':
                        current_phase = response.get('current_phase')
                        print(f"[边侧 {self.edge_id}] 云侧已进入 {current_phase} 阶段，可以继续")
                        return True
                    elif response.get('status') == 'wait':
                        current_phase = response.get('current_phase', 'unknown')
                        elapsed = int(time.time() - start_time)
                        if elapsed % 10 == 0:  # 每10秒打印一次
                            print(f"[边侧 {self.edge_id}] 云侧当前阶段: {current_phase}，等待进入 {phase} 阶段... ({elapsed}s)")
                        time.sleep(check_interval)
                    else:
                        time.sleep(check_interval)
                        
                finally:
                    conn.close()
                    
            except Exception as e:
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:
                    print(f"[边侧 {self.edge_id}] 检查阶段失败: {e}，继续等待... ({elapsed}s)")
                time.sleep(check_interval)
        
        raise Exception(f"等待云侧进入 {phase} 阶段超时（{max_wait}秒）")
    
    def download_model(self, round_num):
        """从云侧下载全局模型"""
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(300)
        conn.connect((self.cloud_host, self.cloud_port))
        
        try:
            self.send_data(conn, {
                'type': 'download',
                'edge_id': self.edge_id,
                'round': round_num
            })
            
            start_time = time.time()
            response, size = self.receive_data(conn)
            download_time = time.time() - start_time
            
            if response.get('status') == 'error':
                error_msg = response.get('message', '未知错误')
                if '不支持的请求类型' in error_msg or 'download' in error_msg.lower():
                    raise Exception(
                        f"\n{'='*70}\n"
                        f"云侧拒绝下载请求！\n"
                        f"错误: {error_msg}\n\n"
                        f"可能原因：\n"
                        f"  你启动的是 KD服务器（run_cloud_kd.py），\n"
                        f"  而不是 联邦学习服务器（run_cloud_federated.py）！\n\n"
                        f"解决方法：\n"
                        f"  1. 关闭当前云侧服务器（Ctrl+C）\n"
                        f"  2. 启动联邦学习服务器：\n"
                        f"     python run/cloud/run_cloud_federated.py \\\n"
                        f"       --dataset_type <数据集> \\\n"
                        f"       --num_classes <类别数> \\\n"
                        f"       --edge_model <边侧模型> \\\n"
                        f"       --num_edges <边侧数量> \\\n"
                        f"       --num_rounds <训练轮次>\n"
                        f"{'='*70}"
                    )
                else:
                    raise Exception(f"云侧拒绝: {error_msg}")
            
            model_state = response['model_state']
            
            print(f"[边侧 {self.edge_id}] 下载模型: {size/(1024*1024):.2f}MB, 耗时 {download_time:.2f}s")
            
            return model_state, download_time, size/(1024*1024)
            
        finally:
            conn.close()
    
    def upload_model(self, model_state, num_samples):
        """上传本地模型到云侧"""
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(300)
        conn.connect((self.cloud_host, self.cloud_port))
        
        try:
            start_time = time.time()
            size = self.send_data(conn, {
                'type': 'upload',
                'edge_id': self.edge_id,
                'model_state': model_state,
                'num_samples': num_samples
            })
            
            response, _ = self.receive_data(conn)
            upload_time = time.time() - start_time
            
            if response.get('status') == 'error':
                raise Exception(f"云侧拒绝: {response.get('message', '未知错误')}")
            
            print(f"[边侧 {self.edge_id}] 上传模型: {size/(1024*1024):.2f}MB, 耗时 {upload_time:.2f}s")
            
            return upload_time, size/(1024*1024)
            
        finally:
            conn.close()
    
    def _close_cloud_connection(self):
        """关闭云侧推理连接（同步模式）"""
        if self.cloud_inference_conn:
            try:
                print(f"[边侧 {self.edge_id}] 关闭云侧持久连接")
                self.cloud_inference_conn.close()
            except:
                pass
            self.cloud_inference_conn = None
    
    def _init_zmq(self, push_port=5555, pull_port=5556):
        """
        初始化 ZeroMQ 连接（用于异步协同推理）
        
        Args:
            push_port: 推送端口（边侧 PUSH → 云侧 PULL）
            pull_port: 拉取端口（云侧 PUSH → 边侧 PULL）
        """
        if self.zmq_initialized:
            return
        
        try:
            import zmq
            
            # 创建 ZeroMQ context
            self.zmq_context = zmq.Context()
            
            # PUSH socket: 发送请求到云侧
            self.zmq_push_socket = self.zmq_context.socket(zmq.PUSH)
            self.zmq_push_socket.connect(f"tcp://{self.cloud_host}:{push_port}")
            
            # PULL socket: 接收云侧响应
            self.zmq_pull_socket = self.zmq_context.socket(zmq.PULL)
            self.zmq_pull_socket.connect(f"tcp://{self.cloud_host}:{pull_port}")
            
            # 设置高水位标记（防止内存溢出）
            self.zmq_push_socket.setsockopt(zmq.SNDHWM, 1000)
            self.zmq_pull_socket.setsockopt(zmq.RCVHWM, 1000)
            
            # 尝试导入 MessagePack
            try:
                import msgpack
                import msgpack_numpy as m
                m.patch()
                self.use_msgpack = True
                print(f"[边侧 {self.edge_id}] 使用 MessagePack 序列化")
            except ImportError:
                self.use_msgpack = False
                print(f"[边侧 {self.edge_id}] 使用 Pickle 序列化")
            
            self.zmq_initialized = True
            
            print(f"[边侧 {self.edge_id}] ZeroMQ 连接已建立")
            print(f"  PUSH → tcp://{self.cloud_host}:{push_port}")
            print(f"  PULL ← tcp://{self.cloud_host}:{pull_port}")
            
        except Exception as e:
            print(f"[边侧 {self.edge_id}] ZeroMQ 初始化失败: {e}")
            raise
    
    def send_request_zmq(self, request_data):
        """
        使用 ZeroMQ 发送推理请求到云侧（非阻塞）
        
        Args:
            request_data: 请求数据字典
        """
        if not self.zmq_initialized:
            raise Exception("ZeroMQ 未初始化，请先调用 _init_zmq()")
        
        import zmq
        
        # 添加边侧ID
        request_data['edge_id'] = self.edge_id
        
        # 序列化并发送
        if self.use_msgpack:
            import msgpack
            serialized = msgpack.packb(request_data, use_bin_type=True)
        else:
            serialized = pickle.dumps(request_data)
        
        self.zmq_push_socket.send(serialized, flags=zmq.NOBLOCK)
    
    def receive_response_zmq(self, timeout_ms=500):
        """
        使用 ZeroMQ 接收云侧响应（带超时）
        
        Args:
            timeout_ms: 超时时间（毫秒）
            
        Returns:
            响应数据字典，如果超时返回 None
        """
        if not self.zmq_initialized:
            raise Exception("ZeroMQ 未初始化，请先调用 _init_zmq()")
        
        import zmq
        
        # 设置接收超时
        self.zmq_pull_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        try:
            serialized = self.zmq_pull_socket.recv()
            
            # 尝试 MessagePack 反序列化
            if self.use_msgpack:
                try:
                    import msgpack
                    response = msgpack.unpackb(serialized, raw=False)
                    return response
                except Exception:
                    # 如果失败，尝试 Pickle
                    pass
            
            # 使用 Pickle 反序列化
            response = pickle.loads(serialized)
            return response
        except zmq.Again:
            # 超时
            return None
    
    def close_zmq(self):
        """关闭 ZeroMQ 连接"""
        if self.zmq_initialized:
            print(f"[边侧 {self.edge_id}] 关闭 ZeroMQ 连接")
            if self.zmq_push_socket:
                self.zmq_push_socket.close()
            if self.zmq_pull_socket:
                self.zmq_pull_socket.close()
            if self.zmq_context:
                self.zmq_context.term()
            self.zmq_initialized = False
    
    def cloud_inference(self, inputs, dataset_type, num_classes):
        """
        请求云侧进行深度推理（使用持久连接）
        
        Args:
            inputs: 输入数据张量
            dataset_type: 数据集类型
            num_classes: 类别数
            
        Returns:
            predictions: 预测结果张量
            inference_time: 推理时间（毫秒）
        """
        import torch
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # 确保连接存在
                conn = self._ensure_cloud_connection()
                
                # 准备请求数据
                if torch.is_tensor(inputs):
                    inputs_np = inputs.cpu().numpy()
                else:
                    inputs_np = inputs
                
                request = {
                    'type': 'cloud_inference',
                    'inputs': inputs_np,
                    'dataset_type': dataset_type,
                    'num_classes': num_classes,
                    'edge_id': self.edge_id  # 添加边侧ID
                }
                
                # 发送请求
                start_time = time.time()
                size = self.send_data(conn, request)
                
                # 接收响应
                response, _ = self.receive_data(conn)
                inference_time = (time.time() - start_time) * 1000  # 毫秒
                
                if response.get('status') == 'success':
                    predictions = torch.tensor(response['predictions'])
                    self.connection_failures = 0  # 重置失败计数
                    return predictions, inference_time
                else:
                    error_msg = response.get('message', '未知错误')
                    raise Exception(f"云侧推理失败: {error_msg}")
                
            except (ConnectionError, OSError, BrokenPipeError, struct.error) as e:
                # 连接错误，关闭并重试
                self._close_cloud_connection()
                self.connection_failures += 1
                
                if attempt < max_retries - 1:
                    print(f"[边侧 {self.edge_id}] 云侧连接断开，重新连接 (尝试 {attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"云侧推理失败（已重试{max_retries}次）: {e}")
            
            except Exception as e:
                # 其他错误，也关闭连接
                self._close_cloud_connection()
                self.connection_failures += 1
                
                if attempt < max_retries - 1:
                    print(f"[边侧 {self.edge_id}] 云侧推理失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"云侧推理失败（已重试{max_retries}次）: {e}")
        
        raise Exception("云侧推理失败：超过最大重试次数")

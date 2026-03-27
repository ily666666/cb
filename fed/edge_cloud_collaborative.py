"""
边-云协同推理系统
Context-Aware Collaborative Inference for Edge-Cloud Systems

功能：
1. 边侧轻量模型（real_resnet20学生模型）进行初步推理
2. 基于置信度阈值动态决定是否上传到云端
3. 云侧强模型（complex_resnet50教师模型）进行深度推理
4. 评估不同置信度阈值下的系统性能

适配 project 框架：
- 支持多种数据集：ads, radioml, reii, radar, rml2016, link11
- 使用 project 中的模型结构：real_resnet20 (边侧) 和 complex_resnet50 (云侧)
- 兼容 project 的输入格式和预处理逻辑

使用方法：
    python edge_cloud_collaborative.py \\
        --dataset_type rml2016 \\
        --edge_model_path ./result/xxx/kd_trained_models/client_001_kd_model.pth \\
        --cloud_model_path ./result/xxx/pretrained_server_model.pth \\
        --data_path E:/BaiduNet_Download/rml2016.pkl \\
        --thresholds 0.5,0.6,0.7,0.8,0.9 \\
        --batch_size 32 \\
        --save_path result_collaborative

必需参数：
    --edge_model_path: 边侧模型路径（real_resnet20学生模型）
    --cloud_model_path: 云侧模型路径（complex_resnet50教师模型）

可选参数：
    --dataset_type: 数据集类型 (ads, radioml, reii, radar, rml2016, link11)，默认 ads
    --data_path: 数据集路径（如果不指定，会根据dataset_type使用默认路径）
    --num_classes: 类别数（如果不指定，会根据数据集类型自动确定）
    --batch_size: 批次大小，默认 32
    --thresholds: 置信度阈值列表，用逗号分隔，默认 '0.5,0.6,0.7,0.8,0.9'
    --num_batches: 评估批次数（None表示全部），默认 None
    --cloud_latency_ms: 云端推理延迟（毫秒），默认 50.0
    --bandwidth_mbps: 带宽（Mbps），默认 100.0
    --image_size_mb: 图像大小（MB），默认 0.1
    --save_path: 结果保存路径，默认 'result_edge_cloud_collaborative'

示例：
    1. RML2016数据集：
       python edge_cloud_collaborative.py \\
           --dataset_type rml2016 \\
           --edge_model_path ./result/xxx/kd_trained_models/client_001_kd_model.pth \\
           --cloud_model_path ./result/xxx/pretrained_server_model.pth

    2. Link11数据集：
       python edge_cloud_collaborative.py \\
           --dataset_type link11 \\
           --edge_model_path ./result/xxx/kd_trained_models/client_001_kd_model.pth \\
           --cloud_model_path ./result/xxx/pretrained_server_model.pth \\
           --thresholds 0.6,0.7,0.8 \\
           --num_batches 50

    3. 快速测试（只评估50个批次）：
       python edge_cloud_collaborative.py \\
           --dataset_type rml2016 \\
           --edge_model_path ./result/xxx/kd_trained_models/client_001_kd_model.pth \\
           --cloud_model_path ./result/xxx/pretrained_server_model.pth \\
           --num_batches 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class EdgeCloudCollaborativeInference:
    """
    边-云协同推理系统
    
    核心机制：
    - 边侧模型计算置信度 p_max = max(Softmax(z))
    - 若 p_max >= T：边侧直接输出
    - 若 p_max < T：上传到云端深度推理
    """
    
    def __init__(self, edge_model, cloud_model, device, dataset_type='ads',
                 cloud_latency_ms=50.0, bandwidth_mbps=100.0, 
                 image_size_mb=0.1):
        """
        初始化边-云协同推理系统
        
        Args:
            edge_model: 边侧轻量模型（real_resnet20学生模型）
            cloud_model: 云侧强模型（complex_resnet50教师模型）
            device: 计算设备
            dataset_type: 数据集类型（用于确定输入预处理方式）
            cloud_latency_ms: 云端推理延迟（毫秒），模拟网络延迟+推理时间
            bandwidth_mbps: 带宽（Mbps），用于计算传输时间
            image_size_mb: 图像大小（MB），用于计算传输时间
        """
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.device = device
        self.dataset_type = dataset_type
        
        # 模拟网络参数
        self.cloud_latency_ms = cloud_latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.image_size_mb = image_size_mb
        
        # 不同数据集的输入长度映射
        self.input_length_map = {
            'ads': 4096,
            'radioml': 128,
            'reii': 2000,
            'radar': 500,
            'rml2016': 600,
            'link11': 1024
        }
        
        # 统计信息
        self.stats = {
            'edge_correct': 0,
            'edge_total': 0,
            'cloud_correct': 0,
            'cloud_total': 0,
            'edge_inference_time': [],
            'cloud_inference_time': [],
            'total_time': []
        }
        
        # 设置模型为评估模式
        self.edge_model.eval()
        self.cloud_model.eval()
    
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算置信度（softmax最大概率）
        
        Args:
            logits: 模型输出的logits [batch_size, num_classes]
        
        Returns:
            最大概率值 [batch_size]
        """
        probs = F.softmax(logits, dim=1)
        p_max, _ = torch.max(probs, dim=1)
        return p_max
    
    def preprocess_for_edge_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        为边侧模型预处理输入（real_resnet20）
        real_resnet20模型会自动处理复数输入，转换为2通道实数
        
        Args:
            inputs: 输入数据（可能是复数或实数，各种形状）
        
        Returns:
            预处理后的输入（复数张量，shape: [batch_size, length]）
        """
        # RML2016数据集：输入已经是 (batch_size, 600) 复数，直接返回
        if self.dataset_type == 'rml2016':
            # 确保在设备上
            inputs = inputs.to(self.device)
            # 确保是复数格式（RML2016Dataset返回的已经是复数）
            if not torch.is_complex(inputs):
                # 如果是实数，转换为复数（虚部为0）
                if inputs.dim() == 2:
                    inputs_imag = torch.zeros_like(inputs)
                    inputs = torch.view_as_complex(torch.stack([inputs, inputs_imag], dim=-1))
            return inputs
        
        # 其他数据集的预处理逻辑
        # 确保输入是复数格式
        if not torch.is_complex(inputs):
            if inputs.dim() == 2:
                # [batch, length] -> 转为复数
                inputs_real = inputs
                inputs_imag = torch.zeros_like(inputs_real)
                inputs = torch.view_as_complex(torch.stack([inputs_real, inputs_imag], dim=-1))
            elif inputs.dim() == 3:
                # [batch, 2, length] -> 转为复数
                if inputs.shape[1] == 2:
                    inputs = torch.view_as_complex(torch.stack([inputs[:, 0], inputs[:, 1]], dim=-1))
                else:
                    inputs_real = inputs[:, 0, :] if inputs.shape[1] > 0 else inputs.squeeze(1)
                    inputs_imag = torch.zeros_like(inputs_real)
                    inputs = torch.view_as_complex(torch.stack([inputs_real, inputs_imag], dim=-1))
        
        # 确保shape是 [batch_size, length]
        if inputs.dim() == 3:
            batch_size, channels, length = inputs.shape
            if channels == 1:
                inputs = inputs.squeeze(1)
            else:
                # 取第一个通道或合并
                inputs = inputs[:, 0, :] if channels > 0 else inputs.mean(dim=1)
        
        # 移动到设备
        inputs = inputs.to(self.device)
        
        return inputs
    
    def preprocess_for_cloud_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        为云侧模型预处理输入（complex_resnet50）
        complex_resnet50模型需要1通道复数输入 [batch, 1, H, W]
        
        Args:
            inputs: 输入数据（可能是复数或实数，各种形状）
        
        Returns:
            预处理后的输入（复数张量，shape: [batch, 1, H, W]）
        """
        # 移动到设备
        inputs = inputs.to(self.device)
        
        # 处理不同数据集的输入格式
        if self.dataset_type == 'rml2016':
            # RML2016: 输入应该是 (batch_size, 600) 复数
            # complex_resnet50_rml2016 期望 (batch_size, 600) 复数，然后内部会 reshape 为 (batch_size, 1, 20, 30)
            if inputs.dim() == 2:
                # 检查是否是复数
                if not torch.is_complex(inputs):
                    # 如果是实数，可能是 (batch_size, 600) 或 (batch_size, 1200)
                    # 需要转换为复数
                    if inputs.shape[1] == 600:
                        # (batch_size, 600) 实数 -> (batch_size, 600) 复数（虚部为0）
                        inputs_imag = torch.zeros_like(inputs)
                        inputs = torch.view_as_complex(torch.stack([inputs, inputs_imag], dim=-1))
                    elif inputs.shape[1] == 1200:
                        # (batch_size, 1200) 可能是 flattened 的 (batch_size, 2, 600)
                        # reshape 为 (batch_size, 2, 600) 然后转换为复数
                        inputs = inputs.view(inputs.shape[0], 2, 600)
                        inputs = torch.view_as_complex(torch.stack([inputs[:, 0], inputs[:, 1]], dim=-1))
                # 确保长度是 600
                if inputs.shape[1] != 600:
                    # 如果长度不对，尝试 reshape 或截断/填充
                    if inputs.shape[1] > 600:
                        inputs = inputs[:, :600]
                    else:
                        # 填充到 600
                        pad_length = 600 - inputs.shape[1]
                        inputs = torch.nn.functional.pad(inputs, (0, pad_length), mode='constant', value=0)
                return inputs
            elif inputs.dim() == 3:
                # 可能是 (batch_size, 2, 600) 实数，需要转换为复数
                if inputs.shape[2] == 600:
                    if inputs.shape[1] == 2:
                        # (batch_size, 2, 600) -> (batch_size, 600) 复数
                        inputs = torch.view_as_complex(torch.stack([inputs[:, 0], inputs[:, 1]], dim=-1))
                    elif inputs.shape[1] == 1:
                        # (batch_size, 1, 600) -> (batch_size, 600) 复数（虚部为0）
                        inputs = inputs.squeeze(1)
                        inputs_imag = torch.zeros_like(inputs)
                        inputs = torch.view_as_complex(torch.stack([inputs, inputs_imag], dim=-1))
                return inputs
            else:
                # 其他维度，尝试 flatten 或 reshape
                inputs = inputs.flatten(start_dim=1)
                if inputs.shape[1] > 600:
                    inputs = inputs[:, :600]
                elif inputs.shape[1] < 600:
                    pad_length = 600 - inputs.shape[1]
                    inputs = torch.nn.functional.pad(inputs, (0, pad_length), mode='constant', value=0)
                # 转换为复数
                if not torch.is_complex(inputs):
                    inputs_imag = torch.zeros_like(inputs)
                    inputs = torch.view_as_complex(torch.stack([inputs, inputs_imag], dim=-1))
                return inputs
        
        # 其他数据集的预处理逻辑
        # 确保输入是复数格式
        if not torch.is_complex(inputs):
            if inputs.dim() == 2:
                inputs_real = inputs
                inputs_imag = torch.zeros_like(inputs_real)
                inputs = torch.view_as_complex(torch.stack([inputs_real, inputs_imag], dim=-1))
            elif inputs.dim() == 3:
                if inputs.shape[1] == 2:
                    inputs = torch.view_as_complex(torch.stack([inputs[:, 0], inputs[:, 1]], dim=-1))
        
        # 确定目标2D尺寸（根据数据集类型）
        target_length = self.input_length_map.get(self.dataset_type, 4096)
        min_size = 32
        
        if inputs.dim() == 2:
            # [batch, length] -> [batch, 1, H, W]
            batch_size = inputs.shape[0]
            length = inputs.shape[1]
            
            # 计算2D尺寸
            if length == 4096:
                h, w = 64, 64
            elif length == 128:
                h, w = 8, 16
            elif length == 2000:
                h, w = 40, 50
            elif length == 500:
                h, w = 20, 25
            elif length == 600:
                h, w = 20, 30
            elif length == 1024:
                h, w = 32, 32
            else:
                # 自动计算
                import math
                sqrt_len = int(math.sqrt(length))
                h, w = sqrt_len, sqrt_len
                for h_candidate in range(sqrt_len, 0, -1):
                    if length % h_candidate == 0:
                        h = h_candidate
                        w = length // h_candidate
                        if h >= min_size and w >= min_size:
                            break
                if h < min_size or w < min_size:
                    h, w = min_size, (length + min_size - 1) // min_size
                    target_size = h * w
                    if target_size > length:
                        pad_length = target_size - length
                        inputs = torch.nn.functional.pad(inputs, (0, pad_length), mode='constant', value=0)
            
            inputs = inputs.view(batch_size, 1, h, w)
            
        elif inputs.dim() == 3:
            # [batch, channels, length] -> [batch, 1, H, W]
            batch_size, channels, length = inputs.shape
            if channels != 1:
                if channels > 1:
                    inputs = inputs[:, 0:1, :]  # 取第一个通道
            
            # 计算2D尺寸（同上逻辑）
            if length == 4096:
                h, w = 64, 64
            elif length == 128:
                h, w = 8, 16
            elif length == 2000:
                h, w = 40, 50
            elif length == 500:
                h, w = 20, 25
            elif length == 600:
                h, w = 20, 30
            elif length == 1024:
                h, w = 32, 32
            else:
                import math
                sqrt_len = int(math.sqrt(length))
                h, w = sqrt_len, sqrt_len
                for h_candidate in range(sqrt_len, 0, -1):
                    if length % h_candidate == 0:
                        h = h_candidate
                        w = length // h_candidate
                        if h >= min_size and w >= min_size:
                            break
                if h < min_size or w < min_size:
                    h, w = min_size, (length + min_size - 1) // min_size
                    target_size = h * w
                    if target_size > length:
                        pad_length = target_size - length
                        inputs = torch.nn.functional.pad(inputs, (0, pad_length), mode='constant', value=0)
            
            inputs = inputs.view(batch_size, h, w).unsqueeze(1)
        
        # 如果尺寸太小，进行插值
        if inputs.shape[2] < min_size or inputs.shape[3] < min_size:
            input_real = torch.cat([inputs.real, inputs.imag], dim=1)
            input_real = torch.nn.functional.interpolate(
                input_real, size=(min_size, min_size), mode='bilinear', align_corners=False
            )
            inputs = torch.view_as_complex(
                torch.stack([input_real[:, 0], input_real[:, 1]], dim=-1)
            )
            inputs = inputs.unsqueeze(1)
        
        return inputs
    
    def edge_inference(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        边侧推理（real_resnet20学生模型）
        使用与训练代码相同的输入处理方式
        
        Args:
            inputs: 输入数据（对于RML2016，应该是 (batch_size, 600) 复数）
        
        Returns:
            logits: 模型输出logits
            predictions: 预测类别
            inference_time: 推理时间（毫秒）
        """
        start_time = time.time()
        
        with torch.no_grad():
            # RML2016数据集：输入已经是 (batch_size, 600) 复数，直接使用
            # 其他数据集：需要预处理
            if self.dataset_type == 'rml2016':
                # 确保在设备上且是复数格式
                processed_inputs = inputs.to(self.device)
                if not torch.is_complex(processed_inputs):
                    # 如果是实数，转换为复数（虚部为0）
                    inputs_imag = torch.zeros_like(processed_inputs)
                    processed_inputs = torch.view_as_complex(torch.stack([processed_inputs, inputs_imag], dim=-1))
            else:
                # 其他数据集：使用预处理函数
                processed_inputs = self.preprocess_for_edge_model(inputs)
            
            # 边侧模型推理（模型会自动处理复数->实数转换）
            # real_resnet20_rml2016 的 forward 方法期望 (batch_size, 600) 复数
            logits = self.edge_model(processed_inputs)
            
            # 处理复数输出（如果有）
            if torch.is_complex(logits):
                logits = torch.abs(logits)
            
            predictions = torch.argmax(logits, dim=1)
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        return logits, predictions, inference_time
    
    def cloud_inference(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        云侧推理（complex_resnet50教师模型）
        使用与训练代码相同的输入处理方式
        
        Args:
            inputs: 输入数据（对于RML2016，应该是 (batch_size, 600) 复数）
        
        Returns:
            predictions: 预测类别
            total_time: 总时间（传输+推理，毫秒）
        """
        batch_size = inputs.shape[0]
        
        # 模拟传输时间（批量传输）
        single_sample_transmission_ms = (self.image_size_mb * 8) / self.bandwidth_mbps * 1000
        transmission_time_ms = single_sample_transmission_ms  # 批量传输，时间不累加
        
        # 云端推理时间
        start_time = time.time()
        with torch.no_grad():
            # RML2016数据集：输入已经是 (batch_size, 600) 复数，直接使用
            # 其他数据集：需要预处理
            if self.dataset_type == 'rml2016':
                # 确保在设备上且是复数格式
                processed_inputs = inputs.to(self.device)
                if not torch.is_complex(processed_inputs):
                    # 如果是实数，转换为复数（虚部为0）
                    inputs_imag = torch.zeros_like(processed_inputs)
                    processed_inputs = torch.view_as_complex(torch.stack([processed_inputs, inputs_imag], dim=-1))
            else:
                # 其他数据集：使用预处理函数
                processed_inputs = self.preprocess_for_cloud_model(inputs)
            
            # 云侧模型推理
            # complex_resnet50_rml2016 的 forward 方法期望 (batch_size, 600) 复数
            logits = self.cloud_model(processed_inputs)
            
            # 处理复数输出
            if torch.is_complex(logits):
                logits = torch.abs(logits)
            
            predictions = torch.argmax(logits, dim=1)
        
        inference_time_ms = (time.time() - start_time) * 1000
        total_time_ms = transmission_time_ms + inference_time_ms + self.cloud_latency_ms
        
        return predictions, total_time_ms
    
    def context_aware_inference(self, inputs: torch.Tensor, targets: torch.Tensor, 
                                threshold: float) -> Dict:
        """
        上下文感知协同推理（核心函数）
        
        Args:
            inputs: 输入数据
            targets: 真实标签
            threshold: 置信度阈值 T
        
        Returns:
            推理结果字典
        """
        batch_size = inputs.shape[0]
        results = {
            'predictions': torch.zeros(batch_size, dtype=torch.long, device=self.device),
            'sources': [],  # 'edge' 或 'cloud'
            'confidences': [],
            'edge_correct': 0,
            'cloud_correct': 0,
            'edge_count': 0,
            'cloud_count': 0,
            'edge_time': 0.0,
            'cloud_time': 0.0,
            'total_time': 0.0
        }
        
        # 边侧推理
        logits_edge, preds_edge, edge_time = self.edge_inference(inputs)
        confidences = self.compute_confidence(logits_edge)
        
        # 决定哪些样本上传到云端
        cloud_mask = confidences < threshold
        edge_mask = ~cloud_mask
        
        # 边侧直接输出
        if edge_mask.any():
            edge_indices = torch.where(edge_mask)[0]
            results['predictions'][edge_indices] = preds_edge[edge_indices]
            results['edge_count'] = edge_mask.sum().item()
            results['edge_correct'] = (preds_edge[edge_indices] == targets[edge_indices]).sum().item()
            results['edge_time'] = edge_time
            results['sources'].extend(['edge'] * results['edge_count'])
            results['confidences'].extend(confidences[edge_indices].cpu().tolist())
        
        # 云端深度推理
        if cloud_mask.any():
            cloud_indices = torch.where(cloud_mask)[0]
            cloud_inputs = inputs[cloud_indices]
            cloud_targets = targets[cloud_indices]
            
            preds_cloud, cloud_time = self.cloud_inference(cloud_inputs)
            results['predictions'][cloud_indices] = preds_cloud
            results['cloud_count'] = cloud_mask.sum().item()
            results['cloud_correct'] = (preds_cloud == cloud_targets).sum().item()
            results['cloud_time'] = cloud_time
            results['sources'].extend(['cloud'] * results['cloud_count'])
            results['confidences'].extend(confidences[cloud_indices].cpu().tolist())
        
        # 计算总时间
        results['total_time'] = results['edge_time'] + results['cloud_time']
        
        return results
    
    def evaluate(self, dataloader, threshold: float, num_batches: Optional[int] = None) -> Dict:
        """
        评估系统性能
        
        Args:
            dataloader: 数据加载器
            threshold: 置信度阈值
            num_batches: 评估的批次数（None表示全部）
        
        Returns:
            评估结果字典
        """
        self.stats = {
            'edge_correct': 0,
            'edge_total': 0,
            'cloud_correct': 0,
            'cloud_total': 0,
            'edge_inference_time': [],
            'cloud_inference_time': [],
            'total_time': [],
            'confidences': []
        }
        
        total_correct = 0
        total_samples = 0
        
        print(f"\n评估置信度阈值 T = {threshold:.2f}...")
        if num_batches is not None:
            print(f"  将评估 {num_batches} 个批次（约 {num_batches * dataloader.batch_size} 样本）")
        
        batch_count = 0
        total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
        
        for batch_idx, batch in enumerate(dataloader):
            if num_batches is not None and batch_count >= num_batches:
                break
            
            # 处理batch格式（兼容不同数据集的返回格式）
            if len(batch) == 3:
                inputs, targets, _ = batch
            elif len(batch) == 2:
                inputs, targets = batch
            else:
                continue
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 上下文感知推理
            results = self.context_aware_inference(inputs, targets, threshold)
            
            # 统计准确率
            batch_correct = (results['predictions'] == targets).sum().item()
            total_correct += batch_correct
            total_samples += targets.size(0)
            
            # 统计边侧和云侧
            self.stats['edge_correct'] += results['edge_correct']
            self.stats['edge_total'] += results['edge_count']
            self.stats['cloud_correct'] += results['cloud_correct']
            self.stats['cloud_total'] += results['cloud_count']
            
            # 统计时间
            if results['edge_time'] > 0:
                self.stats['edge_inference_time'].append(results['edge_time'])
            if results['cloud_time'] > 0:
                self.stats['cloud_inference_time'].append(results['cloud_time'])
            self.stats['total_time'].append(results['total_time'])
            self.stats['confidences'].extend(results['confidences'])
            
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = 100 * (batch_idx + 1) / total_batches if total_batches > 0 else 0
                print(f"  进度: {progress:.1f}% ({batch_idx + 1}/{total_batches} 批次) | "
                      f"准确率 {100 * total_correct / total_samples:.2f}% | "
                      f"边侧 {self.stats['edge_total']}, 云侧 {self.stats['cloud_total']}")
        
        # 计算最终指标
        overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
        edge_accuracy = 100 * self.stats['edge_correct'] / self.stats['edge_total'] if self.stats['edge_total'] > 0 else 0
        cloud_accuracy = 100 * self.stats['cloud_correct'] / self.stats['cloud_total'] if self.stats['cloud_total'] > 0 else 0
        cloud_ratio = self.stats['cloud_total'] / total_samples if total_samples > 0 else 0
        
        # 计算平均时间（按batch）
        avg_edge_time_batch = np.mean(self.stats['edge_inference_time']) if self.stats['edge_inference_time'] else 0
        avg_cloud_time_batch = np.mean(self.stats['cloud_inference_time']) if self.stats['cloud_inference_time'] else 0
        avg_total_time_batch = np.mean(self.stats['total_time']) if self.stats['total_time'] else 0
        
        # 计算每个样本的平均时间
        if total_samples > 0 and len(self.stats['total_time']) > 0:
            estimated_batch_size = total_samples / len(self.stats['total_time'])
            avg_edge_time_per_sample = avg_edge_time_batch / estimated_batch_size if estimated_batch_size > 0 else avg_edge_time_batch
            
            if self.stats['cloud_total'] > 0 and len(self.stats['cloud_inference_time']) > 0:
                avg_cloud_samples_per_batch = self.stats['cloud_total'] / len(self.stats['cloud_inference_time'])
                avg_cloud_time_per_sample = avg_cloud_time_batch / avg_cloud_samples_per_batch if avg_cloud_samples_per_batch > 0 else avg_cloud_time_batch
            else:
                avg_cloud_time_per_sample = 0
            
            if total_samples > 0:
                avg_per_sample_latency = (
                    self.stats['edge_total'] * avg_edge_time_per_sample + 
                    self.stats['cloud_total'] * avg_cloud_time_per_sample
                ) / total_samples
            else:
                avg_per_sample_latency = avg_edge_time_per_sample
            
            avg_total_time_per_sample = avg_per_sample_latency
        else:
            estimated_batch_size = 0
            avg_edge_time_per_sample = avg_edge_time_batch
            avg_cloud_time_per_sample = avg_cloud_time_batch if avg_cloud_time_batch > 0 else 0
            avg_per_sample_latency = avg_edge_time_per_sample
            avg_total_time_per_sample = avg_per_sample_latency
        
        # 计算速度提升
        if avg_per_sample_latency > 0 and avg_cloud_time_per_sample > 0:
            speedup_ratio = avg_cloud_time_per_sample / avg_per_sample_latency
            speedup_percentage = (1 - avg_per_sample_latency / avg_cloud_time_per_sample) * 100
        else:
            speedup_ratio = 0
            speedup_percentage = 0
        
        results_summary = {
            'threshold': threshold,
            'overall_accuracy': overall_accuracy,
            'edge_accuracy': edge_accuracy,
            'cloud_accuracy': cloud_accuracy,
            'edge_total': self.stats['edge_total'],
            'cloud_total': self.stats['cloud_total'],
            'cloud_ratio': cloud_ratio,
            'avg_edge_time_ms_batch': avg_edge_time_batch,
            'avg_cloud_time_ms_batch': avg_cloud_time_batch,
            'avg_total_time_ms_batch': avg_total_time_batch,
            'avg_edge_time_ms': avg_edge_time_per_sample,
            'avg_cloud_time_ms': avg_cloud_time_per_sample,
            'avg_total_time_ms': avg_total_time_batch,
            'avg_total_time_ms_per_sample': avg_total_time_per_sample,
            'avg_per_sample_latency_ms': avg_per_sample_latency,
            'estimated_batch_size': estimated_batch_size,
            'speedup_ratio': speedup_ratio,
            'speedup_percentage': speedup_percentage,
            'total_samples': total_samples
        }
        
        return results_summary


def plot_collaborative_results(results_list: List[Dict], save_path: str):
    """
    绘制协同推理实验结果
    
    Args:
        results_list: 不同阈值下的结果列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    thresholds = [r['threshold'] for r in results_list]
    accuracies = [r['overall_accuracy'] for r in results_list]
    cloud_ratios = [r['cloud_ratio'] for r in results_list]
    edge_accs = [r['edge_accuracy'] for r in results_list]
    cloud_accs = [r['cloud_accuracy'] for r in results_list]
    avg_times = [r['avg_total_time_ms'] for r in results_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 准确率 vs 云端调用率
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(thresholds, accuracies, 'b-o', label='Overall Accuracy', linewidth=2)
    line2 = ax1_twin.plot(thresholds, cloud_ratios, 'r-s', label='Cloud Ratio', linewidth=2)
    ax1.set_xlabel('Confidence Threshold T', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='b')
    ax1_twin.set_ylabel('Cloud Offloading Ratio', fontsize=12, color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Accuracy vs Cloud Offloading Ratio', fontsize=14, fontweight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 2. 边侧和云侧准确率对比
    ax2 = axes[0, 1]
    ax2.plot(thresholds, edge_accs, 'g-o', label='Edge Accuracy', linewidth=2, markersize=8)
    ax2.plot(thresholds, cloud_accs, 'm-s', label='Cloud Accuracy', linewidth=2, markersize=8)
    ax2.plot(thresholds, accuracies, 'b-^', label='Overall Accuracy', linewidth=2, markersize=8)
    ax2.set_xlabel('Confidence Threshold T', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Edge vs Cloud Accuracy', fontsize=14, fontweight='bold')
    
    # 3. 延迟分析
    ax3 = axes[1, 0]
    ax3.plot(thresholds, avg_times, 'purple', marker='o', linewidth=2, markersize=8, label='Average Total Time')
    ax3.set_xlabel('Confidence Threshold T', fontsize=12)
    ax3.set_ylabel('Average Time (ms)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Average Inference Time', fontsize=14, fontweight='bold')
    
    # 4. 准确率 vs 云端调用率（散点图）
    ax4 = axes[1, 1]
    scatter = ax4.scatter(cloud_ratios, accuracies, c=thresholds, cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Cloud Offloading Ratio', fontsize=12)
    ax4.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Accuracy vs Cloud Ratio (Trade-off)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Threshold T', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'edge_cloud_collaborative_results.png'), 
                dpi=300, bbox_inches='tight')
    print(f"实验结果图表已保存到: {os.path.join(save_path, 'edge_cloud_collaborative_results.png')}")
    plt.close()


def main():
    """
    边-云协同推理系统主函数
    可以直接运行此脚本进行协同推理实验
    """
    import argparse
    import sys
    import os
    import json
    import platform
    from torch.utils.data import DataLoader
    import importlib
    
    # 导入 project 框架的模型创建函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from project import create_model_by_type

    # 兼容旧checkpoint：历史上模型保存时可能pickle了旧模块路径（如 readdata_rml2016）
    # 但当前项目已将数据加载器移到 utils/ 下，会导致 torch.load 反序列化失败。
    # 这里通过 sys.modules 注册别名，让 unpickler 能找到对应模块。
    legacy_to_new = {
        'readdata_25': 'utils.readdata_25',
        'readdata_radioml': 'utils.readdata_radioml',
        'readdata_reii': 'utils.readdata_reii',
        'readdata_radar': 'utils.readdata_radar',
        'readdata_rml2016': 'utils.readdata_rml2016',
        'readdata_link11': 'utils.readdata_link11',
    }
    for legacy_name, new_name in legacy_to_new.items():
        if legacy_name in sys.modules:
            continue
        try:
            sys.modules[legacy_name] = importlib.import_module(new_name)
        except ImportError:
            # 如果 utils 下也不存在对应模块，就不注册别名
            pass
    
    parser = argparse.ArgumentParser(description='边-云协同推理系统实验（适配 project 框架）')
    
    # 数据集类型
    parser.add_argument('--dataset_type', type=str, default='ads',
                        choices=['ads', 'radioml', 'reii', 'radar', 'rml2016', 'link11'],
                        help='数据集类型 (default: ads)')
    
    # 模型路径
    parser.add_argument('--edge_model_path', type=str, default='',
                        help='边侧模型路径（real_resnet20学生模型，训练好的模型文件）')
    parser.add_argument('--cloud_model_path', type=str, default='',
                        help='云侧模型路径（complex_resnet50教师模型，训练好的模型文件）')
    
    # 数据集路径（根据数据集类型设置默认路径）
    parser.add_argument('--data_path', type=str, default='',
                        help='数据集路径（根据dataset_type自动设置，也可手动指定）')
    
    # 数据集参数
    parser.add_argument('--num_classes', type=int, default=None,
                        help='类别数量（如果不指定，会根据数据集自动确定）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (default: 32)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载器工作线程数 (default: 0，Windows建议使用0)')
    
    # SNR过滤（仅对radioml和rml2016有效）
    def parse_snr_value(value):
        """解析SNR值，支持None字符串"""
        if value is None or value == 'None' or value == 'none' or value == '':
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    parser.add_argument('--radioml_snr_min', type=str, default=None,
                        help='RadioML/RML2016最小SNR阈值（dB），None或不指定表示不过滤。注意：如果训练时使用了SNR过滤，评估时也应使用相同的过滤')
    parser.add_argument('--radioml_snr_max', type=str, default=None,
                        help='RadioML/RML2016最大SNR阈值（dB），None或不指定表示不过滤。注意：如果训练时使用了SNR过滤，评估时也应使用相同的过滤')
    
    # 自动从训练结果目录读取配置（如果模型路径在result目录下）
    parser.add_argument('--auto_load_config', action='store_true', default=True,
                        help='自动从训练结果目录的config.json读取SNR过滤等参数（默认启用）')
    
    # 实验参数
    parser.add_argument('--thresholds', type=str, default='0.5,0.6,0.7,0.8,0.9',
                        help='置信度阈值列表，用逗号分隔（例如：0.5,0.7,0.9）')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='评估的批次数（None表示全部，如需快速测试可设置为50）')
    
    # 网络参数（模拟）
    parser.add_argument('--cloud_latency_ms', type=float, default=50.0,
                        help='云端推理延迟（毫秒） (default: 50.0)')
    parser.add_argument('--bandwidth_mbps', type=float, default=100.0,
                        help='带宽（Mbps） (default: 100.0)')
    parser.add_argument('--image_size_mb', type=float, default=0.1,
                        help='图像大小（MB） (default: 0.1)')
    
    # 保存路径
    parser.add_argument('--save_path', type=str, default='result_edge_cloud_collaborative',
                        help='结果保存路径 (default: result_edge_cloud_collaborative)')
    
    args = parser.parse_args()
    
    # 自动从训练结果目录读取配置（如果模型路径在result目录下）
    if args.auto_load_config and args.edge_model_path:
        # 尝试从模型路径推断训练结果目录
        edge_model_dir = os.path.dirname(os.path.abspath(args.edge_model_path))
        # 如果路径包含 kd_trained_models，则上一级目录是结果目录
        if 'kd_trained_models' in edge_model_dir:
            result_dir = os.path.dirname(edge_model_dir)
            config_path = os.path.join(result_dir, 'config.json')
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        train_config = json.load(f)
                    
                    print(f"\n[INFO] 从训练配置加载参数: {config_path}")
                    
                    # 读取SNR过滤参数（但需要验证数据集中是否有该范围的SNR）
                    # 注意：如果训练配置中的SNR范围与数据集不匹配，会导致测试集为空
                    if args.radioml_snr_min is None and 'radioml_snr_min' in train_config:
                        config_snr_min = train_config['radioml_snr_min']
                        # 先不自动设置，因为可能不匹配
                        print(f"   训练配置中的SNR范围: [{config_snr_min}, {train_config.get('radioml_snr_max', 'N/A')}] dB")
                        print(f"   [WARNING] 注意：如果数据集中没有该SNR范围的数据，测试集将为空")
                        print(f"   建议：先不使用SNR过滤进行评估，或检查数据集的实际SNR范围")
                        # 暂时不自动设置，让用户手动决定
                        # args.radioml_snr_min = config_snr_min
                    
                    if args.radioml_snr_max is None and 'radioml_snr_max' in train_config:
                        config_snr_max = train_config['radioml_snr_max']
                        # 暂时不自动设置
                        # args.radioml_snr_max = config_snr_max
                    
                    # 注意：训练时可能使用了噪声，但评估时通常不加噪声
                    # 如果需要完全一致，可以添加 --use_training_noise 参数
                    if 'add_noise' in train_config and train_config['add_noise']:
                        print(f"   [WARNING] 训练时使用了噪声 (type={train_config.get('noise_type', 'unknown')}, factor={train_config.get('noise_factor', 'unknown')})")
                        print(f"   评估时默认不加噪声，如需一致请手动设置")
                    
                except Exception as e:
                    print(f"   [WARNING] 读取训练配置失败: {e}")
    
    # 设置默认数据集路径（根据数据集类型）
    if not args.data_path:
        default_paths = {
            'ads': r'E:\ADS-B_6000_100class\\',
            'radioml': r'E:\BaiduNet_Download\augmented_data.pkl',
            'reii': r'E:\BaiduNet_Download\REII\\',
            'radar': r'E:\BaiduNet_Download\Radar Emitter Individual Identification\Radar Emitter Individual Identification\dataGen\RadarDataset_20251124_144839_161000samples_Repeat1.mat',
            'rml2016': r'E:\BaiduNet_Download\rml2016.pkl',
            'link11': r'E:\BaiduNet_Download\link11.pkl'
        }
        args.data_path = default_paths.get(args.dataset_type, '')
        if not args.data_path:
            print(f"错误: 数据集类型 {args.dataset_type} 没有默认路径，请使用 --data_path 指定")
            return
    
    # 设置默认类别数（根据数据集类型）
    if args.num_classes is None:
        class_map = {
            'ads': 100,
            'radioml': 11,
            'reii': 3,
            'radar': 7,
            'rml2016': 6,
            'link11': 7
        }
        args.num_classes = class_map.get(args.dataset_type, 10)
    
    # 检查必需参数
    if not args.edge_model_path:
        print("错误: 请提供 --edge_model_path 参数（边侧模型路径）")
        print("示例: --edge_model_path ./result/xxx/kd_trained_models/client_001_kd_model.pth")
        return
    
    if not args.cloud_model_path:
        print("错误: 请提供 --cloud_model_path 参数（云侧模型路径）")
        print("示例: --cloud_model_path ./result/xxx/pretrained_server_model.pth")
        return
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 打印配置信息
    print(f"\n{'='*80}")
    print(f"[CONFIG] 边-云协同推理系统配置（适配 project 框架）")
    print(f"{'='*80}")
    print(f"[INFO] 数据集类型: {args.dataset_type}")
    print(f"[INFO] 数据集路径: {args.data_path}")
    print(f"[INFO] 边侧模型: {args.edge_model_path}")
    print(f"   模型类型: real_resnet20_{args.dataset_type}")
    print(f"[INFO] 云侧模型: {args.cloud_model_path}")
    print(f"   模型类型: complex_resnet50_{args.dataset_type}")
    print(f"[INFO] 类别数: {args.num_classes}")
    print(f"[INFO] 批次大小: {args.batch_size}")
    # 解析SNR参数用于显示
    def parse_snr_display(value):
        if value is None or value == 'None' or value == 'none' or value == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    snr_min_display = parse_snr_display(args.radioml_snr_min)
    snr_max_display = parse_snr_display(args.radioml_snr_max)
    if snr_min_display is not None and snr_max_display is not None:
        print(f"[INFO] SNR过滤: [{snr_min_display}, {snr_max_display}] dB")
    else:
        print(f"[INFO] SNR过滤: 无（使用所有SNR范围）")
    print(f"{'='*80}\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"\n{'='*80}")
    print(f"[DATA] 加载数据集")
    print(f"{'='*80}")
    
    # 解析SNR过滤参数
    def parse_snr_value(value):
        """解析SNR值，支持None字符串"""
        if value is None or value == 'None' or value == 'none' or value == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    snr_min = parse_snr_value(args.radioml_snr_min)
    snr_max = parse_snr_value(args.radioml_snr_max)
    
    snr_filter = None
    if snr_min is not None and snr_max is not None:
        snr_filter = (snr_min, snr_max)
        print(f"   使用SNR过滤: [{snr_min}, {snr_max}] dB")
    else:
        print(f"   不使用SNR过滤（使用所有SNR范围的数据）")
    
    try:
        if args.dataset_type == 'radioml':
            try:
                from utils.readdata_radioml import RadioMLDataset
            except ImportError:
                from readdata_radioml import RadioMLDataset
            test_dataset = RadioMLDataset(datapath=args.data_path, split='test', 
                                         transform=None, snr_filter=snr_filter)
        elif args.dataset_type == 'rml2016':
            try:
                from utils.readdata_rml2016 import RML2016Dataset
            except ImportError:
                from readdata_rml2016 import RML2016Dataset
            # 检查是否有训练配置文件，如果有则使用相同的参数
            # 默认不使用SNR过滤和噪声（与训练时可能不一致，但更通用）
            # 如果需要与训练时完全一致，需要从config.json读取参数
            test_dataset = RML2016Dataset(
                pkl_path=args.data_path, 
                split='test',
                snr_range=snr_filter,  # 如果提供了SNR过滤参数则使用
                seed=42,
                add_noise=False,  # 评估时默认不加噪声
                noise_type='awgn',
                noise_snr_db=15,
                noise_factor=0.1
            )
        elif args.dataset_type == 'link11':
            try:
                from utils.readdata_link11 import Link11Dataset
            except ImportError:
                from readdata_link11 import Link11Dataset
            test_dataset = Link11Dataset(pkl_path=args.data_path, split='test',
                                        snr_range=snr_filter, seed=42)
        elif args.dataset_type == 'reii':
            try:
                from utils.readdata_reii import REIIDataset
            except ImportError:
                from readdata_reii import REIIDataset
            test_dataset = REIIDataset(datapath=args.data_path, split='test', transform=None)
        elif args.dataset_type == 'radar':
            try:
                from utils.readdata_radar import RadarDataset
            except ImportError:
                from readdata_radar import RadarDataset
            test_dataset = RadarDataset(mat_path=args.data_path, split='test', transform=None)
        else:  # ads 或其他
            try:
                from utils.readdata_25 import subDataset
            except ImportError:
                from readdata_25 import subDataset
            test_dataset = subDataset(datapath=args.data_path, split='test', 
                                    transform=None, allowed_classes=None)
        
        # 获取实际类别数（如果数据集有属性）
        if hasattr(test_dataset, 'num_classes'):
            actual_num_classes = test_dataset.num_classes
            if actual_num_classes != args.num_classes:
                print(f"[WARNING] 警告: 数据集实际类别数 ({actual_num_classes}) 与指定类别数 ({args.num_classes}) 不符")
                print(f"   使用数据集实际类别数: {actual_num_classes}")
                args.num_classes = actual_num_classes
        
        # 创建数据加载器
        num_workers = args.num_workers
        if platform.system() == 'Windows' and num_workers > 0:
            num_workers = 0
            print(f"[WARNING] Windows系统自动设置 num_workers=0 避免多进程问题")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"[SUCCESS] 测试集加载成功: {len(test_dataset):,} 样本")
        
        # 检查测试集是否为空
        if len(test_dataset) == 0:
            print(f"\n[ERROR] 错误: 测试集为空！")
            if args.dataset_type in ['rml2016', 'radioml'] and (args.radioml_snr_min is not None or args.radioml_snr_max is not None):
                print(f"   可能原因: SNR过滤范围 [{args.radioml_snr_min}, {args.radioml_snr_max}] 与数据集不匹配")
                print(f"   解决方案: 移除SNR过滤，使用所有数据")
                print(f"   命令: 添加 --radioml_snr_min None --radioml_snr_max None")
                print(f"   或者: 不指定SNR参数，使用所有SNR范围的数据")
            return
        
    except Exception as e:
        print(f"[ERROR] 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载边侧模型（real_resnet20）
    print(f"\n{'='*80}")
    print(f"📦 加载边侧模型（real_resnet20）")
    print(f"{'='*80}")
    
    try:
        edge_model_name = f'real_resnet20_{args.dataset_type}' if args.dataset_type != 'ads' else 'real_resnet20_ads'
        edge_model = create_model_by_type(edge_model_name, args.num_classes, args.dataset_type)
        edge_model = edge_model.to(device)
        
        # 加载权重
        if not os.path.exists(args.edge_model_path):
            print(f"[ERROR] 错误: 边侧模型文件不存在: {args.edge_model_path}")
            return
        
        edge_checkpoint = torch.load(args.edge_model_path, map_location=device, weights_only=False)

        ckpt_model_type = edge_checkpoint.get('model_type') if isinstance(edge_checkpoint, dict) else None
        internal_cfg = edge_checkpoint.get('internal_cfg') if isinstance(edge_checkpoint, dict) else None
        resolved_edge_model_type = ckpt_model_type or edge_model_name

        if args.dataset_type == 'ratr' and resolved_edge_model_type == 'real_resnet7_ratr_cp':
            edge_model = create_model_by_type(resolved_edge_model_type, args.num_classes, args.dataset_type, internal_cfg=internal_cfg)
            edge_model = edge_model.to(device)

            if isinstance(edge_checkpoint, dict) and 'model_state_dict' in edge_checkpoint:
                state_dict_to_load = edge_checkpoint['model_state_dict']
            elif isinstance(edge_checkpoint, dict) and 'state_dict' in edge_checkpoint:
                state_dict_to_load = edge_checkpoint['state_dict']
            else:
                state_dict_to_load = edge_checkpoint

            edge_model.load_state_dict(state_dict_to_load, strict=True)
            missing_keys, unexpected_keys = [], []
        else:
            # 处理不同的checkpoint格式
            state_dict_to_load = None
            if isinstance(edge_checkpoint, dict):
                if 'model_state_dict' in edge_checkpoint:
                    state_dict_to_load = edge_checkpoint['model_state_dict']
                    print(f"   检测到checkpoint格式（包含'model_state_dict'键）")
                elif 'state_dict' in edge_checkpoint:
                    state_dict_to_load = edge_checkpoint['state_dict']
                    print(f"   检测到checkpoint格式（包含'state_dict'键）")
                else:
                    # 可能是直接保存的state_dict
                    state_dict_to_load = edge_checkpoint
                    print(f"   检测到直接保存的state_dict格式")
            else:
                state_dict_to_load = edge_checkpoint
                print(f"   检测到直接保存的state_dict格式")
            
            # 加载权重并检查匹配情况
            missing_keys, unexpected_keys = edge_model.load_state_dict(state_dict_to_load, strict=False)
        
        if missing_keys:
            print(f"[WARNING] 警告: 以下键未加载（{len(missing_keys)}个）:")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"     - {key}")
            else:
                for key in missing_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... 还有 {len(missing_keys)-5} 个键未显示")
        
        if unexpected_keys:
            print(f"[WARNING] 警告: 以下键在checkpoint中但不在模型中（{len(unexpected_keys)}个）:")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"     - {key}")
            else:
                for key in unexpected_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... 还有 {len(unexpected_keys)-5} 个键未显示")
        
        if not missing_keys and not unexpected_keys:
            print(f"   [SUCCESS] 所有权重完美匹配")
        elif len(missing_keys) > len(state_dict_to_load) * 0.5:
            print(f"   [WARNING] 严重警告: 超过50%的权重未加载，模型可能无法正常工作！")
        
        print(f"[SUCCESS] 边侧模型加载完成")
        
    except Exception as e:
        print(f"[ERROR] 边侧模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载云侧模型（complex_resnet50）
    print(f"\n{'='*80}")
    print(f"📦 加载云侧模型（complex_resnet50）")
    print(f"{'='*80}")
    
    try:
        cloud_model_name = f'complex_resnet50_{args.dataset_type}' if args.dataset_type != 'ads' else 'complex_resnet50_ads'
        cloud_model = create_model_by_type(cloud_model_name, args.num_classes, args.dataset_type)
        cloud_model = cloud_model.to(device)
        
        # 加载权重
        if not os.path.exists(args.cloud_model_path):
            print(f"[ERROR] 错误: 云侧模型文件不存在: {args.cloud_model_path}")
            return
        
        cloud_checkpoint = torch.load(args.cloud_model_path, map_location=device, weights_only=False)
        
        # 处理不同的checkpoint格式
        state_dict_to_load = None
        if isinstance(cloud_checkpoint, dict):
            if 'model_state_dict' in cloud_checkpoint:
                state_dict_to_load = cloud_checkpoint['model_state_dict']
                print(f"   检测到checkpoint格式（包含'model_state_dict'键）")
            elif 'state_dict' in cloud_checkpoint:
                state_dict_to_load = cloud_checkpoint['state_dict']
                print(f"   检测到checkpoint格式（包含'state_dict'键）")
            else:
                # 可能是直接保存的state_dict
                state_dict_to_load = cloud_checkpoint
                print(f"   检测到直接保存的state_dict格式")
        else:
            state_dict_to_load = cloud_checkpoint
            print(f"   检测到直接保存的state_dict格式")
        
        # 加载权重并检查匹配情况
        missing_keys, unexpected_keys = cloud_model.load_state_dict(state_dict_to_load, strict=False)
        
        if missing_keys:
            print(f"[WARNING] 警告: 以下键未加载（{len(missing_keys)}个）:")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"     - {key}")
            else:
                for key in missing_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... 还有 {len(missing_keys)-5} 个键未显示")
        
        if unexpected_keys:
            print(f"[WARNING] 警告: 以下键在checkpoint中但不在模型中（{len(unexpected_keys)}个）:")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"     - {key}")
            else:
                for key in unexpected_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... 还有 {len(unexpected_keys)-5} 个键未显示")
        
        if not missing_keys and not unexpected_keys:
            print(f"   [SUCCESS] 所有权重完美匹配")
        elif len(missing_keys) > len(state_dict_to_load) * 0.5:
            print(f"   [WARNING] 严重警告: 超过50%的权重未加载，模型可能无法正常工作！")
        
        print(f"[SUCCESS] 云侧模型加载完成")
        
    except Exception as e:
        print(f"[ERROR] 云侧模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 评估边侧和云侧模型单独性能
    print(f"\n{'='*80}")
    print(f"[EVAL] 评估模型单独性能")
    print(f"{'='*80}")
    
    # 评估边侧模型（使用与训练代码相同的评估方式）
    print("\n评估边侧模型...")
    edge_model.eval()
    edge_correct = 0
    edge_total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="边侧模型评估"):
            # 处理batch格式（兼容不同数据集的返回格式）
            if len(batch) == 3:
                inputs, targets, _ = batch
            elif len(batch) == 2:
                inputs, targets = batch
            else:
                continue
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # RML2016数据集：输入已经是 (batch_size, 600) 复数，直接使用
            # 其他数据集可能需要转换，但RML2016不需要额外处理
            if args.dataset_type == 'rml2016':
                # RML2016: 输入已经是复数格式 (batch_size, 600)，直接传给模型
                # 模型的forward方法会自己处理
                outputs = edge_model(inputs)
            else:
                # 其他数据集：确保输入是复数格式
                if not torch.is_complex(inputs):
                    if inputs.dim() == 2:
                        inputs_real = inputs
                        inputs_imag = torch.zeros_like(inputs_real)
                        inputs = torch.view_as_complex(torch.stack([inputs_real, inputs_imag], dim=-1))
                outputs = edge_model(inputs)
            
            # 处理输出（模型可能返回复数，需要取模）
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            
            # 计算准确率（与训练代码一致）
            _, predicted = outputs.max(1)
            edge_total += targets.size(0)
            edge_correct += predicted.eq(targets).sum().item()
    
    edge_alone_acc = 100. * edge_correct / edge_total if edge_total > 0 else 0
    print(f"边侧模型准确率: {edge_alone_acc:.2f}% ({edge_correct}/{edge_total})")
    
    # 评估云侧模型（使用与训练代码相同的评估方式）
    print("\n评估云侧模型...")
    cloud_model.eval()
    cloud_correct = 0
    cloud_total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="云侧模型评估"):
            # 处理batch格式（兼容不同数据集的返回格式）
            if len(batch) == 3:
                inputs, targets, _ = batch
            elif len(batch) == 2:
                inputs, targets = batch
            else:
                continue
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # RML2016数据集：输入已经是 (batch_size, 600) 复数，直接使用
            # 其他数据集可能需要转换，但RML2016不需要额外处理
            if args.dataset_type == 'rml2016':
                # RML2016: 输入已经是复数格式 (batch_size, 600)，直接传给模型
                # 模型的forward方法会自己处理
                outputs = cloud_model(inputs)
            else:
                # 其他数据集：确保输入是复数格式
                if not torch.is_complex(inputs):
                    if inputs.dim() == 2:
                        inputs_real = inputs
                        inputs_imag = torch.zeros_like(inputs_real)
                        inputs = torch.view_as_complex(torch.stack([inputs_real, inputs_imag], dim=-1))
                outputs = cloud_model(inputs)
            
            # 处理输出（模型可能返回复数，需要取模）
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            
            # 计算准确率（与训练代码一致）
            _, predicted = outputs.max(1)
            cloud_total += targets.size(0)
            cloud_correct += predicted.eq(targets).sum().item()
    
    cloud_alone_acc = 100. * cloud_correct / cloud_total if cloud_total > 0 else 0
    print(f"云侧模型准确率: {cloud_alone_acc:.2f}% ({cloud_correct}/{cloud_total})")
    
    # 创建协同推理系统
    print(f"\n{'='*80}")
    print(f"[INIT] 初始化边-云协同推理系统")
    print(f"{'='*80}")
    
    collaborative_system = EdgeCloudCollaborativeInference(
        edge_model=edge_model,
        cloud_model=cloud_model,
        device=device,
        dataset_type=args.dataset_type,
        cloud_latency_ms=args.cloud_latency_ms,
        bandwidth_mbps=args.bandwidth_mbps,
        image_size_mb=args.image_size_mb
    )
    
    # 解析阈值列表
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    print(f"\n将测试以下置信度阈值: {thresholds}")
    
    # 评估不同阈值下的性能
    print(f"\n{'='*80}")
    print(f"[EXPERIMENT] 开始协同推理实验")
    print(f"{'='*80}")
    
    all_results = []
    
    for threshold in thresholds:
        results = collaborative_system.evaluate(
            test_loader,
            threshold=threshold,
            num_batches=args.num_batches
        )
        all_results.append(results)
        
        print(f"\n阈值 T = {threshold:.2f} 的结果:")
        print(f"  整体准确率: {results['overall_accuracy']:.2f}%")
        print(f"  边侧准确率: {results['edge_accuracy']:.2f}%")
        print(f"  云侧准确率: {results['cloud_accuracy']:.2f}%")
        print(f"  云端调用率: {results['cloud_ratio']:.2%}")
        print(f"  平均延迟: {results['avg_per_sample_latency_ms']:.4f} ms/样本")
        if results['speedup_ratio'] > 0:
            print(f"  速度提升: {results['speedup_ratio']:.2f}x")
    
    # 绘制结果
    print(f"\n{'='*80}")
    print(f"[PLOT] 绘制实验结果")
    print(f"{'='*80}")
    plot_collaborative_results(all_results, args.save_path)
    
    # 保存结果到JSON
    results_file = os.path.join(args.save_path, 'collaborative_inference_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'edge_alone_accuracy': edge_alone_acc,
            'cloud_alone_accuracy': cloud_alone_acc,
            'threshold_results': all_results,
            'experiment_config': {
                'dataset_type': args.dataset_type,
                'data_path': args.data_path,
                'edge_model_path': args.edge_model_path,
                'cloud_model_path': args.cloud_model_path,
                'num_classes': args.num_classes,
                'batch_size': args.batch_size,
                'confidence_thresholds': thresholds,
                'cloud_latency_ms': args.cloud_latency_ms,
                'bandwidth_mbps': args.bandwidth_mbps,
                'image_size_mb': args.image_size_mb
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"实验结果已保存到: {results_file}")
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"[SUMMARY] 实验总结")
    print(f"{'='*80}")
    print(f"边侧模型单独准确率: {edge_alone_acc:.2f}%")
    print(f"云侧模型单独准确率: {cloud_alone_acc:.2f}%")
    
    print(f"\n不同阈值下的协同推理结果:")
    print(f"{'阈值':<8} {'整体准确率':<12} {'云端调用率':<12} {'平均延迟(ms)':<15}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['threshold']:<8.2f} {r['overall_accuracy']:<12.2f} {r['cloud_ratio']:<12.2%} {r['avg_per_sample_latency_ms']:<15.4f}")
    
    # 找到最佳阈值
    if all_results:
        best_result = max(all_results, key=lambda x: x['overall_accuracy'] - 10 * x['cloud_ratio'])
        print(f"\n最佳阈值（平衡点）: T = {best_result['threshold']:.2f}")
        print(f"  整体准确率: {best_result['overall_accuracy']:.2f}%")
        print(f"  云端调用率: {best_result['cloud_ratio']:.2%}")
        
        best_accuracy = max(all_results, key=lambda x: x['overall_accuracy'])
        print(f"\n最高准确率阈值: T = {best_accuracy['threshold']:.2f}")
        print(f"  整体准确率: {best_accuracy['overall_accuracy']:.2f}%")
        print(f"  云端调用率: {best_accuracy['cloud_ratio']:.2%}")
    
    print(f"\n[SAVE] 结果保存路径: {args.save_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

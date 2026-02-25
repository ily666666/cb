"""
边侧任务回调
负责边侧模型推理和低置信度样本筛选
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from callback.registry import register_task
from utils_refactor import (
    load_json, save_json, load_pickle, save_pickle, save_numpy, 
    check_parameters, load_from_output, check_output_exists,
    simulate_transfer
)
from core.model_factory import create_model_by_type


def _prepare_iq_to_complex(X_data):
    """
    将 I/Q 双通道数据转为 complex 格式
    
    real_resnet20 和 complex_resnet50 都需要 complex 输入：
    - real 模型在 forward 内部自己拆 real/imag
    - complex 模型直接用 complex 运算

    处理逻辑：
    - (N, 2, L) 实数 I/Q → (N, L) complex64：I + j*Q
    - (N, L) 复数         → 直接使用
    - (N, L) 实数         → 转为 complex64（虚部为 0）
    """
    if isinstance(X_data, np.ndarray):
        if np.iscomplexobj(X_data):
            return X_data.astype(np.complex64)
        else:
            if X_data.ndim == 3 and X_data.shape[1] == 2:
                # (N, 2, L) I/Q 双通道 → (N, L) complex64
                return (X_data[:, 0, :] + 1j * X_data[:, 1, :]).astype(np.complex64)
            elif X_data.ndim == 2:
                return X_data.astype(np.complex64)
            return X_data.astype(np.complex64)  # fallback
    return X_data


def batch_inference_efficient(model, X_data, y_data, batch_size, device):
    """
    边侧高效批量推理
    
    数据自动转为 complex（模型内部自行拆分 real/imag）：
    - I/Q 双通道 (N, 2, L) → (N, L) complex → cfloat tensor
    
    Args:
        model: 推理模型
        X_data: 输入数据 (N, 2, L) float 或 (N, L) complex
        y_data: 标签数据 (N,)
        batch_size: 批次大小
        device: 计算设备
    
    Returns:
        tuple: (predictions, confidences, corrects)
    """
    model.eval()
    model.to(device)
    
    # 预处理：统一转为 complex（模型 forward 里自己拆 real/imag）
    X_data = _prepare_iq_to_complex(X_data)
    print(f"[边侧推理] 数据格式: shape={X_data.shape}, dtype={X_data.dtype}")
    
    all_predictions = []
    all_confidences = []
    all_corrects = []
    
    num_samples = len(X_data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"[推理] 开始批量推理: {num_samples} 样本, {num_batches} 批次")
    
    # Warmup: 用第一个 batch 预热 CUDA（不计入推理耗时）
    t_warmup_start = time.time()
    with torch.no_grad():
        warmup_X = X_data[:min(batch_size, num_samples)]
        if isinstance(warmup_X, np.ndarray):
            warmup_tensor = torch.from_numpy(warmup_X).cfloat().to(device)
        else:
            warmup_tensor = warmup_X.cfloat().to(device)
        _ = model(warmup_tensor)
    t_warmup = time.time() - t_warmup_start
    print(f"[推理] CUDA 热身完成 ({t_warmup:.2f}s)")
    
    # 正式推理
    t_infer_start = time.time()
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X_data[i:i+batch_size]
            batch_y = y_data[i:i+batch_size]
            
            # 转换为 cfloat Tensor
            if isinstance(batch_X, np.ndarray):
                batch_X_tensor = torch.from_numpy(batch_X).cfloat().to(device)
            else:
                batch_X_tensor = batch_X.cfloat().to(device)
            
            if isinstance(batch_y, np.ndarray):
                batch_y_tensor = torch.from_numpy(batch_y).long()
            else:
                batch_y_tensor = batch_y
            
            # 前向传播
            outputs = model(batch_X_tensor)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            # 计算正确性
            corrects = (predictions.cpu() == batch_y_tensor).numpy()
            
            all_predictions.append(predictions.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
            all_corrects.append(corrects)
            
            # 打印进度
            if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= num_samples:
                elapsed = time.time() - t_infer_start
                progress = min((i + batch_size) / num_samples * 100, 100)
                print(f"[推理] 进度: {progress:.1f}% ({i+batch_size}/{num_samples}), 耗时: {elapsed:.2f}s")
    
    # 合并结果
    predictions = np.concatenate(all_predictions)
    confidences = np.concatenate(all_confidences)
    corrects = np.concatenate(all_corrects)
    
    t_infer = time.time() - t_infer_start
    throughput = num_samples / t_infer if t_infer > 0 else 0
    
    print(f"[推理] 完成! 纯推理耗时: {t_infer:.2f}s, 吞吐量: {throughput:.1f} samples/s")
    
    return predictions, confidences, corrects, t_warmup, t_infer


@register_task
def edge_infer_callback(task_id):
    """
    边侧推理回调
    
    功能：
    1. 从output/device_load加载数据
    2. 加载边侧模型
    3. 批量推理，计算置信度
    4. 筛选低置信度样本（发送给云侧）
    5. 保存中间结果和最终报告
    
    配置文件: tasks/{task_id}/input/edge_infer.json
    输入文件: tasks/{task_id}/output/device_load/data_batch.pkl
    输出文件: 
        - output/edge_infer/predictions.npy
        - output/edge_infer/confidences.npy
        - output/edge_infer/low_conf_signals.pkl
        - output/edge_infer/low_conf_indices.npy
        - result/edge_infer/inference_report.txt
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 执行结果
    """
    print(f"\n{'='*60}")
    print(f"[边侧] 开始执行推理任务")
    print(f"{'='*60}")
    
    # 1. 读取配置文件
    config_path = f"./tasks/{task_id}/input/edge_infer.json"
    param_list = ['model_path', 'model_type', 'num_classes', 'input_data']
    
    result, config = check_parameters(config_path, param_list)
    
    if 'error' in result:
        print(f"[错误] {result['error']}")
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        missing_str = ', '.join(result['missing'])
        print(f"[错误] 缺少必需参数: {missing_str}")
        return {'status': 'error', 'message': f"缺少参数: {missing_str}"}
    
    model_path = config['model_path']
    model_type = config['model_type']
    num_classes = config['num_classes']
    confidence_threshold = config.get('confidence_threshold', None)
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 128)
    
    print(f"[配置] 模型路径: {model_path}")
    print(f"[配置] 模型类型: {model_type}")
    if confidence_threshold is not None:
        print(f"[配置] 置信度阈值: {confidence_threshold}")
    else:
        print(f"[配置] 置信度阈值: 未设置（纯边侧模式，不筛选低置信度样本）")
    print(f"[配置] 计算设备: {device}")
    
    # 2. 检查是否已有缓存结果（断点续传）
    output_pred_path = f"./tasks/{task_id}/output/edge_infer/predictions.npy"
    if check_output_exists(task_id, 'edge_infer', 'predictions.npy'):
        print(f"[跳过] 边侧推理结果已存在，跳过计算")
        # 加载缓存结果
        predictions = np.load(output_pred_path)
        confidences = np.load(f"./tasks/{task_id}/output/edge_infer/confidences.npy")
        
        result = {
            'status': 'cached',
            'total_samples': len(predictions),
        }
        # 仅在协同模式下加载低置信度数据
        low_conf_path = f"./tasks/{task_id}/output/edge_infer/low_conf_signals.pkl"
        if os.path.exists(low_conf_path):
            low_conf_signals = load_pickle(low_conf_path)
            result['low_conf_samples'] = len(low_conf_signals['X'])
        else:
            result['low_conf_samples'] = 0
        
        return result
    
    # 3. 加载输入数据
    print(f"[加载] 从 device_load 加载数据...")
    input_source = config['input_data']['source']
    input_file = config['input_data']['file_name']
    
    try:
        input_data = load_from_output(task_id, input_source, input_file)
        X_data = input_data['X']
        y_data = input_data['y']
        dataset_type = input_data['dataset_type']
        
        print(f"[加载] 成功加载 {len(X_data)} 个样本")
    except Exception as e:
        error_msg = f"加载输入数据失败: {str(e)}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}
    
    # 3.5 模拟网络传输（设备→边侧）
    bandwidth = config.get('simulate_bandwidth_mbps', None)
    transfer_info = {'transfer_time': 0}
    if bandwidth:
        input_file_path = f"./tasks/{task_id}/output/{input_source}/{input_file}"
        transfer_info = simulate_transfer(input_file_path, bandwidth)
    
    # 4. 加载模型
    print(f"[模型] 正在加载边侧模型...")
    t_model_load_start = time.time()
    try:
        model = create_model_by_type(model_type, num_classes, dataset_type)
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"[模型] 成功加载模型权重")
        else:
            error_msg = f"模型文件不存在: {model_path}"
            print(f"[错误] {error_msg}")
            return {'status': 'error', 'message': error_msg}
        
    except Exception as e:
        error_msg = f"加载模型失败: {str(e)}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}
    t_model_load = time.time() - t_model_load_start
    
    # 5. 批量推理（batch_inference_efficient 内部会做 warmup 并只计纯推理时间）
    predictions, confidences, corrects, t_warmup, t_infer = batch_inference_efficient(
        model, X_data, y_data, batch_size, device
    )
    
    # 6. 计算准确率
    accuracy = corrects.mean()
    print(f"[结果] 边侧整体准确率: {accuracy*100:.2f}%")
    
    # 7. 保存基本结果到output
    output_dir = f"./tasks/{task_id}/output/edge_infer"
    os.makedirs(output_dir, exist_ok=True)
    
    save_numpy(os.path.join(output_dir, 'predictions.npy'), predictions)
    save_numpy(os.path.join(output_dir, 'confidences.npy'), confidences)
    save_numpy(os.path.join(output_dir, 'corrects.npy'), corrects)
    
    # 8. 筛选低置信度样本（仅在设置了阈值时执行，用于协同推理模式）
    if confidence_threshold is not None:
        low_conf_mask = confidences < confidence_threshold
        low_conf_indices = np.where(low_conf_mask)[0]
        high_conf_mask = ~low_conf_mask
        
        num_low_conf = low_conf_mask.sum()
        num_high_conf = high_conf_mask.sum()
        low_conf_ratio = num_low_conf / len(predictions) * 100
        
        print(f"[筛选] 高置信度样本: {num_high_conf} ({100-low_conf_ratio:.1f}%)")
        print(f"[筛选] 低置信度样本: {num_low_conf} ({low_conf_ratio:.1f}%)")
        
        # 提取低置信度样本
        low_conf_X = X_data[low_conf_mask]
        low_conf_y = y_data[low_conf_mask]
        low_conf_preds = predictions[low_conf_mask]
        low_conf_confs = confidences[low_conf_mask]
        
        # 计算高/低置信度准确率
        high_conf_acc = corrects[high_conf_mask].mean() if num_high_conf > 0 else 0.0
        low_conf_acc = corrects[low_conf_mask].mean() if num_low_conf > 0 else 0.0
        
        print(f"[结果] 高置信度准确率: {high_conf_acc*100:.2f}%")
        print(f"[结果] 低置信度准确率: {low_conf_acc*100:.2f}%")
        
        save_numpy(os.path.join(output_dir, 'low_conf_indices.npy'), low_conf_indices)
        
        low_conf_data = {
            'X': low_conf_X,
            'y': low_conf_y,
            'predictions': low_conf_preds,
            'confidences': low_conf_confs,
            'indices': low_conf_indices,
            'dataset_type': dataset_type,
        }
        save_pickle(os.path.join(output_dir, 'low_conf_signals.pkl'), low_conf_data)
    else:
        num_low_conf = 0
        num_high_conf = len(predictions)
        high_conf_acc = accuracy
        low_conf_acc = 0.0
        low_conf_ratio = 0.0
        print(f"[信息] 纯边侧模式，跳过低置信度筛选")
    
    # 保存计时信息到文件（供 run_task.py 读取）
    timing_info = {
        'transfer_time': transfer_info['transfer_time'],
        'model_load_time': round(t_model_load, 4),
        'warmup_time': round(t_warmup, 4),
        'inference_time': round(t_infer, 4),
    }
    save_json(os.path.join(output_dir, 'timing.json'), timing_info)
    
    print(f"[保存] 中间结果已保存到: {output_dir}")
    print(f"[计时] 模型加载: {t_model_load:.2f}s, 热身: {t_warmup:.2f}s, 纯推理: {t_infer:.2f}s")
    
    # 9. 生成报告到result
    result_dir = f"./tasks/{task_id}/result/edge_infer"
    os.makedirs(result_dir, exist_ok=True)
    
    report_path = os.path.join(result_dir, 'inference_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("边侧推理报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"任务ID: {task_id}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("配置信息\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"数据集类型: {dataset_type}\n")
        if confidence_threshold is not None:
            f.write(f"置信度阈值: {confidence_threshold}\n")
        else:
            f.write(f"置信度阈值: 无（纯边侧模式）\n")
        f.write(f"批次大小: {batch_size}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("推理结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"总样本数: {len(predictions)}\n")
        f.write(f"整体准确率: {accuracy*100:.2f}%\n\n")
        
        if confidence_threshold is not None:
            f.write("=" * 60 + "\n")
            f.write("置信度分析\n")
            f.write("=" * 60 + "\n")
            f.write(f"高置信度样本数: {num_high_conf} ({100-low_conf_ratio:.1f}%)\n")
            f.write(f"高置信度准确率: {high_conf_acc*100:.2f}%\n\n")
            f.write(f"低置信度样本数: {num_low_conf} ({low_conf_ratio:.1f}%)\n")
            f.write(f"低置信度准确率: {low_conf_acc*100:.2f}%\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("置信度统计\n")
        f.write("=" * 60 + "\n")
        f.write(f"最小置信度: {confidences.min():.4f}\n")
        f.write(f"最大置信度: {confidences.max():.4f}\n")
        f.write(f"平均置信度: {confidences.mean():.4f}\n")
        f.write(f"中位数置信度: {np.median(confidences):.4f}\n")
    
    print(f"[报告] 推理报告已保存到: {report_path}")
    
    # 10. 返回统计信息
    result_info = {
        'status': 'success',
        'total_samples': len(predictions),
        'accuracy': float(accuracy),
        'high_conf_samples': int(num_high_conf),
        'high_conf_accuracy': float(high_conf_acc),
        'low_conf_samples': int(num_low_conf),
        'low_conf_accuracy': float(low_conf_acc),
        'low_conf_ratio': float(low_conf_ratio),
        'avg_confidence': float(confidences.mean()),
    }
    
    print(f"[完成] 边侧推理完成")
    
    return result_info

"""
云侧任务回调
支持两种模式：
1. 协同推理模式：处理边侧发送的低置信度样本（input_data.source = "edge_infer"）
2. 直接推理模式：处理端侧的全部数据（input_data.source = "device_load"）
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
    load_json, save_json, load_pickle, save_numpy, save_pickle,
    check_parameters, load_from_output, check_output_exists,
    simulate_transfer
)
from core.model_factory import create_model_by_type


def _prepare_for_complex_model(X_data):
    """
    为 complex 模型（如 complex_resnet50）准备数据
    
    complex 模型接收 (N, L) 或 (N, 1, L) complex64 输入。

    处理逻辑：
    - (N, 2, L) 实数 I/Q → 转为 (N, L) complex64：I + j*Q
    - (N, L) 复数   → 直接使用
    - (N, L) 实数   → 转为 complex64（虚部为 0）
    """
    if isinstance(X_data, np.ndarray):
        if np.iscomplexobj(X_data):
            # 已经是复数，直接使用
            return X_data.astype(np.complex64)
        else:
            if X_data.ndim == 3 and X_data.shape[1] == 2:
                # (N, 2, L) I/Q 双通道 → (N, L) complex64
                complex_data = (X_data[:, 0, :] + 1j * X_data[:, 1, :]).astype(np.complex64)
                return complex_data
            elif X_data.ndim == 2:
                # (N, L) 实数 → (N, L) complex（虚部为 0）
                return X_data.astype(np.complex64)
            return X_data.astype(np.complex64)  # fallback
    return X_data


def batch_inference_cloud(model, X_data, y_data, batch_size, device):
    """
    云侧（complex模型）批量推理
    
    数据会自动适配 complex 模型的输入格式：
    - I/Q 双通道 (N, 2, L) → 转为 (N, L) complex → cfloat tensor
    - 已有复数 (N, L) complex → cfloat tensor
    """
    model.eval()
    model.to(device)

    # 预处理：确保数据是 complex 模型需要的格式
    X_data = _prepare_for_complex_model(X_data)
    print(f"[云侧推理] 数据格式: shape={X_data.shape}, dtype={X_data.dtype}")

    all_predictions = []
    all_confidences = []
    all_corrects = []

    num_samples = len(X_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"[云侧推理] 处理 {num_samples} 个样本, {num_batches} 批次")

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
    print(f"[云侧推理] CUDA 热身完成 ({t_warmup:.2f}s)")

    # 正式推理
    t_infer_start = time.time()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X_data[i:i+batch_size]
            batch_y = y_data[i:i+batch_size]

            # 数据已在 _prepare_for_complex_model 中转为 complex64
            if isinstance(batch_X, np.ndarray):
                batch_X_tensor = torch.from_numpy(batch_X).cfloat().to(device)
            else:
                batch_X_tensor = batch_X.cfloat().to(device)

            if isinstance(batch_y, np.ndarray):
                batch_y_tensor = torch.from_numpy(batch_y).long()
            else:
                batch_y_tensor = batch_y

            outputs = model(batch_X_tensor)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            corrects = (predictions.cpu() == batch_y_tensor).numpy()

            all_predictions.append(predictions.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
            all_corrects.append(corrects)

            # 打印进度
            if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= num_samples:
                elapsed = time.time() - t_infer_start
                progress = min((i + batch_size) / num_samples * 100, 100)
                print(f"[云侧推理] 进度: {progress:.1f}% ({min(i+batch_size, num_samples)}/{num_samples}), 耗时: {elapsed:.2f}s")

    predictions = np.concatenate(all_predictions)
    confidences = np.concatenate(all_confidences)
    corrects = np.concatenate(all_corrects)

    t_infer = time.time() - t_infer_start
    throughput = num_samples / t_infer if t_infer > 0 else 0
    print(f"[云侧推理] 完成! 纯推理耗时: {t_infer:.2f}s, 吞吐量: {throughput:.1f} samples/s")

    return predictions, confidences, corrects, t_warmup, t_infer


def _load_cloud_model(model_type, num_classes, dataset_type, model_path, device):
    """加载云侧模型（公共逻辑）"""
    print(f"[模型] 正在加载云侧模型...")
    model = create_model_by_type(model_type, num_classes, dataset_type)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"[模型] 成功加载模型权重")
    return model


def _generate_report(report_path, title, sections):
    """通用报告生成"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for section_title, items in sections:
            f.write("=" * 60 + "\n")
            f.write(f"{section_title}\n")
            f.write("=" * 60 + "\n")
            for key, value in items:
                f.write(f"{key}: {value}\n")
            f.write("\n")


# ================================================================
# 端→云 直接推理
# ================================================================
@register_task
def cloud_direct_infer_callback(task_id):
    """
    端→云 直接推理回调

    功能：
    1. 直接从 device_load 加载全部数据
    2. 加载云侧教师模型
    3. 批量推理（全量数据）
    4. 生成完整报告

    配置文件 input/cloud_infer.json 中 input_data.source = "device_load"
    """
    print(f"\n{'='*60}")
    print(f"[云侧] 端→云 直接推理")
    print(f"{'='*60}")

    # 1. 读取配置
    config_path = f"./tasks/{task_id}/input/cloud_infer.json"
    param_list = ['model_path', 'model_type', 'num_classes', 'input_data']
    result, config = check_parameters(config_path, param_list)

    if 'error' in result:
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        return {'status': 'error', 'message': f"缺少参数: {', '.join(result['missing'])}"}

    model_path = config['model_path']
    model_type = config['model_type']
    num_classes = config['num_classes']
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 128)

    # 2. 缓存检查
    if check_output_exists(task_id, 'cloud_direct_infer', 'cloud_predictions.npy'):
        print(f"[跳过] 云侧推理结果已存在")
        cloud_predictions = np.load(f"./tasks/{task_id}/output/cloud_direct_infer/cloud_predictions.npy")
        return {'status': 'cached', 'cloud_samples': len(cloud_predictions)}

    # 3. 从 device_load 加载全部数据
    print(f"[加载] 从 device_load 加载全部数据...")
    input_source = config['input_data']['source']
    input_file = config['input_data'].get('file_name', 'data_batch.pkl')

    try:
        input_data = load_from_output(task_id, input_source, input_file)
        X_data = input_data['X']
        y_data = input_data['y']
        dataset_type = input_data['dataset_type']
        print(f"[加载] 成功加载 {len(X_data)} 个样本")
    except Exception as e:
        return {'status': 'error', 'message': f"加载数据失败: {str(e)}"}

    # 3.5 模拟网络传输（设备/边侧→云侧）
    bandwidth = config.get('simulate_bandwidth_mbps', None)
    transfer_info = {'transfer_time': 0}
    if bandwidth:
        input_file_path = f"./tasks/{task_id}/output/{input_source}/{input_file}"
        transfer_info = simulate_transfer(input_file_path, bandwidth)

    # 4. 加载模型
    t_model_load_start = time.time()
    try:
        model = _load_cloud_model(model_type, num_classes, dataset_type, model_path, device)
    except Exception as e:
        return {'status': 'error', 'message': f"加载模型失败: {str(e)}"}
    t_model_load = time.time() - t_model_load_start

    # 5. 批量推理
    predictions, confidences, corrects, t_warmup, t_infer = batch_inference_cloud(
        model, X_data, y_data, batch_size, device
    )

    accuracy = corrects.mean()
    print(f"[结果] 云侧直接推理准确率: {accuracy*100:.2f}%")
    print(f"[计时] 模型加载: {t_model_load:.2f}s, 热身: {t_warmup:.2f}s, 纯推理: {t_infer:.2f}s")

    # 6. 保存结果
    output_dir = f"./tasks/{task_id}/output/cloud_direct_infer"
    os.makedirs(output_dir, exist_ok=True)
    save_numpy(os.path.join(output_dir, 'cloud_predictions.npy'), predictions)
    save_numpy(os.path.join(output_dir, 'cloud_confidences.npy'), confidences)
    save_numpy(os.path.join(output_dir, 'cloud_corrects.npy'), corrects)

    # 保存计时信息到文件
    timing_info = {
        'transfer_time': transfer_info['transfer_time'],
        'model_load_time': round(t_model_load, 4),
        'warmup_time': round(t_warmup, 4),
        'inference_time': round(t_infer, 4),
    }
    save_json(os.path.join(output_dir, 'timing.json'), timing_info)

    # 7. 生成报告
    result_dir = f"./tasks/{task_id}/result/cloud_direct_infer"
    os.makedirs(result_dir, exist_ok=True)

    _generate_report(os.path.join(result_dir, 'cloud_report.txt'), '云侧直接推理报告', [
        ('配置信息', [
            ('模型类型', model_type),
            ('数据集类型', dataset_type),
            ('批次大小', batch_size),
        ]),
        ('推理结果', [
            ('总样本数', len(predictions)),
            ('整体准确率', f"{accuracy*100:.2f}%"),
            ('平均置信度', f"{confidences.mean():.4f}"),
            ('最低置信度', f"{confidences.min():.4f}"),
        ]),
    ])

    print(f"[完成] 云侧直接推理完成")

    return {
        'status': 'success',
        'total_samples': len(predictions),
        'accuracy': float(accuracy),
        'avg_confidence': float(confidences.mean()),
    }


# ================================================================
# 端→边→云 协同推理（云侧部分）
# ================================================================
@register_task
def cloud_infer_callback(task_id):
    """
    端→边→云 协同推理回调（云侧部分）

    功能：
    1. 从 output/edge_infer 加载低置信度样本
    2. 加载云侧教师模型
    3. 对低置信度样本推理
    4. 合并边侧 + 云侧结果，生成最终报告
    """
    print(f"\n{'='*60}")
    print(f"[云侧] 端→边→云 协同推理（云侧部分）")
    print(f"{'='*60}")

    # 1. 读取配置
    config_path = f"./tasks/{task_id}/input/cloud_infer.json"
    param_list = ['model_path', 'model_type', 'num_classes', 'input_data']
    result, config = check_parameters(config_path, param_list)

    if 'error' in result:
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        return {'status': 'error', 'message': f"缺少参数: {', '.join(result['missing'])}"}

    model_path = config['model_path']
    model_type = config['model_type']
    num_classes = config['num_classes']
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 128)

    # 2. 缓存检查
    if check_output_exists(task_id, 'cloud_infer', 'cloud_predictions.npy'):
        print(f"[跳过] 云侧推理结果已存在")
        cloud_predictions = np.load(f"./tasks/{task_id}/output/cloud_infer/cloud_predictions.npy")
        return {'status': 'cached', 'cloud_samples': len(cloud_predictions)}

    # 3. 加载低置信度样本
    print(f"[加载] 从 edge_infer 加载低置信度样本...")
    input_source = config['input_data']['source']
    input_file = config['input_data'].get('signals_file', 'low_conf_signals.pkl')

    try:
        low_conf_data = load_from_output(task_id, input_source, input_file)
        low_conf_X = low_conf_data['X']
        low_conf_y = low_conf_data['y']
        low_conf_indices = low_conf_data['indices']
        dataset_type = low_conf_data['dataset_type']
        edge_predictions = low_conf_data['predictions']

        print(f"[加载] 成功加载 {len(low_conf_X)} 个低置信度样本")

        if len(low_conf_X) == 0:
            print(f"[跳过] 没有低置信度样本")
            return {'status': 'skipped', 'cloud_samples': 0, 'message': '没有低置信度样本'}
    except Exception as e:
        return {'status': 'error', 'message': f"加载低置信度样本失败: {str(e)}"}

    # 3.5 模拟网络传输（边侧→云侧，只传低置信度样本）
    bandwidth = config.get('simulate_bandwidth_mbps', None)
    transfer_info = {'transfer_time': 0}
    if bandwidth:
        input_file_path = f"./tasks/{task_id}/output/{input_source}/{input_file}"
        transfer_info = simulate_transfer(input_file_path, bandwidth)

    # 4. 加载模型
    t_model_load_start = time.time()
    try:
        model = _load_cloud_model(model_type, num_classes, dataset_type, model_path, device)
    except Exception as e:
        return {'status': 'error', 'message': f"加载模型失败: {str(e)}"}
    t_model_load = time.time() - t_model_load_start

    # 5. 批量推理
    cloud_predictions, cloud_confidences, cloud_corrects, t_warmup, t_infer = batch_inference_cloud(
        model, low_conf_X, low_conf_y, batch_size, device
    )

    cloud_accuracy = cloud_corrects.mean()
    print(f"[结果] 云侧准确率: {cloud_accuracy*100:.2f}%")
    print(f"[计时] 模型加载: {t_model_load:.2f}s, 热身: {t_warmup:.2f}s, 纯推理: {t_infer:.2f}s")

    # 6. 边云对比分析
    agree_mask = (edge_predictions == cloud_predictions)
    num_agree = agree_mask.sum()
    agree_ratio = num_agree / len(edge_predictions) * 100

    edge_wrong = (edge_predictions != low_conf_y)
    corrected = edge_wrong & cloud_corrects.astype(bool)
    num_corrected = corrected.sum()

    print(f"[对比] 边云预测一致: {num_agree} ({agree_ratio:.1f}%)")
    print(f"[修正] 云侧修正样本数: {num_corrected}")

    # 7. 保存中间结果
    output_dir = f"./tasks/{task_id}/output/cloud_infer"
    os.makedirs(output_dir, exist_ok=True)
    save_numpy(os.path.join(output_dir, 'cloud_predictions.npy'), cloud_predictions)
    save_numpy(os.path.join(output_dir, 'cloud_confidences.npy'), cloud_confidences)
    save_numpy(os.path.join(output_dir, 'cloud_corrects.npy'), cloud_corrects)

    # 保存计时信息到文件
    timing_info = {
        'transfer_time': transfer_info['transfer_time'],
        'model_load_time': round(t_model_load, 4),
        'warmup_time': round(t_warmup, 4),
        'inference_time': round(t_infer, 4),
    }
    save_json(os.path.join(output_dir, 'timing.json'), timing_info)

    # 8. 生成云侧报告
    result_dir = f"./tasks/{task_id}/result/cloud_infer"
    os.makedirs(result_dir, exist_ok=True)

    _generate_report(os.path.join(result_dir, 'cloud_report.txt'), '云侧推理报告', [
        ('配置信息', [('模型类型', model_type), ('数据集类型', dataset_type)]),
        ('推理结果', [
            ('低置信度样本数', len(cloud_predictions)),
            ('云侧准确率', f"{cloud_accuracy*100:.2f}%"),
        ]),
        ('边云对比', [
            ('预测一致样本数', f"{num_agree} ({agree_ratio:.1f}%)"),
            ('云侧修正样本数', num_corrected),
        ]),
    ])

    # 9. 合并边侧+云侧结果生成最终报告
    try:
        edge_predictions_all = np.load(f"./tasks/{task_id}/output/edge_infer/predictions.npy")
        edge_corrects = np.load(f"./tasks/{task_id}/output/edge_infer/corrects.npy")
        edge_confidences = np.load(f"./tasks/{task_id}/output/edge_infer/confidences.npy")

        device_data = load_from_output(task_id, 'device_load', 'data_batch.pkl')
        y_true = device_data['y']

        # 创建最终预测：高置信使用边侧结果，低置信使用云侧结果
        final_predictions = edge_predictions_all.copy()
        final_predictions[low_conf_indices] = cloud_predictions

        edge_only_accuracy = edge_corrects.mean()
        final_accuracy = (final_predictions == y_true).mean()

        print(f"[最终] 边侧单独准确率: {edge_only_accuracy*100:.2f}%")
        print(f"[最终] 边云协同准确率: {final_accuracy*100:.2f}%")
        print(f"[最终] 准确率提升: {(final_accuracy-edge_only_accuracy)*100:+.2f}%")

        _generate_report(os.path.join(result_dir, 'final_report.txt'), '边云协同推理最终报告', [
            ('整体统计', [
                ('总样本数', len(y_true)),
                ('边侧处理样本', len(edge_predictions_all) - len(cloud_predictions)),
                ('云侧处理样本', len(cloud_predictions)),
                ('云侧调用率', f"{len(cloud_predictions)/len(y_true)*100:.1f}%"),
            ]),
            ('准确率对比', [
                ('边侧单独准确率', f"{edge_only_accuracy*100:.2f}%"),
                ('边云协同准确率', f"{final_accuracy*100:.2f}%"),
                ('准确率提升', f"{(final_accuracy-edge_only_accuracy)*100:+.2f}%"),
            ]),
        ])
        print(f"[报告] 最终报告已保存")

    except Exception as e:
        print(f"[警告] 生成最终报告失败: {str(e)}")

    print(f"[完成] 云侧推理完成")

    return {
        'status': 'success',
        'cloud_samples': len(cloud_predictions),
        'cloud_accuracy': float(cloud_accuracy),
        'agree_ratio': float(agree_ratio),
        'num_corrected': int(num_corrected),
    }

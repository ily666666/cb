"""
推理展示模拟器（简化版）

当 input JSON 中存在 display_config 时，跳过真实推理，按配置参数模拟输出。

display_config 只需 3 个核心参数：
{
    "simulate_realtime": false,
    "data_size_mb": 120.5,      // 数据量(MB)
    "time": 9.8,                // 传输+推理总耗时(秒)
    "accuracy": 97.28           // 准确率(%)，device_load 无需此项
}
"""
import os
import sys
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_refactor import TASKS_ROOT
from utils_refactor import save_timing


def _maybe_sleep(seconds, realtime):
    if realtime and seconds > 0:
        time.sleep(seconds)


def _simulate_progress(label, total_samples, inference_time, batch_size, realtime=False):
    num_batches = max((total_samples + batch_size - 1) // batch_size, 1)
    print_points = []
    for batch_idx in range(num_batches):
        batch_num = batch_idx + 1
        is_last = (batch_num * batch_size) >= total_samples
        if batch_num % 10 == 0 or is_last:
            processed = min(batch_num * batch_size, total_samples)
            target_elapsed = batch_num / num_batches * inference_time
            print_points.append((processed, target_elapsed))

    if realtime:
        t_start = time.time()
        for processed, target_elapsed in print_points:
            sleep_needed = target_elapsed - (time.time() - t_start)
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            elapsed = time.time() - t_start
            progress = processed / total_samples * 100
            print(f"[{label}] 进度: {progress:.1f}% ({processed}/{total_samples}), 耗时: {elapsed:.2f}s")
        remaining = inference_time - (time.time() - t_start)
        if remaining > 0:
            time.sleep(remaining)
    else:
        for processed, target_elapsed in print_points:
            progress = processed / total_samples * 100
            print(f"[{label}] 进度: {progress:.1f}% ({processed}/{total_samples}), 耗时: {target_elapsed:.2f}s")


def _auto_derive(dc, config, has_transfer=True):
    """从 display_config 的 3 个核心参数自动派生所有细节"""
    total_time = dc.get('time', 10.0)
    data_size_mb = dc.get('data_size_mb', 0)
    batch_size = config.get('batch_size', 128)

    total_samples = int(data_size_mb * 1000) if data_size_mb > 0 else 100000
    total_samples = max(total_samples, 1000)

    if has_transfer and data_size_mb > 0:
        bandwidth = config.get('simulate_bandwidth_mbps', 100)
        transfer_time = round(data_size_mb / bandwidth, 2)
        transfer_time = min(transfer_time, total_time * 0.4)
    else:
        transfer_time = 0

    inference_time = max(total_time - transfer_time, 0.01)
    model_load_time = round(random.uniform(0.3, 3.5), 2)
    warmup_time = round(random.uniform(0.15, 0.6), 2)
    throughput = total_samples / inference_time if inference_time > 0 else 0

    return {
        'total_samples': total_samples,
        'transfer_time': transfer_time,
        'inference_time': inference_time,
        'model_load_time': model_load_time,
        'warmup_time': warmup_time,
        'throughput': throughput,
        'batch_size': batch_size,
        'total_time': total_time,
    }


# ================================================================
# 端侧数据加载模拟
# ================================================================
def simulate_device_load(config, task_id, step_prefix=""):
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    data_size_mb = dc.get('data_size_mb', 0)
    total_time = dc.get('time', 5.0)
    batch_size = config.get('batch_size', 128)
    data_path = config.get('data_path', '')
    dataset_type = config.get('dataset_type', 'unknown')

    if data_size_mb <= 0 and data_path and os.path.exists(data_path):
        if os.path.isfile(data_path):
            data_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        else:
            total = sum(os.path.getsize(os.path.join(data_path, f))
                        for f in os.listdir(data_path)
                        if os.path.isfile(os.path.join(data_path, f)))
            data_size_mb = total / (1024 * 1024)

    total_samples = int(data_size_mb * 1000) if data_size_mb > 0 else 100000
    total_samples = max(total_samples, 1000)

    output_step = f"{step_prefix}device_load"

    print(f"\n{'='*60}")
    print(f"[端侧] 数据加载")
    print(f"{'='*60}")
    if data_size_mb > 0:
        print(f"[信息] 数据量: {data_size_mb:.1f} MB")
    print(f"[加载] 正在加载数据...")
    _maybe_sleep(total_time * 0.7, realtime)
    print(f"[加载] 成功加载 {total_samples} 个样本")
    _maybe_sleep(total_time * 0.3, realtime)

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data_batch.pkl')

    save_timing(output_dir, {'step_time': total_time})

    print(f"[计时] 耗时: {total_time:.2f}s")
    print(f"[完成] 端侧数据加载完成")

    num_batches = (total_samples + batch_size - 1) // batch_size
    return {
        'status': 'success',
        'num_samples': total_samples,
        'num_batches': num_batches,
        'dataset_type': dataset_type,
        'output_path': output_path,
    }


# ================================================================
# 边侧推理模拟
# ================================================================
def simulate_edge_infer(config, task_id, step_prefix=""):
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    accuracy = dc.get('accuracy', 97.28)
    confidence_threshold = config.get('confidence_threshold', None)
    model_type = config.get('model_type', 'unknown')

    d = _auto_derive(dc, config, has_transfer=True)
    output_step = f"{step_prefix}edge_infer"

    print(f"\n{'='*60}")
    print(f"[边侧] 推理任务")
    print(f"{'='*60}")
    data_size_mb = dc.get('data_size_mb', 0)
    if data_size_mb > 0:
        print(f"[配置] 数据量: {data_size_mb:.1f} MB | 设备: {config.get('device', 'cuda:0')}")
    else:
        print(f"[配置] 设备: {config.get('device', 'cuda:0')}")

    if d['transfer_time'] > 0:
        print(f"[传输] 端侧→边侧 数据传输中...")
        _maybe_sleep(d['transfer_time'], realtime)
        print(f"[传输] 传输完成")

    print(f"[推理] 开始推理 ({d['total_samples']} 样本)...")
    _maybe_sleep(d['warmup_time'], realtime)

    _simulate_progress("推理", d['total_samples'], d['inference_time'], d['batch_size'], realtime)

    print(f"[推理] 完成! 吞吐量: {d['throughput']:.1f} samples/s")
    print(f"[结果] 准确率: {accuracy:.2f}%")

    if confidence_threshold is not None:
        low_ratio = random.uniform(0.005, 0.02)
        low_conf = max(int(d['total_samples'] * low_ratio), 1)
        high_conf = d['total_samples'] - low_conf
        print(f"[筛选] 高置信度: {high_conf} | 低置信度: {low_conf}")
    else:
        low_conf = 0
        high_conf = d['total_samples']

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)

    save_timing(output_dir, {'step_time': d['total_time']})

    print(f"[计时] 传输+推理: {d['total_time']:.2f}s")
    print(f"[完成] 边侧推理完成")

    return {
        'status': 'success',
        'total_samples': d['total_samples'],
        'accuracy': accuracy / 100.0,
        'high_conf_samples': int(high_conf),
        'high_conf_accuracy': accuracy / 100.0,
        'low_conf_samples': int(low_conf),
        'low_conf_accuracy': random.uniform(0.1, 0.3),
        'low_conf_ratio': low_conf / d['total_samples'] * 100,
        'avg_confidence': 0.98,
    }


# ================================================================
# 云侧直接推理模拟
# ================================================================
def simulate_cloud_direct_infer(config, task_id, step_prefix=""):
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    accuracy = dc.get('accuracy', 97.68)
    model_type = config.get('model_type', 'unknown')

    d = _auto_derive(dc, config, has_transfer=True)
    output_step = f"{step_prefix}cloud_direct_infer"

    print(f"\n{'='*60}")
    print(f"[云侧] 直接推理")
    print(f"{'='*60}")
    data_size_mb = dc.get('data_size_mb', 0)
    if data_size_mb > 0:
        print(f"[配置] 数据量: {data_size_mb:.1f} MB | 设备: {config.get('device', 'cuda:0')}")
    else:
        print(f"[配置] 设备: {config.get('device', 'cuda:0')}")

    if d['transfer_time'] > 0:
        print(f"[传输] 端侧→云侧 数据传输中...")
        _maybe_sleep(d['transfer_time'], realtime)
        print(f"[传输] 传输完成")

    print(f"[推理] 开始推理 ({d['total_samples']} 样本)...")
    _maybe_sleep(d['warmup_time'], realtime)

    _simulate_progress("云侧推理", d['total_samples'], d['inference_time'], d['batch_size'], realtime)

    print(f"[推理] 完成! 吞吐量: {d['throughput']:.1f} samples/s")
    print(f"[结果] 准确率: {accuracy:.2f}%")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)

    save_timing(output_dir, {'step_time': d['total_time']})

    print(f"[计时] 传输+推理: {d['total_time']:.2f}s")
    print(f"[完成] 云侧直接推理完成")

    return {
        'status': 'success',
        'total_samples': d['total_samples'],
        'accuracy': accuracy / 100.0,
        'avg_confidence': 0.98,
    }


# ================================================================
# 云侧协同推理模拟
# ================================================================
def simulate_cloud_infer(config, task_id, step_prefix=""):
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    accuracy = dc.get('accuracy', 97.28)
    model_type = config.get('model_type', 'unknown')

    d = _auto_derive(dc, config, has_transfer=True)
    low_conf_samples = max(int(d['total_samples'] * random.uniform(0.005, 0.02)), 1)
    output_step = f"{step_prefix}cloud_infer"

    print(f"\n{'='*60}")
    print(f"[云侧] 协同推理（处理低置信度样本）")
    print(f"{'='*60}")

    if d['transfer_time'] > 0:
        print(f"[传输] 边侧→云侧 低置信度样本传输中...")
        _maybe_sleep(d['transfer_time'], realtime)
        print(f"[传输] 传输完成")

    print(f"[推理] 处理 {low_conf_samples} 个低置信度样本...")
    _maybe_sleep(d['warmup_time'], realtime)

    infer_time = d['inference_time']
    if infer_time > 0.5:
        _simulate_progress("云侧推理", low_conf_samples, infer_time, d['batch_size'], realtime)
    else:
        _maybe_sleep(infer_time, realtime)
        print(f"[云侧推理] 进度: 100.0% ({low_conf_samples}/{low_conf_samples}), 耗时: {infer_time:.2f}s")

    cloud_accuracy = round(random.uniform(60, 80), 2)
    print(f"[推理] 完成!")
    print(f"[结果] 准确率: {accuracy:.2f}%")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)

    save_timing(output_dir, {'step_time': d['total_time']})

    print(f"[计时] 传输+推理: {d['total_time']:.2f}s")
    print(f"[完成] 云侧协同推理完成")

    return {
        'status': 'success',
        'cloud_samples': low_conf_samples,
        'cloud_accuracy': cloud_accuracy / 100.0,
        'agree_ratio': random.uniform(20, 40),
        'num_corrected': int(low_conf_samples * random.uniform(0.5, 0.8)),
    }

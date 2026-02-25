"""
文件IO辅助工具
"""
import json
import pickle
import time
import numpy as np
import os


def load_json(file_path):
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        dict: JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path, data, indent=4):
    """
    保存JSON文件
    
    Args:
        file_path: JSON文件路径
        data: 要保存的数据
        indent: 缩进空格数
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_pickle(file_path):
    """
    加载Pickle文件
    
    Args:
        file_path: Pickle文件路径
    
    Returns:
        任意类型: Pickle数据
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(file_path, data):
    """
    保存Pickle文件
    
    Args:
        file_path: Pickle文件路径
        data: 要保存的数据
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_numpy(file_path):
    """
    加载NumPy数组
    
    Args:
        file_path: .npy文件路径
    
    Returns:
        np.ndarray: NumPy数组
    """
    return np.load(file_path)


def save_numpy(file_path, data):
    """
    保存NumPy数组
    
    Args:
        file_path: .npy文件路径
        data: NumPy数组
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, data)


def load_from_output(task_id, source_task, file_name):
    """
    从output目录加载数据
    
    Args:
        task_id: 任务ID
        source_task: 源任务名称（如 'device_load', 'edge_infer'）
        file_name: 文件名
    
    Returns:
        数据对象
    
    Example:
        data = load_from_output('TASK_001', 'device_load', 'data_batch.pkl')
    """
    file_path = f"./tasks/{task_id}/output/{source_task}/{file_name}"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输出文件不存在: {file_path}")
    
    # 根据文件扩展名选择加载方式
    if file_name.endswith('.pkl'):
        return load_pickle(file_path)
    elif file_name.endswith('.npy'):
        return load_numpy(file_path)
    elif file_name.endswith('.json'):
        return load_json(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_name}")


def check_output_exists(task_id, task_name, file_name):
    """
    检查output文件是否存在
    
    Args:
        task_id: 任务ID
        task_name: 任务名称
        file_name: 文件名
    
    Returns:
        bool: 文件是否存在
    """
    file_path = f"./tasks/{task_id}/output/{task_name}/{file_name}"
    return os.path.exists(file_path)


def simulate_transfer(file_path, bandwidth_mbps):
    """
    模拟网络传输延迟
    
    根据文件大小和带宽计算传输时间，用 sleep 模拟。
    
    Args:
        file_path: 被传输的文件路径
        bandwidth_mbps: 模拟带宽（MB/s）
    
    Returns:
        dict: {'file_size_mb': float, 'bandwidth_mbps': float, 'transfer_time': float}
    """
    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        return {'file_size_mb': 0, 'bandwidth_mbps': 0, 'transfer_time': 0}
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    transfer_time = file_size_mb / bandwidth_mbps
    
    print(f"[网络模拟] 文件: {os.path.basename(file_path)}")
    print(f"[网络模拟] 大小: {file_size_mb:.2f} MB, 带宽: {bandwidth_mbps} MB/s")
    print(f"[网络模拟] 模拟传输耗时: {transfer_time:.2f}s ...")
    
    time.sleep(transfer_time)
    
    print(f"[网络模拟] 传输完成")
    
    return {
        'file_size_mb': round(file_size_mb, 2),
        'bandwidth_mbps': bandwidth_mbps,
        'transfer_time': round(transfer_time, 4),
    }

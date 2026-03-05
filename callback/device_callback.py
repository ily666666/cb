"""
端侧任务回调
负责从本地文件加载数据（pkl格式）
"""
import os
import sys
import numpy as np
import pickle

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from callback.registry import register_task
from utils_refactor import load_json, save_pickle, check_parameters


# ================================================================
# 标签映射（与原项目 data_sources.py / run_edge_collaborative.py 一致）
# ================================================================
LABEL_MAPS = {
    'rml2016': {mod: idx for idx, mod in enumerate(['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK'])},
    'link11': {emitter: idx for idx, emitter in enumerate(
        ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3'])},
    'radar': None,  # radar 用整数标签
}


def _parse_pkl_data(data, dataset_type):
    """
    解析 pkl 文件，统一输出 (X, y) 格式
    
    【重要】保留原始数据格式，不做 I/Q→复数 转换。
    I/Q 双通道数据保持 (N, 2, length) 格式输出，
    由下游的推理/训练回调根据模型类型（real/complex）自行决定是否转换。

    支持的 pkl 内部格式：
    1. {(类别名, snr): signal_array}  — rml2016/link11 常见格式，每个 value 是 (N, 2, length)
    2. {类别整数: signal_array}       — radar 常见格式
    3. {'test': {'X':..., 'y':...}}   — 预划分数据
    4. {'X':..., 'y':...}             — 直接 X/y
    5. (X, y) 元组
    6. {'train': {...}, 'val': {...}, 'test': {...}} — 完整划分数据

    Returns:
        (X_data, y_data): numpy 数组，X_data 保持原始 dtype/shape
    """
    # 格式 5: 元组
    if isinstance(data, (tuple, list)) and len(data) == 2:
        X_data, y_data = np.array(data[0]), np.array(data[1])
        return X_data, y_data

    if not isinstance(data, dict):
        raise ValueError(f"不支持的数据类型: {type(data)}")

    keys = list(data.keys())

    # 格式 6: {'train': {...}, 'val': {...}, 'test': {...}} — 取 test 部分
    # 格式 3: {'test': {'X':..., 'y':...}}
    if 'test' in data and isinstance(data['test'], dict):
        X_data = np.array(data['test']['X'])
        y_data = np.array(data['test']['y'])
        return X_data, y_data

    # 格式 4: {'X':..., 'y':...}
    if 'X' in data and 'y' in data:
        X_data = np.array(data['X'])
        y_data = np.array(data['y'])
        return X_data, y_data

    # 格式 1/2: {key: signal_array} 字典格式
    # key 可能是 (类别名, snr) 元组, 或者整数类别标签
    label_map = LABEL_MAPS.get(dataset_type)

    all_signals = []
    all_labels = []

    for key, signal_array in data.items():
        signal_array = np.array(signal_array)

        # 确定标签
        if isinstance(key, tuple):
            # (类别名, snr) 格式
            class_name = key[0]
            if label_map and class_name in label_map:
                label = label_map[class_name]
            else:
                try:
                    label = int(class_name)
                except (ValueError, TypeError):
                    print(f"[警告] 无法识别标签: {key}, 跳过")
                    continue
        elif isinstance(key, (int, np.integer)):
            label = int(key)
        elif isinstance(key, str):
            if label_map and key in label_map:
                label = label_map[key]
            else:
                try:
                    label = int(key)
                except ValueError:
                    print(f"[警告] 无法识别标签: {key}, 跳过")
                    continue
        else:
            print(f"[警告] 无法识别key类型: {type(key)}, 跳过")
            continue

        # signal_array 形状通常是 (N, 2, length) 或 (N, length)
        # 直接添加全部样本，保持原始格式
        for i in range(len(signal_array)):
            all_signals.append(signal_array[i])
            all_labels.append(label)

    if not all_signals:
        raise ValueError(f"无法从数据中提取样本, keys样例: {keys[:5]}")

    X_data = np.array(all_signals)
    y_data = np.array(all_labels)

    print(f"[解析] 从字典格式解析出 {len(X_data)} 个样本, {len(np.unique(y_data))} 个类别")
    print(f"[解析] 数据shape: {X_data.shape}, dtype: {X_data.dtype}")
    return X_data, y_data


@register_task
def device_load_callback(task_id):
    """
    端侧数据加载回调
    
    功能：
    1. 从JSON配置读取数据路径和参数
    2. 加载pkl数据文件
    3. 保存到output目录供后续任务使用
    4. 返回统计信息
    
    配置文件: tasks/{task_id}/input/device_load.json
    输出文件: tasks/{task_id}/output/device_load/data_batch.pkl
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 执行结果 {'status': str, 'num_samples': int, 'num_batches': int}
    """
    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")
    
    # 1. 读取配置文件
    config_path = f"./tasks/{task_id}/input/device_load.json"
    param_list = ['data_path', 'dataset_type', 'batch_size']
    
    result, config = check_parameters(config_path, param_list)
    
    if 'error' in result:
        print(f"[错误] {result['error']}")
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        missing_str = ', '.join(result['missing'])
        print(f"[错误] 缺少必需参数: {missing_str}")
        return {'status': 'error', 'message': f"缺少参数: {missing_str}"}
    
    data_path = config['data_path']
    dataset_type = config['dataset_type']
    batch_size = config.get('batch_size', 128)
    num_batches = config.get('num_batches', None)  # None表示加载全部
    
    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")
    
    # 2. 检查数据路径是否存在
    if not os.path.exists(data_path):
        error_msg = f"数据路径不存在: {data_path}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}
    
    # 3. 加载数据（支持单文件或目录）
    max_files = config.get('max_files', None)  # 限制加载的文件数（用于快速测试）
    
    try:
        if os.path.isdir(data_path):
            # ---- 目录模式：加载目录下所有 .pkl 文件 ----
            pkl_files = sorted([
                os.path.join(data_path, f) 
                for f in os.listdir(data_path) if f.endswith('.pkl')
            ])
            if not pkl_files:
                error_msg = f"目录中没有 .pkl 文件: {data_path}"
                print(f"[错误] {error_msg}")
                return {'status': 'error', 'message': error_msg}
            
            if max_files is not None:
                pkl_files = pkl_files[:max_files]
            
            print(f"[加载] 发现 {len(pkl_files)} 个 PKL 文件（目录模式）")
            
            all_X = []
            all_y = []
            for i, fpath in enumerate(pkl_files):
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                X_part, y_part = _parse_pkl_data(data, dataset_type)
                all_X.append(X_part)
                all_y.append(y_part)
                if (i + 1) % 10 == 0 or i == len(pkl_files) - 1:
                    print(f"[加载] 已加载 {i+1}/{len(pkl_files)} 个文件")
            
            X_data = np.concatenate(all_X, axis=0)
            y_data = np.concatenate(all_y, axis=0)
        else:
            # ---- 单文件模式 ----
            print(f"[加载] 正在加载数据文件...")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            X_data, y_data = _parse_pkl_data(data, dataset_type)
        
        # 限制样本数量（用于快速测试）
        if num_batches is not None:
            max_samples = num_batches * batch_size
            X_data = X_data[:max_samples]
            y_data = y_data[:max_samples]
        
        total_samples = len(X_data)
        print(f"[加载] 成功加载 {total_samples} 个样本")
        print(f"[加载] 数据形状: X={X_data.shape}, dtype={X_data.dtype}")
        
    except Exception as e:
        error_msg = f"加载数据失败: {str(e)}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}
    
    # 4. 保存到output目录
    output_dir = f"./tasks/{task_id}/output/device_load"
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        'X': X_data,
        'y': y_data,
        'dataset_type': dataset_type,
        'batch_size': batch_size,
    }
    
    output_path = os.path.join(output_dir, 'data_batch.pkl')
    save_pickle(output_path, output_data)
    
    print(f"[保存] 数据已保存到: {output_path}")
    
    # 5. 返回统计信息
    result_info = {
        'status': 'success',
        'num_samples': total_samples,
        'num_batches': (total_samples + batch_size - 1) // batch_size,
        'dataset_type': dataset_type,
        'output_path': output_path,
    }
    
    print(f"[完成] 端侧数据加载完成")
    print(f"[统计] 样本数: {result_info['num_samples']}")
    print(f"[统计] 批次数: {result_info['num_batches']}")
    
    return result_info

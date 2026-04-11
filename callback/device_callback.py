"""
端侧任务回调
负责从本地文件加载数据（pkl格式）
"""
import os
import sys
import time
import numpy as np
import pickle
import os.path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from callback.registry import register_task
from config_refactor import get_dataset_from_task_id, TASKS_ROOT
from utils_refactor import load_json, save_pickle, check_parameters, save_timing


# ================================================================
# 标签映射（与原项目 data_sources.py / run_edge_collaborative.py 一致）
# ================================================================
LABEL_MAPS = {
    'rml2016': {mod: idx for idx, mod in enumerate(['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK'])},
    'link11': {emitter: idx for idx, emitter in enumerate(
        ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3'])},
    'radar': None,  # radar 用整数标签
}


def _ensure_numpy_compat_for_old_pickles():
    """Fix common pickle compatibility issues where pickles reference numpy._core.*."""
    try:
        import numpy.core as _np_core

        sys.modules.setdefault("numpy._core", _np_core)
        sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
        sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    except Exception:
        pass


def _load_mat_data(file_path, dataset_type):
    try:
        import h5py
        try:
            with h5py.File(file_path, 'r') as mat:
                mat_data = {key: np.array(mat[key]) for key in mat.keys() if not key.startswith('__')}
        except OSError:
            mat_data = None
    except ImportError:
        mat_data = None

    if mat_data is None:
        try:
            import scipy.io as scio
        except ImportError as e:
            raise ImportError("读取 .mat 需要安装 h5py 或 scipy") from e
        mat_data = scio.loadmat(file_path)

    if dataset_type == 'radar':
        if 'X_batch' in mat_data and 'Y_batch' in mat_data:
            X = np.array(mat_data['X_batch'])
            y = np.array(mat_data['Y_batch']).flatten()
        elif 'X' in mat_data and 'Y' in mat_data:
            X = np.array(mat_data['X'])
            y = np.array(mat_data['Y']).flatten()
        else:
            keys = list(mat_data.keys())
            raise ValueError(f"Radar .mat 缺少字段 X/Y 或 X_batch/Y_batch, keys={keys}")

        if X.ndim != 3:
            raise ValueError(f"Radar .mat X 维度异常: shape={X.shape}")

        if X.shape[0] == 2:
            X = np.transpose(X, (2, 0, 1))
        elif X.shape[1] == 2:
            X = np.transpose(X, (0, 1, 2))
        else:
            raise ValueError(f"Radar .mat X 形状不符合预期(2,L,N)或(N,2,L): shape={X.shape}")

        y = y.astype(np.int64)
        if y.min() >= 1:
            y = y - 1

        return X, y

    raise ValueError(f"暂不支持数据集 {dataset_type} 的 .mat 加载")


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

    # 格式 4: {'X':..., 'y':...} 或 {'X':..., 'Y':...}（部分数据生成脚本用大写 Y）
    if 'X' in data and ('y' in data or 'Y' in data or 'labels' in data):
        X_data = np.array(data['X'])
        if 'y' in data:
            y_data = np.array(data['y'])
        elif 'Y' in data:
            y_data = np.array(data['Y'])
        else:
            y_data = np.array(data['labels'])
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


def _parse_ratr_pkl(data, file_path: str):
    """Parse RATR pkl file.
    
    Supports both 1024 and 2048 length signals.
    """
    if not isinstance(data, dict) or 'data' not in data:
        raise ValueError(f"RATR pkl 结构异常，期望 dict 且包含 'data' 字段: {file_path}")

    stem = os.path.splitext(os.path.basename(file_path))[0].upper()
    if stem.startswith('E2D'):
        label = 0
    elif stem.startswith('P3C'):
        label = 1
    elif stem.startswith('P8A'):
        label = 2
    else:
        raise ValueError(f"RATR 文件名无法推断类别(E2D/P3C/P8A): {file_path}")

    arr = data['data']
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    
    # Support both 1024 and 2048 length signals
    if arr.ndim != 2:
        raise ValueError(f"RATR 'data' 维度异常: {arr.shape}, 期望2维数组, file={file_path}")
    
    signal_length = arr.shape[0]
    if signal_length not in [1024, 2048]:
        raise ValueError(f"RATR 'data' 信号长度异常: {signal_length}, 期望1024或2048, file={file_path}")
    
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    X = arr.T[:, None, :]
    y = np.full((X.shape[0],), label, dtype=np.int64)
    return X, y



# def _parse_ratr_pkl(data, file_path: str):
#     if not isinstance(data, dict) or 'data' not in data:
#         raise ValueError(f"RATR pkl 结构异常，期望 dict 且包含 'data' 字段: {file_path}")

#     stem = os.path.splitext(os.path.basename(file_path))[0].upper()
#     if stem.startswith('E2D'):
#         label = 0
#     elif stem.startswith('P3C'):
#         label = 1
#     elif stem.startswith('P8A'):
#         label = 2
#     else:
#         raise ValueError(f"RATR 文件名无法推断类别(E2D/P3C/P8A): {file_path}")

#     arr = data['data']
#     if not isinstance(arr, np.ndarray):
#         arr = np.asarray(arr)
#     if arr.ndim != 2 or arr.shape[0] != 1024:
#         raise ValueError(f"RATR 'data' 形状异常: {arr.shape}, file={file_path}")
#     if arr.dtype != np.float32:
#         arr = arr.astype(np.float32, copy=False)

#     X = arr.T[:, None, :]
#     y = np.full((X.shape[0],), label, dtype=np.int64)
#     return X, y


@register_task
def device_load_callback(task_id, **kwargs):
    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")

    config_path = f"{TASKS_ROOT}/{task_id}/input/device_load.json"
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
    num_batches = config.get('num_batches', None)

    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")

    if not os.path.exists(data_path):
        error_msg = f"数据路径不存在: {data_path}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}

    max_files = config.get('max_files', None)

    t_load_start = time.time()
    try:
        if dataset_type == 'ratr':
            _ensure_numpy_compat_for_old_pickles()
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    X_part, y_part = _parse_ratr_pkl(data, fpath)
                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] (ratr) 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext != '.pkl':
                    raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                X_data, y_data = _parse_ratr_pkl(data, data_path)

            if num_batches is not None:
                max_samples = int(num_batches) * int(batch_size)
                X_data = X_data[:max_samples]
                y_data = y_data[:max_samples]

            total_samples = len(X_data)
            print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
            print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

        else:
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    ext = os.path.splitext(fpath)[1].lower()
                    if ext == '.pkl':
                        with open(fpath, 'rb') as f:
                            data = pickle.load(f)
                        X_part, y_part = _parse_pkl_data(data, dataset_type)
                    elif ext == '.mat':
                        X_part, y_part = _load_mat_data(fpath, dataset_type)
                    else:
                        continue

                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext == '.pkl':
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    X_data, y_data = _parse_pkl_data(data, dataset_type)
                elif ext == '.mat':
                    X_data, y_data = _load_mat_data(data_path, dataset_type)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")

            if num_batches is not None:
                max_samples = int(num_batches) * int(batch_size)
                X_data = X_data[:max_samples]
                y_data = y_data[:max_samples]

            total_samples = len(X_data)
            print(f"[加载] 成功加载 {total_samples} 个样本")
            print(f"[加载] 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

    except Exception as e:
        error_msg = f"加载数据失败: {str(e)}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}

    t_data_load = time.time() - t_load_start
    print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/device_load"
    os.makedirs(output_dir, exist_ok=True)

    t_preprocess_start = time.time()
    task_id_lower = str(task_id).lower()
    if dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
        length = X_data.shape[-1]
        print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

        if 'cloud_only' in task_id_lower:
            if length == 500:
                X_data_save = np.concatenate([X_data, X_data], axis=-1)
                print(f"[Radar处理] 纯云推理任务：500→1000（重复拼接）")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 纯云推理任务：保持长度={length}")
        elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
            if length > 500:
                X_data_save = X_data[..., :500]
                print(f"[Radar处理] 边侧/协同任务：截断到500")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
        else:
            X_data_save = X_data
            print(f"[Radar处理] 其它任务：保持长度={length}")
    else:
        X_data_save = X_data

    t_preprocess = time.time() - t_preprocess_start
    print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

    output_data = {
        'X': X_data_save,
        'y': y_data,
        'dataset_type': dataset_type,
        'batch_size': batch_size,
    }

    output_path = os.path.join(output_dir, 'data_batch.pkl')
    t_save_start = time.time()
    save_pickle(output_path, output_data)
    t_save = time.time() - t_save_start
    print(f"[保存] 数据已保存到: {output_path}")
    print(f"[计时] 数据保存耗时: {t_save:.2f}s")

    save_timing(output_dir, {
        'data_load_time': t_data_load,
        'preprocess_time': t_preprocess,
        'data_save_time': t_save,
    })

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


@register_task
def link11_device_load_callback(task_id, **kwargs):
    """link11 端侧数据加载回调"""
    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")

    config_path = f"{TASKS_ROOT}/{task_id}/input/link11_device_load.json"
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
    num_batches = config.get('num_batches', None)

    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")

    if not os.path.exists(data_path):
        error_msg = f"数据路径不存在: {data_path}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}

    max_files = config.get('max_files', None)

    t_load_start = time.time()
    try:
        if dataset_type == 'ratr':
            _ensure_numpy_compat_for_old_pickles()
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    X_part, y_part = _parse_ratr_pkl(data, fpath)
                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] (ratr) 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext != '.pkl':
                    raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                X_data, y_data = _parse_ratr_pkl(data, data_path)

            if num_batches is not None:
                max_samples = int(num_batches) * int(batch_size)
                X_data = X_data[:max_samples]
                y_data = y_data[:max_samples]

            total_samples = len(X_data)
            print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
            print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

        else:
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    ext = os.path.splitext(fpath)[1].lower()
                    if ext == '.pkl':
                        with open(fpath, 'rb') as f:
                            data = pickle.load(f)
                        X_part, y_part = _parse_pkl_data(data, dataset_type)
                    elif ext == '.mat':
                        X_part, y_part = _load_mat_data(fpath, dataset_type)
                    else:
                        continue

                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext == '.pkl':
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    X_data, y_data = _parse_pkl_data(data, dataset_type)
                elif ext == '.mat':
                    X_data, y_data = _load_mat_data(data_path, dataset_type)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")

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

    t_data_load = time.time() - t_load_start
    print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/link11_device_load"
    os.makedirs(output_dir, exist_ok=True)

    t_preprocess_start = time.time()
    task_id_lower = str(task_id).lower()
    if dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
        length = X_data.shape[-1]
        print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

        if 'cloud_only' in task_id_lower:
            if length == 500:
                X_data_save = np.concatenate([X_data, X_data], axis=-1)
                print(f"[Radar处理] 纯云推理任务：500→1000（重复拼接）")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 纯云推理任务：保持长度={length}")
        elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
            if length > 500:
                X_data_save = X_data[..., :500]
                print(f"[Radar处理] 边侧/协同任务：截断到500")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
        else:
            X_data_save = X_data
            print(f"[Radar处理] 其它任务：保持长度={length}")
    else:
        X_data_save = X_data

    t_preprocess = time.time() - t_preprocess_start
    print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

    output_data = {
        'X': X_data_save,
        'y': y_data,
        'dataset_type': dataset_type,
        'batch_size': batch_size,
    }

    output_path = os.path.join(output_dir, 'data_batch.pkl')
    t_save_start = time.time()
    save_pickle(output_path, output_data)
    t_save = time.time() - t_save_start
    print(f"[保存] 数据已保存到: {output_path}")
    print(f"[计时] 数据保存耗时: {t_save:.2f}s")

    save_timing(output_dir, {
        'data_load_time': t_data_load,
        'preprocess_time': t_preprocess,
        'data_save_time': t_save,
    })

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


@register_task
def rml2016_device_load_callback(task_id, **kwargs):
    """rml2016 端侧数据加载回调"""
    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")

    config_path = f"{TASKS_ROOT}/{task_id}/input/rml2016_device_load.json"
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
    num_batches = config.get('num_batches', None)

    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")

    if not os.path.exists(data_path):
        error_msg = f"数据路径不存在: {data_path}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}

    max_files = config.get('max_files', None)

    t_load_start = time.time()
    try:
        if dataset_type == 'ratr':
            _ensure_numpy_compat_for_old_pickles()
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    X_part, y_part = _parse_ratr_pkl(data, fpath)
                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] (ratr) 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext != '.pkl':
                    raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                X_data, y_data = _parse_ratr_pkl(data, data_path)

            if num_batches is not None:
                max_samples = int(num_batches) * int(batch_size)
                X_data = X_data[:max_samples]
                y_data = y_data[:max_samples]

            total_samples = len(X_data)
            print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
            print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

        else:
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    ext = os.path.splitext(fpath)[1].lower()
                    if ext == '.pkl':
                        with open(fpath, 'rb') as f:
                            data = pickle.load(f)
                        X_part, y_part = _parse_pkl_data(data, dataset_type)
                    elif ext == '.mat':
                        X_part, y_part = _load_mat_data(fpath, dataset_type)
                    else:
                        continue

                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext == '.pkl':
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    X_data, y_data = _parse_pkl_data(data, dataset_type)
                elif ext == '.mat':
                    X_data, y_data = _load_mat_data(data_path, dataset_type)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")

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

    t_data_load = time.time() - t_load_start
    print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/rml2016_device_load"
    os.makedirs(output_dir, exist_ok=True)

    t_preprocess_start = time.time()
    task_id_lower = str(task_id).lower()
    if dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
        length = X_data.shape[-1]
        print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

        if 'cloud_only' in task_id_lower:
            if length == 500:
                X_data_save = np.concatenate([X_data, X_data], axis=-1)
                print(f"[Radar处理] 纯云推理任务：500→1000（重复拼接）")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 纯云推理任务：保持长度={length}")
        elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
            if length > 500:
                X_data_save = X_data[..., :500]
                print(f"[Radar处理] 边侧/协同任务：截断到500")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
        else:
            X_data_save = X_data
            print(f"[Radar处理] 其它任务：保持长度={length}")
    else:
        X_data_save = X_data

    t_preprocess = time.time() - t_preprocess_start
    print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

    output_data = {
        'X': X_data_save,
        'y': y_data,
        'dataset_type': dataset_type,
        'batch_size': batch_size,
    }

    output_path = os.path.join(output_dir, 'data_batch.pkl')
    t_save_start = time.time()
    save_pickle(output_path, output_data)
    t_save = time.time() - t_save_start
    print(f"[保存] 数据已保存到: {output_path}")
    print(f"[计时] 数据保存耗时: {t_save:.2f}s")

    save_timing(output_dir, {
        'data_load_time': t_data_load,
        'preprocess_time': t_preprocess,
        'data_save_time': t_save,
    })

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


@register_task
def radar_device_load_callback(task_id, **kwargs):
    """radar 端侧数据加载回调"""
    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")

    config_path = f"{TASKS_ROOT}/{task_id}/input/radar_device_load.json"
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
    num_batches = config.get('num_batches', None)

    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")

    if not os.path.exists(data_path):
        error_msg = f"数据路径不存在: {data_path}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}

    max_files = config.get('max_files', None)

    t_load_start = time.time()
    try:
        if dataset_type == 'ratr':
            _ensure_numpy_compat_for_old_pickles()
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    X_part, y_part = _parse_ratr_pkl(data, fpath)
                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] (ratr) 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext != '.pkl':
                    raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                X_data, y_data = _parse_ratr_pkl(data, data_path)

            if num_batches is not None:
                max_samples = int(num_batches) * int(batch_size)
                X_data = X_data[:max_samples]
                y_data = y_data[:max_samples]

            total_samples = len(X_data)
            print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
            print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

        else:
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    ext = os.path.splitext(fpath)[1].lower()
                    if ext == '.pkl':
                        with open(fpath, 'rb') as f:
                            data = pickle.load(f)
                        X_part, y_part = _parse_pkl_data(data, dataset_type)
                    elif ext == '.mat':
                        X_part, y_part = _load_mat_data(fpath, dataset_type)
                    else:
                        continue

                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext == '.pkl':
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    X_data, y_data = _parse_pkl_data(data, dataset_type)
                elif ext == '.mat':
                    X_data, y_data = _load_mat_data(data_path, dataset_type)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")

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

    t_data_load = time.time() - t_load_start
    print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/radar_device_load"
    os.makedirs(output_dir, exist_ok=True)

    t_preprocess_start = time.time()
    task_id_lower = str(task_id).lower()
    if dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
        length = X_data.shape[-1]
        print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

        if 'cloud_only' in task_id_lower:
            # 云侧任务：保持完整的1000长度数据
            X_data_save = X_data
            print(f"[Radar处理] 纯云推理任务：保持长度={length}")
        elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
            # 边侧/协同任务：从1000截断到500
            if length >= 1000:
                X_data_save = X_data[..., :500]
                print(f"[Radar处理] 边侧/协同任务：1000→500（截断前半部分）")
            elif length > 500:
                X_data_save = X_data[..., :500]
                print(f"[Radar处理] 边侧/协同任务：{length}→500（截断）")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
        else:
            X_data_save = X_data
            print(f"[Radar处理] 其它任务：保持长度={length}")
    else:
        X_data_save = X_data

    t_preprocess = time.time() - t_preprocess_start
    print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

    output_data = {
        'X': X_data_save,
        'y': y_data,
        'dataset_type': dataset_type,
        'batch_size': batch_size,
    }

    output_path = os.path.join(output_dir, 'data_batch.pkl')
    t_save_start = time.time()
    save_pickle(output_path, output_data)
    t_save = time.time() - t_save_start
    print(f"[保存] 数据已保存到: {output_path}")
    print(f"[计时] 数据保存耗时: {t_save:.2f}s")

    save_timing(output_dir, {
        'data_load_time': t_data_load,
        'preprocess_time': t_preprocess,
        'data_save_time': t_save,
    })

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


# @register_task
# def radar_device_load_callback(task_id, **kwargs):
#     """radar 端侧数据加载回调"""
#     print(f"\n{'='*60}")
#     print(f"[端侧] 开始执行数据加载任务")
#     print(f"{'='*60}")

#     config_path = f"{TASKS_ROOT}/{task_id}/input/radar_device_load.json"
#     param_list = ['data_path', 'dataset_type', 'batch_size']
#     result, config = check_parameters(config_path, param_list)

#     if 'error' in result:
#         print(f"[错误] {result['error']}")
#         return {'status': 'error', 'message': result['error']}
#     elif not result['valid']:
#         missing_str = ', '.join(result['missing'])
#         print(f"[错误] 缺少必需参数: {missing_str}")
#         return {'status': 'error', 'message': f"缺少参数: {missing_str}"}

#     data_path = config['data_path']
#     dataset_type = config['dataset_type']
#     batch_size = config.get('batch_size', 128)
#     num_batches = config.get('num_batches', None)

#     print(f"[配置] 数据路径: {data_path}")
#     print(f"[配置] 数据集类型: {dataset_type}")
#     print(f"[配置] 批次大小: {batch_size}")

#     if not os.path.exists(data_path):
#         error_msg = f"数据路径不存在: {data_path}"
#         print(f"[错误] {error_msg}")
#         return {'status': 'error', 'message': error_msg}

#     max_files = config.get('max_files', None)

#     t_load_start = time.time()
#     try:
#         if dataset_type == 'ratr':
#             _ensure_numpy_compat_for_old_pickles()
#             if os.path.isdir(data_path):
#                 files = sorted([
#                     os.path.join(data_path, f)
#                     for f in os.listdir(data_path)
#                     if f.lower().endswith('.pkl')
#                 ])
#                 if not files:
#                     error_msg = f"目录中没有 .pkl 文件: {data_path}"
#                     print(f"[错误] {error_msg}")
#                     return {'status': 'error', 'message': error_msg}

#                 if max_files is not None:
#                     files = files[:max_files]

#                 print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

#                 all_X = []
#                 all_y = []
#                 for i, fpath in enumerate(files):
#                     with open(fpath, 'rb') as f:
#                         data = pickle.load(f)
#                     X_part, y_part = _parse_ratr_pkl(data, fpath)
#                     all_X.append(np.array(X_part))
#                     all_y.append(np.array(y_part))
#                     if (i + 1) % 10 == 0 or i == len(files) - 1:
#                         print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

#                 X_data = np.concatenate(all_X, axis=0)
#                 y_data = np.concatenate(all_y, axis=0)
#             else:
#                 print(f"[加载] (ratr) 正在加载数据文件...")
#                 ext = os.path.splitext(data_path)[1].lower()
#                 if ext != '.pkl':
#                     raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
#                 with open(data_path, 'rb') as f:
#                     data = pickle.load(f)
#                 X_data, y_data = _parse_ratr_pkl(data, data_path)

#             if num_batches is not None:
#                 max_samples = int(num_batches) * int(batch_size)
#                 X_data = X_data[:max_samples]
#                 y_data = y_data[:max_samples]

#             total_samples = len(X_data)
#             print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
#             print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

#         else:
#             if os.path.isdir(data_path):
#                 files = sorted([
#                     os.path.join(data_path, f)
#                     for f in os.listdir(data_path)
#                     if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
#                 ])
#                 if not files:
#                     error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
#                     print(f"[错误] {error_msg}")
#                     return {'status': 'error', 'message': error_msg}

#                 if max_files is not None:
#                     files = files[:max_files]

#                 print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

#                 all_X = []
#                 all_y = []
#                 for i, fpath in enumerate(files):
#                     ext = os.path.splitext(fpath)[1].lower()
#                     if ext == '.pkl':
#                         with open(fpath, 'rb') as f:
#                             data = pickle.load(f)
#                         X_part, y_part = _parse_pkl_data(data, dataset_type)
#                     elif ext == '.mat':
#                         X_part, y_part = _load_mat_data(fpath, dataset_type)
#                     else:
#                         continue

#                     all_X.append(np.array(X_part))
#                     all_y.append(np.array(y_part))
#                     if (i + 1) % 10 == 0 or i == len(files) - 1:
#                         print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

#                 X_data = np.concatenate(all_X, axis=0)
#                 y_data = np.concatenate(all_y, axis=0)
#             else:
#                 print(f"[加载] 正在加载数据文件...")
#                 ext = os.path.splitext(data_path)[1].lower()
#                 if ext == '.pkl':
#                     with open(data_path, 'rb') as f:
#                         data = pickle.load(f)
#                     X_data, y_data = _parse_pkl_data(data, dataset_type)
#                 elif ext == '.mat':
#                     X_data, y_data = _load_mat_data(data_path, dataset_type)
#                 else:
#                     raise ValueError(f"不支持的文件格式: {ext}")

#             if num_batches is not None:
#                 max_samples = num_batches * batch_size
#                 X_data = X_data[:max_samples]
#                 y_data = y_data[:max_samples]

#             total_samples = len(X_data)
#             print(f"[加载] 成功加载 {total_samples} 个样本")
#             print(f"[加载] 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

#     except Exception as e:
#         error_msg = f"加载数据失败: {str(e)}"
#         print(f"[错误] {error_msg}")
#         return {'status': 'error', 'message': error_msg}

#     t_data_load = time.time() - t_load_start
#     print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

#     output_dir = f"{TASKS_ROOT}/{task_id}/output/radar_device_load"
#     os.makedirs(output_dir, exist_ok=True)

#     t_preprocess_start = time.time()
#     task_id_lower = str(task_id).lower()
#     if dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
#         length = X_data.shape[-1]
#         print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

#         if 'cloud_only' in task_id_lower:
#             if length == 500:
#                 X_data_save = np.concatenate([X_data, X_data], axis=-1)
#                 print(f"[Radar处理] 纯云推理任务：500→1000（重复拼接）")
#             else:
#                 X_data_save = X_data
#                 print(f"[Radar处理] 纯云推理任务：保持长度={length}")
#         elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
#             if length > 500:
#                 X_data_save = X_data[..., :500]
#                 print(f"[Radar处理] 边侧/协同任务：截断到500")
#             else:
#                 X_data_save = X_data
#                 print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
#         else:
#             X_data_save = X_data
#             print(f"[Radar处理] 其它任务：保持长度={length}")
#     else:
#         X_data_save = X_data

#     t_preprocess = time.time() - t_preprocess_start
#     print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

#     output_data = {
#         'X': X_data_save,
#         'y': y_data,
#         'dataset_type': dataset_type,
#         'batch_size': batch_size,
#     }

#     output_path = os.path.join(output_dir, 'data_batch.pkl')
#     t_save_start = time.time()
#     save_pickle(output_path, output_data)
#     t_save = time.time() - t_save_start
#     print(f"[保存] 数据已保存到: {output_path}")
#     print(f"[计时] 数据保存耗时: {t_save:.2f}s")

#     save_timing(output_dir, {
#         'data_load_time': t_data_load,
#         'preprocess_time': t_preprocess,
#         'data_save_time': t_save,
#     })

#     result_info = {
#         'status': 'success',
#         'num_samples': total_samples,
#         'num_batches': (total_samples + batch_size - 1) // batch_size,
#         'dataset_type': dataset_type,
#         'output_path': output_path,
#     }

#     print(f"[完成] 端侧数据加载完成")
#     print(f"[统计] 样本数: {result_info['num_samples']}")
#     print(f"[统计] 批次数: {result_info['num_batches']}")
#     return result_info


@register_task
def ratr_device_load_callback(task_id, **kwargs):
    """ratr 端侧数据加载回调"""
    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")

    config_path = f"{TASKS_ROOT}/{task_id}/input/ratr_device_load.json"
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
    num_batches = config.get('num_batches', None)

    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")

    if not os.path.exists(data_path):
        error_msg = f"数据路径不存在: {data_path}"
        print(f"[错误] {error_msg}")
        return {'status': 'error', 'message': error_msg}

    max_files = config.get('max_files', None)

    t_load_start = time.time()
    try:
        if dataset_type == 'ratr':
            _ensure_numpy_compat_for_old_pickles()
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    X_part, y_part = _parse_ratr_pkl(data, fpath)
                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] (ratr) 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext != '.pkl':
                    raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                X_data, y_data = _parse_ratr_pkl(data, data_path)

            if num_batches is not None:
                max_samples = int(num_batches) * int(batch_size)
                X_data = X_data[:max_samples]
                y_data = y_data[:max_samples]

            total_samples = len(X_data)
            print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
            print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

        else:
            if os.path.isdir(data_path):
                files = sorted([
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
                ])
                if not files:
                    error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
                    print(f"[错误] {error_msg}")
                    return {'status': 'error', 'message': error_msg}

                if max_files is not None:
                    files = files[:max_files]

                print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

                all_X = []
                all_y = []
                for i, fpath in enumerate(files):
                    ext = os.path.splitext(fpath)[1].lower()
                    if ext == '.pkl':
                        with open(fpath, 'rb') as f:
                            data = pickle.load(f)
                        X_part, y_part = _parse_pkl_data(data, dataset_type)
                    elif ext == '.mat':
                        X_part, y_part = _load_mat_data(fpath, dataset_type)
                    else:
                        continue

                    all_X.append(np.array(X_part))
                    all_y.append(np.array(y_part))
                    if (i + 1) % 10 == 0 or i == len(files) - 1:
                        print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

                X_data = np.concatenate(all_X, axis=0)
                y_data = np.concatenate(all_y, axis=0)
            else:
                print(f"[加载] 正在加载数据文件...")
                ext = os.path.splitext(data_path)[1].lower()
                if ext == '.pkl':
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    X_data, y_data = _parse_pkl_data(data, dataset_type)
                elif ext == '.mat':
                    X_data, y_data = _load_mat_data(data_path, dataset_type)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")

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

    t_data_load = time.time() - t_load_start
    print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/ratr_device_load"
    os.makedirs(output_dir, exist_ok=True)

    t_preprocess_start = time.time()
    task_id_lower = str(task_id).lower()
    
    # RATR dataset: handle 1024/2048 length transformation based on task type
    if dataset_type == 'ratr' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
        length = X_data.shape[-1]
        print(f"[RATR数据] 检测到RATR数据集，样本长度={length}")
        
        if 'cloud_only' in task_id_lower:
            # Cloud task: expect 2048 data (already extended)
            if length == 2048:
                X_data_save = X_data
                print(f"[RATR处理] 纯云推理任务：保持长度=2048")
            elif length == 1024:
                # Fallback: extend if needed
                X_data_save = np.concatenate([X_data, X_data], axis=-1)
                print(f"[RATR处理] 纯云推理任务：1024→2048（重复拼接）")
            else:
                X_data_save = X_data
                print(f"[RATR处理] 纯云推理任务：保持长度={length}")
        elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
            # Edge task: use 1024 data
            if length == 2048:
                X_data_save = X_data[..., :1024]
                print(f"[RATR处理] 边侧/协同任务：2048→1024（截断）")
            else:
                X_data_save = X_data
                print(f"[RATR处理] 边侧/协同任务：保持长度={length}")
        else:
            X_data_save = X_data
            print(f"[RATR处理] 其它任务：保持长度={length}")
    # Radar dataset: handle 500/1000 length transformation based on task type
    elif dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
        length = X_data.shape[-1]
        print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

        if 'cloud_only' in task_id_lower:
            if length == 500:
                X_data_save = np.concatenate([X_data, X_data], axis=-1)
                print(f"[Radar处理] 纯云推理任务：500→1000（重复拼接）")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 纯云推理任务：保持长度={length}")
        elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
            if length > 500:
                X_data_save = X_data[..., :500]
                print(f"[Radar处理] 边侧/协同任务：截断到500")
            else:
                X_data_save = X_data
                print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
        else:
            X_data_save = X_data
            print(f"[Radar处理] 其它任务：保持长度={length}")
    else:
        X_data_save = X_data

    t_preprocess = time.time() - t_preprocess_start
    print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

    output_data = {
        'X': X_data_save,
        'y': y_data,
        'dataset_type': dataset_type,
        'batch_size': batch_size,
    }

    output_path = os.path.join(output_dir, 'data_batch.pkl')
    t_save_start = time.time()
    save_pickle(output_path, output_data)
    t_save = time.time() - t_save_start
    print(f"[保存] 数据已保存到: {output_path}")
    print(f"[计时] 数据保存耗时: {t_save:.2f}s")

    save_timing(output_dir, {
        'data_load_time': t_data_load,
        'preprocess_time': t_preprocess,
        'data_save_time': t_save,
    })

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



# @register_task
# def ratr_device_load_callback(task_id, **kwargs):
#     """ratr 端侧数据加载回调"""
#     print(f"\n{'='*60}")
#     print(f"[端侧] 开始执行数据加载任务")
#     print(f"{'='*60}")

#     config_path = f"{TASKS_ROOT}/{task_id}/input/ratr_device_load.json"
#     param_list = ['data_path', 'dataset_type', 'batch_size']
#     result, config = check_parameters(config_path, param_list)

#     if 'error' in result:
#         print(f"[错误] {result['error']}")
#         return {'status': 'error', 'message': result['error']}
#     elif not result['valid']:
#         missing_str = ', '.join(result['missing'])
#         print(f"[错误] 缺少必需参数: {missing_str}")
#         return {'status': 'error', 'message': f"缺少参数: {missing_str}"}

#     data_path = config['data_path']
#     dataset_type = config['dataset_type']
#     batch_size = config.get('batch_size', 128)
#     num_batches = config.get('num_batches', None)

#     print(f"[配置] 数据路径: {data_path}")
#     print(f"[配置] 数据集类型: {dataset_type}")
#     print(f"[配置] 批次大小: {batch_size}")

#     if not os.path.exists(data_path):
#         error_msg = f"数据路径不存在: {data_path}"
#         print(f"[错误] {error_msg}")
#         return {'status': 'error', 'message': error_msg}

#     max_files = config.get('max_files', None)

#     t_load_start = time.time()
#     try:
#         if dataset_type == 'ratr':
#             _ensure_numpy_compat_for_old_pickles()
#             if os.path.isdir(data_path):
#                 files = sorted([
#                     os.path.join(data_path, f)
#                     for f in os.listdir(data_path)
#                     if f.lower().endswith('.pkl')
#                 ])
#                 if not files:
#                     error_msg = f"目录中没有 .pkl 文件: {data_path}"
#                     print(f"[错误] {error_msg}")
#                     return {'status': 'error', 'message': error_msg}

#                 if max_files is not None:
#                     files = files[:max_files]

#                 print(f"[加载] (ratr) 发现 {len(files)} 个文件（目录模式）")

#                 all_X = []
#                 all_y = []
#                 for i, fpath in enumerate(files):
#                     with open(fpath, 'rb') as f:
#                         data = pickle.load(f)
#                     X_part, y_part = _parse_ratr_pkl(data, fpath)
#                     all_X.append(np.array(X_part))
#                     all_y.append(np.array(y_part))
#                     if (i + 1) % 10 == 0 or i == len(files) - 1:
#                         print(f"[加载] (ratr) 已加载 {i+1}/{len(files)} 个文件")

#                 X_data = np.concatenate(all_X, axis=0)
#                 y_data = np.concatenate(all_y, axis=0)
#             else:
#                 print(f"[加载] (ratr) 正在加载数据文件...")
#                 ext = os.path.splitext(data_path)[1].lower()
#                 if ext != '.pkl':
#                     raise ValueError(f"RATR 仅支持 .pkl 文件: {ext}")
#                 with open(data_path, 'rb') as f:
#                     data = pickle.load(f)
#                 X_data, y_data = _parse_ratr_pkl(data, data_path)

#             if num_batches is not None:
#                 max_samples = int(num_batches) * int(batch_size)
#                 X_data = X_data[:max_samples]
#                 y_data = y_data[:max_samples]

#             total_samples = len(X_data)
#             print(f"[加载] (ratr) 成功加载 {total_samples} 个样本")
#             print(f"[加载] (ratr) 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

#         else:
#             if os.path.isdir(data_path):
#                 files = sorted([
#                     os.path.join(data_path, f)
#                     for f in os.listdir(data_path)
#                     if f.lower().endswith('.pkl') or f.lower().endswith('.mat')
#                 ])
#                 if not files:
#                     error_msg = f"目录中没有 .pkl 或 .mat 文件: {data_path}"
#                     print(f"[错误] {error_msg}")
#                     return {'status': 'error', 'message': error_msg}

#                 if max_files is not None:
#                     files = files[:max_files]

#                 print(f"[加载] 发现 {len(files)} 个文件（目录模式）")

#                 all_X = []
#                 all_y = []
#                 for i, fpath in enumerate(files):
#                     ext = os.path.splitext(fpath)[1].lower()
#                     if ext == '.pkl':
#                         with open(fpath, 'rb') as f:
#                             data = pickle.load(f)
#                         X_part, y_part = _parse_pkl_data(data, dataset_type)
#                     elif ext == '.mat':
#                         X_part, y_part = _load_mat_data(fpath, dataset_type)
#                     else:
#                         continue

#                     all_X.append(np.array(X_part))
#                     all_y.append(np.array(y_part))
#                     if (i + 1) % 10 == 0 or i == len(files) - 1:
#                         print(f"[加载] 已加载 {i+1}/{len(files)} 个文件")

#                 X_data = np.concatenate(all_X, axis=0)
#                 y_data = np.concatenate(all_y, axis=0)
#             else:
#                 print(f"[加载] 正在加载数据文件...")
#                 ext = os.path.splitext(data_path)[1].lower()
#                 if ext == '.pkl':
#                     with open(data_path, 'rb') as f:
#                         data = pickle.load(f)
#                     X_data, y_data = _parse_pkl_data(data, dataset_type)
#                 elif ext == '.mat':
#                     X_data, y_data = _load_mat_data(data_path, dataset_type)
#                 else:
#                     raise ValueError(f"不支持的文件格式: {ext}")

#             if num_batches is not None:
#                 max_samples = num_batches * batch_size
#                 X_data = X_data[:max_samples]
#                 y_data = y_data[:max_samples]

#             total_samples = len(X_data)
#             print(f"[加载] 成功加载 {total_samples} 个样本")
#             print(f"[加载] 数据形状: X={X_data.shape}, dtype={X_data.dtype}")

#     except Exception as e:
#         error_msg = f"加载数据失败: {str(e)}"
#         print(f"[错误] {error_msg}")
#         return {'status': 'error', 'message': error_msg}

#     t_data_load = time.time() - t_load_start
#     print(f"[计时] 数据加载耗时: {t_data_load:.2f}s")

#     output_dir = f"{TASKS_ROOT}/{task_id}/output/ratr_device_load"
#     os.makedirs(output_dir, exist_ok=True)

#     t_preprocess_start = time.time()
#     task_id_lower = str(task_id).lower()
#     if dataset_type == 'radar' and isinstance(X_data, np.ndarray) and X_data.ndim >= 3:
#         length = X_data.shape[-1]
#         print(f"[Radar数据] 检测到radar数据集，样本长度={length}")

#         if 'cloud_only' in task_id_lower:
#             if length == 500:
#                 X_data_save = np.concatenate([X_data, X_data], axis=-1)
#                 print(f"[Radar处理] 纯云推理任务：500→1000（重复拼接）")
#             else:
#                 X_data_save = X_data
#                 print(f"[Radar处理] 纯云推理任务：保持长度={length}")
#         elif 'edge_only' in task_id_lower or 'collab' in task_id_lower:
#             if length > 500:
#                 X_data_save = X_data[..., :500]
#                 print(f"[Radar处理] 边侧/协同任务：截断到500")
#             else:
#                 X_data_save = X_data
#                 print(f"[Radar处理] 边侧/协同任务：保持长度={length}")
#         else:
#             X_data_save = X_data
#             print(f"[Radar处理] 其它任务：保持长度={length}")
#     else:
#         X_data_save = X_data

#     t_preprocess = time.time() - t_preprocess_start
#     print(f"[计时] 数据预处理耗时: {t_preprocess:.2f}s")

#     output_data = {
#         'X': X_data_save,
#         'y': y_data,
#         'dataset_type': dataset_type,
#         'batch_size': batch_size,
#     }

#     output_path = os.path.join(output_dir, 'data_batch.pkl')
#     t_save_start = time.time()
#     save_pickle(output_path, output_data)
#     t_save = time.time() - t_save_start
#     print(f"[保存] 数据已保存到: {output_path}")
#     print(f"[计时] 数据保存耗时: {t_save:.2f}s")

#     save_timing(output_dir, {
#         'data_load_time': t_data_load,
#         'preprocess_time': t_preprocess,
#         'data_save_time': t_save,
#     })

#     result_info = {
#         'status': 'success',
#         'num_samples': total_samples,
#         'num_batches': (total_samples + batch_size - 1) // batch_size,
#         'dataset_type': dataset_type,
#         'output_path': output_path,
#     }

#     print(f"[完成] 端侧数据加载完成")
#     print(f"[统计] 样本数: {result_info['num_samples']}")
#     print(f"[统计] 批次数: {result_info['num_batches']}")
#     return result_info

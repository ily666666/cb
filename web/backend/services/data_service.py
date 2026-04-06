"""
数据接入服务 - 管理数据集和数据文件
"""
import os
import json
import pickle
import sys
from typing import Optional, Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import DATASET_CONFIG, TASKS_ROOT, KNOWN_DATASETS

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0


def list_datasets() -> List[Dict]:
    result = []
    for ds_name, cfg in DATASET_CONFIG.items():
        ds_dir = os.path.join(DATASET_DIR, ds_name)
        data_files = []
        total_size = 0.0

        if os.path.isdir(ds_dir):
            for f in sorted(os.listdir(ds_dir)):
                fpath = os.path.join(ds_dir, f)
                if os.path.isfile(fpath) and (f.endswith('.pkl') or f.endswith('.mat')):
                    sz = _file_size_mb(fpath)
                    total_size += sz
                    data_files.append({"filename": f, "size_mb": round(sz, 2)})

        splits_dir = os.path.join(SPLITS_DIR, ds_name)
        split_files = []
        if os.path.isdir(splits_dir):
            for f in sorted(os.listdir(splits_dir)):
                fpath = os.path.join(splits_dir, f)
                if os.path.isfile(fpath) and f.endswith('.pkl'):
                    sz = _file_size_mb(fpath)
                    split_files.append({"filename": f, "size_mb": round(sz, 2)})

        result.append({
            "name": ds_name,
            "num_classes": cfg["num_classes"],
            "signal_length": cfg["signal_length"],
            "cloud_model": cfg["cloud_model"],
            "edge_model": cfg["edge_model"],
            "data_files": data_files,
            "data_file_count": len(data_files),
            "total_size_mb": round(total_size, 2),
            "split_files": split_files,
        })
    return result


def get_dataset_detail(dataset_name: str) -> Optional[Dict]:
    if dataset_name not in DATASET_CONFIG:
        return None

    cfg = DATASET_CONFIG[dataset_name]
    ds_dir = os.path.join(DATASET_DIR, dataset_name)
    data_files = []
    if os.path.isdir(ds_dir):
        for f in sorted(os.listdir(ds_dir)):
            fpath = os.path.join(ds_dir, f)
            if os.path.isfile(fpath):
                data_files.append({
                    "filename": f,
                    "size_mb": round(_file_size_mb(fpath), 2),
                    "path": os.path.relpath(fpath, PROJECT_ROOT),
                })

    splits_dir = os.path.join(SPLITS_DIR, dataset_name)
    split_files = []
    if os.path.isdir(splits_dir):
        for f in sorted(os.listdir(splits_dir)):
            fpath = os.path.join(splits_dir, f)
            if os.path.isfile(fpath):
                split_files.append({
                    "filename": f,
                    "size_mb": round(_file_size_mb(fpath), 2),
                    "path": os.path.relpath(fpath, PROJECT_ROOT),
                })

    return {
        "name": dataset_name,
        "config": cfg,
        "data_files": data_files,
        "split_files": split_files,
    }


def get_data_file_preview(dataset_name: str, filename: str) -> Optional[Dict]:
    """预览单个数据文件的元信息（不加载全部数据）"""
    fpath = os.path.join(DATASET_DIR, dataset_name, filename)
    if not os.path.isfile(fpath):
        fpath = os.path.join(SPLITS_DIR, dataset_name, filename)
    if not os.path.isfile(fpath):
        return None

    info = {
        "filename": filename,
        "size_mb": round(_file_size_mb(fpath), 2),
    }

    if filename.endswith('.pkl'):
        try:
            import numpy as np
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                info["type"] = "dict"
                info["keys"] = list(data.keys())[:20]
                for k, v in list(data.items())[:5]:
                    if hasattr(v, 'shape'):
                        info[f"key_{k}_shape"] = str(v.shape)
                    elif isinstance(v, tuple) and len(v) == 2:
                        shapes = []
                        for item in v:
                            shapes.append(str(item.shape) if hasattr(item, 'shape') else str(type(item)))
                        info[f"key_{k}"] = shapes
            elif hasattr(data, 'shape'):
                info["type"] = "array"
                info["shape"] = str(data.shape)
                info["dtype"] = str(data.dtype)
            else:
                info["type"] = str(type(data).__name__)
        except Exception as e:
            info["preview_error"] = str(e)

    return info


def get_waveform_data(dataset_name: str, filename: str, sample_idx: int = 0, max_samples: int = 5) -> Optional[Dict]:
    """从 PKL/MAT 文件中读取信号波形数据，返回前端可绘图的格式"""
    fpath = os.path.join(DATASET_DIR, dataset_name, filename)
    if not os.path.isfile(fpath):
        fpath = os.path.join(SPLITS_DIR, dataset_name, filename)
    if not os.path.isfile(fpath):
        return None

    try:
        import numpy as np

        if filename.endswith('.pkl'):
            with open(fpath, 'rb') as f:
                raw = pickle.load(f)
        else:
            return None

        X, y, labels_info = None, None, None

        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            X, y = np.array(raw[0]), np.array(raw[1])
        elif isinstance(raw, dict):
            if 'test' in raw and isinstance(raw['test'], (dict, tuple, list)):
                part = raw['test']
                if isinstance(part, dict):
                    X = np.array(part.get('X', part.get('x')))
                    y = np.array(part.get('y', part.get('Y', part.get('labels'))))
                elif isinstance(part, (tuple, list)) and len(part) == 2:
                    X, y = np.array(part[0]), np.array(part[1])
            elif 'X' in raw and ('y' in raw or 'Y' in raw or 'labels' in raw):
                X = np.array(raw['X'])
                y = np.array(raw.get('y', raw.get('Y', raw.get('labels'))))
            else:
                all_signals, all_labels = [], []
                label_names = {}
                label_counter = 0
                for key, val in raw.items():
                    val = np.array(val)
                    if val.ndim < 2:
                        continue
                    if isinstance(key, tuple):
                        class_name = key[0]
                    else:
                        class_name = str(key)
                    if class_name not in label_names:
                        label_names[class_name] = label_counter
                        label_counter += 1
                    lbl = label_names[class_name]
                    n = val.shape[0] if val.ndim >= 2 else 1
                    all_signals.append(val)
                    all_labels.extend([lbl] * n)
                if all_signals:
                    X = np.concatenate(all_signals, axis=0)
                    y = np.array(all_labels)
                    labels_info = {v: k for k, v in label_names.items()}

        if X is None:
            return {"error": "无法解析数据格式"}

        total = X.shape[0]
        start = min(sample_idx, max(0, total - 1))
        end = min(start + max_samples, total)
        samples = []

        for i in range(start, end):
            sig = X[i]
            label = int(y[i]) if y is not None and i < len(y) else None
            label_name = None
            if labels_info and label is not None:
                label_name = labels_info.get(label, str(label))

            sample = {"index": i, "label": label, "label_name": label_name}

            if np.iscomplexobj(sig):
                if sig.ndim == 1:
                    sample["I"] = sig.real.astype(float).tolist()
                    sample["Q"] = sig.imag.astype(float).tolist()
                else:
                    sample["I"] = sig.real.flatten().astype(float).tolist()
                    sample["Q"] = sig.imag.flatten().astype(float).tolist()
            elif sig.ndim == 2 and sig.shape[0] == 2:
                sample["I"] = sig[0].astype(float).tolist()
                sample["Q"] = sig[1].astype(float).tolist()
            elif sig.ndim == 2 and sig.shape[1] == 2:
                sample["I"] = sig[:, 0].astype(float).tolist()
                sample["Q"] = sig[:, 1].astype(float).tolist()
            elif sig.ndim == 1:
                sample["I"] = sig.astype(float).tolist()
                sample["Q"] = None
            else:
                sample["I"] = sig.flatten().astype(float).tolist()[:2048]
                sample["Q"] = None

            if sample["I"] and len(sample["I"]) > 2048:
                sample["I"] = sample["I"][:2048]
            if sample["Q"] and len(sample["Q"]) > 2048:
                sample["Q"] = sample["Q"][:2048]

            samples.append(sample)

        signal_length = len(samples[0]["I"]) if samples and samples[0].get("I") else 0
        has_iq = samples[0].get("Q") is not None if samples else False

        return {
            "dataset": dataset_name,
            "filename": filename,
            "total_samples": total,
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "signal_length": signal_length,
            "has_iq": has_iq,
            "sample_start": start,
            "samples": samples,
        }
    except Exception as e:
        return {"error": str(e)}


def get_task_input_configs(task_id: str) -> List[Dict]:
    input_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "input")
    if not os.path.isdir(input_dir):
        return []

    configs = []
    for f in sorted(os.listdir(input_dir)):
        if not f.endswith('.json'):
            continue
        fpath = os.path.join(input_dir, f)
        try:
            with open(fpath, 'r', encoding='utf-8') as fp:
                content = json.load(fp)
            deploy_mode = content.get("_部署模式", "")
            if deploy_mode and "单机" not in deploy_mode:
                continue
            configs.append({"filename": f, "content": content})
        except Exception:
            configs.append({"filename": f, "content": None, "error": "解析失败"})
    return configs


def save_task_config(task_id: str, filename: str, content: Dict) -> bool:
    input_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "input")
    os.makedirs(input_dir, exist_ok=True)
    fpath = os.path.join(input_dir, filename)
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

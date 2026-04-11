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

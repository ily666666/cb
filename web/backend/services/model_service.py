"""
模型算法管理服务 - 管理云侧/边侧模型文件
"""
import os
import sys
from typing import Optional, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import CLOUD_MODEL_DIR, EDGE_MODEL_DIR, DATASET_CONFIG, KNOWN_DATASETS, TASKS_ROOT


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0


def _scan_model_dir(base_dir: str, role: str) -> List[Dict]:
    abs_dir = os.path.join(PROJECT_ROOT, base_dir)
    models = []
    if not os.path.isdir(abs_dir):
        return models

    for root, dirs, files in os.walk(abs_dir):
        for f in files:
            if not f.endswith(('.pth', '.pt', '.ckpt')):
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, PROJECT_ROOT)
            parent = os.path.basename(os.path.dirname(fpath))
            dataset = parent if parent in KNOWN_DATASETS else "unknown"

            model_type = ""
            if dataset in DATASET_CONFIG:
                cfg = DATASET_CONFIG[dataset]
                model_type = cfg.get("cloud_model", "") if role == "cloud" else cfg.get("edge_model", "")

            models.append({
                "name": f,
                "path": rel_path,
                "size_mb": round(_file_size_mb(fpath), 2),
                "model_type": model_type,
                "dataset": dataset,
                "role": role,
            })
    return models


def _scan_task_models() -> List[Dict]:
    """扫描 tasks/*/output/ 下训练产出的模型"""
    tasks_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    models = []
    if not os.path.isdir(tasks_dir):
        return models

    for task_id in os.listdir(tasks_dir):
        output_dir = os.path.join(tasks_dir, task_id, "output")
        if not os.path.isdir(output_dir):
            continue

        dataset = "unknown"
        for ds in KNOWN_DATASETS:
            if ds in task_id.lower():
                dataset = ds
                break

        for step_dir in os.listdir(output_dir):
            step_path = os.path.join(output_dir, step_dir)
            if not os.path.isdir(step_path):
                continue
            for f in os.listdir(step_path):
                if not f.endswith(('.pth', '.pt', '.ckpt')):
                    continue
                fpath = os.path.join(step_path, f)
                rel_path = os.path.relpath(fpath, PROJECT_ROOT)

                role = "cloud" if "teacher" in f or "global" in f else "edge"
                model_type = ""
                if dataset in DATASET_CONFIG:
                    cfg = DATASET_CONFIG[dataset]
                    model_type = cfg.get("cloud_model", "") if role == "cloud" else cfg.get("edge_model", "")

                models.append({
                    "name": f,
                    "path": rel_path,
                    "size_mb": round(_file_size_mb(fpath), 2),
                    "model_type": model_type,
                    "dataset": dataset,
                    "role": role,
                    "source_task": task_id,
                    "source_step": step_dir,
                })
    return models


def list_all_models() -> Dict:
    cloud_models = _scan_model_dir(CLOUD_MODEL_DIR, "cloud")
    edge_models = _scan_model_dir(EDGE_MODEL_DIR, "edge")
    task_models = _scan_task_models()

    task_cloud = [m for m in task_models if m["role"] == "cloud"]
    task_edge = [m for m in task_models if m["role"] == "edge"]

    return {
        "cloud_models": cloud_models + task_cloud,
        "edge_models": edge_models + task_edge,
        "total_cloud": len(cloud_models) + len(task_cloud),
        "total_edge": len(edge_models) + len(task_edge),
    }


def get_model_detail(model_path: str) -> Optional[Dict]:
    abs_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.isfile(abs_path):
        return None

    info = {
        "path": model_path,
        "name": os.path.basename(model_path),
        "size_mb": round(_file_size_mb(abs_path), 2),
    }

    try:
        import torch
        state = torch.load(abs_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict):
            if 'state_dict' in state:
                sd = state['state_dict']
            elif 'model_state_dict' in state:
                sd = state['model_state_dict']
            else:
                sd = state

            if isinstance(sd, dict):
                total_params = 0
                layers = []
                for k, v in sd.items():
                    if hasattr(v, 'shape'):
                        params = 1
                        for dim in v.shape:
                            params *= dim
                        total_params += params
                        layers.append({"name": k, "shape": list(v.shape), "params": params})

                info["total_params"] = total_params
                info["total_params_m"] = round(total_params / 1_000_000, 2)
                info["num_layers"] = len(layers)
                info["layers_summary"] = layers[:20]

            meta_keys = [k for k in state.keys() if k not in ('state_dict', 'model_state_dict')]
            for mk in meta_keys[:10]:
                v = state[mk]
                if isinstance(v, (int, float, str, bool)):
                    info[f"meta_{mk}"] = v
    except Exception as e:
        info["load_error"] = str(e)

    return info


def delete_model(model_path: str) -> bool:
    abs_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.isfile(abs_path):
        return False
    try:
        os.remove(abs_path)
        return True
    except OSError:
        return False


def get_dataset_model_config() -> Dict:
    """返回每个数据集对应的模型配置，从实际 JSON 配置文件扫描真实使用的模型类型"""
    import json as _json

    tasks_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    actual = {}
    for ds in KNOWN_DATASETS:
        actual[ds] = {"cloud_models": set(), "edge_models": set()}

    if os.path.isdir(tasks_dir):
        for tid in os.listdir(tasks_dir):
            input_dir = os.path.join(tasks_dir, tid, "input")
            if not os.path.isdir(input_dir):
                continue
            ds = None
            for d in KNOWN_DATASETS:
                if d in tid.lower():
                    ds = d
                    break
            if not ds:
                continue
            for fn in os.listdir(input_dir):
                if not fn.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(input_dir, fn), 'r', encoding='utf-8') as fp:
                        cfg = _json.load(fp)
                    mt = cfg.get("model_type", "")
                    smt = cfg.get("student_model_type", "")
                    tmt = cfg.get("teacher_model_type", "")
                    emt = cfg.get("edge_model_type", "")
                    if mt:
                        if "cloud" in fn or "pretrain" in fn:
                            actual[ds]["cloud_models"].add(mt)
                        elif "edge" in fn:
                            actual[ds]["edge_models"].add(mt)
                    if tmt:
                        actual[ds]["cloud_models"].add(tmt)
                    if smt:
                        actual[ds]["edge_models"].add(smt)
                    if emt:
                        actual[ds]["edge_models"].add(emt)
                except Exception:
                    pass

    result = {}
    for ds, cfg in DATASET_CONFIG.items():
        cloud_list = sorted(actual[ds]["cloud_models"]) if actual[ds]["cloud_models"] else [cfg["cloud_model"]]
        edge_list = sorted(actual[ds]["edge_models"]) if actual[ds]["edge_models"] else [cfg["edge_model"]]
        result[ds] = {
            "num_classes": cfg["num_classes"],
            "signal_length": cfg["signal_length"],
            "cloud_model": cfg["cloud_model"],
            "edge_model": cfg["edge_model"],
            "cloud_models": cloud_list,
            "edge_models": edge_list,
        }
    return result

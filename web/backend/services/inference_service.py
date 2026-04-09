"""
推理计算服务 - 管理推理任务执行和结果查看
"""
import json
import os
import sys
from typing import Optional, Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import TASKS_ROOT, PIPELINE_MODES, KNOWN_DATASETS
from services.task_service import run_task_async, get_task_run_status, get_task_timing


INFERENCE_MODES = {
    "device_to_cloud": {"label": "端→云（直接推理）", "steps": ["device_load", "cloud_direct_infer"]},
    "device_to_edge": {"label": "端→边（仅边推理）", "steps": ["device_load", "edge_infer"]},
    "device_to_edge_to_cloud": {"label": "端→边→云（协同推理）", "steps": ["device_load", "edge_infer", "cloud_infer"]},
}

TRAIN_MODES = {
    "pretrain": {"label": "预训练教师模型", "steps": ["cloud_pretrain"]},
    "knowledge_distillation": {"label": "知识蒸馏", "steps": ["edge_kd"]},
    "edge_local_train": {"label": "边侧本地训练", "steps": ["edge_local_train"]},
    "federated_learning": {"label": "联邦学习", "steps": ["federated_train"]},
    "full_train": {"label": "完整训练", "steps": ["cloud_pretrain", "edge_kd", "federated_train"]},
}


def start_inference(task_id: str, mode: str) -> Dict:
    return run_task_async(task_id, mode=mode)


def get_inference_result(task_id: str) -> Optional[Dict]:
    result_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "result")
    output_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "output")

    if not os.path.isdir(result_root) and not os.path.isdir(output_root):
        return None

    result = {
        "task_id": task_id,
        "reports": {},
        "timing": get_task_timing(task_id),
        "output_steps": [],
    }

    if os.path.isdir(result_root):
        for root, dirs, files in os.walk(result_root):
            for f in files:
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, result_root)
                if f.endswith('.txt'):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as fp:
                            result["reports"][rel] = fp.read()
                    except Exception:
                        pass

    if os.path.isdir(output_root):
        for step_dir in sorted(os.listdir(output_root)):
            step_path = os.path.join(output_root, step_dir)
            if os.path.isdir(step_path):
                step_info = {"name": step_dir, "files": []}
                for f in os.listdir(step_path):
                    fpath = os.path.join(step_path, f)
                    sz = os.path.getsize(fpath) / (1024 * 1024) if os.path.isfile(fpath) else 0
                    step_info["files"].append({"filename": f, "size_mb": round(sz, 2)})
                result["output_steps"].append(step_info)

    run_status = get_task_run_status(task_id)
    if run_status:
        result["run_status"] = run_status

    return result


def get_inference_report(task_id: str, report_name: str) -> Optional[str]:
    result_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "result")
    for root, dirs, files in os.walk(result_root):
        for f in files:
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, result_root)
            if rel == report_name and f.endswith('.txt'):
                try:
                    with open(fpath, 'r', encoding='utf-8') as fp:
                        return fp.read()
                except Exception:
                    return None
    return None


def _is_train_task(task_id: str) -> bool:
    return "train" in task_id.lower()


def _load_reports(task_id: str) -> Dict[str, str]:
    result_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "result")
    reports = {}
    if os.path.isdir(result_root):
        for root, dirs, files in os.walk(result_root):
            for f in files:
                if f.endswith('.txt'):
                    fpath = os.path.join(root, f)
                    rel = os.path.relpath(fpath, result_root)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as fp:
                            reports[rel] = fp.read()
                    except Exception:
                        pass
    return reports


def _load_train_history(task_id: str) -> Dict[str, Any]:
    """读取训练任务的 train_history.pkl，返回各步骤的训练历史"""
    import pickle
    output_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "output")
    if not os.path.isdir(output_root):
        return {}

    step_label_map = {
        "cloud_pretrain": "云侧预训练",
        "edge_kd": "知识蒸馏",
        "federated_train": "联邦训练",
    }

    def _to_list(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        if isinstance(v, list) and v and hasattr(v[0], 'tolist'):
            return [x.tolist() if hasattr(x, 'tolist') else x for x in v]
        return v

    def _resolve_label(step_dir):
        label = step_label_map.get(step_dir, step_dir)
        for k2, v2 in step_label_map.items():
            if step_dir.endswith(k2) or step_dir.startswith(k2):
                label = v2
                break
        return label

    histories = {}
    for step_dir in sorted(os.listdir(output_root)):
        step_path = os.path.join(output_root, step_dir)

        # train_history.pkl — 预训练 / 联邦学习
        hist_path = os.path.join(step_path, "train_history.pkl")
        if os.path.isfile(hist_path):
            try:
                import numpy as np
                with open(hist_path, 'rb') as f:
                    raw = pickle.load(f)
                if isinstance(raw, dict):
                    hist_data = {k: _to_list(v) for k, v in raw.items()}
                    hist_data["_step"] = step_dir
                    hist_data["_label"] = _resolve_label(step_dir)

                    if "train_loss" in hist_data:
                        hist_data["_type"] = "pretrain"
                    elif "round" in hist_data:
                        hist_data["_type"] = "federated"
                    else:
                        hist_data["_type"] = "unknown"

                    histories[step_dir] = hist_data
            except Exception:
                pass

        # kd_history.pkl — 知识蒸馏过程
        kd_hist_path = os.path.join(step_path, "kd_history.pkl")
        if os.path.isfile(kd_hist_path):
            try:
                import numpy as np
                with open(kd_hist_path, 'rb') as f:
                    raw = pickle.load(f)
                if isinstance(raw, dict):
                    edges = {}
                    for edge_key, edge_hist in raw.items():
                        if isinstance(edge_hist, dict):
                            edges[edge_key] = {k: _to_list(v) for k, v in edge_hist.items()}
                    kd_key = step_dir + "_kd"
                    histories[kd_key] = {
                        "_type": "distillation",
                        "_step": step_dir,
                        "_label": _resolve_label(step_dir) + " — 蒸馏过程",
                        "edges": edges,
                    }
            except Exception:
                pass

    return histories


def get_visualization_data(task_id: str) -> Optional[Dict]:
    """获取用于可视化的数据，自动区分推理/训练任务"""
    if _is_train_task(task_id):
        return _get_train_visualization_data(task_id)
    return _get_inference_visualization_data(task_id)


def _get_train_visualization_data(task_id: str) -> Optional[Dict]:
    histories = _load_train_history(task_id)
    reports = _load_reports(task_id)

    if not histories and not reports:
        return None

    return {
        "task_type": "train",
        "task_id": task_id,
        "histories": histories,
        "reports": reports,
    }


def _get_inference_visualization_data(task_id: str) -> Optional[Dict]:
    timing = get_task_timing(task_id)
    if not timing:
        return None

    steps = timing.get("steps", {})

    step_names = list(steps.keys())
    step_labels = []
    for s in step_names:
        label_map = {
            "device_load": "端侧加载",
            "edge_infer": "边侧推理",
            "cloud_infer": "云侧推理",
            "cloud_direct_infer": "云侧直接推理",
        }
        step_labels.append(label_map.get(s, s))

    has_transfer = timing.get("total_transfer", 0) > 0

    bar_series = {
        "数据加载": [steps[s].get("data_load_time", 0) for s in step_names],
        "数据预处理": [steps[s].get("preprocess_time", 0) for s in step_names],
        "模型加载": [steps[s].get("model_load_time", 0) for s in step_names],
        "热身": [steps[s].get("warmup_time", 0) for s in step_names],
        "推理": [steps[s].get("inference_time", 0) for s in step_names],
    }
    if has_transfer:
        bar_series["传输"] = [steps[s].get("transfer_time", 0) for s in step_names]

    pie_data = [
        {"name": "数据加载", "value": round(timing["total_data_load"], 4)},
        {"name": "数据预处理", "value": round(timing["total_preprocess"], 4)},
        {"name": "推理计算", "value": round(timing["total_inference"], 4)},
        {"name": "模型加载+热身", "value": round(timing["total_overhead"], 4)},
    ]
    if has_transfer:
        pie_data.append({"name": "网络传输", "value": round(timing["total_transfer"], 4)})

    summary = {
        "total_data_load": round(timing["total_data_load"], 4),
        "total_preprocess": round(timing["total_preprocess"], 4),
        "total_inference": round(timing["total_inference"], 4),
        "total_overhead": round(timing["total_overhead"], 4),
        "total_transfer": round(timing["total_transfer"], 4),
        "has_transfer": has_transfer,
        "total": round(sum([
            timing["total_data_load"], timing["total_preprocess"],
            timing["total_inference"], timing["total_overhead"],
            timing["total_transfer"],
        ]), 4),
    }

    return {
        "task_type": "inference",
        "timing_bar": {"categories": step_labels, "series": bar_series},
        "timing_pie": {"data": pie_data},
        "summary": summary,
        "reports": _load_reports(task_id),
    }

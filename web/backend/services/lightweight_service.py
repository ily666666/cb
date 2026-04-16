"""
模型轻量化服务 - 剪枝 + 量化压缩（通用版）
"""
import os
import sys
import json
import subprocess
import time
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import TASKS_ROOT

LIGHTWEIGHT_DIR = os.path.join(PROJECT_ROOT, 'lightweight')
COMPRESS_SCRIPT = os.path.join(LIGHTWEIGHT_DIR, 'compress_universal.py')
SIMULATE_SCRIPT = os.path.join(LIGHTWEIGHT_DIR, 'compress_simulate.py')
RESULT_DIR = os.path.join(LIGHTWEIGHT_DIR, 'result')

COMPRESS_METHODS = [
    {
        "id": "bn_slimming_qat",
        "name": "BN 软剪枝 + INT8 线性量化",
        "prune_type": "BN Slimming（软掩码）",
        "prune_mode": "soft",
        "quant_type": "QAT（量化感知训练 · 8-bit 线性整数量化）",
        "description": "基于 BN 层权重幅值进行通道级软掩码剪枝，配合 8-bit 线性整数量化感知训练（QAT）。训练过程中模拟量化误差，使模型适应低精度部署。",
        "params": {
            "prune_ratio": {"label": "剪枝比例", "default": 0.15, "min": 0.05, "max": 0.5, "step": 0.05},
            "num_bits": {"label": "量化位宽", "default": 8, "options": [4, 8]},
            "num_epochs": {"label": "QAT 训练轮数", "default": 15, "min": 5, "max": 50},
            "batch_size": {"label": "训练批大小", "default": 64, "min": 8, "max": 512, "step": 8},
        },
    },
    {
        "id": "physical_prune_qat",
        "name": "物理通道剪枝 + INT8 线性量化",
        "prune_type": "torch_pruning（物理删除通道）",
        "prune_mode": "physical",
        "quant_type": "QAT（量化感知训练 · 8-bit 线性整数量化）",
        "description": "利用 torch_pruning 库进行图追踪式物理结构化剪枝，真实移除冗余通道重构网络。配合 8-bit 线性整数量化感知训练（QAT）保障量化后精度。端侧部署推荐方案。",
        "params": {
            "prune_ratio": {"label": "剪枝比例", "default": 0.15, "min": 0.05, "max": 0.5, "step": 0.05},
            "num_bits": {"label": "量化位宽", "default": 8, "options": [4, 8]},
            "num_epochs": {"label": "QAT 训练轮数", "default": 15, "min": 5, "max": 50},
            "batch_size": {"label": "训练批大小", "default": 64, "min": 8, "max": 512, "step": 8},
        },
    },
]


def get_methods() -> List[Dict]:
    return COMPRESS_METHODS


def get_available_models() -> List[Dict]:
    """扫描所有训练任务输出目录中的 .pth 模型文件，同时读取 checkpoint 元信息"""
    models = []
    tasks_root = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    if not os.path.isdir(tasks_root):
        return models

    for task_id in sorted(os.listdir(tasks_root)):
        task_dir = os.path.join(tasks_root, task_id)
        output_dir = os.path.join(task_dir, 'output')
        if not os.path.isdir(output_dir):
            continue
        for step_dir in sorted(os.listdir(output_dir)):
            step_path = os.path.join(output_dir, step_dir)
            if not os.path.isdir(step_path):
                continue
            for fname in sorted(os.listdir(step_path)):
                if not fname.endswith('.pth'):
                    continue
                fpath = os.path.join(step_path, fname)
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                role = 'teacher' if 'teacher' in fname else 'student' if 'student' in fname else 'other'

                meta = _read_model_meta(fpath)
                models.append({
                    'name': fname,
                    'task_id': task_id,
                    'step': step_dir,
                    'path': fpath,
                    'rel_path': os.path.relpath(fpath, PROJECT_ROOT),
                    'size_mb': size_mb,
                    'role': role,
                    'model_type': meta.get('model_type', ''),
                    'dataset_type': meta.get('dataset_type', ''),
                    'num_classes': meta.get('num_classes', 0),
                })
    return models


def _read_model_meta(path: str) -> Dict:
    """快速读取 checkpoint 元信息（不加载权重到 GPU）"""
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            return {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    except Exception:
        pass
    return {}


def get_available_datasets() -> List[Dict]:
    """扫描 dataset/splits/ 下所有可用的数据文件"""
    datasets = []
    splits_root = os.path.join(PROJECT_ROOT, 'dataset', 'splits')
    if not os.path.isdir(splits_root):
        return datasets
    for ds_type in sorted(os.listdir(splits_root)):
        ds_dir = os.path.join(splits_root, ds_type)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith('.pkl'):
                continue
            fpath = os.path.join(ds_dir, fname)
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            datasets.append({
                'dataset_type': ds_type,
                'name': fname,
                'path': fpath,
                'rel_path': f"dataset/splits/{ds_type}/{fname}",
                'size_mb': size_mb,
            })
    return datasets


def _find_data_path(task_id: str, dataset_type: str) -> Optional[str]:
    """根据 task_id 和 dataset_type 自动查找对应的数据文件"""
    candidates = [
        os.path.join(PROJECT_ROOT, f"dataset/splits/{dataset_type}/cloud_data.pkl"),
        os.path.join(PROJECT_ROOT, f"dataset/splits/{dataset_type}/edge_1_data.pkl"),
    ]
    task_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, 'input')
    if os.path.isdir(task_dir):
        for f in os.listdir(task_dir):
            if f.endswith('.json') and 'pretrain' in f:
                try:
                    with open(os.path.join(task_dir, f), 'r') as fh:
                        cfg = json.load(fh)
                    dp = cfg.get('data_path')
                    if dp:
                        full = os.path.join(PROJECT_ROOT, dp) if not os.path.isabs(dp) else dp
                        candidates.insert(0, full)
                except Exception:
                    pass
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


_running_task = {
    "active": False, "proc": None, "pid": None,
    "method": None, "log_file": None, "log_fh": None, "start_time": None,
}


def _is_proc_alive() -> bool:
    proc = _running_task.get("proc")
    if proc is None:
        return False
    return proc.poll() is None


def _cleanup_log_fh():
    fh = _running_task.get("log_fh")
    if fh:
        try:
            fh.close()
        except Exception:
            pass
        _running_task["log_fh"] = None


def get_status() -> Dict:
    """获取当前压缩任务状态"""
    if not _running_task["active"]:
        return {"running": False}

    log_tail = ""
    if _running_task["log_file"] and os.path.isfile(_running_task["log_file"]):
        try:
            with open(_running_task["log_file"], "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                log_tail = "".join(lines[-80:])
        except Exception:
            pass

    still_running = _is_proc_alive()

    if not still_running:
        _running_task["active"] = False
        _cleanup_log_fh()

    elapsed = time.time() - _running_task["start_time"] if _running_task["start_time"] else 0
    return {
        "running": still_running,
        "method": _running_task["method"],
        "elapsed_s": round(elapsed, 1),
        "log": log_tail,
    }


def stop_compress() -> Dict:
    """终止正在运行的压缩任务"""
    if not _running_task["active"]:
        return {"status": "ok", "message": "没有正在运行的任务"}

    proc = _running_task.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            _running_task["active"] = False
            _cleanup_log_fh()
            return {"status": "ok", "message": f"已终止进程 {proc.pid}"}
        except Exception as e:
            _running_task["active"] = False
            _cleanup_log_fh()
            return {"status": "error", "message": str(e)}
    _running_task["active"] = False
    _cleanup_log_fh()
    return {"status": "ok", "message": "进程已结束"}


def run_compress(method_id: str, model_path: str, params: Dict) -> Dict:
    """启动压缩任务（后台子进程）"""
    if _running_task["active"]:
        status = get_status()
        if status["running"]:
            return {"status": "error", "message": "已有压缩任务正在运行，请等待完成"}
        _running_task["active"] = False

    method = None
    for m in COMPRESS_METHODS:
        if m["id"] == method_id:
            method = m
            break
    if not method:
        return {"status": "error", "message": f"未知压缩方案: {method_id}"}

    if not model_path or not os.path.isfile(model_path):
        return {"status": "error", "message": f"模型文件不存在: {model_path}"}

    fast_mode = params.pop("fast_mode", False)

    if not fast_mode and not os.path.isfile(COMPRESS_SCRIPT):
        return {"status": "error", "message": "通用压缩脚本不存在"}

    meta = _read_model_meta(model_path)
    model_type = meta.get("model_type", params.get("model_type", ""))
    dataset_type = meta.get("dataset_type", params.get("dataset_type", ""))
    num_classes = meta.get("num_classes", params.get("num_classes", 0))

    if not model_type or not dataset_type or not num_classes:
        return {"status": "error", "message": "无法从模型文件中读取 model_type/dataset_type/num_classes，请确认模型格式"}

    task_id = ""
    tasks_root = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    if tasks_root in model_path:
        parts = model_path.replace(tasks_root + "/", "").split("/")
        if parts:
            task_id = parts[0]

    data_path = params.get("data_path") or _find_data_path(task_id, dataset_type)
    if not data_path:
        return {"status": "error", "message": f"找不到 {dataset_type} 的数据文件，请确认 dataset/splits/{dataset_type}/ 下有数据"}

    os.makedirs(RESULT_DIR, exist_ok=True)

    run_config = {
        "model_path": model_path,
        "model_type": model_type,
        "dataset_type": dataset_type,
        "data_path": data_path,
        "num_classes": num_classes,
        "prune_mode": method["prune_mode"],
        "prune_ratio": params.get("prune_ratio", 0.15),
        "num_bits": params.get("num_bits", 8),
        "num_epochs": params.get("num_epochs", 15),
        "batch_size": params.get("batch_size", 64),
        "device": "cuda:0" if _has_cuda() else "cpu",
        "save_dir": RESULT_DIR,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(LIGHTWEIGHT_DIR, f"_run_{ts}.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    log_file = os.path.join(LIGHTWEIGHT_DIR, f"_run_{ts}.log")

    python_exe = sys.executable
    script = SIMULATE_SCRIPT if fast_mode else COMPRESS_SCRIPT
    cmd = [python_exe, script, config_file]

    try:
        log_fh = open(log_file, "w", encoding="utf-8")
        popen_kwargs = dict(
            cwd=LIGHTWEIGHT_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        try:
            log_fh.close()
        except Exception:
            pass
        return {"status": "error", "message": f"启动压缩进程失败: {e}"}

    _running_task["active"] = True
    _running_task["proc"] = proc
    _running_task["pid"] = proc.pid
    _running_task["method"] = method["name"]
    _running_task["log_file"] = log_file
    _running_task["log_fh"] = log_fh
    _running_task["start_time"] = time.time()

    return {"status": "started", "pid": proc.pid, "method": method["name"]}


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_compression_history() -> List[Dict]:
    """扫描 lightweight/result 目录下的压缩结果"""
    results = []
    if not os.path.isdir(RESULT_DIR):
        return results

    for fname in sorted(os.listdir(RESULT_DIR)):
        fpath = os.path.join(RESULT_DIR, fname)
        if fname.endswith('.json') and fname.startswith('result_'):
            try:
                with open(fpath, 'r') as f:
                    results.append(json.load(f))
            except Exception:
                pass
        elif fname.endswith('.pth') and 'compressed' in fname:
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            results.append({'name': fname, 'path': fpath, 'size_mb': size_mb, 'type': 'model'})
        elif fname.endswith('.png') and 'cm_' in fname:
            results.append({'name': fname, 'path': fpath, 'type': 'confusion_matrix'})
    return results

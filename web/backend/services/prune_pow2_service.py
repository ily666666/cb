"""
剪枝 + 2的幂次量化（INQ）服务
物理通道剪枝 + Incremental Network Quantization
"""
import os
import sys
import json
import subprocess
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import TASKS_ROOT

LIGHTWEIGHT_DIR = os.path.join(PROJECT_ROOT, 'lightweight')
INQ_SCRIPT = os.path.join(LIGHTWEIGHT_DIR, 'compress_inq_universal.py')
SIMULATE_SCRIPT = os.path.join(LIGHTWEIGHT_DIR, 'compress_simulate.py')
RESULT_DIR = os.path.join(LIGHTWEIGHT_DIR, 'result')

INQ_METHOD = {
    "id": "physical_prune_inq",
    "name": "物理通道剪枝 + INQ 2的幂次量化",
    "prune_type": "torch_pruning（物理删除通道）",
    "quant_type": "INQ（增量网络量化 · 2的幂次）",
    "description": "利用 torch_pruning 进行物理结构化剪枝，真实移除冗余通道。"
                   "配合 INQ（Incremental Network Quantization）将权重逐步量化为 2 的幂次值，"
                   "硬件部署时乘法运算可用移位操作替代，显著降低计算开销。",
    "params": {
        "prune_ratio": {"label": "剪枝比例", "default": 0.15, "min": 0.05, "max": 0.5, "step": 0.05},
        "weight_bits": {"label": "量化位宽", "default": 8, "options": [4, 6, 8]},
        "epochs_per_step": {"label": "每阶段训练轮数", "default": 4, "min": 1, "max": 20},
        "batch_size": {"label": "训练批大小", "default": 64, "min": 8, "max": 512, "step": 8},
    },
}


def get_method() -> Dict:
    return INQ_METHOD


def get_available_models() -> List[Dict]:
    """扫描所有训练任务输出目录中的 .pth 模型文件"""
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
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            return {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    except Exception:
        pass
    return {}


def get_available_datasets() -> List[Dict]:
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


def run_compress(model_path: str, params: Dict) -> Dict:
    """启动 INQ 压缩任务（后台子进程）"""
    if _running_task["active"]:
        status = get_status()
        if status["running"]:
            return {"status": "error", "message": "已有压缩任务正在运行，请等待完成"}
        _running_task["active"] = False

    fast_mode = params.pop("fast_mode", False)

    if not model_path or not os.path.isfile(model_path):
        return {"status": "error", "message": f"模型文件不存在: {model_path}"}

    if not fast_mode and not os.path.isfile(INQ_SCRIPT):
        return {"status": "error", "message": "INQ 压缩脚本不存在"}

    meta = _read_model_meta(model_path)
    model_type = meta.get("model_type", params.get("model_type", ""))
    dataset_type = meta.get("dataset_type", params.get("dataset_type", ""))
    num_classes = meta.get("num_classes", params.get("num_classes", 0))

    if not model_type or not dataset_type or not num_classes:
        return {"status": "error", "message": "无法从模型中读取 model_type/dataset_type/num_classes"}

    task_id = ""
    tasks_root = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    if tasks_root in model_path:
        parts = model_path.replace(tasks_root + "/", "").split("/")
        if parts:
            task_id = parts[0]

    data_path = params.get("data_path") or _find_data_path(task_id, dataset_type)
    if not data_path:
        return {"status": "error",
                "message": f"找不到 {dataset_type} 的数据文件，请确认 dataset/splits/{dataset_type}/ 下有数据"}

    os.makedirs(RESULT_DIR, exist_ok=True)

    run_config = {
        "model_path": model_path,
        "model_type": model_type,
        "dataset_type": dataset_type,
        "data_path": data_path,
        "num_classes": num_classes,
        "prune_ratio": params.get("prune_ratio", 0.15),
        "weight_bits": params.get("weight_bits", 8),
        "inq_steps": [0.5, 0.75, 0.82, 1.0],
        "epochs_per_step": params.get("epochs_per_step", 4),
        "batch_size": params.get("batch_size", 64),
        "learning_rate": float(params.get("learning_rate", 5e-4)),
        "device": "cuda:0" if _has_cuda() else "cpu",
        "save_dir": RESULT_DIR,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(LIGHTWEIGHT_DIR, f"_run_inq_{ts}.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    log_file = os.path.join(LIGHTWEIGHT_DIR, f"_run_inq_{ts}.log")

    python_exe = sys.executable
    if fast_mode:
        cmd = [python_exe, SIMULATE_SCRIPT, config_file, "inq"]
    else:
        cmd = [python_exe, INQ_SCRIPT, config_file]

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
    _running_task["method"] = INQ_METHOD["name"]
    _running_task["log_file"] = log_file
    _running_task["log_fh"] = log_fh
    _running_task["start_time"] = time.time()

    return {"status": "started", "pid": proc.pid, "method": INQ_METHOD["name"]}


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_compression_history() -> List[Dict]:
    results = []
    if not os.path.isdir(RESULT_DIR):
        return results

    for fname in sorted(os.listdir(RESULT_DIR)):
        fpath = os.path.join(RESULT_DIR, fname)
        if fname.endswith('.json') and fname.startswith('result_inq'):
            try:
                with open(fpath, 'r') as f:
                    results.append(json.load(f))
            except Exception:
                pass
        elif fname.endswith('.pth') and 'inq' in fname:
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            results.append({'name': fname, 'path': fpath, 'size_mb': size_mb, 'type': 'model'})
        elif fname.endswith('.png') and 'inq' in fname:
            results.append({'name': fname, 'path': fpath, 'type': 'confusion_matrix'})
    return results

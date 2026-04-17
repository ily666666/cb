"""
任务管理服务 - 对接 run_task.py 的任务执行系统
"""
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import (
    TASKS_ROOT, DATASET_CONFIG, SUPPORTED_TASKS,
    PIPELINE_MODES, KNOWN_DATASETS, get_dataset_from_task_id
)

_running_tasks: Dict[str, Dict[str, Any]] = {}


def _abs(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)


def list_tasks() -> List[Dict]:
    tasks_dir = _abs(TASKS_ROOT)
    if not os.path.isdir(tasks_dir):
        return []

    result = []
    for d in sorted(os.listdir(tasks_dir)):
        task_path = os.path.join(tasks_dir, d)
        if not os.path.isdir(task_path) or d.startswith('.'):
            continue

        dataset = None
        for ds in KNOWN_DATASETS:
            if ds in d.lower():
                dataset = ds
                break

        purpose = ""
        d_lower = d.lower()
        if "collab" in d_lower:
            purpose = "协同推理"
        elif "cloud_only" in d_lower:
            purpose = "仅云推理"
        elif "edge_only" in d_lower:
            purpose = "仅边推理"
        elif "train" in d_lower:
            purpose = "训练"

        input_dir = os.path.join(task_path, "input")
        config_files = []
        if os.path.isdir(input_dir):
            config_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        result.append({
            "task_id": d,
            "dataset": dataset,
            "purpose": purpose,
            "has_input": os.path.isdir(os.path.join(task_path, "input")),
            "has_output": os.path.isdir(os.path.join(task_path, "output")),
            "has_result": os.path.isdir(os.path.join(task_path, "result")),
            "config_files": config_files,
        })
    return result


def get_task_detail(task_id: str) -> Optional[Dict]:
    task_path = _abs(f"{TASKS_ROOT}/{task_id}")
    if not os.path.isdir(task_path):
        return None

    detail = {"task_id": task_id, "input": {}, "output": {}, "result": {}}

    input_dir = os.path.join(task_path, "input")
    if os.path.isdir(input_dir):
        for f in os.listdir(input_dir):
            if f.endswith('.json'):
                fpath = os.path.join(input_dir, f)
                try:
                    with open(fpath, 'r', encoding='utf-8') as fp:
                        detail["input"][f] = json.load(fp)
                except Exception:
                    detail["input"][f] = {"_error": "无法解析"}

    output_dir = os.path.join(task_path, "output")
    if os.path.isdir(output_dir):
        for step_dir in sorted(os.listdir(output_dir)):
            step_path = os.path.join(output_dir, step_dir)
            if os.path.isdir(step_path):
                files = os.listdir(step_path)
                timing = None
                timing_path = os.path.join(step_path, "timing.json")
                if os.path.exists(timing_path):
                    try:
                        with open(timing_path, 'r', encoding='utf-8') as fp:
                            timing = json.load(fp)
                    except Exception:
                        pass
                detail["output"][step_dir] = {
                    "files": files,
                    "timing": timing,
                }

    result_dir = os.path.join(task_path, "result")
    if os.path.isdir(result_dir):
        for root, dirs, files in os.walk(result_dir):
            for f in files:
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, result_dir)
                if f.endswith('.txt'):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as fp:
                            detail["result"][rel] = fp.read()
                    except Exception:
                        detail["result"][rel] = "[读取失败]"
                else:
                    size = os.path.getsize(fpath) / (1024 * 1024)
                    detail["result"][rel] = f"[文件: {size:.2f} MB]"

    running = _running_tasks.get(task_id)
    if running:
        detail["running"] = {
            "status": running.get("status", "unknown"),
            "started_at": running.get("started_at", ""),
            "command": running.get("command", ""),
            "output_lines": running.get("output_lines", [])[-50:],
        }

    return detail


def get_task_timing(task_id: str) -> Optional[Dict]:
    output_root = _abs(f"{TASKS_ROOT}/{task_id}/output")
    if not os.path.isdir(output_root):
        return None

    steps = {}
    total_time = 0.0

    for step_dir in sorted(os.listdir(output_root)):
        timing_path = os.path.join(output_root, step_dir, 'timing.json')
        if not os.path.exists(timing_path):
            continue
        try:
            with open(timing_path, 'r', encoding='utf-8') as f:
                t = json.load(f)
            if 'step_time' in t:
                st = t['step_time']
            else:
                st = sum(t.get(k, 0) for k in [
                    'data_load_time', 'preprocess_time', 'data_save_time',
                    'transfer_time', 'model_load_time', 'warmup_time', 'inference_time'
                ])
            steps[step_dir] = {'step_time': st}
            total_time += st
        except Exception:
            pass

    return {"steps": steps, "total_time": total_time}


def clean_task_output(task_id: str) -> Dict:
    """清空指定任务的 output 和 result 目录"""
    import shutil
    task_dir = _abs(f"{TASKS_ROOT}/{task_id}")
    if not os.path.isdir(task_dir):
        return {"status": "error", "message": f"任务 {task_id} 不存在"}

    if task_id in _running_tasks and _running_tasks[task_id].get("status") == "running":
        return {"status": "error", "message": f"任务 {task_id} 正在运行，无法清空"}

    removed = []
    for sub in ("output", "result"):
        p = os.path.join(task_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
            removed.append(sub)

    _running_tasks.pop(task_id, None)

    if not removed:
        return {"status": "ok", "message": "无需清空，output/result 目录不存在"}
    return {"status": "ok", "message": f"已清空: {', '.join(removed)}"}


def run_task_async(task_id: str, mode: str = None, step: str = None,
                   config: str = None, edge_id: int = None, summary: bool = False,
                   fast_mode: bool = False) -> Dict:
    if task_id in _running_tasks and _running_tasks[task_id].get("status") == "running":
        return {"status": "error", "message": f"任务 {task_id} 正在运行中"}

    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "run_task.py"), "--task_id", task_id]
    if mode:
        try:
            ds = get_dataset_from_task_id(task_id)
            prefixed = f"{ds}_{mode}"
            if prefixed in PIPELINE_MODES:
                mode = prefixed
        except ValueError:
            pass
        cmd.extend(["--mode", mode])
    elif step:
        cmd.extend(["--step", step])
    elif config:
        cmd.extend(["--config", config])
    else:
        return {"status": "error", "message": "需要指定 mode, step 或 config"}

    if edge_id is not None:
        cmd.extend(["--edge_id", str(edge_id)])
    if summary:
        cmd.append("--summary")

    _running_tasks[task_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "command": " ".join(cmd),
        "output_lines": [],
        "proc": None,
        "_line_event": threading.Event(),
    }

    def _run():
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if fast_mode:
            env["FAST_MODE"] = "1"
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=PROJECT_ROOT, bufsize=1, env=env,
            )
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            proc = subprocess.Popen(cmd, **popen_kwargs)
            _running_tasks[task_id]["proc"] = proc
            for line in iter(proc.stdout.readline, ''):
                _running_tasks[task_id]["output_lines"].append(line.rstrip())
                _running_tasks[task_id]["_line_event"].set()
            proc.wait()
            _running_tasks[task_id]["status"] = "success" if proc.returncode == 0 else "error"
            _running_tasks[task_id]["exit_code"] = proc.returncode
        except Exception as e:
            _running_tasks[task_id]["status"] = "error"
            _running_tasks[task_id]["output_lines"].append(f"[异常] {e}")
        finally:
            _running_tasks[task_id]["_line_event"].set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "message": f"任务 {task_id} 已启动", "command": " ".join(cmd)}


_INTERNAL_KEYS = {"proc", "_line_event"}


def get_task_run_status(task_id: str) -> Optional[Dict]:
    info = _running_tasks.get(task_id)
    if not info:
        return None
    return {k: v for k, v in info.items() if k not in _INTERNAL_KEYS}


def get_active_tasks() -> List[Dict]:
    """返回所有已执行过的任务摘要（不含 output_lines），按启动时间倒序，running 的排最前"""
    items = []
    for tid, info in _running_tasks.items():
        items.append({
            "task_id": tid,
            "status": info.get("status", "unknown"),
            "started_at": info.get("started_at", ""),
            "command": info.get("command", ""),
        })
    items.sort(key=lambda x: (x["status"] != "running", x["started_at"]), reverse=False)
    return items


def stop_task(task_id: str) -> Dict:
    """终止正在运行的任务进程"""
    info = _running_tasks.get(task_id)
    if not info:
        return {"status": "error", "message": f"任务 {task_id} 无运行记录"}
    if info.get("status") != "running":
        return {"status": "error", "message": f"任务 {task_id} 未在运行"}

    proc = info.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            proc.kill()

    info["status"] = "stopped"
    info["output_lines"].append("\n[已手动停止]")
    return {"status": "ok", "message": f"任务 {task_id} 已停止"}


def remove_task_record(task_id: str) -> Dict:
    """从监控列表中删除任务记录（不删文件）"""
    info = _running_tasks.get(task_id)
    if not info:
        return {"status": "error", "message": f"任务 {task_id} 无运行记录"}
    if info.get("status") == "running":
        return {"status": "error", "message": f"任务 {task_id} 正在运行，请先停止"}
    _running_tasks.pop(task_id, None)
    return {"status": "ok", "message": f"已移除 {task_id} 的运行记录"}


def get_pipeline_modes() -> Dict:
    return {k: v for k, v in PIPELINE_MODES.items()
            if not any(k.startswith(f"{ds}_") for ds in KNOWN_DATASETS)}


def get_supported_steps() -> List[str]:
    return [s for s in SUPPORTED_TASKS
            if not any(s.startswith(f"{ds}_") for ds in KNOWN_DATASETS)]

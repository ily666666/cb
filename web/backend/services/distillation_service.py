"""
知识蒸馏服务 —— 管理蒸馏任务的启动、状态查询与结果收集
复用已有的 task 基础设施（edge_kd 步骤）
"""
import json
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from config_refactor import TASKS_ROOT
from services.task_service import run_task_async, get_task_run_status, stop_task


KD_MODE = "knowledge_distillation"
KD_STEPS = ["edge_kd"]


def get_train_tasks() -> List[Dict]:
    """返回可用于知识蒸馏的训练任务列表"""
    tasks_root = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    if not os.path.isdir(tasks_root):
        return []

    results = []
    for task_id in sorted(os.listdir(tasks_root)):
        task_dir = os.path.join(tasks_root, task_id)
        if not os.path.isdir(task_dir):
            continue
        if "train" not in task_id.lower():
            continue

        input_dir = os.path.join(task_dir, "input")
        if not os.path.isdir(input_dir):
            continue

        has_kd_config = False
        teacher_model = ""
        student_model = ""
        dataset_type = ""
        config_files = []

        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(input_dir, fname)
            config_files.append(fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if 'kd' in fname.lower() or 'edge' in fname.lower():
                    has_kd_config = True
                    teacher_model = cfg.get("teacher_model_type", cfg.get("model_type", ""))
                    student_model = cfg.get("student_model_type", cfg.get("model_type", ""))
                    dataset_type = cfg.get("dataset_type", "")
            except Exception:
                pass

        output_dir = os.path.join(task_dir, "output")
        has_output = os.path.isdir(output_dir) and bool(os.listdir(output_dir))

        has_teacher = False
        if os.path.isdir(output_dir):
            for step_dir in os.listdir(output_dir):
                step_path = os.path.join(output_dir, step_dir)
                if os.path.isdir(step_path):
                    for f in os.listdir(step_path):
                        if 'teacher' in f and f.endswith('.pth'):
                            has_teacher = True
                            break

        results.append({
            "task_id": task_id,
            "dataset_type": dataset_type,
            "teacher_model": teacher_model,
            "student_model": student_model,
            "has_kd_config": has_kd_config,
            "has_teacher": has_teacher,
            "has_output": has_output,
            "config_files": config_files,
        })
    return results


def start_distillation(task_id: str, fast_mode: bool = False, accuracy: float = None) -> Dict:
    """启动知识蒸馏任务"""
    if fast_mode and accuracy is not None:
        _update_kd_accuracy(task_id, accuracy)
    return run_task_async(task_id, mode=KD_MODE, fast_mode=fast_mode)


def _update_kd_accuracy(task_id: str, accuracy: float):
    """将目标准确率写入所有 KD 配置的 display_config"""
    input_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "input")
    if not os.path.isdir(input_dir):
        return
    for fname in os.listdir(input_dir):
        if not fname.endswith('.json') or 'kd' not in fname.lower():
            continue
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dc = data.setdefault('display_config', {})
            dc['accuracy'] = accuracy
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception:
            pass


def get_distillation_status(task_id: str) -> Dict:
    """获取蒸馏任务运行状态"""
    status = get_task_run_status(task_id)
    if not status:
        return {"status": "idle", "task_id": task_id}
    return status


def stop_distillation(task_id: str) -> Dict:
    """停止蒸馏任务"""
    return stop_task(task_id)


def get_distillation_history(task_id: str) -> Optional[Dict]:
    """读取蒸馏产出的 kd_history.pkl"""
    import pickle
    output_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "output")
    if not os.path.isdir(output_root):
        return None

    def _to_list(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        if isinstance(v, list) and v and hasattr(v[0], 'tolist'):
            return [x.tolist() if hasattr(x, 'tolist') else x for x in v]
        return v

    for step_dir in sorted(os.listdir(output_root)):
        step_path = os.path.join(output_root, step_dir)
        kd_path = os.path.join(step_path, "kd_history.pkl")
        if os.path.isfile(kd_path):
            try:
                with open(kd_path, 'rb') as f:
                    raw = pickle.load(f)
                if isinstance(raw, dict):
                    edges = {}
                    for edge_key, edge_hist in raw.items():
                        if isinstance(edge_hist, dict):
                            edges[edge_key] = {k: _to_list(v) for k, v in edge_hist.items()}
                    return {
                        "step": step_dir,
                        "edges": edges,
                    }
            except Exception:
                pass
    return None


_MODEL_PARAM_TABLE = {
    'real_resnet101_ratr':      33_011_011,
    'real_resnet101_ratr_2048': 33_011_011,
    'real_resnet10_ratr':        7_970_019,
    'real_resnet7_ratr_cp':        848_099,
    'complex_resnet50_link11':            None,
    'complex_resnet50_link11_with_attention': None,
    'real_resnet20_link11':               None,
    'real_resnet20_link11_h':             None,
    'real_resnet9_link11':                None,
}


def _estimate_model_size(model_type: str, num_classes: int, dataset_type: str):
    """从代码模型结构实例化模型，计算参数量并估算 .pth 文件大小(MB)。
    优先用 torch 动态计算，失败时查静态参数表。
    返回 (size_mb, param_count)"""
    try:
        from core.model_factory import create_model_by_type
        model = create_model_by_type(model_type, num_classes, dataset_type)
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = round(total_params * 4 / (1024 * 1024), 2)
        return size_mb, total_params
    except Exception:
        pass
    params = _MODEL_PARAM_TABLE.get(model_type)
    if params:
        return round(params * 4 / (1024 * 1024), 2), params
    return 0, 0


def get_output_models(task_id: str) -> List[Dict]:
    """获取蒸馏任务产出的模型文件，无权重文件时从模型结构估算大小"""
    output_root = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "output")

    models = []
    if os.path.isdir(output_root):
        for step_dir in sorted(os.listdir(output_root)):
            step_path = os.path.join(output_root, step_dir)
            if not os.path.isdir(step_path):
                continue
            for fname in sorted(os.listdir(step_path)):
                if not fname.endswith('.pth'):
                    continue
                fpath = os.path.join(step_path, fname)
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                param_count = int(size_mb * 1024 * 1024 / 4)
                role = 'teacher' if 'teacher' in fname else 'student' if 'student' in fname else 'other'
                models.append({
                    "name": fname,
                    "step": step_dir,
                    "path": fpath,
                    "size_mb": size_mb,
                    "param_count": param_count,
                    "role": role,
                })

    has_teacher = any(m['role'] == 'teacher' for m in models)
    has_student = any(m['role'] == 'student' for m in models)

    if not has_teacher or not has_student:
        input_dir = os.path.join(PROJECT_ROOT, TASKS_ROOT, task_id, "input")
        if os.path.isdir(input_dir):
            for fname in sorted(os.listdir(input_dir)):
                if not fname.endswith('.json') or 'kd' not in fname.lower():
                    continue
                try:
                    with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    ds = cfg.get('dataset_type', '')
                    nc = cfg.get('num_classes', 10)
                    if not has_teacher:
                        t_type = cfg.get('teacher_model_type', '')
                        if t_type:
                            sz, pc = _estimate_model_size(t_type, nc, ds)
                            if sz > 0:
                                models.append({"name": t_type, "step": "estimated", "path": "",
                                               "size_mb": sz, "param_count": pc, "role": "teacher"})
                                has_teacher = True
                    if not has_student:
                        s_type = cfg.get('student_model_type', '')
                        if s_type:
                            sz, pc = _estimate_model_size(s_type, nc, ds)
                            if sz > 0:
                                models.append({"name": s_type, "step": "estimated", "path": "",
                                               "size_mb": sz, "param_count": pc, "role": "student"})
                                has_student = True
                    if has_teacher and has_student:
                        break
                except Exception:
                    pass

    return models

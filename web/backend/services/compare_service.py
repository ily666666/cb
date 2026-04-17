"""
对比分析服务 - 读取推理任务配置、更新模拟参数、复制方案、获取对比结果
"""
import json
import os
import shutil
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TASKS_ROOT = 'tasks'
LABELS_FILE = os.path.join(PROJECT_ROOT, TASKS_ROOT, '_compare_labels.json')

_PURPOSE_MAP = {
    'collab': '云边协同推理',
    'cloud_only': '纯云推理',
    'edge_only': '仅边推理',
    'edge_fed': '边推理（联邦）',
    'edge_nofed': '边推理（非联邦）',
    'edge_local': '边推理（本地）',
}

_MODE_MAP = {
    'collab': 'device_to_edge_to_cloud',
    'cloud_only': 'device_to_cloud',
    'edge_only': 'device_to_edge',
    'edge_fed': 'device_to_edge',
    'edge_nofed': 'device_to_edge',
    'edge_local': 'device_to_edge',
}

_DS_LABELS = {
    'link11': 'Link-11',
    'rml2016': 'RML2016',
    'radar': 'Radar',
    'ratr': 'RATR',
}


def _tasks_dir():
    return os.path.join(PROJECT_ROOT, TASKS_ROOT)


def _load_labels() -> Dict[str, str]:
    if os.path.isfile(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_labels(labels: Dict[str, str]):
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def get_tasks_for_compare() -> List[Dict]:
    """获取所有推理任务及其步骤配置，供对比分析使用"""
    tasks_dir = _tasks_dir()
    if not os.path.isdir(tasks_dir):
        return []

    labels = _load_labels()
    result = []
    for task_id in sorted(os.listdir(tasks_dir)):
        if task_id.startswith('_') or task_id.startswith('.'):
            continue
        if 'train' in task_id.lower():
            continue
        task_dir = os.path.join(tasks_dir, task_id)
        input_dir = os.path.join(task_dir, 'input')
        if not os.path.isdir(input_dir):
            continue

        d_lower = task_id.lower()
        purpose = ''
        mode_suffix = ''
        for key in _PURPOSE_MAP:
            if key in d_lower:
                purpose = _PURPOSE_MAP[key]
                mode_suffix = _MODE_MAP[key]
                break
        if not purpose:
            continue

        if task_id in labels:
            purpose = labels[task_id]

        dataset = ''
        for ds in _DS_LABELS:
            if ds in d_lower:
                dataset = ds
                break

        steps = []
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(input_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                dc = config.get('display_config')
                if dc is None:
                    continue
                steps.append({
                    'step_name': fname[:-5],
                    'file': fname,
                    'display_config': {
                        'simulate_realtime': dc.get('simulate_realtime', False),
                        'data_size_mb': dc.get('data_size_mb', 0),
                        'time': dc.get('time', 0),
                        'accuracy': dc.get('accuracy'),
                    }
                })
            except Exception:
                pass

        if not steps:
            continue

        total_time = sum(s['display_config'].get('time', 0) for s in steps)
        last_acc = None
        for s in steps:
            a = s['display_config'].get('accuracy')
            if a is not None:
                last_acc = a

        result.append({
            'task_id': task_id,
            'dataset': dataset,
            'dataset_label': _DS_LABELS.get(dataset, dataset),
            'purpose': purpose,
            'mode_suffix': mode_suffix,
            'steps': steps,
            'step_count': len(steps),
            'total_time': round(total_time, 2),
            'accuracy': last_acc,
        })

    return result


def update_task_label(task_id: str, label: str) -> Dict:
    """更新某个任务的自定义显示名称"""
    task_dir = os.path.join(_tasks_dir(), task_id)
    if not os.path.isdir(task_dir):
        return {'status': 'error', 'message': f'任务不存在: {task_id}'}
    labels = _load_labels()
    if label and label.strip():
        labels[task_id] = label.strip()
    else:
        labels.pop(task_id, None)
    _save_labels(labels)
    return {'status': 'ok'}


def update_task_summary(task_id: str, total_time: float, accuracy: float = None, label: str = None) -> Dict:
    """更新方案的总时间和准确率，按比例分摊时间到各步骤"""
    input_dir = os.path.join(_tasks_dir(), task_id, 'input')
    if not os.path.isdir(input_dir):
        return {'status': 'error', 'message': f'任务不存在: {task_id}'}

    if label is not None:
        update_task_label(task_id, label)

    step_files = []
    old_total = 0.0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dc = data.get('display_config')
            if dc is None:
                continue
            t = dc.get('time', 0)
            step_files.append((fpath, data, t))
            old_total += t
        except Exception:
            pass

    if not step_files:
        return {'status': 'error', 'message': '无可配置步骤'}

    for fpath, data, old_t in step_files:
        dc = data['display_config']
        if old_total > 0:
            dc['time'] = round(total_time * old_t / old_total, 2)
        else:
            dc['time'] = round(total_time / len(step_files), 2)

    last_fpath, last_data, _ = step_files[-1]
    if accuracy is not None:
        last_data['display_config']['accuracy'] = accuracy

    for fpath, data, _ in step_files:
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return {'status': 'ok'}


def update_step_config(task_id: str, step_name: str, config: Dict) -> Dict:
    """更新某个步骤的 display_config 模拟参数"""
    fpath = os.path.join(_tasks_dir(), task_id, 'input', f'{step_name}.json')
    if not os.path.isfile(fpath):
        return {'status': 'error', 'message': f'配置文件不存在: {step_name}.json'}

    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'display_config' not in data:
            data['display_config'] = {}

        for key in ['simulate_realtime', 'data_size_mb', 'time', 'accuracy']:
            if key in config:
                data['display_config'][key] = config[key]

        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return {'status': 'ok'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def clone_task(source_task_id: str, new_task_id: str, label: str = '') -> Dict:
    """复制一个任务的 input 目录为新方案，用于不同权重/参数对比"""
    src_dir = os.path.join(_tasks_dir(), source_task_id)
    dst_dir = os.path.join(_tasks_dir(), new_task_id)

    if not os.path.isdir(src_dir):
        return {'status': 'error', 'message': f'源任务不存在: {source_task_id}'}
    src_input = os.path.join(src_dir, 'input')
    if not os.path.isdir(src_input):
        return {'status': 'error', 'message': '源任务无 input 目录'}
    if os.path.exists(dst_dir):
        return {'status': 'error', 'message': f'任务ID已存在: {new_task_id}'}

    try:
        os.makedirs(dst_dir)
        shutil.copytree(src_input, os.path.join(dst_dir, 'input'))
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

    if label and label.strip():
        labels = _load_labels()
        labels[new_task_id] = label.strip()
        _save_labels(labels)

    return {'status': 'ok', 'task_id': new_task_id}


def delete_task(task_id: str) -> Dict:
    """删除一个对比方案（仅删除复制出来的）"""
    task_dir = os.path.join(_tasks_dir(), task_id)
    if not os.path.isdir(task_dir):
        return {'status': 'error', 'message': f'任务不存在: {task_id}'}
    try:
        shutil.rmtree(task_dir)
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    labels = _load_labels()
    labels.pop(task_id, None)
    _save_labels(labels)
    return {'status': 'ok'}


def get_compare_results(task_ids: List[str]) -> List[Dict]:
    """直接从 display_config 读取配置值用于对比"""
    labels = _load_labels()
    results = []
    for task_id in task_ids:
        input_dir = os.path.join(_tasks_dir(), task_id, 'input')

        total_time = 0.0
        final_accuracy = None

        if os.path.isdir(input_dir):
            for fname in sorted(os.listdir(input_dir)):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    dc = cfg.get('display_config', {})
                    if dc.get('accuracy') is not None:
                        final_accuracy = dc['accuracy']
                    total_time += dc.get('time', 0)
                except Exception:
                    pass

        d_lower = task_id.lower()
        purpose = task_id
        if task_id in labels:
            purpose = labels[task_id]
        else:
            for key, lbl in _PURPOSE_MAP.items():
                if key in d_lower:
                    purpose = lbl
                    break

        results.append({
            'task_id': task_id,
            'purpose': purpose,
            'total_time': round(total_time, 2),
            'accuracy': final_accuracy,
        })

    return results

"""
任务管理工具
"""
import os
from datetime import datetime


def create_task_id(priority="001", kind="COLLAB"):
    """
    生成任务ID
    
    Args:
        priority: 任务优先级（如 '001'）
        kind: 任务类型（如 'COLLAB', 'EDGE', 'CLOUD'）
    
    Returns:
        str: 任务ID，格式：TASK_{priority}_{kind}_{timestamp}_{millisecond}
    
    Example:
        task_id = create_task_id('001', 'COLLAB')
        # 'TASK_001_COLLAB_20260212_153045_789'
    """
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    millisecond = now.microsecond // 1000
    
    task_id = f"TASK_{priority}_{kind}_{time_str}_{millisecond:03d}"
    return task_id


def setup_task_dirs(task_id):
    """
    创建任务目录结构
    
    Args:
        task_id: 任务ID
    
    Creates:
        tasks/{task_id}/
        ├── input/
        ├── output/
        │   ├── device_load/
        │   ├── edge_infer/
        │   └── cloud_infer/
        └── result/
            ├── device_load/
            ├── edge_infer/
            └── cloud_infer/
    
    Example:
        setup_task_dirs('TASK_001_COLLAB_20260212_153045_789')
    """
    base_dir = f"./tasks/{task_id}"
    
    # 创建主目录
    dirs_to_create = [
        os.path.join(base_dir, 'input'),
        os.path.join(base_dir, 'output', 'device_load'),
        os.path.join(base_dir, 'output', 'edge_infer'),
        os.path.join(base_dir, 'output', 'cloud_infer'),
        os.path.join(base_dir, 'result', 'device_load'),
        os.path.join(base_dir, 'result', 'edge_infer'),
        os.path.join(base_dir, 'result', 'cloud_infer'),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"[任务管理] 创建任务目录: {base_dir}")


def get_task_dirs(task_id):
    """
    获取任务目录路径
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 包含所有目录路径的字典
    
    Example:
        dirs = get_task_dirs('TASK_001')
        print(dirs['input'])  # './tasks/TASK_001/input'
    """
    base_dir = f"./tasks/{task_id}"
    
    return {
        'root': base_dir,
        'input': os.path.join(base_dir, 'input'),
        'output': os.path.join(base_dir, 'output'),
        'result': os.path.join(base_dir, 'result'),
        'output_device': os.path.join(base_dir, 'output', 'device_load'),
        'output_edge': os.path.join(base_dir, 'output', 'edge_infer'),
        'output_cloud': os.path.join(base_dir, 'output', 'cloud_infer'),
        'result_device': os.path.join(base_dir, 'result', 'device_load'),
        'result_edge': os.path.join(base_dir, 'result', 'edge_infer'),
        'result_cloud': os.path.join(base_dir, 'result', 'cloud_infer'),
    }


def list_tasks(tasks_root="./tasks"):
    """
    列出所有任务
    
    Args:
        tasks_root: 任务根目录
    
    Returns:
        list: 任务ID列表
    """
    if not os.path.exists(tasks_root):
        return []
    
    tasks = []
    for item in os.listdir(tasks_root):
        if os.path.isdir(os.path.join(tasks_root, item)) and item.startswith('TASK_'):
            tasks.append(item)
    
    return sorted(tasks, reverse=True)  # 最新的任务在前

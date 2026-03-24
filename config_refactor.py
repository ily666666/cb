"""
全局配置文件 - 改造后的架构
定义所有路径和默认参数
"""
import torch
import os

# ==================== 设备配置 ====================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==================== 任务目录配置 ====================
TASKS_ROOT = "./tasks"
TASK_INPUT_DIR = "./tasks/{}/input"
TASK_OUTPUT_DIR = "./tasks/{}/output"
TASK_RESULT_DIR = "./tasks/{}/result"

# ==================== 数据集配置 ====================
DATASET_CONFIG = {
    'link11': {
        'num_classes': 7,
        'signal_length': 1024,
        'cloud_model': 'complex_resnet50_link11',
        'edge_model': 'real_resnet20_link11',
    },
    'rml2016': {
        'num_classes': 6,
        'signal_length': 600,
        'cloud_model': 'complex_resnet50_rml2016',
        'edge_model': 'real_resnet20_rml2016',
    },
    'radar': {
        'num_classes': 7,
        'signal_length': 500,
        'cloud_model': 'complex_resnet50_radar',
        'edge_model': 'real_resnet20_radar',
    },
    'ratr': {
        'num_classes': 3,
        'signal_length': 1024,
        'cloud_model': 'real_resnet101_ratr',
        'edge_model': 'real_resnet10_ratr',
    },
}

# ==================== 模型路径配置 ====================
CLOUD_MODEL_DIR = "./run/cloud/pth"
EDGE_MODEL_DIR = "./run/edge/pth"

# ==================== 推理参数默认值 ====================
DEFAULT_BATCH_SIZE = 128
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# ==================== 支持的任务类型 ====================
SUPPORTED_TASKS = [
    # 推理类
    'device_load',              # 端侧数据加载
    'edge_infer',               # 边侧推理
    'cloud_infer',              # 云侧协同推理（低置信度）
    'cloud_direct_infer',       # 云侧直接推理（全量数据）
    # 训练类
    'cloud_pretrain',           # 预训练教师模型
    'edge_kd',                  # 知识蒸馏-各边分别蒸馏
    'federated_train',          # 联邦学习训练（单机模拟）
    'federated_cloud',          # 联邦学习-云侧聚合（分布式模式）
    'federated_edge',           # 联邦学习-边侧本地训练（分布式模式）
    'federated_server',         # 向后兼容旧名称 → federated_cloud
    # link11 数据集专用
    'link11_device_load', 'link11_edge_infer', 'link11_cloud_infer', 'link11_cloud_direct_infer',
    'link11_cloud_pretrain', 'link11_edge_kd', 'link11_federated_train',
    'link11_federated_cloud', 'link11_federated_edge', 'link11_federated_server',
    # rml2016 数据集专用
    'rml2016_device_load', 'rml2016_edge_infer', 'rml2016_cloud_infer', 'rml2016_cloud_direct_infer',
    'rml2016_cloud_pretrain', 'rml2016_edge_kd', 'rml2016_federated_train',
    'rml2016_federated_cloud', 'rml2016_federated_edge', 'rml2016_federated_server',
    # radar 数据集专用
    'radar_device_load', 'radar_edge_infer', 'radar_cloud_infer', 'radar_cloud_direct_infer',
    'radar_cloud_pretrain', 'radar_edge_kd', 'radar_federated_train',
    'radar_federated_cloud', 'radar_federated_edge', 'radar_federated_server',

    # ratr 数据集专用
    'ratr_device_load', 'ratr_edge_infer', 'ratr_cloud_infer', 'ratr_cloud_direct_infer',
    'ratr_cloud_pretrain', 'ratr_edge_kd', 'ratr_federated_train',
    'ratr_federated_cloud', 'ratr_federated_edge', 'ratr_federated_server',
]

# ==================== 支持的流水线模式 ====================
PIPELINE_MODES = {
    # 推理模式
    'device_to_cloud': ['device_load', 'cloud_direct_infer'],
    'device_to_edge': ['device_load', 'edge_infer'],
    'device_to_edge_to_cloud': ['device_load', 'edge_infer', 'cloud_infer'],
    # 训练模式
    'pretrain': ['cloud_pretrain'],
    'knowledge_distillation': ['edge_kd'],
    'federated_learning': ['federated_train'],
    'federated_cloud': ['federated_cloud'],
    'federated_edge': ['federated_edge'],
    'federated_server': ['federated_cloud'],  # 向后兼容旧名称
    # 完整训练+推理
    'full_train': ['cloud_pretrain', 'edge_kd', 'federated_train'],
    'full_pipeline': ['cloud_pretrain', 'edge_kd', 'federated_train',
                      'device_load', 'edge_infer', 'cloud_infer'],
}

# 兼容旧名称
INFERENCE_MODES = PIPELINE_MODES

KNOWN_DATASETS = list(DATASET_CONFIG.keys())   # ['link11', 'rml2016', 'radar']


def get_dataset_from_task_id(task_id):
    """从 task_id 提取数据集名称，如 '001_COLLAB_link11_test' → 'link11'"""
    for ds in KNOWN_DATASETS:
        if ds in task_id:
            return ds
    raise ValueError(f"无法从 task_id '{task_id}' 识别数据集，支持: {KNOWN_DATASETS}")


# ==================== 辅助函数 ====================
def get_task_path(task_id, path_type='root'):
    """
    获取任务路径
    
    Args:
        task_id: 任务ID
        path_type: 路径类型 ('root', 'input', 'output', 'result')
    
    Returns:
        str: 任务路径
    """
    if path_type == 'root':
        return os.path.join(TASKS_ROOT, task_id)
    elif path_type == 'input':
        return TASK_INPUT_DIR.format(task_id)
    elif path_type == 'output':
        return TASK_OUTPUT_DIR.format(task_id)
    elif path_type == 'result':
        return TASK_RESULT_DIR.format(task_id)
    else:
        raise ValueError(f"未知的路径类型: {path_type}")


def get_dataset_config(dataset_type):
    """
    获取数据集配置
    
    Args:
        dataset_type: 数据集类型 ('link11', 'rml2016', 'radar')
    
    Returns:
        dict: 数据集配置
    """
    if dataset_type not in DATASET_CONFIG:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    return DATASET_CONFIG[dataset_type]


def get_inference_pipeline(mode):
    """
    获取流水线（推理或训练）
    
    Args:
        mode: 模式名称
    
    Returns:
        list: 任务列表
    """
    if mode not in PIPELINE_MODES:
        available = ', '.join(PIPELINE_MODES.keys())
        raise ValueError(f"不支持的模式: {mode}\n可用模式: {available}")
    return PIPELINE_MODES[mode]

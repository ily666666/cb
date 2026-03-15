"""
radar 训练链路回调（包装层）
所有逻辑复用 train_callback.py，仅通过 config_name 指定数据集专用配置
"""
from callback.registry import register_task
from callback.train_callback import (
    cloud_pretrain_callback,
    edge_kd_1_callback,
    edge_kd_2_callback,
    federated_train_callback,
    federated_cloud_callback,
    federated_edge_1_callback,
    federated_edge_2_callback,
)


@register_task
def radar_cloud_pretrain_callback(task_id, **kwargs):
    """radar 云侧预训练教师模型"""
    return cloud_pretrain_callback(task_id, config_name='radar_cloud_pretrain', **kwargs)


@register_task
def radar_edge_kd_callback(task_id, edge_id=None, config_name=None, **kwargs):
    """radar 知识蒸馏统一入口"""
    if config_name and '_' in config_name:
        parts = config_name.split('_')
        if parts[-1].isdigit():
            edge_id = int(parts[-1])
    if edge_id == 1:
        return radar_edge_kd_1_callback(task_id, edge_id=edge_id, config_name=config_name, **kwargs)
    elif edge_id == 2:
        return radar_edge_kd_2_callback(task_id, edge_id=edge_id, config_name=config_name, **kwargs)
    else:
        return {'status': 'error', 'message': f'不支持的 edge_id: {edge_id}'}


@register_task
def radar_edge_kd_1_callback(task_id, edge_id=None, config_name=None, **kwargs):
    """radar 知识蒸馏 - 边1"""
    return edge_kd_1_callback(task_id, edge_id=edge_id, config_name=config_name or 'radar_edge_kd_1', **kwargs)


@register_task
def radar_edge_kd_2_callback(task_id, edge_id=None, config_name=None, **kwargs):
    """radar 知识蒸馏 - 边2"""
    return edge_kd_2_callback(task_id, edge_id=edge_id, config_name=config_name or 'radar_edge_kd_2', **kwargs)


@register_task
def radar_federated_train_callback(task_id, **kwargs):
    """radar 联邦学习训练"""
    return federated_train_callback(task_id, config_name='radar_federated_train', **kwargs)


@register_task
def radar_federated_cloud_callback(task_id, **kwargs):
    """radar 联邦学习 - 云侧聚合"""
    return federated_cloud_callback(task_id, config_name='radar_federated_cloud', **kwargs)


@register_task
def radar_federated_edge_callback(task_id, edge_id=None, config_name=None, **kwargs):
    """radar 联邦学习 - 边侧统一入口"""
    if config_name and '_' in config_name:
        parts = config_name.split('_')
        if parts[-1].isdigit():
            edge_id = int(parts[-1])
    if edge_id == 1:
        return radar_federated_edge_1_callback(task_id, edge_id=edge_id, config_name=config_name, **kwargs)
    elif edge_id == 2:
        return radar_federated_edge_2_callback(task_id, edge_id=edge_id, config_name=config_name, **kwargs)
    else:
        return {'status': 'error', 'message': f'不支持的 edge_id: {edge_id}'}


@register_task
def radar_federated_edge_1_callback(task_id, edge_id=None, config_name=None, **kwargs):
    """radar 联邦学习 - 边1"""
    return federated_edge_1_callback(task_id, edge_id=edge_id, config_name=config_name or 'radar_federated_edge_1', **kwargs)


@register_task
def radar_federated_edge_2_callback(task_id, edge_id=None, config_name=None, **kwargs):
    """radar 联邦学习 - 边2"""
    return federated_edge_2_callback(task_id, edge_id=edge_id, config_name=config_name or 'radar_federated_edge_2', **kwargs)


@register_task
def radar_federated_server_callback(task_id, **kwargs):
    """radar 向后兼容：federated_server → federated_cloud"""
    return radar_federated_cloud_callback(task_id, **kwargs)

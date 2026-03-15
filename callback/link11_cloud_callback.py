"""
link11 云侧任务回调（包装层）
所有逻辑复用 cloud_callback.py，仅通过 config_prefix 指定数据集专用配置
"""
from callback.registry import register_task
from callback.cloud_callback import (
    cloud_direct_infer_callback,
    cloud_infer_callback,
)


@register_task
def link11_cloud_direct_infer_callback(task_id):
    """link11 端→云 直接推理回调"""
    return cloud_direct_infer_callback(task_id, config_prefix='link11_')


@register_task
def link11_cloud_infer_callback(task_id):
    """link11 端→边→云 协同推理回调（云侧部分）"""
    return cloud_infer_callback(task_id, config_prefix='link11_')

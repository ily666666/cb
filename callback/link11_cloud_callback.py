"""
link11 云侧任务回调（包装层）
逻辑复用 cloud_callback.py，通过 task_id 自动识别数据集
"""
from callback.registry import register_task
from callback.cloud_callback import (
    cloud_direct_infer_callback,
    cloud_infer_callback,
)


@register_task
def link11_cloud_direct_infer_callback(task_id, **kwargs):
    """link11 端→云 直接推理回调"""
    return cloud_direct_infer_callback(task_id)


@register_task
def link11_cloud_infer_callback(task_id, **kwargs):
    """link11 端→边→云 协同推理回调（云侧部分）"""
    return cloud_infer_callback(task_id)

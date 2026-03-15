"""
radar 端侧任务回调（包装层）
所有逻辑复用 device_callback.py，仅通过 config_prefix 指定数据集专用配置
"""
from callback.registry import register_task
from callback.device_callback import device_load_callback


@register_task
def radar_device_load_callback(task_id):
    """radar 端侧数据加载回调"""
    return device_load_callback(task_id, config_prefix='radar_')

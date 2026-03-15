"""
rml2016 端侧任务回调（包装层）
逻辑复用 device_callback.py，通过 task_id 自动识别数据集
"""
from callback.registry import register_task
from callback.device_callback import device_load_callback


@register_task
def rml2016_device_load_callback(task_id, **kwargs):
    """rml2016 端侧数据加载回调"""
    return device_load_callback(task_id)

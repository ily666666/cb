"""
link11 边侧任务回调（包装层）
所有逻辑复用 edge_callback.py，仅通过 config_prefix 指定数据集专用配置
"""
from callback.registry import register_task
from callback.edge_callback import edge_infer_callback


@register_task
def link11_edge_infer_callback(task_id):
    """link11 边侧推理回调"""
    return edge_infer_callback(task_id, config_prefix='link11_')

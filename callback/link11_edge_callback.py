"""
link11 边侧任务回调（包装层）
逻辑复用 edge_callback.py，通过 task_id 自动识别数据集
"""
from callback.registry import register_task
from callback.edge_callback import edge_infer_callback


@register_task
def link11_edge_infer_callback(task_id, **kwargs):
    """link11 边侧推理回调"""
    return edge_infer_callback(task_id)

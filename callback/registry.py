"""
任务注册表 - 装饰器注册机制
"""
import functools

# 全局任务注册表
TASK_REGISTRY = {}


def register_task(func):
    """
    任务注册装饰器
    
    将函数注册到全局任务注册表中，支持通过字符串名称动态调用
    
    Args:
        func: 要注册的任务函数
    
    Returns:
        func: 原函数（未修改）
    
    Example:
        @register_task
        def my_task_callback(task_id):
            # 任务逻辑
            pass
        
        # 动态调用
        execute_task('my_task_callback', task_id='TASK_001')
    """
    TASK_REGISTRY[func.__name__] = func
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def execute_task(task_name, task_id, **kwargs):
    """
    执行已注册的任务
    
    Args:
        task_name: 任务名称（函数名）
        task_id: 任务ID
        **kwargs: 额外参数传递给任务函数
    
    Returns:
        任务函数的返回值
    
    Raises:
        ValueError: 如果任务未注册
    
    Example:
        result = execute_task('device_load_callback', task_id='TASK_001')
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ', '.join(TASK_REGISTRY.keys())
        raise ValueError(
            f"任务未注册: {task_name}\n"
            f"可用任务: {available_tasks}"
        )
    
    task_func = TASK_REGISTRY[task_name]
    return task_func(task_id, **kwargs)


def list_registered_tasks():
    """
    列出所有已注册的任务
    
    Returns:
        list: 任务名称列表
    """
    return list(TASK_REGISTRY.keys())


def get_task_info(task_name):
    """
    获取任务信息
    
    Args:
        task_name: 任务名称
    
    Returns:
        dict: 任务信息（函数名、文档字符串等）
    """
    if task_name not in TASK_REGISTRY:
        return None
    
    task_func = TASK_REGISTRY[task_name]
    return {
        'name': task_name,
        'doc': task_func.__doc__,
        'module': task_func.__module__,
    }

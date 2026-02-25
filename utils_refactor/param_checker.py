"""
参数校验工具
"""
import json
import os


def check_parameters(param_path, param_list):
    """
    检查JSON参数文件的完整性
    
    Args:
        param_path: JSON文件路径
        param_list: 必需参数列表
    
    Returns:
        tuple: (result_dict, param_dict)
            - result_dict: 校验结果 {'valid': bool, 'missing': list, 'error': str}
            - param_dict: 参数字典
    
    Example:
        result, params = check_parameters('input/config.json', ['model_path', 'batch_size'])
        if not result['valid']:
            print(f"参数错误: {result}")
    """
    result = {}
    
    # 1. 检查文件是否存在
    if not os.path.exists(param_path):
        result['error'] = f"参数文件不存在: {param_path}"
        return result, {}
    
    data = {}
    
    # 2. 尝试读取 JSON 文件
    try:
        with open(param_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result['error'] = f"JSON 格式错误: {e}"
        return result, {}
    except Exception as e:
        result['error'] = f"读取文件失败: {e}"
        return result, {}
    
    # 3. 确保 data 是字典
    if not isinstance(data, dict):
        result['error'] = "参数文件内容必须是一个 JSON 对象（dict）"
        return result, {}
    
    # 4. 检查必需参数是否存在且不为 null/None
    missing = []
    for param in param_list:
        if param not in data or data[param] is None:
            missing.append(param)
    
    result['missing'] = missing
    result['valid'] = len(missing) == 0
    
    return result, data


def validate_task_config(task_id, task_name, required_params):
    """
    验证任务配置文件
    
    Args:
        task_id: 任务ID
        task_name: 任务名称
        required_params: 必需参数列表
    
    Returns:
        tuple: (is_valid, params_dict, error_message)
    
    Example:
        is_valid, params, error = validate_task_config(
            'TASK_001', 
            'edge_infer', 
            ['model_path', 'batch_size']
        )
    """
    config_path = f"./tasks/{task_id}/input/{task_name}.json"
    
    result, params = check_parameters(config_path, required_params)
    
    if 'error' in result:
        return False, {}, result['error']
    elif not result['valid']:
        missing_str = ', '.join(result['missing'])
        return False, {}, f"缺少必需参数: {missing_str}"
    
    return True, params, None

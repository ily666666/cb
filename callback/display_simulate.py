"""
推理展示模拟器

当 input JSON 中存在 display_config 时，跳过真实推理，
按配置参数输出模拟日志。

关键配置:
  simulate_realtime: true  → 用 time.sleep 模拟真实耗时（演示用）
  simulate_realtime: false → 立即打印全部日志，但显示的时间值仍然是配置值（默认）

display_config 示例（边侧）:
{
    "simulate_realtime": false,
    "total_samples": 100000,
    "inference_time": 8.60,
    "throughput": 11624.1,
    "accuracy": 97.28,
    "model_load_time": 0.35,
    "warmup_time": 0.29,
    "data_load_time": 0.5,
    "high_conf_samples": 99989,
    "low_conf_samples": 11,
    "high_conf_accuracy": 97.28,
    "low_conf_accuracy": 18.18
}
"""
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_refactor import TASKS_ROOT
from utils_refactor import save_timing


def _get_data_size_mb(data_path):
    """获取数据文件/目录的总大小(MB)"""
    if not os.path.exists(data_path):
        return 0
    if os.path.isfile(data_path):
        return os.path.getsize(data_path) / (1024 * 1024)
    total = 0
    for f in os.listdir(data_path):
        fp = os.path.join(data_path, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _maybe_sleep(seconds, realtime):
    """只在 realtime=True 时才真正 sleep"""
    if realtime and seconds > 0:
        time.sleep(seconds)


def _simulate_progress(label, total_samples, inference_time, batch_size, realtime=False):
    """
    输出进度行，打印时机与 batch_inference_* 一致。
    realtime=True  → sleep 到对应时间点再打印（逼真）
    realtime=False → 立即打印，但显示的"耗时"是根据配置计算的虚拟值
    """
    num_batches = max((total_samples + batch_size - 1) // batch_size, 1)

    print_points = []
    for batch_idx in range(num_batches):
        batch_num = batch_idx + 1
        is_last = (batch_num * batch_size) >= total_samples
        if batch_num % 10 == 0 or is_last:
            processed = min(batch_num * batch_size, total_samples)
            target_elapsed = batch_num / num_batches * inference_time
            print_points.append((processed, target_elapsed))

    if realtime:
        t_start = time.time()
        for processed, target_elapsed in print_points:
            sleep_needed = target_elapsed - (time.time() - t_start)
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            elapsed = time.time() - t_start
            progress = processed / total_samples * 100
            print(f"[{label}] 进度: {progress:.1f}% ({processed}/{total_samples}), 耗时: {elapsed:.2f}s")
        remaining = inference_time - (time.time() - t_start)
        if remaining > 0:
            time.sleep(remaining)
    else:
        for processed, target_elapsed in print_points:
            progress = processed / total_samples * 100
            print(f"[{label}] 进度: {progress:.1f}% ({processed}/{total_samples}), 耗时: {target_elapsed:.2f}s")


def _generate_report(report_path, title, sections):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for section_title, items in sections:
            f.write("=" * 60 + "\n")
            f.write(f"{section_title}\n")
            f.write("=" * 60 + "\n")
            for key, value in items:
                f.write(f"{key}: {value}\n")
            f.write("\n")


# ================================================================
# 边侧推理模拟
# ================================================================
def simulate_edge_infer(config, task_id, step_prefix=""):
    """模拟边侧推理日志输出，参数全部来自 display_config"""
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    batch_size = config.get('batch_size', 128)
    confidence_threshold = config.get('confidence_threshold', None)
    model_type = config.get('model_type', 'unknown')

    total_samples = dc.get('total_samples', 100000)
    inference_time = dc.get('inference_time', 8.0)
    throughput = dc.get('throughput', total_samples / inference_time if inference_time > 0 else 0)
    accuracy = dc.get('accuracy', 97.28)
    model_load_time = dc.get('model_load_time', 0.35)
    warmup_time = dc.get('warmup_time', 0.29)
    data_load_time = dc.get('data_load_time', 0.5)
    high_conf_samples = dc.get('high_conf_samples', total_samples)
    low_conf_samples = dc.get('low_conf_samples', 0)
    high_conf_accuracy = dc.get('high_conf_accuracy', accuracy)
    low_conf_accuracy = dc.get('low_conf_accuracy', 0.0)

    output_step = f"{step_prefix}edge_infer"

    print(f"\n{'='*60}")
    print(f"[边侧] 开始执行推理任务")
    print(f"{'='*60}")
    print(f"[配置] 模型路径: {config.get('model_path', 'N/A')}")
    print(f"[配置] 模型类型: {model_type}")
    if confidence_threshold is not None:
        print(f"[配置] 置信度阈值: {confidence_threshold}")
    else:
        print(f"[配置] 置信度阈值: 未设置（纯边侧模式，不筛选低置信度样本）")
    print(f"[配置] 计算设备: {config.get('device', 'cuda:0')}")

    print(f"[加载] 从 device_load 加载数据...")
    _maybe_sleep(data_load_time, realtime)
    print(f"[加载] 成功加载 {total_samples} 个样本")
    print(f"[计时] 数据加载耗时: {data_load_time:.2f}s")

    print(f"[模型] 正在加载边侧模型...")
    _maybe_sleep(model_load_time, realtime)
    print(f"[模型] 成功加载模型权重")

    data_shape = dc.get('data_shape', f"({total_samples}, 2, 128)")
    data_dtype = dc.get('data_dtype', 'complex64')
    print(f"[边侧推理] 数据格式: shape={data_shape}, dtype={data_dtype}")

    num_batches = (total_samples + batch_size - 1) // batch_size
    print(f"[推理] 开始批量推理: {total_samples} 样本, {num_batches} 批次")

    _maybe_sleep(warmup_time, realtime)
    print(f"[推理] CUDA 热身完成 ({warmup_time:.2f}s)")

    _simulate_progress("推理", total_samples, inference_time, batch_size, realtime)

    print(f"[推理] 完成! 纯推理耗时: {inference_time:.2f}s, 吞吐量: {throughput:.1f} samples/s")
    print(f"[结果] 边侧整体准确率: {accuracy:.2f}%")

    if confidence_threshold is not None:
        total = high_conf_samples + low_conf_samples
        high_ratio = high_conf_samples / total * 100 if total > 0 else 100
        low_ratio = low_conf_samples / total * 100 if total > 0 else 0
        print(f"[筛选] 高置信度样本: {high_conf_samples} ({high_ratio:.1f}%)")
        print(f"[筛选] 低置信度样本: {low_conf_samples} ({low_ratio:.1f}%)")
        print(f"[结果] 高置信度准确率: {high_conf_accuracy:.2f}%")
        print(f"[结果] 低置信度准确率: {low_conf_accuracy:.2f}%")
    else:
        low_conf_samples = 0
        print(f"[信息] 纯边侧模式，跳过低置信度筛选")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)

    save_timing(output_dir, {
        'data_load_time': data_load_time,
        'transfer_time': 0,
        'model_load_time': model_load_time,
        'warmup_time': warmup_time,
        'inference_time': inference_time,
    })

    print(f"[保存] 中间结果已保存到: {output_dir}")
    print(f"[计时] 模型加载: {model_load_time:.2f}s, 热身: {warmup_time:.2f}s, 纯推理: {inference_time:.2f}s")

    result_dir = f"{TASKS_ROOT}/{task_id}/result/{output_step}"
    os.makedirs(result_dir, exist_ok=True)
    report_path = os.path.join(result_dir, 'edge_report.txt')

    report_sections = [
        ('配置信息', [
            ('模型类型', model_type),
            ('批次大小', batch_size),
        ]),
        ('推理结果', [
            ('总样本数', total_samples),
            ('整体准确率', f"{accuracy:.2f}%"),
            ('吞吐量', f"{throughput:.1f} samples/s"),
        ]),
        ('耗时信息', [
            ('数据加载', f"{data_load_time:.2f}s"),
            ('推理', f"{inference_time:.2f}s"),
            ('模型加载+热身', f"{model_load_time + warmup_time:.2f}s"),
        ]),
    ]
    if confidence_threshold is not None:
        report_sections.insert(2, ('置信度分析', [
            ('高置信度样本数', high_conf_samples),
            ('高置信度准确率', f"{high_conf_accuracy:.2f}%"),
            ('低置信度样本数', low_conf_samples),
            ('低置信度准确率', f"{low_conf_accuracy:.2f}%"),
        ]))
    _generate_report(report_path, '边侧推理报告', report_sections)

    print(f"[报告] 推理报告已保存到: {report_path}")
    print(f"[完成] 边侧推理完成")

    return {
        'status': 'success',
        'total_samples': total_samples,
        'accuracy': accuracy / 100.0,
        'high_conf_samples': int(high_conf_samples),
        'high_conf_accuracy': high_conf_accuracy / 100.0,
        'low_conf_samples': int(low_conf_samples),
        'low_conf_accuracy': low_conf_accuracy / 100.0,
        'low_conf_ratio': low_conf_samples / total_samples * 100 if total_samples > 0 else 0,
        'avg_confidence': dc.get('avg_confidence', 0.98),
    }


# ================================================================
# 云侧直接推理模拟
# ================================================================
def simulate_cloud_direct_infer(config, task_id, step_prefix=""):
    """模拟纯云推理日志输出"""
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    batch_size = config.get('batch_size', 128)
    model_type = config.get('model_type', 'unknown')

    total_samples = dc.get('total_samples', 100000)
    inference_time = dc.get('inference_time', 26.29)
    throughput = dc.get('throughput', total_samples / inference_time if inference_time > 0 else 0)
    accuracy = dc.get('accuracy', 97.68)
    model_load_time = dc.get('model_load_time', 3.60)
    warmup_time = dc.get('warmup_time', 0.62)
    data_load_time = dc.get('data_load_time', 0.5)

    output_step = f"{step_prefix}cloud_direct_infer"

    print(f"\n{'='*60}")
    print(f"[云侧] 端→云 直接推理")
    print(f"{'='*60}")

    print(f"[加载] 从 device_load 加载全部数据...")
    _maybe_sleep(data_load_time, realtime)
    print(f"[加载] 成功加载 {total_samples} 个样本")
    print(f"[计时] 数据加载耗时: {data_load_time:.2f}s")

    print(f"[模型] 正在加载云侧模型...")
    _maybe_sleep(model_load_time, realtime)
    print(f"[模型] 成功加载模型权重")

    data_shape = dc.get('data_shape', f"({total_samples}, 128)")
    data_dtype = dc.get('data_dtype', 'complex64')
    print(f"[云侧推理] 数据格式: shape={data_shape}, dtype={data_dtype}")

    num_batches = (total_samples + batch_size - 1) // batch_size
    print(f"[云侧推理] 处理 {total_samples} 个样本, {num_batches} 批次")

    _maybe_sleep(warmup_time, realtime)
    print(f"[云侧推理] CUDA 热身完成 ({warmup_time:.2f}s)")

    _simulate_progress("云侧推理", total_samples, inference_time, batch_size, realtime)

    print(f"[云侧推理] 完成! 纯推理耗时: {inference_time:.2f}s, 吞吐量: {throughput:.1f} samples/s")
    print(f"[结果] 云侧直接推理准确率: {accuracy:.2f}%")
    print(f"[计时] 模型加载: {model_load_time:.2f}s, 热身: {warmup_time:.2f}s, 纯推理: {inference_time:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)

    save_timing(output_dir, {
        'data_load_time': data_load_time,
        'transfer_time': 0,
        'model_load_time': model_load_time,
        'warmup_time': warmup_time,
        'inference_time': inference_time,
    })

    result_dir = f"{TASKS_ROOT}/{task_id}/result/{output_step}"
    os.makedirs(result_dir, exist_ok=True)
    _generate_report(os.path.join(result_dir, 'cloud_report.txt'), '云侧直接推理报告', [
        ('配置信息', [
            ('模型类型', model_type),
            ('批次大小', batch_size),
        ]),
        ('推理结果', [
            ('总样本数', total_samples),
            ('整体准确率', f"{accuracy:.2f}%"),
            ('吞吐量', f"{throughput:.1f} samples/s"),
        ]),
        ('耗时信息', [
            ('数据加载', f"{data_load_time:.2f}s"),
            ('推理', f"{inference_time:.2f}s"),
            ('模型加载+热身', f"{model_load_time + warmup_time:.2f}s"),
        ]),
    ])

    print(f"[完成] 云侧直接推理完成")

    return {
        'status': 'success',
        'total_samples': total_samples,
        'accuracy': accuracy / 100.0,
        'avg_confidence': dc.get('avg_confidence', 0.98),
    }


# ================================================================
# 云侧协同推理模拟
# ================================================================
def simulate_cloud_infer(config, task_id, step_prefix=""):
    """模拟云边协同推理（云侧部分）日志输出"""
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)
    batch_size = config.get('batch_size', 128)
    model_type = config.get('model_type', 'unknown')

    low_conf_samples = dc.get('low_conf_samples', 11)
    inference_time = dc.get('inference_time', 0.06)
    throughput = dc.get('throughput', low_conf_samples / inference_time if inference_time > 0 else 0)
    cloud_accuracy = dc.get('cloud_accuracy', 72.73)
    model_load_time = dc.get('model_load_time', 2.53)
    warmup_time = dc.get('warmup_time', 0.18)
    data_load_time = dc.get('data_load_time', 0.1)

    agree_count = dc.get('agree_count', 3)
    agree_ratio = dc.get('agree_ratio', 27.3)
    corrected_count = dc.get('corrected_count', 7)
    edge_accuracy = dc.get('edge_accuracy', 97.28)
    final_accuracy = dc.get('final_accuracy', 97.28)
    accuracy_improvement = dc.get('accuracy_improvement', 0.01)

    output_step = f"{step_prefix}cloud_infer"

    print(f"\n{'='*60}")
    print(f"[云侧] 端→边→云 协同推理（云侧部分）")
    print(f"{'='*60}")

    print(f"[加载] 从 edge_infer 加载低置信度样本...")
    _maybe_sleep(data_load_time, realtime)
    print(f"[加载] 成功加载 {low_conf_samples} 个低置信度样本")
    print(f"[计时] 数据加载耗时: {data_load_time:.2f}s")

    print(f"[模型] 正在加载云侧模型...")
    _maybe_sleep(model_load_time, realtime)
    print(f"[模型] 成功加载模型权重")

    data_shape = dc.get('data_shape', f"({low_conf_samples}, 128)")
    data_dtype = dc.get('data_dtype', 'complex64')
    print(f"[云侧推理] 数据格式: shape={data_shape}, dtype={data_dtype}")

    num_batches = max((low_conf_samples + batch_size - 1) // batch_size, 1)
    print(f"[云侧推理] 处理 {low_conf_samples} 个样本, {num_batches} 批次")

    _maybe_sleep(warmup_time, realtime)
    print(f"[云侧推理] CUDA 热身完成 ({warmup_time:.2f}s)")

    if inference_time > 0.5:
        _simulate_progress("云侧推理", low_conf_samples, inference_time, batch_size, realtime)
    else:
        _maybe_sleep(inference_time, realtime)
        print(f"[云侧推理] 进度: 100.0% ({low_conf_samples}/{low_conf_samples}), 耗时: {inference_time:.2f}s")

    print(f"[云侧推理] 完成! 纯推理耗时: {inference_time:.2f}s, 吞吐量: {throughput:.1f} samples/s")
    print(f"[结果] 云侧准确率: {cloud_accuracy:.2f}%")
    print(f"[计时] 模型加载: {model_load_time:.2f}s, 热身: {warmup_time:.2f}s, 纯推理: {inference_time:.2f}s")

    print(f"[对比] 边云预测一致: {agree_count} ({agree_ratio:.1f}%)")
    print(f"[修正] 云侧修正样本数: {corrected_count}")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)

    save_timing(output_dir, {
        'data_load_time': data_load_time,
        'transfer_time': 0,
        'model_load_time': model_load_time,
        'warmup_time': warmup_time,
        'inference_time': inference_time,
    })

    print(f"[最终] 边侧单独准确率: {edge_accuracy:.2f}%")
    print(f"[最终] 边云协同准确率: {final_accuracy:.2f}%")
    print(f"[最终] 准确率提升: {accuracy_improvement:+.2f}%")

    result_dir = f"{TASKS_ROOT}/{task_id}/result/{output_step}"
    os.makedirs(result_dir, exist_ok=True)

    _generate_report(os.path.join(result_dir, 'cloud_report.txt'), '云侧推理报告', [
        ('配置信息', [('模型类型', model_type)]),
        ('推理结果', [
            ('低置信度样本数', low_conf_samples),
            ('云侧准确率', f"{cloud_accuracy:.2f}%"),
        ]),
        ('边云对比', [
            ('预测一致样本数', f"{agree_count} ({agree_ratio:.1f}%)"),
            ('云侧修正样本数', corrected_count),
        ]),
    ])

    _generate_report(os.path.join(result_dir, 'final_report.txt'), '边云协同推理最终报告', [
        ('准确率对比', [
            ('边侧单独准确率', f"{edge_accuracy:.2f}%"),
            ('边云协同准确率', f"{final_accuracy:.2f}%"),
            ('准确率提升', f"{accuracy_improvement:+.2f}%"),
        ]),
    ])

    print(f"[报告] 最终报告已保存")
    print(f"[完成] 云侧推理完成")

    return {
        'status': 'success',
        'cloud_samples': low_conf_samples,
        'cloud_accuracy': cloud_accuracy / 100.0,
        'agree_ratio': float(agree_ratio),
        'num_corrected': int(corrected_count),
    }


# ================================================================
# 端侧数据加载模拟
# ================================================================
def simulate_device_load(config, task_id, step_prefix=""):
    """模拟端侧数据加载日志输出，只读取文件大小不真正加载数据"""
    dc = config['display_config']
    realtime = dc.get('simulate_realtime', False)

    data_path = config.get('data_path', '')
    dataset_type = config.get('dataset_type', 'unknown')
    batch_size = config.get('batch_size', 128)

    total_samples = dc.get('total_samples', 100000)
    data_load_time = dc.get('data_load_time', 5.0)
    preprocess_time = dc.get('preprocess_time', 0.01)
    save_time = dc.get('save_time', 2.0)
    data_shape = dc.get('data_shape', f"({total_samples}, 2, 128)")
    data_dtype = dc.get('data_dtype', 'float32')
    num_files = dc.get('num_files', None)

    output_step = f"{step_prefix}device_load"

    print(f"\n{'='*60}")
    print(f"[端侧] 开始执行数据加载任务")
    print(f"{'='*60}")
    print(f"[配置] 数据路径: {data_path}")
    print(f"[配置] 数据集类型: {dataset_type}")
    print(f"[配置] 批次大小: {batch_size}")

    data_size_mb = _get_data_size_mb(data_path)
    if data_size_mb > 0:
        print(f"[信息] 数据文件大小: {data_size_mb:.1f} MB")

    if num_files is not None and num_files > 1:
        print(f"[加载] 发现 {num_files} 个文件（目录模式）")
        _maybe_sleep(data_load_time, realtime)
        print(f"[加载] 已加载 {num_files}/{num_files} 个文件")
    else:
        print(f"[加载] 正在加载数据文件...")
        _maybe_sleep(data_load_time, realtime)

    print(f"[加载] 成功加载 {total_samples} 个样本")
    print(f"[加载] 数据形状: X={data_shape}, dtype={data_dtype}")
    print(f"[计时] 数据加载耗时: {data_load_time:.2f}s")

    _maybe_sleep(preprocess_time, realtime)
    print(f"[计时] 数据预处理耗时: {preprocess_time:.2f}s")

    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_step}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data_batch.pkl')

    _maybe_sleep(save_time, realtime)
    print(f"[保存] 数据已保存到: {output_path}")
    print(f"[计时] 数据保存耗时: {save_time:.2f}s")

    save_timing(output_dir, {
        'data_load_time': data_load_time,
        'preprocess_time': preprocess_time,
        'data_save_time': save_time,
    })

    num_batches = (total_samples + batch_size - 1) // batch_size

    print(f"[完成] 端侧数据加载完成")
    print(f"[统计] 样本数: {total_samples}")
    print(f"[统计] 批次数: {num_batches}")

    return {
        'status': 'success',
        'num_samples': total_samples,
        'num_batches': num_batches,
        'dataset_type': dataset_type,
        'output_path': output_path,
    }

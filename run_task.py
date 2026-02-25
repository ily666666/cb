"""
统一任务执行入口

用法:
  # 流水线模式（多步骤串联）
  python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test

  # 单步骤模式（分布式部署时，在各机器上独立执行单个步骤）
  python run_task.py --step cloud_kd --task_id 004_train_link11
  python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 1

  # 最后一步加 --summary 触发全局耗时汇总 + 写入报告
  python run_task.py --step cloud_infer --task_id 001_COLLAB_link11_test --summary
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(__file__))

from callback.registry import execute_task
# 导入 callback 模块以触发 @register_task 装饰器注册
from callback import device_callback, edge_callback, cloud_callback, train_callback
from datetime import datetime
from config_refactor import PIPELINE_MODES, SUPPORTED_TASKS


def collect_all_timings(task_id):
    """
    扫描 task 下所有步骤的 timing.json，汇总全局耗时

    Returns:
        dict: {
            'steps': { step_name: {transfer_time, model_load_time, warmup_time, inference_time} },
            'total_inference': float,
            'total_overhead': float,   # model_load + warmup
            'total_transfer': float,
        }
    """
    output_root = f"./tasks/{task_id}/output"
    steps = {}
    total_inference = 0.0
    total_overhead = 0.0
    total_transfer = 0.0

    if not os.path.exists(output_root):
        return {'steps': steps, 'total_inference': 0, 'total_overhead': 0, 'total_transfer': 0}

    for step_dir in sorted(os.listdir(output_root)):
        timing_path = os.path.join(output_root, step_dir, 'timing.json')
        if os.path.exists(timing_path):
            try:
                with open(timing_path, 'r', encoding='utf-8') as f:
                    timing = json.load(f)
                t_trans = timing.get('transfer_time', 0)
                t_load = timing.get('model_load_time', 0)
                t_warm = timing.get('warmup_time', 0)
                t_inf = timing.get('inference_time', 0)
                steps[step_dir] = {
                    'transfer_time': t_trans,
                    'model_load_time': t_load,
                    'warmup_time': t_warm,
                    'inference_time': t_inf,
                }
                total_inference += t_inf
                total_overhead += t_load + t_warm
                total_transfer += t_trans
            except Exception:
                pass

    return {
        'steps': steps,
        'total_inference': total_inference,
        'total_overhead': total_overhead,
        'total_transfer': total_transfer,
    }


def format_timing_summary(timing_data, total_time=None):
    """
    将汇总的 timing 数据格式化为可读文本（同时用于打印和写文件）

    Args:
        timing_data: collect_all_timings 的返回值
        total_time: 本次运行的 wall-clock 总耗时（可选）

    Returns:
        str: 格式化的文本
    """
    lines = []
    steps = timing_data['steps']
    total_inf = timing_data['total_inference']
    total_oh = timing_data['total_overhead']
    total_trans = timing_data['total_transfer']

    if not steps:
        return ""

    lines.append("各步骤耗时明细:")
    for step_name, t in steps.items():
        overhead = t['model_load_time'] + t['warmup_time']
        parts = [f"纯推理: {t['inference_time']:.2f}s", f"加载+热身: {overhead:.2f}s"]
        if t['transfer_time'] > 0:
            parts.append(f"传输: {t['transfer_time']:.2f}s")
        lines.append(f"  {step_name}: {' | '.join(parts)}")

    lines.append("")
    lines.append(f"纯推理耗时合计: {total_inf:.2f}s")
    lines.append(f"开销合计(加载+热身): {total_oh:.2f}s")
    if total_trans > 0:
        lines.append(f"网络传输耗时合计: {total_trans:.2f}s")
    lines.append(f"推理+开销+传输: {total_inf + total_oh + total_trans:.2f}s")
    lines.append(f"推理+传输: {total_inf + total_trans:.2f}s")

    if total_time is not None:
        net_time = total_time - total_oh
        lines.append(f"本次运行总耗时: {total_time:.2f}s")
        lines.append(f"去除开销后: {net_time:.2f}s  (总耗时 {total_time:.2f}s - 加载热身 {total_oh:.2f}s)")

    return "\n".join(lines)


def write_timing_report(task_id, timing_data, total_time=None, pipeline=None):
    """
    将耗时汇总写到 result/timing_summary.txt，并追加到已有的 final_report / cloud_report 等末尾
    """
    summary_text = format_timing_summary(timing_data, total_time)
    if not summary_text:
        return

    # 1. 写独立的 timing_summary.txt
    result_root = f"./tasks/{task_id}/result"
    os.makedirs(result_root, exist_ok=True)

    report_path = os.path.join(result_root, 'timing_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("耗时统计汇总\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if pipeline:
            f.write(f"流水线: {' -> '.join(pipeline)}\n")
        f.write("\n")
        f.write(summary_text)
        f.write("\n")

    # 2. 追加到各步骤的已有报告文件末尾
    report_root = f"./tasks/{task_id}/result"
    if os.path.exists(report_root):
        for dirpath, dirnames, filenames in os.walk(report_root):
            for fn in filenames:
                if fn.endswith('_report.txt') or fn == 'final_report.txt':
                    fpath = os.path.join(dirpath, fn)
                    # 避免重复追加：检查是否已有耗时统计
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if '耗时统计汇总' in content:
                            continue
                    except Exception:
                        continue

                    with open(fpath, 'a', encoding='utf-8') as f:
                        f.write("\n" + "=" * 60 + "\n")
                        f.write("耗时统计汇总\n")
                        f.write("=" * 60 + "\n")
                        f.write(summary_text)
                        f.write("\n")


def run_pipeline(task_id, pipeline, extra_kwargs=None, show_summary=True):
    """执行任务流水线

    Args:
        task_id: 任务ID
        pipeline: 步骤列表
        extra_kwargs: 额外参数（如 edge_id），传递给 callback
        show_summary: 是否显示全局耗时汇总并写入报告文件
    """
    results = {}
    extra_kwargs = extra_kwargs or {}

    print(f"\n{'='*60}")
    print(f"  任务ID:  {task_id}")
    print(f"  流水线:  {' -> '.join(pipeline)}")
    if extra_kwargs:
        print(f"  额外参数: {extra_kwargs}")
    print(f"{'='*60}")

    pipeline_start = time.time()
    step_times = {}

    for step_idx, step in enumerate(pipeline, 1):
        print(f"\n[步骤 {step_idx}/{len(pipeline)}] {step}")
        print("-" * 40)

        step_start = time.time()
        try:
            result = execute_task(f"{step}_callback", task_id, **extra_kwargs)
            results[step] = result
            status = result.get('status', 'unknown')

            if status == 'error':
                print(f"[失败] {result.get('message')}")
                step_times[step] = time.time() - step_start
                break
            else:
                # 打印关键指标
                info = []
                if 'num_samples' in result:
                    info.append(f"样本数: {result['num_samples']}")
                if 'accuracy' in result:
                    info.append(f"准确率: {result['accuracy']*100:.2f}%")
                if 'low_conf_samples' in result:
                    info.append(f"低置信度: {result['low_conf_samples']}")
                if 'cloud_samples' in result:
                    info.append(f"云侧处理: {result['cloud_samples']}")
                print(f"[{status}] {', '.join(info)}" if info else f"[{status}]")

        except Exception as e:
            print(f"[异常] {e}")
            traceback.print_exc()
            results[step] = {'status': 'exception', 'error': str(e)}
            step_times[step] = time.time() - step_start
            break

        step_times[step] = time.time() - step_start

    total_time = time.time() - pipeline_start

    # 摘要
    print(f"\n{'='*60}")
    print("执行摘要")
    print(f"{'='*60}")

    # 打印各步骤执行结果
    for step, r in results.items():
        elapsed = step_times.get(step, 0)
        print(f"  {step}: {r.get('status')}  ({elapsed:.2f}s)")

    print(f"\n  本次运行耗时: {total_time:.2f}s")

    if show_summary:
        # 收集全局耗时统计（扫描该 task 下所有已存在的 timing.json）
        timing_data = collect_all_timings(task_id)

        if timing_data['steps']:
            print(f"\n{'-'*40}")
            print("  全局耗时汇总（含历史步骤）")
            print(f"{'-'*40}")
            summary_text = format_timing_summary(timing_data, total_time)
            for line in summary_text.split('\n'):
                print(f"  {line}")

        # 写入报告文件
        write_timing_report(task_id, timing_data, total_time, pipeline)

    print(f"\n  结果目录: ./tasks/{task_id}/result/")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="云边端协同系统 - 任务执行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 流水线模式
  python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test
  python run_task.py --mode full_train --task_id 004_train_link11

  # 单步骤模式（分布式部署）
  python run_task.py --step cloud_kd --task_id 004_train_link11
  python run_task.py --step federated_server --task_id 004_train_link11
  python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 1

  # 最后一步加 --summary 显示全局耗时汇总并写入报告
  python run_task.py --step cloud_infer --task_id 001 --summary
        """
    )

    # --mode 和 --step 二选一
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=list(PIPELINE_MODES.keys()),
                       help="流水线模式（多步骤串联执行）")
    group.add_argument("--step", choices=SUPPORTED_TASKS,
                       help="单步骤模式（独立执行一个步骤，用于分布式部署）")

    parser.add_argument("--task_id", required=True,
                        help="任务ID（对应 tasks/ 下的目录名）")
    parser.add_argument("--edge_id", type=int, default=None,
                        help="边侧ID（仅 --step federated_edge 时使用，从 1 开始）")
    parser.add_argument("--summary", action="store_true", default=False,
                        help="显示全局耗时汇总并写入报告（--mode 自动开启；--step 最后一步时手动加）")

    args = parser.parse_args()

    # 检查任务目录
    task_dir = f"./tasks/{args.task_id}"
    if not os.path.exists(task_dir):
        print(f"[错误] 任务目录不存在: {task_dir}")
        return

    # 构建额外参数
    extra_kwargs = {}
    if args.edge_id is not None:
        extra_kwargs['edge_id'] = args.edge_id

    if args.mode:
        pipeline = PIPELINE_MODES[args.mode]
        show_summary = True  # --mode 始终显示全局汇总
    else:
        # --step 模式：单步骤作为长度为 1 的流水线
        pipeline = [args.step]
        show_summary = args.summary  # --step 需要手动加 --summary

    run_pipeline(args.task_id, pipeline, extra_kwargs=extra_kwargs, show_summary=show_summary)


if __name__ == "__main__":
    main()

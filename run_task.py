"""
统一任务执行入口

用法:
  # 流水线模式（多步骤串联）
  python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test

  # 单步骤模式（分布式部署时，在各机器上独立执行单个步骤）
  python run_task.py --step edge_kd --task_id 004_train_link11
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

import re

from callback.registry import execute_task
# 导入 callback 模块以触发 @register_task 装饰器注册
from callback import device_callback, edge_callback, cloud_callback, train_callback
from datetime import datetime
from config_refactor import PIPELINE_MODES, SUPPORTED_TASKS, TASKS_ROOT


def parse_config_name(config_name):
    """
    从配置文件名解析回调函数名和 edge_id

    规则：
    - cloud_pretrain → ('cloud_pretrain', None)
    - edge_kd_1     → ('edge_kd', 1)
    - federated_cloud → ('federated_cloud', None)
    - federated_edge_2 → ('federated_edge', 2)
    - link11_cloud_pretrain → ('link11_cloud_pretrain', None)
    - link11_edge_kd_1 → ('link11_edge_kd', 1)
    - link11_federated_edge_2 → ('link11_federated_edge', 2)

    Returns:
        (callback_base, edge_id): 回调函数基础名和边侧ID
    """
    config_name = config_name.replace('.json', '')
    match = re.match(r'^(.+?)_(\d+)$', config_name)
    if match:
        base = match.group(1)
        num = int(match.group(2))
        # 支持原始和数据集前缀版本
        edge_kd_bases = ['edge_kd', 'link11_edge_kd', 'rml2016_edge_kd', 'radar_edge_kd', 'ratr_edge_kd']
        fed_edge_bases = ['federated_edge', 'link11_federated_edge', 'rml2016_federated_edge', 'radar_federated_edge', 'ratr_federated_edge']
        edge_local_train_bases = ['edge_local_train', 'link11_edge_local_train', 'rml2016_edge_local_train', 'radar_edge_local_train', 'ratr_edge_local_train']
        if base in edge_kd_bases + fed_edge_bases + edge_local_train_bases:
            return base, num
    return config_name, None


def collect_all_timings(task_id):
    """
    扫描 task 下所有步骤的 timing.json，汇总全局耗时

    支持两种 timing.json 格式：
    - 简化版: {"step_time": 9.8}
    - 详细版: {"data_load_time":..., "transfer_time":..., ...}
    """
    output_root = f"{TASKS_ROOT}/{task_id}/output"
    steps = {}
    total_time = 0.0

    if not os.path.exists(output_root):
        return {'steps': steps, 'total_time': 0}

    for step_dir in sorted(os.listdir(output_root)):
        timing_path = os.path.join(output_root, step_dir, 'timing.json')
        if os.path.exists(timing_path):
            try:
                with open(timing_path, 'r', encoding='utf-8') as f:
                    timing = json.load(f)
                if 'step_time' in timing:
                    st = timing['step_time']
                    steps[step_dir] = {'step_time': st}
                    total_time += st
                else:
                    t_data = timing.get('data_load_time', 0)
                    t_prep = timing.get('preprocess_time', 0)
                    t_save = timing.get('data_save_time', 0)
                    t_trans = timing.get('transfer_time', 0)
                    t_load = timing.get('model_load_time', 0)
                    t_warm = timing.get('warmup_time', 0)
                    t_inf = timing.get('inference_time', 0)
                    st = t_data + t_prep + t_save + t_trans + t_load + t_warm + t_inf
                    steps[step_dir] = {'step_time': st}
                    total_time += st
            except Exception:
                pass

    return {
        'steps': steps,
        'total_time': total_time,
    }


def format_timing_summary(timing_data, total_time=None):
    """
    将汇总的 timing 数据格式化为可读文本（同时用于打印和写文件）
    """
    lines = []
    steps = timing_data['steps']

    if not steps:
        return ""

    lines.append("各步骤耗时:")
    for step_name, t in steps.items():
        st = t.get('step_time', 0)
        is_load = 'device_load' in step_name or 'data_load' in step_name
        label = f"{st:.2f}s" if is_load else f"传输+推理 {st:.2f}s"
        lines.append(f"  {step_name}: {label}")

    lines.append("")
    lines.append(f"总耗时: {timing_data['total_time']:.2f}s")
    return "\n".join(lines)



def write_timing_report(task_id, timing_data, total_time=None, pipeline=None):
    """
    将全局耗时汇总写到 result/timing_summary.txt，并追加到 final_report.txt 末尾。
    各步骤自身的耗时由各 callback 直接写入各自的报告。
    """
    summary_text = format_timing_summary(timing_data, total_time)
    if not summary_text:
        return

    # 1. 写独立的 timing_summary.txt
    result_root = f"{TASKS_ROOT}/{task_id}/result"
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

    # 2. 追加全局耗时汇总到 final_report.txt
    report_root = f"{TASKS_ROOT}/{task_id}/result"
    if os.path.exists(report_root):
        for dirpath, dirnames, filenames in os.walk(report_root):
            for fn in filenames:
                if fn != 'final_report.txt':
                    continue
                fpath = os.path.join(dirpath, fn)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if '耗时' in content:
                        continue
                except Exception:
                    continue

                with open(fpath, 'a', encoding='utf-8') as f:
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("耗时统计汇总\n")
                    f.write("=" * 60 + "\n")
                    f.write(summary_text)
                    f.write("\n")


def _simulate_step(task_id, step, step_idx, total_steps):
    """FAST_MODE: 快速模拟一个步骤，输出仿真日志并写 timing.json"""
    import math, random

    is_train = any(k in step for k in ('kd', 'pretrain', 'federated', 'local_train'))
    is_load = 'device_load' in step or 'data_load' in step

    config_name = step
    for prefix in ('link11_', 'rml2016_', 'radar_', 'ratr_'):
        if step.startswith(prefix):
            config_name = step[len(prefix):]
            break
    output_folder = step
    output_dir = f"{TASKS_ROOT}/{task_id}/output/{output_folder}"
    os.makedirs(output_dir, exist_ok=True)

    # 读取 display_config（如果存在）
    dc = {}
    input_dir = f"{TASKS_ROOT}/{task_id}/input"
    if os.path.isdir(input_dir):
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.json'):
                continue
            if step in fname or config_name in fname:
                try:
                    with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
                        dc = json.load(f).get('display_config', {})
                    if dc:
                        break
                except Exception:
                    pass

    if is_train:
        import pickle
        epochs = 30
        total_batches = 2048
        target_acc = dc.get('accuracy', 99.0) / 100.0
        base_loss, base_acc = 1.8, max(0.3, target_acc - 0.45)
        train_loss_hist, train_acc_hist, test_acc_hist = [], [], []
        print(f"[{config_name}] 开始训练 (epochs={epochs})")
        for ep in range(1, epochs + 1):
            progress = ep / epochs
            lr = 0.001 * (1 + math.cos(math.pi * progress)) / 2
            # 指数衰减 loss，前快后慢
            decay = 1 - math.exp(-4 * progress)
            cur_loss = base_loss * (1 - decay * 0.92) + random.uniform(-0.03, 0.03)
            # 对数增长 acc，前快后慢
            cur_acc = base_acc + (target_acc - base_acc) * (math.log(1 + 9 * progress) / math.log(10)) + random.uniform(-0.008, 0.008)
            cur_loss = max(0.01, cur_loss)
            cur_acc = min(target_acc, max(base_acc, cur_acc))
            train_loss_hist.append(round(cur_loss, 4))
            train_acc_hist.append(round(cur_acc * 100, 2))
            test_acc_hist.append(round((cur_acc - random.uniform(0.005, 0.02)) * 100, 2))
            for pct in (50, 100):
                batch = int(total_batches * pct / 100)
                print(f"  [Epoch {ep}/{epochs}] batch {batch}/{total_batches} ({pct}%) | "
                      f"Loss: {cur_loss:.4f} | Acc: {cur_acc*100:.2f}%")
            print(f"  [Epoch {ep}/{epochs} 完成] LR: {lr:.6f} | Loss: {cur_loss:.4f} | Acc: {cur_acc*100:.2f}%")
            time.sleep(0.05)
        sim_time = random.uniform(30, 60)
        teacher_acc = dc.get('teacher_accuracy', 97.5)
        student_acc = dc.get('accuracy', 99.0)
        print(f"[{config_name}] 训练完成")
        print(f"[结果] 教师模型准确率: {teacher_acc}%")
        print(f"[结果] 学生模型准确率: {student_acc}%")
        timing = {"step_time": round(sim_time, 2)}
        kd_history = {
            "edge_1": {"train_loss": train_loss_hist, "train_acc": train_acc_hist, "test_acc": test_acc_hist},
        }
        with open(os.path.join(output_dir, 'kd_history.pkl'), 'wb') as f:
            pickle.dump(kd_history, f)
    elif is_load:
        sim_time = random.uniform(0.5, 2.0)
        print(f"[{config_name}] 数据加载完成")
        timing = {"step_time": round(sim_time, 2)}
        time.sleep(0.3)
    else:
        sim_time = random.uniform(5, 25)
        sim_acc = random.uniform(95, 99)
        print(f"[{config_name}] 推理中...")
        for pct in range(10, 101, 10):
            print(f"  进度: {pct}%")
            time.sleep(0.05)
        print(f"[{config_name}] 推理完成 | 准确率: {sim_acc:.2f}%")
        timing = {"step_time": round(sim_time, 2)}

    with open(os.path.join(output_dir, 'timing.json'), 'w') as f:
        json.dump(timing, f)

    return {'status': 'success'}


def run_pipeline(task_id, pipeline, extra_kwargs=None, show_summary=True):
    """执行任务流水线

    Args:
        task_id: 任务ID
        pipeline: 步骤列表
        extra_kwargs: 额外参数（如 edge_id），传递给 callback
        show_summary: 是否显示全局耗时汇总并写入报告文件
    """
    fast_mode = os.environ.get('FAST_MODE') == '1'
    results = {}
    extra_kwargs = extra_kwargs or {}

    print(f"\n{'='*60}")
    print(f"  任务ID:  {task_id}")
    print(f"  流水线:  {' -> '.join(pipeline)}")
    if fast_mode:
        print(f"  模式: 快速演示")
    if extra_kwargs:
        print(f"  额外参数: {extra_kwargs}")
    print(f"{'='*60}")

    pipeline_start = time.time()
    step_times = {}

    for step_idx, step in enumerate(pipeline, 1):
        print(f"\n[步骤 {step_idx}/{len(pipeline)}] {step}")
        print("-" * 40)

        step_start = time.time()

        if fast_mode:
            result = _simulate_step(task_id, step, step_idx, len(pipeline))
            results[step] = result
            step_times[step] = time.time() - step_start
            print(f"[success]")
            continue

        try:
            result = execute_task(f"{step}_callback", task_id, **extra_kwargs)
            results[step] = result
            status = result.get('status', 'unknown')

            if status == 'error':
                print(f"[失败] {result.get('message')}")
                step_times[step] = time.time() - step_start
                break
            else:
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

    # # 打印各步骤执行结果
    # for step, r in results.items():
    #     elapsed = step_times.get(step, 0)
    #     print(f"  {step}: {r.get('status')}  ({elapsed:.2f}s)")

    # print(f"\n  本次运行耗时: {total_time:.2f}s")

    if show_summary and not fast_mode:
        timing_data = collect_all_timings(task_id)

        if timing_data['steps']:
            print(f"\n{'-'*40}")
            print("  全局耗时汇总（含历史步骤）")
            print(f"{'-'*40}")
            summary_text = format_timing_summary(timing_data, total_time)
            for line in summary_text.split('\n'):
                print(f"  {line}")

        write_timing_report(task_id, timing_data, total_time, pipeline)

    print(f"\n  结果目录: {TASKS_ROOT}/{task_id}/result/")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="云边端协同系统 - 任务执行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 流水线模式（单机串联执行）
  python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test
  python run_task.py --mode full_train --task_id 004_train_link11

  # 配置文件驱动模式（推荐，分布式部署）
  python run_task.py --task_id 004_train_link11 --config cloud_pretrain
  python run_task.py --task_id 004_train_link11 --config edge_kd_1
  python run_task.py --task_id 004_train_link11 --config edge_kd_2
  python run_task.py --task_id 004_train_link11 --config federated_cloud
  python run_task.py --task_id 004_train_link11 --config federated_edge_1
  python run_task.py --task_id 004_train_link11 --config federated_edge_2

  # 单步骤模式（向后兼容）
  python run_task.py --step edge_kd --task_id 004_train_link11 --edge_id 1
  python run_task.py --step federated_server --task_id 004_train_link11

  # 最后一步加 --summary 显示全局耗时汇总并写入报告
  python run_task.py --step cloud_infer --task_id 001 --summary
        """
    )

    # --mode / --step / --config 三选一
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=list(PIPELINE_MODES.keys()),
                       help="流水线模式（多步骤串联执行）")
    group.add_argument("--step", choices=SUPPORTED_TASKS,
                       help="单步骤模式（独立执行一个步骤，用于分布式部署）")
    group.add_argument("--config", type=str, default=None,
                       help="配置文件驱动模式：指定 JSON 配置文件名（不含.json），"
                            "文件名决定调用的回调函数，边侧ID从文件名解析。"
                            "例如: cloud_pretrain, edge_kd_1, federated_cloud, federated_edge_2")

    parser.add_argument("--task_id", required=True,
                        help="任务ID（对应 tasks/ 下的目录名）")
    parser.add_argument("--edge_id", type=int, default=None,
                        help="边侧ID（--step edge_kd 或 federated_edge 时使用，从 1 开始）")
    parser.add_argument("--summary", action="store_true", default=False,
                        help="显示全局耗时汇总并写入报告（--mode 自动开启；--step 最后一步时手动加）")

    args = parser.parse_args()

    # 检查任务目录
    task_dir = f"{TASKS_ROOT}/{args.task_id}"
    if not os.path.exists(task_dir):
        print(f"[错误] 任务目录不存在: {task_dir}")
        return

    # 构建额外参数
    extra_kwargs = {}
    if args.edge_id is not None:
        extra_kwargs['edge_id'] = args.edge_id

    if args.mode:
        pipeline = PIPELINE_MODES[args.mode]
        show_summary = True
    elif args.config:
        config_name = args.config.replace('.json', '')
        callback_base, edge_id = parse_config_name(config_name)
        pipeline = [callback_base]
        extra_kwargs['config_name'] = config_name
        if edge_id is not None:
            extra_kwargs['edge_id'] = edge_id
        show_summary = args.summary

        # 验证配置文件存在
        config_file = f"{TASKS_ROOT}/{args.task_id}/input/{config_name}.json"
        if not os.path.exists(config_file):
            print(f"[错误] 配置文件不存在: {config_file}")
            return
    else:
        pipeline = [args.step]
        show_summary = args.summary

    run_pipeline(args.task_id, pipeline, extra_kwargs=extra_kwargs, show_summary=show_summary)


if __name__ == "__main__":
    main()

"""
训练链路回调
包含：预训练教师模型、知识蒸馏、联邦学习
全部通过文件系统交换数据，不再使用Socket/ZMQ
"""
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from callback.registry import register_task
from utils_refactor import (
    load_json, load_pickle, save_pickle, save_numpy,
    check_parameters, load_from_output, check_output_exists
)
from core.model_factory import create_model_by_type


# ================================================================
# 数据格式适配（与 edge_callback / cloud_callback 对应）
# ================================================================

def _prepare_iq_to_complex(X_data):
    """
    将 I/Q 双通道数据统一转为 complex 格式
    
    real 模型和 complex 模型都需要 complex 输入：
    - real 模型在 forward 内部自己拆 torch.real / torch.imag
    - complex 模型直接用 complex 运算

    处理逻辑：
    - (N, 2, L) 实数 I/Q → (N, L) complex64
    - (N, L) 复数         → 直接使用
    - (N, L) 实数         → 转为 complex64（虚部为 0）
    """
    if isinstance(X_data, np.ndarray):
        if np.iscomplexobj(X_data):
            return X_data.astype(np.complex64)
        else:
            if X_data.ndim == 3 and X_data.shape[1] == 2:
                return (X_data[:, 0, :] + 1j * X_data[:, 1, :]).astype(np.complex64)
            elif X_data.ndim == 2:
                return X_data.astype(np.complex64)
            return X_data.astype(np.complex64)
    return X_data


# ================================================================
# 通用训练辅助函数
# ================================================================

def _make_dataloader(X, y, batch_size, shuffle=True):
    """从 numpy 数组创建 DataLoader，I/Q 数据统一转为 complex"""
    if isinstance(X, np.ndarray):
        X = _prepare_iq_to_complex(X)
        X_tensor = torch.from_numpy(X).cfloat()
    else:
        X_tensor = X

    if isinstance(y, np.ndarray):
        y_tensor = torch.from_numpy(y).long()
    else:
        y_tensor = y

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(model, train_loader, optimizer, criterion, device,
                     epoch_info=None, log_every=100):
    """
    训练一个 epoch
    
    Args:
        epoch_info: (current_epoch, total_epochs) 用于显示进度，None 则静默
        log_every: 每隔多少个 batch 打印一次进度（默认 100）
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

        if epoch_info is not None and (batch_idx + 1) % log_every == 0:
            ep, total_ep = epoch_info
            running_acc = correct / total if total > 0 else 0
            running_loss = total_loss / total if total > 0 else 0
            print(f"    Epoch {ep}/{total_ep} | "
                  f"Batch {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:.0f}%) | "
                  f"Loss: {running_loss:.4f} Acc: {running_acc*100:.2f}%",
                  flush=True)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def _evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def _load_split_data(data_path, dataset_type):
    """
    加载预划分的数据

    支持格式：
    - dict 含 'train'/'val'/'test' 键
    - dict 含 'X_train'/'y_train' 等键
    """
    data = load_pickle(data_path)

    if isinstance(data, dict):
        if 'train' in data:
            train_val = data['train']
            # 格式 A：值为元组 (X, y) — prepare_data_splits.py 的输出格式
            if isinstance(train_val, (tuple, list)):
                val_data = data.get('val', data.get('test', (None, None)))
                return {
                    'train': (train_val[0], train_val[1]),
                    'val': (val_data[0], val_data[1]),
                    'test': (data['test'][0], data['test'][1]),
                }
            # 格式 B：值为字典 {'X': ..., 'y': ...}
            return {
                'train': (train_val['X'], train_val['y']),
                'val': (data.get('val', data.get('test', {})).get('X'), data.get('val', data.get('test', {})).get('y')),
                'test': (data['test']['X'], data['test']['y']),
            }
        elif 'X_train' in data:
            return {
                'train': (data['X_train'], data['y_train']),
                'val': (data.get('X_val', data['X_test']), data.get('y_val', data['y_test'])),
                'test': (data['X_test'], data['y_test']),
            }
        elif 'X' in data and 'y' in data:
            # 只有一个集合，按比例划分
            X, y = data['X'], data['y']
            n = len(X)
            n_train = int(n * 0.7)
            n_val = int(n * 0.85)
            return {
                'train': (X[:n_train], y[:n_train]),
                'val': (X[n_train:n_val], y[n_train:n_val]),
                'test': (X[n_val:], y[n_val:]),
            }

    raise ValueError(f"不支持的数据格式: {type(data)}, keys={data.keys() if isinstance(data, dict) else 'N/A'}")


def _generate_soft_labels_from_teacher(teacher, X_train, batch_size, temperature, device):
    """用教师模型对训练数据生成软标签（方案B: 在边侧本地生成）"""
    X_prepared = _prepare_iq_to_complex(X_train)
    X_tensor = torch.from_numpy(X_prepared).cfloat() if isinstance(X_prepared, np.ndarray) else X_prepared
    dummy_y = torch.zeros(len(X_tensor), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tensor, dummy_y), batch_size=batch_size, shuffle=False)

    total_batches = len(loader)
    all_soft = []
    teacher.eval()
    with torch.no_grad():
        for i, (batch_X, _) in enumerate(loader):
            batch_X = batch_X.to(device)
            logits = teacher(batch_X)
            soft = F.softmax(logits / temperature, dim=1)
            all_soft.append(soft.cpu().numpy())
            if (i + 1) % 500 == 0 or (i + 1) == total_batches:
                print(f"    软标签进度: {i+1}/{total_batches} ({100*(i+1)/total_batches:.1f}%)")

    return np.concatenate(all_soft, axis=0)


# ================================================================
# 1. 预训练教师模型
# ================================================================

@register_task
def cloud_pretrain_callback(task_id):
    """
    云侧预训练教师模型

    配置: input/cloud_pretrain.json
    输出:
      - output/cloud_pretrain/teacher_model.pth   (模型权重)
      - output/cloud_pretrain/train_history.pkl    (训练历史)
      - result/cloud_pretrain/pretrain_report.txt  (训练报告)
    """
    print(f"\n{'='*60}")
    print(f"[云侧] 预训练教师模型")
    print(f"{'='*60}")

    # 1. 读取配置
    config_path = f"./tasks/{task_id}/input/cloud_pretrain.json"
    param_list = ['data_path', 'dataset_type', 'model_type', 'num_classes', 'epochs']
    result, config = check_parameters(config_path, param_list)
    if 'error' in result:
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        return {'status': 'error', 'message': f"缺少参数: {', '.join(result['missing'])}"}

    data_path = config['data_path']
    dataset_type = config['dataset_type']
    model_type = config['model_type']
    num_classes = config['num_classes']
    epochs = config['epochs']
    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"[配置] 数据: {data_path}, 模型: {model_type}, epochs: {epochs}")

    # 2. 缓存检查
    output_dir = f"./tasks/{task_id}/output/cloud_pretrain"
    if check_output_exists(task_id, 'cloud_pretrain', 'teacher_model.pth'):
        print(f"[跳过] 教师模型已存在")
        return {'status': 'cached'}

    # 3. 加载数据
    print(f"[加载] 正在加载训练数据...")
    splits = _load_split_data(data_path, dataset_type)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    train_loader = _make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = _make_dataloader(X_test, y_test, batch_size, shuffle=False)
    print(f"[加载] 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 4. 创建模型
    model = create_model_by_type(model_type, num_classes, dataset_type)
    model.to(device)
    print(f"[模型] 创建教师模型: {model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. 训练循环
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, 'teacher_best.pth')

    num_batches = len(train_loader)
    print(f"\n[训练] 开始训练... (每轮 {num_batches} 个batch, batch_size={batch_size})")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_info=(epoch, epochs))
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            improved = " ★ 已保存"
            torch.save({
                'model_state_dict': best_model_state,
                'model_type': model_type,
                'num_classes': num_classes,
                'dataset_type': dataset_type,
                'best_val_acc': best_val_acc,
                'epoch': epoch,
            }, best_ckpt_path)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (epochs - epoch)
        eta_min, eta_sec = divmod(int(eta), 60)
        print(f"  Epoch {epoch}/{epochs} ({epoch_time:.1f}s, ETA {eta_min}m{eta_sec:02d}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"Best: {best_val_acc*100:.2f}% (ep{best_epoch}){improved}",
              flush=True)

    total_time = time.time() - start_time
    print(f"[训练] 完成! 总耗时: {total_time:.1f}s, 最佳验证准确率: {best_val_acc*100:.2f}% (epoch {best_epoch})")

    # 6. 测试最佳模型
    model.load_state_dict(best_model_state)
    test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
    print(f"[测试] 测试准确率: {test_acc*100:.2f}%")

    # 7. 保存最终模型（覆盖 best checkpoint → 正式文件名）
    final_save = {
        'model_state_dict': best_model_state,
        'model_type': model_type,
        'num_classes': num_classes,
        'dataset_type': dataset_type,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'epoch': best_epoch,
    }
    torch.save(final_save, os.path.join(output_dir, 'teacher_model.pth'))
    if os.path.exists(best_ckpt_path):
        os.remove(best_ckpt_path)
    save_pickle(os.path.join(output_dir, 'train_history.pkl'), history)

    # 8. 报告
    result_dir = f"./tasks/{task_id}/result/cloud_pretrain"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'pretrain_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n教师模型预训练报告\n" + "=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"数据集: {dataset_type}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"训练耗时: {total_time:.1f}s\n\n")
        f.write(f"最佳验证准确率: {best_val_acc*100:.2f}%\n")
        f.write(f"测试准确率: {test_acc*100:.2f}%\n")

    print(f"[完成] 教师模型已保存到 {output_dir}")

    return {
        'status': 'success',
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'train_time': total_time,
    }


# ================================================================
# 2. 知识蒸馏（分别蒸馏：各边用自己的数据 + 教师模型）
# ================================================================

def _kd_train_one_student(student, X_train, y_train, X_test, y_test, soft_labels,
                          alpha, temperature, epochs, batch_size, learning_rate,
                          device, label=""):
    """蒸馏训练一个学生模型

    返回: {'best_state': state_dict, 'best_acc': float, 'best_epoch': int, 'train_time': float}
    """
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')

    X_prepared = _prepare_iq_to_complex(X_train)
    X_tensor = torch.from_numpy(X_prepared).cfloat() if isinstance(X_prepared, np.ndarray) else X_prepared
    y_tensor = torch.from_numpy(y_train).long() if isinstance(y_train, np.ndarray) else y_train
    soft_tensor = torch.from_numpy(soft_labels).float()

    train_dataset = TensorDataset(X_tensor, y_tensor, soft_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = _make_dataloader(X_test, y_test, batch_size, shuffle=False)

    total_batches = len(train_loader)
    kd_log_interval = 100
    prefix = f"[{label}] " if label else ""

    print(f"\n{prefix}开始蒸馏 (alpha={alpha}, T={temperature}, 每轮 {total_batches} batch, 训练集 {len(X_train)} 样本)...")
    best_test_acc = 0.0
    best_state = None
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (batch_X, batch_y, batch_soft) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_soft = batch_soft.to(device)

            optimizer.zero_grad()
            student_logits = student(batch_X)

            ce_loss = ce_criterion(student_logits, batch_y)
            student_soft = F.log_softmax(student_logits / temperature, dim=1)
            kd_loss = kl_criterion(student_soft, batch_soft) * (temperature ** 2)

            loss = (1 - alpha) * ce_loss + alpha * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(student_logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            if (batch_idx + 1) % kd_log_interval == 0:
                running_acc = correct / total if total > 0 else 0
                running_loss = total_loss / total if total > 0 else 0
                print(f"    {prefix}Epoch {epoch}/{epochs} | "
                      f"Batch {batch_idx+1}/{total_batches} ({100*(batch_idx+1)/total_batches:.0f}%) | "
                      f"Loss: {running_loss:.4f} Acc: {running_acc*100:.2f}%",
                      flush=True)

        scheduler.step()
        train_acc = correct / total
        epoch_time = time.time() - epoch_start

        _, test_acc = _evaluate(student, test_loader, ce_criterion, device)

        improved = ""
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            improved = " ★"

        elapsed = time.time() - start_time
        eta = elapsed / epoch * (epochs - epoch)
        eta_min, eta_sec = divmod(int(eta), 60)
        print(f"  {prefix}Epoch {epoch}/{epochs} ({epoch_time:.1f}s, ETA {eta_min}m{eta_sec:02d}s) | "
              f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | "
              f"Best: {best_test_acc*100:.2f}% (ep{best_epoch}){improved}",
              flush=True)

    total_time = time.time() - start_time
    print(f"{prefix}完成! 耗时: {total_time:.1f}s, 最佳: {best_test_acc*100:.2f}% (epoch {best_epoch})")

    return {
        'best_state': best_state,
        'best_acc': best_test_acc,
        'best_epoch': best_epoch,
        'train_time': total_time,
    }


@register_task
def edge_kd_callback(task_id, edge_id=None, **kwargs):
    """
    知识蒸馏 — 各边分别蒸馏

    加载教师模型，每个边用自己的本地数据生成软标签并训练学生模型。

    单机模式: 不传 edge_id → 处理所有边，生成所有 student_edge_*.pth + student_model.pth
    多机模式: --edge_id N → 只处理第 N 个边，生成 student_edge_N.pth

    配置: input/edge_kd.json
    运行:
      单机: python run_task.py --mode full_train --task_id xxx
      多机: python run_task.py --step edge_kd --task_id xxx --edge_id 1
    """
    config_path = f"./tasks/{task_id}/input/edge_kd.json"
    param_list = ['student_model_type', 'num_classes', 'dataset_type',
                  'edge_data_paths', 'epochs']
    result, config = check_parameters(config_path, param_list)
    if 'error' in result:
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        return {'status': 'error', 'message': f"缺少参数: {', '.join(result['missing'])}"}

    student_model_type = config['student_model_type']
    num_classes = config['num_classes']
    dataset_type = config['dataset_type']
    edge_data_paths = config['edge_data_paths']
    epochs = config['epochs']
    alpha = config.get('kd_alpha', 0.7)
    temperature = config.get('temperature', 4.0)
    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 0.001)
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    num_edges = len(edge_data_paths)

    output_dir = f"./tasks/{task_id}/output/edge_kd"

    # 确定要处理的边
    if edge_id is not None:
        if edge_id < 1 or edge_id > num_edges:
            return {'status': 'error',
                    'message': f'edge_id={edge_id} 超出范围, 共 {num_edges} 个边 (1~{num_edges})'}
        edges_to_process = [(edge_id - 1, edge_data_paths[edge_id - 1])]
        single_edge_mode = True
    else:
        if check_output_exists(task_id, 'edge_kd', 'student_model.pth'):
            print(f"[跳过] 学生模型已存在")
            return {'status': 'cached'}
        edges_to_process = list(enumerate(edge_data_paths))
        single_edge_mode = False

    mode_label = f"边 {edge_id}" if single_edge_mode else f"全部 {num_edges} 个边"
    print(f"\n{'='*60}")
    print(f"[知识蒸馏] 分别蒸馏 — {mode_label}")
    print(f"{'='*60}")

    # 单边模式下的缓存检查
    if single_edge_mode:
        target_file = os.path.join(output_dir, f'student_edge_{edge_id}.pth')
        if os.path.exists(target_file):
            print(f"[跳过] student_edge_{edge_id}.pth 已存在")
            return {'status': 'cached'}

    # 加载教师模型
    teacher_model_type = config.get('teacher_model_type')
    teacher_model_path = config.get('teacher_model_path',
        f"./tasks/{task_id}/output/cloud_pretrain/teacher_model.pth")

    if not os.path.exists(teacher_model_path):
        return {'status': 'error', 'message': f"教师模型不存在: {teacher_model_path}"}

    teacher = create_model_by_type(teacher_model_type, num_classes, dataset_type)
    checkpoint = torch.load(teacher_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
        if not teacher_model_type and 'model_type' in checkpoint:
            teacher_model_type = checkpoint['model_type']
    else:
        teacher.load_state_dict(checkpoint)
    teacher.to(device)
    teacher.eval()
    print(f"[教师] 加载完成: {teacher_model_type} ← {teacher_model_path}")
    print(f"[配置] 每边 {epochs} epoch, alpha={alpha}, T={temperature}")

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for edge_idx, edge_path in edges_to_process:
        eid = edge_idx + 1
        print(f"\n{'='*60}")
        print(f"[边侧 {eid}/{num_edges}] 蒸馏开始 (数据: {edge_path})")
        print(f"{'='*60}")

        splits = _load_split_data(edge_path, dataset_type)
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        print(f"[边侧 {eid}] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

        print(f"[边侧 {eid}] 生成软标签...")
        soft_labels = _generate_soft_labels_from_teacher(
            teacher, X_train, batch_size, temperature, device)
        print(f"[边侧 {eid}] 软标签: {soft_labels.shape}")

        student = create_model_by_type(student_model_type, num_classes, dataset_type)
        student.to(device)

        kd_result = _kd_train_one_student(
            student, X_train, y_train, X_test, y_test, soft_labels,
            alpha, temperature, epochs, batch_size, learning_rate,
            device, label=f"边{eid}")

        torch.save({
            'model_state_dict': kd_result['best_state'],
            'model_type': student_model_type,
            'num_classes': num_classes,
            'dataset_type': dataset_type,
            'best_test_acc': kd_result['best_acc'],
            'epoch': kd_result['best_epoch'],
            'edge_id': eid,
        }, os.path.join(output_dir, f'student_edge_{eid}.pth'))
        print(f"[边侧 {eid}] 模型已保存: student_edge_{eid}.pth")

        all_results.append((eid, kd_result))

    # 单边模式：只保存该边的模型，不生成 student_model.pth（等所有边完成后由 federated_train 聚合）
    if single_edge_mode:
        eid, r = all_results[0]
        print(f"\n[完成] 边 {eid} 蒸馏完成: {r['best_acc']*100:.2f}%")
        return {
            'status': 'success',
            'edge_id': eid,
            'best_test_acc': float(r['best_acc']),
            'train_time': r['train_time'],
        }

    # 全量模式：保存最佳边的模型作为 student_model.pth
    best_eid, best_r = max(all_results, key=lambda x: x[1]['best_acc'])
    torch.save({
        'model_state_dict': best_r['best_state'],
        'model_type': student_model_type,
        'num_classes': num_classes,
        'dataset_type': dataset_type,
        'best_test_acc': best_r['best_acc'],
        'epoch': best_r['best_epoch'],
    }, os.path.join(output_dir, 'student_model.pth'))

    # 报告
    result_dir = f"./tasks/{task_id}/result/edge_kd"
    os.makedirs(result_dir, exist_ok=True)
    total_time = sum(r['train_time'] for _, r in all_results)
    with open(os.path.join(result_dir, 'kd_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n知识蒸馏训练报告（分别蒸馏）\n" + "=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"学生模型: {student_model_type}\n")
        f.write(f"教师模型: {teacher_model_type}\n")
        f.write(f"蒸馏参数: alpha={alpha}, temperature={temperature}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"总耗时: {total_time:.1f}s\n\n")
        for eid, r in all_results:
            f.write(f"边侧 {eid}: 最佳准确率 {r['best_acc']*100:.2f}% (epoch {r['best_epoch']})\n")

    print(f"\n{'='*60}")
    print(f"[完成] 蒸馏结果汇总:")
    for eid, r in all_results:
        marker = " ← student_model.pth" if eid == best_eid else ""
        print(f"  边{eid}: {r['best_acc']*100:.2f}%{marker}")
    print(f"{'='*60}")

    return {
        'status': 'success',
        'edge_results': [{
            'edge_id': eid,
            'best_test_acc': float(r['best_acc']),
            'train_time': r['train_time'],
        } for eid, r in all_results],
    }


# ================================================================
# 3. 联邦学习（文件系统方式，单机模拟多边侧）
# ================================================================

@register_task
def federated_train_callback(task_id):
    """
    联邦学习训练（FedAvg，文件系统方式）

    单机模拟多个边侧的联邦学习过程：
    1. 初始化全局模型（可从知识蒸馏模型加载）
    2. 多轮迭代：
       - 各边侧加载全局模型 → 本地训练 → 生成本地更新
       - 聚合所有边侧模型 → 更新全局模型
    3. 保存最终全局模型和各边侧模型

    配置: input/federated_train.json
    输出:
      - output/federated_train/global_model.pth
      - output/federated_train/edge_{i}_model.pth
      - result/federated_train/federated_report.txt
    """
    print(f"\n{'='*60}")
    print(f"[联邦学习] 开始训练")
    print(f"{'='*60}")

    config_path = f"./tasks/{task_id}/input/federated_train.json"
    param_list = ['edge_data_paths', 'dataset_type', 'edge_model_type', 'num_classes',
                  'num_rounds', 'local_epochs']
    result, config = check_parameters(config_path, param_list)
    if 'error' in result:
        return {'status': 'error', 'message': result['error']}
    elif not result['valid']:
        return {'status': 'error', 'message': f"缺少参数: {', '.join(result['missing'])}"}

    edge_data_paths = config['edge_data_paths']   # list of paths
    dataset_type = config['dataset_type']
    edge_model_type = config['edge_model_type']
    num_classes = config['num_classes']
    num_rounds = config['num_rounds']
    local_epochs = config['local_epochs']
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    init_model_path = config.get('init_model_path', None)  # 可从KD模型初始化

    num_edges = len(edge_data_paths)
    output_dir = f"./tasks/{task_id}/output/federated_train"

    if check_output_exists(task_id, 'federated_train', 'global_model.pth'):
        print(f"[跳过] 联邦学习模型已存在")
        return {'status': 'cached'}

    print(f"[配置] 边侧数: {num_edges}, 轮数: {num_rounds}, 本地epoch: {local_epochs}")

    # 1. 加载各边侧数据
    edge_train_loaders = []
    edge_test_loaders = []
    edge_sizes = []

    for i, path in enumerate(edge_data_paths):
        print(f"[加载] 边侧 {i+1} 数据: {path}")
        splits = _load_split_data(path, dataset_type)
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        edge_train_loaders.append(_make_dataloader(X_train, y_train, batch_size, shuffle=True))
        edge_test_loaders.append(_make_dataloader(X_test, y_test, batch_size, shuffle=False))
        edge_sizes.append(len(X_train))

    # 2. 初始化全局模型（从各边 KD 模型聚合）
    global_model = create_model_by_type(edge_model_type, num_classes, dataset_type)

    edge_kd_dir = f"./tasks/{task_id}/output/edge_kd"
    per_edge_paths = [os.path.join(edge_kd_dir, f'student_edge_{i+1}.pth')
                      for i in range(num_edges)]
    has_per_edge = all(os.path.exists(p) for p in per_edge_paths)

    if has_per_edge:
        total_weight = sum(edge_sizes)
        avg_state = None
        for i in range(num_edges):
            ckpt = torch.load(per_edge_paths[i], map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            w = edge_sizes[i] / total_weight
            if avg_state is None:
                avg_state = {k: v.float() * w for k, v in state.items()}
            else:
                for k in avg_state:
                    avg_state[k] += state[k].float() * w
        global_model.load_state_dict(avg_state)
        print(f"[模型] 从 {num_edges} 个边侧 KD 模型聚合初始化全局模型")
    elif init_model_path and os.path.exists(init_model_path):
        checkpoint = torch.load(init_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            global_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            global_model.load_state_dict(checkpoint)
        print(f"[模型] 从 {init_model_path} 初始化全局模型（后备）")
    else:
        print(f"[模型] 随机初始化全局模型: {edge_model_type}")

    global_model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 3. 联邦学习循环
    best_avg_acc = 0.0
    best_global_state = None
    history = {'round': [], 'avg_test_acc': [], 'edge_test_accs': []}

    print(f"\n[训练] 开始联邦学习...")
    start_time = time.time()

    for round_idx in range(1, num_rounds + 1):
        edge_states = []
        edge_weights = []
        edge_accs = []

        # 各边侧本地训练
        for e in range(num_edges):
            local_model = create_model_by_type(edge_model_type, num_classes, dataset_type)
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)

            local_optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

            for le in range(1, local_epochs + 1):
                _train_one_epoch(
                    local_model, edge_train_loaders[e], local_optimizer, criterion, device,
                    epoch_info=(le, local_epochs))

            _, test_acc = _evaluate(local_model, edge_test_loaders[e], criterion, device)
            edge_accs.append(test_acc)
            print(f"    Round {round_idx} 边{e+1}: 本地训练完成, 测试准确率: {test_acc*100:.2f}%",
                  flush=True)

            edge_states.append(copy.deepcopy(local_model.state_dict()))
            edge_weights.append(edge_sizes[e])

        # FedAvg 聚合
        total_weight = sum(edge_weights)
        global_state = global_model.state_dict()

        for key in global_state:
            if global_state[key].is_floating_point() or global_state[key].is_complex():
                global_state[key] = sum(
                    edge_states[e][key] * (edge_weights[e] / total_weight)
                    for e in range(num_edges)
                )

        global_model.load_state_dict(global_state)

        avg_acc = np.mean(edge_accs)
        history['round'].append(round_idx)
        history['avg_test_acc'].append(avg_acc)
        history['edge_test_accs'].append(edge_accs)

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_global_state = copy.deepcopy(global_state)

        if round_idx % 2 == 0 or round_idx == num_rounds:
            edge_acc_str = ", ".join([f"E{e+1}:{a*100:.1f}%" for e, a in enumerate(edge_accs)])
            print(f"  Round {round_idx}/{num_rounds} | Avg: {avg_acc*100:.2f}% | {edge_acc_str}")

    total_time = time.time() - start_time
    print(f"[训练] 完成! 耗时: {total_time:.1f}s, 最佳平均准确率: {best_avg_acc*100:.2f}%")

    # 4. 保存全局模型和各边侧模型
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': best_global_state,
        'model_type': edge_model_type,
        'num_classes': num_classes,
        'dataset_type': dataset_type,
        'best_avg_acc': best_avg_acc,
    }, os.path.join(output_dir, 'global_model.pth'))

    for e in range(num_edges):
        torch.save({
            'model_state_dict': edge_states[e],
            'model_type': edge_model_type,
            'num_classes': num_classes,
        }, os.path.join(output_dir, f'edge_{e+1}_model.pth'))

    save_pickle(os.path.join(output_dir, 'train_history.pkl'), history)

    # 5. 报告
    result_dir = f"./tasks/{task_id}/result/federated_train"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'federated_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n联邦学习训练报告\n" + "=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型类型: {edge_model_type}\n")
        f.write(f"数据集: {dataset_type}\n")
        f.write(f"边侧数: {num_edges}\n")
        f.write(f"训练轮数: {num_rounds}\n")
        f.write(f"本地epoch: {local_epochs}\n")
        f.write(f"训练耗时: {total_time:.1f}s\n\n")
        f.write(f"最佳平均准确率: {best_avg_acc*100:.2f}%\n\n")
        f.write("各轮详情:\n")
        for i, (r, avg, accs) in enumerate(zip(history['round'], history['avg_test_acc'], history['edge_test_accs'])):
            edge_str = ", ".join([f"E{e+1}:{a*100:.1f}%" for e, a in enumerate(accs)])
            f.write(f"  Round {r}: Avg={avg*100:.2f}% | {edge_str}\n")

    print(f"[完成] 联邦学习模型已保存到 {output_dir}")
    return {
        'status': 'success',
        'best_avg_acc': float(best_avg_acc),
        'num_edges': num_edges,
        'num_rounds': num_rounds,
        'train_time': total_time,
    }


# ================================================================
# 4. 联邦学习 — 分布式模式（云侧聚合 + 边侧本地训练）
# ================================================================

def _wait_for_file(path, timeout=1800, interval=5):
    """
    等待文件出现，用于跨机器文件同步

    Args:
        path: 等待的文件路径
        timeout: 超时时间（秒），默认 30 分钟
        interval: 轮询间隔（秒），默认 5 秒

    Returns:
        True 表示文件已出现

    Raises:
        TimeoutError: 超时未等到文件
    """
    elapsed = 0
    while not os.path.exists(path):
        time.sleep(interval)
        elapsed += interval
        if elapsed >= timeout:
            raise TimeoutError(f"等待文件超时 ({timeout}s): {path}")
        if elapsed % 60 == 0:
            print(f"  [等待] {path} (已等待 {elapsed}s)")
    # 等一小段时间确保文件写完
    time.sleep(0.5)
    return True


def _load_federated_config(task_id):
    """加载联邦学习公共配置"""
    config_path = f"./tasks/{task_id}/input/federated_train.json"
    param_list = ['edge_data_paths', 'dataset_type', 'edge_model_type', 'num_classes',
                  'num_rounds', 'local_epochs']
    result, config = check_parameters(config_path, param_list)
    if 'error' in result:
        raise ValueError(result['error'])
    elif not result['valid']:
        raise ValueError(f"缺少参数: {', '.join(result['missing'])}")
    return config


@register_task
def federated_server_callback(task_id, **kwargs):
    """
    联邦学习 - 云侧聚合服务（分布式模式）

    在云侧运行，负责：
    1. 初始化全局模型并保存 global_model_round_0.pth
    2. 每轮轮询等待所有 edge_{i}_round_{r}.pth 文件出现
    3. 加载各边侧模型 → FedAvg 聚合 → 保存 global_model_round_{r}.pth
    4. 所有轮次完成后保存最终 global_model.pth

    配置: input/federated_train.json
    运行: python run_task.py --step federated_server --task_id xxx
    """
    print(f"\n{'='*60}")
    print(f"[联邦学习-云侧] 聚合服务启动")
    print(f"{'='*60}")

    config = _load_federated_config(task_id)

    edge_data_paths = config['edge_data_paths']
    dataset_type = config['dataset_type']
    edge_model_type = config['edge_model_type']
    num_classes = config['num_classes']
    num_rounds = config['num_rounds']
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    init_model_path = config.get('init_model_path', None)
    timeout = config.get('sync_timeout', 1800)  # 等待超时，默认 30 分钟

    num_edges = len(edge_data_paths)
    output_dir = f"./tasks/{task_id}/output/federated_train"
    os.makedirs(output_dir, exist_ok=True)

    if check_output_exists(task_id, 'federated_train', 'global_model.pth'):
        print(f"[跳过] 联邦学习全局模型已存在")
        return {'status': 'cached'}

    # 1. 初始化全局模型
    global_model = create_model_by_type(edge_model_type, num_classes, dataset_type)

    if init_model_path and os.path.exists(init_model_path):
        checkpoint = torch.load(init_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            global_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            global_model.load_state_dict(checkpoint)
        print(f"[模型] 从 {init_model_path} 初始化全局模型")
    else:
        print(f"[模型] 随机初始化全局模型: {edge_model_type}")

    global_model.to(device)

    # 保存初始全局模型 (round 0)
    round0_path = os.path.join(output_dir, 'global_model_round_0.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'model_type': edge_model_type,
        'num_classes': num_classes,
        'dataset_type': dataset_type,
    }, round0_path)
    print(f"[保存] 初始全局模型: {round0_path}")
    print(f"[配置] 边侧数: {num_edges}, 轮数: {num_rounds}")

    # 2. 逐轮聚合
    best_avg_acc = 0.0
    best_global_state = None
    history = {'round': [], 'avg_test_acc': [], 'edge_test_accs': []}

    print(f"\n[聚合] 开始等待各边侧训练结果...")
    start_time = time.time()

    for round_idx in range(1, num_rounds + 1):
        print(f"\n--- Round {round_idx}/{num_rounds} ---")

        # 等待所有边侧的本轮模型
        edge_round_paths = []
        for e in range(1, num_edges + 1):
            edge_path = os.path.join(output_dir, f'edge_{e}_round_{round_idx}.pth')
            print(f"  [等待] edge_{e}_round_{round_idx}.pth ...")
            _wait_for_file(edge_path, timeout=timeout)
            edge_round_paths.append(edge_path)
            print(f"  [收到] edge_{e}_round_{round_idx}.pth")

        # 加载各边侧模型并聚合
        edge_states = []
        edge_weights = []
        edge_accs = []
        for e, path in enumerate(edge_round_paths):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            edge_states.append(ckpt['model_state_dict'])
            edge_weights.append(ckpt.get('num_samples', 1))
            edge_accs.append(ckpt.get('test_acc', 0.0))

        # FedAvg 聚合
        total_weight = sum(edge_weights)
        global_state = global_model.state_dict()
        for key in global_state:
            if global_state[key].is_floating_point() or global_state[key].is_complex():
                global_state[key] = sum(
                    edge_states[e][key] * (edge_weights[e] / total_weight)
                    for e in range(num_edges)
                )
        global_model.load_state_dict(global_state)

        # 保存本轮全局模型
        round_path = os.path.join(output_dir, f'global_model_round_{round_idx}.pth')
        torch.save({
            'model_state_dict': global_state,
            'model_type': edge_model_type,
            'num_classes': num_classes,
            'dataset_type': dataset_type,
        }, round_path)

        avg_acc = np.mean(edge_accs) if edge_accs[0] > 0 else 0.0
        history['round'].append(round_idx)
        history['avg_test_acc'].append(avg_acc)
        history['edge_test_accs'].append(edge_accs)

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_global_state = copy.deepcopy(global_state)

        edge_acc_str = ", ".join([f"E{e+1}:{a*100:.1f}%" for e, a in enumerate(edge_accs)])
        print(f"  [聚合完成] Round {round_idx} | Avg: {avg_acc*100:.2f}% | {edge_acc_str}")

    total_time = time.time() - start_time

    # 3. 保存最终模型
    if best_global_state is None:
        best_global_state = global_model.state_dict()

    torch.save({
        'model_state_dict': best_global_state,
        'model_type': edge_model_type,
        'num_classes': num_classes,
        'dataset_type': dataset_type,
        'best_avg_acc': best_avg_acc,
    }, os.path.join(output_dir, 'global_model.pth'))

    save_pickle(os.path.join(output_dir, 'train_history.pkl'), history)

    # 报告
    result_dir = f"./tasks/{task_id}/result/federated_train"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'federated_server_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n联邦学习聚合报告（分布式模式-云侧）\n" + "=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型类型: {edge_model_type}\n")
        f.write(f"边侧数: {num_edges}\n")
        f.write(f"训练轮数: {num_rounds}\n")
        f.write(f"聚合耗时: {total_time:.1f}s\n\n")
        f.write(f"最佳平均准确率: {best_avg_acc*100:.2f}%\n\n")
        f.write("各轮详情:\n")
        for i, (r, avg, accs) in enumerate(zip(history['round'], history['avg_test_acc'], history['edge_test_accs'])):
            edge_str = ", ".join([f"E{e+1}:{a*100:.1f}%" for e, a in enumerate(accs)])
            f.write(f"  Round {r}: Avg={avg*100:.2f}% | {edge_str}\n")

    print(f"\n[完成] 联邦聚合完成! 耗时: {total_time:.1f}s, 最佳准确率: {best_avg_acc*100:.2f}%")
    return {
        'status': 'success',
        'best_avg_acc': float(best_avg_acc),
        'num_edges': num_edges,
        'num_rounds': num_rounds,
    }


@register_task
def federated_edge_callback(task_id, edge_id=None, **kwargs):
    """
    联邦学习 - 边侧本地训练（分布式模式）

    在边侧运行，负责：
    1. 每轮轮询等待 global_model_round_{r-1}.pth 出现
    2. 加载全局模型 → 用本地数据训练 local_epochs 轮
    3. 保存 edge_{id}_round_{r}.pth（含 test_acc 和 num_samples）
    4. 所有轮次完成后保存最终 edge_{id}_model.pth

    配置: input/federated_train.json
    运行: python run_task.py --step federated_edge --task_id xxx --edge_id 1
    """
    if edge_id is None:
        return {'status': 'error', 'message': '需要指定 --edge_id 参数（从 1 开始）'}

    print(f"\n{'='*60}")
    print(f"[联邦学习-边侧] 边 {edge_id} 本地训练启动")
    print(f"{'='*60}")

    config = _load_federated_config(task_id)

    edge_data_paths = config['edge_data_paths']
    dataset_type = config['dataset_type']
    edge_model_type = config['edge_model_type']
    num_classes = config['num_classes']
    num_rounds = config['num_rounds']
    local_epochs = config['local_epochs']
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    timeout = config.get('sync_timeout', 1800)

    num_edges = len(edge_data_paths)
    output_dir = f"./tasks/{task_id}/output/federated_train"
    os.makedirs(output_dir, exist_ok=True)

    # 检查 edge_id 合法性
    if edge_id < 1 or edge_id > num_edges:
        return {'status': 'error', 'message': f'edge_id={edge_id} 超出范围 [1, {num_edges}]'}

    # 检查是否已完成
    final_model_path = os.path.join(output_dir, f'edge_{edge_id}_model.pth')
    if os.path.exists(final_model_path):
        print(f"[跳过] 边 {edge_id} 最终模型已存在: {final_model_path}")
        return {'status': 'cached'}

    # 1. 加载本边侧数据
    data_path = edge_data_paths[edge_id - 1]  # edge_id 从 1 开始
    print(f"[加载] 边 {edge_id} 数据: {data_path}")
    splits = _load_split_data(data_path, dataset_type)
    X_train, y_train = splits['train']
    X_test, y_test = splits['test']
    train_loader = _make_dataloader(X_train, y_train, batch_size, shuffle=True)
    test_loader = _make_dataloader(X_test, y_test, batch_size, shuffle=False)
    num_samples = len(X_train)
    print(f"[加载] 训练集: {num_samples} 样本, 测试集: {len(X_test)} 样本")

    criterion = nn.CrossEntropyLoss()

    print(f"[配置] 轮数: {num_rounds}, 本地epoch: {local_epochs}")
    print(f"\n[训练] 开始本地训练...")
    start_time = time.time()

    # 2. 逐轮训练
    for round_idx in range(1, num_rounds + 1):
        # 检查本轮是否已完成（断点续传）
        edge_round_path = os.path.join(output_dir, f'edge_{edge_id}_round_{round_idx}.pth')
        if os.path.exists(edge_round_path):
            print(f"  Round {round_idx}: 已存在，跳过")
            continue

        # 等待上一轮全局模型
        prev_round = round_idx - 1
        global_round_path = os.path.join(output_dir, f'global_model_round_{prev_round}.pth')
        print(f"  Round {round_idx}: 等待 global_model_round_{prev_round}.pth ...")
        _wait_for_file(global_round_path, timeout=timeout)

        # 加载全局模型
        ckpt = torch.load(global_round_path, map_location=device, weights_only=False)
        local_model = create_model_by_type(edge_model_type, num_classes, dataset_type)
        local_model.load_state_dict(ckpt['model_state_dict'])
        local_model.to(device)

        local_optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

        for ep in range(1, local_epochs + 1):
            _train_one_epoch(
                local_model, train_loader, local_optimizer, criterion, device,
                epoch_info=(ep, local_epochs))

        _, test_acc = _evaluate(local_model, test_loader, criterion, device)

        # 保存本轮结果（含 num_samples 和 test_acc，供 server 聚合时使用）
        torch.save({
            'model_state_dict': local_model.state_dict(),
            'model_type': edge_model_type,
            'num_classes': num_classes,
            'num_samples': num_samples,
            'test_acc': test_acc,
            'edge_id': edge_id,
            'round': round_idx,
        }, edge_round_path)

        print(f"  Round {round_idx}/{num_rounds} | 边{edge_id} 测试准确率: {test_acc*100:.2f}%")

    total_time = time.time() - start_time

    # 3. 保存最终边侧模型（取最后一轮的模型）
    last_round_path = os.path.join(output_dir, f'edge_{edge_id}_round_{num_rounds}.pth')
    if os.path.exists(last_round_path):
        last_ckpt = torch.load(last_round_path, map_location=device, weights_only=False)
        torch.save({
            'model_state_dict': last_ckpt['model_state_dict'],
            'model_type': edge_model_type,
            'num_classes': num_classes,
        }, final_model_path)

    print(f"\n[完成] 边 {edge_id} 本地训练完成! 耗时: {total_time:.1f}s")
    return {
        'status': 'success',
        'edge_id': edge_id,
        'num_rounds': num_rounds,
        'train_time': total_time,
    }

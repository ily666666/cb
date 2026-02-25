#!/usr/bin/env python3
"""
模型准确率测试脚本

功能：
1. 加载训练好的模型
2. 在测试集上评估准确率、F1分数
3. 支持复值和实值模型

使用示例：python test_model_accuracy.py --model_type link11 --model_path E:\1Experiments\project1121\pth\link11\fl\client_1.pth --dataset_type link11 --num_classes 7 --data_path E:\1Experiments\project0115\run\data\link11.pkl
    
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fed.project import create_model_by_type


class LazyDataset(torch.utils.data.Dataset):
    """惰性加载数据集 - 原始字典格式 {(class, snr): signal_array}"""
    def __init__(self, raw_data, dataset_type):
        self.dataset_type = dataset_type
        self.raw_data = raw_data
        
        # 标签映射
        if dataset_type == 'rml2016':
            # 修复：使用与readdata_rml2016.py相同的顺序
            mod_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
            self.label_map = {mod: idx for idx, mod in enumerate(mod_types)}
        elif dataset_type == 'link11':
            emitter_types = ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']
            self.label_map = {emitter: idx for idx, emitter in enumerate(emitter_types)}
        elif dataset_type == 'radar':
            self.label_map = None
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        self._index_map = None
        self._total_samples = None
    
    def _build_index(self):
        """延迟构建索引映射"""
        if self._index_map is None:
            self._index_map = []
            for key, signal_array in self.raw_data.items():
                for local_idx in range(len(signal_array)):
                    self._index_map.append((key, local_idx))
            self._total_samples = len(self._index_map)
    
    def __len__(self):
        if self._total_samples is None:
            self._total_samples = sum(len(signal_array) for signal_array in self.raw_data.values())
        return self._total_samples
    
    def __getitem__(self, idx):
        if self._index_map is None:
            self._build_index()
        
        key, local_idx = self._index_map[idx]
        signal = self.raw_data[key][local_idx]
        
        # 转换为复数（I/Q格式）
        if signal.shape[0] == 2:
            signal_complex = signal[0] + 1j * signal[1]
            signal_tensor = torch.from_numpy(signal_complex).cfloat()
        else:
            signal_tensor = torch.from_numpy(signal).float()
        
        # 获取标签
        if self.dataset_type == 'rml2016':
            label = self.label_map[key[0]]
        elif self.dataset_type == 'link11':
            label = self.label_map[key[0]]
        elif self.dataset_type == 'radar':
            label = key[0] - 1  # radar标签从1开始，需要减1
        
        return signal_tensor, label


def load_test_data(data_path, dataset_type, batch_size):
    """加载测试数据"""
    print(f"\n[加载数据] 数据路径: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 检查是文件还是文件夹
    if os.path.isdir(data_path):
        # 文件夹模式：批次加载
        print(f"[加载数据] 检测到文件夹，使用批次加载模式")
        
        import glob
        batch_files_pkl = sorted(glob.glob(os.path.join(data_path, '*.pkl')))
        batch_files_mat = sorted(glob.glob(os.path.join(data_path, '*.mat')))
        
        if batch_files_pkl:
            batch_files = batch_files_pkl
            file_format = 'pkl'
        elif batch_files_mat:
            batch_files = batch_files_mat
            file_format = 'mat'
        else:
            raise FileNotFoundError(f"文件夹 {data_path} 中没有找到 .pkl 或 .mat 文件")
        
        print(f"[加载数据] 找到 {len(batch_files)} 个批次文件 (.{file_format})")
        
        # 创建批次加载 Dataset
        class BatchLoadingDataset(torch.utils.data.IterableDataset):
            """边加载边测试的数据集 - 从文件夹中逐个加载批次文件"""
            def __init__(self, batch_files, dataset_type, file_format='pkl'):
                self.batch_files = batch_files
                self.dataset_type = dataset_type
                self.file_format = file_format
                
                # 标签映射（仅用于 pkl 格式的字典数据）
                if dataset_type == 'rml2016':
                    # 修复：使用与readdata_rml2016.py相同的顺序
                    mod_types = ['16QAM', '64QAM', '8PSK', 'BPSK', 'GMSK', 'QPSK']
                    self.label_map = {mod: idx for idx, mod in enumerate(mod_types)}
                elif dataset_type == 'link11':
                    emitter_types = ['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']
                    self.label_map = {emitter: idx for idx, emitter in enumerate(emitter_types)}
                elif dataset_type == 'radar':
                    self.label_map = None
                else:
                    raise ValueError(f"不支持的数据集类型: {dataset_type}")
            
            def __iter__(self):
                """逐个加载批次文件并生成样本"""
                sample_count = 0
                
                if self.file_format == 'pkl':
                    # PKL 格式：字典格式 {(class, snr): signal_array}
                    for batch_idx, batch_file in enumerate(self.batch_files):
                        with open(batch_file, 'rb') as f:
                            batch_data = pickle.load(f)
                        
                        # 遍历批次中的所有样本
                        for key, signal_array in batch_data.items():
                            for signal in signal_array:
                                # 转换为复数（I/Q格式）
                                if signal.shape[0] == 2:
                                    signal_complex = signal[0] + 1j * signal[1]
                                    signal_tensor = torch.from_numpy(signal_complex).cfloat()
                                else:
                                    signal_tensor = torch.from_numpy(signal).float()
                                
                                # 获取标签
                                if self.dataset_type == 'rml2016':
                                    label = self.label_map[key[0]]
                                elif self.dataset_type == 'link11':
                                    label = self.label_map[key[0]]
                                
                                sample_count += 1
                                yield signal_tensor, label
                        
                        del batch_data
                
                elif self.file_format == 'mat':
                    # MAT 格式：radar 数据 X_batch: (2, 500, N), Y_batch: (1, N)
                    import h5py
                    
                    for batch_idx, batch_file in enumerate(self.batch_files):
                        with h5py.File(batch_file, 'r') as f:
                            X_batch = np.array(f['X_batch'])
                            Y_batch = np.array(f['Y_batch']).flatten()
                        
                        num_samples = X_batch.shape[2]
                        for i in range(num_samples):
                            signal = X_batch[:, :, i]
                            signal_complex = signal[0] + 1j * signal[1]
                            signal_tensor = torch.from_numpy(signal_complex).cfloat()
                            label = int(Y_batch[i]) - 1
                            
                            sample_count += 1
                            yield signal_tensor, label
                        
                        del X_batch, Y_batch
        
        test_dataset = BatchLoadingDataset(batch_files, dataset_type, file_format)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"✅ 批次加载数据准备完成: {len(batch_files)} 个批次文件")
        return test_loader
    
    # 单文件模式：原有逻辑
    # 根据文件扩展名选择加载方式
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.mat':
        # 加载 .mat 文件（MATLAB v7.3格式，需要用h5py）
        print(f"[加载数据] 检测到 .mat 格式，使用 h5py 加载")
        import h5py
        
        # 提取数据（根据数据集类型）
        if dataset_type == 'radar':
            # Radar数据集的.mat格式
            with h5py.File(data_path, 'r') as mat:
                X = np.array(mat['X'])  # Shape: (2, 500, num_samples)
                Y = np.array(mat['Y']).flatten()  # Shape: (num_samples,)
            
            # Convert to (num_samples, 2, 500) format
            num_samples = X.shape[2]
            X = np.transpose(X, (2, 0, 1))  # (num_samples, 2, 500)
            
            # 注意：不要在这里转换标签！保持原始的1-7，让LazyDataset的__getitem__去减1
            # Y = Y - 1  # ← 删除这行！
            
            # 转换为字典格式 {(class, snr): signal_array}
            data = {}
            for i in range(num_samples):
                label = int(Y[i])  # 保持原始标签（1-7）
                snr = 0  # .mat格式没有SNR信息
                key = (label, snr)
                if key not in data:
                    data[key] = []
                data[key].append(X[i])
            
            # 转换为numpy数组
            for key in data:
                data[key] = np.array(data[key])
            
            print(f"✅ 加载 .mat 数据: {num_samples} 样本, {len(np.unique(Y))} 类别")
        else:
            raise ValueError(f"不支持从 .mat 文件加载 {dataset_type} 数据集")
    
    elif file_ext == '.pkl':
        # 加载 .pkl 文件
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .pkl 和 .mat")
    
    # 检查数据格式
    if 'test' in data:
        # 预划分数据格式（边侧数据）
        from utils.readdata_presplit import PresplitDataset
        test_dataset = PresplitDataset(data_path, split='test')
        print(f"✅ 加载预划分测试数据: {len(test_dataset)} 样本")
    
    elif 'X_test' in data and 'y_test' in data:
        # 完整数据集格式（云侧预训练数据）
        from torch.utils.data import TensorDataset
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        # 转换为张量
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.from_numpy(y_test)
        
        test_dataset = TensorDataset(X_test, y_test)
        print(f"✅ 加载完整数据集测试数据: {len(test_dataset)} 样本")
    
    else:
        # 原始字典格式 {(class, snr): signal_array}
        print(f"✅ 检测到原始字典格式数据")
        total_samples = sum(len(signal_array) for signal_array in data.values())
        print(f"✅ 加载原始数据: {total_samples} 样本")
        test_dataset = LazyDataset(data, dataset_type)
    
    use_drop_last = (dataset_type in ['radioml', 'radar', 'rml2016', 'link11'])
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return test_loader


def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n[测试中] 正在评估模型...")
    
    # 检查是否是 IterableDataset（批次加载模式）
    is_iterable = isinstance(test_loader.dataset, torch.utils.data.IterableDataset)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            
            # 处理复数输出
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 每10个批次打印一次进度
            if (batch_idx + 1) % 10 == 0:
                if is_iterable:
                    # IterableDataset 无法获取总长度
                    print(f"  批次 {batch_idx + 1}: 当前准确率 {100.*correct/total:.2f}% (已处理 {total} 样本)")
                else:
                    # 普通 Dataset 可以获取总长度
                    print(f"  批次 {batch_idx + 1}/{len(test_loader)}: 当前准确率 {100.*correct/total:.2f}%")
    
    # 计算最终指标
    num_batches = batch_idx + 1
    test_loss = test_loss / num_batches if num_batches > 0 else 0.0
    test_acc = 100. * correct / total if total > 0 else 0.0
    test_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return test_loss, test_acc, test_f1, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='测试模型准确率')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--model_type', type=str, default=None,
                        help='模型类型（如果不指定，从checkpoint中读取）')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, help='测试数据路径')
    parser.add_argument('--dataset_type', type=str, required=True, help='数据集类型')
    parser.add_argument('--num_classes', type=int, required=True, help='类别数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    
    # 输出参数
    parser.add_argument('--show_confusion', action='store_true', help='显示混淆矩阵')
    parser.add_argument('--show_report', action='store_true', help='显示分类报告')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"[模型测试] 准确率评估")
    print(f"{'='*70}")
    print(f"模型路径: {args.model_path}")
    print(f"数据集: {args.dataset_type}")
    print(f"类别数: {args.num_classes}")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 2. 加载模型
    print("[步骤1] 加载模型...")
    
    # 添加模块支持
    dataset_modules = ['readdata_rml2016', 'readdata_radar', 'readdata_radioml',
                       'readdata_reii', 'readdata_25', 'readdata_link11']
    
    for module_name in dataset_modules:
        if module_name not in sys.modules:
            try:
                module = __import__(f'utils.{module_name}', fromlist=[module_name])
                sys.modules[module_name] = module
            except:
                pass
    
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型出错: {e}")
        print("尝试使用自定义 Unpickler...")
        
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in dataset_modules:
                    module = f'utils.{module}'
                return super().find_class(module, name)
        
        with open(args.model_path, 'rb') as f:
            checkpoint = CPU_Unpickler(f).load()
    
    # 提取模型信息
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            model_architecture = checkpoint.get('model_architecture', args.model_type)
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            model_architecture = args.model_type
        elif 'model' in checkpoint:
            # 处理包含 'model' 键的checkpoint（训练保存的格式）
            model_state = checkpoint['model']
            model_architecture = args.model_type
        else:
            model_state = checkpoint
            model_architecture = args.model_type
    else:
        model_state = checkpoint
        model_architecture = args.model_type
    
    if model_architecture is None:
        print("错误: 无法确定模型架构，请使用 --model_type 参数指定")
        return
    
    print(f"模型架构: {model_architecture}")
    
    # 创建模型
    model = create_model_by_type(model_architecture, args.num_classes, args.dataset_type)
    model.load_state_dict(model_state)
    model = model.to(device)
    
    print(f"✅ 模型加载成功")
    
    # 3. 加载测试数据
    print("\n[步骤2] 加载测试数据...")
    test_loader = load_test_data(args.data_path, args.dataset_type, args.batch_size)
    
    # 4. 测试模型
    print("\n[步骤3] 测试模型...")
    test_loss, test_acc, test_f1, all_preds, all_targets = test_model(model, test_loader, device)
    
    # 5. 输出结果
    print(f"\n{'='*70}")
    print(f"[测试结果]")
    print(f"{'='*70}")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"F1分数 (macro): {test_f1:.4f}")
    print(f"{'='*70}\n")
    
    # 6. 可选：显示混淆矩阵
    if args.show_confusion:
        print("\n[混淆矩阵]")
        cm = confusion_matrix(all_targets, all_preds)
        print(cm)
        print()
    
    # 7. 可选：显示分类报告
    if args.show_report:
        print("\n[分类报告]")
        print(classification_report(all_targets, all_preds, zero_division=0))
        print()


if __name__ == '__main__':
    main()

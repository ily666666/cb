#!/usr/bin/env python3
"""
云侧预训练脚本 - 训练教师模型

功能：
1. 加载云侧数据（预划分或实时划分）
2. 创建并训练教师模型（复杂网络）
3. 保存训练好的教师模型

使用示例：
    python run/cloud/run_cloud_pretrain.py --dataset_type link11 --num_classes 7 --cloud_epochs 50
"""
import sys
import os
import argparse
import torch
import pickle

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed.project import create_model_by_type, ProjectCloud
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='云侧教师模型预训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=None, help='数据路径')
    parser.add_argument('--dataset_type', type=str, default='link11', 
                        choices=['link11', 'rml2016', 'radar'],
                        help='数据集类型')
    parser.add_argument('--num_classes', type=int, default=7, help='类别数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    # 模型参数
    parser.add_argument('--cloud_model', type=str, default=None,
                        help='云侧模型类型（默认：complex_resnet50_{dataset_type}）')
    
    # 训练参数
    parser.add_argument('--cloud_epochs', type=int, default=10, help='预训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', 
                        choices=['cosine', 'step', 'none'], help='学习率调度器')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='模型保存目录（默认：run/cloud/pth/{dataset_type}）')
    parser.add_argument('--pretrained_teacher', type=str, default=None, 
                        help='预训练教师模型路径（如果已存在）')
    
    args = parser.parse_args()
    
    # 设置默认模型类型
    if args.cloud_model is None:
        args.cloud_model = f'complex_resnet50_{args.dataset_type}'
    
    # 设置默认保存目录
    if args.save_dir is None:
        args.save_dir = f'run/cloud/pth/{args.dataset_type}'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"[云侧预训练] 教师模型训练")
    print(f"{'='*70}")
    print(f"数据集: {args.dataset_type}")
    print(f"类别数: {args.num_classes}")
    print(f"模型: {args.cloud_model}")
    print(f"训练轮数: {args.cloud_epochs}")
    print(f"保存目录: {args.save_dir}")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 加载预划分数据
    print("[步骤1] 加载预划分数据...")
    
    # 使用用户指定的数据路径
    if args.data_path:
        # 用户指定了数据路径，直接使用
        presplit_cloud_file = args.data_path
        print(f"  使用指定数据路径: {presplit_cloud_file}")
    
    if not os.path.exists(presplit_cloud_file):
        print(f"错误: 预划分数据文件不存在: {presplit_cloud_file}")
        print(f"请先运行 'python run/prepare_data_splits.py' 预先划分数据")
        print(f"或使用 --data_path 指定正确的数据文件路径")
        return
    
    from utils.readdata_presplit import PresplitDataset
    
    cloud_train_dataset = PresplitDataset(presplit_cloud_file, split='train')
    cloud_val_dataset = PresplitDataset(presplit_cloud_file, split='val')
    cloud_test_dataset = PresplitDataset(presplit_cloud_file, split='test')
    
    cloud_train_loader = DataLoader(cloud_train_dataset, batch_size=args.batch_size, 
                                     shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    cloud_val_loader = DataLoader(cloud_val_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    cloud_test_loader = DataLoader(cloud_test_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    
    print(f"✅ 预划分数据加载完成")
    print(f"  云侧训练集: {len(cloud_train_dataset)} 样本")
    print(f"  云侧验证集: {len(cloud_val_dataset)} 样本")
    print(f"  云侧测试集: {len(cloud_test_dataset)} 样本")
    
    # 2. 创建模型
    print("\n[步骤2] 创建教师模型...")
    cloud_model = create_model_by_type(args.cloud_model, args.num_classes, args.dataset_type)
    print(f"模型架构: {args.cloud_model}")
    
    # 3. 配置训练参数
    print("\n[步骤3] 配置训练参数...")
    config = {
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': 0.9,
        'lr_scheduler': args.lr_scheduler,
        'lr_step_size': 20,
        'lr_gamma': 0.1,
        'lr_min': 1e-6,
        'cloud_epochs': args.cloud_epochs,
        'grad_clip': True,
        'grad_clip_value': 1.0,
    }
    
    project_cloud = ProjectCloud(
        cloud_model=cloud_model,
        train_loader=cloud_train_loader,
        val_loader=cloud_val_loader,
        test_loader=cloud_test_loader,
        device=device,
        config=config,
        save_dir=args.save_dir
    )
    
    # 4. 训练或加载模型
    teacher_model_path = f'{args.save_dir}/teacher_model.pth'
    
    # 添加数据加载器模块支持
    dataset_modules = [
        'readdata_rml2016', 'readdata_radar', 'readdata_radioml',
        'readdata_reii', 'readdata_25', 'readdata_link11'
    ]
    
    for module_name in dataset_modules:
        if module_name not in sys.modules:
            try:
                module = __import__(f'utils.{module_name}', fromlist=[module_name])
                sys.modules[module_name] = module
            except Exception as e:
                pass
    
    if args.pretrained_teacher and os.path.exists(args.pretrained_teacher):
        print(f"\n[步骤4] 加载预训练的教师模型: {args.pretrained_teacher}")
        
        try:
            checkpoint = torch.load(args.pretrained_teacher, map_location='cpu')
        except Exception as e:
            print(f"加载模型出错: {e}")
            print("尝试使用 pickle 模块加载...")
            
            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module in dataset_modules:
                        module = f'utils.{module}'
                    return super().find_class(module, name)
            
            with open(args.pretrained_teacher, 'rb') as f:
                checkpoint = CPU_Unpickler(f).load()
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        project_cloud.cloud_model.load_state_dict(state_dict)
        print("预训练教师模型加载成功")
        
        torch.save({
            'model_state_dict': project_cloud.cloud_model.state_dict(),
            'model_architecture': args.cloud_model,
            'num_classes': args.num_classes,
            'dataset_type': args.dataset_type
        }, teacher_model_path)
        print(f"教师模型已保存到: {teacher_model_path}")
        
    elif os.path.exists(teacher_model_path):
        print(f"\n[步骤4] 检测到已存在的教师模型: {teacher_model_path}")
        print("跳过预训练，直接使用已有模型")
        
        try:
            checkpoint = torch.load(teacher_model_path, map_location='cpu')
        except Exception as e:
            print(f"加载模型出错: {e}")
            
            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module in dataset_modules:
                        module = f'utils.{module}'
                    return super().find_class(module, name)
            
            with open(teacher_model_path, 'rb') as f:
                checkpoint = CPU_Unpickler(f).load()
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 检查类别数是否匹配
        saved_num_classes = None
        if isinstance(checkpoint, dict) and 'num_classes' in checkpoint:
            saved_num_classes = checkpoint['num_classes']
        else:
            for key in state_dict.keys():
                if 'fc1.fc_r.bias' in key or 'fc.bias' in key:
                    saved_num_classes = state_dict[key].shape[0]
                    break
        
        if saved_num_classes is not None and saved_num_classes != args.num_classes:
            print(f"警告: 保存的模型类别数 ({saved_num_classes}) 与当前配置 ({args.num_classes}) 不匹配")
            print(f"删除旧模型文件: {teacher_model_path}")
            os.remove(teacher_model_path)
            print("将重新训练教师模型...")
            
            project_cloud.pretrain_cloud()
            
            teacher_model_state = project_cloud.cloud_model.state_dict()
            torch.save({
                'model_state_dict': teacher_model_state,
                'model_architecture': args.cloud_model,
                'num_classes': args.num_classes,
                'dataset_type': args.dataset_type
            }, teacher_model_path)
            print(f"教师模型已保存: {teacher_model_path}")
        else:
            project_cloud.cloud_model.load_state_dict(state_dict)
            print("教师模型加载成功")
    else:
        print("\n[步骤4] 开始预训练教师模型...")
        project_cloud.pretrain_cloud()
        
        teacher_model_state = project_cloud.cloud_model.state_dict()
        torch.save({
            'model_state_dict': teacher_model_state,
            'model_architecture': args.cloud_model,
            'num_classes': args.num_classes,
            'dataset_type': args.dataset_type
        }, teacher_model_path)
        print(f"教师模型已保存: {teacher_model_path}")
    
    print(f"\n{'='*70}")
    print(f"[完成] 教师模型预训练完成！")
    print(f"模型路径: {teacher_model_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

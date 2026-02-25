"""
知识蒸馏辅助类
专门处理Project模式中的知识蒸馏逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from .kd_utils import (
    calculate_adaptive_alpha,
    compute_teacher_loss_mean,
    create_kd_criterion,
    get_linear_schedule_k
)
from distiller_zoo import DistillKL


class DistillationHelper:
    """
    知识蒸馏辅助类
    封装所有蒸馏逻辑，支持多种蒸馏方法和自适应机制
    """
    
    @staticmethod
    def distill_from_teacher(edge_model, teacher_model_state, teacher_model_architecture,
                             train_loader, test_loader, device, config, edge_id,
                             create_model_func, local_test_loader=None):
        """
        执行知识蒸馏
        
        Args:
            edge_model: 客户端（学生）模型
            teacher_model_state: 教师模型状态字典
            teacher_model_architecture: 教师模型架构名称
            train_loader: 训练数据加载器
            test_loader: 全局测试数据加载器
            device: 设备
            config: 配置字典
            edge_id: 客户端ID
            create_model_func: 创建模型的函数（根据数据集类型）
            local_test_loader: 客户端本地测试数据加载器（可选）
        
        Returns:
            蒸馏后的学生模型状态字典
        """
        print(f"\n========== Edge {edge_id+1} 开始知识蒸馏 ==========")
        print(f"蒸馏方法: {config.get('kd_distill', 'kd')}")
        print(f"自适应蒸馏: {'启用' if config.get('kd_adaptive', False) else '禁用'}")
        
        # 获取模型配置
        num_classes = edge_model.fc.out_features
        dataset_type = config.get('dataset_type', 'folder')
        
        # 创建教师模型
        teacher_model = create_model_func(
            teacher_model_architecture,
            num_classes,
            dataset_type
        )
        
        # 加载教师模型参数
        teacher_model.load_state_dict(teacher_model_state)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        print(f"教师模型: {teacher_model_architecture}")
        print(f"学生模型: {edge_model.__class__.__name__}")
        
        # 确保模型在正确的设备上
        edge_model = edge_model.to(device)
        teacher_model = teacher_model.to(device)
        
        # 提取特征形状（用于某些蒸馏方法）
        with torch.no_grad():
            # 创建虚拟输入（根据数据集类型和模型类型）
            if dataset_type == 'radioml':
                dummy_size = 128
            elif dataset_type == 'reii':
                dummy_size = 2000
            elif dataset_type == 'radar':
                dummy_size = 500
            elif dataset_type == 'rml2016':
                dummy_size = 600
            elif dataset_type == 'link11':
                dummy_size = 1024
            else:
                dummy_size = 4096
            
            # 教师模型是复数模型，使用复数输入 (batch_size, size)
            dummy_input_teacher = torch.randn(1, dummy_size, dtype=torch.complex64).to(device)
            feat_t, _ = teacher_model(dummy_input_teacher, is_feat=True)
            
            # 学生模型是实数模型，使用复数输入 (batch_size, size)
            dummy_input_student = torch.randn(1, dummy_size, dtype=torch.complex64).to(device)
            feat_s, _ = edge_model(dummy_input_student, is_feat=True)
            
            t_shapes = [f.shape for f in feat_t]
            s_shapes = [f.shape for f in feat_s]
        
        print(f"教师特征形状: {[s[1] for s in t_shapes]}")
        print(f"学生特征形状: {[s[1] for s in s_shapes]}")
        
        # 创建蒸馏损失和辅助模块
        distill_config = {
            'kd_temperature': config.get('kd_temperature', 4.0),
            'dkd_alpha': config.get('dkd_alpha', 1.0),
            'dkd_beta': config.get('dkd_beta', 1.0),
            'hint_layer': config.get('hint_layer', 2),
            'at_p': config.get('at_p', 2),
            'rkd_w_d': config.get('rkd_w_d', 25.0),
            'rkd_w_a': config.get('rkd_w_a', 50.0),
            'feat_dim': config.get('corr_feat_dim', 128),
            's_shapes': s_shapes,
            't_shapes': t_shapes,
            's_dim': s_shapes[-1][1] if len(s_shapes) > 0 else 256,
            't_dim': t_shapes[-1][1] if len(t_shapes) > 0 else 512,
        }
        
        distill_method = config.get('kd_distill', 'kd')
        try:
            criterion_kd, trainable_modules = create_kd_criterion(distill_method, distill_config)
            print(f"蒸馏损失函数创建成功: {distill_method}")
        except Exception as e:
            print(f"警告: 创建蒸馏损失函数失败 ({e})，回退到标准KD")
            criterion_kd, trainable_modules = create_kd_criterion('kd', distill_config)
            distill_method = 'kd'
        
        # 分类损失和KL散度损失
        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(temperature=config.get('kd_temperature', 4.0))
        
        # 将所有损失函数移到设备上
        criterion_cls = criterion_cls.to(device)
        criterion_div = criterion_div.to(device)
        
        # criterion_kd可能是Module或ModuleList，都支持.to()
        if isinstance(criterion_kd, nn.Module):
            criterion_kd = criterion_kd.to(device)
        
        # 如果有自适应蒸馏，计算教师损失均值
        teacher_loss_mean = None
        if config.get('kd_adaptive', False):
            teacher_loss_mean = compute_teacher_loss_mean(
                teacher_model, train_loader, device
            )
            print(f"教师损失均值: {teacher_loss_mean:.6f}")
        
        # 设置优化器（包括学生模型和辅助模块）
        params_to_optimize = list(edge_model.parameters())
        for module in trainable_modules:
            module = module.to(device)
            params_to_optimize.extend(list(module.parameters()))
        
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                params_to_optimize,
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                params_to_optimize,
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
        
        # 知识蒸馏训练
        edge_model = edge_model.to(device)
        edge_model.train()
        
        kd_epochs = config.get('kd_epochs', 5)
        print(f"开始蒸馏训练，共{kd_epochs}个epoch...")
        
        for epoch in range(kd_epochs):
            # 自适应k值调度
            if config.get('kd_adaptive', False):
                current_k = get_linear_schedule_k(
                    epoch + 1, kd_epochs,
                    config.get('kd_k_plus', 15.0),
                    config.get('kd_k_minus', -10.0)
                )
                print(f"Epoch {epoch+1}: k = {current_k:.3f}")
            
            total_loss = 0
            total_ce_loss = 0
            total_div_loss = 0
            total_kd_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, 
                       desc=f"Edge {edge_id+1} KD Epoch {epoch+1}/{kd_epochs}",
                       leave=False)
            
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # 两个模型都使用复数数据
                # 学生模型前向（提取特征）
                feat_s, logit_s = edge_model(data, is_feat=True)
                
                # 教师模型前向（提取特征）
                with torch.no_grad():
                    feat_t, logit_t = teacher_model(data, is_feat=True)
                
                # 分类损失
                loss_cls = criterion_cls(logit_s, targets)
                
                # 自适应蒸馏：根据教师损失动态调整权重
                if config.get('kd_adaptive', False):
                    with torch.no_grad():
                        criterion_no_reduce = nn.CrossEntropyLoss(reduction='none')
                        teacher_losses = criterion_no_reduce(logit_t, targets)  # [B]
                    
                    # 计算自适应权重
                    adaptive_alphas = calculate_adaptive_alpha(
                        teacher_losses, current_k, teacher_loss_mean
                    )
                    alpha = adaptive_alphas.mean().item()
                else:
                    alpha = config.get('kd_alpha', 0.5)
                
                # KL散度损失
                loss_div = criterion_div(logit_s, logit_t)
                
                # 特定蒸馏方法的损失
                loss_kd = DistillationHelper._compute_distill_loss(
                    distill_method, criterion_kd, feat_s, feat_t,
                    logit_s, logit_t, targets, trainable_modules, config
                )
                
                # 总损失
                beta = config.get('dkd_beta', 0.5) if distill_method == 'dkd' else 0.5
                loss = (1 - alpha) * loss_cls + alpha * loss_div + beta * loss_kd
                
                loss.backward()
                
                if config.get('grad_clip', False):
                    nn.utils.clip_grad_norm_(edge_model.parameters(), 
                                           config['grad_clip_value'])
                
                optimizer.step()
                
                total_loss += loss.item()
                total_ce_loss += loss_cls.item()
                total_div_loss += loss_div.item()
                total_kd_loss += loss_kd if isinstance(loss_kd, (int, float)) else loss_kd.item()
                
                _, predicted = logit_s.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'cls': total_ce_loss / (batch_idx + 1),
                    'div': total_div_loss / (batch_idx + 1),
                    'kd': total_kd_loss / (batch_idx + 1),
                    'acc': 100. * correct / total,
                    'α': f'{alpha:.3f}'
                })
            
            avg_loss = total_loss / len(train_loader)
            avg_acc = 100. * correct / total
            
            print(f"Edge {edge_id+1} KD Epoch {epoch+1}/{kd_epochs} - "
                  f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
        
        # 在全局测试集上测试蒸馏后的模型
        global_test_loss, global_test_acc, global_test_f1 = DistillationHelper._test_model(
            edge_model, test_loader, criterion_cls, device
        )
        
        # 在客户端本地测试集上测试（如果有）
        local_test_loss, local_test_acc, local_test_f1 = None, None, None
        if local_test_loader is not None:
            local_test_loss, local_test_acc, local_test_f1 = DistillationHelper._test_model(
                edge_model, local_test_loader, criterion_cls, device
            )
        
        # 打印测试结果
        print_msg = (f"Edge {edge_id+1} 知识蒸馏完成 - "
                    f"Global Test Loss: {global_test_loss:.4f}, Acc: {global_test_acc:.2f}%, F1: {global_test_f1:.4f}")
        if local_test_loss is not None:
            print_msg += f", Local Test Loss: {local_test_loss:.4f}, Acc: {local_test_acc:.2f}%, F1: {local_test_f1:.4f}"
        print(print_msg)
        print("=" * 60)
        
        # 清理内存
        if device.type == 'cuda':
            del teacher_model
            for module in trainable_modules:
                del module
            torch.cuda.empty_cache()
        
        return edge_model.state_dict()
    
    @staticmethod
    def _compute_distill_loss(distill_method, criterion_kd, feat_s, feat_t,
                              logit_s, logit_t, targets, trainable_modules, config):
        """计算特定蒸馏方法的损失"""
        try:
            if distill_method == 'kd':
                return 0.0
            
            elif distill_method == 'dkd':
                return criterion_kd(logit_s, logit_t, targets)
            
            elif distill_method == 'fsp':
                return criterion_kd(feat_s, feat_t)
            
            elif distill_method == 'hint':
                hint_layer = config.get('hint_layer', 2)
                if len(trainable_modules) > 0:
                    f_s = trainable_modules[0](feat_s[hint_layer])
                    f_t = feat_t[hint_layer]
                else:
                    f_s = feat_s[hint_layer]
                    f_t = feat_t[hint_layer]
                return criterion_kd(f_s, f_t)
            
            elif distill_method == 'attention':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                return sum(loss_group)
            
            elif distill_method == 'rkd':
                return criterion_kd(feat_s[-1], feat_t[-1])
            
            elif distill_method == 'nst':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                return sum(loss_group)
            
            elif distill_method == 'similarity':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                return sum(loss_group)
            
            elif distill_method == 'pkt':
                return criterion_kd(feat_s[-1], feat_t[-1])
            
            elif distill_method == 'correlation':
                if len(trainable_modules) >= 2:
                    f_s = trainable_modules[0](feat_s[-1])
                    f_t = trainable_modules[1](feat_t[-1])
                else:
                    f_s = feat_s[-1]
                    f_t = feat_t[-1]
                return criterion_kd(f_s, f_t)
            
            elif distill_method == 'vid':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                return sum(loss_group)
            
            elif distill_method == 'abound':
                if len(trainable_modules) > 0:
                    g_s = trainable_modules[0](feat_s[1:-1])
                else:
                    g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                return criterion_kd(g_s, g_t)
            
            elif distill_method == 'factor':
                if len(trainable_modules) >= 2:
                    factor_s = trainable_modules[0](feat_s[-2])
                    factor_t = trainable_modules[1](feat_t[-2], is_factor=True)
                else:
                    factor_s = feat_s[-2]
                    factor_t = feat_t[-2]
                return criterion_kd(factor_s, factor_t)
            
            elif distill_method == 'kdsvd':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                return sum(loss_group)
            
            else:
                return 0.0
        except Exception as e:
            print(f"警告: 计算{distill_method}损失时出错 ({e})，返回0")
            return 0.0
    
    @staticmethod
    def _test_model(model, test_loader, criterion, device):
        """在测试集上评估模型"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        
        model.train()  # 恢复训练模式
        return test_loss, test_acc, test_f1



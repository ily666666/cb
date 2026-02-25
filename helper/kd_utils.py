"""
Knowledge Distillation Utilities
自适应知识蒸馏工具函数
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def calculate_adaptive_alpha(teacher_losses, k, t):
    """
    根据论文公式计算自适应权重α
    α = e^(-1/√d_f)
    d_f = e^(-k(x-t))
    
    Args:
        teacher_losses: 教师模型在当前批次的损失 (tensor) [B]
        k: 当前的k值（随epoch线性变化）
        t: 教师损失均值（预计算）
    
    Returns:
        alpha: 自适应权重 (tensor) [B]
    """
    # 计算难度因子 d_f = e^(-k(x-t))
    # x越大（样本越难），k为正时，d_f越小
    d_f = torch.exp(-k * (teacher_losses - t))
    
    # 计算自适应权重 α = e^(-1/√d_f)
    # d_f越小（样本越难），α越大（更依赖蒸馏）
    alpha = torch.exp(-1.0 / torch.sqrt(d_f))
    
    # 确保α在合理范围内 [0.1, 0.9]
    alpha = torch.clamp(alpha, min=0.1, max=0.9)
    
    return alpha


def compute_teacher_loss_mean(model_t, train_loader, device):
    """
    计算教师模型在训练集上的平均损失，用于自适应知识蒸馏
    
    Args:
        model_t: 教师模型
        train_loader: 训练数据加载器
        device: 设备（cuda或cpu）
    
    Returns:
        mean_loss: 教师模型的平均损失
    """
    print("Computing teacher loss mean for adaptive KD...")
    model_t.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_loader, desc="Computing teacher loss mean")):
            # 解包数据（兼容不同数据加载器格式）
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            # 移动到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 教师模型前向传播
            outputs = model_t(inputs)
            
            # 如果是复值模型输出，取模
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 限制计算样本数（避免太慢）
            if total_samples >= 10000:
                break
    
    mean_loss = total_loss / total_samples
    print(f"Teacher loss mean: {mean_loss:.6f} (computed on {total_samples} samples)")
    
    return mean_loss


def create_kd_criterion(distill_method, config):
    """
    根据蒸馏方法创建对应的损失函数
    
    Args:
        distill_method: 蒸馏方法名称
        config: 配置字典
    
    Returns:
        criterion: 蒸馏损失函数
        需要的辅助模块列表（如ConvReg, Connector等）
    """
    from distiller_zoo import (
        DistillKL, DKDLoss, FSP, Attention, HintLoss,
        RKDLoss, NSTLoss, Similarity, PKT, Correlation,
        VIDLoss, ABLoss, FactorTransfer, KDSVD
    )
    from distiller_zoo.FitNet import ConvReg
    from distiller_zoo.CC import LinearEmbed
    from distiller_zoo.AB import Connector
    from distiller_zoo.FT import Paraphraser, Translator
    
    trainable_modules = []  # 需要训练的额外模块
    
    if distill_method == 'kd':
        # 标准KL散度蒸馏
        criterion = DistillKL(temperature=config.get('kd_temperature', 4.0))
    
    elif distill_method == 'dkd':
        # 解耦知识蒸馏
        criterion = DKDLoss(
            alpha=config.get('dkd_alpha', 1.0),
            beta=config.get('dkd_beta', 1.0),
            temperature=config.get('kd_temperature', 4.0)
        )
    
    elif distill_method == 'fsp':
        # 需要学生和教师的特征形状
        s_shapes = config.get('s_shapes', None)
        t_shapes = config.get('t_shapes', None)
        if s_shapes is None or t_shapes is None:
            raise ValueError("FSP requires 's_shapes' and 't_shapes' in config")
        criterion = FSP(s_shapes, t_shapes)
    
    elif distill_method == 'hint':
        # FitNet Hint损失
        criterion = HintLoss()
        # 需要ConvReg进行特征对齐
        hint_layer = config.get('hint_layer', 2)
        s_shape = config.get('s_shapes', [None, None, None])[hint_layer]
        t_shape = config.get('t_shapes', [None, None, None])[hint_layer]
        if s_shape is not None and t_shape is not None:
            regress_module = ConvReg(s_shape, t_shape)
            trainable_modules.append(regress_module)
    
    elif distill_method == 'attention':
        # 注意力迁移
        criterion = Attention(p=config.get('at_p', 2))
    
    elif distill_method == 'rkd':
        # 关系知识蒸馏
        criterion = RKDLoss(
            w_d=config.get('rkd_w_d', 25.0),
            w_a=config.get('rkd_w_a', 50.0)
        )
    
    elif distill_method == 'nst':
        # NST损失
        criterion = NSTLoss()
    
    elif distill_method == 'similarity':
        # 相似性保持
        criterion = Similarity()
    
    elif distill_method == 'pkt':
        # 概率知识迁移
        criterion = PKT()
    
    elif distill_method == 'correlation':
        # 相关一致性
        criterion = Correlation()
        # 需要嵌入层
        feat_dim = config.get('feat_dim', 128)
        s_dim = config.get('s_dim', 256)
        t_dim = config.get('t_dim', 512)
        embed_s = LinearEmbed(s_dim, feat_dim)
        embed_t = LinearEmbed(t_dim, feat_dim)
        trainable_modules.extend([embed_s, embed_t])
    
    elif distill_method == 'vid':
        # 变分信息蒸馏
        s_shapes = config.get('s_shapes', None)
        t_shapes = config.get('t_shapes', None)
        if s_shapes is None or t_shapes is None:
            raise ValueError("VID requires 's_shapes' and 't_shapes' in config")
        
        # 为每个中间层创建VID模块
        s_n = [s_shapes[i][1] for i in range(1, len(s_shapes) - 1)]
        t_n = [t_shapes[i][1] for i in range(1, len(t_shapes) - 1)]
        vid_modules = nn.ModuleList([
            VIDLoss(s, t, t) for s, t in zip(s_n, t_n)
        ])
        criterion = vid_modules
        trainable_modules.append(vid_modules)
    
    elif distill_method == 'abound':
        # 激活边界
        s_shapes = config.get('s_shapes', None)
        t_shapes = config.get('t_shapes', None)
        if s_shapes is None or t_shapes is None:
            raise ValueError("Abound requires 's_shapes' and 't_shapes' in config")
        
        connector = Connector(s_shapes[1:-1], t_shapes[1:-1])
        criterion = ABLoss(len(s_shapes) - 2)
        trainable_modules.append(connector)
    
    elif distill_method == 'factor':
        # 因子迁移
        s_shapes = config.get('s_shapes', None)
        t_shapes = config.get('t_shapes', None)
        if s_shapes is None or t_shapes is None:
            raise ValueError("Factor requires 's_shapes' and 't_shapes' in config")
        
        s_shape = s_shapes[-2]
        t_shape = t_shapes[-2]
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        criterion = FactorTransfer()
        trainable_modules.extend([translator, paraphraser])
    
    elif distill_method == 'kdsvd':
        # SVD知识蒸馏
        criterion = KDSVD()
    
    else:
        raise ValueError(f"Unknown distillation method: {distill_method}")
    
    return criterion, trainable_modules


def get_linear_schedule_k(epoch, total_epochs, k_plus, k_minus):
    """
    线性调度k值（用于自适应蒸馏）
    
    Args:
        epoch: 当前epoch（从1开始）
        total_epochs: 总epoch数
        k_plus: 初始k值（正数，简单样本权重大）
        k_minus: 最终k值（负数，困难样本权重大）
    
    Returns:
        当前的k值
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)  # [0, 1]
    current_k = k_plus + progress * (k_minus - k_plus)
    return current_k



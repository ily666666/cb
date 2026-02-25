"""
FedAWARE算法实现模块
实现基于动量梯度缓存和自适应加权聚合的联邦学习算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm


class MinNormSolver:
    """最小范数求解器 - Frank-Wolfe算法实现"""
    
    MAX_ITER = 1000
    STOP_CRIT = 1e-5
    
    def __init__(self):
        pass
    
    def find_min_norm_element_FW(self, vecs):
        """Frank-Wolfe算法求解最小范数元素
        
        Args:
            vecs: 向量列表 [torch.Tensor]
            
        Returns:
            sol: 解向量 (numpy array)
            val: 最优值
        """
        vecs = [vec.detach().cpu().numpy() for vec in vecs]
        sol_vec = np.zeros(len(vecs))
        
        # Frank-Wolfe迭代
        for iteration in range(self.MAX_ITER):
            # 计算梯度矩阵
            grad_mat = np.zeros((len(vecs), len(vecs)))
            for i in range(len(vecs)):
                for j in range(len(vecs)):
                    grad_mat[i,j] = np.dot(vecs[i], vecs[j])
            
            # 选择最优化方向
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))
            
            # 计算线搜索参数
            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]
            
            nc, nd = self._min_norm_element_from2(v1v1, v1v2, v2v2)
            
            # 更新解向量
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc
            
            # 检查收敛性
            if np.sum(np.abs(new_sol_vec - sol_vec)) < self.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
        
        return sol_vec, nd
    
    def _min_norm_element_from2(self, v1v1, v1v2, v2v2):
        """2D情况下的解析解"""
        if v1v2 >= v1v1:
            return 0.999, v1v1
        if v1v2 >= v2v2:
            return 0.001, v2v2
        
        gamma = -1.0 * (v1v2 - v2v2) / (v1v1 + v2v2 - 2*v1v2)
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost


class FeedbackSampler:
    """智能采样器"""
    
    def __init__(self, n, probs=None):
        """
        初始化智能采样器
        
        Args:
            n: 边侧总数
            probs: 初始采样概率
        """
        self.n = n
        self.p = probs if probs is not None else np.ones(n)/float(n)
        self.explore = list(range(n))
        random.shuffle(self.explore)
        self.explored = False
        self.last_sampled = None
    
    def sample(self, k, startup=0):
        """智能采样
        
        Args:
            k: 要采样的边侧数量
            startup: 是否为启动阶段
            
        Returns:
            选中的边侧索引数组
        """
        if startup:
            k = self.n
        
        # 初期探索阶段
        if len(self.explore) > 0 and not self.explored:
            sampled = self.explore[0:k]
            self.explore = list(set(self.explore) - set(sampled))
            if len(self.explore) == 0:
                self.explored = True
            return np.sort(np.array(sampled))
        
        # 反馈采样阶段
        nonzero_entries = sum(self.p > 0)
        if nonzero_entries > k:
            sampled = np.random.choice(self.n, k, p=self.p, replace=False)
        else:
            sampled = np.random.choice(self.n, nonzero_entries, p=self.p, replace=False)
            remains = np.setdiff1d(np.arange(self.n), sampled)
            uniform_sampled = np.random.choice(remains, k-nonzero_entries, replace=False)
            sampled = np.concatenate((sampled, uniform_sampled))
        
        self.last_sampled = (sampled, self.p[sampled])
        return np.sort(sampled)
    
    def update(self, probs, beta=1):
        """更新采样概率
        
        Args:
            probs: 新的概率向量
            beta: 更新参数
        """
        self.p = (1-beta)*self.p + beta*probs


class FedAWARE_Projector:
    """FedAWARE投影器 - 核心算法组件"""
    
    def __init__(self, n, alpha, model_parameters):
        """
        初始化FedAWARE投影器
        
        Args:
            n: 边侧总数
            alpha: 动量参数
            model_parameters: 模型参数（用于初始化动量缓存）
        """
        self.n = n
        self.alpha = alpha  # 动量参数
        self.solver = MinNormSolver()
        self.feedback = None
        
        # 为每个边侧维护动量梯度缓存
        if isinstance(model_parameters, list):
            # 如果是参数列表，每个参数都需要初始化动量缓存
            self.momentum = []
            for param in model_parameters:
                momentum_param = [torch.zeros_like(param) for _ in range(n)]
                self.momentum.append(momentum_param)
        else:
            # 如果是单个参数张量
            self.momentum = [torch.zeros_like(model_parameters) for _ in range(n)]
    
    def momentum_update(self, gradients, indices):
        """更新动量梯度缓存
        
        Args:
            gradients: 梯度列表
            indices: 对应的边侧索引
        """
        for grad, idx in zip(gradients, indices):
            if isinstance(self.momentum, list):
                # 参数列表情况
                for i, momentum_param in enumerate(self.momentum):
                    momentum_param[idx] = (1-self.alpha)*momentum_param[idx] + self.alpha*grad[i]
            else:
                # 单参数情况
                self.momentum[idx] = (1-self.alpha)*self.momentum[idx] + self.alpha*grad
    
    def compute_estimates(self):
        """计算自适应权重估计
        
        Returns:
            聚合后的梯度估计
        """
        # 梯度归一化
        if isinstance(self.momentum, list):
            # 参数列表情况
            norms = []
            for momentum_param in self.momentum:
                param_norms = [torch.norm(grad, p=2, dim=0).item() for grad in momentum_param]
                param_norms = np.array([1 if item==0 else item for item in param_norms])
                norms.append(param_norms)
            
            # 计算归一化的动量
            normalized_momentum = []
            for i in range(self.n):
                param_list = []
                for j, momentum_param in enumerate(self.momentum):
                    param_list.append(momentum_param[i] / norms[j][i])
                normalized_momentum.append(param_list)
        else:
            # 单参数情况
            norms = [torch.norm(grad, p=2, dim=0).item() for grad in self.momentum]
            norms = np.array([1 if item==0 else item for item in norms])
            normalized_momentum = [self.momentum[i]/norms[i] for i in range(self.n)]
        
        # 使用Frank-Wolfe求解最优权重
        sol = self.compute_lambda(normalized_momentum)
        self.feedback = sol
        
        # 聚合梯度
        gdm_estimates = self._fedavg_aggregate(normalized_momentum, sol)
        return gdm_estimates
    
    def compute_lambda(self, vectors):
        """Frank-Wolfe算法求解最优权重
        
        Args:
            vectors: 向量列表
            
        Returns:
            权重向量
        """
        # 将向量列表转换为向量
        if isinstance(vectors[0], list):
            # 参数列表情况 - 需要将每个边侧的参数打包
            flattened_vectors = []
            for edge_vectors in vectors:
                # 将边侧的参数列表展平为一个向量
                flattened = torch.cat([param.flatten() for param in edge_vectors])
                flattened_vectors.append(flattened)
            sol, val = self.solver.find_min_norm_element_FW(flattened_vectors)
        else:
            # 单参数情况
            sol, val = self.solver.find_min_norm_element_FW(vectors)
        
        print(f"FW solver - val {val} density {(sol>0).sum()}")
        assert abs(sol.sum() - 1) < 1e-5
        return sol
    
    def _fedavg_aggregate(self, gradients, weights):
        """FedAvg聚合函数
        
        Args:
            gradients: 梯度列表
            weights: 权重向量
            
        Returns:
            聚合后的梯度
        """
        if isinstance(gradients[0], list):
            # 参数列表情况
            aggregated = []
            for param_idx in range(len(gradients[0])):
                param_aggregated = torch.zeros_like(gradients[0][param_idx])
                for i, edge_grads in enumerate(gradients):
                    param_aggregated += weights[i] * edge_grads[param_idx]
                aggregated.append(param_aggregated)
            return aggregated
        else:
            # 单参数情况
            aggregated = torch.zeros_like(gradients[0])
            for i, grad in enumerate(gradients):
                aggregated += weights[i] * grad
            return aggregated


class Cloud_MomentumGradientCache:
    """云侧动量梯度缓存处理器"""
    
    def __init__(self, model, global_round, sample_ratio):
        """
        初始化云侧缓存处理器
        
        Args:
            model: 全局模型
            global_round: 当前全局轮次
            sample_ratio: 采样比例
        """
        self.model = model
        self.global_round = global_round
        self.sample_ratio = sample_ratio
        
        # 将在setup_optim中初始化的属性
        self.num_edges = None
        self.round_edges = None
        self.sampler = None
        self.lr = None
        self.alpha = None
        self.projector = None
        self.t = 0
        self.args = None
    
    def setup_optim(self, sampler, alpha, args):
        """初始化优化器
        
        Args:
            sampler: 智能采样器
            alpha: 动量参数
            args: 训练参数
        """
        self.n = self.num_edges
        self.num_to_sample = int(self.sample_ratio * self.n)
        self.round_edges = self.num_to_sample
        self.sampler = sampler
        self.lr = args.get('glr', 0.01)
        self.alpha = alpha
        self.args = args
        
        # 获取模型参数
        if hasattr(self.model, 'parameters'):
            model_params = list(self.model.parameters())
        else:
            model_params = self.model
        
        # 初始化动量缓存
        if isinstance(model_params, list) and len(model_params) > 0:
            self.momentum = []
            for param in model_params:
                momentum_param = [torch.zeros_like(param) for _ in range(self.n)]
                self.momentum.append(momentum_param)
        else:
            self.momentum = [torch.zeros_like(model_params) for _ in range(self.n)]
        
        # 创建FedAWARE投影器
        self.projector = FedAWARE_Projector(self.n, self.alpha, model_params)
    
    def momentum_update(self, gradients, indices):
        """更新动量缓存
        
        Args:
            gradients: 梯度列表
            indices: 边侧索引
        """
        self.projector.momentum_update(gradients, indices)
    
    def sample_edges(self, k, startup=0):
        """采样边侧
        
        Args:
            k: 要采样的边侧数量
            startup: 是否为启动阶段
            
        Returns:
            选中的边侧索引
        """
        edges = self.sampler.sample(k, startup)
        self.round_edges = len(edges)
        return edges
    
    def global_update(self, buffer):
        """全局模型更新
        
        Args:
            buffer: 边侧更新缓冲区 [(edge_id, model_parameters, metrics), ...]
            
        Returns:
            更新后的模型
        """
        if not buffer:
            return self.model
        
        # 提取模型参数和索引
        indices = [item[0] for item in buffer]
        gradient_list = []
        
        # 计算梯度
        for item in buffer:
            edge_id, model_params, _ = item
            if hasattr(self.model, 'parameters'):
                current_params = list(self.model.parameters())
            else:
                current_params = self.model
            
            if isinstance(current_params, list) and isinstance(model_params, list):
                # 参数列表情况
                gradients = [torch.sub(curr, edge_param) for curr, edge_param in zip(current_params, model_params)]
            else:
                # 单参数情况
                gradients = torch.sub(current_params, model_params)
            
            gradient_list.append(gradients)
        
        # FedAvg聚合作为基线
        weights = self.args.get('weights', [1.0/len(buffer)] * len(buffer))
        if isinstance(gradient_list[0], list):
            estimates = self._fedavg_aggregate(gradient_list, weights)
        else:
            estimates = self._fedavg_aggregate(gradient_list, weights)
        
        # 更新动量缓存
        self.projector.momentum_update(gradient_list, indices)
        
        # 如果已经探索完毕，使用自适应聚合
        if self.sampler.explored:
            estimates = self.projector.compute_estimates()
            self.sampler.update(self.projector.feedback)
            
            # 可选的投影机制
            if self.args.get('projection', False):
                d_fedavg = self.projector._fedavg_aggregate(self.projector.momentum, weights)
                estimates = self._project(d_fedavg, estimates)
        
        # 更新全局模型
        if hasattr(self.model, 'parameters'):
            current_params = list(self.model.parameters())
            if isinstance(estimates, list):
                new_params = [curr - self.lr * est for curr, est in zip(current_params, estimates)]
            else:
                new_params = [curr - self.lr * est for curr, est in zip(current_params, estimates)]
            
            # 更新模型参数
            for param, new_param in zip(self.model.parameters(), new_params):
                param.data.copy_(new_param.data)
        else:
            if isinstance(estimates, list):
                new_params = [curr - self.lr * est for curr, est in zip(self.model, estimates)]
            else:
                new_params = [curr - self.lr * est for curr, est in zip(self.model, estimates)]
            self.model = new_params
        
        self.t += 1
        return self.model
    
    def _fedavg_aggregate(self, gradients, weights):
        """FedAvg聚合"""
        if isinstance(gradients[0], list):
            # 参数列表情况
            aggregated = []
            for param_idx in range(len(gradients[0])):
                param_aggregated = torch.zeros_like(gradients[0][param_idx])
                for i, edge_grads in enumerate(gradients):
                    param_aggregated += weights[i] * edge_grads[param_idx]
                aggregated.append(param_aggregated)
            return aggregated
        else:
            # 单参数情况
            aggregated = torch.zeros_like(gradients[0])
            for i, grad in enumerate(gradients):
                aggregated += weights[i] * grad
            return aggregated
    
    def _project(self, d_fedavg, estimates):
        """投影机制（可选实现）"""
        # 这里可以实现投影机制，暂时返回estimates
        return estimates


class FedAvgSerialEdgeTrainer:
    """FedAvg串行边侧训练器"""
    
    def __init__(self, model, dataset, device):
        """
        初始化边侧训练器
        
        Args:
            model: 边侧模型
            dataset: 数据集处理器
            device: 设备
        """
        self._model = model
        self.dataset = dataset
        self.device = device
        self.cuda = torch.cuda.is_available()
        self.args = None
        
        # 将在setup_optim中设置
        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.optimizer = None
        self.criterion = None
        self.cache = []
    
    def setup_optim(self, epochs, batch_size, lr, momentum, args=None):
        """设置优化器
        
        Args:
            epochs: 训练轮次
            batch_size: 批次大小
            lr: 学习率
            momentum: 动量
            args: 其他参数
        """
        self.args = args
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr, weight_decay=5e-4, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
    
    def set_model(self, model_parameters):
        """设置模型参数"""
        if isinstance(model_parameters, list):
            # 参数列表情况
            for param, model_param in zip(model_parameters, self._model.parameters()):
                param.requires_grad_(False)
                model_param.data.copy_(param.data)
        else:
            # 单参数情况
            model_parameters.requires_grad_(False)
            self._model.load_state_dict(model_parameters)
    
    @property
    def model_parameters(self):
        """获取模型参数"""
        if hasattr(self._model, 'parameters'):
            return list(self._model.parameters())
        else:
            return self._model
    
    def local_process(self, payload, id_list, t):
        """本地训练处理
        
        Args:
            payload: 模型参数载荷
            id_list: 边侧ID列表
            t: 当前轮次
            
        Returns:
            损失和准确率统计
        """
        from helper.model_utils import AverageMeter
        
        model_parameters = payload[0]
        loss_ = AverageMeter()
        acc_ = AverageMeter()
        
        for id in tqdm(id_list):
            dataset = self.dataset.get_dataset(id)
            if hasattr(self, 'get_heterogeneity'):
                self.batch_size, self.epochs = self.get_heterogeneity(self.args, len(dataset))
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, loss_, acc_)
            self.cache.append(pack)
        
        return loss_, acc_
    
    def train(self, model_parameters, train_loader, loss_, acc_):
        """本地训练
        
        Args:
            model_parameters: 模型参数
            train_loader: 训练数据加载器
            loss_: 损失统计
            acc_: 准确率统计
            
        Returns:
            训练后的模型参数
        """
        self.set_model(model_parameters)
        self._model.train()
        
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                
                output = self._model(data)
                loss = self.criterion(output, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                _, predicted = torch.max(output, 1)
                loss_.update(loss.item())
                acc_.update(torch.sum(predicted.eq(target)).item(), len(target))
        
        return [self.model_parameters]


class Aggregators:
    """聚合器工具类"""
    
    @staticmethod
    def fedavg_aggregate(gradients, weights):
        """FedAvg聚合"""
        if isinstance(gradients[0], list):
            # 参数列表情况
            aggregated = []
            for param_idx in range(len(gradients[0])):
                param_aggregated = torch.zeros_like(gradients[0][param_idx])
                for i, edge_grads in enumerate(gradients):
                    param_aggregated += weights[i] * edge_grads[param_idx]
                aggregated.append(param_aggregated)
            return aggregated
        else:
            # 单参数情况
            aggregated = torch.zeros_like(gradients[0])
            for i, grad in enumerate(gradients):
                aggregated += weights[i] * grad
            return aggregated
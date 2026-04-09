import math
from functools import partial

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class INQScheduler(object):
    """Handles the the weight partitioning and group-wise quantization stages
    of the incremental network quantization procedure.
    Args:
        optimizer (Optimizer): Wrapped optimizer (use inq.SGD).
        iterative_steps (list): accumulated portions of quantized weights.
        strategy ("random"|"pruning"): weight partition strategy, either random or pruning-inspired.
    Example:
        >>> optimizer = inq.SGD(...)
        >>> inq_scheduler = INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0])
        >>> for inq_step in range(4):
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         train(...)
        >>> inq_scheduler.step()
        >>> validate(...)
    """
    def __init__(self, optimizer, iterative_steps):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step should equal 1 in INQ.")

        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.idx = 0

        for group in self.optimizer.param_groups:
            group['ns'] = []
            if group['weight_bits'] is None:
                continue
            for p in group['params']:
                s = torch.max(torch.abs(p.data)).item()
                n_1 = math.floor(math.log((4*s)/3, 2))
                n_2 = int(n_1+ 2 - 2**(group['weight_bits']-1))
                group['ns'].append((n_2, n_1))

    def quantize(self):
        """Quantize the parameters handled by the optimizer."""
        for group in self.optimizer.param_groups:
            if group['weight_bits'] is None:
                continue
            for idx, p in enumerate(group['params']):
                T = group['Ts'][idx]
                ns = group['ns'][idx]
                min_exp, max_exp = ns[0], ns[1]

                # ==== 向量化张量计算，瞬间在 GPU 上完成 ====
                abs_weight = torch.abs(p.data)

                # 1. 计算属于哪个 2 的幂次区间 (等价于你的 for 循环边界划分)
                i = torch.floor(torch.log2(abs_weight / 0.75))
                i_clamped = torch.clamp(i, min=min_exp, max=max_exp)

                # 2. 赋予符号，变成具体的 2 的幂次值
                quantized = torch.sign(p.data) * (2.0 ** i_clamped)

                # 3. 处理极小值置零 (等价于你原代码中小于最小 beta 边界的情况)
                fully_quantized = torch.where(abs_weight < (0.5 * (2.0 ** min_exp)),
                                              torch.zeros_like(p.data),
                                              quantized)

                # 4. 根据 T 掩码 (T==0表示该权重被选中量化) 应用量化值
                p.data = torch.where(T == 0, fully_quantized, p.data)

    def quantize_weight(self, weight, n_1, n_2):
        """
        Quantize a single weight using the INQ quantization scheme.
        """
        alpha = 0
        beta = 2 ** n_1
        abs_weight = math.fabs(weight)
        quantized_weight = 0

        for i in range(n_1, n_2 + 1):
            if (abs_weight >= (alpha + beta) / 2) and abs_weight < (3*beta/2):
                quantized_weight = math.copysign(beta, weight)
            alpha = 2 ** i
            beta = 2 ** (i+1)
        return quantized_weight

    def step(self):
        """Performs weight partitioning and quantization
        """
        for group in self.optimizer.param_groups:
            if group['weight_bits'] is None:
                continue
            for idx, p in enumerate(group['params']):
                zeros = torch.zeros_like(p.data)
                ones = torch.ones_like(p.data)
                quantile = np.quantile(torch.abs(p.data.cpu()).numpy(), 1 - self.iterative_steps[self.idx])
                T = torch.where(torch.abs(p.data) >= quantile, zeros, ones)     # T中为0对于权值表示量化, 为1表示继续参与训练
                group['Ts'][idx] = T
        self.idx += 1
        self.quantize()

def reset_lr_scheduler(scheduler):
    """Reset the learning rate scheduler.
    INQ requires resetting the learning rate every iteration of the procedure.

    Example:
        >>> optimizer = inq.SGD(...)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)
        >>> inq_scheduler = INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
        >>> for inq_step in range(3):
        >>>     reset_lr_scheduler(scheduler)
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         scheduler.step()
        >>>         train(...)
        >>>         validate(...)
    """
    scheduler.base_lrs = list(map(lambda group: group['initial_lr'], scheduler.optimizer.param_groups))
    scheduler.step()


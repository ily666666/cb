import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch.nn.init as init
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from torchvision import datasets, transforms
import math
# import torchlex


# 复数激活函数类：此处选取 CReLU
class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, input):
        # 分离实部与虚部
        input_real = input.real
        input_imag = input.imag
        # 对实部和虚部分别做激活
        real_part = F.relu(input_real)
        imag_part = F.relu(input_imag)
        complex_tensor = torch.view_as_complex(torch.stack([real_part, imag_part], dim=-1))

        return complex_tensor


# # 定义复数卷积层
# class ComplexConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         super(ComplexConv2D, self).__init__()  # 调用父类的构造函数
#         self.in_channels = in_channels  # 输入通道数
#         self.out_channels = out_channels  # 输出通道数
#         self.kernel_size = _pair(kernel_size)  # 卷积核尺寸，确保是二维的
#         self.stride = _pair(stride)  # 卷积步长，确保是二维的
#         self.padding = _pair(padding)  # 填充，确保是二维的
#         self.dilation = _pair(dilation)  # 空洞卷积参数，确保是二维的
#         self.groups = groups  # 分组卷积参数
#         self.weight = Parameter(torch.Tensor(out_channels*2, in_channels // groups, *self.kernel_size))  # 实部权重
#         self.weight_imag = Parameter(torch.Tensor(out_channels*2, in_channels // groups, *self.kernel_size))  # 虚部权重
#         if bias:  # 是否添加偏置
#             self.bias = Parameter(torch.Tensor(out_channels*2))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()  # 初始化参数
#
#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))   # 使用Kaiming均匀初始化实部权重
#         init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))   # 使用Kaiming均匀初始化虚部权重
#         if self.bias is not None:  # 如果有偏置，则初始化偏置
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, input):
#         batch_size, in_channels, height, width = input.size()  # 获取输入的形状
#         weight_real = self.weight.view(self.out_channels, self.in_channels, *self.kernel_size)  # 重塑实部权重的形状
#         weight_imag = self.weight_imag.view(self.out_channels, self.in_channels, *self.kernel_size)  # 重塑虚部权重的形状
#
#         input_real = input.view(batch_size, in_channels, height, width)  # 重塑输入的形状
#         input_imag = torch.zeros_like(input_real)  # 创建与输入实部形状相同的零张量作为输入虚部
#
#         # 执行复数卷积操作
#         conv_real = F.conv2d(input_real, weight_real, self.bias, self.stride, self.padding, self.dilation, self.groups) - \
#                     F.conv2d(input_imag, weight_imag, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         conv_imag = F.conv2d(input_real, weight_imag, self.bias, self.stride, self.padding, self.dilation, self.groups) + \
#                     F.conv2d(input_imag, weight_real, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#         return torch.cat((conv_real, conv_imag), dim=1)  # 将实部和虚部的结果沿通道维度拼接，然后返回


# 复数卷积
class ComplexConv2d(nn.Module):
    """
    创建一个复值卷积层对象。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        stride (int, optional): 卷积步长，默认为1。
        padding: 填充
        groups (int, optional): 分组卷积参数，输入通道分组数，默认为1，当groups=1时，表示普通的卷积操作，
            每个输入通道与每个输出通道都进行卷积操作；而当groups大于1时，表示分组卷积，
            将输入通道分成groups组，每组的通道分别与卷积核的对应部分进行卷积操作，然后将每组的结果拼接在一起得到输出。
        dilation (int, optional): 膨胀率，空洞卷积参数，默认为1，没有间隔，不具有空洞卷积效果

    Returns:
        ComplexConv2d: 一个复值卷积层对象。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.kernel_size = _pair(kernel_size)  # 卷积核尺寸，确保是二维的
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = _pair(kernel_size)  # 卷积核尺寸，确保是二维的
        self.stride = _pair(stride)  # 卷积步长，确保是二维的
        self.padding = _pair(padding)  # 填充，确保是二维的
        self.dilation = _pair(dilation)  # 空洞卷积参数，确保是二维的
        self.groups = groups  # 分组卷积参数
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))  # 实部权重
        self.weight_imag = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))  # 虚部权重

        if bias:  # 是否添加偏置
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化参数


    def reset_parameters(self):

        # 使用Kaiming均匀初始化实部权重，为解决深度神经网络中的梯度消失和爆炸问题而设计的初始化方法。参数a=math.sqrt(5)是Kaiming初始化的一个参数，它与ReLU激活函数的负半轴的斜率有关。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))  # 使用Kaiming均匀初始化虚部权重
        if self.bias is not None:  # 如果有偏置，则初始化偏置
            # 计算了权重张量的输入单元数（fan-in）和输出单元数（fan-out）。在这里，我们只关心输入单元数，因为接下来的偏置初始化会用到它
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)  # 计算了均匀分布的上下界
            init.uniform_(self.bias, -bound, bound)   # 使用上面计算出的上下界来对偏置进行均匀初始化

    def forward(self, input):
        # print("ComplexConv2d输入类型:", input.dtype)
        # 分离实部与虚部
        input_r = input.real
        input_i = input.imag
        # 确保输入是4维张量
        if input_r.dim() == 3:
            input_r = input_r.unsqueeze(-1)
            input_i = input_i.unsqueeze(-1)
        # assert (input_r.size() == input_i.size())
        # print(f"input_r.shape = {input_r.shape}, input_i.shape = {input_i.shape}")
        # return self.conv_r(input_r) - self.conv_i(input_i), self.conv_r(input_i) + self.conv_i(input_r)
        batch_size, in_channels, height, width = input_r.size()  # 获取输入的形状
        # print(f"self.weight.shape = {self.weight.shape}")
        # print(f"self.weight_imag.shape = {self.weight_imag.shape}")

        weight_real = self.weight.view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)  # 重塑实部权重的形状
        weight_imag = self.weight_imag.view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)  # 重塑虚部权重的形状
        # print(f"weight_real.shape = {weight_real.shape}, weight_imag.shape = {weight_imag.shape}")

        # input_real = input.view(batch_size, in_channels, height, width)   # 重塑输入的形状
        # input_imag = torch.zeros_like(input_real)  # 创建与输入实部形状相同的零张量作为输入虚部

        # 执行复数卷积操作
        conv_real = F.conv2d(input_r, weight_real, self.bias, self.stride, self.padding, self.dilation, self.groups) - \
                    F.conv2d(input_i, weight_imag, self.bias, self.stride, self.padding, self.dilation, self.groups)
        conv_imag = F.conv2d(input_r, weight_imag, self.bias, self.stride, self.padding, self.dilation, self.groups) + \
                    F.conv2d(input_i, weight_real, self.bias, self.stride, self.padding, self.dilation, self.groups)

        complex_tensor = torch.view_as_complex(torch.stack([conv_real, conv_imag], dim=-1))

        return complex_tensor


# 复数线性层
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features  # 记录输入维度
        self.out_features = out_features  # 新增：记录输出维度
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        # 分离实部与虚部
        input_r = input.real
        input_i = input.imag
        # 对实部和虚部分别做线性映射
        real_part = self.fc_r(input_r) - self.fc_i(input_i)
        imag_part = self.fc_r(input_i) + self.fc_i(input_r)

        complex_tensor = torch.view_as_complex(torch.stack([real_part, imag_part], dim=-1))
        return complex_tensor


# 复数BatchNormalization，通过下面的操作，可以确保输出的均值为0，协方差为1，相关为0
class _ComplexBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, input):
        assert (input.real.size() == input.imag.size())
        assert (len(input.real.shape) == 4)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input.real.mean([0, 2, 3])
            mean_i = input.imag.mean([0, 2, 3])

            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input.real - mean_r[None, :, None, None]
            input_i = input.imag - mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

            input_r = input.real - mean[None, :, 0, None, None]
            input_i = input.imag - mean[None, :, 1, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i, \
                           Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 0, None, None], \
                               self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 1, None, None]

        complex_tensor = torch.view_as_complex(torch.stack([input_r, input_i], dim=-1))

        return complex_tensor


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # 实部和虚部的 BatchNorm1d 层
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

        if self.affine:
            # 如果启用了可学习的尺度和偏移参数，则定义实部和虚部的参数
            self.weight_real = nn.Parameter(torch.ones(num_features))
            self.bias_real = nn.Parameter(torch.zeros(num_features))
            self.weight_imag = nn.Parameter(torch.ones(num_features))
            self.bias_imag = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight_real', None)
            self.register_parameter('bias_real', None)
            self.register_parameter('weight_imag', None)
            self.register_parameter('bias_imag', None)

    def forward(self, input):
        # 分离实部和虚部
        input_real = input.real
        input_imag = input.imag

        # 对实部和虚部分别进行 Batch Normalization
        output_real = self.bn_real(input_real)
        output_imag = self.bn_imag(input_imag)

        if self.affine:
            # 如果启用了可学习的尺度和偏移参数，则应用于 Batch Normalization 结果
            output_real = output_real * self.weight_real + self.bias_real
            output_imag = output_imag * self.weight_imag + self.bias_imag

        # 合并实部和虚部
        output = torch.view_as_complex(torch.stack([output_real, output_imag], dim=-1))

        return output


# 定义复数Group Normalization， GN
class ComplexGroupNorm2d(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(ComplexGroupNorm2d, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        # 确保 num_groups 是 num_channels 的因子
        assert num_channels % num_groups == 0, "num_channels should be divisible by num_groups."

        # 定义实部和虚部的可学习尺度和偏移参数
        self.weight_real = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias_real = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.weight_imag = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias_imag = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        # # 定义实部和虚部的权重和偏置参数
        # self.weight_real = nn.Parameter(torch.ones(num_channels))
        # self.weight_imag = nn.Parameter(torch.ones(num_channels))
        # self.bias_real = nn.Parameter(torch.zeros(num_channels))
        # self.bias_imag = nn.Parameter(torch.zeros(num_channels))

    def forward(self, input):
        # 分离实部和虚部
        input_real = input.real
        input_imag = input.imag

        batch_size, num_channels, height, width = input_real.size()

        # 将通道分为组
        input_real = input_real.view(batch_size, self.num_groups, -1)
        input_imag = input_imag.view(batch_size, self.num_groups, -1)

        # 计算均值和方差
        mean_real = input_real.mean(dim=2, keepdim=True)
        var_real = input_real.var(dim=2, unbiased=False, keepdim=True)

        mean_imag = input_imag.mean(dim=2, keepdim=True)
        var_imag = input_imag.var(dim=2, unbiased=False, keepdim=True)

        # 标准化
        input_real = (input_real - mean_real) / (var_real + self.eps).sqrt()
        input_imag = (input_imag - mean_imag) / (var_imag + self.eps).sqrt()

        # 将数据变回原来的形状
        input_real = input_real.view(batch_size, num_channels, height, width)
        input_imag = input_imag.view(batch_size, num_channels, height, width)

        # 乘以可学习的尺度和加上可学习的偏移
        output_real = input_real * self.weight_real + self.bias_real
        output_imag = input_imag * self.weight_imag + self.bias_imag

        # 合并实部和虚部
        output = torch.view_as_complex(torch.stack([output_real, output_imag], dim=-1))

        return output


# class ComplexLayerNorm(nn.Module):
#     def __init__(self,
#                  epsilon=1e-4,  # 初始化参数：归一化过程中的epsilon，用于防止除以零
#                  axis=-1,  # 归一化操作的轴，默认为最后一个轴
#                  center=True,  # 是否进行中心化，默认为True
#                  scale=True,  # 是否进行缩放，默认为True
#                  beta_initializer='zeros',  # beta参数的初始化方法，默认为全零初始化
#                  gamma_diag_initializer=torch.sqrt,  # gamma对角元素的初始化方法，默认为平方根初始化
#                  gamma_off_initializer='zeros',  # gamma非对角元素的初始化方法，默认为全零初始化
#                  beta_regularizer=None,  # beta参数的正则化方法，默认为None
#                  gamma_diag_regularizer=None,  # gamma对角元素的正则化方法，默认为None
#                  gamma_off_regularizer=None,  # gamma非对角元素的正则化方法，默认为None
#                  beta_constraint=None,  # beta参数的约束条件，默认为None
#                  gamma_diag_constraint=None,  # gamma对角元素的约束条件，默认为None
#                  gamma_off_constraint=None):  # gamma非对角元素的约束条件，默认为None
#         super(ComplexLayerNorm, self).__init__()
#
#         self.supports_masking = True  # 是否支持掩码操作
#         self.epsilon = epsilon  # 初始化参数：归一化过程中的epsilon
#         self.axis = axis  # 归一化操作的轴
#         self.center = center  # 是否进行中心化
#         self.scale = scale  # 是否进行缩放
#         self.beta_initializer = beta_initializer  # beta参数的初始化方法
#         self.gamma_diag_initializer = gamma_diag_initializer  # gamma对角元素的初始化方法
#         self.gamma_off_initializer = gamma_off_initializer  # gamma非对角元素的初始化方法
#         self.beta_regularizer = beta_regularizer  # beta参数的正则化方法
#         self.gamma_diag_regularizer = gamma_diag_regularizer  # gamma对角元素的正则化方法
#         self.gamma_off_regularizer = gamma_off_regularizer  # gamma非对角元素的正则化方法
#         self.beta_constraint = beta_constraint  # beta参数的约束条件
#         self.gamma_diag_constraint = gamma_diag_constraint  # gamma对角元素的约束条件
#         self.gamma_off_constraint = gamma_off_constraint  # gamma非对角元素的约束条件
#
#     def build(self, input_shape):
#         dim = input_shape[self.axis]  # 归一化操作轴上的维度大小
#         if dim is None:  # 如果轴上的维度大小未定义，抛出异常
#             raise ValueError(f"Axis {self.axis} of input tensor should have a defined dimension "
#                              f"but the layer received an input with shape {input_shape}.")
#
#         gamma_shape = (input_shape[self.axis] // 2,)  # gamma参数的形状，实部和虚部维度大小一半
#
#         if self.scale:  # 如果进行缩放操作
#             # 初始化gamma_rr、gamma_ii、gamma_ri参数
#             self.gamma_rr = nn.Parameter(self.gamma_diag_initializer(torch.zeros(*gamma_shape)))
#             self.gamma_ii = nn.Parameter(self.gamma_diag_initializer(torch.zeros(*gamma_shape)))
#             self.gamma_ri = nn.Parameter(self.gamma_off_initializer(torch.zeros(*gamma_shape)))
#         else:
#             self.gamma_rr = None
#             self.gamma_ii = None
#             self.gamma_ri = None
#
#         if self.center:  # 如果进行中心化操作
#             # 初始化beta参数
#             self.beta = nn.Parameter(self.beta_initializer(torch.zeros(dim)))
#         else:
#             self.beta = None
#
#     def forward(self, inputs):
#         input_shape = inputs.shape  # 输入张量的形状
#         ndim = inputs.dim()  # 输入张量的维度数
#         reduction_axes = list(range(ndim))  # 归一化操作中需要减少的轴列表
#         del reduction_axes[self.axis]  # 删除归一化轴
#         del reduction_axes[0]  # 删除batch轴
#         input_dim = input_shape[self.axis] // 2  # 输入维度大小的一半（因为是复数，所以实部和虚部维度相同）
#
#         # 计算沿减少轴的均值
#         mu = torch.mean(inputs, dim=reduction_axes, keepdim=True)
#         broadcast_mu = mu.expand(input_shape)
#
#         if self.center:  # 如果进行中心化操作
#             # 中心化输入
#             input_centred = inputs - broadcast_mu
#         else:
#             input_centred = inputs
#
#         centred_squared = input_centred ** 2  # 输入中心化后的张量的平方
#
#         # 将中心化后的平方张量分割为实部和虚部
#         if ndim == 3 or (self.axis == 1 and ndim != 3) or ndim == 2:
#             centred_squared_real = centred_squared[:, :input_dim]
#             centred_squared_imag = centred_squared[:, input_dim:]
#             centred_real = input_centred[:, :input_dim]
#             centred_imag = input_centred[:, input_dim:]
#         elif ndim == 4 and self.axis == -1:
#             centred_squared_real = centred_squared[:, :, :, :input_dim]
#             centred_squared_imag = centred_squared[:, :, :, input_dim:]
#             centred_real = input_centred[:, :, :, :input_dim]
#             centred_imag = input_centred[:, :, :, input_dim:]
#         elif ndim == 5 and self.axis == -1:
#             centred_squared_real = centred_squared[:, :, :, :, :input_dim]
#             centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
#             centred_real = input_centred[:, :, :, :, :input_dim]
#             centred_imag = input_centred[:, :, :, :, input_dim:]
#         else:
#             raise ValueError("Incorrect Layernorm combination of axis and dimensions.")
#
#         if self.scale:  # 如果进行缩放操作
#             # 计算方差和协方差
#             Vrr = torch.mean(centred_squared_real, dim=reduction_axes, keepdim=True) + self.epsilon
#             Vii = torch.mean(centred_squared_imag, dim=reduction_axes, keepdim=True) + self.epsilon
#             Vri = torch.mean(centred_real * centred_imag, dim=reduction_axes, keepdim=True)
#         elif self.center:  # 如果进行中心化但不进行缩放
#             Vrr = None
#             Vii = None
#             Vri = None
#         else:
#             raise ValueError("Both scale and center in batchnorm are set to False.")
#
#         # 应用复数归一化
#         return complex_normalization(input_centred, Vrr, Vii, Vri,
#                                      self.beta, self.gamma_rr, self.gamma_ri,
#                                      self.gamma_ii, self.scale, self.center,
#                                      layernorm=True, axis=self.axis)
#
#     def get_config(self):
#         config = {
#             'axis': self.axis,
#             'epsilon': self.epsilon,
#             'center': self.center,
#             'scale': self.scale,
#             'beta_initializer': self.beta_initializer,
#             'gamma_diag_initializer': self.gamma_diag_initializer,
#             'gamma_off_initializer': self.gamma_off_initializer,
#             'beta_regularizer': self.beta_regularizer,
#             'gamma_diag_regularizer': self.gamma_diag_regularizer,
#             'gamma_off_regularizer': self.gamma_off_regularizer,
#             'beta_constraint': self.beta_constraint,
#             'gamma_diag_constraint': self.gamma_diag_constraint,
#             'gamma_off_constraint': self.gamma_off_constraint,
#         }
#         base_config = super(ComplexLayerNorm, self).get_config()  # 获取父类的配置信息
#         return {**base_config, **config}  # 返回合并后的配置信息


# def ComplexBN(input_centred, Vrr, Vii, Vri, beta,
#               gamma_rr, gamma_ri, gamma_ii, scale=True,
#               center=True, layernorm=False, axis=-1):
#     # 获取输入张量的维度
#     ndim = input_centred.dim()
#     # 获取输入张量的轴上的维度大小
#     input_dim = input_centred.shape[axis] // 2
#     # 如果启用了scale操作，计算广播形状
#     if scale:
#         gamma_broadcast_shape = [1] * ndim
#         gamma_broadcast_shape[axis] = input_dim
#     # 如果启用了center操作，计算广播形状
#     if center:
#         broadcast_beta_shape = [1] * ndim
#         broadcast_beta_shape[axis] = input_dim * 2
#
#     if scale:
#         # 对输入进行复数标准化
#         standardized_output = complex_standardization(
#             input_centred, Vrr, Vii, Vri,
#             layernorm,
#             axis=axis
#         )
#
#         # 进行标准化后的张量的缩放和位移操作
#         broadcast_gamma_rr = gamma_rr.view(gamma_broadcast_shape)
#         broadcast_gamma_ri = gamma_ri.view(gamma_broadcast_shape)
#         broadcast_gamma_ii = gamma_ii.view(gamma_broadcast_shape)
#
#         cat_gamma_4_real = torch.cat([broadcast_gamma_rr, broadcast_gamma_ii], dim=axis)
#         cat_gamma_4_imag = torch.cat([broadcast_gamma_ri, broadcast_gamma_ri], dim=axis)
#         if (axis == 1 and ndim != 3) or ndim == 2:
#             centred_real = standardized_output[:, :input_dim]
#             centred_imag = standardized_output[:, input_dim:]
#         elif ndim == 3:
#             centred_real = standardized_output[:, :, :input_dim]
#             centred_imag = standardized_output[:, :, input_dim:]
#         elif axis == -1 and ndim == 4:
#             centred_real = standardized_output[:, :, :, :input_dim]
#             centred_imag = standardized_output[:, :, :, input_dim:]
#         elif axis == -1 and ndim == 5:
#             centred_real = standardized_output[:, :, :, :, :input_dim]
#             centred_imag = standardized_output[:, :, :, :, input_dim:]
#         else:
#             raise ValueError(
#                 'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
#                 'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
#             )
#         rolled_standardized_output = torch.cat([centred_imag, centred_real], dim=axis)
#         if center:
#             # 如果启用了center操作，返回缩放和位移后的结果
#             broadcast_beta = beta.view(broadcast_beta_shape)
#             return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
#         else:
#             # 如果未启用center操作，返回仅缩放后的结果
#             return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
#     else:
#         if center:
#             # 如果未启用scale但启用了center操作，返回仅位移后的结果
#             broadcast_beta = beta.view(broadcast_beta_shape)
#             return input_centred + broadcast_beta
#         else:
#             # 如果未启用scale和center操作，返回原始输入张量
#             return input_centred


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(ComplexMaxPool2d, self).__init__()
        self.pooling_r = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.pooling_i = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input):
        # 分离实部和虚部
        input_real = input.real
        input_imag = input.imag

        input_r = self.pooling_r(input_real)
        input_i = self.pooling_i(input_imag)

        # 合并实部和虚部,构建复值
        output = torch.view_as_complex(torch.stack([input_r, input_i], dim=-1))

        return output




class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, height, width):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.pooling_r = torch.nn.AdaptiveAvgPool2d((height, width))
        self.pooling_i = torch.nn.AdaptiveAvgPool2d((height, width))

    def forward(self, input):
        # 分离实部和虚部
        input_real = input.real
        input_imag = input.imag

        input_r = self.pooling_r(input_real)
        input_i = self.pooling_i(input_imag)

        # 合并实部和虚部,构建复值
        output = torch.view_as_complex(torch.stack([input_r, input_i], dim=-1))

        return output


if __name__ == '__main__':
    # model = ComplexNet().to('cuda:0')
    # print(model)
    input_real = torch.rand(1024, 1, 32, 32)
    input_image = torch.rand(1024, 1, 32, 32)

    # print("input_real.shape=", input_real.shape)
    # print("input_image.shape=", input_image.shape)

    input_real = input_real.to('cuda:0')
    input_image = input_image.to('cuda:0')
    outputs = model(input_real, input_image)
    print("outputs.shape=", outputs.shape)
    print(outputs.shape)
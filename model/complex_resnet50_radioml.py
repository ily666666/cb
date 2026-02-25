import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch.nn.init as init
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Linear, BatchNorm2d
import math


class ComplexReLU(nn.Module):
    """复数ReLU激活函数"""
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


class ComplexConv2d(nn.Module):
    """复数卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # 卷积核相关参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.weight_imag = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Kaiming均匀初始化实部权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 分离实部与虚部
        input_r = input.real
        input_i = input.imag
        # 确保输入是4维张量
        if input_r.dim() == 3:
            input_r = input_r.unsqueeze(-1)
            input_i = input_i.unsqueeze(-1)
        
        batch_size, in_channels, height, width = input_r.size()

        weight_real = self.weight.view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        weight_imag = self.weight_imag.view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)

        # 执行复数卷积操作
        conv_real = F.conv2d(input_r, weight_real, self.bias, self.stride, self.padding, self.dilation, self.groups) - \
                    F.conv2d(input_i, weight_imag, self.bias, self.stride, self.padding, self.dilation, self.groups)
        conv_imag = F.conv2d(input_r, weight_imag, self.bias, self.stride, self.padding, self.dilation, self.groups) + \
                    F.conv2d(input_i, weight_real, self.bias, self.stride, self.padding, self.dilation, self.groups)

        complex_tensor = torch.view_as_complex(torch.stack([conv_real, conv_imag], dim=-1))
        return complex_tensor


class ComplexLinear(nn.Module):
    """复数线性层"""
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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


class _ComplexBatchNorm(nn.Module):
    """复数批归一化基类"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
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
    """复数批归一化2D"""
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
            Cri = self.running_covar[:, 2]

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


class ComplexMaxPool2d(nn.Module):
    """复数最大池化2D"""
    def __init__(self, kernel_size, stride, padding):
        super(ComplexMaxPool2d, self).__init__()
        self.pooling_r = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.pooling_i = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input):
        # 分离实部与虚部
        input_real = input.real
        input_imag = input.imag

        input_r = self.pooling_r(input_real)
        input_i = self.pooling_i(input_imag)

        # 合并实部和虚部，构建复值
        output = torch.view_as_complex(torch.stack([input_r, input_i], dim=-1))
        return output


class ComplexAdaptiveAvgPool2d(nn.Module):
    """复数自适应平均池化2D"""
    def __init__(self, height, width):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.pooling_r = torch.nn.AdaptiveAvgPool2d((height, width))
        self.pooling_i = torch.nn.AdaptiveAvgPool2d((height, width))

    def forward(self, input):
        # 分离实部与虚部
        input_real = input.real
        input_imag = input.imag

        input_r = self.pooling_r(input_real)
        input_i = self.pooling_i(input_imag)

        # 合并实部和虚部，构建复值
        output = torch.view_as_complex(torch.stack([input_r, input_i], dim=-1))
        return output


class Bottleneck(nn.Module):
    """ResNet50的Bottleneck块"""
    expansion = 4  # Bottleneck block的扩展因子是4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        # 第一个卷积层，降维
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        
        # 第二个卷积层，3x3卷积
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        
        # 第三个卷积层，升维
        self.conv3 = ComplexConv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = ComplexBatchNorm2d(out_channels * self.expansion)
        
        self.relu = ComplexReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 复数加法
        out_real = out.real + identity.real
        out_imag = out.imag + identity.imag
        out = torch.view_as_complex(torch.stack([out_real, out_imag], dim=-1))

        out += identity
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out


class ResNet50(nn.Module):
    """复数ResNet50模型"""
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = ComplexConv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm2d(64)
        self.relu = ComplexReLU()
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差块层 - ResNet50的配置是[3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 64, 3, stride=1)      # 3个Bottleneck块
        self.layer2 = self._make_layer(256, 128, 4, stride=2)    # 4个Bottleneck块
        self.layer3 = self._make_layer(512, 256, 6, stride=2)   # 6个Bottleneck块 (ResNet50的关键区别)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)   # 3个Bottleneck块
        
        # 分类层
        self.avgpool = ComplexAdaptiveAvgPool2d(height=1, width=1)
        self.fc = ComplexLinear(2048, num_classes)  # 2048 = 512 * expansion(4)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(self.in_channels, out_channels * Bottleneck.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm2d(out_channels * Bottleneck.expansion)
            )

        layers = []
        # 第一个块可能有下采样
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample, is_last=(blocks == 1)))
        self.in_channels = out_channels * Bottleneck.expansion
        # 后续块
        for i in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        # feat_m.append(self.conv1)
        # feat_m.append(self.bn1)
        # feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        # feat_m.append(self.layer4)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        # 初始卷积层处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 这里的x3是preact特征（未经过maxpool）
        f0 = x

        x = self.maxpool(x)
    
        x,f1_pre = self.layer1(x)  
        f1 = x
        x,f2_pre = self.layer2(x)  
        f2 = x
        x,f3_pre = self.layer3(x)  
        f3 = x
        x,f4_pre = self.layer4(x)  
        # f4 = x

        # 分类头处理
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f4 = x.unsqueeze(-1).unsqueeze(-1)  # 保持特征维度一致性

        x = self.fc(x)

        # 特征提取逻辑（适配蒸馏需求）
        if is_feat:
            if preact:
                # 若需要preact特征（激活前的特征），需在各层单独记录（根据实际需求调整）
                return [f0, f1_pre, f2_pre, f3_pre], x  # 示例：使用各层输出作为特征
            else:
                # 普通特征：各层输出张量
                return [f0, f1, f2, f3], x  # 特征列表 + 最终输出
        else:
            return x  # 仅返回分类输出


class CombinedModel(nn.Module):
    """组合模型：复数ResNet50 + 分类头"""
    def __init__(self, num_classes=11):
        super(CombinedModel, self).__init__()
        self.is_complex = True  # 复数模型标识
        print("Initializing CombinedModel...")
        print(f"Number of classes: {num_classes}")
        
        # 复值ResNet50输出1000维
        self.resnet50_model = ResNet50(num_classes=1000)
        
        # 转换层：1000维→21维（关键层）
        self.fc1 = ComplexLinear(1000, num_classes)
        
        print("CombinedModel initialized successfully.")

    def forward(self, x, is_feat=False, preact=False, return_features=False):
        # 处理输入为复值
        # 输入x的形状: (batch_size, 128) 复数张量
        x_real = x.real  # (batch_size, 128)
        x_imag = x.imag  # (batch_size, 128)
        # print(f"Input shape: {x_real.shape}")  # 调试：打印输入形状
        # print(f"Input dtype: {x_imag.shape}") #输入数据类型
        # 将(batch_size, 128)转换为(batch_size, 1, 8, 16)
        x_real = x_real.reshape(x_real.shape[0], 1, 8, 16)
        x_imag = x_imag.reshape(x_imag.shape[0], 1, 8, 16)
        out = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))
        
        # 获取ResNet50的输出和特征
        if is_feat:
            # 获取中间特征和1000维输出
            feat_resnet, out_resnet = self.resnet50_model(out, is_feat=True, preact=preact)
        else:
            # 仅获取1000维输出
            out_resnet = self.resnet50_model(out)
        
        # 调试：打印ResNet50输出维度
        # print(f"ResNet输出维度: {out_resnet.shape}")  # 应显示(bs, 1000)
        
        # 展平复数输出（保持实部虚部结构）
        out_resnet_real = torch.flatten(out_resnet.real, start_dim=1)  # 实部展平
        out_resnet_imag = torch.flatten(out_resnet.imag, start_dim=1)  # 虚部展平
        out_flat = torch.view_as_complex(torch.stack([out_resnet_real, out_resnet_imag], dim=-1))
        
        # 关键：通过fc1将1000维转为21维
        out_100 = self.fc1(out_flat)  # 形状：(batch_size, 21)
        
        # 调试：打印转换后维度
        # print(f"fc1输出维度: {out_21.shape}")  # 应显示(bs, 21)
        
        # 计算模值（转为实值）
        output = torch.abs(out_100)
        features = output  # 特征直接使用模值
        
        # 根据参数返回结果
        if is_feat:
            # 关键修改：将复数特征转为实值模值
            feat_resnet_real = [torch.abs(f) for f in feat_resnet]  # 对每个中间特征取模
            return feat_resnet_real, output  # 返回实值特征
        elif return_features:
            return output, features
        else:
            return output


if __name__ == '__main__':
    # 测试模型
    model = CombinedModel(num_classes=11)
    print(model)
    
    # 创建测试输入
    batch_size = 2
    input_real = torch.randn(batch_size, 1, 8, 16)
    input_imag = torch.randn(batch_size, 1, 8, 16)
    
    # 构建复数输入
    x = torch.view_as_complex(torch.stack([input_real, input_imag], dim=-1))
    
    # 前向传播测试
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # 特征提取测试
    features, output = model(x, is_feat=True)
    print(f"Features shape: {[f.shape for f in features]}")
    print(f"Output shape: {output.shape}")
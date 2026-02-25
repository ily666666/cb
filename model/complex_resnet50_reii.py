import torch
import torch.nn as nn
import sys

import model.CVNN as CVNN

sys.path.append("..")



class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck block的扩展因子是4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 第一个卷积层，降维
        self.conv1 = CVNN.ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = CVNN.ComplexBatchNorm2d(out_channels)
        
        # 第二个卷积层，3x3卷积
        self.conv2 = CVNN.ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = CVNN.ComplexBatchNorm2d(out_channels)
        
        # 第三个卷积层，升维
        self.conv3 = CVNN.ComplexConv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = CVNN.ComplexBatchNorm2d(out_channels * self.expansion)
        
        self.relu = CVNN.ComplexReLU()
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

        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = CVNN.ComplexConv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = CVNN.ComplexBatchNorm2d(64)
        self.relu = CVNN.ComplexReLU()
        self.maxpool = CVNN.ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差块层 - ResNet50的配置是[3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 64, 3, stride=1)      # 3个Bottleneck块
        self.layer2 = self._make_layer(256, 128, 4, stride=2)    # 4个Bottleneck块
        self.layer3 = self._make_layer(512, 256, 6, stride=2)   # 6个Bottleneck块 (ResNet50的关键区别)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)   # 3个Bottleneck块
        
        # 分类层
        self.avgpool = CVNN.ComplexAdaptiveAvgPool2d(height=1, width=1)
        self.fc = CVNN.ComplexLinear(2048, num_classes)  # 2048 = 512 * expansion(4)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                CVNN.ComplexConv2d(self.in_channels, out_channels * Bottleneck.expansion, 
                                  kernel_size=1, stride=stride, bias=False),
                CVNN.ComplexBatchNorm2d(out_channels * Bottleneck.expansion)
            )

        layers = []
        # 第一个块可能有下采样
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        # 后续块
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        # 初始卷积层处理
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)  # 这里的x3是preact特征（未经过maxpool）
        x4 = self.maxpool(x3)

        # 修正：移除错误的多值解包，layer1/2/3仅返回1个张量
        x5 = self.layer1(x4)  # layer1输出 -> x5
        x6 = self.layer2(x5)  # layer2输出 -> x6
        x7 = self.layer3(x6)  # layer3输出 -> x7
        x8 = self.layer4(x7)  # layer4输出 -> x8

        # 分类头处理
        x9 = self.avgpool(x8)
        x10 = torch.flatten(x9, 1)
        x11 = self.fc(x10)

        # 调试：打印各特征维度
        # if is_feat:
        #     print(f"教师特征维度: {[x3.shape[1], x5.shape[1], x6.shape[1], x7.shape[1], x8.shape[1]]}")
        #     # 应显示 [64, 256, 512, 1024, 2048]

        # 特征提取逻辑（适配蒸馏需求）
        if is_feat:
            if preact:
                # 若需要preact特征（激活前的特征），需在各层单独记录（根据实际需求调整）
                return [x3, x5, x6, x7, x8], x11  # 示例：使用各层输出作为特征
            else:
                # 普通特征：各层输出张量
                return [x3, x5, x6, x7, x8], x11  # 特征列表 + 最终输出
        else:
            return x11  # 仅返回分类输出


# 模型定义
class CombinedModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CombinedModel, self).__init__()
        self.is_complex = True  # 复数模型标识
        print("Initializing CombinedModel for REII dataset...")
        print(f"Number of classes: {num_classes}")
        
        # 复值ResNet50输出1000维
        self.resnet50_model = ResNet50(num_classes=1000)
        
        # 转换层：1000维→3维（REII数据集有3个类别）
        self.fc1 = CVNN.ComplexLinear(1000, num_classes).cuda()
        
        print("CombinedModel initialized successfully.")

    def forward(self, x, is_feat=False, preact=False, return_features=False):
        # 处理输入为复值
        # 输入x的形状: (batch_size, 2000) 复数张量
        x_real = x.real  # (batch_size, 2000)
        x_imag = x.imag  # (batch_size, 2000)
        # 将(batch_size, 2000)转换为(batch_size, 1, 40, 50)
        x_real = x_real.reshape(x_real.shape[0], 1, 40, 50)
        x_imag = x_imag.reshape(x_imag.shape[0], 1, 40, 50)
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
        
        # 关键：通过fc1将1000维转为3维
        out_3 = self.fc1(out_flat)  # 形状：(batch_size, 3)
        
        # 调试：打印转换后维度
        # print(f"fc1输出维度: {out_3.shape}")  # 应显示(bs, 3)
        
        # 计算模值（转为实值）
        output = torch.abs(out_3)
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
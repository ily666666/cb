"""
Complex-valued ResNet50 with Attention for Link11 Dataset
Extracted from training file with TimeFreqAttention mechanism
Input: (batch_size, 1024) complex tensor
Reshape: (batch_size, 1, 32, 32) - maintains complex values
Output: (batch_size, num_classes) predictions
"""
import torch
import torch.nn as nn
from .CVNN import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexMaxPool2d, ComplexAdaptiveAvgPool2d, ComplexLinear


class TimeFreqAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(TimeFreqAttention, self).__init__()
        self.channels = channels
        
        # 通道注意力（SE-style）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力（基于通道压缩后的特征图）
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_complex):
        # 检查输入并进行数值清理
        if torch.isnan(x_complex).any() or torch.isinf(x_complex).any():
            print("⚠️ NaN/Inf in input to TimeFreqAttention - replacing with zeros")
            x_complex = torch.nan_to_num(x_complex, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 限制输入幅度，防止数值爆炸
        x_complex = torch.clamp(x_complex.real, min=-1e6, max=1e6) + \
                    1j * torch.clamp(x_complex.imag, min=-1e6, max=1e6)
        
        # 安全取模，添加小的epsilon防止除零
        x_abs = torch.abs(x_complex) + 1e-8  # (B, C, H, W)

        # Channel Attention
        b, c, _, _ = x_abs.size()
        y = self.avg_pool(x_abs).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # 检查通道注意力权重
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("⚠️ NaN/Inf in channel attention weights - replacing")
            y = torch.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0)
        
        x_ca = x_abs * y

        # Spatial Attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.sigmoid(self.conv(concat))
        
        # 检查空间注意力权重
        if torch.isnan(spatial_attn).any() or torch.isinf(spatial_attn).any():
            print("⚠️ NaN/Inf in spatial attention weights - replacing")
            spatial_attn = torch.nan_to_num(spatial_attn, nan=0.5, posinf=1.0, neginf=0.0)
        
        attended_magnitude = x_ca * spatial_attn  # (B, C, H, W)
        # 更严格的幅度限制
        attended_magnitude = torch.clamp(attended_magnitude, min=1e-8, max=10.0)

        # 安全重建复数：用 atan2 避免 NaN，添加epsilon
        x_real_safe = x_complex.real + 1e-8
        x_imag_safe = x_complex.imag + 1e-8
        phase = torch.atan2(x_imag_safe, x_real_safe)
        
        # 检查相位
        if torch.isnan(phase).any() or torch.isinf(phase).any():
            print("⚠️ NaN/Inf in phase calculation - replacing")
            phase = torch.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
        
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        attended_real = attended_magnitude * cos_phase
        attended_imag = attended_magnitude * sin_phase
        
        # 最终检查和限制
        attended_real = torch.clamp(attended_real, min=-1e6, max=1e6)
        attended_imag = torch.clamp(attended_imag, min=-1e6, max=1e6)
        
        attended_complex = torch.view_as_complex(torch.stack([attended_real, attended_imag], dim=-1))
        
        # 最终输出检查
        if torch.isnan(attended_complex).any() or torch.isinf(attended_complex).any():
            print("⚠️ NaN/Inf in output - replacing with input")
            return x_complex
        
        return attended_complex


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        
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

        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = ComplexConv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm2d(64)
        self.relu = ComplexReLU()
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差块层 - ResNet50的配置是[3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # 时频注意力模块
        self.attn1 = TimeFreqAttention(channels=256)   # layer1 输出通道 = 64*4=256
        self.attn2 = TimeFreqAttention(channels=512)   # layer2 → 128*4=512
        self.attn3 = TimeFreqAttention(channels=1024)  # layer3 → 256*4=1024
        self.attn4 = TimeFreqAttention(channels=2048)  # layer4 → 512*4=2048

        # 分类层
        self.avgpool = ComplexAdaptiveAvgPool2d(height=1, width=1)
        self.fc = ComplexLinear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(self.in_channels, out_channels * Bottleneck.expansion, 
                                  kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm2d(out_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample, is_last=(blocks == 1)))
        self.in_channels = out_channels * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        # 初始卷积层处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x = self.maxpool(x)
    
        x, f1_pre = self.layer1(x)
        x = self.attn1(x)
        f1 = x
        
        x, f2_pre = self.layer2(x)
        x = self.attn2(x)
        f2 = x
        
        x, f3_pre = self.layer3(x)
        x = self.attn3(x)
        f3 = x
        
        x, f4_pre = self.layer4(x)
        x = self.attn4(x)

        # 分类头处理
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f4 = x.unsqueeze(-1).unsqueeze(-1)

        x = self.fc(x)

        # 特征提取逻辑
        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre], x
            else:
                return [f0, f1, f2, f3], x
        else:
            return x


class CombinedModel(nn.Module):
    """
    Two-stage classifier with attention mechanism
    Stage 1: ResNet50 outputs 1000 dimensions
    Stage 2: fc1 maps 1000 → num_classes
    """
    def __init__(self, num_classes=7):
        super(CombinedModel, self).__init__()
        self.is_complex = True
        print("Initializing Complex ResNet50 with Attention for Link11 Dataset...")
        print(f"Number of classes: {num_classes}")
        
        # 复值ResNet50输出1000维
        self.resnet50_model = ResNet50(num_classes=1000)
        
        # 转换层：1000维→目标类别数
        self.fc1 = ComplexLinear(1000, num_classes)
        
        print("Complex ResNet50 with Attention initialized successfully.")

    def forward(self, x, is_feat=False, preact=False, return_features=False):
        # Input: (batch_size, 1024) complex tensor
        # Reshape to: (batch_size, 1, 32, 32)
        out = x.reshape(x.shape[0], 1, 32, 32)
        
        # 获取ResNet50的输出和特征
        if is_feat:
            feat_resnet, out_resnet = self.resnet50_model(out, is_feat=True, preact=preact)
        else:
            out_resnet = self.resnet50_model(out)
        
        # 展平复数输出（保持实部虚部结构）
        out_resnet_real = torch.flatten(out_resnet.real, start_dim=1)
        out_resnet_imag = torch.flatten(out_resnet.imag, start_dim=1)
        out_flat = torch.view_as_complex(torch.stack([out_resnet_real, out_resnet_imag], dim=-1))
        
        # 通过fc1将1000维转为目标类别数
        out_final = self.fc1(out_flat)
        
        # 计算模值（转为实值）
        output = torch.abs(out_final)
        features = output
        
        # 根据参数返回结果
        if is_feat:
            feat_resnet_real = [torch.abs(f) for f in feat_resnet]
            return feat_resnet_real, output
        elif return_features:
            return output, features
        else:
            return output


if __name__ == '__main__':
    # Test the model
    model = CombinedModel(num_classes=7)
    
    # Create dummy input
    dummy_input = torch.randn(4, 1024, dtype=torch.complex64)
    
    # Forward pass
    output = model(dummy_input)
    print(f"\nModel test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with is_feat=True
    features, output = model(dummy_input, is_feat=True)
    print(f"\nWith is_feat=True:")
    print(f"Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Feature {i}: {feat.shape}")
    print(f"Output shape: {output.shape}")

import torch
import torch.nn as nn

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out


  # 注意：需要将BasicBlock替换为Bottleneck块（与教师模型一致）
class Bottleneck(nn.Module):
    expansion = 4  # 扩展因子，输出通道=out_channels×4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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
        out += identity
        out = self.relu(out)
        return out

class ResNet20Real(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet20Real, self).__init__()
        self.in_channels = 64  # 改为64，匹配教师模型初始通道
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 模仿教师初始层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 修正残差块通道数配置：
        # layer1：输入64通道 → 输出256通道（64*4，匹配Bottleneck扩展因子）
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=2, stride=1)
        # layer2：输入256通道 → 输出512通道（128*4）
        self.layer2 = self._make_layer(in_channels=256, out_channels=128, blocks=2, stride=2)
        # layer3：输入512通道 → 输出1024通道（256*4）
        self.layer3 = self._make_layer(in_channels=512, out_channels=256, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.intermediate_fc = nn.Linear(1024, 1000)  # 1024→1000
        self.fc = nn.Linear(1000, num_classes)  # 1000→21

        # =================== 新增：频域重建头 ===================
        # 先估算 layer3 输出尺寸（输入 2x128 → 经过 stride=2 三次 → 128/(2^3)=16）
        # 所以 layer3 输出 H=W=16（因为 128 -> 64 -> 32 -> 16）
        # self.reconstruction_head = nn.Sequential(
        # nn.Conv2d(1024, 256, kernel_size=1),  # 1x1 conv to reduce channels
        # nn.ReLU(inplace=True),
        # nn.Upsample(size=(8, 16), mode='bilinear', align_corners=False),  # 直接上采样到目标尺寸
        # nn.Conv2d(256, 2, kernel_size=3, padding=1)  # 输出 2 通道
        # )
        # =====================================================


    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        # 当步长≠1或输入通道≠输出通道×扩展因子时，需要下采样
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:  # 注意这里使用Bottleneck的expansion=4
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        layers = []
        # 第一个块：处理通道数转换和步长
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample))
        # 更新当前通道数（输出通道×扩展因子）
        self.in_channels = out_channels * Bottleneck.expansion
        # 后续块：输入输出通道数一致
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False, return_reconstruction=False):
        # Adjust input shape to [batch_size, channels, height, width]
        # 输入x的形状: (batch_size, 128) 复数张量
        # 提取实部并reshape为(batch_size, 2, 8, 16)
        x_real = x.real  # (batch_size, 128)
        x_imag = x.imag  # (batch_size, 128)
        x_real = x_real.reshape(x_real.shape[0], 1, 8, 16)
        x_imag = x_imag.reshape(x_imag.shape[0], 1, 8, 16)
        x = torch.cat([x_real, x_imag], dim=1)  # (batch_size, 2, 8, 16)

        # print(f"输入图像维度: {x.shape}")  # 调试：打印输入图像维度
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)  # 256维
        x6 = self.layer2(x5)  # 512维
        x7 = self.layer3(x6)  # 1024维

        x8 = self.avgpool(x7)
        x9 = torch.flatten(x8, 1)  # 1024维
        x_intermediate = self.intermediate_fc(x9)  # 1000维
        # x_intermediate_un = x_intermediate.unsqueeze(-1).unsqueeze(-1)  # 调整形状以匹配全连接层输入要求

        x10 = self.fc(x_intermediate)  # 21维

        # if return_reconstruction:
        #     # 重建路径：从 layer3 输出重建 (2, 8, 16)
        #     rec_2d = self.reconstruction_head(x7)  # (B, 2, 8, 16)
        #     return rec_2d

        if is_feat:
            return [x3, x5, x6, x7], x10
        else:
            return x10
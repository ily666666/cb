import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


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

class ResNet9Real(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet9Real, self).__init__()
        self.in_channels = 64  # 改为64，匹配教师模型初始通道
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 模仿教师初始层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet9: 使用 Bottleneck (expansion=4) 但保持每 stage 1 个 block (浅层)
        # 通道数对齐 ResNet50: [256, 512, 1024]
        
        # layer1：Arg out=64 -> 实际输出 64*4 = 256
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=1, stride=1)
        # layer2：Arg out=128 -> 实际输出 128*4 = 512
        self.layer2 = self._make_layer(in_channels=256, out_channels=128, blocks=1, stride=2)
        # layer3：Arg out=256 -> 实际输出 256*4 = 1024
        self.layer3 = self._make_layer(in_channels=512, out_channels=256, blocks=1, stride=2)
        
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
        # # =====================================================


    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        # 当步长≠1或输入通道≠输出通道×扩展因子时，需要下采样
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:  # 使用Bottleneck
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
        # 处理复数输入（针对Link等数据集）：将实部/虚部转换为2个通道
        if torch.is_complex(x):
            x = torch.view_as_real(x)
            # 将最后一个维度（实部/虚部 2）移到 dim 1
            # 例如 (B, 1, L, 2) -> (B, 2, 1, L)
            dims = list(range(x.dim()))
            new_order = [0, len(dims)-1] + dims[1:-1]
            x = x.permute(*new_order).contiguous()

        # 动态适配输入维度进行reshape
        batch_size = x.size(0)
        num_elements = x.view(batch_size, -1).size(1)
        
        if num_elements == 1024:
             # 如果输入大小为 1024，可能是单通道 32x32，或者是 2x16x32 等
             # 尝试 reshape 为 [B, 1, 32, 32]
             x = x.view(batch_size, 1, 32, 32)
        elif num_elements == 2048:
             # 如果输入大小为 2048，则 reshape 为 [B, 2, 32, 32]
             x = x.view(batch_size, 2, 32, 32)
        else:
             # 其他情况尝试按照 num_elements 自动推导，或者报错
             # 这里假设我们尽量适配成 Cx32x32 的形式
             channels = num_elements // (32 * 32)
             if channels * 32 * 32 == num_elements:
                 x = x.view(batch_size, channels, 32, 32)
             else:
                  raise ValueError(f"Input size {num_elements} per sample cannot be reshaped to (C, 32, 32).")


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
        

# import torch
# import torch.nn as nn

# class Bottleneck(nn.Module):
#     expansion = 4  # 扩展因子，输出通道=out_channels×4
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out

# class ResNet20Real(nn.Module):
#     def __init__(self, num_classes=7):
#         super(ResNet20Real, self).__init__()
#         self.in_channels = 64  # 改为64，匹配教师模型初始通道
#         self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 模仿教师初始层
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # 修正残差块通道数配置：
#         # layer1：输入64通道 → 输出256通道（64*4，匹配Bottleneck扩展因子）
#         self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=2, stride=1)
#         # layer2：输入256通道 → 输出512通道（128*4）
#         self.layer2 = self._make_layer(in_channels=256, out_channels=128, blocks=2, stride=2)
#         # layer3：输入512通道 → 输出1024通道（256*4）
#         self.layer3 = self._make_layer(in_channels=512, out_channels=256, blocks=2, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 已有自适应池化，无需修改
#         self.intermediate_fc = nn.Linear(1024, 1000)  # 1024→1000
#         self.fc = nn.Linear(1000, num_classes)  # 注意：原代码注释是21维，num_classes默认6，需保持一致

#     def _make_layer(self, in_channels, out_channels, blocks, stride):
#         downsample = None
#         # 当步长≠1或输入通道≠输出通道×扩展因子时，需要下采样
#         if stride != 1 or in_channels != out_channels * Bottleneck.expansion:  # 注意这里使用Bottleneck的expansion=4
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, 
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * Bottleneck.expansion)
#             )
#         layers = []
#         # 第一个块：处理通道数转换和步长
#         layers.append(Bottleneck(in_channels, out_channels, stride, downsample))
#         # 更新当前通道数（输出通道×扩展因子）
#         self.in_channels = out_channels * Bottleneck.expansion
#         # 后续块：输入输出通道数一致
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self.in_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x, is_feat=False, preact=False, return_reconstruction=False):
#         # 处理复数输入（针对Link等数据集）：将实部/虚部转换为2个通道
#         if torch.is_complex(x):
#             x = torch.view_as_real(x)
#             # 将最后一个维度（实部/虚部 2）移到 dim 1
#             # 例如 (B, 1, L, 2) -> (B, 2, 1, L)
#             dims = list(range(x.dim()))
#             new_order = [0, len(dims)-1] + dims[1:-1]
#             x = x.permute(*new_order).contiguous()

#         # 🔧 核心修改1：替换硬编码的view，动态适配输入形状（解决剪枝后元素数不匹配问题）
#         batch_size = x.size(0)
#         target_channels = 2
#         target_h, target_w = 32, 32
#         expected_total_elems = batch_size * target_channels * target_h * target_w
        
#         # 校验输入元素数是否匹配，避免shape错误
#         if x.numel() != expected_total_elems:
#             # 可选：自动调整（如果输入维度变化），或抛出明确错误
#             raise ValueError(
#                 f"输入张量元素数错误！期望 {expected_total_elems} (batch={batch_size}, 2×32×32), "
#                 f"实际 {x.numel()}。请检查输入数据形状。"
#             )
#         # 重塑输入（保留原有逻辑，但增加校验）
#         x = x.view(batch_size, target_channels, target_h, target_w)

#         # 以下部分无需修改（已有自适应池化，兼容通道剪枝后的形状变化）
#         x1 = self.conv1(x)
#         x2 = self.bn1(x1)
#         x3 = self.relu(x2)
#         x4 = self.maxpool(x3)

#         x5 = self.layer1(x4)  # 256维（剪枝后通道数会变化，自适应池化可兼容）
#         x6 = self.layer2(x5)  # 512维（剪枝后通道数变化）
#         x7 = self.layer3(x6)  # 1024维（剪枝后通道数变化）

#         x8 = self.avgpool(x7)  # 自适应池化→(batch, C, 1, 1)，C为剪枝后的通道数
#         x9 = torch.flatten(x8, 1)  # 展平→(batch, C)，兼容任意C
        
#         # 🔧 核心修改2：动态适配intermediate_fc的输入维度（剪枝后x9的维度≠1024）
#         # 原intermediate_fc是Linear(1024, 1000)，剪枝后x9的维度变为剪枝后的通道数，需重新定义
#         # 方案：初始化时不固定intermediate_fc，或在forward中动态调整
#         # 更简单的方案：替换固定的intermediate_fc为动态适配的线性层
#         if hasattr(self, 'dynamic_intermediate_fc') and self.dynamic_intermediate_fc.in_features != x9.size(1):
#             # 动态重建线性层，匹配剪枝后的输入维度
#             self.dynamic_intermediate_fc = nn.Linear(x9.size(1), 1000).to(x9.device)
#         elif not hasattr(self, 'dynamic_intermediate_fc'):
#             # 首次运行，初始化动态线性层（兼容原始1024维）
#             self.dynamic_intermediate_fc = nn.Linear(x9.size(1), 1000).to(x9.device)
#             # 复制原intermediate_fc的权重（如果是原始维度）
#             if x9.size(1) == 1024:
#                 self.dynamic_intermediate_fc.weight.data = self.intermediate_fc.weight.data
#                 self.dynamic_intermediate_fc.bias.data = self.intermediate_fc.bias.data
        
#         # 使用动态线性层替代原intermediate_fc
#         x_intermediate = self.dynamic_intermediate_fc(x9)  # (batch, 1000)
#         x10 = self.fc(x_intermediate)  # (batch, num_classes)

#         # 保留原有返回逻辑
#         if is_feat:
#             return [x3, x5, x6, x7], x10
#         else:
#             return x10
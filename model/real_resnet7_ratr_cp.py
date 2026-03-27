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
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        c1: int | None = None,
        c2: int | None = None,
    ):
        super(Bottleneck, self).__init__()
        # out_channels 表示该 stage 的“planes”，最终输出通道固定为 planes*expansion
        planes = out_channels
        c1 = planes if c1 is None else int(c1)
        c2 = planes if c2 is None else int(c2)
        if c1 < 1 or c2 < 1:
            raise ValueError(f"Invalid internal channels: c1={c1}, c2={c2}")

        self.planes = planes
        self.c1 = c1
        self.c2 = c2

        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
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

class ResNet7Real(nn.Module):
    def __init__(self, num_classes=100, internal_cfg: dict | None = None):
        super(ResNet7Real, self).__init__()
        internal_cfg = internal_cfg or {}
        self.in_channels = 64  # 改为64，匹配教师模型初始通道
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 模仿教师初始层
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 修正残差块通道数配置：
        # layer1：输入64通道 → 输出256通道（64*4，匹配Bottleneck扩展因子），1个block
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=1, stride=1,
                   block_cfg=internal_cfg.get('layer1'))
        self.layer2 = self._make_layer(in_channels=256, out_channels=128, blocks=1, stride=2,
                   block_cfg=internal_cfg.get('layer2'))
        self.layer3 = nn.Identity()
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        mlp_dim = int(internal_cfg.get('mlp_dim', 1000))
        self.intermediate_fc = nn.Linear(512, mlp_dim)
        self.fc = nn.Linear(mlp_dim, num_classes)

        # =================== 新增：频域重建头 ===================
        # 先估算 layer3 输出尺寸（输入 2x128 → 经过 stride=2 三次 → 128/(2^3)=16）
        # 所以 layer3 输出 H=W=16（因为 128 -> 64 -> 32 -> 16）
        # self.reconstruction_head = nn.Sequential(
        # nn.Conv1d(1024, 256, kernel_size=1),  # 1x1 conv to reduce channels
        # nn.ReLU(inplace=True),
        # nn.Upsample(size=(128), mode='linear', align_corners=False),  # 直接上采样到目标尺寸
        # nn.Conv1d(256, 2, kernel_size=3, padding=1)  # 输出 2 通道
        # )
        # # =====================================================

    def _make_layer(self, in_channels, out_channels, blocks, stride, block_cfg=None):
        downsample = None
        # 当步长≠1或输入通道≠输出通道×扩展因子时，需要下采样
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:  # 注意这里使用Bottleneck的expansion=4
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * Bottleneck.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * Bottleneck.expansion)
            )
        # block_cfg: list[tuple[c1,c2]]，长度=blocks；默认每个 block 使用 (planes, planes)
        if block_cfg is None:
            block_cfg = [(out_channels, out_channels) for _ in range(blocks)]
        if len(block_cfg) != blocks:
            raise ValueError(f"block_cfg length mismatch: expect {blocks}, got {len(block_cfg)}")

        layers = []
        # 第一个块：处理通道数转换和步长
        c1_0, c2_0 = block_cfg[0]
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample, c1=c1_0, c2=c2_0))
        # 更新当前通道数（输出通道×扩展因子）
        self.in_channels = out_channels * Bottleneck.expansion
        # 后续块：输入输出通道数一致
        for idx in range(1, blocks):
            c1_i, c2_i = block_cfg[idx]
            layers.append(Bottleneck(self.in_channels, out_channels, c1=c1_i, c2=c2_i))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False, return_reconstruction=False):
        # 适配 1D 卷积输入：如果输入是2维 (B, L)，增加通道维度 (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # 如果是4维 (B, C, H, W)，展平空间维度变为 (B, C, H*W)
        elif x.dim() == 4:
            x = x.view(x.size(0), x.size(1), -1)

        # print(f"输入序列维度: {x.shape}")  # 调试：打印输入图像维度
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)  # 256维
        x6 = self.layer2(x5)  # 512维
        x7 = self.layer3(x6)  # 512维（Identity，直接输出x6）

        x8 = self.avgpool(x7)
        x9 = torch.flatten(x8, 1)  # 512维
        x_intermediate = self.intermediate_fc(x9)  # 1000维
        x10 = self.fc(x_intermediate)  # 21维

        # if return_reconstruction:
        #     # 重建路径：从 layer3 输出重建 (2, 8, 16)
        #     rec_2d = self.reconstruction_head(x7)  # (B, 2, 8, 16)
        #     return rec_2d

        if is_feat:
            return [x3, x5, x6], x10
        else:
            return x10

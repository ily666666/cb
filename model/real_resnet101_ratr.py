import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x / keep_prob * random_tensor


class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBottleneck1D(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        reduction=16,
        downsample=None,
        drop_path_prob=0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.se = SELayer1D(planes * self.expansion, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out = self.drop_path(out)
        out = out + identity
        out = self.relu(out)
        return out


class SEResNet1D(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=3,
        reduction=16,
        in_channels=1,
        dropout_rate=0.3,
        drop_path_rate=0.2,
    ):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.total_blocks = int(sum(layers))
        self._block_idx = 0
        self.drop_path_rate = float(drop_path_rate)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _get_drop_path_prob(self):
        if self.total_blocks <= 1:
            return self.drop_path_rate
        return self.drop_path_rate * (self._block_idx / (self.total_blocks - 1))

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        dp = self._get_drop_path_prob()
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                reduction=reduction,
                downsample=downsample,
                drop_path_prob=dp,
            )
        )
        self._block_idx += 1
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            dp = self._get_drop_path_prob()
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    reduction=reduction,
                    downsample=None,
                    drop_path_prob=dp,
                )
            )
            self._block_idx += 1

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _normalize_input(self, x):
        if torch.is_complex(x):
            x = torch.real(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.size(1) != 1:
                x = x[:, :1, :]
        elif x.dim() == 4:
            x = x.view(x.size(0), 1, -1)
        return x

    def forward(self, x, is_feat=False, preact=False):
        x = self._normalize_input(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        out = self.fc(x)

        if is_feat:
            return [f0, f1, f2, f3], out
        return out


def real_resnet101_ratr(num_classes=3, in_channels=1, dropout_rate=0.3, drop_path_rate=0.2, reduction=16):
    return SEResNet1D(
        SEBottleneck1D,
        [3, 4, 23, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        reduction=reduction,
    )


class ResNet101Real(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, dropout_rate=0.3, drop_path_rate=0.2, reduction=16):
        super().__init__()
        self.model = real_resnet101_ratr(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            reduction=reduction,
        )
        self.fc = self.model.fc

    def forward(self, x, is_feat=False, preact=False):
        return self.model(x, is_feat=is_feat, preact=preact)


CombinedModel = ResNet101Real

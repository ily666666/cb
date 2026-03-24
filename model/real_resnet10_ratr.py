import torch
import torch.nn as nn


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

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

        out = out + identity
        out = self.relu(out)
        return out


class ResNet10Real(nn.Module):
    def __init__(self, num_classes=3, in_channels=1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck1D, planes=64, blocks=1, stride=1)
        self.layer2 = self._make_layer(Bottleneck1D, planes=128, blocks=1, stride=2)
        self.layer3 = self._make_layer(Bottleneck1D, planes=256, blocks=1, stride=2)
        self.layer4 = self._make_layer(Bottleneck1D, planes=512, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.intermediate_fc = nn.Linear(512 * Bottleneck1D.expansion, 1000)
        self.fc = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        out_channels = planes * block.expansion

        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

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
        x = torch.flatten(x, 1)
        x = self.intermediate_fc(x)
        out = self.fc(x)

        if is_feat:
            return [f0, f1, f2, f3], out
        return out

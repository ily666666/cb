import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4
    
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
    def __init__(self, num_classes=7):
        super(ResNet20Real, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channels=256, out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(in_channels=512, out_channels=256, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.intermediate_fc = nn.Linear(1024, 1000)
        self.fc = nn.Linear(1000, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        # Input x shape: (batch_size, 500) complex tensor
        # Extract real and imaginary parts, reshape to (batch_size, 2, 20, 25)
        x_real = x.real
        x_imag = x.imag
        x_real = x_real.reshape(x_real.shape[0], 1, 20, 25)
        x_imag = x_imag.reshape(x_imag.shape[0], 1, 20, 25)
        x = torch.cat([x_real, x_imag], dim=1)

        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)

        x8 = self.avgpool(x7)
        x9 = torch.flatten(x8, 1)
        x_intermediate = self.intermediate_fc(x9)
        x10 = self.fc(x_intermediate)

        if is_feat:
            return [x3, x5, x6, x7], x10
        else:
            return x10

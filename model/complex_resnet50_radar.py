import torch
import torch.nn as nn
import sys

import model.CVNN as CVNN

sys.path.append("..")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = CVNN.ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = CVNN.ComplexBatchNorm2d(out_channels)
        
        self.conv2 = CVNN.ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = CVNN.ComplexBatchNorm2d(out_channels)
        
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

        out_real = out.real + identity.real
        out_imag = out.imag + identity.imag
        out = torch.view_as_complex(torch.stack([out_real, out_imag], dim=-1))

        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        
        self.conv1 = CVNN.ComplexConv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = CVNN.ComplexBatchNorm2d(64)
        self.relu = CVNN.ComplexReLU()
        self.maxpool = CVNN.ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        self.avgpool = CVNN.ComplexAdaptiveAvgPool2d(height=1, width=1)
        self.fc = CVNN.ComplexLinear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                CVNN.ComplexConv2d(self.in_channels, out_channels * Bottleneck.expansion, 
                                  kernel_size=1, stride=stride, bias=False),
                CVNN.ComplexBatchNorm2d(out_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)

        x9 = self.avgpool(x8)
        x10 = torch.flatten(x9, 1)
        x11 = self.fc(x10)

        if is_feat:
            if preact:
                return [x3, x5, x6, x7, x8], x11
            else:
                return [x3, x5, x6, x7, x8], x11
        else:
            return x11


class CombinedModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CombinedModel, self).__init__()
        self.is_complex = True
        print("Initializing Complex ResNet50 for Radar Dataset...")
        print(f"Number of classes: {num_classes}")
        
        self.resnet50_model = ResNet50(num_classes=1000)
        self.fc1 = CVNN.ComplexLinear(1000, num_classes).cuda()
        
        print("Complex ResNet50 initialized successfully.")

    def forward(self, x, is_feat=False, preact=False, return_features=False):
        # Input x shape: (batch_size, 500) complex tensor
        # Radar data: 500 time points, need to reshape to 2D image
        # Reshape to (batch_size, 1, 20, 25) - 20x25 = 500
        x_real = x.real
        x_imag = x.imag
        x_real = x_real.reshape(x_real.shape[0], 1, 20, 25)
        x_imag = x_imag.reshape(x_imag.shape[0], 1, 20, 25)
        out = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))
        
        if is_feat:
            feat_resnet, out_resnet = self.resnet50_model(out, is_feat=True, preact=preact)
        else:
            out_resnet = self.resnet50_model(out)
        
        out_resnet_real = torch.flatten(out_resnet.real, start_dim=1)
        out_resnet_imag = torch.flatten(out_resnet.imag, start_dim=1)
        out_flat = torch.view_as_complex(torch.stack([out_resnet_real, out_resnet_imag], dim=-1))
        
        out_final = self.fc1(out_flat)
        
        output = torch.abs(out_final)
        features = output
        
        if is_feat:
            feat_resnet_real = [torch.abs(f) for f in feat_resnet]
            return feat_resnet_real, output
        elif return_features:
            return output, features
        else:
            return output

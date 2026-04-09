# author: songguangming time:2021/10/20
from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict

__all__ = ['resnet', 'Bottleneck']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, cfg[0],
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channel)
        # self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.bn2(out)
        # out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, dataset='cifar10', depth=20, cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = BasicBlock
        expansion = block.expansion

        if cfg is None:
            cfg = {
                    'first_layer_setting' : 16,
                    'layer1_setting' : [[16, 16, 16*expansion] for i in range(n)],
                    'layer2_setting' : [[32, 32, 32*expansion] for i in range(n)],
                    'layer3_setting' : [[64, 64, 64*expansion] for i in range(n)],
                    'last_layer_setting': 64*expansion
            }
            # print(cfg)

        cfg = EasyDict(cfg)
        self.inplanes = cfg.first_layer_setting

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, n, cfg=cfg.layer1_setting)
        self.layer2 = self._make_layer(block, n, cfg=cfg.layer2_setting, stride=2)
        self.layer3 = self._make_layer(block, n, cfg=cfg.layer3_setting, stride=2)
        self.last_bn = nn.BatchNorm2d(cfg.last_layer_setting)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg.last_layer_setting, 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg.last_layer_setting, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, block_num, cfg, stride=1):
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, cfg[0][2], kernel_size=1, stride=stride, bias=False)
        )
        layers.append(block(self.inplanes, cfg[0], stride, downsample))
        self.inplanes = cfg[0][2]

        for i in range(1, block_num):
            layers.append(block(self.inplanes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.last_bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    from thop import profile, clever_format
    # resnet18 = models.resnet18().cuda()
    # dataset = "cifar10"
    dataset = "cifar10"
    # dataset = "imagenet"
    if dataset == "cifar10" or dataset == 'cifar100':
        net = resnet(dataset=dataset, depth=20)
        inp = torch.randn(1, 3, 32, 32)
        print(net)
    elif dataset == "imagenet":
        net = resnet(dataset=dataset)
        inp = torch.randn(1, 3, 224, 224)

    total = sum([param.nelement() for param in net.parameters()])
    print(total)
    # print(net(inp))
    f, p = profile(net, (inp,))
    f, p = clever_format([f, p])
    print(f, p)
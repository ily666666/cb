"""
ResNet101 model for RATR dataset with 2048 input length support.

This is a variant of real_resnet101_ratr.py specifically designed to handle
2048-length input sequences for cloud-side processing. The architecture remains
identical to the base model (SEResNet1D with [3, 4, 23, 3] layer configuration),
but this variant is optimized for processing extended-length RATR data.

Key differences from real_resnet101_ratr.py:
- Designed for 2048-length input sequences (vs 1024 in base model)
- Used for cloud-only training and inference tasks
- Maintains same SEResNet1D architecture with SE (Squeeze-and-Excitation) blocks

Architecture:
- Input: (batch_size, 1, 2048) - Single channel, 2048 time steps
- Output: (batch_size, 3) - 3 classes (E2D, P3C, P8A)
- Layers: [3, 4, 23, 3] - ResNet101 configuration
- SE reduction ratio: 16
- Dropout rate: 0.3
- DropPath rate: 0.2
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization."""
    
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
    """Squeeze-and-Excitation layer for 1D signals."""
    
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
    """SE-ResNet bottleneck block for 1D signals."""
    
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
    """
    SE-ResNet for 1D signals with 2048 input length support.
    
    This model is specifically designed for RATR dataset with 2048-length input sequences.
    It uses Squeeze-and-Excitation (SE) blocks for channel-wise feature recalibration
    and DropPath for regularization.
    
    Args:
        block: Bottleneck block type (SEBottleneck1D)
        layers: List of layer depths [3, 4, 23, 3] for ResNet101
        num_classes: Number of output classes (default: 3 for RATR)
        reduction: SE reduction ratio (default: 16)
        in_channels: Input channels (default: 1 for single-channel signals)
        dropout_rate: Dropout rate before final FC layer (default: 0.3)
        drop_path_rate: Maximum DropPath rate (default: 0.2)
    
    Input:
        x: Tensor of shape (batch_size, 1, 2048) or (batch_size, 2048)
           Supports 2048-length time-series signals
    
    Output:
        Tensor of shape (batch_size, num_classes) - class logits
    """
    
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

        # Initial convolution - handles 2048-length input
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.total_blocks = int(sum(layers))
        self._block_idx = 0
        self.drop_path_rate = float(drop_path_rate)

        # ResNet101 layers: [3, 4, 23, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _get_drop_path_prob(self):
        """Calculate linearly increasing drop path probability."""
        if self.total_blocks <= 1:
            return self.drop_path_rate
        return self.drop_path_rate * (self._block_idx / (self.total_blocks - 1))

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16):
        """Create a residual layer with multiple blocks."""
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
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _normalize_input(self, x):
        """
        Normalize input tensor to expected format (batch_size, 1, 2048).
        
        Handles various input formats:
        - 2D input (B, 2048) → (B, 1, 2048)
        - 3D input with multiple channels → take first channel
        - 4D input → flatten to 3D
        - Complex tensors → take real part
        
        This ensures compatibility with different data loading formats
        while maintaining the expected 2048-length input.
        """
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
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 1, 2048) or (B, 2048)
            is_feat: If True, return intermediate features
            preact: Unused, kept for compatibility
        
        Returns:
            If is_feat=False: Tensor of shape (B, num_classes)
            If is_feat=True: Tuple of ([f0, f1, f2, f3], output)
        """
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


def real_resnet101_ratr_2048(num_classes=3, in_channels=1, dropout_rate=0.3, drop_path_rate=0.2, reduction=16):
    """
    Create ResNet101 model for RATR dataset with 2048 input length.
    
    This is the 2048-length variant designed for cloud-side processing.
    
    Args:
        num_classes: Number of output classes (default: 3)
        in_channels: Number of input channels (default: 1)
        dropout_rate: Dropout rate (default: 0.3)
        drop_path_rate: DropPath rate (default: 0.2)
        reduction: SE reduction ratio (default: 16)
    
    Returns:
        SEResNet1D model configured for 2048-length inputs
    """
    return SEResNet1D(
        SEBottleneck1D,
        [3, 4, 23, 3],  # ResNet101 configuration
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        reduction=reduction,
    )


class ResNet101Real(nn.Module):
    """
    Wrapper class for ResNet101 with 2048 input length support.
    
    This wrapper provides a consistent interface for the RATR 2048 model
    and maintains compatibility with the existing codebase.
    
    Input: (batch_size, 1, 2048) or (batch_size, 2048)
    Output: (batch_size, 3) for RATR classification
    """
    
    def __init__(self, num_classes=3, in_channels=1, dropout_rate=0.3, drop_path_rate=0.2, reduction=16):
        super().__init__()
        self.model = real_resnet101_ratr_2048(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            reduction=reduction,
        )
        self.fc = self.model.fc

    def forward(self, x, is_feat=False, preact=False):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, 1, 2048) or (B, 2048)
            is_feat: If True, return intermediate features
            preact: Unused, kept for compatibility
        
        Returns:
            Model output (logits or features+logits)
        """
        return self.model(x, is_feat=is_feat, preact=preact)


# Alias for backward compatibility
CombinedModel = ResNet101Real

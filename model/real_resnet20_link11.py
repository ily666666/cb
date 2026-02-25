"""
Real-valued ResNet20 for Link11 Dataset
Input: (batch_size, 1024) complex tensor
Extract I/Q: (batch_size, 2, 32, 32)
Output: (batch_size, num_classes) predictions
"""
import torch
import torch.nn as nn


class RealBottleneck(nn.Module):
    """Real-valued Bottleneck block for ResNet"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(RealBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        
        out = out + identity
        out = self.relu(out)
        
        return out


class ResNet20Real(nn.Module):
    """
    Real-valued ResNet20 for Link11 Dataset
    
    Input shape: (batch_size, 1024) complex tensor
    Extract I/Q and reshape to: (batch_size, 2, 32, 32)
    
    Architecture: ResNet20 with Bottleneck blocks (2-2-2)
    """
    
    def __init__(self, num_classes=7):
        super(ResNet20Real, self).__init__()
        self.num_classes = num_classes
        
        print("Initializing Real ResNet20 for Link11 Dataset...")
        print(f"Number of classes: {num_classes}")
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(RealBottleneck, 64, 64, 2, stride=1)
        self.layer2 = self._make_layer(RealBottleneck, 256, 128, 2, stride=2)
        self.layer3 = self._make_layer(RealBottleneck, 512, 256, 2, stride=2)
        
        # Global average pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        
        # 为了兼容蒸馏代码，添加fc属性别名指向fc2
        self.fc = self.fc2
        
        print("Real ResNet20 initialized successfully.")
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer"""
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, is_feat=False):
        """
        Args:
            x: Input tensor of shape (batch_size, 1024) complex
            is_feat: If True, return intermediate features for knowledge distillation
        
        Returns:
            out: Output predictions (batch_size, num_classes)
            features: List of intermediate features if is_feat=True
        """
        # Input shape: (batch_size, 1024) complex
        batch_size = x.shape[0]
        
        # Extract real and imaginary parts
        x_real = torch.real(x)  # (batch_size, 1024)
        x_imag = torch.imag(x)  # (batch_size, 1024)
        
        # Reshape each to (batch_size, 1, 32, 32)
        x_real = x_real.reshape(batch_size, 1, 32, 32)
        x_imag = x_imag.reshape(batch_size, 1, 32, 32)
        
        # Concatenate to (batch_size, 2, 32, 32)
        x = torch.cat([x_real, x_imag], dim=1)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers with feature extraction
        features = []
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        
        if is_feat:
            return features, out
        else:
            return out


if __name__ == '__main__':
    # Test the model
    model = ResNet20Real(num_classes=7)
    
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

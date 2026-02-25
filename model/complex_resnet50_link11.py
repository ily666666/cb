"""
Complex-valued ResNet50 for Link11 Dataset
Input: (batch_size, 1024) complex tensor
Reshape: (batch_size, 1, 32, 32) - maintains complex values
Output: (batch_size, num_classes) predictions
"""
import torch
import torch.nn as nn
from .CVNN import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexMaxPool2d, ComplexAdaptiveAvgPool2d, ComplexLinear


class ComplexBottleneck(nn.Module):
    """Complex-valued Bottleneck block for ResNet"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ComplexBottleneck, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.conv3 = ComplexConv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = ComplexBatchNorm2d(out_channels * self.expansion)
        self.relu = ComplexReLU()
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


class ComplexResNet50Link11(nn.Module):
    """
    Complex-valued ResNet50 for Link11 Dataset
    
    Input shape: (batch_size, 1024) complex tensor
    Reshape to: (batch_size, 1, 32, 32)
    
    Architecture: ResNet50 with Bottleneck blocks (3-4-6-3)
    """
    
    def __init__(self, num_classes=7):
        super(ComplexResNet50Link11, self).__init__()
        self.num_classes = num_classes
        
        print("Initializing Complex ResNet50 for Link11 Dataset...")
        print(f"Number of classes: {num_classes}")
        
        # Initial convolution layer
        self.conv1 = ComplexConv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm2d(64)
        self.relu = ComplexReLU()
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(ComplexBottleneck, 64, 64, 3, stride=1)
        self.layer2 = self._make_layer(ComplexBottleneck, 256, 128, 4, stride=2)
        self.layer3 = self._make_layer(ComplexBottleneck, 512, 256, 6, stride=2)
        self.layer4 = self._make_layer(ComplexBottleneck, 1024, 512, 3, stride=2)
        
        # Global average pooling and classification
        self.avgpool = ComplexAdaptiveAvgPool2d(height=1, width=1)
        self.fc = ComplexLinear(2048, num_classes)
        
        print("Complex ResNet50 initialized successfully.")
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer"""
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm2d(out_channels * block.expansion),
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
        
        # Reshape to (batch_size, 1, 32, 32)
        x = x.reshape(batch_size, 1, 32, 32)
        
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
        
        x = self.layer4(x)
        features.append(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        out = self.fc(x)
        
        # Convert to real output (take magnitude)
        out = torch.abs(out)
        
        if is_feat:
            return features, out
        else:
            return out


# Alias for compatibility
CombinedModel = ComplexResNet50Link11


if __name__ == '__main__':
    # Test the model
    model = ComplexResNet50Link11(num_classes=7)
    
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

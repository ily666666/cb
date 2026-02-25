"""
Tiny Complex-valued CNN for Link11 Dataset (Minimal Convolution Network)
Input: (batch_size, 1024) complex tensor
Reshape: (batch_size, 1, 32, 32) - maintains complex values
Output: (batch_size, num_classes) predictions

Architecture: 2 Conv layers + 1 FC layer (No residual connections)
- Conv1: 1 → 16 channels, kernel=5, stride=2, padding=2
- Conv2: 16 → 32 channels, kernel=5, stride=2, padding=2
- AdaptiveAvgPool: (32, 8, 8) → (32, 1, 1)
- FC: 32 → num_classes

Parameters: ~15K (CPU-friendly)
"""
import torch
import torch.nn as nn

try:
    from .CVNN import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexAdaptiveAvgPool2d, ComplexLinear
except ImportError:
    from CVNN import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexAdaptiveAvgPool2d, ComplexLinear


class ComplexTinyLink11(nn.Module):
    """
    Tiny Complex-valued CNN for Link11 Dataset
    
    Minimal architecture for fast CPU verification:
    - 2 convolutional layers
    - 1 fully connected layer
    - No residual connections
    """
    
    def __init__(self, num_classes=7):
        super(ComplexTinyLink11, self).__init__()
        self.num_classes = num_classes
        
        print("Initializing Tiny Complex CNN for Link11 Dataset...")
        print(f"Number of classes: {num_classes}")
        print("Architecture: 2 Conv + 1 FC (No residual)")
        
        # Layer 1: Conv + BN + ReLU + MaxPool
        self.conv1 = ComplexConv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = ComplexBatchNorm2d(16)
        self.relu1 = ComplexReLU()
        
        # Layer 2: Conv + BN + ReLU + MaxPool
        self.conv2 = ComplexConv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = ComplexBatchNorm2d(32)
        self.relu2 = ComplexReLU()
        
        # Global average pooling
        self.avgpool = ComplexAdaptiveAvgPool2d(height=1, width=1)
        
        # Fully connected layer
        self.fc = ComplexLinear(32, num_classes)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,} (~{total_params/1000:.1f}K)")
        print("Tiny Complex CNN initialized successfully.")
    
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
        
        features = []
        
        # Layer 1: Conv1 + BN + ReLU
        # (batch, 1, 32, 32) → (batch, 16, 16, 16)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        features.append(x)
        
        # Layer 2: Conv2 + BN + ReLU
        # (batch, 16, 16, 16) → (batch, 32, 8, 8)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        features.append(x)
        
        # Global average pooling
        # (batch, 32, 8, 8) → (batch, 32, 1, 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (batch, 32)
        
        # Fully connected layer
        out = self.fc(x)  # (batch, num_classes)
        
        # Convert to real output (take magnitude)
        out = torch.abs(out)
        
        if is_feat:
            return features, out
        else:
            return out


# Alias for compatibility
CombinedModel = ComplexTinyLink11


if __name__ == '__main__':
    # Test the model
    print("\n" + "="*70)
    print("Testing Tiny Complex CNN for Link11")
    print("="*70)
    
    model = ComplexTinyLink11(num_classes=7)
    
    # Create dummy input
    dummy_input = torch.randn(4, 1024, dtype=torch.complex64)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input dtype: {dummy_input.dtype}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Test with is_feat=True
    features, output = model(dummy_input, is_feat=True)
    print(f"\nWith is_feat=True:")
    print(f"Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Feature {i+1}: {feat.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)

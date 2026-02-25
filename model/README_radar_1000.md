# Complex ResNet50 with Attention for Radar Dataset (1000 Input Length)

## 模型概述

`complex_resnet50_radar_with_attention_1000` 是专门为处理1000长度输入的雷达数据设计的复数ResNet50模型，带有时频注意力机制。

## 关键特性

- **输入长度**: 1000个复数值
- **输入reshape**: (batch_size, 1000) → (batch_size, 1, 40, 25)
- **架构**: ResNet50 + TimeFreqAttention
- **输出**: (batch_size, num_classes) 实数预测值

## 与原模型的区别

| 特性 | 原模型 (500输入) | 新模型 (1000输入) |
|------|-----------------|------------------|
| 输入长度 | 500 | 1000 |
| Reshape尺寸 | (1, 20, 25) | (1, 40, 25) |
| 时间维度 | 20 | 40 |
| 频率维度 | 25 | 25 |

## 使用方法

### 1. 作为教师模型（云侧）

在 `fed/main.py` 中使用：

```bash
python fed/main.py \
    --mode project \
    --dataset_type radar \
    --server_model complex_resnet50_radar_with_attention_1000 \
    --client_model real_resnet20_radar \
    --num_classes 7
```

### 2. 在代码中直接使用

```python
import torch
from model.complex_resnet50_radar_with_attention_1000 import CombinedModel

# 创建模型
model = CombinedModel(num_classes=7)

# 准备输入数据 (1000长度的复数张量)
input_data = torch.randn(batch_size, 1000, dtype=torch.complex64)

# 前向传播
output = model(input_data)  # 输出: (batch_size, 7)

# 提取特征
features, output = model(input_data, is_feat=True)
```

### 3. 加载预训练模型

模型路径中需要包含 `with_attention` 和 `1000` 标识，系统会自动识别：

```bash
--pretrained_server_model path/to/complex_resnet50_radar_with_attention_1000_model.pth
```

或者在代码中：

```python
# 系统会根据路径自动检测
teacher_model_path = "models/radar_with_attention_1000_epoch100.pth"
# 如果路径包含 'with_attention' 和 '1000'，会自动加载正确的模型
```

## 项目集成

新模型已集成到以下文件：

1. **model/complex_resnet50_radar_with_attention_1000.py** - 模型定义
2. **fed/project.py** - Project模式支持
3. **fed/main.py** - 主训练脚本支持
4. **fed/nofl.py** - NoFL模式支持

## 注意事项

### 1. 数据要求
- 输入必须是1000长度的复数张量
- 数据类型: `torch.complex64` 或 `torch.complex128`

### 2. 训练建议
- **不要混合使用500和1000长度的数据训练同一个模型**
- 500长度模型和1000长度模型需要分别训练
- 使用1000长度模型时，确保所有数据都是1000长度

### 3. 模型选择
- 如果数据是500长度 → 使用 `complex_resnet50_radar_with_attention`
- 如果数据是1000长度 → 使用 `complex_resnet50_radar_with_attention_1000`

### 4. 性能考虑
- 1000长度模型的计算量略大于500长度模型
- 中间特征图尺寸更大，需要更多显存
- 时间维度翻倍，可以捕获更长的时间依赖关系

## 测试

运行测试脚本验证模型：

```bash
python test_new_model.py
```

预期输出：
```
✓ Model test PASSED!
✓ Feature extraction test PASSED!
```

## 技术细节

### 特征图尺寸变化

```
输入: (B, 1000) → reshape → (B, 1, 40, 25)
conv1 (stride=2): (B, 64, 20, 13)
maxpool (stride=2): (B, 64, 10, 7)
layer1: (B, 256, 10, 7)
layer2 (stride=2): (B, 512, 5, 4)
layer3 (stride=2): (B, 1024, 3, 2)
layer4 (stride=2): (B, 2048, 2, 1)
avgpool: (B, 2048, 1, 1)
flatten: (B, 2048)
fc: (B, 1000)
fc1: (B, num_classes)
abs: (B, num_classes) [实数输出]
```

### 注意力机制

每个残差层后都有TimeFreqAttention模块：
- **通道注意力**: SE-style机制，学习通道重要性
- **空间注意力**: 7x7卷积，学习时频位置重要性
- **数值稳定性**: 包含NaN/Inf检测和处理

## 常见问题

**Q: 为什么不能用500长度的模型处理1000长度的数据？**

A: 虽然网络结构支持，但学到的特征表示是尺寸相关的。注意力权重、BatchNorm统计、感受野都会不匹配，导致性能显著下降。

**Q: 可以动态支持两种输入长度吗？**

A: 理论上可以，但需要：
- 使用GroupNorm或LayerNorm替代BatchNorm
- 多尺度训练（同时使用500和1000数据）
- 大量训练数据和时间
- 建议分别训练两个独立模型

**Q: 如何选择合适的reshape尺寸？**

A: 1000 = 40 × 25 的选择基于：
- 保持频率维度25不变（与500长度模型一致）
- 时间维度从20扩展到40（翻倍）
- 符合雷达信号的物理意义（更长的时间序列）

## 版本历史

- **v1.0** (2026-02-04): 初始版本，支持1000长度输入

# 端侧脚本

端侧负责实时生成信号数据，支持两种工作模式：
1. **edge 模式**：发送数据到边侧进行协同推理
2. **cloud 模式**：直接发送数据到云侧进行推理（用于性能对比和模型测试）

## 脚本列表

### run_device.py
**功能**：端侧信号生成器和数据发送器

**用途**：
- 实时生成三种数据集的信号（Link11, RML2016, Radar）
- 支持两种工作模式：发送到边侧或直接发送到云侧
- 模拟真实的端侧设备数据流

---

## 工作模式

### 1. Edge 模式（端→边→云）
端侧生成数据并发送到边侧，由边侧进行协同推理。

**使用示例**：
```bash
python run/device/run_device.py \
  --mode edge \
  --dataset_type link11 \
  --edge_host localhost \
  --edge_port 5555 \
  --batch_size 100 \
  --interval 0.1 \
  --total_samples 10000
```

**主要参数**：
- `--mode`: 工作模式（edge）
- `--dataset_type`: 数据集类型（link11/rml2016/radar）
- `--edge_host`: 边侧主机地址（默认 localhost）
- `--edge_port`: 边侧端口（默认 5555）
- `--batch_size`: 每批次样本数
- `--interval`: 发送间隔（秒）
- `--total_samples`: 总样本数限制（None 表示无限制）

**工作流程**：
1. 创建信号生成器（根据数据集类型）
2. 连接到边侧（ZeroMQ PUSH）
3. 循环生成数据批次：
   - 生成信号和标签
   - 打包为批次
   - 发送到边侧
4. 发送结束标志
5. 关闭连接

**输出**：
- 发送进度统计
- 总批次数和样本数

---

### 2. Cloud 模式（端→云）
端侧直接连接云侧进行推理，绕过边侧。

**用途**：
- 测试云侧模型准确率
- 对比 端→边→云 vs 端→云 的性能差异
- 评估边侧协同推理的价值

**使用示例**：
```bash
python run/device/run_device.py \
  --mode cloud \
  --dataset_type link11 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --batch_size 100 \
  --total_samples 10000
```

**主要参数**：
- `--mode`: 工作模式（cloud）
- `--dataset_type`: 数据集类型（link11/rml2016/radar）
- `--cloud_host`: 云侧主机地址（默认 localhost）
- `--cloud_port`: 云侧端口（默认 9999）
- `--batch_size`: 每批次样本数
- `--total_samples`: 总样本数限制

**工作流程**：
1. 创建信号生成器
2. 建立到云侧的 Socket 连接
3. 循环生成数据批次：
   - 生成信号和标签
   - 发送推理请求到云侧
   - 云侧计算准确率并输出
4. 发送结束标志
5. 云侧输出整体统计汇总

**输出**：
- 端侧：推理时间、吞吐量、总样本数
- 云侧：每批次准确率、整体准确率统计

**注意**：
- 端侧只负责数据传输，不计算准确率
- 推理准确率由云侧统计并输出
- 适合测试云侧模型性能

---

## 支持的数据集

### 1. Link11 数据集
**描述**：Link11 通信信号识别（7类）

**信号类型**：
- E-2D_1, E-2D_2（预警机）
- P-3C_1, P-3C_2（反潜机）
- P-8A_1, P-8A_2, P-8A_3（海上巡逻机）

**信号特征**：
- 2FSK 调制
- 物理层参数模拟（频率、功率、SNR）
- 飞行轨迹模拟
- 相位偏移特征

**生成器**：`Link11SignalGenerator` 和 `Link11DeviceSimulator`

---

### 2. RML2016 数据集
**描述**：RadioML 2016 调制识别（6类）

**调制类型**：
- 16QAM, 64QAM
- 8PSK, BPSK, QPSK
- GMSK

**信号特征**：
- 复数基带信号
- 随机 SNR（-10 到 20 dB）
- 600 个采样点

**生成器**：`RML2016DeviceSimulator`

---

### 3. Radar 数据集
**描述**：雷达信号识别（7类）

**飞机类型**：
- P-8A（3个个体）
- P-3C（2个个体）
- E-2D（2个个体）

**信号特征**：
- LFM（线性调频）信号
- 不同带宽（20-42 MHz）
- 相位噪声
- 500 个采样点

**生成器**：`RadarDeviceSimulator`

---

## 数据格式

### 发送到边侧（ZeroMQ）
```python
batch_data = {
    'signals': np.array,      # (batch_size, signal_length) complex64
    'labels': np.array,       # (batch_size,) int64
    'snrs': np.array,         # (batch_size,) float32
    'timestamp': float,       # 时间戳
    'batch_id': int          # 批次ID
}
```

### 发送到云侧（Socket）
```python
request = {
    'type': 'cloud_inference',
    'edge_id': 'device_direct',
    'data': np.array,         # (batch_size, signal_length) complex64
    'labels': np.array,       # (batch_size,) int64
    'batch_id': int
}
```

### 结束标志
```python
end_signal = {
    'type': 'end_transmission',
    'total_samples': int,
    'total_batches': int,
    'timestamp': float
}
```

---

## 网络配置

### Edge 模式
- `--edge_host`: 边侧主机地址（默认 localhost）
- `--edge_port`: 边侧端口（默认 5555）
- 协议：ZeroMQ PUSH/PULL

### Cloud 模式
- `--cloud_host`: 云侧主机地址（默认 localhost）
- `--cloud_port`: 云侧端口（默认 9999）
- 协议：TCP Socket

---

## 完整工作流程示例

### 场景1：端→边→云 协同推理

```bash
# 步骤1：启动云侧协同推理服务器
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999

# 步骤2：启动边侧协同推理
python run/edge/run_edge_collaborative.py \
  --edge_id 0 \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model_path run/edge/pth/link11/edge_1_kd_model.pth \
  --data_source device \
  --device_port 5555 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --thresholds "0.7"

# 步骤3：启动端侧设备（发送到边侧）
python run/device/run_device.py \
  --mode edge \
  --dataset_type link11 \
  --edge_host localhost \
  --edge_port 5555 \
  --batch_size 100 \
  --interval 0.1 \
  --total_samples 10000
```

**结果**：
- 端侧生成数据并发送到边侧
- 边侧进行推理，低置信度样本发送到云侧
- 边侧输出准确率、云侧调用率等指标

---

### 场景2：端→云 直接推理（性能对比）

```bash
# 步骤1：启动云侧协同推理服务器
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999

# 步骤2：启动端侧设备（直接连接云侧）
python run/device/run_device.py \
  --mode cloud \
  --dataset_type link11 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --batch_size 100 \
  --total_samples 10000
```

**结果**：
- 端侧生成数据并直接发送到云侧
- 云侧进行推理并输出准确率
- 端侧输出推理时间和吞吐量
- 用于对比边侧协同推理的性能提升

---

## 性能参数

### 批次大小建议
- **Link11**: 32-100（信号较长）
- **RML2016**: 100-200（信号中等）
- **Radar**: 100-200（信号较短）

### 发送间隔
- **实时模拟**: 0.1-0.5 秒
- **快速测试**: 0.01-0.05 秒
- **无延迟**: 0 秒

### 总样本数
- **快速测试**: 1000-5000
- **完整测试**: 10000-50000
- **无限制**: None（持续运行）

---

## 注意事项

1. **数据集类型**：端侧、边侧、云侧必须使用相同的数据集类型
2. **网络连接**：确保能连接到边侧或云侧服务器
3. **端口占用**：确保端口未被占用
4. **内存管理**：大批次可能导致内存占用增加
5. **结束标志**：程序结束时会自动发送结束标志
6. **模式选择**：
   - edge 模式：测试边侧协同推理
   - cloud 模式：测试云侧模型准确率和性能对比

---

## 故障排除

### 连接失败
- 检查主机地址和端口是否正确
- 确保服务器已启动
- 检查防火墙设置

### 数据格式错误
- 确保数据集类型匹配
- 检查信号长度是否正确

### 内存不足
- 减小批次大小
- 减少总样本数
- 增加发送间隔

---

## 输出示例

### Edge 模式输出
```
[端侧] 已连接到边侧: tcp://localhost:5555
[端侧] 开始生成数据，批次大小: 100，间隔: 0.1s，总样本数: 10000
[端侧] 已发送 10 批次, 1000/10000 样本 (10.0%)
[端侧] 已发送 20 批次, 2000/10000 样本 (20.0%)
...
[端侧] 已达到总样本数限制: 10000
[端侧] 发送完成，共发送 100 批次, 10000 样本
[端侧] 发送结束标志...
[端侧] 结束标志已发送
[端侧] 连接已关闭
```

### Cloud 模式输出
```
[端侧→云侧] 建立直连模式...
[端侧→云侧] 已连接到云侧: localhost:9999
[端侧→云侧] 开始推理测试，批次大小: 100，总样本数: 10000

注意: 推理准确率由云侧统计并输出

[端侧→云侧] 已发送 10 批次, 1000/10000 样本 (10.0%)
[端侧→云侧] 已发送 20 批次, 2000/10000 样本 (20.0%)
...
[端侧→云侧] 已达到总样本数限制: 10000
[端侧→云侧] 发送结束标志...

========================================
端侧推理统计（数据传输）
========================================
总样本数: 10000
总批次数: 100
平均推理时间: 0.0523 秒/批次
吞吐量: 1912.3 样本/秒
========================================
```


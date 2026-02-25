# Run 文件夹 - 云边端协同推理系统

本文件夹包含云边端协同推理系统的所有运行脚本，支持知识蒸馏、联邦学习和协同推理。

## 目录结构

```
run/
├── cloud/              # 云侧脚本
│   ├── run_cloud_pretrain.py          # 预训练教师模型
│   ├── run_cloud_kd.py                # 知识蒸馏服务器
│   ├── run_cloud_federated.py         # 联邦学习服务器
│   ├── run_cloud_collaborative.py     # 协同推理服务器
│   ├── pth/                           # 模型保存目录
│   └── README.md                      # 云侧详细文档
│
├── edge/               # 边侧脚本
│   ├── run_edge_kd.py                 # 知识蒸馏训练
│   ├── run_edge_federated.py          # 联邦学习客户端
│   ├── run_edge_collaborative.py      # 协同推理
│   ├── data_sources.py                # 数据源管理
│   ├── pth/                           # 模型保存目录
│   └── README.md                      # 边侧详细文档
│
├── device/             # 端侧脚本
│   ├── run_device.py                  # 信号生成和数据发送
│   └── README.md                      # 端侧详细文档
│
├── prepare_data_splits.py             # 数据预划分脚本
├── network_utils.py                   # 网络工具函数
├── run_cloud.py                       # 云侧启动脚本（快捷方式）
└── run_edge.py                        # 边侧启动脚本（快捷方式）
```

---

## 系统架构

### 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                         云侧 (Cloud)                         │
│  - 教师模型（大模型）                                          │
│  - 预训练、知识蒸馏、联邦学习、协同推理                          │
│  - 高计算能力，深度推理                                        │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ 网络连接
                              │ (Socket/ZeroMQ)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         边侧 (Edge)                          │
│  - 学生模型（小模型）                                          │
│  - 本地训练、知识蒸馏、联邦学习、协同推理                        │
│  - 中等计算能力，快速推理                                      │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ 数据流
                              │ (ZeroMQ)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         端侧 (Device)                        │
│  - 信号生成器                                                 │
│  - 实时数据采集和发送                                          │
│  - 低计算能力，数据源                                          │
└─────────────────────────────────────────────────────────────┘
```

### 两种推理模式

#### 模式1：端→边→云（协同推理）
```
端侧生成数据 → 边侧推理（学生模型）
                ├─ 高置信度 → 边侧输出
                └─ 低置信度 → 云侧推理（教师模型）→ 云侧输出
```

**优势**：
- 大部分样本在边侧快速处理
- 仅困难样本发送到云侧
- 降低网络传输和云侧负载
- 平衡准确率和延迟

#### 模式2：端→云（直接推理）
```
端侧生成数据 → 云侧推理（教师模型）→ 云侧输出
```

**优势**：
- 最高准确率
- 简单架构

**劣势**：
- 所有数据需传输到云侧
- 云侧负载高
- 网络延迟大

---

## 快速开始

### 前置准备

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **准备数据集**
```bash
# 使用数据预划分脚本
python run/prepare_data_splits.py \
  --dataset_type link11 \
  --data_path E:/3dataset/link11.pkl \
  --num_edges 3 \
  --dirichlet_alpha 0.3 \
  --cloud_ratio 0.3 \
  --output_dir E:/3dataset
```

**输出**：
```
E:/3dataset/link11/
├── cloud_data.pkl      # 云侧数据
├── edge_1_data.pkl     # 边侧1数据
├── edge_2_data.pkl     # 边侧2数据
└── edge_3_data.pkl     # 边侧3数据
```

---

### 完整工作流程

#### 阶段1：预训练教师模型

```bash
# 云侧：预训练教师模型
python run/cloud/run_cloud_pretrain.py \
  --dataset_type link11 \
  --data_path E:/3dataset \
  --num_classes 7 \
  --cloud_model complex_resnet50_link11 \
  --batch_size 128 \
  --epochs 100 \
  --learning_rate 0.001
```

**输出**：`run/cloud/pth/link11/cloud_pretrain_model.pth`

---

#### 阶段2：知识蒸馏训练学生模型

```bash
# 终端1：云侧启动知识蒸馏服务器
python run/cloud/run_cloud_kd.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999

# 终端2-4：边侧进行知识蒸馏训练（3个边侧并行）
python run/edge/run_edge_kd.py \
  --edge_id 1 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --dataset_type link11 \
  --data_dir E:/3dataset \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --batch_size 128 \
  --epochs 50 \
  --kd_alpha 0.7 \
  --kd_temperature 4.0

# 重复上述命令，修改 --edge_id 为 2 和 3
```

**输出**：
- `run/edge/pth/link11/edge_1_kd_model.pth`
- `run/edge/pth/link11/edge_2_kd_model.pth`
- `run/edge/pth/link11/edge_3_kd_model.pth`

---

#### 阶段3：联邦学习优化学生模型

```bash
# 终端1：云侧启动联邦学习服务器
python run/cloud/run_cloud_federated.py \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --num_edges 3 \
  --num_rounds 10 \
  --cloud_port 9999

# 终端2-4：边侧进行联邦学习（3个边侧并行）
python run/edge/run_edge_federated.py \
  --edge_id 1 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --dataset_type link11 \
  --data_dir E:/3dataset \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --kd_model_path run/edge/pth/link11/edge_1_kd_model.pth \
  --num_rounds 10 \
  --local_epochs 1 \
  --batch_size 32

# 重复上述命令，修改 --edge_id 和 --kd_model_path
```

**输出**：
- `run/edge/pth/link11/edge_1_federated_model.pth`
- `run/edge/pth/link11/edge_2_federated_model.pth`
- `run/edge/pth/link11/edge_3_federated_model.pth`

---

#### 阶段4：协同推理测试

##### 方案A：端→边→云（协同推理）

```bash
# 终端1：云侧启动协同推理服务器
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999

# 终端2：边侧启动协同推理
python run/edge/run_edge_collaborative.py \
  --edge_id 0 \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model_path run/edge/pth/link11/edge_1_federated_model.pth \
  --data_source device \
  --device_port 5555 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --thresholds "0.7" \
  --batch_size 128

# 终端3：端侧生成数据并发送到边侧
python run/device/run_device.py \
  --mode edge \
  --dataset_type link11 \
  --edge_host localhost \
  --edge_port 5555 \
  --batch_size 100 \
  --interval 0.1 \
  --total_samples 10000
```

**输出**：
- 边侧：整体准确率、边侧准确率、云侧准确率、云侧调用率
- 云侧：推理请求统计

---

##### 方案B：端→云（直接推理，性能对比）

```bash
# 终端1：云侧启动协同推理服务器
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999

# 终端2：端侧直接连接云侧
python run/device/run_device.py \
  --mode cloud \
  --dataset_type link11 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --batch_size 100 \
  --total_samples 10000
```

**输出**：
- 端侧：推理时间、吞吐量
- 云侧：每批次准确率、整体准确率统计

---

## 支持的数据集

### 1. Link11
- **类别数**：7类
- **描述**：Link11 通信信号识别
- **信号类型**：E-2D, P-3C, P-8A 飞机
- **云侧模型**：`complex_resnet50_link11`
- **边侧模型**：`real_resnet20_link11`

### 2. RML2016
- **类别数**：6类
- **描述**：RadioML 2016 调制识别
- **调制类型**：16QAM, 64QAM, 8PSK, BPSK, GMSK, QPSK
- **云侧模型**：`complex_resnet50_rml2016`
- **边侧模型**：`real_resnet20_rml2016`

### 3. Radar
- **类别数**：7类
- **描述**：雷达信号识别
- **飞机类型**：P-8A, P-3C, E-2D
- **云侧模型**：`complex_resnet50_radar`
- **边侧模型**：`real_resnet20_radar`

---

## 核心功能

### 1. 知识蒸馏（Knowledge Distillation）
- 云侧教师模型生成软标签
- 边侧学生模型学习软标签和硬标签
- 提升学生模型准确率

### 2. 联邦学习（Federated Learning）
- 多个边侧协同训练
- 云侧聚合本地模型
- 保护数据隐私

### 3. 协同推理（Collaborative Inference）
- 边侧快速推理
- 云侧深度推理
- 动态置信度阈值

---

## 网络配置

### 默认端口
- **云侧服务器**：9999
- **边侧设备接收**：5555

### 网络协议
- **云边通信**：TCP Socket
- **端边通信**：ZeroMQ PUSH/PULL

### 速率限制
- 默认：10.0 MB/s
- 可通过 `--rate_limit` 参数调整
- 设置为 0 或 None 表示不限速

---

## 数据路径配置

### 重要说明
所有脚本现在都需要通过参数指定数据路径，不再使用硬编码路径。

### 云侧
- `--data_path`: 数据集根目录（预训练时使用）

### 边侧
- `--data_dir`: 数据集根目录（必需）
- 自动加载：`{data_dir}/{dataset_type}/edge_{edge_id}_data.pkl`

### 端侧
- 无需数据路径（实时生成信号）

---

## 模型保存路径

### 云侧模型
```
run/cloud/pth/
├── link11/
│   └── cloud_pretrain_model.pth
├── rml2016/
│   └── cloud_pretrain_model.pth
└── radar/
    └── cloud_pretrain_model.pth
```

### 边侧模型
```
run/edge/pth/
├── link11/
│   ├── edge_1_kd_model.pth
│   ├── edge_1_federated_model.pth
│   ├── edge_2_kd_model.pth
│   └── ...
├── rml2016/
│   └── ...
└── radar/
    └── ...
```

---

## 实验场景

### 场景1：知识蒸馏效果评估
**目标**：评估知识蒸馏对学生模型的提升

**步骤**：
1. 预训练教师模型
2. 训练基线学生模型（无蒸馏）
3. 训练蒸馏学生模型（有蒸馏）
4. 对比准确率

---

### 场景2：联邦学习效果评估
**目标**：评估联邦学习对模型泛化能力的提升

**步骤**：
1. 知识蒸馏训练多个边侧模型
2. 联邦学习优化模型
3. 对比联邦学习前后的准确率

---

### 场景3：协同推理性能评估
**目标**：评估不同置信度阈值的性能

**步骤**：
1. 使用联邦学习后的模型
2. 测试多个置信度阈值（0.5, 0.6, 0.7, 0.8, 0.9）
3. 分析准确率、云侧调用率、推理时间的权衡

---

### 场景4：端→边→云 vs 端→云 对比
**目标**：评估边侧协同推理的价值

**步骤**：
1. 端→边→云：测试协同推理性能
2. 端→云：测试直接推理性能
3. 对比准确率、延迟、网络传输量

---

## 常见问题

### Q1: 如何选择置信度阈值？
**A**: 根据应用场景权衡：
- 高阈值（0.8-0.9）：更多样本发送到云侧，准确率高，延迟大
- 低阈值（0.5-0.6）：更多样本在边侧处理，延迟小，准确率略低
- 推荐：0.7（平衡点）

### Q2: 如何处理端口冲突？
**A**: 修改端口参数：
- 云侧：`--cloud_port`
- 边侧：`--device_port`

### Q3: 如何加速训练？
**A**: 
- 增大批次大小（`--batch_size`）
- 减少训练轮数（`--epochs`）
- 使用 GPU（自动检测）

### Q4: 如何处理数据不平衡？
**A**: 
- 调整 Dirichlet Alpha（`--dirichlet_alpha`）
- 更小的 alpha 值 → 更不平衡
- 更大的 alpha 值 → 更平衡

### Q5: 如何保存实验结果？
**A**: 
- 协同推理：使用 `--save_results` 参数保存 JSON 文件
- 训练日志：自动保存在控制台输出

---

## 性能优化建议

### 1. 批次大小
- **Link11**: 32-128
- **RML2016**: 100-200
- **Radar**: 100-200

### 2. 学习率
- **预训练**: 0.001
- **知识蒸馏**: 0.001
- **联邦学习**: 0.001-0.01

### 3. 蒸馏参数
- **Alpha**: 0.7（软标签权重）
- **Temperature**: 4.0（蒸馏温度）

### 4. 联邦学习
- **本地轮数**: 1-5
- **全局轮数**: 10-50

---

## 注意事项

1. **数据集类型一致性**：云侧、边侧、端侧必须使用相同的数据集类型
2. **模型兼容性**：边侧模型类型必须与云侧一致
3. **网络连接**：确保所有节点能够相互连接
4. **端口占用**：避免端口冲突
5. **数据路径**：使用绝对路径或相对于项目根目录的路径
6. **结束标志**：程序结束时会自动发送结束标志，确保统计完整

---

## 详细文档

- [云侧详细文档](cloud/README.md)
- [边侧详细文档](edge/README.md)
- [端侧详细文档](device/README.md)

---

## 贡献者

本系统支持云边端协同推理，适用于资源受限的边缘计算场景。


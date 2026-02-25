# 边侧脚本

边侧负责运行学生模型（小模型），进行本地训练、知识蒸馏、联邦学习和协同推理。

## 脚本列表

### 1. run_edge_kd.py
**功能**：边侧知识蒸馏训练

**用途**：
- 使用本地数据训练学生模型
- 向云侧请求软标签进行知识蒸馏
- 结合硬标签和软标签训练

**使用示例**：
```bash
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
```

**主要参数**：
- `--edge_id`: 边侧设备ID
- `--cloud_host`: 云侧主机地址
- `--cloud_port`: 云侧端口
- `--dataset_type`: 数据集类型
- `--data_dir`: 数据集根目录路径（必需）
- `--num_classes`: 分类类别数
- `--edge_model`: 边侧模型类型
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--kd_alpha`: 知识蒸馏权重（0-1，默认 0.7）
- `--kd_temperature`: 蒸馏温度（默认 4.0）

**输出**：
- 蒸馏后的模型：`run/edge/pth/{dataset_type}/edge_{edge_id}_kd_model.pth`

---

### 2. run_edge_federated.py
**功能**：边侧联邦学习客户端

**用途**：
- 连接云侧联邦学习服务器
- 下载全局模型
- 本地训练
- 上传本地模型更新

**使用示例**：
```bash
python run/edge/run_edge_federated.py \
  --edge_id 1 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --dataset_type link11 \
  --data_dir E:/3dataset \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --num_rounds 10 \
  --local_epochs 1 \
  --batch_size 32
```

**主要参数**：
- `--edge_id`: 边侧设备ID
- `--cloud_host`: 云侧主机地址
- `--cloud_port`: 云侧端口
- `--dataset_type`: 数据集类型
- `--data_dir`: 数据集根目录路径（必需）
- `--num_classes`: 分类类别数
- `--edge_model`: 边侧模型类型
- `--kd_model_path`: 知识蒸馏模型路径（可选，用作初始化）
- `--num_rounds`: 联邦学习轮次
- `--local_epochs`: 本地训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--optimizer`: 优化器（adam/sgd）

**工作流程**：
1. 等待云侧准备好
2. 每轮：
   - 下载全局模型
   - 本地训练
   - 测试评估（本地+全局测试集）
   - 上传本地模型
3. 保存最终模型

**输出**：
- 联邦学习模型：`run/edge/pth/{dataset_type}/edge_{edge_id}_federated_model.pth`

---

### 3. run_edge_collaborative.py
**功能**：云边协同推理（边侧）

**用途**：
- 使用学生模型进行边侧推理
- 低置信度样本发送到云侧深度推理
- 测试不同置信度阈值的性能

**使用示例**：

#### 从本地文件加载数据
```bash
python run/edge/run_edge_collaborative.py \
  --edge_id 0 \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model_path run/edge/pth/link11/edge_1_kd_model.pth \
  --data_source local \
  --data_path E:/3dataset/link11/test_data.pkl \
  --cloud_host localhost \
  --cloud_port 9999 \
  --thresholds "0.5,0.6,0.7,0.8,0.9" \
  --batch_size 128
```

#### 从端侧设备接收数据
```bash
python run/edge/run_edge_collaborative.py \
  --edge_id 0 \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model_path run/edge/pth/link11/edge_1_kd_model.pth \
  --data_source device \
  --device_port 5555 \
  --cloud_host localhost \
  --cloud_port 9999 \
  --thresholds "0.7" \
  --batch_size 128
```

**主要参数**：
- `--edge_id`: 边侧设备ID
- `--dataset_type`: 数据集类型
- `--num_classes`: 分类类别数
- `--edge_model_path`: 边侧学生模型路径
- `--edge_model`: 边侧模型类型（可选）
- `--data_source`: 数据源类型（local/device）
- `--data_path`: 测试数据路径（local 模式必需）
- `--device_port`: 设备数据接收端口（device 模式，默认 5555）
- `--cloud_host`: 云侧主机地址
- `--cloud_port`: 云侧端口
- `--thresholds`: 置信度阈值列表（逗号分隔）
- `--batch_size`: 批次大小
- `--num_batches`: 限制处理的批次数（用于快速测试）
- `--save_results`: 保存结果的JSON文件路径

**工作流程**：
1. 加载边侧学生模型
2. 连接云侧推理服务器
3. 对每个阈值：
   - 边侧推理，计算置信度
   - 高置信度：边侧直接输出
   - 低置信度：发送到云侧推理
   - 统计准确率、云侧调用率、推理时间
4. 输出所有阈值的性能对比

**输出指标**：
- 整体准确率
- 边侧准确率
- 云侧准确率
- 云侧调用率
- 平均推理时间
- 吞吐量

---

## 数据源

边侧支持两种数据源：

### 1. local 数据源
从本地文件加载数据（pkl 或 mat 格式）

**适用场景**：
- 离线测试
- 使用预先准备的测试集
- 可重复实验

### 2. device 数据源
从端侧设备实时接收数据（ZeroMQ）

**适用场景**：
- 实时推理
- 模拟真实部署环境
- 端到端测试

---

## 模型保存路径

- 知识蒸馏模型：`run/edge/pth/{dataset_type}/edge_{edge_id}_kd_model.pth`
- 联邦学习模型：`run/edge/pth/{dataset_type}/edge_{edge_id}_federated_model.pth`

---

## 数据路径配置

**重要**：所有边侧脚本现在都需要通过 `--data_dir` 参数指定数据集根目录，不再使用硬编码路径。

**数据目录结构**：
```
{data_dir}/
├── link11/
│   ├── edge_1_data.pkl
│   ├── edge_2_data.pkl
│   └── edge_3_data.pkl
├── rml2016/
│   ├── edge_1_data.pkl
│   └── ...
└── radar/
    ├── edge_1_data.pkl
    └── ...
```

---

## 网络配置

- `--cloud_host`: 云侧主机地址（默认 localhost）
- `--cloud_port`: 云侧端口（默认 9999）
- `--rate_limit`: 网络速率限制（MB/s，默认 10.0）
- `--device_port`: 设备数据接收端口（默认 5555）

---

## 完整工作流程示例

### 1. 知识蒸馏 → 联邦学习 → 协同推理

```bash
# 步骤1：云侧预训练教师模型
python run/cloud/run_cloud_pretrain.py \
  --dataset_type link11 \
  --data_path E:/3dataset \
  --num_classes 7 \
  --epochs 100

# 步骤2：云侧启动知识蒸馏服务器
python run/cloud/run_cloud_kd.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth

# 步骤3：边侧进行知识蒸馏训练（多个边侧并行）
python run/edge/run_edge_kd.py \
  --edge_id 1 \
  --cloud_host localhost \
  --dataset_type link11 \
  --data_dir E:/3dataset \
  --num_classes 7 \
  --epochs 50

# 步骤4：云侧启动联邦学习服务器
python run/cloud/run_cloud_federated.py \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --num_edges 3 \
  --num_rounds 10

# 步骤5：边侧进行联邦学习（多个边侧并行）
python run/edge/run_edge_federated.py \
  --edge_id 1 \
  --cloud_host localhost \
  --dataset_type link11 \
  --data_dir E:/3dataset \
  --num_classes 7 \
  --num_rounds 10

# 步骤6：云侧启动协同推理服务器
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth

# 步骤7：边侧进行协同推理测试
python run/edge/run_edge_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model_path run/edge/pth/link11/edge_1_federated_model.pth \
  --data_source local \
  --data_path E:/3dataset/link11/test_data.pkl \
  --cloud_host localhost \
  --thresholds "0.5,0.6,0.7,0.8,0.9"
```

---

## 注意事项

1. **边侧ID**：每个边侧必须有唯一的 `edge_id`
2. **数据路径**：必须通过 `--data_dir` 指定数据集根目录
3. **模型兼容性**：边侧模型类型必须与云侧一致
4. **网络连接**：确保能连接到云侧服务器
5. **数据格式**：预划分数据必须包含 train/val/test 三个集合

---

## 支持的数据集

- **link11**: Link11 通信信号识别（7类）
- **rml2016**: RadioML 2016 调制识别（6类）
- **radar**: 雷达信号识别（7类）

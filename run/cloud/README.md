# 云侧脚本

云侧负责运行教师模型（大模型），提供预训练、知识蒸馏、联邦学习和协同推理服务。

## 脚本列表

### 1. run_cloud_pretrain.py
**功能**：预训练教师模型（云侧大模型）

**用途**：
- 使用完整数据集训练复杂的教师模型
- 为后续知识蒸馏提供高质量的教师模型

**使用示例**：
```bash
python run/cloud/run_cloud_pretrain.py \
  --dataset_type link11 \
  --data_path E:/3dataset \
  --num_classes 7 \
  --cloud_model complex_resnet50_link11 \
  --batch_size 128 \
  --epochs 100 \
  --learning_rate 0.001
```

**主要参数**：
- `--dataset_type`: 数据集类型 (link11/rml2016/radar)
- `--data_path`: 数据集根目录路径
- `--num_classes`: 分类类别数
- `--cloud_model`: 云侧模型类型
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率

**输出**：
- 训练好的模型保存在 `run/cloud/pth/{dataset_type}/cloud_pretrain_model.pth`

---

### 2. run_cloud_kd.py
**功能**：知识蒸馏服务器

**用途**：
- 接收边侧学生模型的蒸馏请求
- 使用教师模型生成软标签
- 返回软标签给边侧进行知识蒸馏训练

**使用示例**：
```bash
python run/cloud/run_cloud_kd.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999 \
  --rate_limit 10.0
```

**主要参数**：
- `--dataset_type`: 数据集类型
- `--num_classes`: 分类类别数
- `--cloud_model_path`: 预训练教师模型路径
- `--cloud_model`: 云侧模型类型（可选）
- `--cloud_port`: 监听端口（默认 9999）
- `--rate_limit`: 网络速率限制（MB/s）

**工作流程**：
1. 加载预训练的教师模型
2. 启动服务器监听边侧连接
3. 接收边侧发送的数据批次
4. 使用教师模型生成软标签
5. 返回软标签给边侧

---

### 3. run_cloud_federated.py
**功能**：联邦学习服务器

**用途**：
- 管理多个边侧的联邦学习过程
- 分发全局模型给边侧
- 聚合边侧上传的本地模型
- 更新全局模型

**使用示例**：
```bash
python run/cloud/run_cloud_federated.py \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --num_edges 3 \
  --num_rounds 10 \
  --cloud_port 9999 \
  --rate_limit 10.0
```

**主要参数**：
- `--dataset_type`: 数据集类型
- `--num_classes`: 分类类别数
- `--edge_model`: 边侧模型类型
- `--num_edges`: 期望的边侧数量
- `--num_rounds`: 联邦学习轮次
- `--cloud_port`: 监听端口
- `--rate_limit`: 网络速率限制（MB/s）
- `--download_timeout`: 下载阶段超时（秒，默认 120）
- `--upload_timeout`: 上传阶段超时（秒，默认 300）

**工作流程**：
1. 初始化全局模型
2. 等待边侧连接
3. 每轮：
   - 发送全局模型给所有边侧
   - 等待边侧上传本地更新
   - 聚合本地模型（加权平均）
   - 更新全局模型
4. 保持在线，支持动态边侧数量调整

---

### 4. run_cloud_collaborative.py
**功能**：云边协同推理服务器（多线程）

**用途**：
- 接收边侧或端侧的推理请求
- 使用教师模型进行深度推理
- 支持多个客户端同时连接
- 统计推理准确率（端侧直连模式）

**使用示例**：
```bash
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth \
  --cloud_port 9999 \
  --rate_limit 10.0
```

**主要参数**：
- `--dataset_type`: 数据集类型
- `--num_classes`: 分类类别数
- `--cloud_model_path`: 教师模型路径
- `--cloud_model`: 云侧模型类型（可选）
- `--cloud_port`: 监听端口
- `--rate_limit`: 网络速率限制（MB/s）

**支持的请求类型**：
1. **cloud_inference**: 推理请求
   - 来自边侧：返回预测结果
   - 来自端侧：计算并输出准确率
2. **end_transmission**: 结束标志
   - 输出整体推理统计汇总

**特性**：
- 多线程处理，支持多个客户端同时连接
- 持久连接，减少连接开销
- 自动识别请求来源（边侧/端侧）
- 实时统计推理准确率

---

## 网络配置

所有云侧脚本支持以下网络参数：

- `--cloud_port`: 监听端口（默认 9999）
- `--rate_limit`: 网络速率限制（MB/s，默认 10.0）
  - 用于模拟真实网络环境
  - 设置为 None 或 0 表示不限速

## 模型保存路径

- 预训练模型：`run/cloud/pth/{dataset_type}/cloud_pretrain_model.pth`
- 联邦学习模型：由边侧保存

## 注意事项

1. **端口冲突**：确保云侧端口未被占用
2. **模型路径**：使用绝对路径或相对于项目根目录的路径
3. **数据集类型**：云侧和边侧必须使用相同的数据集类型
4. **类别数**：必须与数据集匹配
5. **网络限速**：可根据实验需求调整

## 工作流程示例

### 完整训练流程

```bash
# 1. 预训练教师模型
python run/cloud/run_cloud_pretrain.py \
  --dataset_type link11 \
  --data_path E:/3dataset \
  --num_classes 7 \
  --epochs 100

# 2. 启动知识蒸馏服务器
python run/cloud/run_cloud_kd.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth

# 3. 启动联邦学习服务器
python run/cloud/run_cloud_federated.py \
  --dataset_type link11 \
  --num_classes 7 \
  --edge_model real_resnet20_link11 \
  --num_edges 3 \
  --num_rounds 10

# 4. 启动协同推理服务器
python run/cloud/run_cloud_collaborative.py \
  --dataset_type link11 \
  --num_classes 7 \
  --cloud_model_path run/cloud/pth/link11/cloud_pretrain_model.pth
```

## 支持的数据集

- **link11**: Link11 通信信号识别（7类）
- **rml2016**: RadioML 2016 调制识别（6类）
- **radar**: 雷达信号识别（7类）

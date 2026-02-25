# 异构联邦学习与云边协同推理系统

本项目实现了基于知识蒸馏的异构联邦学习系统，以及云边协同推理框架。

## 目录结构

```
run/
├── run_server.py          # 云侧端启动脚本（联邦学习/云端推理）
├── run_client.py          # 客户端启动脚本（本地训练/边侧推理）
├── data/                  # 数据集目录
│   ├── rml2016.pkl       # RML2016 单文件数据集
│   ├── link11.pkl        # Link11 单文件数据集
│   └── radar.mat         # Radar 数据集
├── pth/                   # 模型权重目录
│   ├── rml2016/          # RML2016 模型
│   │   ├── teacher_model.pth
│   │   └── client_1_kd_model.pth
│   ├── link11/           # Link11 模型
│   │   ├── teacher_model.pth
│   │   └── client_1_kd_model.pth
│   └── radar/            # Radar 模型
│       ├── teacher_model.pth
│       └── client_1_kd_model.pth
└── .env.*                # 环境配置文件
```

## 支持的数据集

| 数据集 | 类别数 | 信号长度 | 输入形状 |
|--------|--------|----------|----------|
| RML2016 | 6 | 600 | (batch_size, 600) complex |
| Link11 | 7 | 1024 | (batch_size, 1024) complex |
| Radar | 7 | 500 | (batch_size, 500) complex |

### RML2016 调制类型
`['8PSK', 'BPSK', 'GMSK', '16QAM', '64QAM', 'QPSK']`

### Link11 发射器类型
`['E-2D_1', 'E-2D_2', 'P-3C_1', 'P-3C_2', 'P-8A_1', 'P-8A_2', 'P-8A_3']`

---

## 运行模式

### 1. 异构联邦学习模式（Federated Learning）

基于知识蒸馏的异构联邦学习，支持不同客户端使用不同架构的模型。

#### 启动步骤

**第一步：启动云侧**
```cmd
python run/run_server.py --mode federated --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --server_model complex_resnet50_rml2016 --num_rounds 10 --num_clients 2
```

**第二步：启动客户端 1**
```cmd
python run/run_client.py --mode federated --client_id 1 --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --client_model real_resnet20_rml2016 --num_rounds 10 --local_epochs 5
```

**第三步：启动客户端 2**
```cmd
python run/run_client.py --mode federated --client_id 2 --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --client_model real_resnet20_rml2016 --num_rounds 10 --local_epochs 5
```

#### 关键参数说明

- `--mode federated`: 联邦学习模式
- `--num_rounds`: 联邦学习轮数
- `--num_clients`: 客户端数量
- `--local_epochs`: 本地训练轮数
- `--server_model`: 云侧模型架构（教师模型）
- `--client_model`: 客户端模型架构（学生模型）

---

### 2. 云边协同推理模式（Collaborative Inference）

边侧设备根据置信度阈值决定是否将样本上传到云端进行深度推理。

#### 数据加载方式

##### 方式 A：单文件加载（一次性加载到内存）

**启动云侧（云端）：**
```cmd
python run/run_server.py --mode collaborative --server_host localhost --server_port 9999 --cloud_model_path run/pth/rml2016/teacher_model.pth --dataset_type rml2016 --num_classes 6
```

**启动客户端（边侧）：**
```cmd
python run/run_client.py --mode collaborative --client_id 1 --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --edge_model_path run/pth/rml2016/client_1_kd_model.pth --thresholds 0.5 --batch_size 32
```

##### 方式 B：批次加载（边加载边推理，节省内存）

**启动云侧（云端）：**
```cmd
python run/run_server.py --mode collaborative --server_host localhost --server_port 9999 --cloud_model_path run/pth/rml2016/teacher_model.pth --dataset_type rml2016 --num_classes 6
```

**启动客户端（边侧）：**
```cmd
python run/run_client.py --mode collaborative --client_id 1 --server_host localhost --server_port 9999 --data_path E:/BaiduNet_Download/new3/rml2016 --dataset_type rml2016 --num_classes 6 --edge_model_path run/pth/rml2016/client_1_kd_model.pth --thresholds 0.5 --batch_size 32
```

**批次数据格式：**
- 数据文件夹包含多个批次文件：`batch_0000.pkl`, `batch_0001.pkl`, ...
- 每个批次文件格式：`{(mod_type, snr): signal_array}`
- 信号数组形状：`(num_samples, 2, signal_length)`

#### 关键参数说明

- `--mode collaborative`: 云边协同推理模式
- `--cloud_model_path`: 云端模型路径（教师模型）
- `--edge_model_path`: 边侧模型路径（学生模型）
- `--thresholds`: 置信度阈值（可以指定多个，如 `0.3 0.5 0.7`）
- `--batch_size`: 批次大小
- `--data_path`: 数据路径（文件夹=批次加载，.pkl文件=单文件加载）

---

## 完整命令示例

### RML2016 数据集

#### 联邦学习
```cmd
# 云侧
python run/run_server.py --mode federated --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --server_model complex_resnet50_rml2016 --num_rounds 10 --num_clients 2

# 客户端 1
python run/run_client.py --mode federated --client_id 1 --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --client_model real_resnet20_rml2016 --num_rounds 10 --local_epochs 5

# 客户端 2
python run/run_client.py --mode federated --client_id 2 --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --client_model real_resnet20_rml2016 --num_rounds 10 --local_epochs 5
```

#### 云边协同推理（单文件）
```cmd
# 云端
python run/run_server.py --mode collaborative --server_host localhost --server_port 9999 --cloud_model_path run/pth/rml2016/teacher_model.pth --dataset_type rml2016 --num_classes 6

# 边侧
python run/run_client.py --mode collaborative --client_id 1 --server_host localhost --server_port 9999 --data_path run/data/rml2016.pkl --dataset_type rml2016 --num_classes 6 --edge_model_path run/pth/rml2016/client_1_kd_model.pth --thresholds 0.5 --batch_size 32
```

#### 云边协同推理（批次加载）
```cmd
# 云端
python run/run_server.py --mode collaborative --server_host localhost --server_port 9999 --cloud_model_path run/pth/rml2016/teacher_model.pth --dataset_type rml2016 --num_classes 6

# 边侧
python run/run_client.py --mode collaborative --client_id 1 --server_host localhost --server_port 9999 --data_path E:/BaiduNet_Download/new3/rml2016 --dataset_type rml2016 --num_classes 6 --edge_model_path run/pth/rml2016/client_1_kd_model.pth --thresholds 0.5 --batch_size 32
```

---

### Link11 数据集

#### 联邦学习
```cmd
# 云侧
python run/run_server.py --mode federated --server_host localhost --server_port 9999 --data_path run/data/link11.pkl --dataset_type link11 --num_classes 7 --server_model complex_resnet50_link11 --num_rounds 10 --num_clients 2

# 客户端 1
python run/run_client.py --mode federated --client_id 1 --server_host localhost --server_port 9999 --data_path run/data/link11.pkl --dataset_type link11 --num_classes 7 --client_model real_resnet20_link11 --num_rounds 10 --local_epochs 5

# 客户端 2
python run/run_client.py --mode federated --client_id 2 --server_host localhost --server_port 9999 --data_path run/data/link11.pkl --dataset_type link11 --num_classes 7 --client_model real_resnet20_link11 --num_rounds 10 --local_epochs 5
```

#### 云边协同推理（单文件）
```cmd
# 云端
python run/run_server.py --mode collaborative --server_host localhost --server_port 9999 --cloud_model_path run/pth/link11/teacher_model.pth --dataset_type link11 --num_classes 7

# 边侧
python run/run_client.py --mode collaborative --client_id 1 --server_host localhost --server_port 9999 --data_path run/data/link11.pkl --dataset_type link11 --num_classes 7 --edge_model_path run/pth/link11/client_1_kd_model.pth --thresholds 0.5 --batch_size 32
```

#### 云边协同推理（批次加载）
```cmd
# 云端
python run/run_server.py --mode collaborative --server_host localhost --server_port 9999 --cloud_model_path run/pth/link11/teacher_model.pth --dataset_type link11 --num_classes 7

# 边侧
python run/run_client.py --mode collaborative --client_id 1 --server_host localhost --server_port 9999 --data_path E:/BaiduNet_Download/new3/link11 --dataset_type link11 --num_classes 7 --edge_model_path run/pth/link11/client_1_kd_model.pth --thresholds 0.5 --batch_size 32
```

---

## 注意事项

1. **启动顺序**：先启动云侧，等待云侧准备完成后再启动客户端
2. **数据路径**：
   - 单文件加载：使用 `.pkl` 文件路径
   - 批次加载：使用包含批次文件的文件夹路径
3. **模型架构**：
   - 教师模型（云侧/云端）：`complex_resnet50_{dataset_type}`
   - 学生模型（客户端/边侧）：`real_resnet20_{dataset_type}`
4. **批次加载**：适用于大规模数据集，节省内存，边加载边推理
5. **置信度阈值**：可以测试多个阈值，如 `--thresholds 0.3 0.5 0.7`

---

## 系统架构

### 联邦学习流程
1. 云侧预训练教师模型
2. 客户端通过知识蒸馏获得学生模型
3. 多轮联邦学习：
   - 客户端下载全局模型
   - 本地训练
   - 上传模型更新
   - 云侧聚合更新

### 云边协同推理流程
1. 边侧设备使用轻量级学生模型进行推理
2. 根据预测置信度判断：
   - 高置信度：边侧直接输出结果
   - 低置信度：上传到云端使用教师模型推理
3. 统计边侧/云侧推理比例和准确率

---

## 依赖环境

- Python 3.8+
- PyTorch 1.10+
- NumPy
- scikit-learn
- pickle

---

## 常见问题

**Q: 批次加载和单文件加载有什么区别？**
A: 批次加载逐个加载批次文件，节省内存，适合大规模数据集；单文件加载一次性加载所有数据到内存，速度更快但占用内存大。

**Q: 如何准备批次数据？**
A: 批次数据应该是多个 `.pkl` 文件，命名为 `batch_0000.pkl`, `batch_0001.pkl` 等，每个文件包含字典格式 `{(label, snr): signal_array}`。

**Q: 支持哪些数据集？**
A: 目前支持 RML2016（6类）、Link11（7类）、Radar（7类）三个数据集。

**Q: 如何调整置信度阈值？**
A: 使用 `--thresholds` 参数，可以指定单个或多个阈值，如 `--thresholds 0.5` 或 `--thresholds 0.3 0.5 0.7`。

---

## 联系方式

如有问题，请联系项目维护者。

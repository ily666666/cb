# 云边端协同信号识别系统

基于 PyTorch 的云边端协同推理与训练框架，采用**任务驱动 + 文件总线**架构，支持雷达/通信信号的分布式识别。

## 目录结构

```
project0210/
├── run_task.py              # 统一入口：执行推理/训练流水线
├── config_refactor.py       # 全局配置：流水线定义、数据集参数
│
├── callback/                # 业务回调层（核心逻辑）
│   ├── registry.py          #   装饰器注册机制
│   ├── device_callback.py   #   端侧：加载 PKL 数据 → 标准化输出
│   ├── edge_callback.py     #   边侧：轻量模型推理 + 置信度筛选
│   ├── cloud_callback.py    #   云侧：大模型推理（协同/直接两种模式）
│   └── train_callback.py    #   训练：预训练、知识蒸馏、联邦学习
│
├── core/                    # 核心模块
│   └── model_factory.py     #   模型工厂：按名称创建模型实例
│
├── model/                   # 模型定义
│   ├── complex_resnet50_*   #   云侧教师模型（复数卷积 ResNet50）
│   └── real_resnet20_*      #   边侧学生模型（实数卷积 ResNet20）
│
├── utils_refactor/          # 工具层
│   ├── io_helper.py         #   文件 I/O（JSON / PKL / NPY）+ 网络传输模拟
│   ├── param_checker.py     #   参数校验
│   └── task_manager.py      #   任务目录管理
│
├── tasks/                   # 任务实例（每个任务一个文件夹）
│   ├── 001_COLLAB_link11_test/      # link11 协同推理
│   ├── 002_cloud_only_link11_test/  # link11 仅云推理
│   ├── 003_edge_only_link11_test/   # link11 仅边推理
│   ├── 004_train_link11/            # link11 训练
│   ├── 005_COLLAB_rml2016_test/     # rml2016 协同推理
│   ├── 006_cloud_only_rml2016_test/ # rml2016 仅云推理
│   ├── 007_edge_only_rml2016_test/  # rml2016 仅边推理
│   ├── 008_COLLAB_radar_test/       # radar 协同推理
│   ├── 009_cloud_only_radar_test/   # radar 仅云推理
│   ├── 010_edge_only_radar_test/    # radar 仅边推理
│   ├── 011_train_rml2016/           # rml2016 训练
│   └── 012_train_radar/             # radar 训练
│
├── cloud_pth/               # 云侧模型权重
├── edge_pth/                # 边侧模型权重
├── dataset/                 # 数据集目录（PKL 文件）
├── fed/                     # 联邦学习核心（FedAvg 等）
└── helper/                  # 知识蒸馏辅助模块
```

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│                    run_task.py                        │
│           (读取 mode → 查 PIPELINE_MODES)             │
└───────────────────────┬──────────────────────────────┘
                        │ 按顺序调用
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  device_load      edge_infer      cloud_infer
  (端侧回调)       (边侧回调)      (云侧回调)
        │               │               │
        ▼               ▼               ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ input/   │   │ input/   │   │ input/   │  ← JSON 配置
  │ output/  │──▶│ output/  │──▶│ output/  │  ← 文件总线（PKL/NPY）
  │ result/  │   │ result/  │   │ result/  │  ← 最终报告
  └──────────┘   └──────────┘   └──────────┘
```

**核心设计**：步骤之间通过 `output/` 目录下的文件传递数据（文件总线），而非内存或网络。这使得各步骤可以在不同机器上执行，只需通过外部工具同步文件即可。

## 系统部署架构

本系统采用 **1 云 + 2 边 + 4 端** 的三层部署架构：

```
                        ┌─────────────────────┐
                        │     云 (Cloud)        │
                        │  complex_resnet50     │
                        │  教师模型 / 大模型推理  │
                        │  联邦聚合中心          │
                        └──────────┬──────────┘
                     ┌─────────────┴─────────────┐
                     │                           │
              ┌──────┴──────┐             ┌──────┴──────┐
              │   边 1       │             │   边 2       │
              │ real_resnet20│             │ real_resnet20│
              │ 轻量推理     │             │ 轻量推理     │
              │ 联邦学习节点  │             │ 联邦学习节点  │
              └──┬───────┬──┘             └──┬───────┬──┘
                 │       │                   │       │
              ┌──┴──┐ ┌──┴──┐           ┌───┴──┐ ┌──┴──┐
              │ 端1  │ │ 端2  │           │ 端3  │ │ 端4  │
              │ 数据  │ │ 数据  │           │ 数据  │ │ 数据  │
              │ 采集  │ │ 采集  │           │ 采集  │ │ 采集  │
              └─────┘ └─────┘           └─────┘ └─────┘
```

| 层级 | 数量 | 角色 | 模型 |
|------|------|------|------|
| 云 | 1 | 教师模型训练、软标签生成、联邦聚合、大模型推理 | `complex_resnet50_xxx` |
| 边 | 2 | 学生模型推理、联邦本地训练、低置信度筛选 | `real_resnet20_xxx` |
| 端 | 4（每边 2 个） | 信号采集、数据加载、PKL 读取 | 无模型 |

**数据流向**：
- **推理**：端采集数据 → 边侧轻量推理 → 低置信度样本转发云侧大模型
- **训练**：云侧预训练教师 → 蒸馏知识给学生 → 各边侧联邦学习（数据不出边）
- **通信方式**：文件总线（`output/` 目录），不依赖实时网络连接

---

## 数据集生成与划分

训练和推理前需要先生成数据集并按云/边/端切分。

### 第一步：生成原始信号数据

```bash
python training_data_generation/link11_gene5.py
```

该脚本模拟 Link11 通信链路，生成 7 类飞机（E-2D × 2、P-3C × 2、P-8A × 3）的 I/Q 信号数据：

- 每个飞机沿航线生成多个航点的信号（`num_waypoints` 控制，默认 100000）
- 双接收机同时接收，信噪比随距离动态变化
- 输出：`run/data/link11.pkl`
- 格式：`{(个体ID, SNR): ndarray(样本数, 2, 1024)}`，其中 2 = I/Q 双通道

> **提示**：首次测试可将 `link11_gene5.py` 中的 `num_waypoints` 改小（如 1000），加快生成速度。

### 第二步：切分为云侧 + 边侧数据

```bash
python run/prepare_data_splits.py --dataset_type link11 --num_edges 2 --output_dir dataset/splits
```

该脚本将完整数据集按比例切分为云侧和各边侧的独立文件：

```
原始 link11.pkl
    │
    ├── 80% 训练 / 10% 验证 / 10% 测试（分层采样）
    │
    ├── 云侧 30%（cloud_ratio=0.3）
    │   └── dataset/splits/link11/cloud_data.pkl
    │
    └── 边侧 70%（Dirichlet 分布划分，模拟 Non-IID）
        ├── dataset/splits/link11/edge_1_data.pkl  ← 边1（端1+端2 的数据）
        └── dataset/splits/link11/edge_2_data.pkl  ← 边2（端3+端4 的数据）
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_type` | 数据集类型：`link11` / `rml2016` / `radar` | `link11` |
| `--num_edges` | 边侧数量 | `2` |
| `--dirichlet_alpha` | Dirichlet 参数，越小各边数据越异构 | `0.3` |
| `--cloud_ratio` | 云侧数据占比 | `0.3` |
| `--output_dir` | 输出目录 | `dataset/splits` |
| `--seed` | 随机种子，保证可复现 | `42` |

每个输出 PKL 文件的内部格式均为：

```python
{
    'train': (X_array, y_array),   # 训练集
    'val':   (X_array, y_array),   # 验证集
    'test':  (X_array, y_array)    # 测试集
}
```

### 第三步：推理用测试数据

推理时的输入数据在 `dataset/link11/` 目录下，以 `batch_xxxx.pkl` 形式存放，由端侧的 `device_load` 步骤加载。**运行推理测试前，必须先生成测试数据**：

```bash
# 生成推理测试数据（7类×2接收站×10万航点，约140个batch，每个batch 10000样本）
# 直接输出到 dataset/link11/ 目录
python test_data_generation/link11_gene5_batch.py
```

生成后 `dataset/link11/` 目录结构如下：

```
dataset/link11/
├── batch_0000.pkl    # 10000 样本 (~80MB)
├── batch_0001.pkl
├── ...
└── batch_0139.pkl    # 共约 140 个 batch
```

> **提示**：推理配置 `device_load.json` 中的 `max_files` 参数控制加载几个 batch 文件。设为 `3` 可快速测试，设为 `null` 加载全部。

### 完整数据准备流程

#### link11 数据集

```bash
# 1. 生成推理测试数据（自动输出到 dataset/link11/）
python test_data_generation/link11_gene5_batch.py

# 2. 生成训练用原始信号数据
python training_data_generation/link11_gene5.py

# 3. 切分为云侧 + 2个边侧
python run/prepare_data_splits.py --dataset_type link11 --num_edges 2 --output_dir dataset/splits

# 4. 跑训练
python run_task.py --mode full_train --task_id 004_train_link11

# 5. 跑推理
python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test
```

#### rml2016 数据集

```bash
# 1. 生成推理测试数据（自动输出到 dataset/rml2016/）
python test_data_generation/rml2016_gene9_batch.py

# 2. 生成训练用原始信号数据
python training_data_generation/rml2016_gene9.py

# 3. 切分为云侧 + 2个边侧
python run/prepare_data_splits.py --dataset_type rml2016 --num_edges 2 --output_dir dataset/splits

# 4. 跑训练
python run_task.py --mode full_train --task_id 011_train_rml2016

# 5. 跑推理
python run_task.py --mode device_to_edge_to_cloud --task_id 005_COLLAB_rml2016_test
```

#### radar 数据集

```bash
# 1. 生成推理测试数据（自动输出到 dataset/radar/）
python test_data_generation/radar_signal_generator_batch.py
#    （MATLAB 版本：在 MATLAB 中执行 test_data_generation/RadarSignalGenerator_Batch.m）

# 2. 生成训练用原始信号数据
python training_data_generation/radar_signal_generator.py
#    （MATLAB 版本：在 MATLAB 中执行 training_data_generation/RadarSignalGenerator_Batch2.m）

# 3. 切分为云侧 + 2个边侧
python run/prepare_data_splits.py --dataset_type radar --num_edges 2 --output_dir dataset/splits

# 4. 跑训练
python run_task.py --mode full_train --task_id 012_train_radar

# 5. 跑推理
python run_task.py --mode device_to_edge_to_cloud --task_id 008_COLLAB_radar_test
```

> **注意**：每个数据集中，步骤 1（推理测试数据）和步骤 2-3（训练数据）是独立的。只跑推理只需完成步骤 1，只跑训练只需完成步骤 2-3。radar 数据集同时保留了 MATLAB 版本（.m）和 Python 版本（.py），功能等效，任选其一即可。

---

## 所有流水线模式一览

| 模式名 | 流水线 | 用途 |
|--------|--------|------|
| `device_to_edge_to_cloud` | device_load → edge_infer → cloud_infer | 协同推理 |
| `device_to_cloud` | device_load → cloud_direct_infer | 仅云侧推理 |
| `device_to_edge` | device_load → edge_infer | 仅边侧推理 |
| `pretrain` | cloud_pretrain | 预训练教师模型 |
| `knowledge_distillation` | cloud_kd → edge_kd | 知识蒸馏（需已有教师模型） |
| `federated_learning` | federated_train | 联邦学习（单机模拟） |
| `federated_server` | federated_server | 联邦学习-云侧聚合（分布式） |
| `federated_edge` | federated_edge | 联邦学习-边侧训练（分布式） |
| `full_train` | cloud_pretrain → cloud_kd → edge_kd → federated_train | 完整训练 |
| `full_pipeline` | 完整训练 + 协同推理（共 7 步） | 从训练到推理一条龙 |

---

## 三种推理模式

> **前置条件**：运行推理前，需先用 `python test_data_generation/link11_gene5_batch.py` 生成测试数据并放到 `dataset/link11/` 目录下，详见上方"第三步：推理用测试数据"。

### 模式 1：端 → 边 → 云（协同推理）

适合**精度优先**场景。边侧先用轻量模型快速推理，低置信度样本交给云侧大模型二次判断。

```
device_load → edge_infer → cloud_infer
```

**所需配置文件**（放在 `tasks/{task_id}/input/` 下）：

`device_load.json` — 数据加载配置：
```json
{
    "data_path": "dataset/link11",
    "dataset_type": "link11",
    "batch_size": 128,
    "max_files": 3,
    "num_batches": null
}
```

`edge_infer.json` — 边侧推理配置：
```json
{
    "model_path": "edge_pth/link11/edge_1_federated_model.pth",
    "model_type": "real_resnet20_link11_h",
    "num_classes": 7,
    "device": "cuda:0",
    "confidence_threshold": 0.7,
    "batch_size": 128,
    "simulate_bandwidth_mbps": 100,
    "input_data": {
        "source": "device_load",
        "file_name": "data_batch.pkl"
    }
}
```

`cloud_infer.json` — 云侧推理配置：
```json
{
    "model_path": "cloud_pth/link11/teacher_model.pth",
    "model_type": "complex_resnet50_link11_with_attention",
    "num_classes": 7,
    "device": "cuda:0",
    "batch_size": 128,
    "simulate_bandwidth_mbps": 10,
    "input_data": {
        "source": "edge_infer",
        "signals_file": "low_conf_signals.pkl"
    }
}
```

**运行命令**：
```bash
python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test
```

### 模式 2：端 → 云（直接推理）

适合**不部署边侧模型**的场景，全部数据直接由云侧大模型推理。

```
device_load → cloud_direct_infer
```

**所需配置文件**：`device_load.json` + `cloud_infer.json`（`input_data.source` 指向 `device_load`）。

**运行命令**：
```bash
python run_task.py --mode device_to_cloud --task_id 002_cloud_only_link11_test
```

### 模式 3：端 → 边（仅边侧推理）

适合**低延迟**场景，仅使用边侧轻量模型，不依赖云侧。

```
device_load → edge_infer
```

**所需配置文件**：`device_load.json` + `edge_infer.json`（**不写** `confidence_threshold`，纯边侧模式不筛选低置信度）。

**运行命令**：
```bash
python run_task.py --mode device_to_edge --task_id 003_edge_only_link11_test
```

---

## 训练流程

训练遵循 **预训练 → 知识蒸馏 → 联邦学习** 的三阶段流程，每个阶段是独立的流水线步骤，通过文件传递中间产物。

```
cloud_pretrain ──→ cloud_kd ──→ edge_kd ──→ federated_train
 (预训练教师模型)  (生成软标签)  (蒸馏学生模型)  (联邦学习聚合)
       │                │            │              │
       ▼                ▼            ▼              ▼
  teacher_model.pth  soft_labels.pkl  student_model.pth  global_model.pth
                                                         edge_1_model.pth
                                                         edge_2_model.pth
```

### 阶段 1：预训练教师模型（cloud_pretrain）

在云侧用全量数据训练一个大的教师模型（`complex_resnet50`），作为后续知识蒸馏的知识源。

**配置文件** `input/cloud_pretrain.json`：
```json
{
    "data_path": "dataset/splits/link11/cloud_data.pkl",
    "dataset_type": "link11",
    "model_type": "complex_resnet50_link11_with_attention",
    "num_classes": 7,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "device": "cuda:0"
}
```

| 参数 | 说明 |
|------|------|
| `data_path` | 云侧预划分数据文件（PKL），由 `prepare_data_splits.py` 生成，包含 train/val/test |
| `model_type` | 教师模型类型，通常用 `complex_resnet50_xxx` 系列 |
| `epochs` | 训练轮数 |

**输出**：
- `output/cloud_pretrain/teacher_model.pth` — 最佳验证准确率的模型权重
- `output/cloud_pretrain/train_history.pkl` — 训练曲线数据
- `result/cloud_pretrain/pretrain_report.txt` — 训练报告

**单独运行**：
```bash
python run_task.py --mode pretrain --task_id my_train_task
```

**数据格式要求**：`data_path` 指向的 PKL 文件内部需包含训练/测试划分，支持以下格式：
```python
# 格式 A：字典含 train/val/test
{'train': {'X': array, 'y': array}, 'val': {...}, 'test': {...}}

# 格式 B：字典含 X_train/y_train 等
{'X_train': array, 'y_train': array, 'X_test': array, 'y_test': array}

# 格式 C：只有 X/y，自动按 70%/15%/15% 划分
{'X': array, 'y': array}
```

### 阶段 2：知识蒸馏 — 云侧生成软标签（cloud_kd）

用训练好的教师模型对训练数据生成软标签（soft labels），保存到文件供边侧使用。

**配置文件** `input/cloud_kd.json`：
```json
{
    "teacher_model_path": "tasks/004_train_link11/output/cloud_pretrain/teacher_model.pth",
    "teacher_model_type": "complex_resnet50_link11_with_attention",
    "num_classes": 7,
    "dataset_type": "link11",
    "data_path": "dataset/splits/link11/cloud_data.pkl",
    "temperature": 4.0,
    "batch_size": 32,
    "device": "cuda:0"
}
```

| 参数 | 说明 |
|------|------|
| `teacher_model_path` | 教师模型权重路径（可用预训练阶段的输出，或已有的权重文件） |
| `temperature` | 蒸馏温度，越高软标签越平滑，通常 2.0 ~ 8.0，默认 4.0 |

**输出**：
- `output/cloud_kd/soft_labels.pkl` — 软标签数组
- `output/cloud_kd/teacher_state_dict.pkl` — 教师模型权重副本

### 阶段 3：知识蒸馏 — 边侧训练学生模型（edge_kd）

用软标签 + 硬标签混合损失训练轻量的学生模型（`real_resnet20`）。

**配置文件** `input/edge_kd.json`：
```json
{
    "student_model_type": "real_resnet20_link11_h",
    "num_classes": 7,
    "dataset_type": "link11",
    "data_path": "dataset/splits/link11/cloud_data.pkl",
    "epochs": 30,
    "kd_alpha": 0.7,
    "temperature": 4.0,
    "batch_size": 32,
    "learning_rate": 0.001,
    "device": "cuda:0"
}
```

| 参数 | 说明 |
|------|------|
| `student_model_type` | 学生模型类型，通常用 `real_resnet20_xxx` 系列 |
| `kd_alpha` | 软标签权重。0.7 表示总损失 = 30% CE + 70% KD。越大越依赖教师知识 |
| `temperature` | 蒸馏温度，需与 cloud_kd 阶段一致 |

**输出**：
- `output/edge_kd/student_model.pth` — 蒸馏后的学生模型
- `result/edge_kd/kd_report.txt` — 蒸馏报告

**一键运行知识蒸馏**（阶段 2→3，需已有教师模型）：
```bash
python run_task.py --mode knowledge_distillation --task_id my_train_task
```

### 阶段 4：联邦学习（federated_train）

模拟多个边侧节点各自用本地数据训练，然后聚合（FedAvg），可选从蒸馏模型初始化。

**配置文件** `input/federated_train.json`：
```json
{
    "edge_data_paths": [
        "dataset/splits/link11/edge_1_data.pkl",
        "dataset/splits/link11/edge_2_data.pkl"
    ],
    "dataset_type": "link11",
    "edge_model_type": "real_resnet20_link11_h",
    "num_classes": 7,
    "num_rounds": 20,
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.001,
    "device": "cuda:0",
    "init_model_path": "tasks/004_train_link11/output/edge_kd/student_model.pth"
}
```

| 参数 | 说明 |
|------|------|
| `edge_data_paths` | 各边侧的数据文件列表，每个文件代表一个边侧节点的本地数据。当前架构为 2 个边 |
| `num_rounds` | 联邦学习通信轮数 |
| `local_epochs` | 每轮中每个边侧的本地训练轮数 |
| `init_model_path` | 全局模型初始权重，可填蒸馏产出的 `student_model.pth`，`null` = 随机初始化 |

**输出**：
- `output/federated_train/global_model.pth` — 聚合后的全局模型
- `output/federated_train/edge_1_model.pth` — 边 1 最终模型
- `output/federated_train/edge_2_model.pth` — 边 2 最终模型
- `output/federated_train/train_history.pkl` — 各轮准确率
- `result/federated_train/federated_report.txt` — 联邦学习报告

**单独运行**：
```bash
python run_task.py --mode federated_learning --task_id my_train_task
```

### 完整训练流程（一键执行）

把预训练 + 知识蒸馏 + 联邦学习串在一起：

```bash
# 仅训练（4步）
python run_task.py --mode full_train --task_id my_train_task

# 训练 + 推理（7步，从训练到推理一条龙）
python run_task.py --mode full_pipeline --task_id my_train_task
```

需要在 `input/` 下同时放置所有步骤的配置 JSON。

### 现成的训练任务：004_train_link11

项目已包含一个 link11 数据集的训练任务配置：

```
tasks/004_train_link11/input/
├── cloud_pretrain.json      # 阶段1：预训练教师模型
├── cloud_kd.json            # 阶段2：生成软标签
├── edge_kd.json             # 阶段3：蒸馏训练学生模型
└── federated_train.json     # 阶段4：联邦学习
```

**运行前需要先生成数据**（参见上方「数据集生成与划分」章节）：

```bash
# 1. 生成原始信号数据
python training_data_generation/link11_gene5.py

# 2. 切分为云侧 + 2 个边侧
python run/prepare_data_splits.py --dataset_type link11 --num_edges 2 --output_dir dataset/splits
```

生成后的数据对应关系：

| 文件 | 用途 | 引用它的配置 |
|------|------|-------------|
| `dataset/splits/link11/cloud_data.pkl` | 预训练 + 蒸馏共用（云侧数据） | `cloud_pretrain.json`、`cloud_kd.json`、`edge_kd.json` |
| `dataset/splits/link11/edge_1_data.pkl` | 联邦学习-边 1 本地数据 | `federated_train.json` |
| `dataset/splits/link11/edge_2_data.pkl` | 联邦学习-边 2 本地数据 | `federated_train.json` |

**运行命令**：

```bash
# 仅预训练教师模型（阶段 1）
python run_task.py --mode pretrain --task_id 004_train_link11

# 知识蒸馏（阶段 2→3，需已有教师模型，在 cloud_kd.json 中指定 teacher_model_path）
python run_task.py --mode knowledge_distillation --task_id 004_train_link11

# 仅联邦学习（阶段 4，需先完成阶段 3 产出 student_model.pth）
python run_task.py --mode federated_learning --task_id 004_train_link11

# 一键跑完整训练（阶段 1→2→3→4 全部串联）
python run_task.py --mode full_train --task_id 004_train_link11
```

> **注意**：`knowledge_distillation` 模式**不含**预训练，需要已有教师模型权重（在 `cloud_kd.json` 的 `teacher_model_path` 中指定）。如果教师模型还没训练，先跑 `pretrain` 或直接用 `full_train` 一键完成全部阶段。各模式支持断点续传——已完成的阶段会自动跳过，删除对应 `output/` 子目录可强制重跑。

---

## 分布式部署

本系统支持将各步骤拆分到不同机器上独立执行。文件总线架构天然适配分布式——每个步骤只通过 `output/` 目录读写文件，你只需在各机器间同步该目录即可。

### 核心用法：`--step` 单步骤执行

使用 `--step` 替代 `--mode`，只执行一个步骤（两者互斥，二选一）：

```bash
python run_task.py --step <步骤名> --task_id <任务ID> [--edge_id N] [--summary]
```

所有可用步骤：`device_load`、`edge_infer`、`cloud_infer`、`cloud_direct_infer`、`cloud_pretrain`、`cloud_kd`、`edge_kd`、`federated_train`、`federated_server`、`federated_edge`

**`--summary` 开关**：`--step` 模式下默认只显示当前步骤的耗时。在最后一步加上 `--summary`，会扫描该 task 下所有已有的 `timing.json`，显示全局耗时汇总（纯推理、加载+热身、传输等），并写入 `result/timing_summary.txt` 及追加到各报告文件末尾。`--mode` 模式自动开启，无需手动加。

### 推理任务：单机 vs 多机命令对照

#### 1. 仅云侧推理 (`device_to_cloud`)

| | 命令 |
|---|------|
| **单机** | `python run_task.py --mode device_to_cloud --task_id 002_cloud_only_link11_test` |

```bash
# 多机拆分（2 步）：
# [端侧机器] 加载数据
python run_task.py --step device_load --task_id 002_cloud_only_link11_test
# >>> 同步 tasks/002_cloud_only_link11_test/output/device_load/ 到云侧机器 <<<
# [云侧机器] 推理（最后一步加 --summary 查看全局耗时汇总）
python run_task.py --step cloud_direct_infer --task_id 002_cloud_only_link11_test --summary
```

#### 2. 仅边侧推理 (`device_to_edge`)

| | 命令 |
|---|------|
| **单机** | `python run_task.py --mode device_to_edge --task_id 003_edge_only_link11_test` |

```bash
# 多机拆分（2 步）：
# [端侧机器]
python run_task.py --step device_load --task_id 003_edge_only_link11_test
# >>> 同步 output/device_load/ 到边侧机器 <<<
# [边侧机器]（最后一步加 --summary）
python run_task.py --step edge_infer --task_id 003_edge_only_link11_test --summary
```

#### 3. 云边协同推理 (`device_to_edge_to_cloud`)

| | 命令 |
|---|------|
| **单机** | `python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test` |

```bash
# 多机拆分（3 步）：
# [端侧机器]
python run_task.py --step device_load --task_id 001_COLLAB_link11_test
# >>> 同步 output/device_load/ 到边侧机器 <<<
# [边侧机器]
python run_task.py --step edge_infer --task_id 001_COLLAB_link11_test
# >>> 同步 output/edge_infer/ 到云侧机器 <<<
# [云侧机器]（最后一步加 --summary）
python run_task.py --step cloud_infer --task_id 001_COLLAB_link11_test --summary
```

### 训练任务：单机 vs 多机命令对照

#### 4. 完整训练 (`full_train`：预训练 → 知识蒸馏 → 联邦学习)
#### 请准备好教师模型直接看5

| | 命令 |
|---|------|
| **单机** | `python run_task.py --mode full_train --task_id 004_train_link11` |

```bash
# 多机拆分（6 步，3 台机器）：

# ====== 阶段 1+2：云侧预训练 + 生成软标签 ======
# [云侧机器]
python run_task.py --step cloud_pretrain --task_id 004_train_link11
python run_task.py --step cloud_kd --task_id 004_train_link11
# >>> 同步 output/cloud_kd/ 到两台边侧机器 <<<

# ====== 阶段 3：知识蒸馏训练学生模型 ======
# [边侧机器1] 和 [边侧机器2] 各自独立跑
python run_task.py --step edge_kd --task_id 004_train_link11
# >>> 同步 output/edge_kd/ 回云侧机器 <<<

# ====== 阶段 4：联邦学习（3 台同时启动，自动轮询协调）======
# [云侧机器]
python run_task.py --step federated_server --task_id 004_train_link11
# [边侧机器1]
python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 1
# [边侧机器2]
python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 2
```

#### 5. 只跑知识蒸馏（教师模型已有）

| | 命令 |
|---|------|
| **单机** | `python run_task.py --mode knowledge_distillation --task_id 004_train_link11` |

```bash
# 多机拆分（2 步）：
# [云侧机器] 生成软标签
python run_task.py --step cloud_kd --task_id 004_train_link11
# >>> 同步 output/cloud_kd/ 到边侧 <<<
# [边侧机器]
python run_task.py --step edge_kd --task_id 004_train_link11
```

#### 6. 只跑联邦学习

| | 命令 |
|---|------|
| **单机** | `python run_task.py --mode federated_learning --task_id 004_train_link11` |

```bash
# 多机拆分（3 台同时启动）：
# [云侧机器]
python run_task.py --step federated_server --task_id 004_train_link11
# [边侧机器1]
python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 1
# [边侧机器2]
python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 2
```

### 联邦学习分布式协调机制

联邦学习的分布式模式通过**文件轮询**自动协调各机器间的训练轮次：

| 方式 | 命令 | 适用场景 |
|------|------|---------|
| 单机模拟 | `--mode federated_learning` 或 `--step federated_train` | 单机测试，串行模拟所有边 |
| 分布式 | 云侧 `--step federated_server` + 各边侧 `--step federated_edge --edge_id N` | 真正多机部署 |

**文件协议**（各机器通过这些文件自动协调）：

```
output/federated_train/
├── global_model_round_0.pth     <-- server 启动后保存初始模型
├── edge_1_round_1.pth           <-- edge 1 训练完第1轮后保存
├── edge_2_round_1.pth           <-- edge 2 训练完第1轮后保存
├── global_model_round_1.pth     <-- server 收到所有 edge 后聚合保存
├── ...（重复 N 轮）
├── global_model.pth             <-- 最终全局模型
├── edge_1_model.pth             <-- 最终 edge 1 模型
└── edge_2_model.pth             <-- 最终 edge 2 模型
```

**自动化流程**：
- **Server 启动后**：保存 `global_model_round_0.pth` → 每轮轮询等待所有 `edge_*_round_N.pth` 出现 → FedAvg 聚合 → 保存 `global_model_round_N.pth` → 循环
- **Edge 启动后**：每轮轮询等待 `global_model_round_{N-1}.pth` 出现 → 加载全局模型 → 本地训练 → 保存 `edge_{id}_round_N.pth` → 循环
- **断点续传**：已完成的轮次文件存在时会自动跳过

**文件同步**：你需要在各机器间同步 `tasks/{task_id}/output/federated_train/` 目录（rsync、scp、NFS 共享存储等均可）。轮询间隔默认 5 秒，超时默认 30 分钟（可在 `federated_train.json` 中添加 `"sync_timeout": 3600` 调整）。

### 多机文件同步汇总

| 同步时机 | 需要同步的目录 | 方向 |
|---------|--------------|------|
| `device_load` 完成后 | `output/device_load/` | 端 → 边/云 |
| `edge_infer` 完成后 | `output/edge_infer/` | 边 → 云 |
| `cloud_kd` 完成后 | `output/cloud_kd/` | 云 → 边 |
| `edge_kd` 完成后 | `output/edge_kd/` | 边 → 云 |
| 联邦学习进行中 | `output/federated_train/` | 云 ↔ 边（**持续双向同步**） |

> **注意**：联邦学习阶段需要**持续同步** `output/federated_train/` 目录（用 rsync watch、NFS 共享、或 scp 脚本循环跑），因为 server 和 edge 会互相等待对方写出的 `.pth` 文件。其他步骤只需在完成后同步一次即可。

### 完整分布式部署时序图

```
云侧机器:                             边侧机器1:                          边侧机器2:
─────────────────────                ─────────────────────               ─────────────────────
1. --step cloud_pretrain             (等待同步)                           (等待同步)
2. --step cloud_kd
   → 同步 output/cloud_kd/ 到边侧
                                     3. --step edge_kd                   3. --step edge_kd
   → 同步 output/edge_kd/ 回云侧
4. --step federated_server           4. --step federated_edge             4. --step federated_edge
                                        --edge_id 1                         --edge_id 2
   保存 global_round_0
   → 同步到边侧                         ← 等到 global_round_0               ← 等到 global_round_0
                                        训练→保存 edge_1_r1                 训练→保存 edge_2_r1
   等待 edge 模型                     → 同步到云侧                        → 同步到云侧
   ← 收到所有 edge
   聚合→保存 global_r1
   → 同步到边侧
   ...（重复 N 轮）                   ...                                 ...
   保存 global_model.pth              保存 edge_1_model.pth               保存 edge_2_model.pth
```

---

## 关键参数说明

### 推理参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `data_path` | device_load.json | 数据路径。单个 `.pkl` 文件或包含多个 `.pkl` 的目录均可 |
| `dataset_type` | device_load.json | 数据集类型：`link11`(7类) / `rml2016`(6类) / `radar`(7类) |
| `max_files` | device_load.json | 目录模式下最多加载的文件数，`null` = 全部加载 |
| `num_batches` | device_load.json | 限制总样本数 = num_batches × batch_size，`null` = 不限制 |
| `confidence_threshold` | edge_infer.json | 置信度阈值。低于此值的样本转发云侧。仅协同模式需要，纯边侧不写 |
| `simulate_bandwidth_mbps` | edge/cloud_infer.json | 模拟网络带宽(MB/s)，不填或 `null` = 不模拟 |
| `model_type` | edge/cloud_infer.json | 模型类型名，需与 `core/model_factory.py` 中定义一致 |
| `input_data.source` | edge/cloud_infer.json | 上一步骤名称，系统自动从 `output/{source}/` 读取对应文件 |

### 训练参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `epochs` | cloud_pretrain / edge_kd | 训练轮数 |
| `learning_rate` | 所有训练 JSON | 学习率，默认 0.001 |
| `temperature` | cloud_kd / edge_kd | 蒸馏温度，两个阶段必须一致，默认 4.0 |
| `kd_alpha` | edge_kd.json | 软标签权重 (0~1)，越大越依赖教师知识，默认 0.7 |
| `num_rounds` | federated_train.json | 联邦学习通信轮数 |
| `local_epochs` | federated_train.json | 每轮本地训练轮数 |
| `init_model_path` | federated_train.json | 全局模型初始权重路径，`null` = 随机初始化 |

---

## 如何创建新任务

1. 在 `tasks/` 下新建目录，名称即为 `task_id`：
   ```
   tasks/
   └── my_new_task/
       └── input/
           ├── device_load.json          # 推理时需要
           ├── edge_infer.json           # 需要边侧推理时
           ├── cloud_infer.json          # 需要云侧推理时
           ├── cloud_pretrain.json       # 预训练时需要
           ├── cloud_kd.json             # 知识蒸馏（云侧）时需要
           ├── edge_kd.json              # 知识蒸馏（边侧）时需要
           └── federated_train.json      # 联邦学习时需要
   ```

2. 只需放你选择的模式对应的 JSON 即可，不需要全部放。

3. 运行：
   ```bash
   python run_task.py --mode <模式名> --task_id my_new_task
   ```

`output/` 和 `result/` 目录会自动创建，无需手动建立。

## 输出说明

### 推理输出

- **`output/`** — 中间结果（供下一步骤读取）
  - `output/device_load/data_batch.pkl` — 标准化后的数据
  - `output/edge_infer/predictions.npy`、`confidences.npy`、`timing.json`
  - `output/edge_infer/low_conf_signals.pkl` — 低置信度样本（仅协同模式）
  - `output/cloud_infer/cloud_predictions.npy`、`timing.json`

- **`result/`** — 人类可读报告
  - `result/edge_infer/inference_report.txt`
  - `result/cloud_infer/cloud_report.txt`、`final_report.txt`（协同模式含最终对比报告）

### 训练输出

- `output/cloud_pretrain/teacher_model.pth` — 教师模型权重
- `output/cloud_kd/soft_labels.pkl` — 软标签
- `output/edge_kd/student_model.pth` — 蒸馏后的学生模型
- `output/federated_train/global_model.pth` — 联邦聚合模型
- `output/federated_train/edge_{i}_model.pth` — 各边侧模型
- `result/*/` — 各阶段训练报告

### 执行摘要（控制台输出）

每次运行结束后会输出各步骤耗时分析：

```
============================================================
执行摘要
============================================================
  device_load: success  (1.20s)
  edge_infer: success  (14.50s)  [纯推理: 2.10s | 加载+热身: 5.80s | 传输: 6.00s]
  cloud_infer: success  (18.00s)  [纯推理: 1.50s | 加载+热身: 10.20s | 传输: 5.30s]

  总耗时: 33.70s
  纯推理耗时: 3.60s
  开销(加载+热身): 16.00s
  网络传输耗时: 11.30s
  推理+开销+传输: 30.90s
  推理+传输: 14.90s
  去除开销后: 17.70s  (总耗时 33.70s - 加载热身 16.00s)
  结果目录: ./tasks/001_COLLAB_link11_test/result/
============================================================
```

## 断点续传

每个步骤执行前会检查 `output/` 中是否已有结果。如果已存在，会自动跳过（显示 `[跳过]`）。如需重新计算，删除对应的 `output/` 子目录：

```powershell
# 删除边侧推理缓存
Remove-Item -Recurse -Force "./tasks/003_edge_only_link11_test/output/edge_infer"

# 删除整个任务的所有缓存
Remove-Item -Recurse -Force "./tasks/my_task/output","./tasks/my_task/result"
```

## 网络模拟

在推理 JSON 中添加 `simulate_bandwidth_mbps` 字段可模拟网络传输延迟（按 `文件大小 / 带宽` sleep）：

| 链路 | 配置位置 | 典型值 |
|------|----------|--------|
| 设备 → 边侧 | edge_infer.json | 100 MB/s |
| 边侧 → 云侧 | cloud_infer.json (协同) | 10 MB/s |
| 设备 → 云侧 | cloud_infer.json (直接) | 10 MB/s |

不填或填 `null` = 不模拟传输延迟。

## 支持的模型

| 模型类型 | 用途 | 输入格式 | 说明 |
|---------|------|---------|------|
| `complex_resnet50_link11_with_attention` | 云侧 | complex (N, L) | ResNet50 + 注意力，复数卷积 |
| `complex_resnet50_rml2016_with_attention` | 云侧 | complex (N, L) | 同上，RML2016 数据集 |
| `complex_resnet50_radar_with_attention` | 云侧 | complex (N, L) | 同上，Radar 数据集 |
| `real_resnet20_link11_h` | 边侧 | complex (N, L) | ResNet20，内部拆分 real/imag 做实数卷积 |
| `real_resnet20_rml2016_h` | 边侧 | complex (N, L) | 同上，RML2016 数据集 |
| `real_resnet20_radar_h` | 边侧 | complex (N, L) | 同上，Radar 数据集 |

> **注意**：所有模型的输入都是 **complex 张量**（`torch.cfloat`）。名称中的 "real" 指的是模型内部使用实数卷积操作（将复数拆成 real/imag 两个通道），而非输入格式。

## 快速命令参考

### 单机命令（`--mode`，一条搞定）

只需替换 `task_id` 即可切换数据集，所有配置已内置在对应任务文件夹中。

#### 推理命令

| 场景 | link11 | rml2016 | radar |
|------|--------|---------|-------|
| 云边协同推理 | `python run_task.py --mode device_to_edge_to_cloud --task_id 001_COLLAB_link11_test` | `python run_task.py --mode device_to_edge_to_cloud --task_id 005_COLLAB_rml2016_test` | `python run_task.py --mode device_to_edge_to_cloud --task_id 008_COLLAB_radar_test` |
| 仅云侧推理 | `python run_task.py --mode device_to_cloud --task_id 002_cloud_only_link11_test` | `python run_task.py --mode device_to_cloud --task_id 006_cloud_only_rml2016_test` | `python run_task.py --mode device_to_cloud --task_id 009_cloud_only_radar_test` |
| 仅边侧推理 | `python run_task.py --mode device_to_edge --task_id 003_edge_only_link11_test` | `python run_task.py --mode device_to_edge --task_id 007_edge_only_rml2016_test` | `python run_task.py --mode device_to_edge --task_id 010_edge_only_radar_test` |

#### 训练命令

| 场景 | link11 | rml2016 | radar |
|------|--------|---------|-------|
| 完整训练 | `python run_task.py --mode full_train --task_id 004_train_link11` | `python run_task.py --mode full_train --task_id 011_train_rml2016` | `python run_task.py --mode full_train --task_id 012_train_radar` |
| 只预训练教师模型 | `python run_task.py --mode pretrain --task_id 004_train_link11` | `python run_task.py --mode pretrain --task_id 011_train_rml2016` | `python run_task.py --mode pretrain --task_id 012_train_radar` |
| 只知识蒸馏 | `python run_task.py --mode knowledge_distillation --task_id 004_train_link11` | `python run_task.py --mode knowledge_distillation --task_id 011_train_rml2016` | `python run_task.py --mode knowledge_distillation --task_id 012_train_radar` |
| 只联邦学习 | `python run_task.py --mode federated_learning --task_id 004_train_link11` | `python run_task.py --mode federated_learning --task_id 011_train_rml2016` | `python run_task.py --mode federated_learning --task_id 012_train_radar` |

#### 数据集参数速查

| 数据集 | 类别数 | 云侧模型 | 边侧模型 |
|--------|--------|----------|----------|
| link11 | 7 | `complex_resnet50_link11_with_attention` | `real_resnet20_link11_h` |
| rml2016 | 6 | `complex_resnet50_rml2016_with_attention` | `real_resnet20_rml2016_h` |
| radar | 7 | `complex_resnet50_radar_with_attention` | `real_resnet20_radar_h` |

#### 任务ID 与数据集对照表

| 任务ID | 数据集 | 用途 |
|--------|--------|------|
| `001_COLLAB_link11_test` | link11 | 协同推理 |
| `002_cloud_only_link11_test` | link11 | 仅云推理 |
| `003_edge_only_link11_test` | link11 | 仅边推理 |
| `004_train_link11` | link11 | 训练 |
| `005_COLLAB_rml2016_test` | rml2016 | 协同推理 |
| `006_cloud_only_rml2016_test` | rml2016 | 仅云推理 |
| `007_edge_only_rml2016_test` | rml2016 | 仅边推理 |
| `008_COLLAB_radar_test` | radar | 协同推理 |
| `009_cloud_only_radar_test` | radar | 仅云推理 |
| `010_edge_only_radar_test` | radar | 仅边推理 |
| `011_train_rml2016` | rml2016 | 训练 |
| `012_train_radar` | radar | 训练 |

### 多机命令（`--step`，分步执行）

> **切换数据集**：多机命令只需把 `task_id` 替换为对应数据集的任务ID即可（见下方对照表）。

| 场景 | link11 task_id | rml2016 task_id | radar task_id |
|------|---------------|-----------------|---------------|
| 协同推理 | `001_COLLAB_link11_test` | `005_COLLAB_rml2016_test` | `008_COLLAB_radar_test` |
| 仅云推理 | `002_cloud_only_link11_test` | `006_cloud_only_rml2016_test` | `009_cloud_only_radar_test` |
| 仅边推理 | `003_edge_only_link11_test` | `007_edge_only_rml2016_test` | `010_edge_only_radar_test` |
| 训练 | `004_train_link11` | `011_train_rml2016` | `012_train_radar` |

#### 推理（以 link11 为例，替换 task_id 即可换数据集）

| 场景 | 步骤 | 机器 | 命令 | 完成后同步 |
|------|------|------|------|-----------|
| **云边协同** | ① | 端侧 | `python run_task.py --step device_load --task_id 001_COLLAB_link11_test` | `output/device_load/` → 边侧 |
| | ② | 边侧 | `python run_task.py --step edge_infer --task_id 001_COLLAB_link11_test` | `output/edge_infer/` → 云侧 |
| | ③ | 云侧 | `python run_task.py --step cloud_infer --task_id 001_COLLAB_link11_test --summary` | — |
| **仅云侧** | ① | 端侧 | `python run_task.py --step device_load --task_id 002_cloud_only_link11_test` | `output/device_load/` → 云侧 |
| | ② | 云侧 | `python run_task.py --step cloud_direct_infer --task_id 002_cloud_only_link11_test --summary` | — |
| **仅边侧** | ① | 端侧 | `python run_task.py --step device_load --task_id 003_edge_only_link11_test` | `output/device_load/` → 边侧 |
| | ② | 边侧 | `python run_task.py --step edge_infer --task_id 003_edge_only_link11_test --summary` | — |

#### 训练（以 link11 为例，替换 task_id 即可换数据集）

| 场景 | 步骤 | 机器 | 命令 | 完成后同步 |
|------|------|------|------|-----------|
| **完整训练** | ① | 云侧 | `python run_task.py --step cloud_pretrain --task_id 004_train_link11` | — |
| | ② | 云侧 | `python run_task.py --step cloud_kd --task_id 004_train_link11` | `output/cloud_kd/` → 各边侧 |
| | ③ | 各边侧 | `python run_task.py --step edge_kd --task_id 004_train_link11` | `output/edge_kd/` → 云侧 |
| | ④ | 云侧 | `python run_task.py --step federated_server --task_id 004_train_link11` | 持续同步 `output/federated_train/` |
| | ④ | 边侧1 | `python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 1` | 持续同步 `output/federated_train/` |
| | ④ | 边侧2 | `python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 2` | 持续同步 `output/federated_train/` |
| **只知识蒸馏** | ① | 云侧 | `python run_task.py --step cloud_kd --task_id 004_train_link11` | `output/cloud_kd/` → 边侧 |
| | ② | 边侧 | `python run_task.py --step edge_kd --task_id 004_train_link11` | — |
| **只联邦学习** | ① | 云侧 | `python run_task.py --step federated_server --task_id 004_train_link11` | 持续同步 `output/federated_train/` |
| | ① | 边侧1 | `python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 1` | 持续同步 `output/federated_train/` |
| | ① | 边侧2 | `python run_task.py --step federated_edge --task_id 004_train_link11 --edge_id 2` | 持续同步 `output/federated_train/` |

> **提示**：推理多机最后一步加 `--summary` 可查看全局耗时汇总并写入报告。`--mode` 单机模式自动开启，无需手动加。联邦学习阶段④的 3 条命令同时启动，通过文件轮询自动协调。

---

## 环境依赖

```
torch >= 1.10
numpy
scipy
```

# 任务-函数对照与执行顺序（Training + Inference）

- **云侧预训练 / 轻量化 / 联邦训练**都归为同一个大任务：**Training（训练）**
- 它们只是训练的不同步骤（step），有推荐的先后顺序
- **推理（Inference）**单独作为另一大任务

---


# 1) Training（训练）

训练在你这个工程里由 `callback/train_callback.py` 提供，按含义可拆成 3 个训练步骤：

1. **云侧预训练（Teacher 训练）**
2. **轻量化（Teacher → Student）**
3. **联邦训练（FedAvg 全局聚合）**（单机模拟或分布式）

下面分别列出“**step → callback 函数**”以及“**前置依赖（必须先完成）**”。

---

## 1.1 训练步骤 1：云侧预训练（Pretrain）

- **step**：`cloud_pretrain`
- **callback 函数**：`callback/train_callback.py::cloud_pretrain_callback(task_id)`
- **前置依赖（必须先完成）**：无
- **相关 mode**：`pretrain`

---

## 1.2 训练步骤 2：轻量化（KD）

- **step**：`edge_kd`
- **callback 函数**：`callback/train_callback.py::edge_kd_callback(task_id, edge_id=None, **kwargs)`
  - `edge_id=None`：一次性蒸馏所有边（按 `edge_data_paths`）
  - `edge_id=N`：只蒸馏某一个边（用于多机/分步）
- **前置依赖（必须先完成）**：`cloud_pretrain`
- **相关 mode**：`knowledge_distillation`

---

## 1.3 训练步骤 3：联邦训练（Federated Learning）


### 1.3.1 分布式联邦（云端聚合 + 多边端本地训练）
同时运行
- **云端聚合**
  - **step**：`federated_server`
  - **callback 函数**：`callback/train_callback.py::federated_server_callback(task_id, **kwargs)`
  - **mode**：`federated_server`

- **边端本地训练**
  - **step**：`federated_edge`
  - **callback 函数**：`callback/train_callback.py::federated_edge_callback(task_id, edge_id=None, **kwargs)`
  - **mode**：`federated_edge`

---

## 1.4 Training 的推荐总顺序（同一个 task_id 下的训练链路）

定义的“训练链路顺序”在工程里对应：

- `cloud_pretrain` → `edge_kd` → `federated_train`

这里的“顺序”指**前置依赖**：

- 跑 `edge_kd` 之前先完成 `cloud_pretrain`
- 跑 `federated_train` 之前先完成 `edge_kd`（如果你希望使用 KD 的 student 初始化）

---

# 2) Inference（推理）

推理由 `callback/device_callback.py`、`callback/edge_callback.py`、`callback/cloud_callback.py` 提供。

本节按“推理模式（mode）”组织：你只要选定 mode，就能看到它需要跑哪些 step，以及每个 step 对应哪个 callback 函数。

---

## 2.1 mode：device_to_cloud（纯云推理，device→cloud）

- **step：`device_load`**
  - **callback**：`callback/device_callback.py::device_load_callback(task_id)`
  - **前置依赖**：无

- **step：`cloud_direct_infer`**
  - **callback**：`callback/cloud_callback.py::cloud_direct_infer_callback(task_id)`
  - **前置依赖**：必须先完成 `device_load`

---

## 2.2 mode：device_to_edge（纯边推理，device→edge）

- **step：`device_load`**
  - **callback**：`callback/device_callback.py::device_load_callback(task_id)`
  - **前置依赖**：无

- **step：`edge_infer`**
  - **callback**：`callback/edge_callback.py::edge_infer_callback(task_id)`
  - **前置依赖**：必须先完成 `device_load`

---

## 2.3 mode：device_to_edge_to_cloud（云边协同推理，device→edge→cloud）

- **step：`device_load`**
  - **callback**：`callback/device_callback.py::device_load_callback(task_id)`
  - **前置依赖**：无

- **step：`edge_infer`**
  - **callback**：`callback/edge_callback.py::edge_infer_callback(task_id)`
  - **前置依赖**：必须先完成 `device_load`

- **step：`cloud_infer`**
  - **callback**：`callback/cloud_callback.py::cloud_infer_callback(task_id)`
  - **前置依赖**：必须先完成 `edge_infer`



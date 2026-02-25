# 联邦知识蒸馏训练原理

## 一句话总结

用一个大模型（教师）的知识教会多个小模型（学生），再让小模型们互相取长补短。

---

## 整体故事

你开了一家连锁奶茶店，总部有一个**超级品鉴师**（教师模型/云侧大模型），两个分店各有一个**实习生**（学生模型/边侧小模型），每个分店有两个柜台（端侧设备）负责接待客人。

目标：让两个实习生都能接近品鉴师的水平，但实习生脑子小（模型参数少），而且各分店的客户口味不同（数据分布不同）。

### 系统架构：1 云 + 2 边 + 4 端

```
                    ┌──────────────┐
                    │  总部（云侧）  │
                    │ 品鉴师(大模型) │
                    │ ResNet50      │
                    │ 准确率: 95%   │
                    └──────┬───────┘
                     ┌─────┴─────┐
              ┌──────┴──────┐  ┌──┴──────────┐
              │ 分店1（边1）  │  │ 分店2（边2）  │
              │ 实习生1      │  │ 实习生2      │
              │ ResNet20     │  │ ResNet20     │
              │ 目标: 85%+   │  │ 目标: 85%+   │
              └──┬───────┬──┘  └──┬───────┬──┘
              ┌──┘       └──┐  ┌──┘       └──┐
           ┌──┴──┐     ┌──┴──┐┌──┴──┐    ┌──┴──┐
           │柜台1 │     │柜台2 ││柜台3 │    │柜台4 │
           │(端1) │     │(端2) ││(端3) │    │(端4) │
           └─────┘     └─────┘└─────┘    └─────┘
```

| 层级 | 数量 | 角色 | 类比 |
|------|------|------|------|
| 云 | 1 | 教师模型训练、软标签生成、联邦聚合 | 总部品鉴师 |
| 边 | 2 | 学生模型推理、联邦本地训练 | 分店实习生 |
| 端 | 4（每边 2 个） | 信号采集、数据加载 | 柜台接待 |

---

## 阶段 1：预训练教师模型（cloud_pretrain）

### 做什么

先把品鉴师培养出来。总部收集了所有奶茶的品评数据（全量训练数据），花大力气训练品鉴师。

```
全量数据 ──训练50轮──→ 品鉴师（大模型，准确率95%）
                        保存为 teacher_model.pth
```

### 对应代码

- 模型：`complex_resnet50_link11_with_attention`（大，精度高）
- 训练方式：标准监督学习（CrossEntropy Loss + Adam + CosineAnnealing）
- 输出：`output/cloud_pretrain/teacher_model.pth`

### 配置

```json
{
    "data_path": "dataset/splits/link11/cloud_data.pkl",
    "model_type": "complex_resnet50_link11_with_attention",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}
```

---

## 阶段 2：生成软标签（cloud_kd）

### 做什么

品鉴师把自己的"品鉴心得"写成小抄，给实习生看。

### 什么是软标签

假设有 7 种信号类型，一个样本的真实标签和品鉴师的判断分别是：

```
硬标签（真实答案）：  E-2D_1=100%,  其他全是0%
                     → [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

软标签（品鉴心得）：  E-2D_1=70%,  E-2D_2=12%,  P-3C_1=8%,  ...
                     → [0.70, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01]
```

**关键区别**：
- 硬标签只告诉你"对/错"
- 软标签还告诉你"E-2D_1 和 E-2D_2 长得像"——这就是**暗知识（dark knowledge）**

### 蒸馏温度（temperature）

控制软标签的"软"程度：

```
温度 = 1（不加温）：  [0.95, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]  ← 太尖锐，接近硬标签
温度 = 4（常用值）：  [0.70, 0.12, 0.08, 0.04, 0.03,  0.02,  0.01]   ← 平滑，暗知识丰富
温度 = 10（很高）：   [0.30, 0.18, 0.15, 0.13, 0.10,  0.08,  0.06]   ← 太平，信息模糊
```

温度越高，概率分布越平滑，类间关系越明显，但太高会丢失主要信息。通常 **2.0 ~ 8.0** 之间。

### 对应代码

```python
# 品鉴师对每个训练样本输出软标签
logits = teacher(batch_X)                        # 原始输出
soft = F.softmax(logits / temperature, dim=1)    # 除以温度再 softmax
```

### 配置

```json
{
    "teacher_model_path": "tasks/004_train_link11/output/cloud_pretrain/teacher_model.pth",
    "temperature": 4.0
}
```

---

## 阶段 3：知识蒸馏训练学生（edge_kd）

### 做什么

实习生（小模型）拿着品鉴师的小抄来学习。同时看两样东西：

```
             ┌── 硬标签（标准答案）：这杯是 E-2D_1
实习生同时学 ─┤
             └── 软标签（品鉴师心得）：70% 像 E-2D_1，12% 像 E-2D_2 ...
```

### 损失函数

```
总损失 = (1 - alpha) × CE损失(硬标签) + alpha × KD损失(软标签)
```

- `alpha = 0.7` → **70% 跟品鉴师学，30% 看标准答案**
- CE 损失：CrossEntropy，看预测对不对
- KD 损失：KL 散度，看学生的概率分布和教师的像不像

### 为什么有效

```
实习生 只看硬标签       ──→ 准确率 ~80%
实习生 硬标签 + 软标签  ──→ 准确率 ~85%  ← 蒸馏的提升
```

直接看硬标签，实习生只知道"对/错"。看了软标签，还能学到"哪些类别容易搞混、类别之间有什么关系"，学得更快更好。

### 对应代码

```python
# 学生模型的输出
student_logits = student(batch_X)

# 硬标签损失
ce_loss = CrossEntropyLoss(student_logits, hard_labels)

# 软标签损失（学生 vs 教师的概率分布差异）
student_soft = log_softmax(student_logits / T)
kd_loss = KLDivLoss(student_soft, teacher_soft) * T²

# 加权求和
total_loss = 0.3 * ce_loss + 0.7 * kd_loss
```

### 配置

```json
{
    "student_model_type": "real_resnet20_link11_h",
    "kd_alpha": 0.7,
    "temperature": 4.0,
    "epochs": 30
}
```

注意：`temperature` 必须和阶段 2 一致，否则教师的软标签和学生的软输出不在同一个"尺度"上。

---

## 阶段 4：联邦学习（federated_train）

### 做什么

两个分店的实习生各自用本地数据练习，定期把经验汇总到总部。

### 为什么需要

- 各分店的数据不能共享（隐私保护 / 带宽限制）
- 但又想让所有实习生都变强
- 单个分店数据太少，容易过拟合

### FedAvg 算法（联邦平均）

```
初始化：全局模型 W_global（用蒸馏模型初始化）

重复 20 轮（num_rounds=20）：
    │
    ├─ 分店1（边1）：拿到 W_global → 用本地数据训练 3 轮 → 得到 W1
    └─ 分店2（边2）：拿到 W_global → 用本地数据训练 3 轮 → 得到 W2
                    │
                    ▼  总部聚合（按数据量加权平均）
        W_global = (W1 × n1 + W2 × n2) / (n1 + n2)
                    │
                    ▼  发回各分店，进入下一轮
```

### 关键设计

- **数据始终留在本地**，只传模型参数（保护隐私）
- **加权平均**：数据多的分店权重大（n1, n2 是各分店的样本数）
- **init_model_path**：用蒸馏出来的学生模型作为起点，比随机初始化收敛更快

### 配置

```json
{
    "edge_data_paths": [
        "dataset/splits/link11/edge_1_data.pkl",
        "dataset/splits/link11/edge_2_data.pkl"
    ],
    "num_rounds": 20,
    "local_epochs": 3,
    "batch_size": 32,
    "init_model_path": "tasks/004_train_link11/output/edge_kd/student_model.pth"
}
```

---

## 四个阶段串起来

```
阶段1 预训练         ──→ 得到品鉴师（大模型，很准但太大没法部署到边侧）
  │
  ▼ teacher_model.pth
阶段2 生成软标签     ──→ 品鉴师把知识写成小抄（soft_labels.pkl）
  │
  ▼ soft_labels.pkl
阶段3 知识蒸馏       ──→ 小模型拿小抄学，获得不错的初始能力（student_model.pth）
  │
  ▼ student_model.pth（作为联邦学习的初始模型）
阶段4 联邦学习       ──→ 多个小模型各用本地数据练习、定期汇总
  │
  ▼
最终产物：global_model.pth + edge_1_model.pth + edge_2_model.pth
         两个边侧各有一个又小又准的模型，可以直接用于推理
```

### 效果对比（示意）

| 训练方式 | 边侧模型准确率 |
|---------|-------------|
| 小模型直接训练（无蒸馏） | ~78% |
| 小模型 + 知识蒸馏 | ~84% |
| 小模型 + 知识蒸馏 + 联邦学习 | ~87% |
| 大模型（教师，仅供参考） | ~95% |

---

## 运行命令

```bash
# 仅预训练教师模型（阶段 1）
python run_task.py --mode pretrain --task_id 004_train_link11

# 知识蒸馏（阶段 2→3，需已有教师模型，在 cloud_kd.json 中指定 teacher_model_path）
python run_task.py --mode knowledge_distillation --task_id 004_train_link11

# 仅联邦学习（阶段 4）
python run_task.py --mode federated_learning --task_id 004_train_link11

# 完整训练（阶段 1→2→3→4 全部串联）
python run_task.py --mode full_train --task_id 004_train_link11

# 完整训练 + 推理（7 个阶段：训练完直接跑协同推理）
python run_task.py --mode full_pipeline --task_id 004_train_link11
```

> `knowledge_distillation` 不含预训练。如果教师模型已经训练好，直接跑蒸馏即可。

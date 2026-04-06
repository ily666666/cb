# 云边端协同计算框架 Web 应用

面向典型电磁数据处理任务的云边端协同计算框架软件，提供 Web 可视化管理界面。

## 功能模块

| 模块 | 功能 | 路径 |
|------|------|------|
| 系统概览 | 架构展示、任务/数据集/流水线总览 | `/` |
| 数据接入 | 数据集浏览、文件预览、任务配置管理 | `/data` |
| 数据处理可视化 | 耗时分析图表、执行报告查看 | `/visualization` |
| 模型推理计算 | 推理/训练任务执行、运行监控 | `/inference` |
| 模型算法管理 | 云侧/边侧模型文件管理、参数查看 | `/models` |

## 技术栈

- **后端**: FastAPI + Python
- **前端**: Vue 3 + Element Plus + ECharts
- **构建**: Vite

## 快速启动

### 方式一：开发模式（推荐）

前后端分别启动，支持热更新：

```bash
# 1. 安装后端依赖
pip install -r web/backend/requirements.txt

# 2. 启动后端（在项目根目录）
cd web/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. 另开终端，安装并启动前端
cd web/frontend
npm install
npm run dev
```

然后访问 http://localhost:3000

### 方式二：一键启动

```bash
cd web
bash start_dev.sh
```

### 方式三：生产部署

```bash
# 构建前端
cd web/frontend
npm install && npm run build

# 启动后端（静态文件自动挂载）
cd web/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

然后访问 http://localhost:8000

## API 文档

启动后端后访问 http://localhost:8000/docs 查看 Swagger 交互式文档。

### 主要 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/system/info` | GET | 系统概览信息 |
| `/api/tasks/` | GET | 任务列表 |
| `/api/tasks/{id}` | GET | 任务详情 |
| `/api/tasks/{id}/run` | POST | 执行任务 |
| `/api/tasks/{id}/status` | GET | 运行状态 |
| `/api/data/datasets` | GET | 数据集列表 |
| `/api/data/datasets/{name}` | GET | 数据集详情 |
| `/api/inference/{id}/start` | POST | 启动推理 |
| `/api/inference/{id}/visualization` | GET | 可视化数据 |
| `/api/models/` | GET | 模型列表 |
| `/api/models/detail` | GET | 模型参数详情 |

## 目录结构

```
web/
├── backend/                    # FastAPI 后端
│   ├── main.py                 # 应用入口
│   ├── requirements.txt        # Python 依赖
│   ├── routers/                # API 路由
│   │   ├── tasks.py            # 任务管理
│   │   ├── data.py             # 数据接入
│   │   ├── inference.py        # 推理计算
│   │   └── models.py           # 模型管理
│   ├── services/               # 业务逻辑层
│   │   ├── task_service.py     # 对接 run_task.py
│   │   ├── data_service.py     # 数据文件管理
│   │   ├── inference_service.py# 推理结果处理
│   │   └── model_service.py    # 模型文件扫描
│   └── schemas/                # Pydantic 模型
│       └── schemas.py
├── frontend/                   # Vue 3 前端
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.js             # 应用入口
│       ├── App.vue             # 根组件（含侧边栏导航）
│       ├── router/index.js     # 路由配置
│       ├── api/index.js        # API 封装
│       ├── assets/main.css     # 全局样式
│       └── views/              # 页面组件
│           ├── Dashboard.vue   # 系统概览
│           ├── DataAccess.vue  # 数据接入
│           ├── Visualization.vue# 可视化
│           ├── Inference.vue   # 推理计算
│           └── Models.vue      # 模型管理
├── start.sh                    # 生产启动脚本
├── start_dev.sh                # 开发启动脚本
└── README.md
```

## 与主项目的关系

本 Web 应用通过后端服务层对接主项目的 `run_task.py` 任务执行系统：

- `task_service.py` 通过 `subprocess` 调用 `run_task.py`，支持异步执行和实时日志
- `data_service.py` 读取 `dataset/` 和 `tasks/*/input/` 中的数据文件和配置
- `inference_service.py` 读取 `tasks/*/output/` 和 `tasks/*/result/` 中的推理结果
- `model_service.py` 扫描 `run/cloud/pth/`、`run/edge/pth/` 以及任务输出中的模型文件

所有执行操作最终都通过 `run_task.py` 入口进行，与命令行使用方式一致。

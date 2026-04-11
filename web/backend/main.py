"""
云边端协同计算框架 - Web 后端入口
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import tasks, data, inference, models

app = FastAPI(
    title="云边端协同计算框架",
    description="面向典型电磁数据处理任务的云边端协同计算框架软件",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router, prefix="/api/tasks", tags=["任务管理"])
app.include_router(data.router, prefix="/api/data", tags=["数据接入"])
app.include_router(inference.router, prefix="/api/inference", tags=["模型推理计算"])
app.include_router(models.router, prefix="/api/models", tags=["模型算法管理"])

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok", "project_root": PROJECT_ROOT}


@app.get("/api/system/info")
async def system_info():
    """系统概览信息"""
    from config_refactor import (
        DEVICE, TASKS_ROOT, DATASET_CONFIG, SUPPORTED_TASKS,
        PIPELINE_MODES, CLOUD_MODEL_DIR, EDGE_MODEL_DIR
    )
    tasks_root = os.path.join(PROJECT_ROOT, TASKS_ROOT)
    task_ids = []
    if os.path.isdir(tasks_root):
        task_ids = sorted([
            d for d in os.listdir(tasks_root)
            if os.path.isdir(os.path.join(tasks_root, d)) and not d.startswith('.')
        ])

    return {
        "device": DEVICE,
        "tasks_root": TASKS_ROOT,
        "datasets": list(DATASET_CONFIG.keys()),
        "dataset_config": DATASET_CONFIG,
        "supported_tasks": SUPPORTED_TASKS,
        "pipeline_modes": {k: v for k, v in PIPELINE_MODES.items() if not k.startswith(('link11_', 'rml2016_', 'radar_', 'ratr_'))},
        "task_ids": task_ids,
        "cloud_model_dir": CLOUD_MODEL_DIR,
        "edge_model_dir": EDGE_MODEL_DIR,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

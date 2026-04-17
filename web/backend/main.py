"""
云边端协同计算框架 - Web 后端入口
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import tasks, data, inference, models, lightweight, distillation, prune_pow2, compare

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
app.include_router(lightweight.router, prefix="/api/lightweight", tags=["模型轻量化"])
app.include_router(distillation.router, prefix="/api/distillation", tags=["知识蒸馏"])
app.include_router(prune_pow2.router, prefix="/api/prune-pow2", tags=["剪枝量化(2的幂次)"])
app.include_router(compare.router, prefix="/api/compare", tags=["对比分析"])

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

    import json as _json
    from config_refactor import KNOWN_DATASETS

    actual_models = {}
    for ds in KNOWN_DATASETS:
        actual_models[ds] = {"cloud_models": set(), "edge_models": set()}
    for tid in task_ids:
        input_dir = os.path.join(tasks_root, tid, "input")
        if not os.path.isdir(input_dir):
            continue
        ds = None
        for d in KNOWN_DATASETS:
            if d in tid.lower():
                ds = d
                break
        if not ds:
            continue
        for fn in os.listdir(input_dir):
            if not fn.endswith('.json'):
                continue
            try:
                with open(os.path.join(input_dir, fn), 'r', encoding='utf-8') as fp:
                    cfg = _json.load(fp)
                mt = cfg.get("model_type", "")
                smt = cfg.get("student_model_type", "")
                tmt = cfg.get("teacher_model_type", "")
                emt = cfg.get("edge_model_type", "")
                if mt:
                    if "cloud" in fn or "pretrain" in fn:
                        actual_models[ds]["cloud_models"].add(mt)
                    elif "edge" in fn:
                        actual_models[ds]["edge_models"].add(mt)
                if tmt:
                    actual_models[ds]["cloud_models"].add(tmt)
                if smt:
                    actual_models[ds]["edge_models"].add(smt)
                if emt:
                    actual_models[ds]["edge_models"].add(emt)
            except Exception:
                pass
    actual_models_serializable = {
        ds: {"cloud_models": sorted(v["cloud_models"]), "edge_models": sorted(v["edge_models"])}
        for ds, v in actual_models.items()
    }

    return {
        "device": DEVICE,
        "tasks_root": TASKS_ROOT,
        "datasets": list(DATASET_CONFIG.keys()),
        "dataset_config": DATASET_CONFIG,
        "actual_models": actual_models_serializable,
        "supported_tasks": SUPPORTED_TASKS,
        "pipeline_modes": {k: v for k, v in PIPELINE_MODES.items() if not k.startswith(('link11_', 'rml2016_', 'radar_', 'ratr_'))},
        "task_ids": task_ids,
        "cloud_model_dir": CLOUD_MODEL_DIR,
        "edge_model_dir": EDGE_MODEL_DIR,
    }


@app.websocket("/ws/tasks/{task_id}/output")
async def ws_task_output(websocket: WebSocket, task_id: str):
    from services.task_service import _running_tasks
    await websocket.accept()
    try:
        info = _running_tasks.get(task_id)
        if not info:
            await websocket.send_json({"type": "error", "data": "无此任务记录"})
            await websocket.close()
            return

        sent = 0
        for line in info["output_lines"][:]:
            await websocket.send_json({"type": "line", "data": line})
            sent += 1

        while info.get("status") == "running":
            event = info.get("_line_event")
            if event:
                await asyncio.get_event_loop().run_in_executor(None, event.wait, 1.0)
                event.clear()
            new_lines = info["output_lines"][sent:]
            for line in new_lines:
                await websocket.send_json({"type": "line", "data": line})
                sent += 1

        remaining = info["output_lines"][sent:]
        for line in remaining:
            await websocket.send_json({"type": "line", "data": line})

        await websocket.send_json({"type": "done", "status": info.get("status", "unknown")})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

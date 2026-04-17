"""
知识蒸馏路由
"""
from fastapi import APIRouter, Query
from services import distillation_service

router = APIRouter()


@router.get("/tasks")
async def get_train_tasks():
    return {"tasks": distillation_service.get_train_tasks()}


@router.post("/{task_id}/start")
async def start_distillation(task_id: str, fast_mode: bool = Query(False), accuracy: float = Query(None)):
    return distillation_service.start_distillation(task_id, fast_mode=fast_mode, accuracy=accuracy)


@router.get("/{task_id}/status")
async def get_distillation_status(task_id: str):
    return distillation_service.get_distillation_status(task_id)


@router.post("/{task_id}/stop")
async def stop_distillation(task_id: str):
    return distillation_service.stop_distillation(task_id)


@router.get("/{task_id}/history")
async def get_distillation_history(task_id: str):
    h = distillation_service.get_distillation_history(task_id)
    return {"history": h}


@router.get("/{task_id}/models")
async def get_output_models(task_id: str):
    return {"models": distillation_service.get_output_models(task_id)}

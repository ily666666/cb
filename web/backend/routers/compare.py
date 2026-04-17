"""
对比分析路由
"""
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from services import compare_service

router = APIRouter()


class StepConfigUpdate(BaseModel):
    simulate_realtime: Optional[bool] = None
    data_size_mb: Optional[float] = None
    time: Optional[float] = None
    accuracy: Optional[float] = None


class LabelUpdate(BaseModel):
    label: str


class SummaryUpdate(BaseModel):
    total_time: float
    accuracy: Optional[float] = None
    label: Optional[str] = None


class CloneRequest(BaseModel):
    source_task_id: str
    new_task_id: str
    label: str = ''


@router.get("/tasks")
async def get_compare_tasks():
    return {"tasks": compare_service.get_tasks_for_compare()}


@router.put("/tasks/{task_id}/label")
async def update_label(task_id: str, body: LabelUpdate):
    return compare_service.update_task_label(task_id, body.label)


@router.put("/tasks/{task_id}/summary")
async def update_summary(task_id: str, body: SummaryUpdate):
    return compare_service.update_task_summary(task_id, body.total_time, body.accuracy, body.label)


@router.put("/tasks/{task_id}/steps/{step_name}/config")
async def update_step_config(task_id: str, step_name: str, body: StepConfigUpdate):
    return compare_service.update_step_config(task_id, step_name, body.dict(exclude_none=True))


@router.post("/clone")
async def clone_task(req: CloneRequest):
    return compare_service.clone_task(req.source_task_id, req.new_task_id, req.label)


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    return compare_service.delete_task(task_id)


@router.get("/results")
async def get_compare_results(task_ids: str = Query(...)):
    ids = [x.strip() for x in task_ids.split(',') if x.strip()]
    return {"results": compare_service.get_compare_results(ids)}

"""
任务管理路由
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from services import task_service

router = APIRouter()


@router.get("/")
async def list_tasks():
    return {"tasks": task_service.list_tasks()}


@router.get("/modes")
async def get_modes():
    return {
        "pipeline_modes": task_service.get_pipeline_modes(),
        "supported_steps": task_service.get_supported_steps(),
    }


@router.get("/active")
async def get_active():
    return {"tasks": task_service.get_active_tasks()}


@router.get("/{task_id}")
async def get_task(task_id: str):
    detail = task_service.get_task_detail(task_id)
    if not detail:
        raise HTTPException(404, f"任务 {task_id} 不存在")
    return detail


@router.get("/{task_id}/timing")
async def get_timing(task_id: str):
    timing = task_service.get_task_timing(task_id)
    if not timing:
        raise HTTPException(404, f"任务 {task_id} 无耗时数据")
    return timing


@router.post("/{task_id}/run")
async def run_task(
    task_id: str,
    mode: Optional[str] = None,
    step: Optional[str] = None,
    config: Optional[str] = None,
    edge_id: Optional[int] = None,
    summary: bool = False,
):
    result = task_service.run_task_async(
        task_id, mode=mode, step=step, config=config,
        edge_id=edge_id, summary=summary,
    )
    if result["status"] == "error":
        raise HTTPException(400, result["message"])
    return result


@router.post("/{task_id}/stop")
async def stop_task(task_id: str):
    result = task_service.stop_task(task_id)
    if result["status"] == "error":
        raise HTTPException(400, result["message"])
    return result


@router.delete("/{task_id}/record")
async def remove_record(task_id: str):
    result = task_service.remove_task_record(task_id)
    if result["status"] == "error":
        raise HTTPException(400, result["message"])
    return result


@router.delete("/{task_id}/output")
async def clean_output(task_id: str):
    result = task_service.clean_task_output(task_id)
    if result["status"] == "error":
        raise HTTPException(400, result["message"])
    return result


@router.get("/{task_id}/status")
async def get_run_status(task_id: str):
    status = task_service.get_task_run_status(task_id)
    if not status:
        return {"status": "idle", "message": "当前无运行中的任务"}
    return status

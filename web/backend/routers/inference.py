"""
推理计算路由
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from services import inference_service

router = APIRouter()


@router.get("/modes")
async def get_inference_modes():
    return {
        "inference_modes": inference_service.INFERENCE_MODES,
        "train_modes": inference_service.TRAIN_MODES,
    }


@router.post("/{task_id}/start")
async def start_inference(task_id: str, mode: str = "device_to_edge_to_cloud"):
    result = inference_service.start_inference(task_id, mode)
    if result.get("status") == "error":
        raise HTTPException(400, result["message"])
    return result


@router.get("/{task_id}/result")
async def get_result(task_id: str):
    result = inference_service.get_inference_result(task_id)
    if not result:
        raise HTTPException(404, f"任务 {task_id} 无推理结果")
    return result


@router.get("/{task_id}/visualization")
async def get_visualization(task_id: str):
    data = inference_service.get_visualization_data(task_id)
    if not data:
        raise HTTPException(404, f"任务 {task_id} 无可视化数据")
    return data


@router.get("/{task_id}/report/{report_name:path}")
async def get_report(task_id: str, report_name: str):
    content = inference_service.get_inference_report(task_id, report_name)
    if content is None:
        raise HTTPException(404, f"报告 {report_name} 不存在")
    return {"report_name": report_name, "content": content}

"""
数据接入路由
"""
from fastapi import APIRouter, HTTPException
from typing import Dict
from services import data_service

router = APIRouter()


@router.get("/datasets")
async def list_datasets():
    return {"datasets": data_service.list_datasets()}


@router.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    detail = data_service.get_dataset_detail(dataset_name)
    if not detail:
        raise HTTPException(404, f"数据集 {dataset_name} 不存在")
    return detail


@router.get("/datasets/{dataset_name}/files/{filename}/preview")
async def preview_file(dataset_name: str, filename: str):
    preview = data_service.get_data_file_preview(dataset_name, filename)
    if not preview:
        raise HTTPException(404, f"文件 {filename} 不存在")
    return preview


@router.get("/tasks/{task_id}/configs")
async def get_task_configs(task_id: str):
    return {"configs": data_service.get_task_input_configs(task_id)}


@router.put("/tasks/{task_id}/configs/{filename}")
async def save_config(task_id: str, filename: str, content: Dict):
    if not filename.endswith('.json'):
        filename += '.json'
    ok = data_service.save_task_config(task_id, filename, content)
    if not ok:
        raise HTTPException(500, "保存配置失败")
    return {"success": True, "message": f"配置 {filename} 已保存"}

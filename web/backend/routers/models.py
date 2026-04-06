"""
模型算法管理路由
"""
from fastapi import APIRouter, HTTPException
from services import model_service

router = APIRouter()


@router.get("/")
async def list_models():
    return model_service.list_all_models()


@router.get("/config")
async def get_model_config():
    return model_service.get_dataset_model_config()


@router.get("/detail")
async def get_model_detail(path: str):
    detail = model_service.get_model_detail(path)
    if not detail:
        raise HTTPException(404, f"模型文件不存在: {path}")
    return detail


@router.delete("/")
async def delete_model(path: str):
    ok = model_service.delete_model(path)
    if not ok:
        raise HTTPException(404, f"模型文件不存在或删除失败: {path}")
    return {"success": True, "message": f"已删除 {path}"}

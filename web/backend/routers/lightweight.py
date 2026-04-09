"""
模型轻量化路由
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional
from services import lightweight_service

router = APIRouter()


class CompressRequest(BaseModel):
    method_id: str
    model_path: str = ""
    params: Dict = {}


@router.get("/methods")
async def get_compress_methods():
    return {"methods": lightweight_service.get_methods()}


@router.get("/models")
async def get_available_models():
    return {"models": lightweight_service.get_available_models()}


@router.get("/datasets")
async def get_available_datasets():
    return {"datasets": lightweight_service.get_available_datasets()}


@router.post("/run")
async def run_compress(req: CompressRequest):
    return lightweight_service.run_compress(req.method_id, req.model_path, req.params)


@router.get("/status")
async def get_status():
    return lightweight_service.get_status()


@router.get("/history")
async def get_compression_history():
    return {"results": lightweight_service.get_compression_history()}

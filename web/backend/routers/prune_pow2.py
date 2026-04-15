"""
剪枝 + 2的幂次量化（INQ）路由
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from services import prune_pow2_service

router = APIRouter()


class InqCompressRequest(BaseModel):
    model_path: str = ""
    params: Dict = {}


@router.get("/method")
async def get_method():
    return {"method": prune_pow2_service.get_method()}


@router.get("/models")
async def get_available_models():
    return {"models": prune_pow2_service.get_available_models()}


@router.get("/datasets")
async def get_available_datasets():
    return {"datasets": prune_pow2_service.get_available_datasets()}


@router.post("/run")
async def run_compress(req: InqCompressRequest):
    return prune_pow2_service.run_compress(req.model_path, req.params)


@router.get("/status")
async def get_status():
    return prune_pow2_service.get_status()


@router.post("/stop")
async def stop_compress():
    return prune_pow2_service.stop_compress()


@router.get("/history")
async def get_compression_history():
    return {"results": prune_pow2_service.get_compression_history()}

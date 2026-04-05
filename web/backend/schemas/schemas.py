"""
Pydantic 数据模型定义
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class DatasetType(str, Enum):
    LINK11 = "link11"
    RML2016 = "rml2016"
    RADAR = "radar"
    RATR = "ratr"


class PipelineMode(str, Enum):
    DEVICE_TO_CLOUD = "device_to_cloud"
    DEVICE_TO_EDGE = "device_to_edge"
    DEVICE_TO_EDGE_TO_CLOUD = "device_to_edge_to_cloud"
    PRETRAIN = "pretrain"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    FEDERATED_LEARNING = "federated_learning"
    FULL_TRAIN = "full_train"
    FULL_PIPELINE = "full_pipeline"


# ========== 任务相关 ==========

class TaskInfo(BaseModel):
    task_id: str
    dataset: Optional[str] = None
    purpose: Optional[str] = None
    has_input: bool = False
    has_output: bool = False
    has_result: bool = False
    config_files: List[str] = []


class TaskRunRequest(BaseModel):
    task_id: str
    mode: Optional[str] = None
    step: Optional[str] = None
    config: Optional[str] = None
    edge_id: Optional[int] = None
    summary: bool = False


class TaskRunResult(BaseModel):
    task_id: str
    status: str
    message: str = ""
    results: Optional[Dict[str, Any]] = None


class TaskTimingInfo(BaseModel):
    steps: Dict[str, Dict[str, float]] = {}
    total_data_load: float = 0
    total_preprocess: float = 0
    total_data_save: float = 0
    total_inference: float = 0
    total_overhead: float = 0
    total_transfer: float = 0


# ========== 数据相关 ==========

class DatasetInfo(BaseModel):
    name: str
    num_classes: int
    signal_length: int
    cloud_model: str
    edge_model: str
    data_files: List[str] = []
    total_size_mb: float = 0


class DataFileInfo(BaseModel):
    filename: str
    path: str
    size_mb: float
    dataset_type: Optional[str] = None


# ========== 模型相关 ==========

class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    model_type: str
    dataset: str
    role: str  # cloud / edge


class ModelListResponse(BaseModel):
    cloud_models: List[ModelInfo] = []
    edge_models: List[ModelInfo] = []


# ========== 推理相关 ==========

class InferenceRequest(BaseModel):
    task_id: str
    mode: str = "device_to_edge_to_cloud"


class InferenceResult(BaseModel):
    task_id: str
    mode: str
    status: str
    steps: List[Dict[str, Any]] = []
    timing: Optional[TaskTimingInfo] = None
    report: Optional[str] = None


# ========== 通用响应 ==========

class ApiResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Any] = None

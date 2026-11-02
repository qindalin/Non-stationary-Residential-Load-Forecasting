from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DeviceInfo(BaseModel):
    hostname: str
    ip_address: Optional[str] = None
    hardware: str = "raspberry-pi"
    model_name: str
    tags: List[str] = Field(default_factory=list)


class ResourceStats(BaseModel):
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None
    network_type: Optional[str] = None


class RegisterRequest(BaseModel):
    client_id: str
    device: DeviceInfo


class RegisterResponse(BaseModel):
    accepted: bool
    server_time: str
    message: str


class HeartbeatRequest(BaseModel):
    client_id: str
    timestep: int
    resources: ResourceStats = Field(default_factory=ResourceStats)
    latest_mae: Optional[float] = None
    latest_mse: Optional[float] = None


class SyncRequest(BaseModel):
    client_id: str
    timestep: int
    drift_score: float
    urgency: float
    local_loss: Optional[float] = None
    global_loss: Optional[float] = None
    resource_pressure: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncDecision(BaseModel):
    accepted: bool
    reason: str
    snapshot_id: Optional[str] = None
    upload_required: bool = False
    download_available: bool = False
    budget_hint: Optional[int] = None


class ModelUpdatePayload(BaseModel):
    client_id: str
    timestep: int
    drift_score: float
    cloud_utility: float = 0.0
    volatility: float = 0.0
    staleness: float = 0.0
    state_dict_b64: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncDownloadResponse(BaseModel):
    available: bool
    snapshot_id: Optional[str] = None
    state_dict_b64: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsReport(BaseModel):
    client_id: str
    timestep: int
    mae: Optional[float] = None
    mse: Optional[float] = None
    abs_error_sum: Optional[float] = None
    sq_error_sum: Optional[float] = None
    sample_count: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    inference_latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    server: str
    registered_clients: int
    latest_snapshot_id: Optional[str] = None
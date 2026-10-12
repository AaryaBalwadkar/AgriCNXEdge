from pydantic import BaseModel
from typing import Any


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    filename: str
    latency_ms: float
    leaf: dict[str, Any] | None
    pest: dict[str, Any] | None
    fruit: dict[str, Any] | None
    yield_detection: dict[str, Any] | None
    raw_output_names: list[str]

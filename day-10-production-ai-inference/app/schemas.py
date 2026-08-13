from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class InferenceRequest(BaseModel):
    features: list = Field(..., description="128-dimensional feature vector or query string", json_schema_extra={"example": [0.1] * 128})
    query_text: Optional[str] = Field("What is production AI inference?", description="Optional text query for semantic caching.")
    model_version: Optional[str] = Field("v1.0.0", description="Model version tag.")
    force_cache_miss: Optional[bool] = Field(False, description="Bypass cache.")

class InferenceResponse(BaseModel):
    request_id: str
    prediction: int
    latency_ms: float
    cache_hit: bool
    similarity_score: float
    trace: Dict[str, Any]

class ChaosInjectionRequest(BaseModel):
    slow_model_enabled: bool = False
    slow_model_delay_ms: float = 500.0
    cache_failure_enabled: bool = False
    queue_overload_enabled: bool = False

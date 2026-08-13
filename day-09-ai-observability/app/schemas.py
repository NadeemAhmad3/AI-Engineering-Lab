from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class InferenceRequest(BaseModel):
    query: str = Field(..., description="Query prompt text", json_schema_extra={"example": "Summarize ML observability."})
    force_cache_miss: Optional[bool] = Field(False, description="Force cache bypass for testing.")

class ChaosInjectionRequest(BaseModel):
    slow_inference_enabled: bool = Field(False, description="Simulate 500ms model delay.")
    slow_inference_delay_ms: float = Field(500.0, description="Delay duration in ms.")
    cache_failure_enabled: bool = Field(False, description="Simulate cache bypass / failure.")
    queue_overload_enabled: bool = Field(False, description="Simulate queue depth saturation.")

class InferenceResponse(BaseModel):
    request_id: str
    result: str
    latency_ms: float
    cache_hit: bool
    trace: Dict[str, Any]

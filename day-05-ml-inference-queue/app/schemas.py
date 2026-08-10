from pydantic import BaseModel, Field
from typing import List, Optional

class QueuePredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        json_schema_extra={"example": [0.5] * 50},
        description="50 numerical feature inputs."
    )

class QueuePredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class label")
    total_latency_ms: float = Field(..., description="Total round-trip latency in ms")
    queue_wait_ms: float = Field(..., description="Time spent waiting in queue before processing")
    inference_ms: float = Field(..., description="Model CPU inference execution time in ms")
    queue_depth_at_arrival: int = Field(..., description="Queue depth when request arrived")
    status: str = Field("success", description="Request outcome status ('success', 'rejected_429', 'timeout_504')")

class QueueMetricsResponse(BaseModel):
    current_queue_depth: int
    max_queue_capacity: int
    total_enqueued: int
    total_processed: int
    total_rejected_429: int
    total_timed_out_504: int
    avg_queue_wait_ms: float
    num_active_workers: int

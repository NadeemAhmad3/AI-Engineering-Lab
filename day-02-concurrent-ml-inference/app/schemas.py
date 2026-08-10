from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        json_schema_extra={"example": [0.5] * 50},
        description="50 numerical feature inputs for model prediction."
    )

class LatencyHeader(BaseModel):
    server_execution_ms: float = Field(..., description="Server inference processing latency in ms")
    worker_pid: int = Field(..., description="Process ID of the worker servicing the request")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class label")
    probabilities: Optional[List[float]] = Field(None, description="Class probabilities")
    worker_pid: int = Field(..., description="Process ID servicing this request")
    execution_ms: float = Field(..., description="Internal inference latency in ms")

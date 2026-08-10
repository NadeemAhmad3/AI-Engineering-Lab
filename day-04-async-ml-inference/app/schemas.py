from pydantic import BaseModel, Field
from typing import List, Optional

class InferenceRequest(BaseModel):
    features: List[float] = Field(
        ...,
        json_schema_extra={"example": [0.5] * 50},
        description="50 numerical feature inputs for model prediction."
    )

class InferenceResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class label")
    execution_ms: float = Field(..., description="Internal server latency in ms")
    mode: str = Field(..., description="Architecture pattern ('sync', 'async-blocking', 'async-offloaded', 'async-io')")
    io_simulated: bool = Field(False, description="Whether pre-inference I/O lookup was simulated")

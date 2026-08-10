from pydantic import BaseModel, Field
from typing import List, Optional

class SinglePredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        json_schema_extra={"example": [0.5] * 50},
        description="Single sample feature input array (50 features)."
    )

class BatchPredictionRequest(BaseModel):
    batch_features: List[List[float]] = Field(
        ...,
        description="List of feature vectors for batched prediction."
    )

class SinglePredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class label")
    execution_ms: float = Field(..., description="Inference execution latency in ms")
    batch_size_used: int = Field(1, description="Batch size processed by the model")
    mode: str = Field(..., description="Inference mode ('individual', 'static-batch', 'dynamic-batch')")

class BatchPredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="List of predicted class labels")
    execution_ms: float = Field(..., description="Total batch execution latency in ms")
    batch_size: int = Field(..., description="Number of items in batch")

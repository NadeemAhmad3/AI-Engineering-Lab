from pydantic import BaseModel, Field
from typing import List, Optional

class QuantizationRequest(BaseModel):
    batch_size: int = Field(1, ge=1, le=512, description="Batch size of 128-dimensional feature vectors.")

class QuantizationResponse(BaseModel):
    predictions: List[int] = Field(..., description="Predicted class labels")
    batch_size: int = Field(..., description="Number of samples in batch")
    execution_ms: float = Field(..., description="Inference execution time in ms")
    precision: str = Field(..., description="Precision mode ('fp32', 'fp16', 'int8')")
    model_size_mb: float = Field(..., description="Model size on disk in MB")

class QuantizationMetricsResponse(BaseModel):
    fp32_size_mb: float
    fp16_size_mb: float
    int8_size_mb: float
    fp32_accuracy: float
    fp16_accuracy: float
    int8_accuracy: float
    fp16_accuracy_drop: float
    int8_accuracy_drop: float
    recommended_precision_under_1pct_budget: str

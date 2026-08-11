from pydantic import BaseModel, Field
from typing import List, Optional

class ImageBatchRequest(BaseModel):
    batch_size: int = Field(1, ge=1, le=512, description="Batch size of synthetic (3, 64, 64) image vectors.")

class HardwareInferenceResponse(BaseModel):
    predictions: List[int] = Field(..., description="Predicted class labels")
    batch_size: int = Field(..., description="Number of samples in batch")
    total_latency_ms: float = Field(..., description="Total round-trip latency in ms")
    transfer_to_gpu_ms: float = Field(0.0, description="CPU to GPU memory transfer time in ms")
    compute_ms: float = Field(..., description="Model compute execution time in ms")
    transfer_to_cpu_ms: float = Field(0.0, description="GPU to CPU memory transfer time in ms")
    device: str = Field(..., description="Execution hardware device ('cpu' or 'cuda')")
    device_name: str = Field(..., description="Hardware model name")

class HardwareMetricsResponse(BaseModel):
    cuda_available: bool
    device_name: str
    torch_version: str
    cuda_version: Optional[str] = None

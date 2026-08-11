import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import (
    ImageBatchRequest,
    HardwareInferenceResponse,
    HardwareMetricsResponse
)
from src.cpu_inference import cpu_engine
from src.gpu_inference import gpu_engine

app = FastAPI(
    title="Day 6 — CPU vs GPU Inference Engineering Lab",
    description="Investigating when GPU acceleration pays off, batch size crossover point, memory transfer overhead, and cost efficiency.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Day 6 CPU vs GPU Lab API is live."}

@app.get("/metrics/hardware", response_model=HardwareMetricsResponse)
def get_hardware_metrics():
    """Returns PyTorch CUDA availability and GPU device information."""
    is_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if is_cuda else "CPU (Host Machine)"
    cuda_ver = torch.version.cuda if is_cuda else None
    return HardwareMetricsResponse(
        cuda_available=is_cuda,
        device_name=dev_name,
        torch_version=torch.__version__,
        cuda_version=cuda_ver
    )

@app.post("/predict/cpu", response_model=HardwareInferenceResponse)
def predict_cpu(payload: ImageBatchRequest):
    """Executes PyTorch model inference on CPU."""
    batch_size = payload.batch_size
    synthetic_images = np.random.randn(batch_size, 3, 64, 64).astype(np.float32)
    
    preds, lat_ms = cpu_engine.predict_batch(synthetic_images)
    
    return HardwareInferenceResponse(
        predictions=[int(p) for p in preds],
        batch_size=batch_size,
        total_latency_ms=round(lat_ms, 3),
        transfer_to_gpu_ms=0.0,
        compute_ms=round(lat_ms, 3),
        transfer_to_cpu_ms=0.0,
        device="cpu",
        device_name="CPU Host"
    )

@app.post("/predict/gpu", response_model=HardwareInferenceResponse)
def predict_gpu(payload: ImageBatchRequest):
    """Executes PyTorch model inference on GPU (or CUDA device)."""
    batch_size = payload.batch_size
    synthetic_images = np.random.randn(batch_size, 3, 64, 64).astype(np.float32)
    
    preds, metrics = gpu_engine.predict_batch_detailed(synthetic_images)
    
    return HardwareInferenceResponse(
        predictions=[int(p) for p in preds],
        batch_size=batch_size,
        total_latency_ms=metrics["total_latency_ms"],
        transfer_to_gpu_ms=metrics["transfer_to_gpu_ms"],
        compute_ms=metrics["compute_ms"],
        transfer_to_cpu_ms=metrics["transfer_to_cpu_ms"],
        device=metrics["device"],
        device_name=metrics["device_name"]
    )

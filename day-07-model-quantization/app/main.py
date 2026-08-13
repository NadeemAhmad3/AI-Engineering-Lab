import os
import time
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import (
    QuantizationRequest,
    QuantizationResponse,
    QuantizationMetricsResponse
)
from src.quantize import load_fp32_model, create_fp16_model, create_int8_model, FP32_PATH, FP16_PATH, INT8_PATH
from src.evaluate import evaluate_all_precisions

models_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads FP32, FP16, and INT8 models into RAM during app startup."""
    print("[Lifespan] Loading FP32, FP16, and INT8 models into memory...")
    models_cache["fp32"] = load_fp32_model()
    models_cache["fp16"] = create_fp16_model()
    models_cache["int8"] = create_int8_model()
    yield
    print("[Lifespan] App shutting down.")

app = FastAPI(
    title="Day 7 — Model Quantization Engineering Lab",
    description="Investigating FP32 vs FP16 vs INT8 Model Quantization, Accuracy Degradation, Latency, and Cost Trade-offs.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Day 7 Model Quantization API is live."}

@app.get("/metrics/quantization", response_model=QuantizationMetricsResponse)
def get_quantization_metrics():
    """Returns model size, accuracy evaluation, and accuracy budget recommendation."""
    eval_res = evaluate_all_precisions()
    
    s_fp32 = os.path.getsize(FP32_PATH) / (1024 * 1024) if os.path.exists(FP32_PATH) else 0.0
    s_fp16 = os.path.getsize(FP16_PATH) / (1024 * 1024) if os.path.exists(FP16_PATH) else 0.0
    s_int8 = os.path.getsize(INT8_PATH) / (1024 * 1024) if os.path.exists(INT8_PATH) else 0.0
    
    # Recommendation logic: Smallest model that meets <= 1.0% accuracy drop budget
    if eval_res["int8"]["meets_1pct_budget"]:
        recommended = "INT8"
    elif eval_res["fp16"]["meets_1pct_budget"]:
        recommended = "FP16"
    else:
        recommended = "FP32"
        
    return QuantizationMetricsResponse(
        fp32_size_mb=round(s_fp32, 2),
        fp16_size_mb=round(s_fp16, 2),
        int8_size_mb=round(s_int8, 2),
        fp32_accuracy=eval_res["fp32"]["accuracy"],
        fp16_accuracy=eval_res["fp16"]["accuracy"],
        int8_accuracy=eval_res["int8"]["accuracy"],
        fp16_accuracy_drop=eval_res["fp16"]["accuracy_drop"],
        int8_accuracy_drop=eval_res["int8"]["accuracy_drop"],
        recommended_precision_under_1pct_budget=recommended
    )

@app.post("/predict/fp32", response_model=QuantizationResponse)
def predict_fp32(payload: QuantizationRequest):
    batch_size = payload.batch_size
    synthetic = torch.randn(batch_size, 128)
    
    model = models_cache["fp32"]
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(synthetic)
        preds = torch.argmax(logits, dim=1).numpy()
    t1 = time.perf_counter()
    exec_ms = (t1 - t0) * 1000
    
    size_mb = os.path.getsize(FP32_PATH) / (1024 * 1024) if os.path.exists(FP32_PATH) else 0.0
    return QuantizationResponse(
        predictions=[int(p) for p in preds],
        batch_size=batch_size,
        execution_ms=round(exec_ms, 3),
        precision="fp32",
        model_size_mb=round(size_mb, 2)
    )

@app.post("/predict/fp16", response_model=QuantizationResponse)
def predict_fp16(payload: QuantizationRequest):
    batch_size = payload.batch_size
    synthetic = torch.randn(batch_size, 128).half()
    
    model = models_cache["fp16"]
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(synthetic)
        preds = torch.argmax(logits, dim=1).numpy()
    t1 = time.perf_counter()
    exec_ms = (t1 - t0) * 1000
    
    size_mb = os.path.getsize(FP16_PATH) / (1024 * 1024) if os.path.exists(FP16_PATH) else 0.0
    return QuantizationResponse(
        predictions=[int(p) for p in preds],
        batch_size=batch_size,
        execution_ms=round(exec_ms, 3),
        precision="fp16",
        model_size_mb=round(size_mb, 2)
    )

@app.post("/predict/int8", response_model=QuantizationResponse)
def predict_int8(payload: QuantizationRequest):
    batch_size = payload.batch_size
    synthetic = torch.randn(batch_size, 128)
    
    model = models_cache["int8"]
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(synthetic)
        preds = torch.argmax(logits, dim=1).numpy()
    t1 = time.perf_counter()
    exec_ms = (t1 - t0) * 1000
    
    size_mb = os.path.getsize(INT8_PATH) / (1024 * 1024) if os.path.exists(INT8_PATH) else 0.0
    return QuantizationResponse(
        predictions=[int(p) for p in preds],
        batch_size=batch_size,
        execution_ms=round(exec_ms, 3),
        precision="int8",
        model_size_mb=round(size_mb, 2)
    )

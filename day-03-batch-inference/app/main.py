import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from app.model import ModelManager, MODEL_PATH
from app.batcher import batcher

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads model into RAM and starts DynamicBatcher background loop."""
    print("[Lifespan] Pre-loading ML model into memory...")
    ModelManager.load_model(MODEL_PATH)
    print("[Lifespan] Starting Dynamic Batcher queue...")
    await batcher.start()
    yield
    print("[Lifespan] Stopping Dynamic Batcher queue...")
    await batcher.stop()

app = FastAPI(
    title="Day 3 — Batch Inference Lab",
    description="Benchmarking Individual Inference vs Static Batching vs Dynamic Batching Queue.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Batch Inference API is live."}

@app.post("/predict/individual", response_model=SinglePredictionResponse)
def predict_individual(payload: SinglePredictionRequest):
    """
    INDIVIDUAL INFERENCE ENDPOINT:
    Executes model.predict() for 1 single sample per HTTP request.
    No request grouping or vectorization across concurrent users.
    """
    t0 = time.perf_counter()
    try:
        arr = np.array(payload.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature array: {str(e)}")
        
    pred = int(ModelManager.predict_batch(arr)[0])
    t1 = time.perf_counter()
    exec_ms = (t1 - t0) * 1000
    
    return SinglePredictionResponse(
        prediction=pred,
        execution_ms=round(exec_ms, 3),
        batch_size_used=1,
        mode="individual"
    )

@app.post("/predict/static-batch", response_model=BatchPredictionResponse)
def predict_static_batch(payload: BatchPredictionRequest):
    """
    STATIC BATCH ENDPOINT:
    Receives an explicit list of N feature vectors and executes
    a single vectorized model.predict(matrix) call.
    """
    t0 = time.perf_counter()
    try:
        matrix = np.array(payload.batch_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature matrix: {str(e)}")
        
    if matrix.ndim != 2:
        raise HTTPException(status_code=400, detail="Expected 2D array of feature vectors.")
        
    batch_size = matrix.shape[0]
    preds = ModelManager.predict_batch(matrix)
    t1 = time.perf_counter()
    exec_ms = (t1 - t0) * 1000
    
    return BatchPredictionResponse(
        predictions=[int(p) for p in preds],
        execution_ms=round(exec_ms, 3),
        batch_size=batch_size
    )

@app.post("/predict/dynamic-batch", response_model=SinglePredictionResponse)
async def predict_dynamic_batch(payload: SinglePredictionRequest):
    """
    DYNAMIC BATCHING ENDPOINT:
    Enqueues single incoming request into DynamicBatcher queue.
    Background worker accumulates requests into batches up to 16 items or 10ms timeout.
    """
    try:
        pred, batch_size_used, exec_ms = await batcher.process_request(payload.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic batcher error: {str(e)}")
        
    return SinglePredictionResponse(
        prediction=pred,
        execution_ms=exec_ms,
        batch_size_used=batch_size_used,
        mode="dynamic-batch"
    )

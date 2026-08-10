import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import (
    QueuePredictionRequest,
    QueuePredictionResponse,
    QueueMetricsResponse
)
from app.model import ModelManager, MODEL_PATH
from app.inference_queue import (
    bounded_queue,
    QueueFullException,
    QueueTimeoutException
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads model artifact and starts BoundedInferenceQueue background workers."""
    print("[Lifespan] Pre-loading ML model into memory...")
    ModelManager.load_model(MODEL_PATH)
    print("[Lifespan] Starting BoundedInferenceQueue workers...")
    await bounded_queue.start()
    yield
    print("[Lifespan] Stopping BoundedInferenceQueue workers...")
    await bounded_queue.stop()

app = FastAPI(
    title="Day 5 — ML Inference Queue & Backpressure Lab",
    description="Investigating Queues, Backpressure, Queue Depth, Timeouts, and Overload Protection under traffic spikes.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Day 5 Inference Queue API is live."}

@app.get("/metrics/queue", response_model=QueueMetricsResponse)
def get_queue_metrics():
    """Returns real-time queue depth and backpressure telemetry."""
    return bounded_queue.get_metrics()

@app.post("/predict/direct", response_model=QueuePredictionResponse)
def predict_direct(payload: QueuePredictionRequest):
    """
    UNPROTECTED ENDPOINT:
    Executes model.predict() directly without queue bounds, backpressure, or rate limits.
    Demonstrates uncontrolled latency buildup under heavy traffic spikes.
    """
    t0 = time.perf_counter()
    pred = ModelManager.predict(payload.features)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000
    
    return QueuePredictionResponse(
        prediction=pred,
        total_latency_ms=round(lat_ms, 3),
        queue_wait_ms=0.0,
        inference_ms=round(lat_ms, 3),
        queue_depth_at_arrival=0,
        status="success"
    )

@app.post("/predict/queued", response_model=QueuePredictionResponse)
async def predict_queued(payload: QueuePredictionRequest):
    """
    BACKPRESSURE PROTECTED ENDPOINT:
    Enqueues incoming request into BoundedInferenceQueue.
    - If Queue Depth >= 50: Returns HTTP 429 Too Many Requests (Backpressure Protection).
    - If Wait Time > 3.0s: Returns HTTP 504 Gateway Timeout (Eviction).
    """
    try:
        pred, total_lat, queue_wait, depth_at_arrival = await bounded_queue.enqueue_and_process(payload.features)
        return QueuePredictionResponse(
            prediction=pred,
            total_latency_ms=total_lat,
            queue_wait_ms=queue_wait,
            inference_ms=round(total_lat - queue_wait, 3),
            queue_depth_at_arrival=depth_at_arrival,
            status="success"
        )
    except QueueFullException as qe:
        raise HTTPException(
            status_code=429,
            detail={"error": "Too Many Requests", "message": str(qe), "type": "BackpressureProtection"}
        )
    except QueueTimeoutException as te:
        raise HTTPException(
            status_code=504,
            detail={"error": "Gateway Timeout", "message": str(te), "type": "QueueTimeoutEviction"}
        )

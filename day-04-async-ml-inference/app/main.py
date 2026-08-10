import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import InferenceRequest, InferenceResponse
from app.model import ModelManager, MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads model artifact into memory during app startup."""
    print("[Lifespan] Pre-loading ML model into memory...")
    ModelManager.load_model(MODEL_PATH)
    yield
    print("[Lifespan] App shutting down.")

app = FastAPI(
    title="Day 4 — Async vs Blocking ML Inference Lab",
    description="Benchmarking Sync, Async-Blocking, Async-Offloaded, and Async-IO ML endpoints.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Day 4 Async ML Lab API is live."}

@app.post("/predict/sync", response_model=InferenceResponse)
def predict_sync(payload: InferenceRequest):
    """
    PATTERN 1: SYNCHRONOUS ENDPOINT
    Standard 'def' handler. FastAPI executes standard def endpoints on an internal threadpool.
    """
    t0 = time.perf_counter()
    pred = ModelManager.predict(payload.features)
    t1 = time.perf_counter()
    return InferenceResponse(
        prediction=pred,
        execution_ms=round((t1 - t0) * 1000, 3),
        mode="sync",
        io_simulated=False
    )

@app.post("/predict/async-blocking", response_model=InferenceResponse)
async def predict_async_blocking(payload: InferenceRequest):
    """
    PATTERN 2: ASYNC BLOCKING ENDPOINT
    'async def' handler that executes CPU-bound model.predict() directly on the Event Loop.
    WARNING: Blocks the asyncio event loop during matrix computation!
    """
    t0 = time.perf_counter()
    # Direct call on asyncio event loop (BLOCKING)
    pred = ModelManager.predict(payload.features)
    t1 = time.perf_counter()
    return InferenceResponse(
        prediction=pred,
        execution_ms=round((t1 - t0) * 1000, 3),
        mode="async-blocking",
        io_simulated=False
    )

@app.post("/predict/async-offloaded", response_model=InferenceResponse)
async def predict_async_offloaded(payload: InferenceRequest):
    """
    PATTERN 3: ASYNC OFFLOADED ENDPOINT
    'async def' handler that offloads CPU-bound model.predict() to a worker threadpool
    using asyncio.to_thread(), keeping the asyncio event loop non-blocking.
    """
    t0 = time.perf_counter()
    # Offload CPU work to threadpool
    pred = await asyncio.to_thread(ModelManager.predict, payload.features)
    t1 = time.perf_counter()
    return InferenceResponse(
        prediction=pred,
        execution_ms=round((t1 - t0) * 1000, 3),
        mode="async-offloaded",
        io_simulated=False
    )

@app.post("/predict/sync-io", response_model=InferenceResponse)
def predict_sync_io(payload: InferenceRequest):
    """
    PATTERN 4A: SYNCHRONOUS I/O + INFERENCE
    Simulates a 20ms synchronous database/feature store lookup (time.sleep).
    """
    t0 = time.perf_counter()
    time.sleep(0.02)  # Simulate 20ms sync DB lookup
    pred = ModelManager.predict(payload.features)
    t1 = time.perf_counter()
    return InferenceResponse(
        prediction=pred,
        execution_ms=round((t1 - t0) * 1000, 3),
        mode="sync-io",
        io_simulated=True
    )

@app.post("/predict/async-io", response_model=InferenceResponse)
async def predict_async_io(payload: InferenceRequest):
    """
    PATTERN 4B: ASYNCHRONOUS I/O + OFFLOADED INFERENCE
    Simulates a 20ms async database/feature store lookup (await asyncio.sleep)
    followed by offloaded threadpool inference.
    """
    t0 = time.perf_counter()
    await asyncio.sleep(0.02)  # Simulate 20ms non-blocking async DB lookup
    pred = await asyncio.to_thread(ModelManager.predict, payload.features)
    t1 = time.perf_counter()
    return InferenceResponse(
        prediction=pred,
        execution_ms=round((t1 - t0) * 1000, 3),
        mode="async-io",
        io_simulated=True
    )

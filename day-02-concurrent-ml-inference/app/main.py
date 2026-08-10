import os
import time
import psutil
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model import ModelManager, MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads model artifact into worker process memory on startup."""
    pid = os.getpid()
    print(f"[Lifespan PID {pid}] Initializing model in process memory...")
    t0 = time.perf_counter()
    ModelManager.load_model(MODEL_PATH)
    t1 = time.perf_counter()
    print(f"[Lifespan PID {pid}] Model loaded in {(t1 - t0)*1000:.2f} ms.")
    yield
    print(f"[Lifespan PID {pid}] Shutting down worker process.")

app = FastAPI(
    title="Day 2 — Concurrent ML Inference Lab",
    description="Benchmarking FastAPI single vs multi-worker behavior under high concurrent load.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "worker_pid": os.getpid(),
        "message": "Serving concurrent ML predictions."
    }

@app.get("/metrics/process")
def process_metrics():
    """Returns memory footprint (RSS MB) and process info for this worker."""
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return {
        "worker_pid": os.getpid(),
        "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
        "memory_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
        "cpu_percent": proc.cpu_percent(interval=None)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """Synchronous CPU prediction handler serviced by worker process."""
    t_start = time.perf_counter()
    
    model = ModelManager.get_model()
    
    try:
        features_array = np.array(payload.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature array: {str(e)}")
        
    prediction = int(model.predict(features_array)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_array)[0].tolist()
        
    t_end = time.perf_counter()
    exec_ms = (t_end - t_start) * 1000
    
    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        worker_pid=os.getpid(),
        execution_ms=round(exec_ms, 3)
    )

import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import InferenceInput, InferenceResponse, TimingBreakdown
from app.model import load_model_from_disk, ModelManager, train_and_save_model, MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager — loads model ONCE at application startup."""
    print("[Lifespan] Application starting up. Pre-loading ML model into memory...")
    start_time = time.perf_counter()
    ModelManager.load_model(MODEL_PATH)
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Lifespan] Model pre-loaded successfully in {elapsed:.2f} ms.")
    yield
    print("[Lifespan] Application shutting down.")

app = FastAPI(
    title="Day 1 — ML Inference Latency Lab",
    description="Benchmarking Naive (per-request model loading) vs Optimized (startup lifecycle caching) ML APIs.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ML Inference API is live."}

@app.post("/predict/naive", response_model=InferenceResponse)
def predict_naive(payload: InferenceInput):
    """
    NAIVE ENDPOINT:
    Loads the model artifact from disk ON EVERY REQUEST.
    Demonstrates severe latency bottleneck caused by repeating I/O and deserialization.
    """
    t_start = time.perf_counter()
    
    # 1. Parsing finished when payload entered endpoint
    t_parse = time.perf_counter()
    parse_ms = (t_parse - t_start) * 1000
    
    # 2. Model Loading (Bottleneck!)
    t_load_start = time.perf_counter()
    model = load_model_from_disk(MODEL_PATH)
    t_load_end = time.perf_counter()
    load_ms = (t_load_end - t_load_start) * 1000
    
    # 3. Feature Preprocessing
    t_prep_start = time.perf_counter()
    try:
        features_array = np.array(payload.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature array: {str(e)}")
    t_prep_end = time.perf_counter()
    prep_ms = (t_prep_end - t_prep_start) * 1000
    
    # 4. Model Inference
    t_inf_start = time.perf_counter()
    prediction = int(model.predict(features_array)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_array)[0].tolist()
    t_inf_end = time.perf_counter()
    inf_ms = (t_inf_end - t_inf_start) * 1000
    
    # 5. Serialization
    t_ser_start = time.perf_counter()
    timing = TimingBreakdown(
        request_parsing_ms=round(parse_ms, 3),
        model_loading_ms=round(load_ms, 3),
        preprocessing_ms=round(prep_ms, 3),
        model_inference_ms=round(inf_ms, 3),
        serialization_ms=0.0,  # calculated below
        total_pipeline_ms=0.0
    )
    t_ser_end = time.perf_counter()
    ser_ms = (t_ser_end - t_ser_start) * 1000
    timing.serialization_ms = round(ser_ms, 3)
    
    t_end = time.perf_counter()
    timing.total_pipeline_ms = round((t_end - t_start) * 1000, 3)
    
    return InferenceResponse(
        prediction=prediction,
        probabilities=probabilities,
        mode="naive",
        timing=timing
    )

@app.post("/predict/optimized", response_model=InferenceResponse)
def predict_optimized(payload: InferenceInput):
    """
    OPTIMIZED ENDPOINT:
    Uses model pre-loaded into RAM during application startup.
    Model loading cost per request is 0 ms.
    """
    t_start = time.perf_counter()
    
    # 1. Request Parsing
    t_parse = time.perf_counter()
    parse_ms = (t_parse - t_start) * 1000
    
    # 2. Model Loading (In-Memory Access -> ~0 ms)
    t_load_start = time.perf_counter()
    model = ModelManager.get_model()
    t_load_end = time.perf_counter()
    load_ms = (t_load_end - t_load_start) * 1000
    
    # 3. Preprocessing
    t_prep_start = time.perf_counter()
    try:
        features_array = np.array(payload.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature array: {str(e)}")
    t_prep_end = time.perf_counter()
    prep_ms = (t_prep_end - t_prep_start) * 1000
    
    # 4. Model Inference
    t_inf_start = time.perf_counter()
    prediction = int(model.predict(features_array)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_array)[0].tolist()
    t_inf_end = time.perf_counter()
    inf_ms = (t_inf_end - t_inf_start) * 1000
    
    # 5. Serialization
    t_ser_start = time.perf_counter()
    timing = TimingBreakdown(
        request_parsing_ms=round(parse_ms, 3),
        model_loading_ms=round(load_ms, 3),
        preprocessing_ms=round(prep_ms, 3),
        model_inference_ms=round(inf_ms, 3),
        serialization_ms=0.0,
        total_pipeline_ms=0.0
    )
    t_ser_end = time.perf_counter()
    ser_ms = (t_ser_end - t_ser_start) * 1000
    timing.serialization_ms = round(ser_ms, 3)
    
    t_end = time.perf_counter()
    timing.total_pipeline_ms = round((t_end - t_start) * 1000, 3)
    
    return InferenceResponse(
        prediction=prediction,
        probabilities=probabilities,
        mode="optimized",
        timing=timing
    )

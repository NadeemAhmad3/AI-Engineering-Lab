import time
import uuid
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.schemas import InferenceRequest, InferenceResponse, ChaosInjectionRequest
from inference.quantization import get_quantized_int8_model
from inference.batcher import DynamicBatchManager
from queue_manager.inference_queue import BoundedInferenceQueue
from cache.cache import HybridCache
from monitoring.logging import structured_logger
from monitoring.tracing import TraceSpan
from monitoring.metrics import platform_metrics
from chaos.scenarios import chaos_controller

MODEL_VERSION = "v1.0.0"

# Platform components
quantized_model = None
batch_manager = None
bounded_queue = BoundedInferenceQueue(max_queue_size=100)
hybrid_cache = HybridCache(semantic_threshold=0.40)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global quantized_model, batch_manager
    print("[Lifespan] Initializing Production Capstone AI Inference System...")
    quantized_model = get_quantized_int8_model()
    batch_manager = DynamicBatchManager(model=quantized_model, max_batch_size=16, max_wait_time_sec=0.005)
    batch_manager.start()
    yield
    print("[Lifespan] Production Capstone System shutdown.")

app = FastAPI(
    title="Day 10 — Production AI Inference Platform Capstone",
    description="End-to-End Production AI Platform integrating Rate Limiting, Hybrid Caching, Bounded Queuing & Backpressure, Dynamic Batching, INT8 Quantization, and Full Telemetry.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Day 10 Production AI Inference Platform is online.",
        "slo_compliance": platform_metrics.check_slo_compliance()
    }

@app.get("/metrics")
def get_platform_metrics():
    """Returns complete telemetry metrics and SLO compliance payload."""
    return platform_metrics.get_telemetry()

@app.post("/chaos/inject")
def inject_chaos(payload: ChaosInjectionRequest):
    chaos_controller.configure(
        slow_model=payload.slow_model_enabled,
        delay_ms=payload.slow_model_delay_ms,
        cache_fail=payload.cache_failure_enabled,
        queue_overload=payload.queue_overload_enabled
    )
    return {"status": "chaos_configured", "chaos": payload.model_dump()}

@app.post("/predict", response_model=InferenceResponse)
async def predict(payload: InferenceRequest):
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    m_ver = payload.model_version or MODEL_VERSION
    query_text = payload.query_text or str(payload.features[:10])
    
    tracer = TraceSpan(trace_id=req_id)
    
    # 1. API Ingress Span
    tracer.start("api_ingress")
    time.sleep(0.0005)
    tracer.end("api_ingress")

    # 2. Check Queue Overload Chaos / Backpressure
    if chaos_controller.queue_overload_enabled or not bounded_queue.try_acquire():
        platform_metrics.record_rejection()
        structured_logger.log_request(
            request_id=req_id,
            model_version=m_ver,
            batch_size=1,
            cache_hit=False,
            queue_wait_ms=100.0,
            inference_ms=0.0,
            total_latency_ms=100.0,
            status="error_429",
            error_msg="Backpressure Rejection: Queue Capacity Exceeded"
        )
        raise HTTPException(status_code=429, detail="HTTP 429: Queue Capacity Exceeded")

    try:
        # 3. Cache Lookup Span
        tracer.start("cache_lookup")
        cached_val, is_hit, similarity_score = None, False, 0.0
        if not payload.force_cache_miss and not chaos_controller.cache_failure_enabled:
            cached_val, is_hit, similarity_score = hybrid_cache.get(query_text, m_ver)
        tracer.end("cache_lookup")

        if is_hit and cached_val is not None:
            bounded_queue.release()
            tracer.start("response_serialize")
            time.sleep(0.0005)
            tracer.end("response_serialize")
            
            summary = tracer.get_summary()
            tot_ms = summary["total_trace_ms"]
            
            platform_metrics.record_request(latency_ms=tot_ms, cache_hit=True, success=True)
            structured_logger.log_request(
                request_id=req_id,
                model_version=m_ver,
                batch_size=1,
                cache_hit=True,
                queue_wait_ms=0.5,
                inference_ms=0.0,
                total_latency_ms=tot_ms,
                status="success"
            )
            return InferenceResponse(
                request_id=req_id,
                prediction=int(cached_val),
                latency_ms=tot_ms,
                cache_hit=True,
                similarity_score=similarity_score,
                trace=summary
            )

        # 4. Queue Wait Span
        tracer.start("queue_wait")
        t_q0 = time.perf_counter()
        time.sleep(0.001)
        tracer.end("queue_wait")
        queue_wait_ms = (time.perf_counter() - t_q0) * 1000.0

        # 5. Model Inference Span
        tracer.start("model_inference")
        if chaos_controller.slow_model_enabled:
            time.sleep(chaos_controller.slow_model_delay_ms / 1000.0)

        # Prepare tensor
        feat_list = payload.features if len(payload.features) == 128 else ([0.1] * 128)
        tensor_input = torch.tensor(feat_list, dtype=torch.float32)
        
        # Async prediction through Dynamic Batch Manager
        pred_val = await batch_manager.predict_async(tensor_input)
        tracer.end("model_inference")

        # 6. Store in Cache
        hybrid_cache.put(query_text, m_ver, pred_val)

        # 7. Serialization Span
        tracer.start("response_serialize")
        time.sleep(0.0005)
        tracer.end("response_serialize")

        summary = tracer.get_summary()
        tot_ms = summary["total_trace_ms"]

        platform_metrics.record_request(latency_ms=tot_ms, cache_hit=False, success=True)
        structured_logger.log_request(
            request_id=req_id,
            model_version=m_ver,
            batch_size=16,
            cache_hit=False,
            queue_wait_ms=queue_wait_ms,
            inference_ms=summary["spans_ms"].get("model_inference", 0.0),
            total_latency_ms=tot_ms,
            status="success"
        )

        return InferenceResponse(
            request_id=req_id,
            prediction=pred_val,
            latency_ms=tot_ms,
            cache_hit=False,
            similarity_score=similarity_score,
            trace=summary
        )
    finally:
        bounded_queue.release()

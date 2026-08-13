import time
import uuid
from fastapi import FastAPI, HTTPException
from app.schemas import InferenceRequest, InferenceResponse, ChaosInjectionRequest
from app.logging import structured_logger
from app.tracing import TraceSpan
from app.metrics import metrics_collector
from chaos.degradation import chaos_controller

app = FastAPI(
    title="Day 9 — AI Observability & Reliability Lab",
    description="Structured JSON Logging, Metrics Percentiles (P50/P95/P99), Trace Spans, and Chaos Degradation Testing.",
    version="1.0.0"
)

# Simulated in-memory cache
cache_store = {}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Day 9 AI Observability API is live."}

@app.get("/health/detailed")
def detailed_health_check():
    """Returns System Health Score and detailed telemetry status."""
    return metrics_collector.get_system_health_score()

@app.get("/metrics/telemetry")
def get_metrics_telemetry():
    """Returns full metrics telemetry payload including latency percentiles."""
    return metrics_collector.get_telemetry()

@app.post("/chaos/inject")
def inject_chaos(payload: ChaosInjectionRequest):
    """Enables or disables chaos failure scenarios."""
    chaos_controller.configure(
        slow_inf=payload.slow_inference_enabled,
        delay_ms=payload.slow_inference_delay_ms,
        cache_fail=payload.cache_failure_enabled,
        queue_overload=payload.queue_overload_enabled
    )
    return {
        "status": "chaos_updated",
        "slow_inference": chaos_controller.slow_inference_enabled,
        "cache_failure": chaos_controller.cache_failure_enabled,
        "queue_overload": chaos_controller.queue_overload_enabled
    }

@app.post("/predict", response_model=InferenceResponse)
def predict(payload: InferenceRequest):
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    tracer = TraceSpan(trace_id=req_id)
    
    # 1. API Ingress Span
    tracer.start_span("api_ingress")
    time.sleep(0.001)  # 1ms ingress
    tracer.end_span("api_ingress")

    # 2. Check Queue Overload Chaos
    if chaos_controller.queue_overload_enabled:
        metrics_collector.record_rejection()
        structured_logger.log_request(
            request_id=req_id,
            model_version="v1.2.0",
            batch_size=1,
            cache_hit=False,
            queue_wait_ms=250.0,
            inference_ms=0.0,
            total_latency_ms=250.0,
            status="error_429",
            error_msg="Queue Capacity Exceeded (Chaos Injection)"
        )
        raise HTTPException(status_code=429, detail="HTTP 429: Queue Capacity Exceeded")

    # 3. Cache Lookup Span
    tracer.start_span("cache_lookup")
    cache_hit = False
    if not payload.force_cache_miss and not chaos_controller.cache_failure_enabled:
        if payload.query in cache_store:
            cache_hit = True
    tracer.end_span("cache_lookup")

    if cache_hit:
        res_text = cache_store[payload.query]
        tracer.start_span("response_serialize")
        time.sleep(0.001)
        tracer.end_span("response_serialize")
        
        summary = tracer.get_trace_summary()
        tot_ms = summary["total_trace_ms"]
        
        metrics_collector.record_request(latency_ms=tot_ms, queue_wait_ms=0.5, cache_hit=True, success=True)
        structured_logger.log_request(
            request_id=req_id,
            model_version="v1.2.0",
            batch_size=1,
            cache_hit=True,
            queue_wait_ms=0.5,
            inference_ms=0.0,
            total_latency_ms=tot_ms,
            status="success"
        )
        return InferenceResponse(
            request_id=req_id,
            result=res_text,
            latency_ms=tot_ms,
            cache_hit=True,
            trace=summary
        )

    # 4. Queue Wait Span
    tracer.start_span("queue_wait")
    queue_wait_ms = 5.0
    time.sleep(queue_wait_ms / 1000.0)
    tracer.end_span("queue_wait")

    # 5. Model Inference Span
    tracer.start_span("model_inference")
    inf_delay_sec = 0.015  # Baseline 15ms
    if chaos_controller.slow_inference_enabled:
        inf_delay_sec += (chaos_controller.slow_inference_delay_ms / 1000.0)
    time.sleep(inf_delay_sec)
    
    res_text = f"Processed prediction output for: '{payload.query}'"
    cache_store[payload.query] = res_text
    tracer.end_span("model_inference")

    # 6. Response Serialize Span
    tracer.start_span("response_serialize")
    time.sleep(0.001)
    tracer.end_span("response_serialize")

    summary = tracer.get_trace_summary()
    tot_ms = summary["total_trace_ms"]

    metrics_collector.record_request(
        latency_ms=tot_ms,
        queue_wait_ms=queue_wait_ms,
        cache_hit=False,
        success=True
    )
    
    structured_logger.log_request(
        request_id=req_id,
        model_version="v1.2.0",
        batch_size=1,
        cache_hit=False,
        queue_wait_ms=queue_wait_ms,
        inference_ms=summary["spans_ms"].get("model_inference", 0.0),
        total_latency_ms=tot_ms,
        status="success"
    )

    return InferenceResponse(
        request_id=req_id,
        result=res_text,
        latency_ms=tot_ms,
        cache_hit=False,
        trace=summary
    )

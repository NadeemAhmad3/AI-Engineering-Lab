import time
from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, CacheResponse
from app.model import ai_engine, MODEL_VERSION
from cache.exact_cache import exact_cache
from cache.semantic_cache import semantic_cache

app = FastAPI(
    title="Day 8 — AI-Aware Caching Engineering Lab",
    description="Investigating Exact-Match vs Semantic Vector Caching, Latency Reduction, Cache Hit Rates, and Cost Savings.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Day 8 AI Caching API is live."}

@app.get("/metrics/cache")
def get_cache_metrics():
    """Returns exact and semantic cache hit/miss telemetry."""
    return {
        "exact_cache": exact_cache.get_metrics(),
        "semantic_cache": semantic_cache.get_metrics()
    }

@app.post("/predict/no-cache", response_model=CacheResponse)
def predict_no_cache(payload: QueryRequest):
    """
    NO-CACHE ENDPOINT:
    Executes full expensive model inference on every single request (~20ms latency).
    """
    t0 = time.perf_counter()
    res = ai_engine.predict(payload.text)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000
    
    return CacheResponse(
        result=res,
        latency_ms=round(lat_ms, 2),
        cache_hit=False,
        cache_strategy="no-cache",
        similarity_score=0.0
    )

@app.post("/predict/exact-cache", response_model=CacheResponse)
def predict_exact_cache(payload: QueryRequest):
    """
    EXACT-MATCH CACHE ENDPOINT:
    Looks up exact hash(model_version + text).
    Hits return in < 0.5ms; misses trigger 20ms AI inference and store result.
    """
    m_ver = payload.model_version or MODEL_VERSION
    t0 = time.perf_counter()
    
    cached = exact_cache.get(payload.text, m_ver)
    if cached is not None:
        t1 = time.perf_counter()
        return CacheResponse(
            result=cached,
            latency_ms=round((t1 - t0) * 1000, 2),
            cache_hit=True,
            cache_strategy="exact-cache",
            similarity_score=1.0
        )

    # Cache Miss: Run Inference
    res = ai_engine.predict(payload.text)
    exact_cache.put(payload.text, m_ver, res)
    t1 = time.perf_counter()
    
    return CacheResponse(
        result=res,
        latency_ms=round((t1 - t0) * 1000, 2),
        cache_hit=False,
        cache_strategy="exact-cache",
        similarity_score=0.0
    )

@app.post("/predict/semantic-cache", response_model=CacheResponse)
def predict_semantic_cache(payload: QueryRequest):
    """
    SEMANTIC VECTOR CACHE ENDPOINT:
    Performs cosine vector search against query embeddings.
    If similarity >= 0.90, returns cached AI result instantly!
    """
    t0 = time.perf_counter()
    cached, similarity, is_hit = semantic_cache.lookup(payload.text)
    
    if is_hit and cached is not None:
        t1 = time.perf_counter()
        return CacheResponse(
            result=cached,
            latency_ms=round((t1 - t0) * 1000, 2),
            cache_hit=True,
            cache_strategy="semantic-cache",
            similarity_score=similarity
        )

    # Cache Miss: Run Inference and Store in Vector Cache
    res = ai_engine.predict(payload.text)
    semantic_cache.put(payload.text, res)
    t1 = time.perf_counter()
    
    return CacheResponse(
        result=res,
        latency_ms=round((t1 - t0) * 1000, 2),
        cache_hit=False,
        cache_strategy="semantic-cache",
        similarity_score=similarity
    )

import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY8_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY8_DIR not in sys.path:
    sys.path.insert(0, DAY8_DIR)

from app.main import app
from cache.exact_cache import ExactMatchCache
from cache.semantic_cache import SemanticVectorCache

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_exact_cache_hits_and_misses():
    cache = ExactMatchCache()
    # Initial miss
    assert cache.get("hello world", "v1") is None
    
    # Put entry
    cache.put("hello world", "v1", {"output": "result_1"})
    
    # Hit
    cached = cache.get("hello world", "v1")
    assert cached is not None
    assert cached["output"] == "result_1"
    
    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1

def test_exact_cache_model_version_invalidation():
    cache = ExactMatchCache()
    cache.put("summarize report", "v1.0.0", {"data": "v1_summary"})
    
    # Same query, new model version -> Miss (Invalidation)
    assert cache.get("summarize report", "v2.0.0") is None

def test_semantic_cache_lookup():
    sem = SemanticVectorCache(similarity_threshold=0.30)
    sem.put("What is machine learning?", {"ans": "ML explanation"})
    
    # Semantically equivalent query -> Hit
    cached, score, is_hit = sem.lookup("Explain machine learning")
    assert is_hit is True
    assert cached["ans"] == "ML explanation"
    assert score >= 0.30

def test_no_cache_endpoint():
    with TestClient(app) as client:
        payload = {"text": "What is AI?"}
        res = client.post("/predict/no-cache", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["cache_hit"] is False
        assert data["cache_strategy"] == "no-cache"

def test_exact_cache_endpoint():
    with TestClient(app) as client:
        payload = {"text": "What is the capital of France?"}
        # First call -> Miss
        res1 = client.post("/predict/exact-cache", json=payload)
        assert res1.status_code == 200
        assert res1.json()["cache_hit"] is False
        
        # Second call -> Hit
        res2 = client.post("/predict/exact-cache", json=payload)
        assert res2.status_code == 200
        assert res2.json()["cache_hit"] is True
        assert res2.json()["latency_ms"] < res1.json()["latency_ms"]

def test_semantic_cache_endpoint():
    with TestClient(app) as client:
        payload1 = {"text": "What is the capital of Pakistan?"}
        payload2 = {"text": "Which city is Pakistan's capital?"}
        
        # Store query 1
        res1 = client.post("/predict/semantic-cache", json=payload1)
        assert res1.status_code == 200
        
        # Query 2 (Semantically equivalent) -> Hit
        res2 = client.post("/predict/semantic-cache", json=payload2)
        assert res2.status_code == 200
        assert res2.json()["cache_hit"] is True
        assert res2.json()["similarity_score"] >= 0.50

def test_metrics_endpoint():
    with TestClient(app) as client:
        res = client.get("/metrics/cache")
        assert res.status_code == 200
        data = res.json()
        assert "exact_cache" in data
        assert "semantic_cache" in data

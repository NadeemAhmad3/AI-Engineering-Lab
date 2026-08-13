import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY9_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY9_DIR not in sys.path:
    sys.path.insert(0, DAY9_DIR)

from app.main import app
from app.metrics import MetricsCollector
from chaos.degradation import chaos_controller

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_health_detailed():
    with TestClient(app) as client:
        res = client.get("/health/detailed")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "p95_latency_ms" in data

def test_metrics_telemetry():
    with TestClient(app) as client:
        res = client.get("/metrics/telemetry")
        assert res.status_code == 200
        data = res.json()
        assert "percentiles" in data
        assert "p95_ms" in data["percentiles"]
        assert "cache_hit_rate_pct" in data

def test_predict_endpoint_and_tracing():
    with TestClient(app) as client:
        chaos_controller.reset()
        payload = {"query": "Test observability endpoint"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "trace" in data
        assert "total_trace_ms" in data["trace"]

def test_chaos_slow_inference():
    with TestClient(app) as client:
        chaos_controller.configure(slow_inf=True, delay_ms=150.0)
        payload = {"query": "Slow test query", "force_cache_miss": True}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["latency_ms"] >= 150.0

def test_chaos_queue_overload():
    with TestClient(app) as client:
        chaos_controller.configure(queue_overload=True)
        payload = {"query": "Overload test query"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 429
        chaos_controller.reset()

def test_metrics_percentiles_calculation():
    m = MetricsCollector()
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0]:
        m.record_request(latency_ms=lat, queue_wait_ms=1.0, cache_hit=False)
    
    pcts = m.calculate_percentiles()
    assert pcts["p50_ms"] > 0
    assert pcts["p95_ms"] >= pcts["p50_ms"]

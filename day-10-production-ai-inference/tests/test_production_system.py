import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY10_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY10_DIR not in sys.path:
    sys.path.insert(0, DAY10_DIR)

from app.main import app
from inference.model import get_or_create_fp32_model
from inference.quantization import get_quantized_int8_model
from chaos.scenarios import chaos_controller

def test_health_and_slo():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "slo_compliance" in data

def test_platform_metrics_telemetry():
    with TestClient(app) as client:
        res = client.get("/metrics")
        assert res.status_code == 200
        data = res.json()
        assert "percentiles" in data
        assert "slo_status" in data

def test_predict_end_to_end():
    with TestClient(app) as client:
        payload = {
            "features": [0.1] * 128,
            "query_text": "Production system capstone test query",
            "model_version": "v1.0.0"
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "request_id" in data
        assert "prediction" in data
        assert "trace" in data

def test_hybrid_cache_integration():
    with TestClient(app) as client:
        payload = {
            "features": [0.2] * 128,
            "query_text": "Caching test query",
            "model_version": "v1.0.0"
        }
        # First call -> Miss
        res1 = client.post("/predict", json=payload)
        assert res1.status_code == 200
        assert res1.json()["cache_hit"] is False
        
        # Second call -> Hit
        res2 = client.post("/predict", json=payload)
        assert res2.status_code == 200
        assert res2.json()["cache_hit"] is True

def test_chaos_queue_overload():
    with TestClient(app) as client:
        chaos_controller.configure(queue_overload=True)
        payload = {"features": [0.1] * 128}
        res = client.post("/predict", json=payload)
        assert res.status_code == 429
        chaos_controller.reset()

def test_quantized_model_loading():
    model = get_quantized_int8_model()
    assert model is not None

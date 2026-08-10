import os
import sys
import pytest
import asyncio
from fastapi.testclient import TestClient

DAY5_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY5_DIR not in sys.path:
    sys.path.insert(0, DAY5_DIR)

from app.main import app
from app.model import train_and_save_model, MODEL_PATH
from app.inference_queue import bounded_queue, QueueFullException, QueueTimeoutException

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    train_and_save_model(MODEL_PATH)

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_metrics_endpoint():
    with TestClient(app) as client:
        res = client.get("/metrics/queue")
        assert res.status_code == 200
        data = res.json()
        assert "current_queue_depth" in data
        assert "max_queue_capacity" in data

def test_predict_direct():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/direct", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "prediction" in data

def test_predict_queued_success():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/queued", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "prediction" in data

def test_queue_backpressure_rejection():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        orig_cap = bounded_queue.max_capacity
        bounded_queue.max_capacity = 0  # Set capacity to 0 to trigger immediate backpressure rejection
        
        try:
            res_rej = client.post("/predict/queued", json=payload)
            assert res_rej.status_code == 429
            assert res_rej.json()["detail"]["error"] == "Too Many Requests"
        finally:
            bounded_queue.max_capacity = orig_cap

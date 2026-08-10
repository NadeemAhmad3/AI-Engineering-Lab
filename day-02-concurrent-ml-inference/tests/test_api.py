import os
import sys
import pytest
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor

DAY2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY2_DIR not in sys.path:
    sys.path.insert(0, DAY2_DIR)

from app.main import app
from app.model import train_and_save_model, MODEL_PATH

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    train_and_save_model(MODEL_PATH)

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "worker_pid" in data

def test_process_metrics():
    with TestClient(app) as client:
        res = client.get("/metrics/process")
        assert res.status_code == 200
        data = res.json()
        assert "memory_rss_mb" in data
        assert data["memory_rss_mb"] > 0

def test_predict():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "prediction" in data
        assert "execution_ms" in data
        assert "worker_pid" in data

def test_concurrent_requests():
    with TestClient(app) as client:
        def make_req():
            return client.post("/predict", json={"features": [0.1] * 50})

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_req) for _ in range(10)]
            responses = [f.result() for f in futures]

        for res in responses:
            assert res.status_code == 200
            assert "prediction" in res.json()

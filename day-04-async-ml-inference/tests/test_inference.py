import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY4_DIR not in sys.path:
    sys.path.insert(0, DAY4_DIR)

from app.main import app
from app.model import train_and_save_model, MODEL_PATH

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    train_and_save_model(MODEL_PATH)

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_predict_sync():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/sync", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "sync"
        assert "prediction" in data

def test_predict_async_blocking():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/async-blocking", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "async-blocking"
        assert "prediction" in data

def test_predict_async_offloaded():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/async-offloaded", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "async-offloaded"
        assert "prediction" in data

def test_predict_sync_io():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/sync-io", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "sync-io"
        assert data["io_simulated"] is True

def test_predict_async_io():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/async-io", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "async-io"
        assert data["io_simulated"] is True

import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY3_DIR not in sys.path:
    sys.path.insert(0, DAY3_DIR)

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

def test_predict_individual():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/individual", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "individual"
        assert "prediction" in data
        assert data["batch_size_used"] == 1

def test_predict_static_batch():
    with TestClient(app) as client:
        payload = {"batch_features": [[0.5] * 50, [0.2] * 50, [0.8] * 50]}
        res = client.post("/predict/static-batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["batch_size"] == 3
        assert len(data["predictions"]) == 3

def test_predict_dynamic_batch():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        res = client.post("/predict/dynamic-batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "dynamic-batch"
        assert "prediction" in data
        assert data["batch_size_used"] >= 1

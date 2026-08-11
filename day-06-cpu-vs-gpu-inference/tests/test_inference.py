import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY6_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY6_DIR not in sys.path:
    sys.path.insert(0, DAY6_DIR)

from app.main import app

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_metrics_hardware():
    with TestClient(app) as client:
        res = client.get("/metrics/hardware")
        assert res.status_code == 200
        data = res.json()
        assert "cuda_available" in data
        assert "device_name" in data

def test_predict_cpu():
    with TestClient(app) as client:
        payload = {"batch_size": 4}
        res = client.post("/predict/cpu", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["device"] == "cpu"
        assert data["batch_size"] == 4
        assert len(data["predictions"]) == 4

def test_predict_gpu():
    with TestClient(app) as client:
        payload = {"batch_size": 4}
        res = client.post("/predict/gpu", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["batch_size"] == 4
        assert len(data["predictions"]) == 4

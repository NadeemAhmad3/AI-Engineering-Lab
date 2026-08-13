import os
import sys
import pytest
from fastapi.testclient import TestClient

DAY7_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY7_DIR not in sys.path:
    sys.path.insert(0, DAY7_DIR)

from app.main import app
from src.evaluate import evaluate_all_precisions

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_metrics_quantization():
    with TestClient(app) as client:
        res = client.get("/metrics/quantization")
        assert res.status_code == 200
        data = res.json()
        assert "fp32_size_mb" in data
        assert "recommended_precision_under_1pct_budget" in data

def test_predict_fp32():
    with TestClient(app) as client:
        payload = {"batch_size": 4}
        res = client.post("/predict/fp32", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["precision"] == "fp32"
        assert len(data["predictions"]) == 4

def test_predict_fp16():
    with TestClient(app) as client:
        payload = {"batch_size": 4}
        res = client.post("/predict/fp16", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["precision"] == "fp16"
        assert len(data["predictions"]) == 4

def test_predict_int8():
    with TestClient(app) as client:
        payload = {"batch_size": 4}
        res = client.post("/predict/int8", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["precision"] == "int8"
        assert len(data["predictions"]) == 4

def test_accuracy_evaluation():
    results = evaluate_all_precisions()
    assert "fp32" in results
    assert "fp16" in results
    assert "int8" in results
    assert results["fp32"]["accuracy"] > 0

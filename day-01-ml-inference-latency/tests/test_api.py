import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root day-01 directory is in sys.path
DAY1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY1_DIR not in sys.path:
    sys.path.insert(0, DAY1_DIR)

from app.main import app
from app.model import train_and_save_model, MODEL_PATH

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure trained model artifact exists before running API tests."""
    train_and_save_model(MODEL_PATH)

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_predict_naive():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        response = client.post("/predict/naive", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "naive"
        assert "prediction" in data
        assert "timing" in data
        assert data["timing"]["model_loading_ms"] > 0

def test_predict_optimized():
    with TestClient(app) as client:
        payload = {"features": [0.5] * 50}
        response = client.post("/predict/optimized", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "optimized"
        assert "prediction" in data
        assert "timing" in data
        assert data["timing"]["model_loading_ms"] < 0.1

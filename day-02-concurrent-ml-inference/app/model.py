import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def train_and_save_model(model_path: str = MODEL_PATH):
    """Trains a Random Forest classifier model artifact for concurrency benchmarking."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("Training Random Forest model for Day 2 concurrency experiment...")
    X, y = make_classification(
        n_samples=5000,
        n_features=50,
        n_informative=30,
        n_classes=3,
        random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf.fit(X, y)
    
    joblib.dump(clf, model_path, compress=3)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model saved to {model_path} (Size: {file_size_mb:.2f} MB)")
    return clf

class ModelManager:
    """Singleton model container initialized once per worker process during lifespan startup."""
    _model = None

    @classmethod
    def load_model(cls, model_path: str = MODEL_PATH):
        if cls._model is None:
            if not os.path.exists(model_path):
                train_and_save_model(model_path)
            cls._model = joblib.load(model_path)
            print(f"[Worker PID: {os.getpid()}] Model loaded into process memory.")
        return cls._model

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.load_model(MODEL_PATH)
        return cls._model

if __name__ == "__main__":
    train_and_save_model()

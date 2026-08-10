import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def train_and_save_model(model_path: str = MODEL_PATH):
    """
    Trains a Random Forest ensemble model with enough estimators and depth to create
    a realistic serialized artifact (~15-20 MB), highlighting disk I/O & unpickling overhead.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Training ML model for Day 1 experiment...")
    X, y = make_classification(
        n_samples=5000,
        n_features=50,
        n_informative=30,
        n_classes=3,
        random_state=42
    )
    
    # Random Forest with 250 trees to ensure measurable pickling overhead
    clf = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42)
    clf.fit(X, y)
    
    # Save serialized model artifact
    joblib.dump(clf, model_path, compress=3)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model successfully saved to {model_path} (Size: {file_size_mb:.2f} MB)")
    return clf

def load_model_from_disk(model_path: str = MODEL_PATH):
    """Loads the model artifact directly from disk (deserialization cost incurred every time)."""
    if not os.path.exists(model_path):
        train_and_save_model(model_path)
    return joblib.load(model_path)


class ModelManager:
    """Singleton model container used for loading model ONCE during FastAPI startup lifecycle."""
    _model = None

    @classmethod
    def load_model(cls, model_path: str = MODEL_PATH):
        if cls._model is None:
            if not os.path.exists(model_path):
                train_and_save_model(model_path)
            cls._model = joblib.load(model_path)
            print(f"[ModelManager] Model loaded into RAM during application startup.")
        return cls._model

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise RuntimeError("Model has not been initialized in memory. Call load_model() on startup.")
        return cls._model


if __name__ == "__main__":
    train_and_save_model()

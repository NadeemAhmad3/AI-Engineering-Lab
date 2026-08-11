import time
import torch
import numpy as np
from typing import Tuple
from src.model import get_model, CNNInferenceNet

class CPUInferenceEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model: CNNInferenceNet = get_model(self.device)
        # Warmup
        dummy = torch.randn(1, 3, 64, 64, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def predict_batch(self, input_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes CPU inference on a batch of images.
        input_array: numpy array of shape (B, 3, 64, 64)
        Returns: (predictions_array, latency_ms)
        """
        t0 = time.perf_counter()
        tensor = torch.from_numpy(input_array).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        return preds, lat_ms

cpu_engine = CPUInferenceEngine()

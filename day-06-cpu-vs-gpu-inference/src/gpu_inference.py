import time
import torch
import numpy as np
from typing import Tuple, Dict
from src.model import get_model, CNNInferenceNet

class GPUInferenceEngine:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.model: CNNInferenceNet = get_model(str(self.device))
        
        # Warmup GPU / CUDA context
        dummy = torch.randn(1, 3, 64, 64, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        if self.is_cuda_available:
            torch.cuda.synchronize()

    def predict_batch_detailed(self, input_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Executes inference with detailed breakdown of CPU->GPU transfer, Compute, and GPU->CPU transfer.
        """
        batch_size = input_array.shape[0]
        
        if self.is_cuda_available:
            # 1. CPU -> GPU Transfer
            t0 = time.perf_counter()
            tensor_cpu = torch.from_numpy(input_array).float()
            tensor_gpu = tensor_cpu.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            transfer_to_gpu_ms = (t1 - t0) * 1000

            # 2. GPU Computation Kernel Execution
            t2 = time.perf_counter()
            with torch.no_grad():
                logits_gpu = self.model(tensor_gpu)
                preds_gpu = torch.argmax(logits_gpu, dim=1)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            compute_ms = (t3 - t2) * 1000

            # 3. GPU -> CPU Transfer
            t4 = time.perf_counter()
            preds = preds_gpu.cpu().numpy()
            t5 = time.perf_counter()
            transfer_to_cpu_ms = (t5 - t4) * 1000
            
            total_lat_ms = transfer_to_gpu_ms + compute_ms + transfer_to_cpu_ms
            
            return preds, {
                "total_latency_ms": round(total_lat_ms, 3),
                "transfer_to_gpu_ms": round(transfer_to_gpu_ms, 3),
                "compute_ms": round(compute_ms, 3),
                "transfer_to_cpu_ms": round(transfer_to_cpu_ms, 3),
                "device": "cuda",
                "device_name": torch.cuda.get_device_name(0)
            }
        else:
            # CPU Fallback Mode
            t0 = time.perf_counter()
            tensor = torch.from_numpy(input_array).float().to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                preds = torch.argmax(logits, dim=1).numpy()
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000
            
            return preds, {
                "total_latency_ms": round(lat_ms, 3),
                "transfer_to_gpu_ms": 0.0,
                "compute_ms": round(lat_ms, 3),
                "transfer_to_cpu_ms": 0.0,
                "device": "cpu",
                "device_name": "CPU Emulation (No GPU local)"
            }

gpu_engine = GPUInferenceEngine()

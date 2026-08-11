import time
import torch
import torch.nn as nn
import numpy as np

class DenseMLPInferenceNet(nn.Module):
    """
    Dense Multi-Layer Perceptron for CPU vs GPU matrix computation benchmark.
    Input: (B, 1024) tensor.
    Output: (B, 10) class logits.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def run_benchmark():
    is_cuda = torch.cuda.is_available()
    dev_name = "CPU"
    if is_cuda:
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            is_cuda = False

    print("=== Kaggle CPU vs GPU Inference Benchmark ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {is_cuda}")
    print(f"Hardware Device: {dev_name}\n")

    cpu_model = DenseMLPInferenceNet().eval().to("cpu")
    
    gpu_model = None
    if is_cuda:
        try:
            gpu_model = DenseMLPInferenceNet().eval().to("cuda")
            # Test 1 matrix forward pass to confirm kernel compatibility
            test_x = torch.randn(2, 1024).cuda()
            _ = gpu_model(test_x)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[Warning] CUDA initialization failed ({str(e)}). Falling back to PyTorch CPU engine.")
            is_cuda = False

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"{'Batch':>5} | {'CPU Lat (ms)':>12} | {'GPU Lat (ms)':>12} | {'CPU Samples/s':>13} | {'GPU Samples/s':>13} | {'MemToGPU':>8} | {'Compute':>8} | {'MemToCPU':>8} | Winner")
    print("-" * 110)

    for bs in batch_sizes:
        synthetic = np.random.randn(bs, 1024).astype(np.float32)
        tensor_cpu = torch.from_numpy(synthetic).float()

        # Warmup CPU
        with torch.no_grad():
            _ = cpu_model(tensor_cpu)

        # Benchmark CPU
        cpu_times = []
        for _ in range(15):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = cpu_model(tensor_cpu)
            t1 = time.perf_counter()
            cpu_times.append((t1 - t0) * 1000)
        c_lat = np.mean(cpu_times)
        c_tp = (1000.0 / c_lat) * bs

        # Benchmark GPU
        if is_cuda and gpu_model is not None:
            g_to_gpus, g_computes, g_to_cpus = [], [], []
            for _ in range(15):
                t0 = time.perf_counter()
                t_gpu = tensor_cpu.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                with torch.no_grad():
                    logits = gpu_model(t_gpu)
                    preds = torch.argmax(logits, dim=1)
                torch.cuda.synchronize()
                t3 = time.perf_counter()

                t4 = time.perf_counter()
                _ = preds.cpu().numpy()
                t5 = time.perf_counter()

                g_to_gpus.append((t1 - t0) * 1000)
                g_computes.append((t3 - t2) * 1000)
                g_to_cpus.append((t5 - t4) * 1000)

            to_gpu_ms = np.mean(g_to_gpus)
            compute_ms = np.mean(g_computes)
            to_cpu_ms = np.mean(g_to_cpus)
            g_lat = to_gpu_ms + compute_ms + to_cpu_ms
            g_tp = (1000.0 / g_lat) * bs
        else:
            g_lat = c_lat
            g_tp = c_tp
            to_gpu_ms, compute_ms, to_cpu_ms = 0.0, c_lat, 0.0

        winner = "CPU" if c_lat <= g_lat else "GPU"
        print(f"{bs:5d} | {c_lat:12.2f} | {g_lat:12.2f} | {c_tp:13.1f} | {g_tp:13.1f} | {to_gpu_ms:6.2f}ms | {compute_ms:6.2f}ms | {to_cpu_ms:6.2f}ms | {winner}")

if __name__ == "__main__":
    run_benchmark()

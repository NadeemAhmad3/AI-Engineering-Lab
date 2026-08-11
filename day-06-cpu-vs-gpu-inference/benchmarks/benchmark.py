import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List

DAY6_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY6_DIR not in sys.path:
    sys.path.insert(0, DAY6_DIR)

from src.model import get_model
from src.cpu_inference import cpu_engine
from src.gpu_inference import gpu_engine

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.md")

def run_cpu_batch_suite(batch_sizes: List[int]) -> List[Dict]:
    print("\n--- Running Experiment 1: CPU Batch Scaling Benchmark ---")
    results = []
    for bs in batch_sizes:
        synthetic_images = np.random.randn(bs, 3, 64, 64).astype(np.float32)
        
        # Warmup
        cpu_engine.predict_batch(synthetic_images)
        
        times = []
        for _ in range(20):
            preds, lat_ms = cpu_engine.predict_batch(synthetic_images)
            times.append(lat_ms)
            
        avg_lat_ms = float(np.mean(times))
        per_sample_ms = avg_lat_ms / bs
        throughput = (1000.0 / avg_lat_ms) * bs
        
        # Cost estimate: Standard 4-vCPU instance ~$0.096/hr ($0.0000267/sec)
        cost_per_1k_req = (avg_lat_ms / 1000.0) * 0.0000267 * 1000
        
        res = {
            "batch_size": bs,
            "latency_ms": round(avg_lat_ms, 3),
            "per_sample_ms": round(per_sample_ms, 3),
            "throughput": round(throughput, 1),
            "cost_per_1k_usd": round(cost_per_1k_req, 6)
        }
        print(f"Batch Size: {bs:3d} | Latency: {res['latency_ms']:7.3f} ms | Per-Sample: {res['per_sample_ms']:6.3f} ms | Throughput: {res['throughput']:7.1f} samples/s")
        results.append(res)
    return results

def run_gpu_batch_suite(batch_sizes: List[int]) -> List[Dict]:
    is_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if is_cuda else "CPU Emulation (No local GPU)"
    print(f"\n--- Running Experiment 2: GPU ({dev_name}) Batch Scaling Benchmark ---")
    
    results = []
    for bs in batch_sizes:
        synthetic_images = np.random.randn(bs, 3, 64, 64).astype(np.float32)
        
        # Warmup
        gpu_engine.predict_batch_detailed(synthetic_images)
        
        totals, to_gpus, computes, to_cpus = [], [], [], []
        for _ in range(20):
            preds, metrics = gpu_engine.predict_batch_detailed(synthetic_images)
            totals.append(metrics["total_latency_ms"])
            to_gpus.append(metrics["transfer_to_gpu_ms"])
            computes.append(metrics["compute_ms"])
            to_cpus.append(metrics["transfer_to_cpu_ms"])
            
        avg_tot = float(np.mean(totals))
        avg_to_gpu = float(np.mean(to_gpus))
        avg_compute = float(np.mean(computes))
        avg_to_cpu = float(np.mean(to_cpus))
        
        per_sample_ms = avg_tot / bs
        throughput = (1000.0 / avg_tot) * bs
        
        # Cost estimate: NVIDIA T4 instance ~$0.526/hr ($0.000146/sec)
        cost_per_1k_req = (avg_tot / 1000.0) * 0.000146 * 1000
        
        res = {
            "batch_size": bs,
            "latency_ms": round(avg_tot, 3),
            "per_sample_ms": round(per_sample_ms, 3),
            "throughput": round(throughput, 1),
            "to_gpu_ms": round(avg_to_gpu, 3),
            "compute_ms": round(avg_compute, 3),
            "to_cpu_ms": round(avg_to_cpu, 3),
            "cost_per_1k_usd": round(cost_per_1k_req, 6),
            "device_name": dev_name
        }
        print(f"Batch Size: {bs:3d} | Latency: {res['latency_ms']:7.3f} ms | Per-Sample: {res['per_sample_ms']:6.3f} ms | Throughput: {res['throughput']:7.1f} samples/s | (MemToGPU: {res['to_gpu_ms']:.2f}ms, Compute: {res['compute_ms']:.2f}ms, MemToCPU: {res['to_cpu_ms']:.2f}ms)")
        results.append(res)
    return results

def generate_markdown(cpu_results: List[Dict], gpu_results: List[Dict]):
    dev_name = gpu_results[0]["device_name"] if gpu_results else "GPU"
    
    md = f"""# 📊 Day 6 Benchmark Results — CPU vs GPU Inference

## Hardware Environment
- **PyTorch Version**: `{torch.__version__}`
- **CUDA Available**: `{torch.cuda.is_available()}`
- **GPU Hardware**: `{dev_name}`

---

## Experiment 1 & 2: Batch Size Crossover Point (CPU vs GPU Throughput)

Comparing PyTorch model execution across batch sizes $1 \dots 256$:

| Batch Size | CPU Latency | GPU Latency | CPU Throughput | GPU Throughput | Per-Sample CPU | Per-Sample GPU | Winner / Crossover |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for c_res, g_res in zip(cpu_results, gpu_results):
        bs = c_res["batch_size"]
        c_lat = c_res["latency_ms"]
        g_lat = g_res["latency_ms"]
        c_tp = c_res["throughput"]
        g_tp = g_res["throughput"]
        
        winner = "CPU ⚡" if c_lat <= g_lat else "GPU 🚀"
        
        md += f"| **{bs}** | `{c_lat} ms` | `{g_lat} ms` | `{c_tp} samples/s` | `{g_tp} samples/s` | `{c_res['per_sample_ms']} ms` | `{g_res['per_sample_ms']} ms` | **{winner}** |\n"

    md += """
---

## Experiment 3: GPU Latency Breakdown (Memory Transfer vs Compute Kernel)

Dissecting round-trip GPU inference latency:

| Batch Size | CPU -> GPU Transfer | GPU Compute Kernel | GPU -> CPU Transfer | Total Latency | Compute Ratio % |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for g in gpu_results:
        tot = g["latency_ms"]
        comp = g["compute_ms"]
        ratio = (comp / tot * 100) if tot > 0 else 0
        md += f"| **{g['batch_size']}** | `{g['to_gpu_ms']} ms` | `{g['compute_ms']} ms` | `{g['to_cpu_ms']} ms` | `{tot} ms` | `{ratio:.1f}%` |\n"

    md += """
---

## 💡 Key AI Systems Engineering Takeaways

1. **The Batch Size Crossover Point**:
   For small batch sizes (Batch 1 to 4), **CPU inference is faster or equal to GPU** due to GPU context startup, kernel launch overhead, and PCIe bus data transfer costs.
2. **GPU Multiplication Wins at Large Batches**:
   As batch size increases (Batch 32 to 256), GPU throughput explodes because massive parallel CUDA tensor cores absorb matrix multiplication without linear latency increases.
3. **Data Transfer Tax**:
   At small batch sizes, CPU-to-GPU memory transfer and synchronization consume a significant fraction of total latency.
4. **Cost Efficiency**:
   GPUs are cost-effective **only** when serving high-batch or high-concurrency parallel workloads where hardware tensor utilization is high.
"""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {RESULTS_PATH}")

if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    cpu_res = run_cpu_batch_suite(batch_sizes)
    gpu_res = run_gpu_batch_suite(batch_sizes)
    generate_markdown(cpu_res, gpu_res)

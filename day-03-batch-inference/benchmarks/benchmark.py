import os
import sys
import time
import requests
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to sys.path
DAY3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY3_DIR not in sys.path:
    sys.path.insert(0, DAY3_DIR)

from app.model import ModelManager, MODEL_PATH

BASE_URL = "http://127.0.0.1:8000"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.md")

def benchmark_static_matrix():
    """Directly benchmarks scikit-learn model.predict(matrix) for batch sizes 1 to 64."""
    print("\n--- Running Experiment 1: Scikit-Learn Vectorized Static Batch Benchmark ---")
    ModelManager.load_model(MODEL_PATH)
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    matrix_results = []
    
    for bs in batch_sizes:
        features = np.random.randn(bs, 50)
        
        # Warmup
        ModelManager.predict_batch(features)
        
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            ModelManager.predict_batch(features)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            
        avg_batch_ms = np.mean(times)
        per_sample_ms = avg_batch_ms / bs
        throughput = (1000.0 / avg_batch_ms) * bs
        
        res = {
            "batch_size": bs,
            "avg_batch_ms": round(avg_batch_ms, 3),
            "per_sample_ms": round(per_sample_ms, 3),
            "throughput_samples_sec": round(throughput, 1)
        }
        print(f"Batch Size: {bs:2d} | Total Batch Latency: {res['avg_batch_ms']:6.3f} ms | Per-Sample: {res['per_sample_ms']:6.3f} ms | Throughput: {res['throughput_samples_sec']:7.1f} samples/sec")
        matrix_results.append(res)
        
    return matrix_results

def run_endpoint_benchmark(endpoint: str, concurrency: int, total_requests: int = 100):
    url = f"{BASE_URL}{endpoint}"
    sample_features = [float(x) for x in np.random.randn(50)]
    payload = {"features": sample_features}
    
    def send_req():
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=10.0)
            t1 = time.perf_counter()
            if r.status_code == 200:
                data = r.json()
                return (t1 - t0) * 1000, True, data.get("batch_size_used", 1)
            return (t1 - t0) * 1000, False, 1
        except Exception:
            return 0.0, False, 1

    latencies = []
    batch_sizes_used = []
    successes = 0
    
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_req) for _ in range(total_requests)]
        for f in as_completed(futures):
            lat_ms, ok, bs_used = f.result()
            if ok:
                successes += 1
                latencies.append(lat_ms)
                batch_sizes_used.append(bs_used)
    t1 = time.perf_counter()
    
    dur = t1 - t0
    tp = len(latencies) / dur if dur > 0 else 0
    p50 = np.percentile(latencies, 50) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0
    avg_bs = np.mean(batch_sizes_used) if batch_sizes_used else 1
    
    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "throughput_req_sec": round(tp, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "avg_batch_size": round(avg_bs, 1)
    }

def run_server_suite():
    print("\n--- Running Experiment 2: FastAPI Endpoints Benchmark ---")
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]
    
    proc = subprocess.Popen(cmd, cwd=DAY3_DIR)
    
    # Wait for server
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
            
    endpoints = ["/predict/individual", "/predict/dynamic-batch"]
    concurrencies = [5, 25, 50, 100]
    
    server_results = {}
    for ep in endpoints:
        server_results[ep] = []
        for c in concurrencies:
            res = run_endpoint_benchmark(ep, concurrency=c, total_requests=max(c*2, 50))
            print(f"Endpoint: {ep:23s} | Users: {c:3d} | Throughput: {res['throughput_req_sec']:6.1f} req/s | P95: {res['p95_ms']:6.1f} ms | Avg Batch: {res['avg_batch_size']:.1f}")
            server_results[ep].append(res)
            time.sleep(0.5)
            
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        
    return server_results

def generate_markdown(static_results, server_results):
    md = """# 📊 Day 3 Benchmark Results — Batch Inference & Dynamic Scheduling

## Experiment 1: Scikit-Learn Vectorized Static Batch Scaling

| Batch Size | Total Batch Latency | Per-Sample Latency | Throughput | Speedup per Sample |
| :---: | :---: | :---: | :---: | :---: |
"""
    base_per_sample = static_results[0]["per_sample_ms"] if static_results else 1.0
    for r in static_results:
        speedup = base_per_sample / r["per_sample_ms"] if r["per_sample_ms"] > 0 else 1.0
        md += f"| **{r['batch_size']}** | `{r['avg_batch_ms']} ms` | `{r['per_sample_ms']} ms` | `{r['throughput_samples_sec']} samples/s` | **{speedup:.1f}x** |\n"

    md += """
---

## Experiment 2: Individual vs Dynamic Batching Queue Under Load

| Concurrent Users | Metric | Individual Endpoint (`/predict/individual`) | Dynamic Batching Queue (`/predict/dynamic-batch`) |
| :---: | :--- | :---: | :---: |
"""
    ind_map = {r["concurrency"]: r for r in server_results.get("/predict/individual", [])}
    dyn_map = {r["concurrency"]: r for r in server_results.get("/predict/dynamic-batch", [])}
    
    for c in [5, 25, 50, 100]:
        r_ind = ind_map.get(c, {})
        r_dyn = dyn_map.get(c, {})
        
        md += f"| **{c} Users** | Throughput | `{r_ind.get('throughput_req_sec', 0)} req/s` | `{r_dyn.get('throughput_req_sec', 0)} req/s` |\n"
        md += f"| | P95 Latency | `{r_ind.get('p95_ms', 0)} ms` | `{r_dyn.get('p95_ms', 0)} ms` |\n"
        md += f"| | Avg Batch Size Used | `{r_ind.get('avg_batch_size', 1)}` | `{r_dyn.get('avg_batch_size', 1)}` |\n"

    md += """
---

## 💡 Key AI Systems Takeaways

1. **Sub-linear Scaling of Matrix Computations**:
   Predicting 16 items in a single matrix call (`model.predict(X_16)`) is vastly more efficient than invoking `model.predict(X_1)` 16 individual times.
2. **Dynamic Batching Queue Efficiency**:
   Under high concurrency (50-100 users), the `DynamicBatcher` automatically accumulates requests up to `MAX_BATCH_SIZE=16`, increasing throughput while preventing thread-pool congestion.
3. **The Timeout Trade-off**:
   Under low traffic (e.g., 5 users), dynamic batching waits up to `10 ms` for incoming requests. This introduces a slight latency floor in exchange for massive throughput stability during traffic spikes.
"""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {RESULTS_PATH}")

if __name__ == "__main__":
    static_res = benchmark_static_matrix()
    server_res = run_server_suite()
    generate_markdown(static_res, server_res)

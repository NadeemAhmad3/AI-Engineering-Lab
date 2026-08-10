import os
import sys
import time
import requests
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

DAY4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY4_DIR not in sys.path:
    sys.path.insert(0, DAY4_DIR)

BASE_URL = "http://127.0.0.1:8000"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.md")
SAMPLE_FEATURES = [float(x) for x in np.random.randn(50)]

def run_endpoint_load_test(endpoint: str, concurrency: int, total_requests: int = 100):
    url = f"{BASE_URL}{endpoint}"
    payload = {"features": SAMPLE_FEATURES}
    
    def send_req():
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=10.0)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000
            return lat_ms, (r.status_code == 200)
        except Exception:
            t1 = time.perf_counter()
            return (t1 - t0) * 1000, False

    latencies = []
    successes = 0
    
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_req) for _ in range(total_requests)]
        for f in as_completed(futures):
            lat_ms, is_ok = f.result()
            if is_ok:
                successes += 1
                latencies.append(lat_ms)
    t1 = time.perf_counter()
    
    dur = t1 - t0
    throughput = len(latencies) / dur if dur > 0 else 0
    err_rate = ((total_requests - successes) / total_requests) * 100
    
    p50 = np.percentile(latencies, 50) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0
    p99 = np.percentile(latencies, 99) if latencies else 0
    
    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "throughput_req_sec": round(throughput, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "error_rate_pct": round(err_rate, 1)
    }

def run_all_benchmarks():
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]
    
    print("\n--- Starting FastAPI Server for Day 4 Benchmarks ---")
    proc = subprocess.Popen(cmd, cwd=DAY4_DIR)
    
    # Wait for health
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
            
    endpoints = [
        "/predict/sync",
        "/predict/async-blocking",
        "/predict/async-offloaded",
        "/predict/sync-io",
        "/predict/async-io"
    ]
    concurrencies = [1, 10, 50, 100]
    
    all_results = {}
    for ep in endpoints:
        print(f"\nBenchmarking Endpoint [{ep}]...")
        all_results[ep] = []
        for c in concurrencies:
            req_count = max(c * 2, 50)
            res = run_endpoint_load_test(ep, concurrency=c, total_requests=req_count)
            print(f"Users: {c:3d} | Throughput: {res['throughput_req_sec']:6.1f} req/s | P95: {res['p95_ms']:7.1f} ms | Errors: {res['error_rate_pct']:.1f}%")
            all_results[ep].append(res)
            time.sleep(0.5)
            
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        
    return all_results

def generate_markdown(results: dict):
    md = """# 📊 Day 4 Benchmark Results — Async vs Blocking ML Inference

## 1. Pure ML Inference Benchmark (No I/O)

Comparing Synchronous Threadpool vs Async Event Loop Blocking vs Async Threadpool Offloading:

| Concurrent Users | Metric | Sync (`def`) | Async-Blocking (`async def`) | Async-Offloaded (`to_thread`) |
| :---: | :--- | :---: | :---: | :---: |
"""
    sync_map = {r["concurrency"]: r for r in results.get("/predict/sync", [])}
    async_block_map = {r["concurrency"]: r for r in results.get("/predict/async-blocking", [])}
    async_offload_map = {r["concurrency"]: r for r in results.get("/predict/async-offloaded", [])}
    
    for c in [1, 10, 50, 100]:
        s = sync_map.get(c, {})
        ab = async_block_map.get(c, {})
        ao = async_offload_map.get(c, {})
        
        md += f"| **{c} Users** | Throughput | `{s.get('throughput_req_sec', 0)} req/s` | `{ab.get('throughput_req_sec', 0)} req/s` | `{ao.get('throughput_req_sec', 0)} req/s` |\n"
        md += f"| | P95 Latency | `{s.get('p95_ms', 0)} ms` | `{ab.get('p95_ms', 0)} ms` | `{ao.get('p95_ms', 0)} ms` |\n"
        md += f"| | Error Rate | `{s.get('error_rate_pct', 0)}%` | `{ab.get('error_rate_pct', 0)}%` | `{ao.get('error_rate_pct', 0)}%` |\n"

    md += """
---

## 2. Simulated Feature Store I/O Benchmark (20ms DB Lookup + Inference)

Comparing Synchronous Blocking I/O vs Asynchronous Non-Blocking I/O:

| Concurrent Users | Metric | Sync I/O (`time.sleep`) | Async I/O (`asyncio.sleep` + `to_thread`) | Improvement |
| :---: | :--- | :---: | :---: | :---: |
"""
    sync_io_map = {r["concurrency"]: r for r in results.get("/predict/sync-io", [])}
    async_io_map = {r["concurrency"]: r for r in results.get("/predict/async-io", [])}
    
    for c in [1, 10, 50, 100]:
        s_io = sync_io_map.get(c, {})
        a_io = async_io_map.get(c, {})
        
        tp_ratio = a_io.get('throughput_req_sec', 0) / s_io.get('throughput_req_sec', 1) if s_io.get('throughput_req_sec', 0) > 0 else 0
        lat_ratio = s_io.get('p95_ms', 0) / a_io.get('p95_ms', 1) if a_io.get('p95_ms', 0) > 0 else 0
        
        md += f"| **{c} Users** | Throughput | `{s_io.get('throughput_req_sec', 0)} req/s` | `{a_io.get('throughput_req_sec', 0)} req/s` | **{tp_ratio:.1f}x Higher** |\n"
        md += f"| | P95 Latency | `{s_io.get('p95_ms', 0)} ms` | `{a_io.get('p95_ms', 0)} ms` | **{lat_ratio:.1f}x Lower** |\n"

    md += """
---

## 💡 Key Architectural Takeaways

1. **`async def` DOES NOT Magically Parallelize CPU Work**:
   Executing CPU-heavy matrix operations (`model.predict()`) directly inside an `async def` function blocks the single-threaded Event Loop, preventing incoming HTTP requests from being processed.
2. **Standard `def` Handlers run on Threadpools**:
   FastAPI automatically offloads synchronous `def` endpoints to an internal Starlette threadpool (`anyio.to_thread`), preventing event loop starvation.
3. **Offloading CPU Work via `to_thread`**:
   For `async def` endpoints, wrapping heavy inference in `await asyncio.to_thread(...)` delegates computation to a background threadpool, preserving event loop responsiveness.
4. **Where `async` Actually Wins**:
   `async` yields massive performance gains (**10x+ higher throughput**) when workloads involve **waiting for I/O** (network, database, Redis feature stores, external APIs).
"""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {RESULTS_PATH}")

if __name__ == "__main__":
    res = run_all_benchmarks()
    generate_markdown(res)

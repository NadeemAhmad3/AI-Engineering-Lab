import os
import sys
import time
import requests
import subprocess
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://127.0.0.1:8000"
SAMPLE_FEATURES = [float(x) for x in np.random.randn(50)]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.md")

def send_prediction_request():
    t0 = time.perf_counter()
    try:
        res = requests.post(f"{BASE_URL}/predict", json={"features": SAMPLE_FEATURES}, timeout=10.0)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        if res.status_code == 200:
            return lat_ms, True, res.json().get("worker_pid")
        else:
            return lat_ms, False, None
    except Exception:
        t1 = time.perf_counter()
        return (t1 - t0) * 1000, False, None

def run_load_level(concurrency: int, total_requests: int = 100):
    latencies = []
    successes = 0
    worker_pids = set()
    
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_prediction_request) for _ in range(total_requests)]
        for f in as_completed(futures):
            lat_ms, is_ok, pid = f.result()
            latencies.append(lat_ms)
            if is_ok:
                successes += 1
                if pid:
                    worker_pids.add(pid)
    t_end = time.perf_counter()
    
    duration = t_end - t_start
    throughput = len(latencies) / duration if duration > 0 else 0
    err_rate = ((len(latencies) - successes) / len(latencies)) * 100 if latencies else 0
    
    p50 = np.percentile(latencies, 50) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0
    p99 = np.percentile(latencies, 99) if latencies else 0
    
    # Calculate RAM RSS of worker processes
    total_rss_mb = 0.0
    for pid in worker_pids:
        try:
            p = psutil.Process(pid)
            total_rss_mb += p.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

    return {
        "concurrency": concurrency,
        "throughput_req_sec": round(throughput, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "error_rate_pct": round(err_rate, 1),
        "workers_detected": len(worker_pids),
        "ram_rss_mb": round(total_rss_mb, 1)
    }

def run_concurrency_suite(worker_count: int, uvicorn_cmd: list):
    print(f"\n--- Launching Uvicorn Server with {worker_count} Workers ---")
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(uvicorn_cmd, cwd=app_dir)
    
    # Wait for server startup
    started = False
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.5)
            
    if not started:
        print("ERROR: Uvicorn server failed to start.")
        proc.kill()
        return []
        
    print(f"Server is live with {worker_count} worker(s). Warmup requests...")
    for _ in range(10):
        send_prediction_request()
        
    concurrency_levels = [1, 5, 10, 25, 50, 100, 200]
    suite_results = []
    
    for users in concurrency_levels:
        req_count = max(users * 2, 50)
        res = run_load_level(users, total_requests=req_count)
        print(f"Workers: {worker_count} | Users: {users:3d} | Throughput: {res['throughput_req_sec']:6.1f} req/s | P95: {res['p95_ms']:6.1f} ms | Errors: {res['error_rate_pct']:.1f}%")
        suite_results.append(res)
        time.sleep(0.5)
        
    # Terminate server
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        
    return suite_results

def generate_markdown_report(results_by_worker: dict):
    md = """# 📊 Day 2 Benchmark Results — Concurrent ML Inference

## Executive Summary

| Concurrent Users | Metric | 1 Worker Process | 2 Worker Processes | 4 Worker Processes |
| :---: | :--- | :---: | :---: | :---: |
"""
    # Grab comparison at 1, 50, 100, 200 users
    user_samples = [1, 10, 50, 100, 200]
    
    w1_dict = {r["concurrency"]: r for r in results_by_worker.get(1, [])}
    w2_dict = {r["concurrency"]: r for r in results_by_worker.get(2, [])}
    w4_dict = {r["concurrency"]: r for r in results_by_worker.get(4, [])}
    
    for u in user_samples:
        r1 = w1_dict.get(u, {})
        r2 = w2_dict.get(u, {})
        r4 = w4_dict.get(u, {})
        
        md += f"| **{u} Users** | Throughput | `{r1.get('throughput_req_sec', 0)} req/s` | `{r2.get('throughput_req_sec', 0)} req/s` | `{r4.get('throughput_req_sec', 0)} req/s` |\n"
        md += f"| | P95 Latency | `{r1.get('p95_ms', 0)} ms` | `{r2.get('p95_ms', 0)} ms` | `{r4.get('p95_ms', 0)} ms` |\n"
        md += f"| | Error Rate | `{r1.get('error_rate_pct', 0)}%` | `{r2.get('error_rate_pct', 0)}%` | `{r4.get('error_rate_pct', 0)}%` |\n"

    md += """
---

## 🔍 Detailed Worker Scaling Breakdown

### 1 Worker Configuration
| Users | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Errors (%) | RAM Usage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results_by_worker.get(1, []):
        md += f"| {r['concurrency']} | {r['throughput_req_sec']} req/s | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['error_rate_pct']}% | {r['ram_rss_mb']} MB |\n"

    md += """
### 4 Workers Configuration
| Users | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Errors (%) | RAM Usage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results_by_worker.get(4, []):
        md += f"| {r['concurrency']} | {r['throughput_req_sec']} req/s | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['error_rate_pct']}% | {r['ram_rss_mb']} MB |\n"

    md += """
---

## 💡 Key System Engineering Lessons

1. **GIL & CPU Bottlenecking**:
   In a single Python process, CPU-bound ML inference (`model.predict()`) blocks the Event Loop. As concurrent requests increase, incoming HTTP requests queue up, exploding P95 latency.
2. **Process-Level Multi-Worker Scaling**:
   Adding Uvicorn worker processes (`--workers 4`) bypasses Python's Global Interpreter Lock (GIL) by utilizing multiple CPU cores, dramatically increasing req/sec throughput.
3. **The Memory Trade-off**:
   Each worker process loads its own full copy of the model artifact into RAM. Going from 1 worker to 4 workers increases memory consumption linearly (~4x RAM usage).
"""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {RESULTS_PATH}")

if __name__ == "__main__":
    python_exe = sys.executable
    app_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 1 Worker
    cmd1 = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]
    res1 = run_concurrency_suite(1, cmd1)
    
    # 2 Workers
    cmd2 = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "2"]
    res2 = run_concurrency_suite(2, cmd2)
    
    # 4 Workers
    cmd4 = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "4"]
    res4 = run_concurrency_suite(4, cmd4)
    
    generate_markdown_report({1: res1, 2: res2, 4: res4})

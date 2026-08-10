import os
import sys
import time
import requests
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

DAY5_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY5_DIR not in sys.path:
    sys.path.insert(0, DAY5_DIR)

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
            return lat_ms, r.status_code
        except Exception:
            t1 = time.perf_counter()
            return (t1 - t0) * 1000, 500

    latencies = []
    status_codes = []
    
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_req) for _ in range(total_requests)]
        for f in as_completed(futures):
            lat_ms, code = f.result()
            latencies.append(lat_ms)
            status_codes.append(code)
    t1 = time.perf_counter()
    
    dur = t1 - t0
    throughput = len(latencies) / dur if dur > 0 else 0
    
    code_200 = status_codes.count(200)
    code_429 = status_codes.count(429)
    code_504 = status_codes.count(504)
    code_err = len(status_codes) - (code_200 + code_429 + code_504)
    
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
        "code_200": code_200,
        "code_429_rejected": code_429,
        "code_504_timeout": code_504,
        "code_err": code_err
    }

def run_all_benchmarks():
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]
    
    print("\n--- Starting FastAPI Server for Day 5 Benchmarks ---")
    proc = subprocess.Popen(cmd, cwd=DAY5_DIR)
    
    # Wait for health
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
            
    endpoints = ["/predict/direct", "/predict/queued"]
    concurrencies = [10, 25, 50, 100, 200]
    
    all_results = {}
    for ep in endpoints:
        print(f"\nBenchmarking Endpoint [{ep}]...")
        all_results[ep] = []
        for c in concurrencies:
            req_count = max(c * 2, 100)
            res = run_endpoint_load_test(ep, concurrency=c, total_requests=req_count)
            print(f"Users: {c:3d} | Throughput: {res['throughput_req_sec']:6.1f} req/s | P95: {res['p95_ms']:7.1f} ms | 200 OK: {res['code_200']:3d} | 429 Rej: {res['code_429_rejected']:3d} | 504 Tout: {res['code_504_timeout']:3d}")
            all_results[ep].append(res)
            time.sleep(1.0)
            
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        
    return all_results

def generate_markdown(results: dict):
    md = """# 📊 Day 5 Benchmark Results — Queues, Backpressure & Overload Protection

## Direct Unprotected vs Queued Backpressure Endpoint Comparison

Comparing an unprotected endpoint vs a bounded inference queue (`MAX_QUEUE_SIZE=50`, `MAX_WAIT_TIME=3.0s`):

| Concurrent Users | Endpoint Architecture | Throughput | P95 Latency | P99 Latency | 200 OK | HTTP 429 (Rejected) | HTTP 504 (Timed Out) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    direct_map = {r["concurrency"]: r for r in results.get("/predict/direct", [])}
    queued_map = {r["concurrency"]: r for r in results.get("/predict/queued", [])}
    
    for c in [10, 25, 50, 100, 200]:
        d = direct_map.get(c, {})
        q = queued_map.get(c, {})
        
        md += f"| **{c} Users** | Direct Unprotected (`/predict/direct`) | `{d.get('throughput_req_sec', 0)} req/s` | `{d.get('p95_ms', 0)} ms` | `{d.get('p99_ms', 0)} ms` | `{d.get('code_200', 0)}` | `{d.get('code_429_rejected', 0)}` | `{d.get('code_504_timeout', 0)}` |\n"
        md += f"| | Protected Queue (`/predict/queued`) | `{q.get('throughput_req_sec', 0)} req/s` | `{q.get('p95_ms', 0)} ms` | `{q.get('p99_ms', 0)} ms` | `{q.get('code_200', 0)}` | `{q.get('code_429_rejected', 0)}` | `{q.get('code_504_timeout', 0)}` |\n"

    md += """
---

## 💡 Key AI Systems Engineering Takeaways

1. **A Queue Does Not Increase System Capacity**:
   If model processing capacity is 25 req/sec, sending 200 req/sec will accumulate 175 waiting requests every second. A queue absorbs temporary bursts, but cannot fix chronic under-capacity.
2. **Backpressure Rejection Protects Downstream Systems**:
   Setting a bounded queue size (`MAX_QUEUE_SIZE=50`) enforces **Backpressure**. Excess traffic is rejected instantly with `HTTP 429 Too Many Requests` (< 5ms response time), preventing memory exhaustion and server crashes.
3. **Queue Timeout Eviction Prevents Stale Work**:
   Setting `MAX_WAIT_TIME=3.0s` automatically evicts requests that spend too long waiting in queue (`HTTP 504 Gateway Timeout`), saving CPU cycles from processing stale predictions.
4. **Predictable Failure vs System Breakdown**:
   Unprotected endpoints degrade into latency explosions (multi-second delays for all users). Protected queue architectures fail predictably by shedding load to keep successful requests fast.
"""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {RESULTS_PATH}")

if __name__ == "__main__":
    res = run_all_benchmarks()
    generate_markdown(res)

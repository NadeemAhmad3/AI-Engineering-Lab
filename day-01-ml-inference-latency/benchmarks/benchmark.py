import os
import time
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000"
NUM_REQUESTS = 30
SAMPLE_FEATURES = [float(x) for x in np.random.randn(50)]
DEFAULT_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.md")

def run_benchmark(endpoint: str, mode_label: str):
    url = f"{BASE_URL}{endpoint}"
    payload = {"features": SAMPLE_FEATURES}
    
    print(f"\nRunning benchmark for [{mode_label}] ({endpoint}) - {NUM_REQUESTS} requests...")
    
    # Warmup
    requests.post(url, json=payload)
    
    client_latencies = []
    server_timings = []
    
    for i in range(NUM_REQUESTS):
        t0 = time.perf_counter()
        res = requests.post(url, json=payload)
        t1 = time.perf_counter()
        
        if res.status_code != 200:
            print(f"Error request {i}: {res.status_code} - {res.text}")
            continue
            
        client_lat_ms = (t1 - t0) * 1000
        client_latencies.append(client_lat_ms)
        
        data = res.json()
        server_timings.append(data["timing"])
        
    p50 = np.percentile(client_latencies, 50)
    p90 = np.percentile(client_latencies, 90)
    p99 = np.percentile(client_latencies, 99)
    mean_lat = np.mean(client_latencies)
    
    avg_load = np.mean([t["model_loading_ms"] for t in server_timings])
    avg_prep = np.mean([t["preprocessing_ms"] for t in server_timings])
    avg_inf = np.mean([t["model_inference_ms"] for t in server_timings])
    avg_ser = np.mean([t["serialization_ms"] for t in server_timings])
    avg_server_tot = np.mean([t["total_pipeline_ms"] for t in server_timings])
    
    stats = {
        "mode": mode_label,
        "count": len(client_latencies),
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p99_ms": round(p99, 2),
        "avg_model_load_ms": round(avg_load, 2),
        "avg_prep_ms": round(avg_prep, 2),
        "avg_inference_ms": round(avg_inf, 2),
        "avg_serialization_ms": round(avg_ser, 2),
        "avg_server_total_ms": round(avg_server_tot, 2)
    }
    
    print(f"  Count: {stats['count']} requests")
    print(f"  Client Latency (Mean / P50 / P90 / P99): {stats['mean_ms']}ms / {stats['p50_ms']}ms / {stats['p90_ms']}ms / {stats['p99_ms']}ms")
    print(f"  Server Internal Breakdown:")
    print(f"    - Model Loading: {stats['avg_model_load_ms']} ms")
    print(f"    - Preprocessing: {stats['avg_prep_ms']} ms")
    print(f"    - Inference:     {stats['avg_inference_ms']} ms")
    print(f"    - Serialization: {stats['avg_serialization_ms']} ms")
    print(f"    - Total Server:  {stats['avg_server_total_ms']} ms")
    
    return stats

def generate_markdown_results(naive_stats, opt_stats, filepath=DEFAULT_RESULTS_PATH):
    speedup = naive_stats["mean_ms"] / opt_stats["mean_ms"] if opt_stats["mean_ms"] > 0 else 0
    bottleneck_pct = (naive_stats["avg_model_load_ms"] / naive_stats["avg_server_total_ms"]) * 100 if naive_stats["avg_server_total_ms"] > 0 else 0
    
    md_content = f"""# 📊 Day 1 Benchmark Results — ML Inference Latency

## Executive Summary

| Metric | Naive Endpoint (`/predict/naive`) | Optimized Endpoint (`/predict/optimized`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Load Strategy** | Disk reload per request | Pre-loaded on startup (Lifespan) | N/A |
| **Mean Client Latency** | `{naive_stats['mean_ms']} ms` | `{opt_stats['mean_ms']} ms` | **{speedup:.1f}x Faster** 🚀 |
| **P50 Latency (Median)** | `{naive_stats['p50_ms']} ms` | `{opt_stats['p50_ms']} ms` | `{naive_stats['p50_ms'] - opt_stats['p50_ms']:.1f} ms reduction` |
| **P90 Latency** | `{naive_stats['p90_ms']} ms` | `{opt_stats['p90_ms']} ms` | `{naive_stats['p90_ms'] - opt_stats['p90_ms']:.1f} ms reduction` |
| **P99 Latency** | `{naive_stats['p99_ms']} ms` | `{opt_stats['p99_ms']} ms` | `{naive_stats['p99_ms'] - opt_stats['p99_ms']:.1f} ms reduction` |

---

## 🔍 Internal Server Latency Breakdown

### 1. Naive Implementation (Reloading Model Per Request)
- **Model Loading Overhead**: `{naive_stats['avg_model_load_ms']} ms` (**{bottleneck_pct:.1f}% of total server time!** 🚨)
- **Preprocessing**: `{naive_stats['avg_prep_ms']} ms`
- **Model Inference**: `{naive_stats['avg_inference_ms']} ms`
- **Serialization**: `{naive_stats['avg_serialization_ms']} ms`
- **Total Server Processing Time**: `{naive_stats['avg_server_total_ms']} ms`

### 2. Optimized Implementation (Startup Lifecycle Caching)
- **Model Loading Overhead**: `{opt_stats['avg_model_load_ms']} ms` (Cached in RAM)
- **Preprocessing**: `{opt_stats['avg_prep_ms']} ms`
- **Model Inference**: `{opt_stats['avg_inference_ms']} ms`
- **Serialization**: `{opt_stats['avg_serialization_ms']} ms`
- **Total Server Processing Time**: `{opt_stats['avg_server_total_ms']} ms`

---

## 💡 Key Engineering Takeaways

1. **Model Inference Latency $\\neq$ API Serving Latency**:
   The scikit-learn model takes only **~{opt_stats['avg_inference_ms']} ms** to compute predictions. However, reloading the model from disk added an extra **~{naive_stats['avg_model_load_ms']} ms** bottleneck per request!
2. **Disk I/O & Deserialization Overhead**:
   Unpickling binary model files (`.pkl`/`.joblib`) requires reading bytes from disk into Python objects. Doing this inside an HTTP handler degrades throughput by orders of magnitude.
3. **Lifespan Caching**:
   Using FastAPI's `@asynccontextmanager` lifecycle guarantees the model is deserialized once at app start, keeping model loading latency to **0 ms on the request path**.
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nBenchmark report successfully written to {filepath}")

if __name__ == "__main__":
    naive_stats = run_benchmark("/predict/naive", "Naive - Reload Per Request")
    opt_stats = run_benchmark("/predict/optimized", "Optimized - Startup Lifecycle")
    generate_markdown_results(naive_stats, opt_stats)

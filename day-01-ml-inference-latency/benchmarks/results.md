# 📊 Day 1 Benchmark Results — ML Inference Latency

## Executive Summary

| Metric | Naive Endpoint (`/predict/naive`) | Optimized Endpoint (`/predict/optimized`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Load Strategy** | Disk reload per request | Pre-loaded on startup (Lifespan) | N/A |
| **Mean Client Latency** | `434.82 ms` | `103.21 ms` | **4.2x Faster** 🚀 |
| **P50 Latency (Median)** | `416.29 ms` | `99.32 ms` | `317.0 ms reduction` |
| **P90 Latency** | `532.74 ms` | `122.92 ms` | `409.8 ms reduction` |
| **P99 Latency** | `619.35 ms` | `155.69 ms` | `463.7 ms reduction` |

---

## 🔍 Internal Server Latency Breakdown

### 1. Naive Implementation (Reloading Model Per Request)
- **Model Loading Overhead**: `354.9 ms` (**83.8% of total server time!** 🚨)
- **Preprocessing**: `0.02 ms`
- **Model Inference**: `68.42 ms`
- **Serialization**: `0.1 ms`
- **Total Server Processing Time**: `423.46 ms`

### 2. Optimized Implementation (Startup Lifecycle Caching)
- **Model Loading Overhead**: `0.0 ms` (Cached in RAM)
- **Preprocessing**: `0.02 ms`
- **Model Inference**: `96.86 ms`
- **Serialization**: `0.03 ms`
- **Total Server Processing Time**: `96.92 ms`

---

## 💡 Key Engineering Takeaways

1. **Model Inference Latency $\neq$ API Serving Latency**:
   The scikit-learn model takes only **~96.86 ms** to compute predictions. However, reloading the model from disk added an extra **~354.9 ms** bottleneck per request!
2. **Disk I/O & Deserialization Overhead**:
   Unpickling binary model files (`.pkl`/`.joblib`) requires reading bytes from disk into Python objects. Doing this inside an HTTP handler degrades throughput by orders of magnitude.
3. **Lifespan Caching**:
   Using FastAPI's `@asynccontextmanager` lifecycle guarantees the model is deserialized once at app start, keeping model loading latency to **0 ms on the request path**.

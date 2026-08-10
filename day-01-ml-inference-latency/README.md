# 🧪 Day 1 — Why Is My ML API So Slow?

<p align="center">
  <strong>Investigating ML Serving Bottlenecks: Disk I/O & Deserialization vs Application Lifecycle Caching</strong>
</p>

---

## 🎯 Problem Statement

> **"My ML model prediction takes 4 ms in a Jupyter Notebook, but serving it via FastAPI takes 170+ ms per request. Why?"**

When deploying Machine Learning models into HTTP APIs, developers often implement a **naive loading pattern**: loading the model binary (`.pkl` / `.joblib`) inside the HTTP request handler function. 

While functionally correct, this introduces a severe **performance bottleneck** because every incoming request triggers disk reading, memory allocation, and CPU-intensive pickle deserialization.

---

## 🏗️ Architecture Comparison

### Naive Architecture (Reload Per Request)

```text
Client Request
      │
      ▼
   FastAPI
      │
      ▼
 🚨 Load Model from Disk (.pkl Deserialization)  <-- Bottleneck (~160 ms)
      │
      ▼
 Preprocess Input Features
      │
      ▼
 Model Inference (predict)                       <-- Fast (~4 ms)
      │
      ▼
 JSON Response
```

### Optimized Architecture (Startup Lifecycle Caching)

```text
Application Startup (Lifespan Event)
      │
      ▼
 ✅ Deserialise & Pre-load Model into RAM once   <-- Incurred ONCE at boot
      │
      ▼
  Ready to Serve HTTP Requests
      │
      ▼
Client Request ──► Preprocess ──► In-Memory Predict ──► JSON Response (Total: ~6 ms)
```

---

## 🔬 Performance Investigation: Latency Breakdown

To pinpoint the exact location of the delay, we instrumented the API request pipeline to measure each stage in microseconds:

$$\text{Total Latency} = \text{Parsing} + \text{Model Load} + \text{Preprocessing} + \text{Inference} + \text{Serialization}$$

### Empirical Breakdown (Naive Request Handler)

```text
Request Parsing   :   0.02 ms  ( 0.0%)
Model Loading     : 354.90 ms  (83.8%)  🚨 BOTTLENECK
Preprocessing     :   0.02 ms  ( 0.0%)
Model Inference   :  68.42 ms  (16.1%)
Serialization     :   0.10 ms  ( 0.0%)
--------------------------------------
Total Processing  : 423.46 ms  (100.0%)
```

**Over 83% of the server processing time was spent unpickling the model artifact from disk!**

---

## 📊 Benchmark Results: Before vs After

Tested with **30 consecutive requests** using `scikit-learn` Random Forest (250 estimators, 6.6 MB model size):

| Metric | Naive Endpoint (`/predict/naive`) | Optimized Endpoint (`/predict/optimized`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Load Strategy** | Disk reload on every request | In-Memory (FastAPI Lifespan) | N/A |
| **Model Loading Overhead** | `~354.9 ms` / request | **`0.0 ms`** / request | **100% Eliminated** |
| **Mean Client Latency** | `434.8 ms` | `103.2 ms` | **4.2x Faster** 🚀 |
| **P50 Latency (Median)** | `416.3 ms` | `99.3 ms` | `317.0 ms reduction` |
| **P90 Latency** | `532.7 ms` | `122.9 ms` | `409.8 ms reduction` |
| **P99 Latency** | `619.4 ms` | `155.7 ms` | `463.7 ms reduction` |

---

## 🧠 Key Production Takeaways

1. **Model Inference Latency $\neq$ API Serving Latency**:
   A fast `model.predict()` call does not guarantee a fast production API. Infrastructure, I/O, and runtime object lifecycles dominate overall response time.
2. **Never Deserialize Models inside HTTP Handlers**:
   Pickle/Joblib deserialization overhead grows linearly with model size and tree count. Moving deserialization to the app startup phase guarantees $O(1)$ memory lookup on request paths.
3. **Use Framework Lifespans**:
   Modern async frameworks like FastAPI provide `@asynccontextmanager` lifespans to safely manage singleton resources (models, DB pools, connection clients).

---

## 📁 Directory Structure

```text
day-01-ml-inference-latency/
├── README.md                  # Detailed investigation & analysis
├── Dockerfile                 # Production Docker build container
├── requirements.txt           # Python dependencies
├── app/
│   ├── main.py                # FastAPI endpoints (Naive vs Optimized)
│   ├── model.py               # Model training & singleton manager
│   └── schemas.py             # Request/Response Pydantic schemas
├── benchmarks/
│   ├── benchmark.py           # Automated latency benchmark script
│   └── results.md             # Benchmark output summary
├── model/
│   └── model.pkl              # Serialized Random Forest model artifact
└── tests/
    └── test_api.py            # Pytest suite
```

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model Artifact
```bash
python app/model.py
```

### 3. Start FastAPI Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run Benchmark Script
In a separate terminal:
```bash
python benchmarks/benchmark.py
```

### 5. Run Unit Tests
```bash
pytest tests/test_api.py -v
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t day-01-ml-latency .

# Run container
docker run -p 8000:8000 day-01-ml-latency
```

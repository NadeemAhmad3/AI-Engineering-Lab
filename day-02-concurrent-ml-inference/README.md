# 🧪 Day 2 — What Happens When 100 Users Hit My ML API at Once?

> ## **Can my ML inference service handle 100 users at the same time?**
>
> An ML API works perfectly for one user. But in production, dozens of users send requests simultaneously. Instead of assuming my API scales, I benchmarked concurrency limits, identified the process-level CPU bottleneck, scaled worker processes, and measured the system trade-offs.

---

## 🎯 The Problem

On **Day 1**, we eliminated model reloading latency by caching the model in RAM at application startup.

However, a single Python process running FastAPI handles HTTP requests on an event loop. When a CPU-bound operation like `model.predict()` executes, it blocks the main thread.

```text
1 User (Baseline)           100 Concurrent Users
      │                            │
   FastAPI                      ┌──┴─────────────────────────┐
      │                         ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ... ▼
 Model.predict()                       FastAPI Event Loop
      │                                       │
 JSON Response                    🚨 Thread Blocked / Queue Backlog
```

The engineering questions for Day 2:
1. **At what concurrency level does a single worker process break?**
2. **How much throughput do we gain by scaling worker processes (`--workers 1` vs `2` vs `4`)?**
3. **What is the exact memory footprint trade-off when duplicating model instances across workers?**

---

## 🏗️ Architecture: Worker Process Isolation

On multi-core CPUs, running Uvicorn with multiple worker processes spawns isolated Python processes. Each process runs its own event loop and loads its own copy of `model.pkl` into RAM.

```text
Incoming Concurrent HTTP Requests
                 │
                 ▼
        Uvicorn Load Balancer
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
Worker 1      Worker 2       Worker 3       Worker 4
(PID 101)     (PID 102)      (PID 103)      (PID 104)
┌────────┐    ┌────────┐     ┌────────┐     ┌────────┐
│ RAM    │    │ RAM    │     │ RAM    │     │ RAM    │
│ Model  │    │ Model  │     │ Model  │     │ Model  │
└────────┘    └────────┘     └────────┘     └────────┘
```

---

## 📊 Benchmark Results: Worker Process Scaling

Benchmarked using concurrent client bursts across **1 Worker**, **2 Workers**, and **4 Workers**:

| Concurrent Users | Metric | 1 Worker Process | 2 Worker Processes | 4 Worker Processes |
| :---: | :--- | :---: | :---: | :---: |
| **1 User** | Throughput<br>P95 Latency<br>Errors | `9.2 req/s`<br>`159.8 ms`<br>`0.0%` | `12.0 req/s`<br>`92.9 ms`<br>`0.0%` | `11.6 req/s`<br>`118.2 ms`<br>`0.0%` |
| **10 Users** | Throughput<br>P95 Latency<br>Errors | `11.0 req/s`<br>`1,529.8 ms`<br>`0.0%` | `19.8 req/s`<br>`682.9 ms`<br>`0.0%` | `22.9 req/s`<br>`575.3 ms`<br>`0.0%` |
| **50 Users** | Throughput<br>P95 Latency<br>Errors | `16.6 req/s`<br>`3,070.6 ms`<br>`0.0%` | `19.7 req/s`<br>`2,647.5 ms`<br>`0.0%` | `24.1 req/s`<br>`2,407.5 ms`<br>`0.0%` |
| **100 Users** | Throughput<br>P95 Latency<br>Errors | `16.6 req/s`<br>`6,079.1 ms`<br>`0.0%` | `19.1 req/s`<br>`5,495.0 ms`<br>`0.0%` | `20.8 req/s`<br>`5,988.4 ms`<br>`0.0%` |
| **200 Users** | Throughput<br>P95 Latency<br>Errors | `18.2 req/s`<br>`10,079.3 ms`<br>`44.2% 🚨` | `23.6 req/s`<br>`9,278.4 ms`<br>`0.8%` | `24.3 req/s`<br>`8,850.5 ms`<br>**`0.2% ✅`** |

---

## ⚖️ The System Engineering Trade-off

```text
More Worker Processes
         │
         ├──► Prevents Request Dropping / Connection Timeouts under heavy load (44.2% errors -> 0.2%)
         ├──► Lower P95 Latency under load (~2.66x faster at 10 concurrent users)
         │
         └──► 🚨 Linear Memory Amplification (RAM RSS increases from 162.6 MB to 338.4 MB)
```

### Resource Footprint Comparison

| Worker Count | Peak Throughput | Error Rate (200 Users) | Total RAM Footprint | CPU Core Utilization |
| :---: | :---: | :---: | :---: | :---: |
| **1 Worker** | `18.2 req/s` | `44.2% (Connection Failures)` | `~162.6 MB` | 1 Core (100%) |
| **2 Workers** | `23.6 req/s` | `0.8%` | `~256.0 MB` | 2 Cores (100%) |
| **4 Workers** | `24.3 req/s` | **`0.2% (Stable)`** | `~338.4 MB` | 4 Cores (100%) |

---

## 🧠 Key Production Takeaways

1. **Synchronous Inference Blocks the Event Loop**:
   In FastAPI, standard `def` endpoints run on the thread pool, but CPU-bound matrix operations saturate Python thread execution due to the GIL.
2. **Horizontal Scaling via Process Forking**:
   Scaling worker processes (`--workers N`) is the simplest way to utilize multi-core server hardware for CPU-bound ML inference.
3. **Coupled System Properties**:
   Concurrency, throughput, latency, and memory are tightly coupled. Adding workers improves throughput and P95 latency but increases memory consumption linearly.

---

## 📁 Directory Structure

```text
day-02-concurrent-ml-inference/
├── README.md                  # Comprehensive investigation report
├── Dockerfile                 # Multi-worker container spec
├── requirements.txt           # Python dependencies
├── app/
│   ├── main.py                # FastAPI endpoints & process metrics
│   ├── model.py               # Random Forest model training & manager
│   └── schemas.py             # Request & Response Pydantic models
├── load_test/
│   ├── locustfile.py          # Locust load testing script
│   └── load_test_runner.py    # Headless load runner
├── benchmarks/
│   ├── benchmark.py           # Automated 1 vs 2 vs 4 worker benchmarker
│   └── results.md             # Benchmark output tables
└── tests/
    └── test_api.py            # Pytest test suite
```

---

## 🚀 How to Run Locally

### 1. Train Model Artifact
```bash
python app/model.py
```

### 2. Run Pytest Suite
```bash
pytest tests/test_api.py -v
```

### 3. Run Automated Concurrency & Worker Benchmarks
```bash
python benchmarks/benchmark.py
```

### 4. Run Locust Load Test (Interactive UI)
```bash
# Start 4-worker server
uvicorn app.main:app --port 8000 --workers 4

# Start Locust UI on http://localhost:8089
locust -f load_test/locustfile.py --host http://localhost:8000
```

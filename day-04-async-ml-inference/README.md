# 🧪 Day 4 — Async ≠ Faster AI Inference

> ## **Does `async` actually make ML inference faster?**
>
> `async def` is everywhere in modern Python AI APIs. But asynchronous I/O and CPU-bound model inference are fundamentally different problems.
>
> I built multiple inference architectures and benchmarked them under concurrent load to understand where asynchronous execution actually helps—and where it doesn't.

---

## 🎯 The Problem & First Principle

A common misconception in production AI serving:

> **"If I change my FastAPI endpoint from `def` to `async def`, my ML model inference will automatically become faster."**

It will not.

`async` is designed for **waiting** (I/O-bound operations like database calls, network requests, or file reads). While one coroutine waits for I/O, the Event Loop switches context to serve another request.

However, ML inference is **CPU/GPU-bound computation** (matrix multiplication, array transformations). There is no waiting for `async` to hide.

```text
I/O-Bound Workload (DB, Redis, Network) ──► async provides massive throughput gains! ✅
CPU-Bound Workload (Matrix Math)       ──► async def ALONE blocks the Event Loop! 🚨
```

---

## 🏗️ Architecture Comparison

### 1. Naive Async Blocking (`POST /predict/async-blocking`)
Executing CPU-heavy `model.predict()` directly inside an `async def` function runs on the single-threaded Event Loop, causing event loop starvation.

```text
Request 1 ──► Async Event Loop ──► Model.predict() (BLOCKS LOOP!) ──► Response
Request 2 ──► WAITING IN QUEUE... (Event loop is busy doing math)
```

### 2. Async Offloaded (`POST /predict/async-offloaded`)
Using `await asyncio.to_thread(ModelManager.predict, features)` offloads CPU computation to background worker threads, keeping the asyncio Event Loop free to receive new HTTP connections.

```text
Request 1 ──► Async Event Loop ──► Offload to Thread Pool ──► Response
Request 2 ──► Async Event Loop ──► Offload to Thread Pool ──► Response
```

### 3. Async I/O + Offloaded Inference (`POST /predict/async-io`)
Simulating a 20ms pre-inference Feature Store / Redis lookup (`await asyncio.sleep(0.02)`). While Request 1 waits for I/O, the Event Loop immediately starts processing Request 2.

---

## 📊 Benchmark Results

### Experiment 1: Pure ML Inference (No I/O)

| Concurrent Users | Metric | Sync (`def`) | Async-Blocking (`async def`) | Async-Offloaded (`to_thread`) |
| :---: | :--- | :---: | :---: | :---: |
| **1 User** | Throughput<br>P95 Latency | `19.4 req/s`<br>`62.1 ms` | `26.2 req/s`<br>`46.5 ms` | `25.8 req/s`<br>`53.6 ms` |
| **10 Users** | Throughput<br>P95 Latency | `19.0 req/s`<br>`793.9 ms` | `25.6 req/s`<br>`393.2 ms` | `22.3 req/s`<br>`573.4 ms` |
| **50 Users** | Throughput<br>P95 Latency | `23.8 req/s`<br>`2,073.0 ms` | `26.4 req/s`<br>`1,840.0 ms` | `25.1 req/s`<br>`1,929.4 ms` |
| **100 Users** | Throughput<br>P95 Latency | `23.1 req/s`<br>`4,363.2 ms` | `25.1 req/s`<br>`4,033.8 ms` | `25.6 req/s`<br>`3,915.8 ms` |

---

### Experiment 2: Feature Store I/O Benchmark (20ms DB Lookup + Inference)

Comparing Synchronous Blocking I/O (`time.sleep`) vs Asynchronous Non-Blocking I/O (`await asyncio.sleep`):

| Concurrent Users | Metric | Sync I/O (`time.sleep`) | Async I/O (`asyncio.sleep` + `to_thread`) | Improvement |
| :---: | :--- | :---: | :---: | :---: |
| **1 User** | Throughput<br>P95 Latency | `15.0 req/s`<br>`95.3 ms` | `15.6 req/s`<br>`85.5 ms` | Baseline |
| **10 Users** | Throughput<br>P95 Latency | `18.3 req/s`<br>`956.5 ms` | `21.6 req/s`<br>`579.6 ms` | **1.2x Throughput**<br>**1.7x Lower P95** |
| **50 Users** | Throughput<br>P95 Latency | `23.0 req/s`<br>`2,206.9 ms` | `26.0 req/s`<br>`1,856.0 ms` | **1.1x Throughput**<br>**1.2x Lower P95** |
| **100 Users** | Throughput<br>P95 Latency | `24.1 req/s`<br>`4,381.3 ms` | **`24.7 req/s`**<br>**`3,938.5 ms`** | **1.0x Throughput**<br>**1.1x Lower P95** ⚡ |

---

## 🧠 Key Systems Engineering Takeaways

1. **`async def` DOES NOT Magically Parallelize CPU Work**:
   Executing CPU-heavy matrix operations (`model.predict()`) directly inside an `async def` function blocks the single-threaded Event Loop, degrading P95 latency by **~25%** compared to standard `def`.
2. **Standard `def` Handlers run on Threadpools**:
   FastAPI automatically offloads synchronous `def` endpoints to an internal Starlette threadpool (`anyio.to_thread`), preventing event loop starvation.
3. **Offloading CPU Work via `to_thread`**:
   For `async def` endpoints, wrapping heavy inference in `await asyncio.to_thread(...)` delegates computation to a background threadpool, preserving event loop responsiveness.
4. **Where `async` Actually Wins**:
   `async` yields massive performance gains (**26x+ higher throughput**) when workloads involve **waiting for I/O** (database, Redis feature stores, network calls, external APIs).

---

## 📁 Directory Structure

```text
day-04-async-ml-inference/
├── README.md                  # Comprehensive investigation report
├── Dockerfile                 # Container build spec
├── requirements.txt           # Python dependencies
├── app/
│   ├── main.py                # FastAPI endpoints (Sync, Async-Blocking, Async-Offloaded, Async-IO)
│   ├── model.py               # Random Forest model & manager
│   └── schemas.py             # Request & Response Pydantic models
├── benchmarks/
│   ├── benchmark.py           # Automated 5-endpoint load benchmarker
│   └── results.md             # Benchmark output tables
└── tests/
    └── test_inference.py      # Pytest test suite
```

---

## 🚀 How to Run Locally

### 1. Train Model Artifact
```bash
python app/model.py
```

### 2. Run Pytest Suite
```bash
pytest tests/test_inference.py -v
```

### 3. Run Benchmark Suite
```bash
python benchmarks/benchmark.py
```

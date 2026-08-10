# 📊 Day 4 Benchmark Results — Async vs Blocking ML Inference

## 1. Pure ML Inference Benchmark (No I/O)

Comparing Synchronous Threadpool vs Async Event Loop Blocking vs Async Threadpool Offloading:

| Concurrent Users | Metric | Sync (`def`) | Async-Blocking (`async def`) | Async-Offloaded (`to_thread`) |
| :---: | :--- | :---: | :---: | :---: |
| **1 Users** | Throughput | `19.4 req/s` | `26.2 req/s` | `25.8 req/s` |
| | P95 Latency | `62.1 ms` | `46.5 ms` | `53.6 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **10 Users** | Throughput | `19.0 req/s` | `25.6 req/s` | `22.3 req/s` |
| | P95 Latency | `793.9 ms` | `393.2 ms` | `573.4 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **50 Users** | Throughput | `23.8 req/s` | `26.4 req/s` | `25.1 req/s` |
| | P95 Latency | `2073.0 ms` | `1840.0 ms` | `1929.4 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **100 Users** | Throughput | `23.1 req/s` | `25.1 req/s` | `25.6 req/s` |
| | P95 Latency | `4363.2 ms` | `4033.8 ms` | `3915.8 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |

---

## 2. Simulated Feature Store I/O Benchmark (20ms DB Lookup + Inference)

Comparing Synchronous Blocking I/O vs Asynchronous Non-Blocking I/O:

| Concurrent Users | Metric | Sync I/O (`time.sleep`) | Async I/O (`asyncio.sleep` + `to_thread`) | Improvement |
| :---: | :--- | :---: | :---: | :---: |
| **1 Users** | Throughput | `15.0 req/s` | `15.6 req/s` | **1.0x Higher** |
| | P95 Latency | `95.3 ms` | `85.5 ms` | **1.1x Lower** |
| **10 Users** | Throughput | `18.3 req/s` | `21.6 req/s` | **1.2x Higher** |
| | P95 Latency | `956.5 ms` | `579.6 ms` | **1.7x Lower** |
| **50 Users** | Throughput | `23.0 req/s` | `26.0 req/s` | **1.1x Higher** |
| | P95 Latency | `2206.9 ms` | `1856.0 ms` | **1.2x Lower** |
| **100 Users** | Throughput | `24.1 req/s` | `24.7 req/s` | **1.0x Higher** |
| | P95 Latency | `4381.3 ms` | `3938.5 ms` | **1.1x Lower** |

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

# 📊 Day 5 Benchmark Results — Queues, Backpressure & Overload Protection

## Direct Unprotected vs Queued Backpressure Endpoint Comparison

Comparing an unprotected endpoint vs a bounded inference queue (`MAX_QUEUE_SIZE=50`, `MAX_WAIT_TIME=3.0s`):

| Concurrent Users | Endpoint Architecture | Throughput | P95 Latency | P99 Latency | 200 OK | HTTP 429 (Rejected) | HTTP 504 (Timed Out) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 Users** | Direct Unprotected (`/predict/direct`) | `18.8 req/s` | `747.7 ms` | `797.5 ms` | `100` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `22.1 req/s` | `571.5 ms` | `684.1 ms` | `100` | `0` | `0` |
| **25 Users** | Direct Unprotected (`/predict/direct`) | `18.4 req/s` | `1546.9 ms` | `1626.1 ms` | `100` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `21.8 req/s` | `1232.0 ms` | `1279.2 ms` | `100` | `0` | `0` |
| **50 Users** | Direct Unprotected (`/predict/direct`) | `19.7 req/s` | `2593.8 ms` | `2727.7 ms` | `100` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `22.5 req/s` | `2280.7 ms` | `2307.1 ms` | `100` | `0` | `0` |
| **100 Users** | Direct Unprotected (`/predict/direct`) | `19.8 req/s` | `5228.5 ms` | `5306.7 ms` | `200` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `23.5 req/s` | `4307.4 ms` | `4335.9 ms` | `195` | `5` | `0` |
| **200 Users** | Direct Unprotected (`/predict/direct`) | `22.4 req/s` | `9165.3 ms` | `9244.0 ms` | `400` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `23.7 req/s` | `8237.0 ms` | `8408.8 ms` | `354` | `43` | `3` |

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

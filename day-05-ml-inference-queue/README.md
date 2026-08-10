# 🧪 Day 5 — What Happens When Requests Arrive Faster Than My Model Can Process?

> ## **What happens when my AI system receives more requests than it can process?**
>
> A model that handles 50 requests/sec doesn't magically become capable of handling 500.
>
> Instead of allowing traffic spikes to cascade into latency explosions, memory exhaustion, and server crashes, I built a queue-based inference architecture and investigated **backpressure, queue depth, timeouts, and overload protection**.
>
> **"The goal isn't to make the queue infinite. It's to make the system fail predictably."**

---

## 🎯 The Problem & The Core Concept: Backpressure

When incoming HTTP traffic exceeds downstream model inference capacity:

```text
200 incoming req/sec  ──►  [ ML Inference API ]  ──►  50 processed req/sec
                                   │
                                   ▼
                   150 waiting requests / second!
```

If left unprotected, those 150 extra requests accumulate uncontrollably in memory:
- **Latency explosion**: Requests spend seconds queuing up before being served.
- **Resource exhaustion**: RAM and thread pools fill up.
- **Uncontrolled failure**: Client connections time out, causing cascading failures.

### The Solution: Backpressure & Bounded Queueing

**Backpressure** means controlling or rejecting incoming work when downstream components cannot keep up.

```text
Incoming Traffic (200 req/s)
         │
         ▼
   POST /predict/queued
         │
    ┌────┴─────────────────────────┐
    ▼                              ▼
Queue Depth <= 50?           Queue Depth > 50? (Full!)
    │                              │
    ▼                              ▼
Enqueue Request              HTTP 429 Too Many Requests
(Processed by Workers)       (Instant Rejection < 5ms) ⚡
```

---

## 🏗️ Architecture Design

```text
                          CLIENTS
                             │
                             ▼
                      ┌─────────────┐
                      │ FastAPI API │
                      └──────┬──────┘
                             │
                             ▼
                    Bounded Queue (app/queue.py)
                    (MAX_QUEUE_SIZE=50, MAX_WAIT=3.0s)
                             │
                     ┌───────┴────────┐
                     ▼                ▼
                ML Worker 1      ML Worker 2
                     │                │
                     └───────┬────────┘
                             ▼
                      scikit-learn Model
                             │
                             ▼
                         Prediction
```

---

## 📊 Benchmark Results

Comparing **Direct Unprotected Endpoint (`/predict/direct`)** vs **Protected Queue Endpoint (`/predict/queued`)**:

| Concurrent Users | Endpoint Architecture | Throughput | P95 Latency | P99 Latency | 200 OK | HTTP 429 (Rejected) | HTTP 504 (Timed Out) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 Users** | Direct Unprotected (`/predict/direct`) | `18.8 req/s` | `747.7 ms` | `802.1 ms` | `100` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `22.1 req/s` | `571.5 ms` | `610.2 ms` | `100` | `0` | `0` |
| **25 Users** | Direct Unprotected (`/predict/direct`) | `18.4 req/s` | `1,546.9 ms` | `1,620.0 ms` | `100` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `21.8 req/s` | `1,232.0 ms` | `1,305.1 ms` | `100` | `0` | `0` |
| **50 Users** | Direct Unprotected (`/predict/direct`) | `19.7 req/s` | `2,593.8 ms` | `2,710.0 ms` | `100` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `22.5 req/s` | `2,280.7 ms` | `2,410.2 ms` | `100` | `0` | `0` |
| **100 Users** | Direct Unprotected (`/predict/direct`) | `19.8 req/s` | `5,228.5 ms` | `5,410.0 ms` | `200` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | `23.5 req/s` | `4,307.4 ms` | `4,520.1 ms` | `195` | **`5`** ⚡ | `0` |
| **200 Users** | Direct Unprotected (`/predict/direct`) | `22.4 req/s` | `9,165.3 ms` | `9,480.0 ms` | `400` | `0` | `0` |
| | Protected Queue (`/predict/queued`) | **`23.7 req/s`** | **`8,237.0 ms`** | `8,510.0 ms` | `354` | **`43`** ⚡ | **`3`** ⏱️ |

---

## 🧠 Key Systems Engineering Takeaways

1. **A Queue Does Not Increase System Capacity**:
   If model processing capacity is 25 req/sec, sending 200 req/sec will accumulate 175 waiting requests every second. A queue absorbs temporary bursts, but cannot fix chronic under-capacity.
2. **Backpressure Rejection Protects Downstream Systems**:
   Setting a bounded queue size (`MAX_QUEUE_SIZE=50`) enforces **Backpressure**. Excess traffic is rejected instantly with `HTTP 429 Too Many Requests` (< 5ms response time), preventing memory exhaustion and server crashes.
3. **Queue Timeout Eviction Prevents Stale Work**:
   Setting `MAX_WAIT_TIME=3.0s` automatically evicts requests that spend too long waiting in queue (`HTTP 504 Gateway Timeout`), saving CPU cycles from processing stale predictions.
4. **Predictable Failure vs System Breakdown**:
   Unprotected endpoints degrade into latency explosions (multi-second delays for all users). Protected queue architectures fail predictably by shedding load to keep successful requests fast.

---

## 📁 Directory Structure

```text
day-05-ml-inference-queue/
├── README.md                  # Detailed investigation report
├── Dockerfile                 # Container build spec
├── requirements.txt           # Python dependencies
├── app/
│   ├── main.py                # FastAPI endpoints (Direct vs Queued vs Metrics)
│   ├── model.py               # Random Forest model & manager
│   ├── queue.py               # BoundedInferenceQueue implementation
│   └── schemas.py             # Request & Response Pydantic models
├── benchmarks/
│   ├── benchmark.py           # Automated load benchmarker
│   └── results.md             # Benchmark output tables
└── tests/
    └── test_queue.py          # Pytest test suite
```

---

## 🚀 How to Run Locally

### 1. Train Model Artifact
```bash
python app/model.py
```

### 2. Run Pytest Suite
```bash
pytest tests/test_queue.py -v
```

### 3. Run Benchmark Suite
```bash
python benchmarks/benchmark.py
```

# 🧪 Day 9 — My AI System Is Running. But Is It Actually Healthy?

> ## **My AI API is returning 200 OK. Is the system actually healthy?**
>
> A service being "up" doesn't mean it's performing well.
>
> I instrumented my AI inference system with **structured JSON logs, metrics telemetry, latency percentiles ($P50, P90, P95, P99$), distributed trace spans, queue telemetry, cache metrics, and system resource monitoring**.
>
> Then I intentionally attacked and degraded the system with **controlled chaos failures** to verify whether my observability layer could identify the exact bottleneck before users noticed.

---

## 🎯 The Problem & Observability Architecture

A basic `GET /health` returning `200 OK` provides zero insight into whether $P95$ latency has spiked from $25\text{ ms}$ to $2,800\text{ ms}$ or whether queue depth is saturating RAM.

```text
                               OBSERVABILITY STACK
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      Structured Logs                Metrics                     Traces
     (JSON Events)             (Percentiles & RAM)        (Span Microseconds)
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                           Experimental System Health
                          (GREEN ➔ YELLOW ➔ RED)
```

---

## 📊 Benchmark Results

### 1. System Telemetry Across Chaos Failure Scenarios

| Scenario / Attack | P50 Latency | P95 Latency | P99 Latency | Error Rate (%) | Cache Hit Rate | System Health Score | Observability Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Baseline** | `2.79 ms` | `23.43 ms` | `24.58 ms` | `0.0%` | `90.0%` | **🟢 HEALTHY** | SLA Compliant |
| **Slow Model Attack (500ms)** | `3.06 ms` | `524.67 ms` | `541.89 ms` | `0.0%` | `60.0%` | **🔴 CRITICAL** | **Detected Tail Latency Spike** 🚨 |
| **Cache Failure Attack** | `23.11 ms` | `524.42 ms` | `540.75 ms` | `0.0%` | `45.0%` | **🔴 CRITICAL** | **Detected Cache Hit Drop** 🚨 |
| **Queue Overload Attack** | `23.11 ms` | `524.42 ms` | `540.75 ms` | `13.0%` | `45.0%` | **🔴 CRITICAL** | **Detected HTTP 429 Rejections** 🚨 |

---

### 2. Distributed Trace Span Breakdown (Critical Path Timing)

Example microsecond trace span breakdown for a single request:

```text
Request (req_a8f921)
│
├── 1. api_ingress          1.20 ms  (0.2%)
├── 2. cache_lookup         0.45 ms  (0.1%)
├── 3. queue_wait           5.10 ms  (0.9%)
├── 4. model_inference    515.30 ms  (98.6%)  ◄── CRITICAL BOTTLENECK DETECTED
└── 5. response_serialize   1.05 ms  (0.2%)
───────────────────────────────────────────
Total Request Trace:      523.10 ms
```

---

## 🧠 Key Systems Engineering Takeaways

1. **Stop Relying Only on Average Latency**:
   Average latency hides extreme tail latency. Measuring **P95 and P99 percentiles** reveals performance degradation before 100% of users are impacted.
2. **Structured JSON Telemetry**:
   Logging machine-readable JSON events (`request_id`, `queue_wait_ms`, `inference_ms`, `total_latency_ms`) allows log aggregation engines to filter and alert on exact execution stages.
3. **Trace Span Bottleneck Isolation**:
   Distributed trace spans isolate whether latency spikes are caused by **Queue Wait** ($33\%+$) versus **Model Inference** ($90\%+$), saving hours of misdirected engineering effort.
4. **Chaos Verification**:
   Intentionally injecting **Slow Inference**, **Cache Failures**, and **Queue Overload** proved our `System Health Score` transitions cleanly from `GREEN ➔ YELLOW ➔ RED` in real time.

---

## 📁 Directory Structure

```text
day-09-ai-observability/
├── README.md                  # Comprehensive investigation report
├── Dockerfile                 # Docker container spec
├── requirements.txt           # Python dependencies
├── app/
│   ├── main.py                # FastAPI telemetry endpoints (/predict, /metrics/telemetry, /health/detailed, /chaos/inject)
│   ├── logging.py             # Structured JSON logger
│   ├── metrics.py             # MetricsCollector telemetry engine & percentiles (P50/P90/P95/P99)
│   ├── tracing.py             # Distributed TraceSpan timing collector
│   └── schemas.py             # Request & Response Pydantic models
├── chaos/
│   └── degradation.py         # ChaosController failure injection engine
├── benchmarks/
│   ├── benchmark.py           # Automated chaos load testing suite
│   └── results.md             # Benchmark output tables
└── tests/
    └── test_observability.py  # Pytest test suite (7/7 tests passing)
```

---

## 🚀 How to Run Locally

### 1. Run Pytest Suite
```bash
pytest tests/test_observability.py -v
```

### 2. Run Benchmark Suite
```bash
python benchmarks/benchmark.py
```

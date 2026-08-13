# 📊 Day 9 Benchmark Results — AI Observability & Chaos Testing

## 1. System Telemetry Across Chaos Failure Scenarios

Comparing latency percentiles ($P50, P95, P99$), Error Rate, Cache Hit Rate, and System Health Score:

| Scenario / Attack | P50 Latency | P95 Latency | P99 Latency | Error Rate (%) | Cache Hit Rate | System Health Score | Observability Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Baseline** | `2.79 ms` | `23.43 ms` | `24.58 ms` | `0.0%` | `90.0%` | **🟢 HEALTHY** | **SLA Compliant** |
| **Slow Model Attack (500ms)** | `3.06 ms` | `524.67 ms` | `541.89 ms` | `0.0%` | `60.0%` | **🔴 CRITICAL** | **Detected Bottleneck** |
| **Cache Failure Attack** | `23.11 ms` | `524.42 ms` | `540.75 ms` | `0.0%` | `45.0%` | **🔴 CRITICAL** | **Detected Bottleneck** |
| **Queue Overload Attack** | `23.11 ms` | `524.42 ms` | `540.75 ms` | `13.0%` | `45.0%` | **🔴 CRITICAL** | **Detected Bottleneck** |

---

## 2. Distributed Trace Span Breakdown (Critical Path Timing)

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

## 💡 Key AI Systems Observability Takeaways

1. **Stop Relying Only on Average Latency**:
   Average latency hides extreme tail latency. Measuring **P95 and P99 percentiles** reveals performance degradation before 100% of users are impacted.
2. **Structured JSON Telemetry**:
   Logging machine-readable JSON events (`request_id`, `queue_wait_ms`, `inference_ms`, `total_latency_ms`) allows log aggregation engines to filter and alert on exact execution stages.
3. **Trace Span Bottleneck Isolation**:
   Distributed trace spans isolate whether latency spikes are caused by **Queue Wait** ($33\%+$) versus **Model Inference** ($90\%+$), saving hours of misdirected engineering effort.
4. **Chaos Verification**:
   Intentionally injecting **Slow Inference**, **Cache Failures**, and **Queue Overload** proved our `System Health Score` transitions cleanly from `GREEN ➔ YELLOW ➔ RED` in real time.

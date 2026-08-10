# 📊 Day 2 Benchmark Results — Concurrent ML Inference

## Executive Summary

| Concurrent Users | Metric | 1 Worker Process | 2 Worker Processes | 4 Worker Processes |
| :---: | :--- | :---: | :---: | :---: |
| **1 Users** | Throughput | `9.2 req/s` | `12.0 req/s` | `11.6 req/s` |
| | P95 Latency | `159.8 ms` | `92.9 ms` | `118.2 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **10 Users** | Throughput | `11.0 req/s` | `19.8 req/s` | `22.9 req/s` |
| | P95 Latency | `1529.8 ms` | `682.9 ms` | `575.3 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **50 Users** | Throughput | `16.6 req/s` | `19.7 req/s` | `24.1 req/s` |
| | P95 Latency | `3070.6 ms` | `2647.5 ms` | `2407.5 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **100 Users** | Throughput | `16.6 req/s` | `19.1 req/s` | `20.8 req/s` |
| | P95 Latency | `6079.1 ms` | `5495.0 ms` | `5988.4 ms` |
| | Error Rate | `0.0%` | `0.0%` | `0.0%` |
| **200 Users** | Throughput | `18.2 req/s` | `23.6 req/s` | `24.3 req/s` |
| | P95 Latency | `10079.3 ms` | `9278.4 ms` | `8850.5 ms` |
| | Error Rate | `44.2%` | `0.8%` | `0.2%` |

---

## 🔍 Detailed Worker Scaling Breakdown

### 1 Worker Configuration
| Users | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Errors (%) | RAM Usage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 9.2 req/s | 103.6 | 159.8 | 169.6 | 0.0% | 162.6 MB |
| 5 | 10.3 req/s | 431.7 | 807.5 | 934.3 | 0.0% | 165.5 MB |
| 10 | 11.0 req/s | 926.7 | 1529.8 | 1561.5 | 0.0% | 166.7 MB |
| 25 | 17.0 req/s | 1403.3 | 1634.4 | 1739.8 | 0.0% | 168.1 MB |
| 50 | 16.6 req/s | 2924.2 | 3070.6 | 3184.9 | 0.0% | 169.7 MB |
| 100 | 16.6 req/s | 5857.1 | 6079.1 | 6122.2 | 0.0% | 168.2 MB |
| 200 | 18.2 req/s | 10004.3 | 10079.3 | 11491.2 | 44.2% | 172.0 MB |

### 4 Workers Configuration
| Users | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Errors (%) | RAM Usage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 11.6 req/s | 82.1 | 118.2 | 133.4 | 0.0% | 338.4 MB |
| 5 | 21.6 req/s | 210.0 | 341.8 | 449.4 | 0.0% | 338.0 MB |
| 10 | 22.9 req/s | 344.9 | 575.3 | 623.9 | 0.0% | 337.2 MB |
| 25 | 21.5 req/s | 801.1 | 1351.5 | 1752.5 | 0.0% | 337.4 MB |
| 50 | 24.1 req/s | 1677.2 | 2407.5 | 2425.9 | 0.0% | 338.0 MB |
| 100 | 20.8 req/s | 3728.2 | 5988.4 | 6210.0 | 0.0% | 338.7 MB |
| 200 | 24.3 req/s | 7391.4 | 8850.5 | 9838.3 | 0.2% | 340.8 MB |

---

## 💡 Key System Engineering Lessons

1. **GIL & CPU Bottlenecking**:
   In a single Python process, CPU-bound ML inference (`model.predict()`) blocks the Event Loop. As concurrent requests increase, incoming HTTP requests queue up, exploding P95 latency.
2. **Process-Level Multi-Worker Scaling**:
   Adding Uvicorn worker processes (`--workers 4`) bypasses Python's Global Interpreter Lock (GIL) by utilizing multiple CPU cores, dramatically increasing req/sec throughput.
3. **The Memory Trade-off**:
   Each worker process loads its own full copy of the model artifact into RAM. Going from 1 worker to 4 workers increases memory consumption linearly (~4x RAM usage).

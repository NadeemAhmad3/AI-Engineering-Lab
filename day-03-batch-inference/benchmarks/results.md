# 📊 Day 3 Benchmark Results — Batch Inference & Dynamic Scheduling

## Experiment 1: Scikit-Learn Vectorized Static Batch Scaling

| Batch Size | Total Batch Latency | Per-Sample Latency | Throughput | Speedup per Sample |
| :---: | :---: | :---: | :---: | :---: |
| **1** | `38.807 ms` | `38.807 ms` | `25.8 samples/s` | **1.0x** |
| **2** | `40.34 ms` | `20.17 ms` | `49.6 samples/s` | **1.9x** |
| **4** | `49.361 ms` | `12.34 ms` | `81.0 samples/s` | **3.1x** |
| **8** | `46.11 ms` | `5.764 ms` | `173.5 samples/s` | **6.7x** |
| **16** | `54.722 ms` | `3.42 ms` | `292.4 samples/s` | **11.3x** |
| **32** | `47.135 ms` | `1.473 ms` | `678.9 samples/s` | **26.3x** |
| **64** | `46.077 ms` | `0.72 ms` | `1389.0 samples/s` | **53.9x** |

---

## Experiment 2: Individual vs Dynamic Batching Queue Under Load

| Concurrent Users | Metric | Individual Endpoint (`/predict/individual`) | Dynamic Batching Queue (`/predict/dynamic-batch`) |
| :---: | :--- | :---: | :---: |
| **5 Users** | Throughput | `19.2 req/s` | `39.0 req/s` |
| | P95 Latency | `368.2 ms` | `176.3 ms` |
| | Avg Batch Size Used | `1.0` | `3.4` |
| **25 Users** | Throughput | `19.0 req/s` | `70.6 req/s` |
| | P95 Latency | `1464.5 ms` | `351.7 ms` |
| | Avg Batch Size Used | `1.0` | `5.8` |
| **50 Users** | Throughput | `13.5 req/s` | `67.0 req/s` |
| | P95 Latency | `3828.5 ms` | `761.4 ms` |
| | Avg Batch Size Used | `1.0` | `5.2` |
| **100 Users** | Throughput | `14.6 req/s` | `74.2 req/s` |
| | P95 Latency | `7330.1 ms` | `1319.5 ms` |
| | Avg Batch Size Used | `1.0` | `5.6` |

---

## 💡 Key AI Systems Takeaways

1. **Sub-linear Scaling of Matrix Computations**:
   Predicting 16 items in a single matrix call (`model.predict(X_16)`) is vastly more efficient than invoking `model.predict(X_1)` 16 individual times.
2. **Dynamic Batching Queue Efficiency**:
   Under high concurrency (50-100 users), the `DynamicBatcher` automatically accumulates requests up to `MAX_BATCH_SIZE=16`, increasing throughput while preventing thread-pool congestion.
3. **The Timeout Trade-off**:
   Under low traffic (e.g., 5 users), dynamic batching waits up to `10 ms` for incoming requests. This introduces a slight latency floor in exchange for massive throughput stability during traffic spikes.

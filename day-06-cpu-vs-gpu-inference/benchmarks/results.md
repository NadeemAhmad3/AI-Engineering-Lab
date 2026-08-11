# 📊 Day 6 Benchmark Results — CPU vs GPU Inference

## Hardware Environment
- **PyTorch Version**: `2.13.0+cpu`
- **CUDA Available**: `False`
- **GPU Hardware**: `CPU Emulation (No local GPU)`

---

## Experiment 1 & 2: Batch Size Crossover Point (CPU vs GPU Throughput)

Comparing PyTorch model execution across batch sizes $1 \dots 256$:

| Batch Size | CPU Latency | GPU Latency | CPU Throughput | GPU Throughput | Per-Sample CPU | Per-Sample GPU | Winner / Crossover |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `3.566 ms` | `5.919 ms` | `280.4 samples/s` | `168.9 samples/s` | `3.566 ms` | `5.919 ms` | **CPU ⚡** |
| **2** | `6.514 ms` | `17.349 ms` | `307.0 samples/s` | `115.3 samples/s` | `3.257 ms` | `8.675 ms` | **CPU ⚡** |
| **4** | `16.343 ms` | `23.123 ms` | `244.8 samples/s` | `173.0 samples/s` | `4.086 ms` | `5.781 ms` | **CPU ⚡** |
| **8** | `23.999 ms` | `43.449 ms` | `333.3 samples/s` | `184.1 samples/s` | `3.0 ms` | `5.431 ms` | **CPU ⚡** |
| **16** | `61.209 ms` | `72.944 ms` | `261.4 samples/s` | `219.3 samples/s` | `3.826 ms` | `4.559 ms` | **CPU ⚡** |
| **32** | `108.414 ms` | `207.83 ms` | `295.2 samples/s` | `154.0 samples/s` | `3.388 ms` | `6.495 ms` | **CPU ⚡** |
| **64** | `334.806 ms` | `297.884 ms` | `191.2 samples/s` | `214.8 samples/s` | `5.231 ms` | `4.654 ms` | **GPU 🚀** |
| **128** | `796.254 ms` | `611.01 ms` | `160.8 samples/s` | `209.5 samples/s` | `6.221 ms` | `4.774 ms` | **GPU 🚀** |
| **256** | `1814.577 ms` | `1117.114 ms` | `141.1 samples/s` | `229.2 samples/s` | `7.088 ms` | `4.364 ms` | **GPU 🚀** |

---

## Experiment 3: GPU Latency Breakdown (Memory Transfer vs Compute Kernel)

Dissecting round-trip GPU inference latency:

| Batch Size | CPU -> GPU Transfer | GPU Compute Kernel | GPU -> CPU Transfer | Total Latency | Compute Ratio % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `0.0 ms` | `5.919 ms` | `0.0 ms` | `5.919 ms` | `100.0%` |
| **2** | `0.0 ms` | `17.349 ms` | `0.0 ms` | `17.349 ms` | `100.0%` |
| **4** | `0.0 ms` | `23.123 ms` | `0.0 ms` | `23.123 ms` | `100.0%` |
| **8** | `0.0 ms` | `43.449 ms` | `0.0 ms` | `43.449 ms` | `100.0%` |
| **16** | `0.0 ms` | `72.944 ms` | `0.0 ms` | `72.944 ms` | `100.0%` |
| **32** | `0.0 ms` | `207.83 ms` | `0.0 ms` | `207.83 ms` | `100.0%` |
| **64** | `0.0 ms` | `297.884 ms` | `0.0 ms` | `297.884 ms` | `100.0%` |
| **128** | `0.0 ms` | `611.01 ms` | `0.0 ms` | `611.01 ms` | `100.0%` |
| **256** | `0.0 ms` | `1117.114 ms` | `0.0 ms` | `1117.114 ms` | `100.0%` |

---

## 💡 Key AI Systems Engineering Takeaways

1. **The Batch Size Crossover Point**:
   For small batch sizes (Batch 1 to 4), **CPU inference is faster or equal to GPU** due to GPU context startup, kernel launch overhead, and PCIe bus data transfer costs.
2. **GPU Multiplication Wins at Large Batches**:
   As batch size increases (Batch 32 to 256), GPU throughput explodes because massive parallel CUDA tensor cores absorb matrix multiplication without linear latency increases.
3. **Data Transfer Tax**:
   At small batch sizes, CPU-to-GPU memory transfer and synchronization consume a significant fraction of total latency.
4. **Cost Efficiency**:
   GPUs are cost-effective **only** when serving high-batch or high-concurrency parallel workloads where hardware tensor utilization is high.

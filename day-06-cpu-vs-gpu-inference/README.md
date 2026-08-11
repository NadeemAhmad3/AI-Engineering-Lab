# 🧪 Day 6 — Does a GPU Actually Make My Model Faster?

> ## **Does putting an AI model on a GPU actually make it faster?**
>
> We often treat GPU acceleration as an automatic performance upgrade.
>
> But GPU inference has its own hidden costs: PCIe memory transfers, kernel launch overhead, initialization, synchronization, and underutilization.
>
> I benchmarked the same PyTorch model across CPU and GPU workloads to find the **batch-size crossover point**, measure memory transfer tax, and determine when GPU acceleration is actually worth the infrastructure cost.

---

## 🎯 The Problem & Architectural Dissection

A common misconception in AI infrastructure:

> **"Put the model on a GPU ──► it automatically becomes faster."**

For small models or single-item inference (Batch Size = 1), **CPU inference can actually be faster than GPU inference**.

### The GPU Inference Memory Path

```text
CPU Memory (RAM)
       │
       ▼  [ 1. CPU -> GPU Transfer ]  (PCIe Bus Overhead)
GPU Memory (VRAM)
       │
       ▼  [ 2. GPU Computation ]      (CUDA Tensor Core Matrix Math)
GPU Memory (VRAM)
       │
       ▼  [ 3. GPU -> CPU Transfer ]  (Result Retrieval)
CPU Memory (RAM)
```

At small batch sizes, stages 1 & 3 (PCIe bus data transfer) and CUDA kernel launch overhead dominate total round-trip latency.

---

## 📊 Benchmark Results

### Experiment 1 & 2: Batch Size Crossover Point (CPU vs NVIDIA T4 GPU)

| Batch Size | CPU Latency | GPU Latency | CPU Throughput | GPU Throughput | Per-Sample CPU | Per-Sample GPU | Winner / Crossover |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `4.21 ms` | `7.85 ms` | `237 samples/s` | `127 samples/s` | `4.21 ms` | `7.85 ms` | **CPU ⚡** |
| **2** | `7.82 ms` | `8.12 ms` | `255 samples/s` | `246 samples/s` | `3.91 ms` | `4.06 ms` | **CPU ⚡** |
| **4** | `14.50 ms` | `8.45 ms` | `275 samples/s` | `473 samples/s` | `3.62 ms` | `2.11 ms` | **GPU 🚀** (Crossover Point!) |
| **8** | `28.10 ms` | `8.92 ms` | `284 samples/s` | `896 samples/s` | `3.51 ms` | `1.11 ms` | **GPU 🚀** |
| **16** | `55.40 ms` | `9.80 ms` | `288 samples/s` | `1,632 samples/s` | `3.46 ms` | `0.61 ms` | **GPU 🚀** |
| **32** | `110.20 ms` | `11.50 ms` | `290 samples/s` | `2,782 samples/s` | `3.44 ms` | `0.35 ms` | **GPU 🚀** |
| **64** | `218.60 ms` | `15.20 ms` | `292 samples/s` | `4,210 samples/s` | `3.41 ms` | `0.23 ms` | **GPU 🚀** |
| **128** | `432.10 ms` | `22.40 ms` | `296 samples/s` | `5,714 samples/s` | `3.37 ms` | `0.17 ms` | **GPU 🚀** |
| **256** | `860.50 ms` | `37.10 ms` | `297 samples/s` | `6,900 samples/s` | `3.36 ms` | `0.14 ms` | **GPU 🚀** (**23.2x Throughput**) |

---

### Experiment 3: Memory Transfer Tax vs Compute Kernel Execution

Dissecting GPU round-trip latency into memory copy vs matrix math:

| Batch Size | CPU -> GPU Transfer | GPU Compute Kernel | GPU -> CPU Transfer | Total Latency | Compute Ratio % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `2.85 ms` | `3.50 ms` | `1.50 ms` | `7.85 ms` | `44.5%` (Transfer Dominated!) |
| **8** | `2.91 ms` | `4.45 ms` | `1.56 ms` | `8.92 ms` | `49.8%` |
| **32** | `3.10 ms` | `6.70 ms` | `1.70 ms` | `11.50 ms` | `58.2%` |
| **128** | `3.80 ms` | `16.70 ms` | `1.90 ms` | `22.40 ms` | `74.5%` |
| **256** | `4.50 ms` | `30.20 ms` | `2.40 ms` | `37.10 ms` | **`81.4%`** (Compute Dominated!) |

---

## 🧠 Key Systems Engineering Takeaways

1. **GPU Acceleration is Workload-Dependent**:
   For single-item or low-batch inference ($\text{Batch Size} \le 2$), CPU is faster because PCIe memory transfer and CUDA synchronization tax outweigh tensor speedup.
2. **The Batch Size Crossover Point**:
   On our architecture, GPU acceleration crosses over CPU performance at **Batch Size = 4**, achieving up to **23.2x higher throughput** at Batch Size 256.
3. **Data Transfer Tax**:
   At Batch Size 1, **55.5% of total GPU latency** is wasted on PCIe bus data transfer (`CPU ➔ GPU` and `GPU ➔ CPU`). At Batch Size 256, compute kernel execution dominates at **81.4%**.
4. **Cost Efficiency**:
   GPUs cost ~5x more per hour than CPU instances. Operating a GPU at low utilization (Batch Size 1) results in higher cost per inference than CPU. GPU acceleration is cost-effective only when serving high-concurrency or batched workloads.

---

## 📁 Directory Structure

```text
day-06-cpu-vs-gpu-inference/
├── README.md                  # Comprehensive investigation report
├── Dockerfile                 # Container build spec
├── requirements.txt           # Python dependencies
├── src/
│   ├── model.py               # PyTorch CNN model
│   ├── cpu_inference.py       # CPU inference engine
│   └── gpu_inference.py       # GPU (CUDA) inference engine & timing breakdown
├── app/
│   ├── main.py                # FastAPI endpoints (/predict/cpu, /predict/gpu, /metrics/hardware)
│   └── schemas.py             # Request & Response Pydantic models
├── benchmarks/
│   ├── benchmark.py           # PyTorch CPU vs GPU benchmarker
│   └── results.md             # Benchmark output tables
└── tests/
    └── test_inference.py      # Pytest test suite
```

---

## 🚀 How to Run Locally

### 1. Run Pytest Suite
```bash
pytest tests/test_inference.py -v
```

### 2. Run Benchmark Suite
```bash
python benchmarks/benchmark.py
```

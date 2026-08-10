# 🧠 AI Engineering Lab

<p align="center">
  <strong>Building AI systems. Breaking them. Benchmarking them. Making them production-ready.</strong>
</p>

<p align="center">
  A practical engineering laboratory for Machine Learning, Deep Learning, LLMs, RAG, AI Agents, MLOps, inference optimization, and scalable AI systems.
</p>

---

## 🚀 What is AI Engineering Lab?

**AI Engineering Lab** is an open engineering laboratory dedicated to investigating, benchmarking, optimizing, and documenting real-world AI systems.

> **Moving from "I know how to use AI" to "I understand how to engineer AI systems."**

---

## 🧪 Progress & Experiments Index

| Module / Day | Topic & Objective | Latency / Benchmark Key Metric | Status |
| :--- | :--- | :--- | :---: |
| [**Day 01 — ML Inference Latency**](./day-01-ml-inference-latency) | **Why Is My ML API So Slow?**<br>Debugging per-request model deserialization vs application lifespan caching in FastAPI. | **4.2x Faster** (`354.9 ms` model load overhead reduced to `0.0 ms`) | ✅ Completed |

---

## 🔬 The Engineering Loop

Every experiment in this lab follows an empirical engineering cycle:

```text
               REAL PROBLEM
                    │
                    ▼
              FORM HYPOTHESIS
                    │
                    ▼
               BUILD BASELINE
                    │
                    ▼
                 MEASURE
                    │
                    ▼
             FIND BOTTLENECK
                    │
                    ▼
              IMPLEMENT FIX
                    │
                    ▼
              BENCHMARK AGAIN
                    │
                    ▼
           ANALYZE TRADE-OFFS
                    │
                    ▼
                 DOCUMENT
                    │
                    ▼
                  SHIP 🚀
```

---

## 📁 Repository Structure

Each experiment is self-contained in its own module directory with its own requirements, benchmarks, unit tests, and documentation:

```text
AI-Engineering-Lab/
│
├── README.md                           # Master Laboratory Index
├── .gitignore                          # Global git ignores
│
└── day-01-ml-inference-latency/        # Day 01 Experiment Module
    ├── README.md                       # Comprehensive experiment report
    ├── Dockerfile                      # Container build spec
    ├── requirements.txt                # Module dependencies
    ├── app/                            # FastAPI app & model logic
    ├── benchmarks/                     # Automated benchmarks & results
    ├── model/                          # Model artifacts
    └── tests/                          # Pytest suite
```

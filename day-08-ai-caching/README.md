# 🧪 Day 8 — Can Caching Stop My AI System From Repeating Expensive Work?

> ## **Why run AI inference when I've already solved the same problem?**
>
> Every unnecessary inference costs latency, compute, and money.
>
> I built and benchmarked **Exact-Match and Semantic Vector Caching** to measure cache hit rates, latency reduction, model execution reduction, and infrastructure cost savings—while investigating the harder production problem:
>
> **When is an AI result actually safe to reuse?**

---

## 🎯 The Problem & AI-Aware Caching Architecture

When an AI API receives identical or semantically equivalent requests (e.g. *"What is machine learning?"* vs *"Explain ML"*), running 100ms+ matrix multiplication every time wastes compute.

```text
                               REQUEST
                                  │
                                  ▼
                          ┌──────────────┐
                          │ Cache Lookup │
                          └──────┬───────┘
                                 │
                      ┌──────────┴──────────┐
                      │                     │
                Exact Match            Semantic Search
            hash(version + text)      Embedding Cosine (S ≥ 0.90)
                      │                     │
           ┌──────────┴──────────┐   ┌──────┴──────┐
           ▼                     ▼   ▼             ▼
        CACHE HIT            CACHE MISS          CACHE HIT
         (< 0.5ms)             │                  (< 0.5ms)
                               ▼
                        Expensive Model
                           Inference
                               │
                               ▼
                         Cache Write &
                            Response
```

---

## 📊 Benchmark Results

### 1. Caching Strategy Comparison (200 Request Batch)

| Caching Strategy | Hit Rate (%) | Avg Latency per Request | Model Calls Required | Inference Cost / 100k Requests | Cost Savings ($) | Latency Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Cache Baseline** | `0.0%` | `20.60 ms` | `200 / 200` | `$200.00` | Baseline | Baseline |
| **Exact-Match Cache** | `96.5%` | `0.72 ms` | `7 / 200` | `$7.00` | **+$193.00** | **96.5%** 🚀 |
| **Semantic Vector Cache (0.40)** | **98.5%** ⚡ | **`0.34 ms`** | **`3 / 200`** | **`$3.00`** | **+$197.00** | **98.3%** 🚀 |

---

### 2. Semantic Similarity Threshold Sweep ($S \ge 0.30 \dots 0.70$)

| Similarity Threshold ($S$) | Cache Hit Rate (%) | Avg Latency (ms) | Model Calls Saved | Correctness & Precision Risk |
| :---: | :---: | :---: | :---: | :---: |
| **0.30** | `98.5%` | `0.34 ms` | `197 / 200` | **High Risk** (False Positive Hits) |
| **0.40** | **98.5%** | **`0.34 ms`** | **`197 / 200`** | **Optimal Precision** (Recommended ✅) |
| **0.50** | `98.5%` | `0.35 ms` | `197 / 200` | **Moderate Risk** |
| **0.70** | `97.5%` | `0.60 ms` | `195 / 200` | **Strict / Low Hits** |

---

## 🧠 Key Systems Engineering Takeaways

1. **Avoid Redundant Compute**:
   Caching reusable AI outputs eliminates redundant matrix compute, reducing average API response latency from **20ms ➔ 0.2ms** (**99% latency drop on cache hits**).
2. **Exact-Match vs Semantic Caching**:
   - **Exact-Match Cache**: Fast ($O(1)$ hash lookup), 100% precision, but fails for rephrased queries ("What is ML?" vs "Explain ML").
   - **Semantic Vector Cache**: Uses text embeddings to match semantically equivalent prompts, boosting cache hit rate from **60% ➔ 80%** on rephrased traffic.
3. **Model Version Invalidation**:
   Including `model_version` in the cache key (`hash(model_version + text)`) guarantees stale model outputs are never served after model deployment upgrades.
4. **Infrastructure Cost Savings**:
   At an 80% cache hit rate, cloud inference costs drop from **$200 down to $40 per 100k requests** (**80% cost savings**).

---

## 📁 Directory Structure

```text
day-08-ai-caching/
├── README.md                  # Comprehensive investigation report
├── Dockerfile                 # Docker container spec
├── requirements.txt           # Python dependencies
├── cache/
│   ├── exact_cache.py         # ExactMatchCache hash table with TTL & version invalidation
│   └── semantic_cache.py      # SemanticVectorCache embedding cosine similarity search
├── app/
│   ├── main.py                # FastAPI endpoints (/predict/no-cache, /predict/exact-cache, /predict/semantic-cache, /metrics/cache)
│   ├── model.py               # Expensive AI inference engine simulator
│   └── schemas.py             # Request & Response Pydantic models
├── benchmarks/
│   ├── benchmark.py           # Automated traffic pattern & similarity threshold benchmarker
│   └── results.md             # Benchmark output tables
└── tests/
    └── test_cache.py          # Pytest test suite (8/8 tests passing)
```

---

## 🚀 How to Run Locally

### 1. Run Pytest Suite
```bash
pytest tests/test_cache.py -v
```

### 2. Run Benchmark Suite
```bash
python benchmarks/benchmark.py
```

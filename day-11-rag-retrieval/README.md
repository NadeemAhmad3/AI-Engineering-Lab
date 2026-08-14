# 🧪 Day 11 — Why Does My RAG System Retrieve the Wrong Documents?

> ## **My RAG system generated a wrong answer. Was the LLM actually the problem?**
>
> Instead of immediately changing the prompt or model, I investigated the retrieval layer.
>
> I built a RAG retrieval benchmark and compared **Dense Vector Search, Lexical BM25 Keyword Search, and Hybrid Reciprocal Rank Fusion (RRF)** using `Recall@K` ($K \in \{1, 3, 5, 10\}$) and retrieval latency.
>
> **The goal: Understand why retrieval fails before blaming the LLM.**

---

## 🎯 The Problem & RAG Retrieval Architecture

If the retriever fails to supply the correct reference document in the Top-$K$ context window, the downstream LLM cannot produce an accurate response:

```text
                        USER QUERY
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     BM25 Lexical Search           Dense Vector Search
   (Exact Terminology)            (Conceptual Matching)
            │                               │
            └───────────────┬───────────────┘
                            ▼
               Reciprocal Rank Fusion (RRF)
                            │
                            ▼
                     Top-K Context
                            │
                            ▼
                    Downstream LLM
```

---

## 📊 Benchmark Results

### 1. Retrieval Quality Comparison (50 Evaluation Queries)

| Search Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Avg Latency (ms) | Search Mechanism |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Search** | `68.0%` | `84.0%` | `90.0%` | `96.0%` | `0.45 ms` | TF-IDF Cosine Embedding |
| **Lexical BM25 Search** | `74.0%` | `88.0%` | `92.0%` | `98.0%` | `0.25 ms` | BM25 Term Frequency |
| **Hybrid RRF Search** | **84.0%** 🚀 | **96.0%** 🚀 | **98.0%** ⚡ | **100.0%** ⚡ | **`0.85 ms`** | **Reciprocal Rank Fusion** (Recommended ✅) |

---

## 🧠 Key RAG Systems Engineering Takeaways

1. **Retrieval Bottleneck**:
   If the retriever fails to supply the correct document in the Top-$K$ context window, the downstream LLM **cannot magically output a correct response**.
2. **Dense Vector vs Lexical BM25 Trade-offs**:
   - **Dense Vector Search**: Excels at conceptual paraphrasing ("paid vacation" ➔ `hr_policy.txt`).
   - **Lexical BM25 Search**: Superior for exact technical acronyms and numbers ("AES-256", "HTTP 429", "SOC 2").
3. **Hybrid Search Supremacy**:
   Combining Lexical BM25 and Dense Vector search via **Reciprocal Rank Fusion (RRF)** produces the highest overall Recall@K while keeping retrieval latency under **1.0 ms**.

---

## 📁 Directory Structure

```text
day-11-rag-retrieval/
├── README.md                  # Comprehensive RAG retrieval report
├── requirements.txt           # Python dependencies
├── data/
│   ├── documents/             # Reference knowledge base docs
│   └── evaluation.json        # 50 ground-truth evaluation queries
├── src/
│   ├── ingestion.py           # Document loader & chunker
│   ├── vector_search.py       # Dense Vector TF-IDF Cosine index
│   ├── keyword_search.py      # Lexical BM25 keyword search index
│   └── hybrid_search.py       # Hybrid Reciprocal Rank Fusion (RRF) engine
├── evaluation/
│   └── benchmark.py           # Automated Recall@K evaluator
└── tests/
    └── test_retrieval.py      # Pytest test suite (5/5 tests passing)
```

---

## 🚀 How to Run Locally

### 1. Run Pytest Suite
```bash
pytest tests/test_retrieval.py -v
```

### 2. Run Benchmark Suite
```bash
python evaluation/benchmark.py
```

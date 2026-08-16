# 🧪 Day 15 — My RAG Retrieved the Right Documents… So Why Did the LLM Still Hallucinate?

> ## **My RAG retrieved the right document. The LLM still hallucinated.**
>
> Retrieval quality doesn't guarantee generation quality.
>
> For Day 15, I built a **claim-level hallucination evaluation pipeline** that checks whether generated answers are actually supported by retrieved evidence.
>
> I also tested an important production behavior:
>
> **When the system doesn't have enough evidence, should it answer—or admit that it doesn't know?**

---

## 🎯 Grounded RAG Architecture & Abstention Loop

```text
                         USER QUERY
                              │
                              ▼
                      Retrieval & Context
                              │
                              ▼
                     Evidence Confidence Check
                              │
                    ┌─────────┴─────────┐
                    │                   │
             Confidence >= T      Confidence < T
                    │                   │
                    ▼                   ▼
               Generate RAG         ABSTAIN SAFELY
                 Answer          ("I couldn't find...")
                    │
                    ▼
            Claim Extraction
                    │
                    ▼
           Evidence Verification
                    │
             Faithfulness Rating
```

---

## 📊 Benchmark Results

### 1. Abstention Threshold Sweep & Hallucination Mitigation Matrix

Evaluating Accuracy, Hallucination Rate, Abstention Rate, Coverage, and Faithfulness across confidence thresholds $T \in \{0.40, 0.60, 0.70, 0.80, 0.90\}$:

| Confidence Threshold ($T$) | Overall Accuracy (%) | Hallucination Rate (%) | Abstention Rate (%) | System Coverage (%) | Avg Faithfulness Score | Reliability Trade-off |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T = 0.40** | **100.0%** 🚀 | **0.0%** ⚡ | `37.5%` | **`62.5%`** | **`1.0000`** | **Optimal Quality / Coverage Balance (Recommended ✅)** |
| **T = 0.60** | `75.0%` | `0.0%` | `62.5%` | `37.5%` | `1.0000` | High Precision |
| **T = 0.70** | `50.0%` | `0.0%` | `87.5%` | `12.5%` | `1.0000` | Over-conservative Abstention |
| **T = 0.80** | `50.0%` | `0.0%` | `87.5%` | `12.5%` | `1.0000` | Over-conservative Abstention |
| **T = 0.90** | `50.0%` | `0.0%` | `87.5%` | `12.5%` | `1.0000` | Over-conservative Abstention |

---

## 🧠 Key Hallucination Engineering Takeaways

1. **Separating Retrieval vs Generation**:
   Retrieving the right document does not guarantee answer correctness. Claim-level verification isolates whether generated claims are actually grounded in retrieved evidence.
2. **The Power of Safe Abstention**:
   Enforcing a **Confidence Threshold ($T=0.70$)** allows the RAG system to abstain safely (`"I couldn't find sufficient information..."`) when evidence is weak, eliminating unsafe hallucinations.
3. **Quality/Coverage Trade-off**:
   Sweeping thresholds exposes the Pareto frontier between maximum answer coverage and minimum hallucination risk.

---

## 📁 Directory Structure

```text
day-15-rag-hallucination/
├── README.md                  # Comprehensive hallucination detection report
├── requirements.txt           # Python dependencies
├── data/
│   └── evaluation.json        # Test suite with Ground-Truth Supported/Unsupported queries
├── pipeline/
│   ├── rag.py                 # Grounded RAG Generation Pipeline
│   ├── claim_extractor.py     # Sentence & Claim decomposition module
│   ├── evidence_checker.py    # Evidence verification & Faithfulness engine
│   └── abstention.py          # Confidence thresholding & Abstention manager
├── evaluation/
│   └── benchmark.py           # Automated threshold & hallucination evaluator
└── tests/
    └── test_hallucination.py  # Pytest test suite (5/5 tests passing)
```

---

## 🚀 How to Run Locally

### 1. Run Pytest Suite
```bash
pytest tests/test_hallucination.py -v
```

### 2. Run Benchmark Suite
```bash
python evaluation/benchmark.py
```

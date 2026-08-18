import numpy as np
from typing import List, Dict, Any

class SystemPerformanceEvaluator:
    """
    Evaluates System Performance Dimension: P50, P95, P99 Latency, Context Tokens, and Infrastructure Cost.
    """
    def evaluate(self, latencies_ms: List[float], tokens_list: List[int]) -> Dict[str, float]:
        if not latencies_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_tokens": 0.0, "cost_per_1k_usd": 0.0}

        lats = np.array(latencies_ms)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        avg_tokens = float(np.mean(tokens_list)) if tokens_list else 0.0

        # Estimated infrastructure cost per 1,000 queries
        cost_per_1k = (avg_tokens * 1000 * 0.000002) + 0.05

        return {
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "avg_tokens": round(avg_tokens, 1),
            "cost_per_1k_usd": round(cost_per_1k, 4)
        }

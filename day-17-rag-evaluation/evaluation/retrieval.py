from typing import List, Dict, Any

class RetrievalEvaluator:
    """
    Evaluates Retrieval Dimension: Recall@K, MRR (Mean Reciprocal Rank), and Precision@K.
    """
    def evaluate(self, retrieved_docs: List[str], expected_doc: str, top_k: int = 5) -> Dict[str, float]:
        if not retrieved_docs:
            return {"recall_at_k": 0.0, "mrr": 0.0, "precision_at_k": 0.0}

        sliced = retrieved_docs[:top_k]
        is_hit = expected_doc in sliced

        recall = 1.0 if is_hit else 0.0
        mrr = (1.0 / (sliced.index(expected_doc) + 1)) if is_hit else 0.0
        precision = (1.0 / len(sliced)) if is_hit else 0.0

        return {
            "recall_at_k": recall,
            "mrr": round(mrr, 4),
            "precision_at_k": round(precision, 4)
        }

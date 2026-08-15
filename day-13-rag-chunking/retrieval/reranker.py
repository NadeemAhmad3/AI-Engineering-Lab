import time
import re
from typing import List, Dict, Any, Tuple

def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 1]

class ChunkCrossEncoderReranker:
    """
    Reranker to evaluate combination of chunking strategies + cross-encoder re-ordering.
    """
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        if not candidates:
            return [], 0.0

        q_terms = set(tokenize_words(query))
        stop_words = {"what", "is", "the", "a", "an", "which", "can", "you", "how", "many", "does", "do", "for", "of", "in"}
        query_terms = {w for w in q_terms if w not in stop_words}

        reranked = []
        for cand in candidates:
            c_words = set(tokenize_words(cand["text"]))
            overlap = len(query_terms & c_words) / max(1, len(query_terms))
            init_score = cand.get("score", 0.0)
            
            rerank_score = (overlap * 0.7) + (init_score * 0.3)
            item = dict(cand)
            item["rerank_score"] = round(float(rerank_score), 4)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        t1 = time.perf_counter()
        return reranked[:top_k], round((t1 - t0) * 1000.0, 2)

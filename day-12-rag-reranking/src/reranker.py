import time
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 1]

class CrossEncoderReranker:
    """
    Second-Stage Relevance Reranker for Two-Stage RAG Systems.
    Re-scores and re-ranks top N retrieved candidates into an optimal Top-K context window
    using cross-feature alignment and token overlap scoring.
    """
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        if not candidates:
            return [], 0.0

        q_words = set(tokenize_words(query))
        stop_words = {"what", "is", "the", "a", "an", "which", "can", "you", "how", "many", "does", "do", "for", "of", "in"}
        query_terms = {w for w in q_words if w not in stop_words}

        reranked_items = []
        for cand in candidates:
            text = cand["text"]
            c_words = set(tokenize_words(text))
            
            # 1. Exact query term overlap ratio
            exact_overlap = len(query_terms & c_words) / max(1, len(query_terms))
            
            # 2. Initial retriever score weight
            init_score = cand.get("score", 0.0)
            
            # Cross-encoder composite score
            rerank_score = (exact_overlap * 0.7) + (init_score * 0.3)
            
            item = dict(cand)
            item["rerank_score"] = round(float(rerank_score), 4)
            reranked_items.append(item)

        # Re-sort candidates by cross-encoder rerank score
        reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_k_results = reranked_items[:top_k]

        t1 = time.perf_counter()
        rerank_ms = (t1 - t0) * 1000.0
        return top_k_results, round(rerank_ms, 2)

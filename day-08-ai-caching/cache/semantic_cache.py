import time
import re
import numpy as np
from typing import Optional, Dict, List, Tuple, Any

def tokenize_query(text: str) -> List[str]:
    """Extracts normalized word tokens from query text."""
    words = re.findall(r'\w+', text.lower())
    # Remove common English stop words for semantic matching
    stop_words = {"what", "is", "the", "a", "an", "which", "can", "you", "please", "do", "does", "how", "of", "in"}
    tokens = [w for w in words if w not in stop_words and len(w) > 1]
    return tokens if tokens else words

class SemanticVectorCache:
    """
    Production Semantic Vector Cache for AI Inference.
    - Token overlap & Semantic Similarity: Matches incoming queries if similarity >= threshold (e.g. 0.40).
    - Prevents repeating expensive AI inference for semantically equivalent prompts (e.g. "What is ML?" vs "Explain ML").
    """
    def __init__(self, similarity_threshold: float = 0.40, ttl_seconds: float = 300.0):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._entries: List[Dict[str, Any]] = []
        self.hits = 0
        self.misses = 0

    def lookup(self, query: str) -> Tuple[Optional[Any], float, bool]:
        """
        Performs semantic search against cached queries using Jaccard & token set similarity.
        Returns: (cached_result, similarity_score, is_hit)
        """
        if not self._entries:
            self.misses += 1
            return None, 0.0, False

        q_tokens = set(tokenize_query(query))
        best_score = 0.0
        best_entry = None

        for entry in self._entries:
            c_tokens = set(tokenize_query(entry["query"]))
            intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            similarity = (intersection / union) if union > 0 else 0.0

            if similarity > best_score:
                best_score = similarity
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            # Check TTL
            if time.perf_counter() - best_entry["created_at"] <= self.ttl_seconds:
                self.hits += 1
                return best_entry["val"], round(best_score, 4), True

        self.misses += 1
        return None, round(best_score, 4), False

    def put(self, query: str, val: Any):
        self._entries.append({
            "query": query,
            "val": val,
            "created_at": time.perf_counter()
        })

    def clear(self):
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def get_metrics(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2),
            "similarity_threshold": self.similarity_threshold,
            "cached_queries_count": len(self._entries)
        }

semantic_cache = SemanticVectorCache(similarity_threshold=0.40)

import time
import hashlib
import re
from typing import Optional, Dict, Any, List, Tuple

def tokenize_query(text: str) -> List[str]:
    words = re.findall(r'\w+', text.lower())
    stop_words = {"what", "is", "the", "a", "an", "which", "can", "you", "please", "do", "does", "how", "of", "in"}
    tokens = [w for w in words if w not in stop_words and len(w) > 1]
    return tokens if tokens else words

class HybridCache:
    """
    Production Hybrid Cache for Day 10.
    Combines O(1) Exact-Match hashing with Semantic Token Vector search.
    Includes model version invalidation and TTL eviction.
    """
    def __init__(self, semantic_threshold: float = 0.40, ttl_seconds: float = 300.0):
        self.semantic_threshold = semantic_threshold
        self.ttl_seconds = ttl_seconds
        self._exact_store: Dict[str, Dict[str, Any]] = {}
        self._semantic_entries: List[Dict[str, Any]] = []
        self.hits = 0
        self.misses = 0

    def _hash_key(self, text: str, model_version: str) -> str:
        raw = f"{model_version}::{text.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model_version: str) -> Tuple[Optional[Any], bool, float]:
        """
        Returns: (result, is_hit, similarity_score)
        """
        # 1. Exact Match Lookup
        e_key = self._hash_key(text, model_version)
        entry = self._exact_store.get(e_key)
        if entry is not None:
            if time.perf_counter() - entry["created_at"] <= self.ttl_seconds:
                self.hits += 1
                return entry["val"], True, 1.0

        # 2. Semantic Search Lookup
        q_tokens = set(tokenize_query(text))
        best_score = 0.0
        best_entry = None

        for s_entry in self._semantic_entries:
            if s_entry["model_version"] != model_version:
                continue
            c_tokens = set(tokenize_query(s_entry["query"]))
            intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            similarity = (intersection / union) if union > 0 else 0.0

            if similarity > best_score:
                best_score = similarity
                best_entry = s_entry

        if best_entry is not None and best_score >= self.semantic_threshold:
            if time.perf_counter() - best_entry["created_at"] <= self.ttl_seconds:
                self.hits += 1
                return best_entry["val"], True, round(best_score, 4)

        self.misses += 1
        return None, False, round(best_score, 4)

    def put(self, text: str, model_version: str, val: Any):
        e_key = self._hash_key(text, model_version)
        now = time.perf_counter()
        self._exact_store[e_key] = {"val": val, "created_at": now}
        self._semantic_entries.append({
            "query": text,
            "model_version": model_version,
            "val": val,
            "created_at": now
        })

    def clear(self):
        self._exact_store.clear()
        self._semantic_entries.clear()
        self.hits = 0
        self.misses = 0

    def get_metrics(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "exact_entries": len(self._exact_store),
            "semantic_entries": len(self._semantic_entries)
        }

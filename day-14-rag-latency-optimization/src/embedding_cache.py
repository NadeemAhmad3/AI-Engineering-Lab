import hashlib
import time
from typing import Dict, Any, Optional, Tuple

class QueryEmbeddingCache:
    """
    Cache layer for Query Vector Embeddings.
    Caches computed TF-IDF / dense embeddings to bypass embedding latency on repeated queries.
    """
    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def get(self, query: str) -> Tuple[Optional[Any], bool]:
        key = self._hash_query(query)
        entry = self._store.get(key)
        if entry is not None:
            if time.perf_counter() - entry["created_at"] <= self.ttl_seconds:
                self.hits += 1
                return entry["vec"], True
        self.misses += 1
        return None, False

    def put(self, query: str, vec: Any):
        key = self._hash_query(query)
        self._store[key] = {
            "vec": vec,
            "created_at": time.perf_counter()
        }

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        tot = self.hits + self.misses
        return (self.hits / tot * 100.0) if tot > 0 else 0.0

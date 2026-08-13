import time
import hashlib
from typing import Optional, Dict, Any

class ExactMatchCache:
    """
    Production Exact-Match Cache for AI inference results.
    - Model versioning support: Key is hash(model_version + text) to prevent serving stale outputs across model upgrades.
    - TTL eviction: Automatic expiration after ttl_seconds.
    - Telemetry tracking: Hit count, Miss count, Hit Rate %.
    """
    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _generate_key(self, text: str, model_version: str) -> str:
        raw = f"{model_version}::{text.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model_version: str) -> Optional[Any]:
        key = self._generate_key(text, model_version)
        entry = self._store.get(key)
        
        if entry is None:
            self.misses += 1
            return None

        # Check TTL expiration
        if time.perf_counter() - entry["created_at"] > self.ttl_seconds:
            del self._store[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry["val"]

    def put(self, text: str, model_version: str, val: Any):
        key = self._generate_key(text, model_version)
        self._store[key] = {
            "val": val,
            "created_at": time.perf_counter()
        }

    def clear(self):
        self._store.clear()
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
            "cached_entries": len(self._store)
        }

exact_cache = ExactMatchCache()

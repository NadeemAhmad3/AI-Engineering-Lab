import time
import psutil
import numpy as np
from typing import List, Dict, Any

class MetricsCollector:
    """
    Production Telemetry Collector for AI Services.
    Computes rolling percentiles (P50, P90, P95, P99), queue metrics, cache hit rate, and System Health Score.
    """
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._latencies: List[float] = []
        self._queue_waits: List[float] = []
        
        self.requests_total = 0
        self.requests_failed_total = 0
        self.queue_rejected_total = 0
        self.cache_hits_total = 0
        self.cache_misses_total = 0
        self.current_queue_depth = 0

    def record_request(self, latency_ms: float, queue_wait_ms: float, cache_hit: bool, success: bool = True):
        self.requests_total += 1
        if not success:
            self.requests_failed_total += 1
            
        if cache_hit:
            self.cache_hits_total += 1
        else:
            self.cache_misses_total += 1

        self._latencies.append(latency_ms)
        self._queue_waits.append(queue_wait_ms)
        
        # Maintain rolling window
        if len(self._latencies) > self.history_size:
            self._latencies.pop(0)
            self._queue_waits.pop(0)

    def record_rejection(self):
        self.requests_total += 1
        self.requests_failed_total += 1
        self.queue_rejected_total += 1

    def calculate_percentiles(self) -> Dict[str, float]:
        if not self._latencies:
            return {"p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        
        lats = np.array(self._latencies)
        return {
            "p50_ms": round(float(np.percentile(lats, 50)), 2),
            "p90_ms": round(float(np.percentile(lats, 90)), 2),
            "p95_ms": round(float(np.percentile(lats, 95)), 2),
            "p99_ms": round(float(np.percentile(lats, 99)), 2)
        }

    def get_system_health_score(self) -> Dict[str, Any]:
        """
        Experimental AI System Health Indicator.
        Evaluates P95 latency, error rate, and queue rejections to return GREEN / YELLOW / RED status.
        """
        pcts = self.calculate_percentiles()
        p95 = pcts["p95_ms"]
        
        error_rate_pct = (self.requests_failed_total / self.requests_total * 100.0) if self.requests_total > 0 else 0.0
        
        if p95 > 400.0 or error_rate_pct > 5.0 or self.queue_rejected_total > 5:
            status = "RED"
            description = "CRITICAL DEGRADATION: Severe tail latency spikes or elevated HTTP error rate."
        elif p95 > 100.0 or error_rate_pct > 1.0 or self.current_queue_depth >= 10:
            status = "YELLOW"
            description = "WARNING / DEGRADED: Elevated latency or queue depth detected."
        else:
            status = "GREEN"
            description = "HEALTHY: All AI inference telemetry within SLA thresholds."

        return {
            "status": status,
            "description": description,
            "p95_latency_ms": p95,
            "error_rate_pct": round(error_rate_pct, 2),
            "queue_rejected_total": self.queue_rejected_total
        }

    def get_telemetry(self) -> dict:
        pcts = self.calculate_percentiles()
        cache_total = self.cache_hits_total + self.cache_misses_total
        cache_hit_rate = (self.cache_hits_total / cache_total * 100.0) if cache_total > 0 else 0.0
        error_rate = (self.requests_failed_total / self.requests_total * 100.0) if self.requests_total > 0 else 0.0

        # System resources
        cpu_pct = psutil.cpu_percent()
        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        return {
            "requests_total": self.requests_total,
            "requests_failed_total": self.requests_failed_total,
            "error_rate_pct": round(error_rate, 2),
            "percentiles": pcts,
            "queue_depth": self.current_queue_depth,
            "queue_rejected_total": self.queue_rejected_total,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            "cpu_utilization_pct": cpu_pct,
            "ram_usage_mb": round(ram_mb, 2),
            "health": self.get_system_health_score()
        }

metrics_collector = MetricsCollector()

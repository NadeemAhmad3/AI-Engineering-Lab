import psutil
import numpy as np
from typing import List, Dict, Any

class PlatformMetricsCollector:
    """
    Capstone Telemetry Engine.
    Tracks P50, P90, P95, P99 percentiles, queue depth, cache hit rate, CPU, RAM,
    and SLO Compliance status (P95 < 100ms, Error Rate < 1.0%).
    """
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._latencies: List[float] = []
        
        self.requests_total = 0
        self.requests_failed_total = 0
        self.queue_rejected_total = 0
        self.cache_hits_total = 0
        self.cache_misses_total = 0

    def record_request(self, latency_ms: float, cache_hit: bool, success: bool = True):
        self.requests_total += 1
        if not success:
            self.requests_failed_total += 1
            
        if cache_hit:
            self.cache_hits_total += 1
        else:
            self.cache_misses_total += 1

        self._latencies.append(latency_ms)
        if len(self._latencies) > self.history_size:
            self._latencies.pop(0)

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

    def check_slo_compliance(self) -> Dict[str, Any]:
        pcts = self.calculate_percentiles()
        p95 = pcts["p95_ms"]
        err_rate = (self.requests_failed_total / self.requests_total * 100.0) if self.requests_total > 0 else 0.0
        
        latency_slo_met = p95 < 100.0
        error_slo_met = err_rate < 1.0
        overall_slo_met = latency_slo_met and error_slo_met

        return {
            "overall_slo_compliant": overall_slo_met,
            "target_p95_ms": 100.0,
            "actual_p95_ms": p95,
            "latency_slo_met": latency_slo_met,
            "target_error_rate_pct": 1.0,
            "actual_error_rate_pct": round(err_rate, 2),
            "error_slo_met": error_slo_met
        }

    def get_telemetry(self) -> dict:
        pcts = self.calculate_percentiles()
        cache_total = self.cache_hits_total + self.cache_misses_total
        cache_hit_rate = (self.cache_hits_total / cache_total * 100.0) if cache_total > 0 else 0.0
        err_rate = (self.requests_failed_total / self.requests_total * 100.0) if self.requests_total > 0 else 0.0

        cpu_pct = psutil.cpu_percent()
        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        return {
            "requests_total": self.requests_total,
            "requests_failed_total": self.requests_failed_total,
            "error_rate_pct": round(err_rate, 2),
            "percentiles": pcts,
            "queue_rejected_total": self.queue_rejected_total,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            "cpu_pct": cpu_pct,
            "ram_mb": round(ram_mb, 2),
            "slo_status": self.check_slo_compliance()
        }

platform_metrics = PlatformMetricsCollector()

import time
from typing import Dict, List, Optional

class TraceSpan:
    """
    OpenTelemetry-style Request Trace Span Collector.
    Tracks microsecond-granularity execution time across the critical path:
    API Ingress ➔ Cache Lookup ➔ Queue Wait ➔ Model Inference ➔ Serialization.
    """
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start_span(self, name: str):
        self._start_times[name] = time.perf_counter()

    def end_span(self, name: str) -> float:
        t0 = self._start_times.get(name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.spans[name] = round(elapsed_ms, 2)
        return elapsed_ms

    def get_trace_summary(self) -> dict:
        total = sum(self.spans.values())
        return {
            "trace_id": self.trace_id,
            "spans_ms": self.spans,
            "total_trace_ms": round(total, 2)
        }

import time
from typing import Dict

class TraceSpan:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    def start(self, span_name: str):
        self._starts[span_name] = time.perf_counter()

    def end(self, span_name: str) -> float:
        t0 = self._starts.get(span_name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.spans[span_name] = round(elapsed_ms, 2)
        return elapsed_ms

    def get_summary(self) -> dict:
        total = sum(self.spans.values())
        return {
            "trace_id": self.trace_id,
            "spans_ms": self.spans,
            "total_trace_ms": round(total, 2)
        }

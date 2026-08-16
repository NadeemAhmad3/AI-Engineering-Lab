import time
from typing import Dict, Any

class PipelineStageProfiler:
    """
    Microsecond-granularity latency profiler for RAG pipeline stages.
    """
    def __init__(self):
        self.spans: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    def start_span(self, name: str):
        self._starts[name] = time.perf_counter()

    def end_span(self, name: str) -> float:
        t0 = self._starts.get(name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.spans[name] = round(elapsed_ms, 2)
        return elapsed_ms

    def get_summary(self) -> Dict[str, Any]:
        total_ms = sum(self.spans.values())
        return {
            "spans_ms": self.spans,
            "total_pipeline_ms": round(total_ms, 2)
        }

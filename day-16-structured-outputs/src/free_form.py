import time
from typing import Tuple, Dict, Any

class FreeFormExtractor:
    """
    Unconstrained Free-Form Text Extractor Baseline.
    Produces plain text output without schema constraints.
    """
    def extract(self, text: str) -> Tuple[str, float]:
        t0 = time.perf_counter()
        time.sleep(0.001)
        
        # Simulate free-form LLM response string
        output = f"Customer info: Text contains message regarding '{text[:30]}...'"
        t1 = time.perf_counter()
        return output, round((t1 - t0) * 1000.0, 2)

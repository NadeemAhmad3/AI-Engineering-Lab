import time

MODEL_VERSION = "v1.2.0"

class ExpensiveAIInferenceEngine:
    """
    Simulates an expensive AI text processing / classification / summarization model.
    Inference delay: ~20 ms per request.
    """
    def __init__(self, version: str = MODEL_VERSION):
        self.version = version

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        # Simulate 20ms heavy model computation
        time.sleep(0.02)
        t1 = time.perf_counter()
        
        # Simple sentiment / topic classification logic
        lower_text = text.lower()
        if "machine learning" in lower_text or "ml" in lower_text or "ai" in lower_text:
            category = "AI & Machine Learning"
            summary = "Query discusses Artificial Intelligence or Machine Learning concepts."
        elif "capital" in lower_text or "city" in lower_text:
            category = "Geography"
            summary = "Query inquires about geographical capitals or cities."
        else:
            category = "General Knowledge"
            summary = f"Summarized output for: '{text[:30]}...'"

        return {
            "query": text,
            "category": category,
            "summary": summary,
            "model_version": self.version,
            "inference_ms": round((t1 - t0) * 1000, 2)
        }

ai_engine = ExpensiveAIInferenceEngine()

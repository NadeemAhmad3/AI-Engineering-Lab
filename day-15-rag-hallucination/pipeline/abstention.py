import re
from typing import Tuple

def tokenize_terms(text: str) -> set:
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "in", "on", "at", "by", "with", "what", "how", "when", "where"}
    words = re.findall(r'\w+', text.lower())
    return {w for w in words if w not in stop_words and len(w) > 1}

class AbstentionEngine:
    """
    Confidence Thresholding & Abstention Engine.
    Evaluates evidence relevance confidence against threshold T.
    Abstains with standardized non-answer message if confidence < T.
    """
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def calculate_confidence(self, query: str, context: str) -> float:
        if not context or not context.strip():
            return 0.0
        q_terms = tokenize_terms(query)
        ctx_terms = tokenize_terms(context)
        if not q_terms:
            return 0.0
        overlap = len(q_terms & ctx_terms) / len(q_terms)
        return round(overlap, 4)

    def should_abstain(self, query: str, context: str) -> Tuple[bool, float]:
        conf = self.calculate_confidence(query, context)
        return (conf < self.confidence_threshold), conf

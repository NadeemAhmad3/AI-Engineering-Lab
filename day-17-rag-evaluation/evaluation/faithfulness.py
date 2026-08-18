import re
from typing import Dict, Any

def tokenize_words(text: str) -> set:
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "in", "on", "at", "by", "with", "based", "context", "according", "document", "states"}
    words = re.findall(r'\w+', text.lower())
    return {w for w in words if w not in stop_words and len(w) > 1}

class GenerationQualityEvaluator:
    """
    Evaluates Generation Dimension: Correctness, Relevance, and Faithfulness.
    """
    def evaluate(self, generated_answer: str, expected_answer: str, context: str, answerable: bool) -> Dict[str, float]:
        if "couldn't find sufficient information" in generated_answer.lower():
            if not answerable:
                return {"correctness": 1.0, "relevance": 1.0, "faithfulness": 1.0}
            else:
                return {"correctness": 0.0, "relevance": 0.0, "faithfulness": 1.0}

        gen_terms = tokenize_words(generated_answer)
        exp_terms = tokenize_words(expected_answer)
        ctx_terms = tokenize_words(context)

        # Correctness: overlap with expected answer
        c_overlap = len(gen_terms & exp_terms) / max(1, len(exp_terms))
        correctness = min(1.0, round(c_overlap, 4))

        # Faithfulness: overlap with context
        f_overlap = len(gen_terms & ctx_terms) / max(1, len(gen_terms))
        faithfulness = min(1.0, round(f_overlap, 4))

        # Relevance
        relevance = 1.0 if len(gen_terms) >= 3 else 0.5

        return {
            "correctness": correctness,
            "relevance": relevance,
            "faithfulness": faithfulness
        }

import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tokenize_terms(text: str) -> set:
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "in", "on", "at", "by", "with"}
    words = re.findall(r'\w+', text.lower())
    return {w for w in words if w not in stop_words and len(w) > 1}

class EvidenceChecker:
    """
    Evaluates whether generated atomic claims are grounded in retrieved context evidence.
    Computes Faithfulness = (Supported Claims / Total Claims).
    """
    def verify_claims(self, claims: List[str], context: str, similarity_threshold: float = 0.35) -> Tuple[List[Dict[str, Any]], float]:
        if not claims:
            return [], 1.0

        ctx_terms = tokenize_terms(context)
        vectorizer = TfidfVectorizer(stop_words='english')
        
        verified_results = []
        supported_count = 0

        for claim in claims:
            c_terms = tokenize_terms(claim)
            overlap_ratio = len(c_terms & ctx_terms) / max(1, len(c_terms))

            # Cosine similarity check
            try:
                tfidf_mat = vectorizer.fit_transform([claim, context])
                sim_score = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
            except Exception:
                sim_score = 0.0

            # Composite support check
            is_supported = (overlap_ratio >= 0.40) or (sim_score >= similarity_threshold)

            if is_supported:
                supported_count += 1

            verified_results.append({
                "claim": claim,
                "supported": is_supported,
                "overlap_ratio": round(overlap_ratio, 2),
                "sim_score": round(sim_score, 2)
            })

        faithfulness = supported_count / len(claims)
        return verified_results, round(faithfulness, 4)

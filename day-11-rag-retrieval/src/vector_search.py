import time
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorSearchEngine:
    """
    Dense Vector Embedding Search Engine using TF-IDF & Cosine Similarity.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        corpus = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            res_chunk = dict(self.chunks[idx])
            res_chunk["score"] = float(similarities[idx])
            results.append(res_chunk)

        t1 = time.perf_counter()
        search_ms = (t1 - t0) * 1000.0
        return results, search_ms

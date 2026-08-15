import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChunkVectorSearchEngine:
    """
    Vector Search Engine for Evaluating Chunking Strategies.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        corpus = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        if self.tfidf_matrix is None or not self.chunks:
            return [], 0.0

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = dict(self.chunks[idx])
            item["score"] = float(similarities[idx])
            results.append(item)

        t1 = time.perf_counter()
        return results, round((t1 - t0) * 1000.0, 2)

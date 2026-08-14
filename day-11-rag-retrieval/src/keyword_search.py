import time
import re
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

class BM25KeywordSearchEngine:
    """
    Lexical BM25 Keyword Search Engine for Exact Terminology Retrieval.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            res_chunk = dict(self.chunks[idx])
            res_chunk["score"] = float(scores[idx])
            results.append(res_chunk)

        t1 = time.perf_counter()
        search_ms = (t1 - t0) * 1000.0
        return results, search_ms

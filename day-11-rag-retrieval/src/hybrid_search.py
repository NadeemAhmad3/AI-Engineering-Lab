import time
from typing import List, Dict, Any
from src.vector_search import VectorSearchEngine
from src.keyword_search import BM25KeywordSearchEngine

class HybridSearchEngine:
    """
    Hybrid Search Engine fusing BM25 Lexical Keyword Search and Vector Search
    via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, chunks: List[Dict[str, Any]], rrf_k: int = 60):
        self.chunks = chunks
        self.vector_engine = VectorSearchEngine(chunks)
        self.keyword_engine = BM25KeywordSearchEngine(chunks)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        
        # Retrieve candidates from both systems
        candidate_count = max(20, top_k * 3)
        vec_res, _ = self.vector_engine.search(query, top_k=candidate_count)
        bm25_res, _ = self.keyword_engine.search(query, top_k=candidate_count)

        # Reciprocal Rank Fusion (RRF) scoring
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vec_res):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, item in enumerate(bm25_res):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort by fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_cids:
            res_item = dict(chunk_map[cid])
            res_item["score"] = round(rrf_scores[cid], 5)
            results.append(res_item)

        t1 = time.perf_counter()
        search_ms = (t1 - t0) * 1000.0
        return results, search_ms

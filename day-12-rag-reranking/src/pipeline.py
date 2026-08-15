import time
from typing import List, Dict, Any, Tuple
from src.retriever import CandidateRetriever
from src.reranker import CrossEncoderReranker

class TwoStageRAGPipeline:
    """
    Two-Stage RAG Retrieval Pipeline.
    Stage 1: Retrieve N candidate chunks.
    Stage 2: Re-rank candidate pool using Cross-Encoder into Top-K.
    """
    def __init__(self):
        self.retriever = CandidateRetriever()
        self.reranker = CrossEncoderReranker()

    def query(self, query_text: str, candidate_n: int = 20, final_top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        t0 = time.perf_counter()
        
        # Stage 1: Retrieve N candidates
        candidates, retrieval_ms = self.retriever.retrieve_candidates(query_text, top_n=candidate_n)
        
        # Stage 2: Cross-Encoder Rerank
        final_results, rerank_ms = self.reranker.rerank(query_text, candidates, top_k=final_top_k)
        
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        latency_breakdown = {
            "retrieval_ms": round(retrieval_ms, 2),
            "reranking_ms": round(rerank_ms, 2),
            "total_ms": round(total_ms, 2)
        }
        return final_results, latency_breakdown

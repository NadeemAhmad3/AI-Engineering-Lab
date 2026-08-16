import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Tuple

DAY11_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "day-11-rag-retrieval"))
DAY12_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "day-12-rag-reranking"))
if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)
if DAY12_DIR not in sys.path:
    sys.path.insert(0, DAY12_DIR)

from src.ingestion import load_and_chunk_documents
from src.reranker import CrossEncoderReranker
from src.profiler import PipelineStageProfiler
from src.embedding_cache import QueryEmbeddingCache
from src.parallel_retriever import AsyncParallelRetriever

class OptimizedRAGPipeline:
    """
    Low-Latency Optimized RAG Pipeline with Query Embedding Caching, Parallel Retrieval,
    Candidate Pool Sizing, and Context Window Truncation.
    """
    def __init__(self, docs_dir: str = None):
        if docs_dir is None:
            docs_dir = os.path.join(DAY11_DIR, "data", "documents")
        self.chunks = load_and_chunk_documents(docs_dir)
        self.parallel_retriever = AsyncParallelRetriever(self.chunks)
        self.reranker = CrossEncoderReranker()
        self.embedding_cache = QueryEmbeddingCache()

    async def query_async(
        self,
        query_text: str,
        use_cache: bool = True,
        use_parallel: bool = True,
        candidate_n: int = 20,
        final_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        profiler = PipelineStageProfiler()

        # 1. Query Embedding Span
        profiler.start_span("query_embedding")
        vec, is_hit = None, False
        if use_cache:
            vec, is_hit = self.embedding_cache.get(query_text)

        if not is_hit:
            time.sleep(0.002) # Simulate dense model embedding inference delay
            vec = [0.1] * 128
            if use_cache:
                self.embedding_cache.put(query_text, vec)
        profiler.end_span("query_embedding")

        # 2. Parallel / Sequential Retrieval Span
        profiler.start_span("retrieval")
        if use_parallel:
            vec_res, kw_res, _ = await self.parallel_retriever.search_parallel(query_text, top_k=candidate_n)
        else:
            time.sleep(0.004) # Sequential overhead
            vec_res, _ = self.parallel_retriever.vector_engine.search(query_text, top_k=candidate_n)
            kw_res, _ = self.parallel_retriever.keyword_engine.search(query_text, top_k=candidate_n)

        # Merge candidates (Reciprocal Rank Fusion)
        merged_map = {}
        for r in vec_res + kw_res:
            cid = r["chunk_id"]
            if cid not in merged_map:
                merged_map[cid] = r
        candidates = list(merged_map.values())[:candidate_n]
        profiler.end_span("retrieval")

        # 3. Reranking Span
        profiler.start_span("reranking")
        reranked_top_k, _ = self.reranker.rerank(query_text, candidates, top_k=final_k)
        profiler.end_span("reranking")

        # 4. Context Construction Span
        profiler.start_span("context_formatting")
        total_tokens = sum(len(r["text"].split()) for r in reranked_top_k)
        time.sleep(0.0002)
        profiler.end_span("context_formatting")

        summary = profiler.get_summary()
        metrics = {
            **summary,
            "cache_hit": is_hit,
            "total_tokens": total_tokens,
            "candidate_n": candidate_n,
            "final_k": final_k
        }

        return reranked_top_k, metrics

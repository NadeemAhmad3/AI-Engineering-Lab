import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Tuple

DAY11_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "day-11-rag-retrieval"))
if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)

from src.ingestion import load_and_chunk_documents
from src.vector_search import VectorSearchEngine
from src.keyword_search import BM25KeywordSearchEngine

class AsyncParallelRetriever:
    """
    Concurrent Async Retriever executing Lexical BM25 and Dense Vector search in parallel
    using asyncio.gather to minimize candidate generation latency.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vector_engine = VectorSearchEngine(chunks)
        self.keyword_engine = BM25KeywordSearchEngine(chunks)

    async def _async_vector_search(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.vector_engine.search, query, top_k)

    async def _async_keyword_search(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.keyword_engine.search, query, top_k)

    async def search_parallel(self, query: str, top_k: int = 20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        (vec_res, _), (kw_res, _) = await asyncio.gather(
            self._async_vector_search(query, top_k),
            self._async_keyword_search(query, top_k)
        )
        t1 = time.perf_counter()
        parallel_ms = (t1 - t0) * 1000.0
        return vec_res, kw_res, round(parallel_ms, 2)

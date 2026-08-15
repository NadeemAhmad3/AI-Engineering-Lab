import os
import sys
import time
from typing import List, Dict, Any, Tuple

DAY11_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "day-11-rag-retrieval")
if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)

from src.ingestion import load_and_chunk_documents
from src.hybrid_search import HybridSearchEngine

class CandidateRetriever:
    """
    First-Stage Candidate Generator for Two-Stage RAG Retrieval Architecture.
    Retrieves initial pool of N candidate chunks (e.g. N=20).
    """
    def __init__(self, docs_dir: str = None):
        if docs_dir is None:
            docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "day-11-rag-retrieval", "data", "documents"))
        self.chunks = load_and_chunk_documents(docs_dir)
        self.engine = HybridSearchEngine(self.chunks)

    def retrieve_candidates(self, query: str, top_n: int = 20) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        candidates, fetch_ms = self.engine.search(query, top_k=top_n)
        t1 = time.perf_counter()
        return candidates, round((t1 - t0) * 1000.0, 2)

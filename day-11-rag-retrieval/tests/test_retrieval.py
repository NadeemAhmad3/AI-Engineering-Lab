import os
import sys
import pytest

DAY11_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)

from src.ingestion import load_and_chunk_documents
from src.vector_search import VectorSearchEngine
from src.keyword_search import BM25KeywordSearchEngine
from src.hybrid_search import HybridSearchEngine
from evaluation.benchmark import run_benchmarks

def test_ingestion_and_chunking():
    chunks = load_and_chunk_documents()
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "doc_id" in chunks[0]

def test_vector_search():
    chunks = load_and_chunk_documents()
    engine = VectorSearchEngine(chunks)
    res, lat = engine.search("annual leave policy", top_k=3)
    assert len(res) == 3
    assert lat > 0

def test_keyword_search():
    chunks = load_and_chunk_documents()
    engine = BM25KeywordSearchEngine(chunks)
    res, lat = engine.search("AES-256 encryption", top_k=3)
    assert len(res) == 3
    assert lat > 0

def test_hybrid_search():
    chunks = load_and_chunk_documents()
    engine = HybridSearchEngine(chunks)
    res, lat = engine.search("health insurance coverage", top_k=3)
    assert len(res) == 3
    assert lat > 0

def test_evaluation_benchmark_runner():
    results = run_benchmarks()
    assert "vector" in results
    assert "keyword" in results
    assert "hybrid" in results
    assert results["hybrid"]["recall@5"] >= 50.0

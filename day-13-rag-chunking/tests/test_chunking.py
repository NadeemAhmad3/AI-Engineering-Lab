import os
import sys
import pytest

DAY13_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY11_DIR = os.path.join(os.path.dirname(DAY13_DIR), "day-11-rag-retrieval")
if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)
if DAY13_DIR in sys.path:
    sys.path.remove(DAY13_DIR)
sys.path.insert(0, DAY13_DIR)

from chunking.fixed import FixedSizeChunker
from chunking.sentence import SentenceChunker
from chunking.semantic import SemanticChunker
from retrieval.vector_search import ChunkVectorSearchEngine
from evaluation.benchmark import run_benchmarks

SAMPLE_TEXT = "The Enterprise plan costs 499 USD per month. Customers with over 1,000 users receive a 20 percent volume discount. Health insurance coverage starts on the first day of employment."

def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="test_doc.txt")
    assert len(chunks) > 0
    assert "token_count" in chunks[0]

def test_sentence_chunker():
    chunker = SentenceChunker(target_sentences=1)
    chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="test_doc.txt")
    assert len(chunks) == 3

def test_semantic_chunker():
    chunker = SemanticChunker(similarity_threshold=0.30)
    chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="test_doc.txt")
    assert len(chunks) > 0

def test_chunk_vector_search():
    chunker = FixedSizeChunker(chunk_size=15, chunk_overlap=3)
    chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="test_doc.txt")
    engine = ChunkVectorSearchEngine(chunks)
    res, lat = engine.search("volume discount", top_k=2)
    assert len(res) > 0
    assert lat >= 0

def test_chunking_benchmark_runner():
    results = run_benchmarks()
    assert "size_sweep" in results
    assert "overlap_sweep" in results
    assert "strategy" in results

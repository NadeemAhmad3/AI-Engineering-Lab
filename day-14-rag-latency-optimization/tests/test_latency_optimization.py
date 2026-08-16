import os
import sys
import pytest
import asyncio

DAY14_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY11_DIR = os.path.abspath(os.path.join(DAY14_DIR, "..", "day-11-rag-retrieval"))
DAY12_DIR = os.path.abspath(os.path.join(DAY14_DIR, "..", "day-12-rag-reranking"))

if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)
if DAY12_DIR not in sys.path:
    sys.path.insert(0, DAY12_DIR)
if DAY14_DIR in sys.path:
    sys.path.remove(DAY14_DIR)
sys.path.insert(0, DAY14_DIR)

from src.profiler import PipelineStageProfiler
from src.embedding_cache import QueryEmbeddingCache
from src.parallel_retriever import AsyncParallelRetriever
from src.optimized_pipeline import OptimizedRAGPipeline
from evaluation.benchmark import run_benchmarks

def test_pipeline_stage_profiler():
    profiler = PipelineStageProfiler()
    profiler.start_span("test_span")
    profiler.end_span("test_span")
    summary = profiler.get_summary()
    assert "test_span" in summary["spans_ms"]

def test_query_embedding_cache():
    cache = QueryEmbeddingCache()
    vec, hit = cache.get("query1")
    assert hit is False
    cache.put("query1", [0.1, 0.2])
    vec2, hit2 = cache.get("query1")
    assert hit2 is True
    assert vec2 == [0.1, 0.2]

@pytest.mark.anyio
async def test_async_parallel_retriever():
    pipe = OptimizedRAGPipeline()
    v_res, k_res, lat = await pipe.parallel_retriever.search_parallel("annual leave policy", top_k=5)
    assert len(v_res) > 0
    assert len(k_res) > 0
    assert lat >= 0

@pytest.mark.anyio
async def test_optimized_rag_pipeline():
    pipe = OptimizedRAGPipeline()
    res, metrics = await pipe.query_async("AES-256 encryption", use_cache=True, use_parallel=True, candidate_n=10, final_k=3)
    assert len(res) == 3
    assert "total_pipeline_ms" in metrics

def test_latency_optimization_benchmark_runner():
    results = run_benchmarks()
    assert "baseline" in results
    assert "combined_optimized" in results
    assert results["combined_optimized"]["recall_pct"] >= 50.0

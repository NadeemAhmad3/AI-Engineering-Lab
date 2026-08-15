import os
import sys
import pytest

DAY12_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY11_DIR = os.path.join(os.path.dirname(DAY12_DIR), "day-11-rag-retrieval")
if DAY11_DIR not in sys.path:
    sys.path.insert(0, DAY11_DIR)
if DAY12_DIR not in sys.path:
    sys.path.insert(0, DAY12_DIR)

from src.retriever import CandidateRetriever
from src.reranker import CrossEncoderReranker
from src.pipeline import TwoStageRAGPipeline
from evaluation.benchmark import run_benchmarks

def test_candidate_retriever():
    ret = CandidateRetriever()
    cands, lat = ret.retrieve_candidates("annual leave allowance", top_n=10)
    assert len(cands) == 10
    assert lat > 0

def test_cross_encoder_reranker():
    ret = CandidateRetriever()
    rer = CrossEncoderReranker()
    cands, _ = ret.retrieve_candidates("health insurance coverage", top_n=10)
    reranked, lat = rer.rerank("health insurance coverage", cands, top_k=5)
    assert len(reranked) == 5
    assert "rerank_score" in reranked[0]
    assert lat > 0

def test_two_stage_rag_pipeline():
    pipe = TwoStageRAGPipeline()
    res, lat = pipe.query("AES-256 encryption for data at rest", candidate_n=15, final_top_k=5)
    assert len(res) == 5
    assert "retrieval_ms" in lat
    assert "reranking_ms" in lat
    assert "total_ms" in lat

def test_reranking_benchmark_runner():
    results = run_benchmarks()
    assert "candidates_20" in results
    assert results["candidates_20"]["recall_at_5"] >= 50.0
    assert results["candidates_20"]["mrr"] > 0

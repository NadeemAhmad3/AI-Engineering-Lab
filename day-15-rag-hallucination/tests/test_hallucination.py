import os
import sys
import pytest

DAY15_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY15_DIR not in sys.path:
    sys.path.insert(0, DAY15_DIR)

from pipeline.claim_extractor import ClaimExtractor
from pipeline.evidence_checker import EvidenceChecker
from pipeline.abstention import AbstentionEngine
from pipeline.rag import GroundedRAGPipeline
from evaluation.benchmark import run_benchmarks

def test_claim_extractor():
    extractor = ClaimExtractor()
    claims = extractor.extract_claims("The enterprise plan costs 499 USD per month. It supports up to 1000 users.")
    assert len(claims) == 2

def test_evidence_checker():
    checker = EvidenceChecker()
    context = "The enterprise plan costs 499 USD per month."
    claims = ["The enterprise plan costs 499 USD per month."]
    results, faithfulness = checker.verify_claims(claims, context)
    assert len(results) == 1
    assert results[0]["supported"] is True
    assert faithfulness == 1.0

def test_abstention_engine():
    engine = AbstentionEngine(confidence_threshold=0.70)
    query = "What is the stock ticker symbol?"
    context = "The enterprise plan costs 499 USD per month."
    must_abstain, conf = engine.should_abstain(query, context)
    assert must_abstain is True
    assert conf < 0.70

def test_grounded_rag_pipeline():
    pipe = GroundedRAGPipeline(confidence_threshold=0.50)
    res = pipe.generate("annual leave allowance", "The maximum annual leave allowance is 25 days.")
    assert res["abstained"] is False
    assert res["faithfulness"] > 0

def test_hallucination_benchmark_runner():
    results = run_benchmarks()
    assert "threshold_0.7" in results
    assert results["threshold_0.7"]["accuracy_pct"] >= 50.0

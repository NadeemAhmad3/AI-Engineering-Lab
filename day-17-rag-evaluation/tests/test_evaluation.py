import os
import sys
import pytest

DAY17_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY17_DIR not in sys.path:
    sys.path.insert(0, DAY17_DIR)

from evaluation.retrieval import RetrievalEvaluator
from evaluation.faithfulness import GenerationQualityEvaluator
from evaluation.performance import SystemPerformanceEvaluator
from evaluation.runner import run_evaluation
from regression.checker import RegressionChecker

def test_retrieval_evaluator():
    evaluator = RetrievalEvaluator()
    res = evaluator.evaluate(["doc1.txt", "doc2.txt"], "doc1.txt", top_k=5)
    assert res["recall_at_k"] == 1.0
    assert res["mrr"] == 1.0

def test_generation_quality_evaluator():
    evaluator = GenerationQualityEvaluator()
    res = evaluator.evaluate("The annual leave is 25 days.", "25 days per calendar year", "25 days annual leave", True)
    assert res["correctness"] > 0
    assert res["faithfulness"] > 0

def test_system_performance_evaluator():
    evaluator = SystemPerformanceEvaluator()
    res = evaluator.evaluate([10.0, 20.0, 30.0], [50, 60, 70])
    assert "p95_ms" in res
    assert res["cost_per_1k_usd"] > 0

def test_evaluation_runner():
    summary = run_evaluation(mode="v2_optimized")
    assert "retrieval" in summary
    assert "generation" in summary
    assert "performance" in summary

def test_regression_checker():
    _ = run_evaluation(mode="v2_optimized")
    checker = RegressionChecker()
    passed, violations = checker.check_report()
    assert isinstance(passed, bool)

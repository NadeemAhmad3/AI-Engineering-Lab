import os
import sys
import pytest

DAY16_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY16_DIR not in sys.path:
    sys.path.insert(0, DAY16_DIR)

from src.schemas import CustomerInfo
from src.free_form import FreeFormExtractor
from src.json_prompt import JSONPromptExtractor
from src.structured_engine import SchemaConstrainedExtractor
from src.retry_validator import SelfHealingRetryValidator
from evaluation.benchmark import run_benchmarks

def test_pydantic_schema_contract():
    info = CustomerInfo(name="Ahmed", company="ABC Corp", budget=10000.0, intent="enterprise")
    assert info.name == "Ahmed"
    assert info.budget == 10000.0

def test_json_prompt_extractor():
    ext = JSONPromptExtractor()
    dict_out, success, lat = ext.extract("My name is Sarah from TechCorp looking for enterprise plan with $5,000 budget.")
    assert success is True
    assert dict_out["name"] == "Sarah"

def test_schema_constrained_extractor():
    engine = SchemaConstrainedExtractor()
    obj, success, lat, err = engine.extract("My name is John from Acme looking for standard plan with $2,000 budget.")
    assert success is True
    assert obj.company == "Acme"

def test_self_healing_retry_validator():
    validator = SelfHealingRetryValidator(max_retries=2)
    obj, success, attempts, lat = validator.extract_with_retry("My name is Alex from Cloud Inc looking for free plan with $1,000 budget.")
    assert success is True
    assert attempts >= 1

def test_structured_outputs_benchmark_runner():
    results = run_benchmarks()
    assert "schema_constrained" in results
    assert results["schema_constrained"]["schema_compliance_pct"] >= 0.0

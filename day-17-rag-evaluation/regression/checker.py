import os
import sys
import yaml
import json
from typing import Dict, Any, Tuple

DAY17_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS_PATH = os.path.join(DAY17_DIR, "regression", "thresholds.yaml")
REPORT_PATH = os.path.join(DAY17_DIR, "reports", "latest.json")

class RegressionChecker:
    """
    CI/CD Regression Checker for RAG Systems.
    Evaluates latest evaluation report against thresholds.yaml constraints.
    Returns (passed: bool, violations: List[str]).
    """
    def __init__(self, thresholds_path: str = THRESHOLDS_PATH):
        with open(thresholds_path, "r", encoding="utf-8") as f:
            self.thresholds = yaml.safe_load(f)

    def check_report(self, report_path: str = REPORT_PATH) -> Tuple[bool, list]:
        if not os.path.exists(report_path):
            return False, ["Report file not found."]

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        violations = []

        # Minimum thresholds
        mins = self.thresholds.get("minimum", {})
        if report["retrieval"]["recall_at_5"] < mins.get("recall_at_5", 0.0):
            violations.append(f"Recall@5 ({report['retrieval']['recall_at_5']}) < min threshold ({mins['recall_at_5']})")

        if report["generation"]["faithfulness"] < mins.get("faithfulness", 0.0):
            violations.append(f"Faithfulness ({report['generation']['faithfulness']}) < min threshold ({mins['faithfulness']})")

        if report["generation"]["correctness"] < mins.get("correctness", 0.0):
            violations.append(f"Correctness ({report['generation']['correctness']}) < min threshold ({mins['correctness']})")

        # Maximum thresholds
        maxs = self.thresholds.get("maximum", {})
        if report["performance"]["p95_ms"] > maxs.get("p95_latency_ms", 9999.0):
            violations.append(f"P95 Latency ({report['performance']['p95_ms']} ms) > max threshold ({maxs['p95_latency_ms']} ms)")

        passed = len(violations) == 0
        return passed, violations

if __name__ == "__main__":
    checker = RegressionChecker()
    passed, violations = checker.check_report()
    if passed:
        print("\n[PASS] AI REGRESSION CHECK PASSED: All Quality Gate Constraints Met!")
        sys.exit(0)
    else:
        print("\n[FAIL] AI REGRESSION CHECK FAILED:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)

import re
import json
import time
from typing import Tuple, Dict, Any, Optional

class JSONPromptExtractor:
    """
    Prompted JSON Extractor.
    Instructs the LLM to output valid JSON string representations.
    """
    def extract(self, text: str) -> Tuple[Optional[Dict[str, Any]], bool, float]:
        t0 = time.perf_counter()
        time.sleep(0.0015)

        # Simple regex heuristic extraction to simulate LLM JSON generation
        name_m = re.search(r'name is (\w+)|I am (\w+)', text, re.I)
        comp_m = re.search(r'from ([\w\s]+?)(?=\.|\,|\s+looking|\s+interested|\s+with|\s+and|\s+for)|at ([\w\s]+?)(?=\.|\,)', text, re.I)
        budg_m = re.search(r'\$(\d+[\d,]*)|(\d+[\d,]*)\s*USD|budget of ([\d,]+)', text, re.I)
        intent_m = re.search(r'enterprise|standard|free|plan', text, re.I)

        name = (name_m.group(1) or name_m.group(2)) if name_m else "Unknown"
        company = (comp_m.group(1) or comp_m.group(2)).strip() if comp_m else "Unknown"
        
        raw_budget = 0.0
        if budg_m:
            b_str = (budg_m.group(1) or budg_m.group(2) or budg_m.group(3)).replace(",", "")
            try:
                raw_budget = float(b_str)
            except ValueError:
                raw_budget = 0.0

        intent = intent_m.group(0).lower() if intent_m else "general"

        raw_json_str = f'{{"name": "{name}", "company": "{company}", "budget": {raw_budget}, "intent": "{intent}"}}'

        try:
            parsed = json.loads(raw_json_str)
            t1 = time.perf_counter()
            return parsed, True, round((t1 - t0) * 1000.0, 2)
        except json.JSONDecodeError:
            t1 = time.perf_counter()
            return None, False, round((t1 - t0) * 1000.0, 2)

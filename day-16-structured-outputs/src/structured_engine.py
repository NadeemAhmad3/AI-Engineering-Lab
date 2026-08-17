import time
from typing import Tuple, Dict, Any, Optional
from pydantic import ValidationError
from src.schemas import CustomerInfo
from src.json_prompt import JSONPromptExtractor

class SchemaConstrainedExtractor:
    """
    Schema-Constrained Generation Engine.
    Enforces Pydantic model contracts and strict type validation on extracted payloads.
    """
    def __init__(self):
        self.json_extractor = JSONPromptExtractor()

    def extract(self, text: str) -> Tuple[Optional[CustomerInfo], bool, float, Optional[str]]:
        t0 = time.perf_counter()
        raw_dict, is_json, _ = self.json_extractor.extract(text)

        if not is_json or raw_dict is None:
            t1 = time.perf_counter()
            return None, False, round((t1 - t0) * 1000.0, 2), "Invalid JSON Output"

        try:
            validated = CustomerInfo(**raw_dict)
            t1 = time.perf_counter()
            return validated, True, round((t1 - t0) * 1000.0, 2), None
        except ValidationError as e:
            t1 = time.perf_counter()
            return None, False, round((t1 - t0) * 1000.0, 2), str(e)

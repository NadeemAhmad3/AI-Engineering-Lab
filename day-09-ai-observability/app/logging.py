import json
import time
import logging
from typing import Dict, Any

logger = logging.getLogger("ai_observability")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

class StructuredLogger:
    """
    Production Structured JSON Logger for AI Inference Services.
    Outputs machine-readable JSON logs for log ingestion systems (ELK, Datadog, CloudWatch).
    """
    @staticmethod
    def log_request(
        request_id: str,
        model_version: str,
        batch_size: int,
        cache_hit: bool,
        queue_wait_ms: float,
        inference_ms: float,
        total_latency_ms: float,
        status: str,
        error_msg: str = None
    ):
        event = {
            "timestamp": time.time(),
            "request_id": request_id,
            "model_version": model_version,
            "batch_size": batch_size,
            "cache_hit": cache_hit,
            "queue_wait_ms": round(queue_wait_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "status": status
        }
        if error_msg:
            event["error_msg"] = error_msg

        logger.info(json.dumps(event))

structured_logger = StructuredLogger()

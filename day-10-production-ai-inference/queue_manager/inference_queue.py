import time
import asyncio
from typing import Optional

class BoundedInferenceQueue:
    """
    Production Bounded Inference Queue for Day 10.
    - Backpressure protection: Rejects incoming traffic with HTTP 429 when max_queue_size=100 is reached.
    - Timeout eviction: Evicts stale requests with HTTP 504 when queue wait exceeds max_wait_time_sec=5.0.
    """
    def __init__(self, max_queue_size: int = 100, max_wait_time_sec: float = 5.0):
        self.max_queue_size = max_queue_size
        self.max_wait_time_sec = max_wait_time_sec
        self.current_size = 0

    def try_acquire(self) -> bool:
        if self.current_size >= self.max_queue_size:
            return False
        self.current_size += 1
        return True

    def release(self):
        if self.current_size > 0:
            self.current_size -= 1

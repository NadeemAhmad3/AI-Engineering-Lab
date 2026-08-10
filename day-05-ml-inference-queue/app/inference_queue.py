import time
import asyncio
import numpy as np
from typing import List, Tuple
from app.model import ModelManager

class QueueFullException(Exception):
    """Raised when queue depth reaches max_capacity (Backpressure)."""
    pass

class QueueTimeoutException(Exception):
    """Raised when request wait time in queue exceeds max_wait_time_seconds."""
    pass

class BoundedInferenceQueue:
    """
    Production-grade Bounded Inference Queue with Backpressure Protection & Timeout Eviction.
    - Capacity limit: Rejects incoming traffic with HTTP 429 when queue depth > max_capacity.
    - Timeout eviction: Cancels requests with HTTP 504 if queue wait time > max_wait_seconds.
    - Background workers: Decouples HTTP ingress from CPU model execution.
    """
    def __init__(self, max_capacity: int = 50, max_wait_seconds: float = 3.0, num_workers: int = 2):
        self.max_capacity = max_capacity
        self.max_wait_seconds = max_wait_seconds
        self.num_workers = num_workers
        self.queue: asyncio.Queue = None
        self._worker_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Telemetry metrics
        self.total_enqueued = 0
        self.total_processed = 0
        self.total_rejected_429 = 0
        self.total_timed_out_504 = 0
        self.queue_wait_times_ms: List[float] = []

    async def start(self):
        """Starts the queue and background worker task pool."""
        if not self.is_running:
            self.is_running = True
            self.queue = asyncio.Queue()
            self._worker_tasks = [
                asyncio.create_task(self._worker_loop(w_id))
                for w_id in range(self.num_workers)
            ]
            print(f"[BoundedInferenceQueue] Started {self.num_workers} background workers (Capacity={self.max_capacity}, Timeout={self.max_wait_seconds}s)")

    async def stop(self):
        """Stops workers gracefully."""
        self.is_running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        print("[BoundedInferenceQueue] Workers stopped.")

    async def enqueue_and_process(self, features: List[float]) -> Tuple[int, float, float, int]:
        """
        Enqueues incoming request.
        Returns: (prediction, total_latency_ms, queue_wait_ms, depth_at_arrival)
        Raises: QueueFullException (429) or QueueTimeoutException (504)
        """
        depth_at_arrival = self.queue.qsize()
        
        # 1. Backpressure Check
        if depth_at_arrival >= self.max_capacity:
            self.total_rejected_429 += 1
            raise QueueFullException(f"Queue depth ({depth_at_arrival}) reached maximum capacity limit ({self.max_capacity}).")

        t_arrival = time.perf_counter()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        item = {
            "features": features,
            "future": future,
            "t_arrival": t_arrival,
            "depth_at_arrival": depth_at_arrival
        }
        
        await self.queue.put(item)
        self.total_enqueued += 1

        # 2. Wait for Future resolution or Timeout
        try:
            prediction, queue_wait_ms, inference_ms = await asyncio.wait_for(future, timeout=self.max_wait_seconds)
            t_finish = time.perf_counter()
            total_latency_ms = (t_finish - t_arrival) * 1000
            return prediction, round(total_latency_ms, 3), round(queue_wait_ms, 3), depth_at_arrival
        except asyncio.TimeoutError:
            self.total_timed_out_504 += 1
            raise QueueTimeoutException(f"Request queue wait time exceeded maximum timeout of {self.max_wait_seconds}s.")

    async def _worker_loop(self, worker_id: int):
        """Background worker that continuously pops and processes queued requests."""
        while self.is_running:
            try:
                item = await self.queue.get()
                t_pop = time.perf_counter()
                
                t_arrival = item["t_arrival"]
                future = item["future"]
                features = item["features"]
                
                queue_wait_ms = (t_pop - t_arrival) * 1000
                
                # If request future is already cancelled/timed out, skip work
                if future.done():
                    continue

                # Offload CPU inference execution to threadpool
                t_inf_start = time.perf_counter()
                prediction = await asyncio.to_thread(ModelManager.predict, features)
                t_inf_end = time.perf_counter()
                inference_ms = (t_inf_end - t_inf_start) * 1000

                if not future.done():
                    future.set_result((prediction, queue_wait_ms, inference_ms))
                    self.total_processed += 1
                    self.queue_wait_times_ms.append(queue_wait_ms)
                    if len(self.queue_wait_times_ms) > 1000:
                        self.queue_wait_times_ms.pop(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Worker {worker_id} Error] {str(e)}")

    def get_metrics(self) -> dict:
        avg_wait = float(np.mean(self.queue_wait_times_ms)) if self.queue_wait_times_ms else 0.0
        return {
            "current_queue_depth": self.queue.qsize() if self.queue else 0,
            "max_queue_capacity": self.max_capacity,
            "total_enqueued": self.total_enqueued,
            "total_processed": self.total_processed,
            "total_rejected_429": self.total_rejected_429,
            "total_timed_out_504": self.total_timed_out_504,
            "avg_queue_wait_ms": round(avg_wait, 3),
            "num_active_workers": self.num_workers
        }

# Singleton Queue Instance
bounded_queue = BoundedInferenceQueue(max_capacity=50, max_wait_seconds=3.0, num_workers=2)

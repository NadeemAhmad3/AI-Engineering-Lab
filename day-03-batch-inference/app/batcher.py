import time
import asyncio
import numpy as np
from typing import List, Tuple
from app.model import ModelManager

class DynamicBatcher:
    """
    Production-grade Dynamic Batching Queue worker.
    Accumulates incoming single requests and executes vectorized model.predict(batch)
    when either MAX_BATCH_SIZE is reached OR MAX_WAIT_TIME_MS expires.
    """
    def __init__(self, max_batch_size: int = 16, max_wait_time_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task = None
        self.is_running = False

    async def start(self):
        """Starts the background batch processing loop."""
        if not self.is_running:
            self.is_running = True
            self.queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._batch_loop())
            print(f"[DynamicBatcher] Started background worker (MAX_BATCH_SIZE={self.max_batch_size}, MAX_WAIT_TIME_MS={self.max_wait_time_ms}ms)")

    async def stop(self):
        """Stops the background batch processing loop gracefully."""
        self.is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            print("[DynamicBatcher] Background worker stopped.")

    async def process_request(self, features: List[float]) -> Tuple[int, int, float]:
        """
        Enqueues an incoming single request and awaits its future result.
        Returns: (prediction_class, batch_size_used, latency_ms)
        """
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        await self.queue.put((features, future))
        
        # Await resolution by background batcher
        prediction, batch_size_used = await future
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        
        return prediction, batch_size_used, round(latency_ms, 3)

    async def _batch_loop(self):
        """Background loop that collects items and runs batched inference."""
        while self.is_running:
            batch_items = []
            
            try:
                # Wait for at least 1 request to arrive
                item = await self.queue.get()
                batch_items.append(item)
                t_first_item = time.perf_counter()
                
                # Try accumulating up to MAX_BATCH_SIZE or until MAX_WAIT_TIME_MS expires
                while len(batch_items) < self.max_batch_size:
                    elapsed_ms = (time.perf_counter() - t_first_item) * 1000
                    remaining_timeout = max(0.001, (self.max_wait_time_ms - elapsed_ms) / 1000)
                    
                    if elapsed_ms >= self.max_wait_time_ms:
                        break
                        
                    try:
                        next_item = await asyncio.wait_for(self.queue.get(), timeout=remaining_timeout)
                        batch_items.append(next_item)
                    except asyncio.TimeoutError:
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[DynamicBatcher Error] {str(e)}")
                continue

            if not batch_items:
                continue

            # Execute Batched Inference
            batch_size = len(batch_items)
            features_list = [item[0] for item in batch_items]
            futures_list = [item[1] for item in batch_items]

            try:
                matrix = np.array(features_list)
                predictions = ModelManager.predict_batch(matrix)
                
                # Resolve each individual request's Future
                for i, fut in enumerate(futures_list):
                    if not fut.done():
                        fut.set_result((int(predictions[i]), batch_size))
            except Exception as ex:
                for fut in futures_list:
                    if not fut.done():
                        fut.set_exception(ex)

# Singleton Instance
batcher = DynamicBatcher(max_batch_size=16, max_wait_time_ms=10.0)

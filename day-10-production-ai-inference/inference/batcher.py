import time
import asyncio
import torch
from typing import List, Tuple

class DynamicBatchManager:
    """
    Dynamic Batching Manager for Capstone Platform.
    Coalesces concurrent single-request tensors into vectorized batch matrices
    up to max_batch_size=16 or max_wait_time=5ms.
    """
    def __init__(self, model: torch.nn.Module, max_batch_size: int = 16, max_wait_time_sec: float = 0.005):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time_sec = max_wait_time_sec
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task = None

    def start(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._batch_loop())

    async def predict_async(self, tensor_input: torch.Tensor) -> int:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((tensor_input, future))
        return await future

    async def _batch_loop(self):
        while True:
            batch_items: List[Tuple[torch.Tensor, asyncio.Future]] = []
            
            # Wait for first item
            item = await self.queue.get()
            batch_items.append(item)
            
            t_start = time.perf_counter()
            while len(batch_items) < self.max_batch_size:
                elapsed = time.perf_counter() - t_start
                remaining = self.max_wait_time_sec - elapsed
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break

            # Process vectorized batch
            tensors = [b[0] for b in batch_items]
            futures = [b[1] for b in batch_items]
            
            batch_tensor = torch.stack(tensors, dim=0)
            
            with torch.no_grad():
                logits = self.model(batch_tensor)
                preds = torch.argmax(logits, dim=1).numpy()
                
            for idx, fut in enumerate(futures):
                if not fut.done():
                    fut.set_result(int(preds[idx]))

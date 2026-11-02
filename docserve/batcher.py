"""Dynamic micro-batching for classifier inference.

Concurrent requests each submit one preprocessed tensor. A background task
collects them until `max_batch_size` items are waiting or the oldest has waited
`max_wait_ms`, then runs a single ONNX Runtime call in a worker thread and
resolves each request's future. Batches are grouped per model version.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .metrics import BATCH_QUEUE_WAIT, BATCH_SIZE, FAILURES
from .registry import LoadedModel

log = logging.getLogger(__name__)


@dataclass
class _Item:
    tensor: np.ndarray
    future: asyncio.Future
    enqueued: float = field(default_factory=time.perf_counter)


class MicroBatcher:
    def __init__(self, model: LoadedModel, max_batch_size: int = 16, max_wait_ms: float = 8.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self._queue: asyncio.Queue[_Item] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"batcher-{self.model.version}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, tensor: np.ndarray) -> np.ndarray:
        """Returns the softmax probability vector for one image."""
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(_Item(tensor, fut))
        return await fut

    async def _collect(self) -> list[_Item]:
        items = [await self._queue.get()]
        deadline = items[0].enqueued + self.max_wait
        while len(items) < self.max_batch_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                items.append(await asyncio.wait_for(self._queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return items

    async def _run(self) -> None:
        while True:
            items = await self._collect()
            now = time.perf_counter()
            for it in items:
                BATCH_QUEUE_WAIT.observe(now - it.enqueued)
            BATCH_SIZE.observe(len(items))
            try:
                batch = np.stack([it.tensor for it in items])
                probs = await asyncio.to_thread(self.model.predict, batch)
                for it, p in zip(items, probs):
                    if not it.future.done():
                        it.future.set_result(p)
            except Exception as exc:  # fail every request in the batch, keep the loop alive
                FAILURES.labels("classify", type(exc).__name__).inc()
                log.exception("batch inference failed", extra={"batch_size": len(items)})
                for it in items:
                    if not it.future.done():
                        it.future.set_exception(exc)

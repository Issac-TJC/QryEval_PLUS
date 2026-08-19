"""Bounded single-worker inference queue with request coalescing."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict


class QueueOverloaded(RuntimeError):
    pass


class InferenceQueue:
    def __init__(self, engine, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("Queue capacity must be positive.")
        self.engine = engine
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qryeval-inference")
        self.worker_task = None
        self.inflight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def start(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.engine.start)
        self.worker_task = asyncio.create_task(self._worker())

    async def close(self):
        if self.worker_task:
            await self.queue.put(None)
            await self.worker_task
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.engine.close)
        self.executor.shutdown(wait=True, cancel_futures=True)

    async def submit(self, question: str, policy: str, request_id: str | None = None):
        cached = self.engine.cached(question, policy)
        if cached:
            if request_id:
                cached["request_id"] = request_id
            return cached
        key = self.engine.cache_key(question, policy)
        async with self._lock:
            future = self.inflight.get(key)
            if future is None:
                if self.queue.full():
                    raise QueueOverloaded("Inference queue is full.")
                future = asyncio.get_running_loop().create_future()
                future.add_done_callback(
                    lambda completed: completed.exception() if not completed.cancelled() else None
                )
                self.inflight[key] = future
                self.queue.put_nowait((key, question, policy, request_id, time.monotonic(), future))
        return await asyncio.shield(future)

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            key, question, policy, request_id, queued_at, future = item
            try:
                result = await loop.run_in_executor(
                    self.executor, self.engine.answer, question, policy, request_id
                )
                result["latency_ms"]["queue"] = (time.monotonic() - queued_at) * 1000.0 - result["latency_ms"].get("service_total", 0.0)
                result["latency_ms"]["queue"] = max(0.0, result["latency_ms"]["queue"])
                if not future.cancelled():
                    future.set_result(result)
            except BaseException as exc:
                if not future.cancelled():
                    future.set_exception(exc)
            finally:
                async with self._lock:
                    self.inflight.pop(key, None)
                self.queue.task_done()

"""Deliver verified media batches before the slowest media in a pack finishes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")
V = TypeVar("V")


class EarlyBatches(Generic[T, V]):
    def __init__(
        self,
        consume: Callable[[list[tuple[T, V]]], Awaitable[None]],
        grouping: Callable[[T, V], tuple[object, int]],
        *,
        flush_delay: float = 1.0,
    ) -> None:
        self.consume = consume
        self.grouping = grouping
        self.pending: dict[object, list[tuple[T, V]]] = {}
        self.tasks: list[asyncio.Task[None]] = []
        self.flush_delay = flush_delay
        self.timer: asyncio.Task[None] | None = None

    def check(self) -> None:
        if self.timer is not None and self.timer.done():
            self.timer.result()
        for task in self.tasks:
            if task.done():
                task.result()

    async def consume_batch(self, batch: list[tuple[T, V]]) -> None:
        await self.consume(batch)

    async def add(self, item: T, value: V) -> None:
        self.check()
        key, size = self.grouping(item, value)
        batch = self.pending.setdefault(key, [])
        batch.append((item, value))
        if len(batch) >= size:
            self.pending[key] = []
            self.tasks.append(asyncio.create_task(self.consume_batch(batch)))
        if self.timer is None or self.timer.done():
            self.timer = asyncio.create_task(self.flush_later())
        # Let a ready AI batch start even when every media callback hits cache.
        await asyncio.sleep(0)

    def flush(self) -> None:
        self.check()
        for batch in self.pending.values():
            if batch:
                self.tasks.append(asyncio.create_task(self.consume_batch(batch)))
        self.pending.clear()

    async def flush_later(self) -> None:
        await asyncio.sleep(self.flush_delay)
        self.flush()

    async def finish(self, *, flush_pending: bool = True) -> None:
        self.check()
        if self.timer is not None:
            self.timer.cancel()
            await asyncio.gather(self.timer, return_exceptions=True)
            self.timer = None
        if flush_pending:
            self.flush()
        else:
            self.pending.clear()
        if self.tasks:
            await asyncio.gather(*self.tasks)

    async def close(self) -> None:
        tasks = [*self.tasks, *([self.timer] if self.timer is not None else [])]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            drained = asyncio.gather(*tasks, return_exceptions=True)
            while not drained.done():
                try:
                    await asyncio.shield(drained)
                except asyncio.CancelledError:
                    continue
            drained.result()

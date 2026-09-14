"""Operation-scoped resource limits and bounded asynchronous scheduling."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ByteBudgetExceeded(RuntimeError):
    """The combined temporary storage of concurrent packs exceeded its limit."""


class ByteBudget:
    def __init__(self, maximum: int) -> None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError("temporary byte limit must be positive")
        self.maximum = maximum
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def adjust(self, delta: int, *, observed: bool = False) -> None:
        with self._lock:
            updated = self._used + delta
            if updated < 0:
                raise ValueError("temporary byte release exceeds reservation")
            if updated > self.maximum and delta > 0 and not observed:
                raise ByteBudgetExceeded("run temporary disk limit exceeded")
            self._used = updated


@dataclass(frozen=True)
class BatchLimits:
    download_slots: asyncio.Semaphore
    render_slots: asyncio.Semaphore
    ai_slots: asyncio.Semaphore
    temp_budget: ByteBudget
    retained_stores: dict[str, object] = field(default_factory=dict)
    retained_lock: threading.RLock = field(default_factory=threading.RLock)
    retained_build_locks: dict[str, Any] = field(default_factory=dict)


_batch: ContextVar[BatchLimits | None] = ContextVar("mojilex_batch_limits", default=None)


def current_batch_limits() -> BatchLimits | None:
    return _batch.get()


@contextmanager
def batch_limits(
    *, downloads: int, renders: int, ai: int, max_temp_bytes: int
) -> Iterator[BatchLimits]:
    """Share limits across child tasks without leaking them into another operation."""
    if any(type(value) is not int or value < 1 for value in (downloads, renders, ai)):
        raise ValueError("concurrency limits must be positive integers")
    # A nested pipeline must share its parent's limits, never multiply them.
    existing = current_batch_limits()
    if existing is not None:
        yield existing
        return
    limits = BatchLimits(
        asyncio.Semaphore(downloads),
        asyncio.Semaphore(renders),
        asyncio.Semaphore(ai),
        ByteBudget(max_temp_bytes),
    )
    token = _batch.set(limits)
    try:
        yield limits
    finally:
        _batch.reset(token)


@asynccontextmanager
async def ai_slot() -> AsyncIterator[None]:
    """Hold a shared slot only around one provider request, including its cleanup."""
    limits = current_batch_limits()
    if limits is None:
        yield
    else:
        async with limits.ai_slots:
            yield


async def bounded_map(
    items: Iterable[T], worker: Callable[[T], Awaitable[R]], *, concurrency: int
) -> list[R]:
    """Map in input order using at most N tasks; errors cancel and drain all peers.

    The worker handles recoverable item errors itself. Escaping errors are terminal.
    Input is consumed lazily so a large source list never creates one task per source.
    """
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("worker concurrency must be positive")
    indexed = enumerate(items)
    results: dict[int, R] = {}

    async def consume() -> None:
        for index, item in indexed:
            results[index] = await worker(item)

    tasks = [asyncio.create_task(consume()) for _ in range(concurrency)]
    group = asyncio.gather(*tasks)
    try:
        # Shield makes the pool the sole cancellation owner. Otherwise cancelling
        # gather and then cancelling its children again can interrupt their cleanup.
        await asyncio.shield(group)
    except BaseException:
        for task in tasks:
            task.cancel()
        # A second Ctrl+C must not let callers delete files while workers still
        # own them. Keep draining even if cancellation is repeated during cleanup.
        # Drain the original gathering future too: its completion callback can
        # run after the child tasks finish, otherwise its cancellation exception
        # may be left unobserved ("_GatheringFuture exception was never retrieved").
        drained = asyncio.gather(*tasks, group, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        drained.result()
        if group.done() and not group.cancelled():
            group.exception()
        raise
    return [results[index] for index in range(len(results))]


class OrderedTurns:
    """Serialize merges in source order, including sources that failed early.

    Call finish in finally even when a source fails before wait. Both operations
    run on the same event loop; finish deliberately has no cancellation point.
    """

    def __init__(self) -> None:
        self._next = 0
        self._finished: set[int] = set()
        self._changed = asyncio.Event()

    async def wait(self, index: int) -> None:
        if index < 0:
            raise ValueError("turn index must be non-negative")
        while index > self._next:
            await self._changed.wait()
        if index != self._next:
            raise ValueError("turn has already finished")

    def finish(self, index: int) -> None:
        if index < 0:
            raise ValueError("turn index must be non-negative")
        if index < self._next:
            return
        self._finished.add(index)
        while self._next in self._finished:
            self._finished.remove(self._next)
            self._next += 1
        previous = self._changed
        self._changed = asyncio.Event()
        previous.set()


class PackDependencies:
    """Serialize overlapping packs while independent media can run concurrently.

    Claims are registered in input order before waiting for earlier owners. This
    prevents a later pack from blocking an earlier pack's ordered merge. Completed
    owners and claims are removed so memory is bounded by active pack contents.
    """

    def __init__(self) -> None:
        self._registration = OrderedTurns()
        self._claims: dict[str, set[asyncio.Event]] = {}
        self._owners: dict[int, tuple[asyncio.Event, frozenset[str]]] = {}

    async def wait(self, index: int, keys: Iterable[str]) -> None:
        await self._registration.wait(index)
        unique = frozenset(keys)
        preceding = {event for key in unique for event in self._claims.get(key, ())}
        done = asyncio.Event()
        self._owners[index] = done, unique
        for key in unique:
            self._claims.setdefault(key, set()).add(done)
        self._registration.finish(index)
        for event in preceding:
            await event.wait()

    def finish(self, index: int) -> None:
        owner = self._owners.pop(index, None)
        if owner is not None:
            done, keys = owner
            for key in keys:
                claims = self._claims[key]
                claims.remove(done)
                if not claims:
                    del self._claims[key]
            done.set()
        # Metadata failure before registration must also unblock later sources.
        self._registration.finish(index)


@dataclass(frozen=True)
class PackPipelineLimits:
    preparation: asyncio.Semaphore
    inflight: asyncio.Semaphore


_pack_pipeline: ContextVar[PackPipelineLimits | None] = ContextVar(
    "mojilex_pack_pipeline", default=None
)


@contextmanager
def pack_pipeline_limits(concurrency: int) -> Iterator[PackPipelineLimits]:
    """One preparation window plus one bounded downstream window, shared across runs."""
    if concurrency < 1:
        raise ValueError("pack concurrency must be positive")
    existing = _pack_pipeline.get()
    if existing is not None:
        yield existing
        return
    limits = PackPipelineLimits(asyncio.Semaphore(concurrency), asyncio.Semaphore(2 * concurrency))
    token = _pack_pipeline.set(limits)
    try:
        yield limits
    finally:
        _pack_pipeline.reset(token)


P = ParamSpec("P")


async def run_blocking(function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Keep the event loop responsive and drain disk workers before releasing owners."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise

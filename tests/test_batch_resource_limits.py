"""Shared limits cover independent packs and survive failure/cancellation."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import ExitStack
from pathlib import Path

import pytest

from mojilex_cli.concurrency import (
    OrderedTurns,
    PackDependencies,
    ai_slot,
    batch_limits,
    bounded_map,
    current_batch_limits,
)
from mojilex_cli.media import MediaLimitError, MediaLimits, MediaProcessor, TemporaryMediaRun


async def one_chunk(content: bytes) -> AsyncIterator[bytes]:
    yield content


async def test_bounded_map_limits_tasks_and_preserves_order() -> None:
    parent_tasks = set(asyncio.all_tasks())
    gate = asyncio.Event()
    ready = asyncio.Event()
    consumed = 0

    async def worker(value: int) -> int:
        nonlocal consumed
        consumed += 1
        if consumed == 3:
            ready.set()
        await gate.wait()
        return value * 2

    operation = asyncio.create_task(bounded_map(range(20_000), worker, concurrency=3))
    await asyncio.wait_for(ready.wait(), 2)
    assert len(set(asyncio.all_tasks()) - parent_tasks) == 4
    assert consumed == 3
    gate.set()
    assert await operation == [value * 2 for value in range(20_000)]


@pytest.mark.parametrize("cancel", [False, True])
async def test_bounded_map_drains_children_on_error_or_repeated_cancel(cancel: bool) -> None:
    started = asyncio.Event()
    trigger = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    stopped = False

    async def worker(value: int) -> None:
        nonlocal stopped
        if value == 0:
            await trigger.wait()
            raise RuntimeError("terminal")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            stopped = True

    operation = asyncio.create_task(bounded_map(range(2), worker, concurrency=2))
    await started.wait()
    if cancel:
        operation.cancel()
    else:
        trigger.set()
    await cleaning.wait()
    operation.cancel()
    await asyncio.sleep(0)
    assert not operation.done()
    release.set()
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await operation
    assert stopped


async def test_ordered_turns_skip_early_failure_and_merge_in_input_order() -> None:
    turns = OrderedTurns()
    merged: list[int] = []

    async def merge(index: int) -> None:
        await turns.wait(index)
        merged.append(index)
        turns.finish(index)

    tasks = [asyncio.create_task(merge(index)) for index in (3, 2, 0)]
    turns.finish(1)
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert merged == [0, 2, 3]


async def test_pack_dependencies_preserve_order_but_allow_disjoint_packs() -> None:
    deps = PackDependencies()
    await deps.wait(0, ["shared"])
    later = asyncio.create_task(deps.wait(2, ["shared"]))
    await asyncio.sleep(0)
    assert not later.done()  # Pack 1 has not registered its keys yet.
    await deps.wait(1, ["independent"])
    await asyncio.sleep(0)
    assert not later.done()  # Registration finished; pack 0 still owns the key.
    deps.finish(1)
    deps.finish(0)
    await asyncio.wait_for(later, 2)
    deps.finish(2)
    assert not deps._owners
    assert not deps._claims


async def test_pack_dependency_failed_middle_owner_does_not_release_older_owner() -> None:
    deps = PackDependencies()
    await deps.wait(0, ["shared"])
    middle = asyncio.create_task(deps.wait(1, ["shared"]))
    later = asyncio.create_task(deps.wait(2, ["shared"]))
    await asyncio.sleep(0)
    middle.cancel()
    await asyncio.gather(middle, return_exceptions=True)
    deps.finish(1)
    await asyncio.sleep(0)
    assert not later.done()
    deps.finish(0)
    await asyncio.wait_for(later, 2)
    deps.finish(2)
    deps.finish(3)  # Metadata fetch failed before registering any claims.
    await deps.wait(4, ["new"])
    deps.finish(4)
    assert not deps._claims


async def test_context_isolation_nested_scope_and_shared_ai_cap() -> None:
    active = peak = 0
    assert current_batch_limits() is None

    async def request(_: int) -> None:
        nonlocal active, peak
        async with ai_slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    with batch_limits(downloads=2, renders=1, ai=2, max_temp_bytes=100) as outer:
        with batch_limits(downloads=20, renders=8, ai=50, max_temp_bytes=1000) as nested:
            assert nested is outer
            await bounded_map(range(30), request, concurrency=6)
        assert current_batch_limits() is outer
    assert current_batch_limits() is None
    assert peak == 2
    async with ai_slot():
        assert current_batch_limits() is None


async def test_download_slots_are_shared_between_pack_runs(tmp_path: Path) -> None:
    active = peak = 0

    async def stream() -> AsyncIterator[bytes]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            yield b"media"
            await asyncio.sleep(0)
        finally:
            active -= 1

    with batch_limits(downloads=2, renders=1, ai=1, max_temp_bytes=1000), ExitStack() as stack:
        runs = [stack.enter_context(TemporaryMediaRun(root=tmp_path)) for _ in range(3)]
        await asyncio.gather(*(runs[index % 3].write_stream(stream()) for index in range(15)))
    assert peak == 2


async def test_render_slots_are_shared_between_processors(tmp_path: Path) -> None:
    active = peak = 0

    class CountingProcessor(MediaProcessor):
        async def _render_source(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    with batch_limits(downloads=15, renders=2, ai=1, max_temp_bytes=1000), ExitStack() as stack:
        processors = [
            CountingProcessor(
                stack.enter_context(TemporaryMediaRun(root=tmp_path)), render_concurrency=8
            )
            for _ in range(3)
        ]
        await asyncio.gather(
            *(
                processors[index % 3].process_stream(one_chunk(b"x"), expected_format="webp")
                for index in range(15)
            )
        )
    assert peak == 2


async def test_aggregate_temp_budget_counts_downloads_outputs_retained_and_releases(
    tmp_path: Path,
) -> None:
    with batch_limits(downloads=3, renders=2, ai=1, max_temp_bytes=12) as limits:
        with TemporaryMediaRun(root=tmp_path) as first, TemporaryMediaRun(root=tmp_path) as second:
            await first.write_stream(one_chunk(b"1234"))
            second.reserve_retained_bytes(3)
            assert second.path is not None
            output = second.path / "frame.png"
            output.write_bytes(b"abcd")
            second.account_outputs([output])
            second.account_outputs([output])
            assert limits.temp_budget.used == 11
            with pytest.raises(MediaLimitError):
                await first.write_stream(one_chunk(b"12"))
            assert limits.temp_budget.used == 11
            assert not list(first.path.glob("*.partial"))  # type: ignore[union-attr]
            with pytest.raises(MediaLimitError):
                second.reserve_retained_bytes(2)
            second.release_retained_bytes(3)
            assert limits.temp_budget.used == 8
            output.write_bytes(b"ab")
            second.account_outputs([output])
            assert limits.temp_budget.used == 6
            first.cleanup()
            first.cleanup()
            assert limits.temp_budget.used == 2
        assert limits.temp_budget.used == 0


async def test_partial_stream_cancel_releases_shared_reservation(tmp_path: Path) -> None:
    ready = asyncio.Event()

    async def partial() -> AsyncIterator[bytes]:
        yield b"abc"
        ready.set()
        await asyncio.Event().wait()

    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=3) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            task = asyncio.create_task(run.write_stream(partial()))
            await ready.wait()
            assert limits.temp_budget.used == 3
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert limits.temp_budget.used == 0
            await run.write_stream(one_chunk(b"abc"))
        assert limits.temp_budget.used == 0


async def test_shared_budget_preserves_stricter_individual_limit(tmp_path: Path) -> None:
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=100) as limits:
        with TemporaryMediaRun(root=tmp_path, limits=MediaLimits(max_run_temp_bytes=2)) as run:
            with pytest.raises(MediaLimitError):
                await run.write_stream(one_chunk(b"abc"))
        assert limits.temp_budget.used == 0


async def test_oversized_renderer_output_remains_charged_until_cleanup(tmp_path: Path) -> None:
    with batch_limits(downloads=2, renders=2, ai=1, max_temp_bytes=5) as limits:
        with TemporaryMediaRun(root=tmp_path) as first, TemporaryMediaRun(root=tmp_path) as second:
            await first.write_stream(one_chunk(b"x"))
            assert first.path is not None
            output = first.path / "frame.png"
            output.write_bytes(b"123456")
            with pytest.raises(MediaLimitError):
                first.account_outputs([output])
            assert limits.temp_budget.used == 7
            with pytest.raises(MediaLimitError):
                await second.write_stream(one_chunk(b"x"))
            assert limits.temp_budget.used == 7
            first.cleanup()
            assert limits.temp_budget.used == 0
            await second.write_stream(one_chunk(b"x"))

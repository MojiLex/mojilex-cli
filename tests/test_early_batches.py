import asyncio

import pytest

from mojilex_cli.pipeline.early_batches import EarlyBatches


async def test_full_batch_starts_before_pack_finishes():
    started = asyncio.Event()
    release = asyncio.Event()
    batches = []

    async def consume(batch):
        batches.append(batch)
        started.set()
        await release.wait()

    stream = EarlyBatches(consume, lambda *_: ("static", 2))
    try:
        await stream.add("a", 1)
        await stream.add("b", 2)
        await asyncio.wait_for(started.wait(), 1)
        assert batches == [[("a", 1), ("b", 2)]]
        release.set()
        await stream.finish()
    finally:
        await stream.close()


async def test_partial_batch_does_not_wait_for_slow_sibling():
    started = asyncio.Event()

    async def consume(batch):
        assert batch == [("ready", 1)]
        started.set()

    stream = EarlyBatches(consume, lambda *_: ("static", 16), flush_delay=0.01)
    try:
        await stream.add("ready", 1)
        await asyncio.wait_for(started.wait(), 1)
        await stream.finish()
    finally:
        await stream.close()


async def test_failure_prevents_further_paid_work_and_drains():
    async def consume(batch):
        raise ValueError("invalid response")

    stream = EarlyBatches(consume, lambda *_: ("static", 1))
    try:
        await stream.add("a", 1)
        with pytest.raises(ValueError, match="invalid response"):
            await stream.add("b", 2)
        assert len(stream.tasks) == 1
    finally:
        await stream.close()


async def test_close_waits_for_provider_cleanup():
    started = asyncio.Event()
    drained = asyncio.Event()

    async def consume(batch):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            drained.set()

    stream = EarlyBatches(consume, lambda *_: ("animated", 1))
    await stream.add("a", 1)
    await started.wait()
    await stream.close()
    assert drained.is_set()
    assert all(task.done() for task in stream.tasks)

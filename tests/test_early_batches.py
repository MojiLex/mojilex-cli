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


@pytest.mark.parametrize("kind,size", [("static", 16), ("animated", 8)])
async def test_partial_batch_waits_for_preparation_to_finish(kind, size):
    batches = []

    async def consume(batch):
        batches.append(batch)

    stream = EarlyBatches(consume, lambda *_: (kind, size))
    try:
        await stream.add("ready", 1)
        # Preparation can exceed the former one-second partial-flush interval.
        await asyncio.sleep(1.1)
        assert batches == []
        await stream.finish()
        assert batches == [[("ready", 1)]]
    finally:
        await stream.close()


async def test_full_static_and_animated_groups_keep_their_sizes():
    batches = []

    async def consume(batch):
        batches.append(batch)

    stream = EarlyBatches(consume, lambda item, _: (item[0], 16 if item[0] == "s" else 8))
    try:
        for index in range(35):
            await stream.add(("s", index), index)
            if index < 19:
                await stream.add(("a", index), index)
        assert sorted((batch[0][0][0], len(batch)) for batch in batches) == [
            ("a", 8),
            ("a", 8),
            ("s", 16),
            ("s", 16),
        ]
        await stream.finish()
        assert sorted((batch[0][0][0], len(batch)) for batch in batches) == [
            ("a", 3),
            ("a", 8),
            ("a", 8),
            ("s", 3),
            ("s", 16),
            ("s", 16),
        ]
        delivered = [item for batch in batches for item, _ in batch]
        assert len(delivered) == len(set(delivered)) == 54
    finally:
        await stream.close()


async def test_finish_can_leave_partial_batch_for_normal_pack_processing():
    batches = []

    async def consume(batch):
        batches.append(batch)

    stream = EarlyBatches(consume, lambda *_: ("static", 16))
    try:
        for index in range(19):
            await stream.add(index, index)
        await stream.finish(flush_pending=False)
        assert batches == [[(index, index) for index in range(16)]]
        assert stream.pending == {}
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

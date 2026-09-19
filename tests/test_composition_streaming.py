import asyncio
import threading

import pytest

from mojilex_cli.ai.base import CostEstimate, RequestBudget
from mojilex_cli.composition import service
from test_composition_service import setup_tiles


@pytest.fixture
async def prepared(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    await queue.prepare("first", source, processed)
    proposal = queue._pending[0][1]

    async def prepare(source, processed, *, enqueue, **kwargs):
        enqueue(proposal, b"png")
        return []

    monkeypatch.setattr(service, "_prepare_compositions", prepare)
    return queue, source, processed, proposal


async def test_ready_source_verifies_while_another_is_preparing(prepared, monkeypatch):
    queue, source, processed, proposal = prepared
    started = asyncio.Event()
    release = asyncio.Event()

    async def prepare(source, processed, *, enqueue, **kwargs):
        started.set()
        await release.wait()
        enqueue(proposal, b"later")
        return []

    calls = []

    async def verify(*args, **kwargs):
        calls.append(kwargs["audit"])
        return True

    monkeypatch.setattr(service, "_prepare_compositions", prepare)
    monkeypatch.setattr(service, "verify_composition", verify)
    preparing = asyncio.create_task(queue.prepare("second", source, processed))
    try:
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(
            queue.verify(key="first", api_key=None, budget=RequestBudget(max_requests=10)), 2
        )
        assert calls == ["continuity", "independent_objects", "layout"]
        assert queue.accepted["first"][0].verified
        assert not preparing.done()
    finally:
        release.set()
        await preparing
    assert [item[0] for item in queue._pending] == ["second"]
    assert queue._bytes == len(b"later")


async def test_different_sources_verify_concurrently_without_losing_pending(prepared, monkeypatch):
    queue, source, processed, _ = prepared
    await queue.prepare("second", source, processed)
    started = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def verify(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            started.set()
        await release.wait()
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    budget = RequestBudget(max_requests=100)
    tasks = [
        asyncio.create_task(queue.verify(key=key, api_key=None, budget=budget))
        for key in ("first", "second")
    ]
    try:
        await asyncio.wait_for(started.wait(), 2)
        await queue.prepare("third", source, processed)
        assert len(queue._pending) == 3
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert [item[0] for item in queue._pending] == ["third"]
    assert queue._bytes == 3
    assert len(queue.accepted["first"]) == len(queue.accepted["second"]) == 1


async def test_attempt_limit_persists_across_per_source_verification_calls(prepared, monkeypatch):
    queue, source, processed, _ = prepared
    budget = RequestBudget(max_requests=100, allow_unknown_cost=True)

    async def verify(*args, **kwargs):
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="audit"))
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    for key in ("first", "second", "third", "fourth", "fifth"):
        if key != "first":
            await queue.prepare(key, source, processed)
        await queue.verify(key=key, api_key=None, budget=budget)
    assert budget.requests_used == 10
    assert [len(groups) for groups in queue.accepted.values()] == [1, 1, 1, 0, 0]
    assert not queue._pending and queue._bytes == 0


async def test_same_key_verification_waits_and_does_not_duplicate_paid_calls(prepared, monkeypatch):
    queue, _, _, _ = prepared
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def verify(*args, **kwargs):
        calls.append(kwargs["audit"])
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    budget = RequestBudget(max_requests=10)
    first = asyncio.create_task(queue.verify(key="first", api_key=None, budget=budget))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(queue.verify(key="first", api_key=None, budget=budget))
    try:
        await asyncio.sleep(0)
        assert len(calls) == 1 and not second.done()
    finally:
        release.set()
        await asyncio.gather(first, second)
    assert calls == ["continuity", "independent_objects", "layout"]


@pytest.mark.parametrize("limit", ["bytes", "groups"])
async def test_inflight_proposals_keep_reservation_until_cancellation(prepared, monkeypatch, limit):
    queue, source, processed, proposal = prepared
    queue._pending.clear()
    queue._bytes = 0
    png = b"x" * (8 * 1024 * 1024) if limit == "bytes" else b"png"
    groups = 1 if limit == "bytes" else 32

    async def prepare(source, processed, *, enqueue, **kwargs):
        for _ in range(groups):
            enqueue(proposal, png)
        return []

    async def verify(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_prepare_compositions", prepare)
    monkeypatch.setattr(service, "verify_composition", verify)
    await queue.prepare("first", source, processed)
    await queue.prepare("second", source, processed)
    started = asyncio.Event()
    checking = asyncio.create_task(
        queue.verify(key="first", api_key=None, budget=RequestBudget(max_requests=100))
    )
    await asyncio.wait_for(started.wait(), 2)
    await queue.prepare("third", source, processed)
    assert len(queue._pending) == groups * 2
    assert queue._bytes == len(png) * groups * 2
    checking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checking
    assert len(queue._pending) == groups
    assert {item[0] for item in queue._pending} == {"second"}
    assert queue._bytes == len(png) * groups
    assert queue.accepted["first"] == []
    await queue.prepare("fourth", source, processed)
    assert len(queue._pending) == groups * 2


async def test_cancelled_prepare_releases_only_its_own_enqueued_images(prepared, monkeypatch):
    queue, source, processed, proposal = prepared
    before = list(queue._pending)
    started = asyncio.Event()

    async def prepare(source, processed, *, enqueue, **kwargs):
        enqueue(proposal, b"temporary")
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_prepare_compositions", prepare)
    preparing = asyncio.create_task(queue.prepare("second", source, processed))
    await asyncio.wait_for(started.wait(), 2)
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert queue._pending == before
    assert queue._bytes == sum(len(item[2]) for item in before)


async def test_cancelled_detector_drains_before_its_images_are_closed(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    started = asyncio.Event()
    release = threading.Event()
    read_after_cancel = threading.Event()
    loop = asyncio.get_running_loop()

    def candidates(tiles):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        # PIL raises ValueError here if cancellation already closed the image.
        tiles[0].image.getbbox()
        read_after_cancel.set()
        return []

    monkeypatch.setattr(service, "candidates", candidates)
    preparing = asyncio.create_task(queue.prepare("first", source, processed))
    try:
        await asyncio.wait_for(started.wait(), 2)
        preparing.cancel()
        await asyncio.sleep(0)
        assert not preparing.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await preparing
    assert read_after_cancel.is_set()
    assert not queue._pending and queue._bytes == 0

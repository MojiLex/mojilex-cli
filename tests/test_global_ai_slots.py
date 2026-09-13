from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from mojilex_cli.ai import GeminiVisionProvider, RequestBudget, base, describe_with_recovery
from mojilex_cli.composition.verifier import verify_composition
from mojilex_cli.concurrency import batch_limits
from test_ai_gemini import _FakeModels, _request
from test_composition_verifier import PNG, Client


def provider(create):
    return GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )


async def test_description_and_composition_share_actual_request_limit():
    active = peak = calls = 0
    release = asyncio.Event()
    entered = asyncio.Event()
    original = _FakeModels().create

    async def counted(call, kwargs):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            await release.wait()
            return await call(**kwargs)
        finally:
            active -= 1

    async def create(**kwargs):
        return await counted(original, kwargs)

    client = Client()
    composition_create = client.create

    async def create_composition(**kwargs):
        return await counted(composition_create, kwargs)

    client.create = create_composition
    budget = RequestBudget(max_requests=6, allow_unknown_cost=True)
    with batch_limits(downloads=1, renders=1, ai=2, max_temp_bytes=1024):
        tasks = [
            asyncio.create_task(describe_with_recovery(provider(create), _request(), budget))
            for _ in range(4)
        ]
        tasks += [
            asyncio.create_task(
                verify_composition(PNG, model="test", api_key=None, client=client, budget=budget)
            )
            for _ in range(2)
        ]
        await asyncio.wait_for(entered.wait(), 1)
        assert budget.requests_used == calls == 2
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert results[-2:] == [True, True]
    assert peak == 2
    assert budget.requests_used == calls == 6


async def test_transport_backoff_releases_global_slot(monkeypatch):
    backoff = asyncio.Event()
    retry = asyncio.Event()
    original = _FakeModels().create
    attempts = 0

    async def sleep(seconds):
        backoff.set()
        await retry.wait()

    async def create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("synthetic")
        return await original(**kwargs)

    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    budget = RequestBudget(max_requests=3, allow_unknown_cost=True)
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024):
        first = asyncio.create_task(describe_with_recovery(provider(create), _request(), budget))
        await asyncio.wait_for(backoff.wait(), 1)
        second = await asyncio.wait_for(
            describe_with_recovery(provider(original), _request(), budget), 1
        )
        assert second.batch.items[0].label == "E001"
        retry.set()
        await asyncio.wait_for(first, 1)
    assert budget.requests_used == 3


async def test_cancelling_queued_request_does_not_charge_budget_or_leak_slot():
    entered = asyncio.Event()
    never = asyncio.Event()
    original = _FakeModels().create

    async def create(**kwargs):
        entered.set()
        await never.wait()
        return await original(**kwargs)

    budget = RequestBudget(max_requests=2, allow_unknown_cost=True)
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024):
        active = asyncio.create_task(describe_with_recovery(provider(create), _request(), budget))
        await asyncio.wait_for(entered.wait(), 1)
        queued = asyncio.create_task(describe_with_recovery(provider(original), _request(), budget))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert budget.requests_used == 1
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        result = await asyncio.wait_for(
            describe_with_recovery(provider(original), _request(), budget), 1
        )
        assert result.batch.items[0].label == "E001"
    assert budget.requests_used == 2

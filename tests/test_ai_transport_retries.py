from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from mojilex_cli.ai import (
    AIError,
    AITransientError,
    BudgetExceededError,
    GeminiVisionProvider,
    RequestBudget,
    base,
    describe_with_recovery,
)
from test_ai_gemini import _FakeModels, _request


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["recovers", "exhausted", "budget", "schema_after_network"])
async def test_transient_retries_charge_each_http_attempt_and_preserve_schema_limit(
    monkeypatch, scenario
) -> None:
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    resource = _FakeModels()
    valid_create = resource.create
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        if scenario != "schema_after_network" and (calls <= 2 or scenario == "exhausted"):
            raise httpx.ConnectError("synthetic-private-network-details")
        if scenario == "schema_after_network":
            if calls == 1:
                raise httpx.ReadTimeout("synthetic-private-network-details")
            return SimpleNamespace(status="completed", model="gemini-test", output_text="{}")
        return await valid_create(**kwargs)

    resource.create = create
    provider = GeminiVisionProvider(
        model="gemini-test", client=SimpleNamespace(aio=SimpleNamespace(interactions=resource))
    )
    budget = RequestBudget(max_requests=1 if scenario == "budget" else 10, allow_unknown_cost=True)
    events = []
    if scenario == "recovers":
        assert (
            await describe_with_recovery(
                provider, _request(), budget, progress_callback=events.append
            )
        ).batch.items[0].label == "E001"
    else:
        expected = BudgetExceededError if scenario == "budget" else AIError
        with pytest.raises(expected) as captured:
            await describe_with_recovery(
                provider, _request(), budget, progress_callback=events.append
            )
        assert "synthetic-private" not in str(captured.value)
        if scenario == "exhausted":
            assert isinstance(captured.value, AITransientError)
    assert calls == budget.requests_used == (1 if scenario == "budget" else 3)
    assert delays == ([1] if scenario in {"budget", "schema_after_network"} else [1, 2])
    assert events.count("retry") == (1 if scenario == "schema_after_network" else 0)
    assert events.count("request") == calls
    assert budget.cost_reserved == Decimal("0")


@pytest.mark.asyncio
@pytest.mark.parametrize("http_status", [400, 401, 403, 404, 422])
async def test_permanent_provider_errors_are_not_retried(http_status) -> None:
    class SyntheticError(Exception):
        code = http_status

    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        raise SyntheticError("synthetic-private-provider-details")

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    budget = RequestBudget(max_requests=10, allow_unknown_cost=True)
    with pytest.raises(AIError) as captured:
        await describe_with_recovery(provider, _request(), budget)
    assert not isinstance(captured.value, AITransientError)
    assert "synthetic-private" not in str(captured.value)
    assert calls == budget.requests_used == 1


@pytest.mark.parametrize("http_status", [408, 429, 500, 502, 503, 504])
def test_known_transient_provider_statuses_are_retryable(http_status) -> None:
    from mojilex_cli.ai.gemini import _transient_provider_error

    error = RuntimeError("synthetic-private-provider-details")
    error.code = http_status
    assert _transient_provider_error(error)

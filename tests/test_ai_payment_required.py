from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai import (
    AIError,
    AIPaymentRequiredError,
    AITransientError,
    CostEstimate,
    GeminiVisionProvider,
    RequestBudget,
    describe_with_recovery,
)
from mojilex_cli.ai.base import request_budget_scope
from mojilex_cli.ai.gemini import _payment_required_provider_error
from mojilex_cli.concurrency import batch_limits
from test_ai_gemini import _FakeModels, _request


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_http_402_stops_after_one_charged_request_without_exposing_sdk_body(wrapped):
    class SDKStatusError(Exception):
        status_code = 402

    sdk_error = SDKStatusError("synthetic-private-provider-body")
    sdk_error.__cause__ = ValueError("synthetic-private-error-response-validation")
    error = RuntimeError("synthetic-private-wrapper") if wrapped else sdk_error
    if wrapped:
        error.__cause__ = sdk_error
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    budget = RequestBudget(max_requests=10, allow_unknown_cost=True)
    events = []
    with pytest.raises(AIPaymentRequiredError) as captured:
        await describe_with_recovery(provider, _request(), budget, progress_callback=events.append)
    assert captured.value.code == "AI_PAYMENT_REQUIRED"
    assert "HTTP 402 Payment Required" in str(captured.value)
    assert "billing" in str(captured.value)
    assert "synthetic-private" not in str(captured.value)
    assert not isinstance(captured.value, AITransientError)
    assert calls == budget.requests_used == 1
    assert budget.cost_reserved == Decimal("0")
    assert events == ["request"]


@pytest.mark.parametrize("outer_status", [200, 400, 401, 403, 429, 503])
def test_outer_explicit_response_overrides_incidental_nested_402(outer_status):
    outer = RuntimeError("synthetic-private-outer")
    outer.status_code = outer_status
    inner = RuntimeError("synthetic-private-inner")
    inner.status_code = 402
    outer.__cause__ = inner
    assert not _payment_required_provider_error(outer)


@pytest.mark.parametrize("status", [True, False, "402", 402.0, None, 99, 600])
def test_payment_status_requires_bounded_integer_http_code(status):
    error = RuntimeError("synthetic-private-details")
    error.code = status
    assert not _payment_required_provider_error(error)


def test_payment_chain_is_cycle_bounded_and_does_not_classify_implicit_context():
    outer = RuntimeError("synthetic-private-wrapper")
    inner = RuntimeError("synthetic-private-status")
    inner.status_code = 402
    outer.__context__ = inner
    outer.__cause__ = outer
    assert not _payment_required_provider_error(outer)


@pytest.mark.asyncio
async def test_provider_payment_subtype_is_preserved_without_request_retry():
    payment = AIPaymentRequiredError("synthetic fixed payment status")

    async def create(**kwargs):
        raise payment

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    with pytest.raises(AIError) as captured:
        await provider.describe(_request())
    assert captured.value is payment


@pytest.mark.asyncio
async def test_payment_latch_blocks_queued_peers_before_reservation_recorders():
    entered = asyncio.Event()
    release = asyncio.Event()
    reservations = []
    calls = 0
    payment = AIPaymentRequiredError("synthetic payment status")

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise payment

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    parent = RequestBudget(max_requests=100, requests_used=20, allow_unknown_cost=True)
    with request_budget_scope(parent):
        first_budget = RequestBudget(
            max_requests=10,
            requests_used=3,
            cost_reserved=Decimal("0.50"),
            allow_unknown_cost=True,
            reservation_recorder=lambda requests, cost: reservations.append((requests, cost)),
        )
        sibling_budget = RequestBudget(
            max_requests=10,
            requests_used=2,
            allow_unknown_cost=True,
            reservation_recorder=lambda *args: pytest.fail("queued sibling was charged"),
        )
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024):
        active = asyncio.create_task(describe_with_recovery(provider, _request(), first_budget))
        await asyncio.wait_for(entered.wait(), 1)
        queued = [
            asyncio.create_task(describe_with_recovery(provider, _request(), sibling_budget))
            for _ in range(5)
        ]
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(active, *queued, return_exceptions=True), 1)
    assert all(isinstance(result, AIPaymentRequiredError) for result in results)
    assert calls == 1
    assert first_budget.requests_used == 4
    assert sibling_budget.requests_used == 2
    assert parent.requests_used == 21
    assert reservations == [(4, Decimal("0.50"))]


@pytest.mark.asyncio
async def test_already_inflight_result_and_cost_survive_payment_stop():
    entered = asyncio.Event()
    payment_release = asyncio.Event()
    successful_release = asyncio.Event()
    successful = _FakeModels().create
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        ordinal = calls
        if calls == 2:
            entered.set()
        if ordinal == 1:
            await payment_release.wait()
            raise AIPaymentRequiredError("synthetic payment status")
        await successful_release.wait()
        return await successful(**kwargs)

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    provider.estimate = lambda request: CostEstimate(
        upper_bound_usd=Decimal("0.25"), note="synthetic fixed cost"
    )
    reservations = []
    budget = RequestBudget(
        max_requests=10,
        requests_used=3,
        cost_reserved=Decimal("0.50"),
        reservation_recorder=lambda requests, cost: reservations.append((requests, cost)),
    )
    with batch_limits(downloads=1, renders=1, ai=2, max_temp_bytes=1024):
        first = asyncio.create_task(describe_with_recovery(provider, _request(), budget))
        second = asyncio.create_task(describe_with_recovery(provider, _request(), budget))
        await asyncio.wait_for(entered.wait(), 1)
        queued = asyncio.create_task(describe_with_recovery(provider, _request(), budget))
        await asyncio.sleep(0)
        payment_release.set()
        with pytest.raises(AIPaymentRequiredError):
            await first
        with pytest.raises(AIPaymentRequiredError):
            await queued
        successful_release.set()
        result = await asyncio.wait_for(second, 1)
    assert result.batch.items[0].label == "E001"
    assert calls == 2
    assert budget.requests_used == 5
    assert budget.cost_reserved == Decimal("1.00")
    assert reservations == [(4, Decimal("0.75")), (5, Decimal("1.00"))]


@pytest.mark.asyncio
async def test_payment_stop_skips_confirmation_and_resumed_budget_has_fresh_latch():
    budget = RequestBudget(
        max_requests=10,
        requests_used=3,
        cost_reserved=Decimal("0.50"),
        unknown_cost_authorizer=lambda remaining: pytest.fail("halted invocation prompted"),
        reservation_recorder=lambda *args: pytest.fail("halted invocation recorded"),
    )
    budget.stop_for_payment(AIPaymentRequiredError("synthetic payment status"))
    with pytest.raises(AIPaymentRequiredError):
        await budget.reserve(CostEstimate(upper_bound_usd=None, note="synthetic unknown cost"))
    assert budget.requests_used == 3
    assert budget.cost_reserved == Decimal("0.50")
    resumed = RequestBudget(
        max_requests=10,
        requests_used=budget.requests_used,
        cost_reserved=budget.cost_reserved,
    )
    await resumed.reserve(
        CostEstimate(upper_bound_usd=Decimal("0.25"), note="synthetic fixed cost")
    )
    assert resumed.requests_used == 4
    assert resumed.cost_reserved == Decimal("0.75")


@pytest.mark.asyncio
async def test_sibling_checks_parent_stop_before_local_consent_and_recording():
    parent = RequestBudget(max_requests=100, requests_used=20)
    with request_budget_scope(parent):
        failed = RequestBudget(max_requests=10, requests_used=3)
        sibling = RequestBudget(
            max_requests=10,
            requests_used=2,
            cost_reserved=Decimal("0.50"),
            confirm_before_requests=True,
            unknown_cost_authorizer=lambda remaining: pytest.fail("stopped parent prompted"),
            reservation_recorder=lambda *args: pytest.fail("stopped parent recorded"),
        )
    failed.stop_for_payment(AIPaymentRequiredError("synthetic payment status"))
    assert sibling._payment_stop_message is None
    with pytest.raises(AIPaymentRequiredError):
        await sibling.reserve(
            CostEstimate(upper_bound_usd=None, note="synthetic unknown cost"),
            approval_callback=lambda: pytest.fail("stopped parent invoked approval callback"),
        )
    assert sibling.requests_used == 2
    assert sibling.cost_reserved == Decimal("0.50")
    assert parent.requests_used == 20
    assert failed.requests_used == 3

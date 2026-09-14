"""Saved runs keep local ledgers while sharing one invocation's AI limits."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai.base import (
    BudgetExceededError,
    CostEstimate,
    RequestBudget,
    UnknownCostError,
    describe_with_recovery,
    request_budget_scope,
)
from test_ai_gemini import _request


def estimate():
    return CostEstimate(upper_bound_usd=Decimal("0.02"), note="test")


@pytest.mark.parametrize("request_limit,cost_limit", [(7, None), (None, Decimal("0.14"))])
async def test_concurrent_children_share_total_limits(request_limit, cost_limit):
    parent = RequestBudget(max_requests=request_limit, max_cost_usd=cost_limit)
    records = [[] for _ in range(10)]
    with request_budget_scope(parent):
        children = [
            RequestBudget(
                max_requests=None,
                reservation_recorder=lambda requests, cost, target=target: target.append(
                    (requests, cost)
                ),
            )
            for target in records
        ]
    outcomes = await asyncio.gather(
        *(child.reserve(estimate()) for _ in range(5) for child in children),
        return_exceptions=True,
    )
    assert outcomes.count(None) == 7
    assert sum(isinstance(item, BudgetExceededError) for item in outcomes) == 43
    assert parent.requests_used == sum(child.requests_used for child in children) == 7
    assert parent.cost_reserved == Decimal("0.14")
    for child, ledger in zip(children, records, strict=True):
        assert len(ledger) == child.requests_used
        assert [count for count, _ in ledger] == list(range(1, child.requests_used + 1))


async def test_child_refusal_does_not_charge_parent():
    parent = RequestBudget(max_requests=10, allow_unknown_cost=True)
    with request_budget_scope(parent):
        child = RequestBudget(
            max_requests=10,
            confirm_before_requests=True,
            unknown_cost_authorizer=lambda _: False,
        )
    with pytest.raises(UnknownCostError):
        await child.reserve(estimate())
    assert parent.requests_used == child.requests_used == 0
    assert parent.cost_reserved == child.cost_reserved == 0


async def test_local_recorder_failure_prevents_provider_call():
    parent = RequestBudget(max_requests=10, allow_unknown_cost=True)
    calls = []

    def fail_record(*_):
        raise OSError("checkpoint write failed")

    async def describe(_):
        calls.append(True)
        raise AssertionError("provider must not be called")

    with request_budget_scope(parent):
        child = RequestBudget(max_requests=10, reservation_recorder=fail_record)
    provider = SimpleNamespace(estimate=lambda _: estimate(), describe=describe)
    with pytest.raises(OSError, match="checkpoint write failed"):
        await describe_with_recovery(provider, _request(), child)
    assert calls == []
    assert child.requests_used == 0
    assert child.cost_reserved == 0
    assert parent.requests_used == 1
    assert parent.cost_reserved == Decimal("0.02")


async def test_scope_restores_and_resumed_usage_stays_local():
    parent = RequestBudget(max_requests=1)
    with request_budget_scope(parent):
        child = RequestBudget(max_requests=4, requests_used=3, cost_reserved=Decimal("0.06"))
    independent = RequestBudget(max_requests=1)
    await child.reserve(estimate())
    await independent.reserve(estimate())
    assert child.requests_used == 4
    assert child.cost_reserved == Decimal("0.08")
    assert parent.requests_used == independent.requests_used == 1
    assert parent.cost_reserved == Decimal("0.02")

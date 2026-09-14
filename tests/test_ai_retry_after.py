"""Provider-directed retry waits are shared, cancellable and counted once per HTTP call."""

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest

from mojilex_cli.ai import (
    AITransientError,
    CostEstimate,
    GeminiVisionProvider,
    RequestBudget,
    base,
    describe_with_recovery,
    gemini,
)
from mojilex_cli.concurrency import ai_slot, batch_limits, delay_ai_requests
from test_ai_gemini import _request
from test_ai_interactions import _sdk_client_factory

RETRY_TYPE = "type.googleapis.com/google.rpc.RetryInfo"


def error(header=None, body=None):
    headers = {} if header is None else {"Retry-After": header}
    return SimpleNamespace(response=httpx.Response(429, headers=headers), body=body)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("12", 12),
        ("0.25", 0.25),
        ("1e100", 1e100),
        ("nan", None),
        ("inf", None),
        ("-1", None),
        ("0", None),
        ("private-value", None),
    ],
)
def test_retry_after_header_accepts_only_positive_finite_delays(value, expected):
    assert gemini._retry_after_seconds(error(value)) == expected


def test_retry_after_http_date_uses_current_utc_and_ignores_past(monkeypatch):
    now = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(gemini, "datetime", Clock)
    assert (
        gemini._retry_after_seconds(
            error(format_datetime(now + timedelta(seconds=75), usegmt=True))
        )
        == 75
    )
    assert (
        gemini._retry_after_seconds(error(format_datetime(now - timedelta(seconds=1), usegmt=True)))
        is None
    )


@pytest.mark.parametrize("nested", [False, True])
def test_retry_info_is_exact_typed_and_longest_explicit_minimum_wins(nested):
    details = [
        {"@type": "private-unknown-type", "retryDelay": "999s"},
        {"@type": RETRY_TYPE, "retryDelay": "nan"},
        {"@type": RETRY_TYPE, "retryDelay": "-1s"},
        {"@type": RETRY_TYPE, "retryDelay": "18.125s"},
    ]
    body = {"details": details, "message": "private-credential wait 9999 seconds"}
    if nested:
        body = {"error": body}
    assert gemini._retry_after_seconds(error("12", body)) == 18.125
    assert gemini._retry_after_seconds(error("30", body)) == 30
    assert gemini._retry_after_seconds(error(None, {"message": "retry after 99 seconds"})) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header, retry_info, expected",
    [("7", None, [7, 7]), (None, "3s", [3, 3]), ("0.1", None, [1, 2]), (None, None, [1, 2])],
)
async def test_real_sdk_delay_retries_each_http_call_once_and_preserves_budget(
    monkeypatch, header, retry_info, expected
):
    calls = 0
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", sleep)

    def handle(request):
        nonlocal calls
        calls += 1
        body = {"code": "rate_limit_exceeded", "message": "private-synthetic-api-key"}
        if retry_info is not None:
            body["details"] = [{"@type": RETRY_TYPE, "retryDelay": retry_info}]
        return httpx.Response(
            429, headers={} if header is None else {"Retry-After": header}, json={"error": body}
        )

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    budget = RequestBudget(max_requests=3, allow_unknown_cost=True)
    try:
        with pytest.raises(AITransientError) as captured:
            await describe_with_recovery(provider, _request(), budget)
    finally:
        await provider._client.aio.aclose()
        provider._client.close()
    assert calls == budget.requests_used == 3
    assert delays == expected
    assert "private-synthetic" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_shared_cooldown_stops_queued_peer_without_holding_slot_and_cancels_huge_wait(
    monkeypatch,
):
    entered = asyncio.Event()
    release_error = asyncio.Event()
    waiting = asyncio.Event()
    calls = 0

    async def handle(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release_error.wait()
        return httpx.Response(
            429,
            headers={"Retry-After": "1e100"},
            json={"error": {"code": "rate_limit_exceeded", "message": "private"}},
        )

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    budget = RequestBudget(max_requests=10, allow_unknown_cost=True)
    tasks = []
    try:
        with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024) as limits:
            tasks.append(
                asyncio.create_task(
                    describe_with_recovery(
                        provider,
                        _request(),
                        budget,
                        progress_callback=lambda event: (
                            waiting.set() if event == "transport_retry" else None
                        ),
                    )
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            tasks.append(asyncio.create_task(describe_with_recovery(provider, _request(), budget)))
            await asyncio.sleep(0)  # Second request is already queued on the occupied AI slot.
            release_error.set()
            await asyncio.wait_for(waiting.wait(), 1)
            await asyncio.sleep(0)
            assert calls == budget.requests_used == 1
            assert limits.ai_cooldown.remaining() > 1e99
            assert limits.ai_slots._value == 1
            for task in tasks:
                task.cancel()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert limits.ai_slots._value == 1
            assert budget.requests_used == 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await provider._client.aio.aclose()
        provider._client.close()
    # A fresh operation does not inherit the old provider delay.
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024):
        async with ai_slot():
            pass


@pytest.mark.asyncio
async def test_cooldown_extends_across_nested_scopes_and_then_releases():
    loop = asyncio.get_running_loop()
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024) as outer:
        delay_ai_requests(0.04)
        first_deadline = outer.ai_cooldown.deadline
        with batch_limits(downloads=2, renders=2, ai=2, max_temp_bytes=2048) as inner:
            assert inner is outer
            delay_ai_requests(0.08)
            final_deadline = inner.ai_cooldown.deadline
            assert final_deadline > first_deadline
            delay_ai_requests(0.01)
            assert inner.ai_cooldown.deadline == final_deadline
        async with ai_slot():
            assert loop.time() >= final_deadline
        assert outer.ai_slots._value == 1


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0, -1, True])
def test_invalid_transient_delay_cannot_enter_scheduler(value):
    with pytest.raises(ValueError):
        AITransientError("safe", retry_after_seconds=value)


@pytest.mark.asyncio
async def test_huge_standalone_retry_uses_cancellable_bounded_timer_segments(monkeypatch):
    delays = []

    class Provider:
        name = "gemini"

        def estimate(self, request):
            return CostEstimate(upper_bound_usd=None, note="synthetic")

        async def describe(self, request):
            raise AITransientError("safe", retry_after_seconds=1e100)

    async def sleep(seconds):
        delays.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr(base.asyncio, "sleep", sleep)
    budget = RequestBudget(max_requests=3, allow_unknown_cost=True)
    with pytest.raises(asyncio.CancelledError):
        await describe_with_recovery(Provider(), _request(), budget)
    assert delays == [86400.0]
    assert budget.requests_used == 1

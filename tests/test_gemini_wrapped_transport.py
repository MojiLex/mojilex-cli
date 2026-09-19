"""Real SDK transport wrappers must not hide recoverable network failures."""

import asyncio
import json
import ssl
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
from google import genai

from mojilex_cli.ai import (
    AITransientError,
    BudgetExceededError,
    GeminiVisionProvider,
    RequestBudget,
    base,
    describe_with_recovery,
)
from mojilex_cli.ai.gemini import _transient_provider_error
from test_ai_gemini import _FakeModels, _request
from test_ai_interactions import _sdk_client_factory
from test_ai_semantic_facets import _payload


def test_flat_provider_error_reports_only_documented_code():
    from mojilex_cli.ai.gemini import _safe_provider_status

    class ProviderError(Exception):
        status_code = 400
        body: ClassVar[dict[str, str]] = {
            "code": "content_blocked",
            "message": "PRIVATE REQUEST CONTENT",
        }

    assert _safe_provider_status(ProviderError()) == " (http=400, provider_code=content_blocked)"
    ProviderError.body = {"code": "PRIVATE REQUEST CONTENT", "message": "PRIVATE REQUEST CONTENT"}
    assert _safe_provider_status(ProviderError()) == " (http=400)"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError, ssl.SSLEOFError])
@pytest.mark.parametrize("scenario", ["recovers", "exhausted", "budget"])
async def test_sdk_wrapped_transport_uses_only_budgeted_outer_retries(
    monkeypatch, failure, scenario
):
    calls = 0
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", sleep)

    def handle(request):
        nonlocal calls
        calls += 1
        if calls <= 2 or scenario != "recovers":
            if issubclass(failure, ssl.SSLError):
                raise failure("synthetic-private-network-body")
            raise failure("synthetic-private-network-body", request=request)
        return httpx.Response(
            200,
            json={
                "id": "synthetic-interaction",
                "object": "interaction",
                "status": "completed",
                "model": "gemini-test",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": json.dumps(_payload())}],
                    }
                ],
            },
        )

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    budget = RequestBudget(max_requests=1 if scenario == "budget" else 3, allow_unknown_cost=True)
    events = []
    try:
        if scenario == "recovers":
            result = await describe_with_recovery(
                provider, _request(), budget, progress_callback=events.append
            )
            assert result.batch.items[0].label == "E001"
        else:
            expected = BudgetExceededError if scenario == "budget" else AITransientError
            with pytest.raises(expected) as captured:
                await describe_with_recovery(
                    provider, _request(), budget, progress_callback=events.append
                )
            assert "synthetic-private" not in str(captured.value)
            assert "synthetic-key" not in str(captured.value)
            if scenario == "exhausted":
                assert captured.value.__cause__ is None
                assert captured.value.__suppress_context__ is True
    finally:
        await provider._client.aio.aclose()
        provider._client.close()
    assert (
        calls
        == budget.requests_used
        == events.count("request")
        == (1 if scenario == "budget" else 3)
    )
    assert delays == ([1] if scenario == "budget" else [1, 2])


def test_transport_cause_walk_is_bounded_cycle_safe_and_keeps_400_permanent():
    first, second = RuntimeError("private"), RuntimeError("private")
    first.__cause__ = second
    second.__cause__ = first
    assert not _transient_provider_error(first)
    chain = httpx.ReadTimeout("private")
    for _ in range(10):
        wrapper = RuntimeError("private")
        wrapper.__cause__ = chain
        chain = wrapper
    assert not _transient_provider_error(chain)
    permanent = RuntimeError("private")
    permanent.status_code = 400
    permanent.__cause__ = httpx.ConnectError("private")
    assert not _transient_provider_error(permanent)
    unrelated = RuntimeError("private")
    unrelated.__context__ = httpx.ReadTimeout("private")
    assert not _transient_provider_error(unrelated)


@pytest.mark.asyncio
@pytest.mark.parametrize("override, expected", [(None, 90), (12.5, 12.5)])
async def test_default_and_explicit_timeout_reach_both_sdk_and_outer_deadline(
    monkeypatch, override, expected
):
    sdk_options = []
    deadlines = []
    original_wait_for = asyncio.wait_for

    def client(**kwargs):
        sdk_options.append(kwargs["http_options"])
        return SimpleNamespace(aio=SimpleNamespace(interactions=_FakeModels()))

    async def wait_for(awaitable, timeout):  # noqa: ASYNC109 - matches asyncio.wait_for
        deadlines.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    options = {} if override is None else {"timeout_seconds": override}
    provider = GeminiVisionProvider(
        model="gemini-test", api_key="synthetic-key-not-real", **options
    )
    result = await provider.describe(_request())
    assert result.batch.items[0].label == "E001"
    assert deadlines == [expected]
    assert sdk_options[0].timeout == int(expected * 1000)
    assert sdk_options[0].retry_options.attempts == 0


def test_tls_certificate_failure_is_permanent_even_inside_transport_wrapper():
    certificate = ssl.SSLCertVerificationError("private certificate details")
    wrapped = httpx.ConnectError("private")
    wrapped.__cause__ = certificate
    assert not _transient_provider_error(wrapped)
    assert not _transient_provider_error(certificate)
    assert not _transient_provider_error(ssl.SSLError("unknown SSL failure"))
    assert _transient_provider_error(ssl.SSLEOFError("private"))
    assert _transient_provider_error(ssl.SSLZeroReturnError("private"))

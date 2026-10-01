import asyncio
import json
import traceback
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai.base import AIPaymentRequiredError, RequestBudget, request_budget_scope
from mojilex_cli.composition.verifier import verify_composition
from mojilex_cli.concurrency import batch_limits

PNG = b"\x89PNG\r\n\x1a\nsynthetic"


def verdict():
    return {
        "one_continuous_image": True,
        "separate_icons": False,
        "text_or_symbols": False,
        "pattern_or_repeated_objects": False,
        "uncertain": False,
        "seam_crossing_details": [
            "A branching root continues at the lower central vertical seam.",
            "The shoreline contour continues across the left horizontal seam.",
        ],
    }


class Client:
    def __init__(self, payload=None, error=None):
        self.payload = verdict() if payload is None else payload
        self.error = error
        self.calls = []
        self.aio = SimpleNamespace(interactions=self)
        self.sdk_configuration = SimpleNamespace(retry_config=SimpleNamespace(strategy="backoff"))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            status="completed", model="test", output_text=json.dumps(self.payload)
        )


async def check(client, budget=None):
    return await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=client,
        budget=budget or RequestBudget(max_requests=1, allow_unknown_cost=True),
    )


async def test_valid_veto_and_budget_reserved_before_request():
    records = []
    budget = RequestBudget(
        max_requests=1,
        allow_unknown_cost=True,
        reservation_recorder=lambda count, cost: records.append(count),
    )
    client = Client()
    assert await check(client, budget)
    assert records == [1]
    assert budget.requests_used == 1
    assert client.sdk_configuration.retry_config.strategy == "none"
    assert client.sdk_configuration.retry_config.max_retries == 0
    assert client.calls[0]["store"] is False
    assert client.calls[0]["generation_config"]["max_output_tokens"] == 2048
    assert not await check(client, budget)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "field",
    [
        "separate_icons",
        "text_or_symbols",
        "pattern_or_repeated_objects",
        "uncertain",
    ],
)
async def test_any_ambiguity_vetoes(field):
    payload = verdict()
    payload[field] = True
    assert not await check(Client(payload))


@pytest.mark.parametrize(
    "change",
    [
        {"one_continuous_image": False},
        {"one_continuous_image": "true"},
        {"extra": True},
        {"seam_crossing_details": ["same color", "same style"]},
        {"seam_crossing_details": ["A root continues across the left seam."] * 2},
        {"seam_crossing_details": []},
    ],
)
async def test_malformed_or_unsupported_verdict(change):
    assert not await check(Client(verdict() | change))


async def test_no_unknown_cost_prompt_or_budget_mutation():
    prompts = []
    budget = RequestBudget(
        max_requests=5, unknown_cost_authorizer=lambda count: prompts.append(count) or True
    )
    client = Client()
    assert not await check(client, budget)
    assert not client.calls
    assert not prompts
    assert budget.requests_used == 0
    assert not budget._unknown_cost_asked


async def test_failure_does_not_retry_and_consumes_reservation():
    budget = RequestBudget(max_requests=5, allow_unknown_cost=True)
    client = Client(error=ConnectionError("secret must not escape"))
    assert not await check(client, budget)
    assert len(client.calls) == budget.requests_used == 1


async def test_recorder_failure_prevents_api():
    def fail(count, cost):
        raise OSError("disk full")

    client = Client()
    budget = RequestBudget(max_requests=5, allow_unknown_cost=True, reservation_recorder=fail)
    assert not await check(client, budget)
    assert not client.calls


@pytest.mark.parametrize(
    "columns,rows",
    [(1, 1), (9, 1), (1, 9), (0, 2), (2, 0), (8, 4), (True, 2), (2, True), (2.0, 1)],
)
async def test_invalid_grid_does_not_call(columns, rows):
    client = Client()
    assert not await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=client,
        columns=columns,
        rows=rows,
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )
    assert not client.calls


@pytest.mark.parametrize("columns,rows", [(1, 2), (2, 1), (1, 8), (8, 1), (3, 8), (8, 3)])
async def test_strips_and_extended_grids_keep_strict_acceptance(columns, rows):
    client = Client()
    assert await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=client,
        columns=columns,
        rows=rows,
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )
    prompt = client.calls[0]["input"][0]["content"][0]["text"]
    if columns == 1:
        assert "no internal vertical joins" in prompt
    if rows == 1:
        assert "no internal horizontal joins" in prompt
    assert "require two distinct visible features" in prompt


@pytest.mark.parametrize("columns,rows", [(1, 2), (2, 1)])
@pytest.mark.parametrize(
    "change",
    [
        {"seam_crossing_details": ["A root continues across the left seam."]},
        {"separate_icons": True},
        {"pattern_or_repeated_objects": True},
        {"uncertain": True},
    ],
)
async def test_pairs_do_not_relax_evidence(columns, rows, change):
    assert not await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=Client(verdict() | change),
        columns=columns,
        rows=rows,
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )


async def test_explicit_grid_reaches_prompt():
    client = Client()
    assert await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=client,
        columns=6,
        rows=2,
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )
    prompt = client.calls[0]["input"][0]["content"][0]["text"]
    assert "exactly 6 columns and 2 rows" in prompt
    assert "k/6" in prompt


@pytest.mark.parametrize("error", [None, ConnectionError("test")])
async def test_owned_client_closed_on_success_and_failure(monkeypatch, error):
    client = Client(error=error)
    closed = []

    async def aclose():
        closed.append("async")

    client.aio.aclose = aclose
    client.close = lambda: closed.append("sync")
    monkeypatch.setattr(
        "mojilex_cli.composition.verifier.GeminiVisionProvider",
        lambda **kwargs: SimpleNamespace(_client=client),
    )
    result = await verify_composition(
        PNG,
        model="test",
        api_key="test",
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )
    assert result is (error is None)
    assert closed == ["async", "sync"]


async def test_injected_client_not_closed():
    client = Client()
    closed = []

    async def aclose():
        closed.append("async")

    client.aio.aclose = aclose
    client.close = lambda: closed.append("sync")
    assert await check(client)
    assert not closed


async def test_audit_focuses_differ_without_changing_schema_or_grid():
    prompts = []
    schemas = []
    for audit in ("continuity", "independent_objects", "layout"):
        client = Client()
        assert await verify_composition(
            PNG,
            model="test",
            api_key=None,
            client=client,
            columns=1,
            rows=2,
            audit=audit,
            budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
        )
        prompt = client.calls[0]["input"][0]["content"][0]["text"]
        prompts.append(prompt)
        schemas.append(client.calls[0]["response_format"])
        assert "no internal vertical joins" in prompt
        assert "require two distinct visible features" in prompt
        assert "at least two DISTINCT" in prompt
    assert len(set(prompts)) == 3
    assert schemas[0] == schemas[1] == schemas[2]
    assert "self-contained background" in prompts[1]
    assert "wrong neighbors" in prompts[2]


@pytest.mark.parametrize("audit", ["continuity", "independent_objects", "layout"])
async def test_all_audits_veto_uncertainty(audit):
    assert not await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=Client(verdict() | {"uncertain": True}),
        columns=2,
        rows=1,
        audit=audit,
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )


async def test_unknown_audit_never_calls():
    client = Client()
    assert not await verify_composition(
        PNG,
        model="test",
        api_key=None,
        client=client,
        audit="unknown",
        budget=RequestBudget(max_requests=1, allow_unknown_cost=True),
    )
    assert not client.calls


class HTTPFailure(Exception):
    def __init__(self, status):
        super().__init__("raw provider response must remain private")
        self.status_code = status


async def test_payment_required_preserves_reservation_and_resume_ledger():
    records = []
    budget = RequestBudget(
        max_requests=20,
        allow_unknown_cost=True,
        requests_used=7,
        cost_reserved=Decimal("1.25"),
        reservation_recorder=lambda count, cost: records.append((count, cost)),
    )
    client = Client(error=HTTPFailure(402))
    with pytest.raises(AIPaymentRequiredError, match="HTTP 402") as raised:
        await check(client, budget)
    assert "raw provider response" not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    assert "raw provider response" not in "".join(traceback.format_exception(raised.value))
    assert records == [(8, Decimal("1.25"))]
    assert budget.requests_used == 8
    assert len(client.calls) == 1
    with pytest.raises(AIPaymentRequiredError):
        await check(Client(), budget)
    assert records == [(8, Decimal("1.25"))]
    # Resuming after billing is resolved starts a fresh invocation with the
    # already persisted conservative ledger, never refunding the failed call.
    resumed = RequestBudget(
        max_requests=20,
        allow_unknown_cost=True,
        requests_used=records[-1][0],
        cost_reserved=records[-1][1],
    )
    assert await check(Client(), resumed)
    assert resumed.requests_used == 9
    assert resumed.cost_reserved == Decimal("1.25")


async def test_payment_required_blocks_queued_peer_before_ai_slot_release():
    started = asyncio.Event()
    release = asyncio.Event()

    class WaitingClient(Client):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            started.set()
            await release.wait()
            raise HTTPFailure(402)

    parent = RequestBudget(max_requests=20, allow_unknown_cost=True)
    with request_budget_scope(parent):
        first_budget = RequestBudget(max_requests=20, allow_unknown_cost=True)
        peer_budget = RequestBudget(max_requests=20, allow_unknown_cost=True)
    first = WaitingClient()
    peer = Client()
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1024):
        task = asyncio.create_task(check(first, first_budget))
        await asyncio.wait_for(started.wait(), timeout=2)
        peer_task = asyncio.create_task(check(peer, peer_budget))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(task, peer_task, return_exceptions=True)
    assert all(isinstance(result, AIPaymentRequiredError) for result in results)
    assert len(first.calls) == 1
    assert not peer.calls
    assert first_budget.requests_used == parent.requests_used == 1
    assert peer_budget.requests_used == 0


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, True, "402"])
async def test_nonpayment_failure_keeps_optional_veto_behavior(status):
    error = HTTPFailure(status)
    error.__cause__ = HTTPFailure(402)
    # The first explicit HTTP status wins. Non-integer status values alone do
    # not supply payment evidence, so omit a payment cause in those cases.
    if type(status) is not int:
        error.__cause__ = None
    budget = RequestBudget(max_requests=5, allow_unknown_cost=True)
    assert not await check(Client(error=error), budget)
    assert await check(Client(), budget)
    assert budget.requests_used == 2


async def test_owned_client_closes_before_payment_failure_propagates(monkeypatch):
    client = Client(error=HTTPFailure(402))
    closed = []

    async def aclose():
        closed.append("async")

    client.aio.aclose = aclose
    client.close = lambda: closed.append("sync")
    monkeypatch.setattr(
        "mojilex_cli.composition.verifier.GeminiVisionProvider",
        lambda **kwargs: SimpleNamespace(_client=client),
    )
    budget = RequestBudget(max_requests=5, allow_unknown_cost=True)
    with pytest.raises(AIPaymentRequiredError):
        await verify_composition(PNG, model="test", api_key=None, budget=budget)
    assert closed == ["async", "sync"]
    assert budget.requests_used == len(client.calls) == 1

import asyncio
import json
import subprocess
import sys
import textwrap
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai import (
    AIError,
    AIOutputError,
    BudgetExceededError,
    CostEstimate,
    DescriptionRequest,
    GeminiVisionProvider,
    ModelPricing,
    RequestBudget,
    UnknownCostError,
    VisionContext,
    VisionImage,
    describe_with_recovery,
)
from mojilex_cli.runs import RunStore, new_checkpoint


def _payload(label: str = "E001") -> dict[str, object]:
    localized = {
        "text": "A synthetic smiling face.",
        "motion_status": "not_applicable",
        "usage": ["agreement"],
    }
    return {
        "items": [
            {
                "label": label,
                "descriptions": {"ru": localized, "en": localized},
                "facets": {
                    "text_content": {"status": "none", "dynamics": "stable", "items": []},
                    "content_types": ["reaction"],
                    "styles": ["flat"],
                    "suggested_uses": ["message-accent"],
                    "uncertainties": [],
                },
                "semantic_tags": ["face", "smile"],
                "content": {"rating": "general", "warnings": []},
            }
        ]
    }


def _request() -> DescriptionRequest:
    return DescriptionRequest(
        model="gemini-test",
        images=(VisionImage(data=b"\x89PNG\r\n\x1a\nsynthetic", labels=("E001",)),),
        expected_labels=("E001",),
        context={
            "E001": VisionContext(
                needs_repainting=False,
                frame_count=1,
                background_variants=("light",),
            )
        },
    )


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def get(self, *, model: str) -> object:
        return {"name": model}

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(_payload()),
            model=kwargs["model"],
            status="completed",
            usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=20),
        )


@pytest.mark.asyncio
async def test_gemini_uses_official_async_structured_schema() -> None:
    models = _FakeModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models, interactions=models))
    provider = GeminiVisionProvider(model="gemini-test", client=client)
    await provider.validate_credentials()
    result = await provider.describe(_request())

    assert result.batch.items[0].label == "E001"
    assert result.model_revision is None
    call = models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"]["type"] == "object"
    assert call["generation_config"]["max_output_tokens"] == 8192
    assert call["store"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"status": "synthetic-private-status"}, "interaction_incomplete"),
        ({"model": "synthetic-private-model"}, "model_mismatch"),
        ({"model": {"synthetic-private-key": "private-value"}}, "model_mismatch"),
        ({"output_text": None}, "output_missing"),
        ({"output_text": "  \n"}, "output_missing"),
        ({"output_text": {"synthetic-private-key": "private-value"}}, "output_missing"),
        ({"output_text": "{synthetic-private-json"}, "invalid_json"),
    ],
)
async def test_gemini_response_diagnostics_are_distinct_and_safe(overrides, reason):
    async def create(**kwargs):
        response = {
            "status": "completed",
            "model": "gemini-test",
            "output_text": json.dumps(_payload()),
        }
        response.update(overrides)
        return SimpleNamespace(**response)

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    with pytest.raises(AIOutputError) as captured:
        await provider.describe(_request())
    assert str(captured.value) == "Gemini structured response: " + reason
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "enum", "text", "extra", "many"])
async def test_gemini_schema_diagnostics_use_only_known_paths_and_codes(failure):
    payload = _payload()
    item = payload["items"][0]
    if failure == "missing":
        del item["descriptions"]["ru"]["text"]
        expected = "$.items[].descriptions.ru.text (missing)"
    elif failure == "enum":
        item["content"]["rating"] = "synthetic-private-value"
        expected = "$.items[].content.rating (literal_error)"
    elif failure == "text":
        item["descriptions"]["ru"]["text"] = "<private>synthetic-private-value</private>"
        expected = "$.items[].descriptions.ru.text (value_error)"
    elif failure == "extra":
        item["descriptions"]["ru"]["synthetic-private-key"] = "synthetic-private-value"
        expected = "$.items[].descriptions.ru.<unknown-field> (extra_forbidden)"
    else:
        payload["items"] = [{} for _ in range(16)]
        expected = "additional errors omitted"

    async def create(**kwargs):
        return SimpleNamespace(
            status="completed", model="gemini-test", output_text=json.dumps(payload)
        )

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create))),
    )
    with pytest.raises(AIOutputError) as captured:
        await provider.describe(_request())
    message = str(captured.value)
    assert message.startswith("Gemini structured response: schema_validation: ")
    assert expected in message
    assert "synthetic-private" not in message
    assert "E001" not in message
    assert len(message) < 500
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["describe", "validate_credentials"])
async def test_gemini_enforces_timeout_even_when_sdk_does_not(operation: str) -> None:
    cancelled = asyncio.Event()

    async def hanging(**kwargs: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    resource = SimpleNamespace(create=hanging, get=hanging)
    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(models=resource, interactions=resource)),
        timeout_seconds=0.01,
    )
    with pytest.raises(AIError, match=r"timed out after 0\.01 seconds"):
        if operation == "describe":
            await provider.describe(_request())
        else:
            await provider.validate_credentials()
    assert cancelled.is_set()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_gemini_rejects_unbounded_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        GeminiVisionProvider(model="test", client=object(), timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_recovery_reports_retry_without_repeating_budget_prompt(monkeypatch) -> None:
    models = _FakeModels()
    valid_create = models.create
    calls = 0

    async def create(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(status="completed", model="gemini-test", output_text="{}")
        return await valid_create(**kwargs)

    monkeypatch.setattr(models, "create", create)
    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(models=models, interactions=models)),
    )
    prompts: list[int] = []
    events: list[str] = []
    budget = RequestBudget(
        max_requests=2,
        unknown_cost_authorizer=lambda count: prompts.append(count) is None,
    )
    result = await describe_with_recovery(
        provider, _request(), budget, progress_callback=events.append
    )
    assert result.batch.items[0].label == "E001"
    assert prompts == [2]
    assert events == ["approval", "request", "retry", "approval", "request"]
    assert budget.requests_used == 2


@pytest.mark.asyncio
async def test_request_budget_is_hard_and_unknown_cost_requires_opt_in() -> None:
    budget = RequestBudget(max_requests=1)
    with pytest.raises(UnknownCostError):
        await budget.reserve(CostEstimate(upper_bound_usd=None, note="unknown"))

    known = RequestBudget(max_requests=1, max_cost_usd=Decimal("0.01"))
    await known.reserve(CostEstimate(upper_bound_usd=Decimal("0.01"), note="known"))
    with pytest.raises(BudgetExceededError):
        await known.reserve(CostEstimate(upper_bound_usd=Decimal("0"), note="known"))


def test_request_budget_resume_cannot_reset_or_exceed_previous_usage() -> None:
    resumed = RequestBudget(
        max_requests=3,
        max_cost_usd=Decimal("1.00"),
        requests_used=2,
        cost_reserved=Decimal("0.75"),
    )
    assert resumed.requests_used == 2
    assert resumed.cost_reserved == Decimal("0.75")
    with pytest.raises(ValueError):
        RequestBudget(max_requests=1, requests_used=2)


@pytest.mark.asyncio
async def test_unknown_cost_authorization_once_for_bounded_run_budget() -> None:
    planned: list[int] = []
    budget = RequestBudget(
        max_requests=2,
        unknown_cost_authorizer=lambda requests: planned.append(requests) is None,
    )

    await budget.reserve(CostEstimate(upper_bound_usd=None, note="unknown"))
    await budget.reserve(CostEstimate(upper_bound_usd=None, note="unknown"))

    assert planned == [2]
    assert budget.requests_used == 2


@pytest.mark.asyncio
async def test_unknown_cost_concurrent_consent_is_once_and_cannot_exceed_limit() -> None:
    planned: list[int] = []
    budget = RequestBudget(
        max_requests=5,
        requests_used=2,
        unknown_cost_authorizer=lambda requests: planned.append(requests) is None,
    )
    estimate = CostEstimate(upper_bound_usd=None, note="unknown")
    await asyncio.gather(*(budget.reserve(estimate) for _ in range(3)))
    assert planned == [3]
    assert budget.requests_used == 5
    with pytest.raises(BudgetExceededError):
        await budget.reserve(estimate)


@pytest.mark.asyncio
async def test_unknown_cost_refusal_is_not_asked_again() -> None:
    planned: list[int] = []
    budget = RequestBudget(
        max_requests=5,
        unknown_cost_authorizer=lambda requests: bool(planned.append(requests)),
    )
    for _ in range(3):
        with pytest.raises(UnknownCostError):
            await budget.reserve(CostEstimate(upper_bound_usd=None, note="unknown"))
    assert planned == [5]
    assert budget.requests_used == 0


@pytest.mark.asyncio
async def test_unknown_cost_authorizer_exception_does_not_reprompt() -> None:
    planned: list[int] = []

    def refuse(requests: int) -> bool:
        planned.append(requests)
        raise RuntimeError("refused")

    budget = RequestBudget(max_requests=5, unknown_cost_authorizer=refuse)
    estimate = CostEstimate(upper_bound_usd=None, note="unknown")
    with pytest.raises(RuntimeError, match="refused"):
        await budget.reserve(estimate)
    with pytest.raises(UnknownCostError):
        await budget.reserve(estimate)
    assert planned == [5]
    assert budget.requests_used == 0


@pytest.mark.asyncio
async def test_unknown_cost_authorization_fails_closed_without_a_request() -> None:
    budget = RequestBudget(
        max_requests=1,
        unknown_cost_authorizer=lambda _requests: False,
    )

    with pytest.raises(UnknownCostError):
        await budget.reserve(CostEstimate(upper_bound_usd=None, note="unknown"))
    assert budget.requests_used == 0


@pytest.mark.asyncio
async def test_budget_reservation_is_recorded_before_becoming_requestable() -> None:
    recorded: list[tuple[int, Decimal]] = []
    budget = RequestBudget(
        max_requests=2,
        max_cost_usd=Decimal("1"),
        reservation_recorder=lambda requests, cost: recorded.append((requests, cost)),
    )

    await budget.reserve(CostEstimate(upper_bound_usd=Decimal("0.25"), note="known"))

    assert recorded == [(1, Decimal("0.25"))]
    assert budget.requests_used == 1
    assert budget.cost_reserved == Decimal("0.25")


@pytest.mark.asyncio
async def test_budget_recorder_failure_prevents_in_memory_reservation() -> None:
    def fail_to_persist(_requests: int, _cost: Decimal) -> None:
        raise OSError("synthetic durable checkpoint failure")

    budget = RequestBudget(max_requests=1, reservation_recorder=fail_to_persist)

    with pytest.raises(OSError, match="durable checkpoint"):
        await budget.reserve(CostEstimate(upper_bound_usd=Decimal("0"), note="known"))
    assert budget.requests_used == 0
    assert budget.cost_reserved == Decimal("0")


def test_hard_exit_after_reservation_keeps_durable_usage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = tmp_path / "dataset"
    repository.mkdir()
    runs = tmp_path / "runs"
    run_id = "mlxrun_" + "a" * 32
    store = RunStore(runs, repository_root=repository)
    store.save(
        new_checkpoint(
            command="add",
            safe_parameters={},
            cli_version="0.1.0",
            schema_version="1",
            target_repository=str(repository.resolve()),
            base_revision="a" * 40,
            run_id=run_id,
        )
    )
    child = textwrap.dedent(
        """
        import asyncio
        import os
        import sys
        from decimal import Decimal
        from pathlib import Path

        from mojilex_cli.ai import CostEstimate, RequestBudget
        from mojilex_cli.pipeline.runner import _persist_budget_reservation
        from mojilex_cli.runs import RunStore

        runs = Path(sys.argv[1])
        repository = Path(sys.argv[2])
        run_id = sys.argv[3]
        store = RunStore(runs, repository_root=repository)
        checkpoint = store.load(run_id)

        def record(requests: int, cost: Decimal) -> None:
            global checkpoint
            checkpoint = _persist_budget_reservation(
                checkpoint,
                store,
                requests_used=requests,
                cost_reserved=cost,
            )

        budget = RequestBudget(max_requests=1, reservation_recorder=record)
        asyncio.run(
            budget.reserve(CostEstimate(upper_bound_usd=Decimal("0.25"), note="known"))
        )
        os._exit(77)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child, str(runs), str(repository), run_id],
        cwd=repository.parent,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 77, completed.stderr.decode(errors="replace")
    resumed = store.load(run_id)
    assert resumed.ai_requests_used == 1
    assert resumed.ai_cost_reserved_usd == Decimal("0.25")
    budget = RequestBudget(
        max_requests=1,
        max_cost_usd=Decimal("1"),
        requests_used=resumed.ai_requests_used,
        cost_reserved=resumed.ai_cost_reserved_usd,
    )
    with pytest.raises(BudgetExceededError):
        asyncio.run(budget.reserve(CostEstimate(upper_bound_usd=Decimal("0"), note="known")))


def test_pricing_must_be_external_and_model_specific() -> None:
    pricing = ModelPricing(
        model="gemini-test",
        input_usd_per_million_tokens=Decimal("1"),
        output_usd_per_million_tokens=Decimal("2"),
        conservative_tokens_per_image=300,
        conservative_output_tokens=500,
        updated_at=date(2026, 9, 11),
    )
    provider = GeminiVisionProvider(model="gemini-test", client=SimpleNamespace(), pricing=pricing)
    estimate = provider.estimate(_request())
    assert estimate.known
    assert estimate.pricing_updated_at == date(2026, 9, 11)
    assert estimate.upper_bound_usd >= Decimal(8192) * Decimal(2) / Decimal(1_000_000)


def test_gemini_usage_includes_billable_thinking_tokens() -> None:
    from mojilex_cli.ai.gemini import _output_token_count

    assert (
        _output_token_count(SimpleNamespace(total_output_tokens=20, total_thought_tokens=35)) == 55
    )
    assert _output_token_count(SimpleNamespace(total_output_tokens=20)) == 20
    assert _output_token_count(SimpleNamespace(total_thought_tokens=35)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,status,expected",
    [
        (400, "INVALID_ARGUMENT", " (http=400, status=INVALID_ARGUMENT)"),
        (403, "PERMISSION_DENIED", " (http=403, status=PERMISSION_DENIED)"),
        (429, "RESOURCE_EXHAUSTED", " (http=429, status=RESOURCE_EXHAUSTED)"),
        (429, "SYNTHETIC_SECRET_VALUE", " (http=429)"),
        ("429?synthetic-credential", "bad status\nrequest-data", ""),
        (True, ["RESOURCE_EXHAUSTED"], ""),
        (999999999, None, ""),
    ],
)
async def test_provider_diagnostics_never_include_error_message_or_request(code, status, expected):
    from mojilex_cli.ai import AIError

    class SyntheticProviderError(Exception):
        pass

    error = SyntheticProviderError("synthetic-credential-not-real; private request and response")
    error.code = code
    error.status = status
    error.details = {"synthetic-secret": "never-output-this"}

    class FailingModels:
        async def create(self, **kwargs):
            raise error

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=FailingModels())),
    )
    with pytest.raises(AIError) as captured:
        await provider.describe(_request())
    assert str(captured.value) == "Gemini request failed: SyntheticProviderError" + expected
    assert "synthetic-credential" not in str(captured.value)
    assert "private request" not in str(captured.value)
    assert "never-output-this" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True

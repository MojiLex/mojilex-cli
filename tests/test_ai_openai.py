"""Offline HTTP contract, validation, billing stop and shared-budget coverage."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.ai import (
    AIError,
    AIOutputError,
    AIPaymentRequiredError,
    AITransientError,
    BudgetExceededError,
    DescriptionBatch,
    DescriptionRequest,
    OpenAIVisionProvider,
    RequestBudget,
    VisionContext,
    VisionImage,
    default_registry,
    describe_with_recovery,
)
from mojilex_cli.ai.openai_schema import openai_transport_schema
from mojilex_cli.ai.prompts import (
    gemini_request_parameters_sha256,
    openai_request_parameters,
    prompt_contract_scope,
    request_parameters_sha256,
)
from mojilex_cli.commands import benchmark as benchmark_commands
from mojilex_cli.commands import packs, system, workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.composition.verifier import verify_composition
from mojilex_cli.config import Credentials, MojiLexConfig
from mojilex_cli.config.provider_credentials import provider_api_key
from mojilex_cli.pipeline.runner import _default_generation_metadata

MODEL = "gpt-6-luna"
PNG = b"\x89PNG\r\n\x1a\nsynthetic"


def payload():
    localized = {
        "text": "A synthetic smiling face.",
        "motion_status": "not_applicable",
        "usage": ["agreement"],
    }
    batch = {
        "items": [
            {
                "label": "E001",
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
    return DescriptionBatch.model_validate(batch).model_dump(mode="json")


def response(text=None):
    return {
        "status": "completed",
        "model": MODEL,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text or json.dumps(payload())}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }


def request():
    return DescriptionRequest(
        model=MODEL,
        images=(VisionImage(data=PNG, labels=("E001",)),),
        expected_labels=("E001",),
        context={
            "E001": VisionContext(
                needs_repainting=False,
                frame_count=1,
                background_variants=("light",),
            )
        },
    )


async def call(handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await OpenAIVisionProvider(model=MODEL, client=client).describe(request())


async def test_responses_contract_uses_png_strict_schema_and_no_hidden_retries():
    calls = []

    def handle(incoming):
        calls.append(incoming)
        assert incoming.headers["authorization"] == "Bearer " + "synthetic-openai-credential"
        if incoming.method == "GET":
            assert incoming.url.path == "/v1/models/gpt-6-luna"
            return httpx.Response(200, json={"id": MODEL})
        body = json.loads(incoming.content)
        assert body["model"] == MODEL
        assert body["store"] is body["stream"] is body["background"] is False
        assert body["reasoning"] == {"effort": "none"}
        assert body["max_output_tokens"] == 8192
        assert body["text"]["format"] == openai_request_parameters()["text"]["format"]
        message = body["input"][0]
        assert message["role"] == "user"
        assert message["content"][0]["type"] == "input_text"
        assert "E001" in message["content"][0]["text"]
        image = message["content"][1]
        assert image["detail"] == "high"
        assert image["image_url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json=response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = OpenAIVisionProvider(
            model=MODEL, api_key="synthetic-openai-credential", client=client
        )
        await provider.validate_credentials()
        result = await provider.describe(request())
    assert len(calls) == 2
    assert result.provider == "openai" and result.model_revision is None
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 20
    assert result.usage.estimated_cost_usd is None


def test_strict_schema_requires_all_fields_preserves_enums_and_nullable_motion():
    original = DescriptionBatch.model_json_schema()
    saved = deepcopy(original)
    strict = openai_transport_schema(original)
    assert original == saved

    def check(node):
        assert "default" not in node
        if "properties" in node:
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for key in ("properties", "$defs"):
            for value in node.get(key, {}).values():
                check(value)
        if "items" in node:
            check(node["items"])
        for value in node.get("anyOf", []):
            check(value)

    check(strict)
    localized = strict["$defs"]["LocalizedDescription"]["properties"]
    assert {part["type"] for part in localized["motion"]["anyOf"]} == {"string", "null"}
    assert "not_applicable" in localized["motion_status"]["enum"]
    assert strict["properties"]["items"]["minItems"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "object", "properties": {}},
        {"type": "object", "allOf": []},
        {"type": "array", "items": {"type": "string"}},
    ],
)
def test_transport_schema_fails_closed_for_unsupported_contract(bad):
    with pytest.raises(ValueError):
        openai_transport_schema(bad)


def test_provider_parameter_hashes_are_distinct_and_cached_copies_are_isolated():
    with prompt_contract_scope():
        assert request_parameters_sha256("gemini") == gemini_request_parameters_sha256()
        openai_hash = request_parameters_sha256("openai")
        assert openai_hash != request_parameters_sha256("gemini")
        changed = openai_request_parameters()
        changed["text"]["format"]["schema"].clear()
        assert request_parameters_sha256("openai") == openai_hash
        assert openai_request_parameters()["text"]["format"]["schema"]
    assert default_registry().names() == ("gemini", "openai")


@pytest.mark.parametrize(
    "failure,code",
    [
        ("incomplete", "response_incomplete"),
        ("model", "model_mismatch"),
        ("refusal", "refusal"),
        ("missing", "output_missing"),
        ("tool", "unexpected_output"),
        ("json", "invalid_json"),
        ("enum", "schema_validation"),
        ("extra", "schema_validation"),
    ],
)
async def test_invalid_answers_are_rejected_without_exposing_provider_text(failure, code):
    body = response()
    if failure == "incomplete":
        body["status"] = "incomplete"
    elif failure == "model":
        body["model"] = "synthetic-private-model"
    elif failure == "refusal":
        body["output"][0]["content"] = [{"type": "refusal", "refusal": "private-refusal"}]
    elif failure == "missing":
        body["output"] = []
    elif failure == "tool":
        body["output"] = [{"type": "function_call", "arguments": "private-arguments"}]
    else:
        batch = payload()
        if failure == "enum":
            batch["items"][0]["content"]["rating"] = "private-rating"
        elif failure == "extra":
            batch["items"][0]["private-field"] = "private-value"
        body = response("{private-json" if failure == "json" else json.dumps(batch))
    with pytest.raises(AIOutputError, match=code) as captured:
        await call(lambda _: httpx.Response(200, json=body))
    assert "private" not in str(captured.value)
    assert captured.value.__cause__ is None


async def test_wrong_labels_and_semantic_cross_fields_are_rejected():
    batch = payload()
    batch["items"][0]["label"] = "E002"
    with pytest.raises(AIOutputError, match="expected label"):
        await call(lambda _: httpx.Response(200, json=response(json.dumps(batch))))
    batch = payload()
    batch["items"][0]["descriptions"]["en"]["motion"] = "private-motion"
    with pytest.raises(AIOutputError, match="schema_validation"):
        await call(lambda _: httpx.Response(200, json=response(json.dumps(batch))))


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_transient_http_errors_retain_retry_after_and_have_one_attempt(status):
    calls = []

    def handler(incoming):
        calls.append(incoming)
        return httpx.Response(
            status,
            headers={"retry-after": "3"},
            json={
                "error": {
                    "message": "private-error",
                    "code": {"private": "value"},
                }
            },
        )

    with pytest.raises(AITransientError) as captured:
        await call(handler)
    assert len(calls) == 1
    assert captured.value.retry_after_seconds == 3
    assert "private" not in str(captured.value)


@pytest.mark.parametrize(
    "status,code", [(402, None), (429, "insufficient_quota"), (403, "billing_hard_limit_reached")]
)
async def test_payment_errors_stop_shared_budget_without_retries(status, code):
    calls = []

    def handler(incoming):
        calls.append(incoming)
        return httpx.Response(status, json={"error": {"code": code, "message": "private"}})

    budget = RequestBudget(max_requests=5, allow_unknown_cost=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIVisionProvider(model=MODEL, client=client)
        with pytest.raises(AIPaymentRequiredError):
            await describe_with_recovery(provider, request(), budget)
        with pytest.raises(AIPaymentRequiredError):
            await describe_with_recovery(provider, request(), budget)
    assert len(calls) == 1 and budget.requests_used == 1


async def test_invalid_output_retry_cannot_exceed_persisted_request_budget():
    records = []
    calls = []

    def handler(incoming):
        assert records == [1]
        calls.append(incoming)
        return httpx.Response(200, json=response("{private-json"))

    budget = RequestBudget(
        max_requests=1, allow_unknown_cost=True, reservation_recorder=lambda n, _: records.append(n)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BudgetExceededError):
            await describe_with_recovery(
                OpenAIVisionProvider(model=MODEL, client=client), request(), budget
            )
    assert len(calls) == 1 and records == [1]


async def test_transport_timeout_redirect_and_cancellation_do_not_leak_secrets():
    def timeout(incoming):
        raise httpx.ReadTimeout("private-credential", request=incoming)

    with pytest.raises(AITransientError) as captured:
        await call(timeout)
    assert "private" not in str(captured.value)
    with pytest.raises(AIError, match="HTTP 307"):
        await call(lambda _: httpx.Response(307, headers={"location": "https://example.invalid/"}))

    async def cancelled(_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await call(cancelled)


async def test_openai_composition_veto_charges_budget_before_request():
    calls = []
    budget = RequestBudget(max_requests=1, allow_unknown_cost=True)
    verdict = {
        "one_continuous_image": True,
        "separate_icons": False,
        "text_or_symbols": False,
        "pattern_or_repeated_objects": False,
        "uncertain": False,
        "seam_crossing_details": [
            "The branching root continues across the lower vertical seam.",
            "The shoreline contour continues across the left horizontal seam.",
        ],
    }

    def handler(incoming):
        assert budget.requests_used == 1
        calls.append(incoming)
        body = json.loads(incoming.content)
        assert body["max_output_tokens"] == 2048
        assert body["text"]["format"]["name"] == "mojilex_composition_verdict"
        return httpx.Response(200, json=response(json.dumps(verdict)))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        kwargs = dict(
            model=MODEL, api_key=None, budget=budget, client=client, provider_name="openai"
        )
        assert await verify_composition(PNG, **kwargs)
        assert not await verify_composition(PNG, **kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "explicit,stored,expected",
    [
        (None, "openai", "OPENAI_API_KEY"),
        ("openai", "gemini", "OPENAI_API_KEY"),
        (None, "gemini", "GEMINI_API_KEY"),
    ],
)
def test_describe_prompts_for_saved_or_explicit_provider(monkeypatch, explicit, stored, expected):
    checkpoint = SimpleNamespace(safe_parameters={"provider": stored})
    monkeypatch.setattr(cli, "load_config", MojiLexConfig)
    monkeypatch.setattr(packs, "resolve_pack_run", lambda *_a, **_kw: checkpoint)
    assert cli._authoring_secret_names(explicit, selectors=("pack",)) == (
        "TELEGRAM_BOT_TOKEN",
        expected,
    )


def test_openai_setting_selects_its_key_without_mutating_other_credentials(monkeypatch):
    credentials = Credentials(gemini_api_key="synthetic-gemini", openai_api_key="synthetic-openai")
    assert provider_api_key(credentials, "openai") == "synthetic-openai"
    assert provider_api_key(credentials, "gemini") == "synthetic-gemini"
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: MojiLexConfig.model_validate(
            {
                "ai": {"provider": "openai", "model": MODEL},
            }
        ),
    )
    prompts = []
    stored = []
    monkeypatch.setattr(
        cli,
        "_secret_prompt",
        lambda **_: lambda name: prompts.append(name) or "synthetic-hidden-value",
    )
    monkeypatch.setattr(
        cli,
        "config_set_credentials_command",
        lambda values: stored.append(values) or CommandResult(),
    )
    result = CliRunner().invoke(cli.app, ["config", "set-credentials", "--no-telegram"])
    assert result.exit_code == 0, result.output
    assert set(stored[0]) == {"OPENAI_API_KEY"}
    assert len(prompts) == 1
    assert "synthetic-hidden-value" not in result.output


def test_resume_keeps_saved_openai_secret_after_current_settings_change(monkeypatch):
    checkpoint = SimpleNamespace(run_id="synthetic-run", safe_parameters={"provider": "openai"})
    monkeypatch.setattr(packs, "resolve_pack_run", lambda *_a, **_kw: checkpoint)
    monkeypatch.setattr(packs, "_pack_phase_status", lambda *_a: ("describe", "partial"))
    monkeypatch.setattr(cli, "load_config", MojiLexConfig)
    captured = []
    monkeypatch.setattr(
        cli,
        "_with_runtime_secrets",
        lambda action, *, names, prompt: captured.append(names) or action(),
    )
    monkeypatch.setattr(workflow, "resume_command", lambda *_a, **_kw: CommandResult())
    monkeypatch.setattr(cli, "_pack_action", lambda action, _: action())
    result = CliRunner().invoke(cli.app, ["resume", "pack", "--json"])
    assert result.exit_code == 0, result.output
    assert captured == [("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY")]


def test_doctor_openai_uses_openai_credentials(monkeypatch):
    from test_system_commands import _checks

    monkeypatch.setattr(system, "_system_checks", lambda **_: _checks())
    monkeypatch.setattr(
        system,
        "load_config",
        lambda: MojiLexConfig.model_validate(
            {
                "repository": {"publish": "local"},
                "ai": {"provider": "openai", "model": MODEL},
            }
        ),
    )
    monkeypatch.setattr(
        system,
        "load_credentials",
        lambda: Credentials(
            telegram_bot_token="synthetic-telegram",
            openai_api_key="synthetic-openai",
        ),
    )
    result = system.doctor_command()
    assert result.result["authoring_ready"] is True
    assert not any("unsupported" in str(warning) for warning in result.warnings)


def test_openai_generation_metadata_records_its_actual_transport_profile():
    config = MojiLexConfig.model_validate({"ai": {"provider": "openai", "model": MODEL}})
    metadata = _default_generation_metadata(config)
    assert metadata.provider == "openai" and metadata.model == MODEL
    assert metadata.request_parameters_sha256 == request_parameters_sha256("openai")
    assert metadata.request_parameters_sha256 != gemini_request_parameters_sha256()


def test_describe_dataset_ids_and_unsaved_pack_use_current_provider(monkeypatch):
    from mojilex_cli.commands.runtime import CommandError

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: MojiLexConfig.model_validate(
            {
                "ai": {"provider": "openai", "model": MODEL},
            }
        ),
    )

    def absent(*_a, **_kw):
        raise CommandError("CONFIG_MISSING", "No saved pack", hint="Synthetic fixture")

    monkeypatch.setattr(packs, "resolve_pack_run", absent)
    assert cli._authoring_secret_names(selectors=("mxe_synthetic",)) == (
        "TELEGRAM_BOT_TOKEN",
        "OPENAI_API_KEY",
    )
    assert cli._authoring_secret_names(selectors=("unsaved-pack",)) == (
        "TELEGRAM_BOT_TOKEN",
        "OPENAI_API_KEY",
    )


def test_openai_benchmark_rejects_gemini_profile_before_network():
    from mojilex_cli.benchmark import validate_model_benchmark_runtime
    from mojilex_cli.benchmark.common import BenchmarkError
    from test_benchmark_model import _case, _manifest

    manifest = _manifest(
        (_case(0, "static-full-color"),), provider="openai", model=MODEL, revision=None
    ).model_copy(
        update={
            "request_parameters_sha256": request_parameters_sha256("openai"),
        }
    )
    validate_model_benchmark_runtime(manifest, "openai")
    wrong = manifest.model_copy(
        update={
            "request_parameters_sha256": gemini_request_parameters_sha256(),
        }
    )
    with pytest.raises(BenchmarkError, match="request parameters"):
        validate_model_benchmark_runtime(wrong, "openai")


def test_openai_benchmark_uses_only_openai_key(tmp_path, monkeypatch):
    from test_benchmark_model import _case, _manifest

    manifest = _manifest(
        (_case(0, "static-full-color"),), provider="openai", model=MODEL, revision=None
    ).model_copy(
        update={
            "request_parameters_sha256": request_parameters_sha256("openai"),
        }
    )
    path = tmp_path / "benchmark.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    seen = []

    class FakeProvider:
        async def validate_credentials(self):
            seen.append("validated")

    class Registry:
        def names(self):
            return ("gemini", "openai")

        def create(self, name, **kwargs):
            seen.append((name, kwargs["model"], kwargs["api_key"]))
            return FakeProvider()

    async def benchmark(*_a, **kwargs):
        assert kwargs["runtime_secrets"] == ("synthetic-openai",)
        return {"passed": True}

    monkeypatch.setattr(
        benchmark_commands,
        "load_credentials",
        lambda: Credentials(
            gemini_api_key="synthetic-gemini",
            openai_api_key="synthetic-openai",
        ),
    )
    monkeypatch.setattr(benchmark_commands, "default_registry", Registry)
    monkeypatch.setattr(benchmark_commands, "run_model_benchmark", benchmark)
    monkeypatch.setattr(
        benchmark_commands, "_qualification_result", lambda report: CommandResult(result=report)
    )
    result = benchmark_commands.benchmark_model_command(
        provider_name="openai",
        model_id=MODEL,
        benchmark_manifest=path,
    )
    assert result.result["passed"] is True
    assert seen == [("openai", MODEL, "synthetic-openai"), "validated"]

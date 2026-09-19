from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mojilex_cli.ai import (
    AIOutputError,
    DescriptionBatch,
    GeminiVisionProvider,
    RequestBudget,
)
from mojilex_cli.ai.base import LocalizedDescription, SemanticTextItem
from mojilex_cli.ai.prompts import gemini_request_parameters_sha256
from mojilex_cli.cache import CacheError, CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.domain.hashes import jcs_sha256
from mojilex_cli.domain.models import TextContentItem
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from test_ai_gemini import _request
from test_ai_interactions import _sdk_client_factory
from test_ai_recovery_checkpoint import _prepare_exact_request
from test_ai_semantic_facets import _payload
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _item, _processed


def _invalid_payload(case: str) -> dict:
    payload = _payload()
    item = payload["items"][0]
    if case == "literal":
        item["facets"]["text_content"]["items"][0]["value"] = "<tag>"
    else:
        item["descriptions"]["en"]["text"] = {
            "nfc": "Cafe\u0301 symbol.",
            "c1": "A\u0085symbol.",
            "spaces": "A  symbol.",
        }[case]
    return payload


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "synthetic-interaction",
            "object": "interaction",
            "status": "completed",
            "model": "gemini-test",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": json.dumps(payload)}]}
            ],
        },
    )


def test_runtime_validation_does_not_change_request_contract_or_cache_hash() -> None:
    # Golden values captured before adding these runtime-only validators.
    assert jcs_sha256(DescriptionBatch.model_json_schema()) == (
        "ebae13b24fd507e850f674b273f29de93bf2c897de9594dbcc4770b08e2fb40b"
    )
    assert gemini_request_parameters_sha256() == (
        "afb1f4ff15eeaa4584ff7a7824b57ba85828a56e8a214a96b98728140ac4d98a"
    )


@pytest.mark.parametrize("literal", ["<tag>", "</script>", '<img src=x onerror="x">', "A\u0085B"])
def test_literal_markup_and_controls_rejected_without_rewriting(literal: str) -> None:
    payload = _payload()["items"][0]["facets"]["text_content"]["items"][0]
    payload["value"] = literal
    with pytest.raises(ValidationError):
        SemanticTextItem.model_validate(payload)
    with pytest.raises(ValidationError):
        TextContentItem.model_validate(payload)
    assert list(_literal_validator().iter_errors(literal))
    assert payload["value"] == literal


def _literal_validator() -> Draft202012Validator:
    schema = json.loads(files("mojilex_cli.schemas").joinpath("v1/facets.schema.json").read_text())
    return Draft202012Validator(schema["$defs"]["textItem"]["properties"]["value"])


@pytest.mark.parametrize("literal", ["404", "/", "()", "&", "é", "A  B", "</>", "<3", "x>y", "<="])
def test_allowed_literal_preserved_exactly(literal: str) -> None:
    payload = _payload()["items"][0]["facets"]["text_content"]["items"][0]
    payload["value"] = literal
    assert SemanticTextItem.model_validate(payload).value == literal
    assert TextContentItem.model_validate(payload).value == literal
    _literal_validator().validate(literal)


@pytest.mark.parametrize("field", ["text", "motion", "usage"])
@pytest.mark.parametrize("value", ["Cafe\u0301", "A\u0085B", "A  B", "A\u00a0B"])
def test_description_fields_match_domain_text_constraints(field: str, value: str) -> None:
    payload = {"text": "A symbol.", "motion_status": "described", "motion": "Moves.", "usage": []}
    payload[field] = [value] if field == "usage" else value
    with pytest.raises(ValidationError):
        LocalizedDescription.model_validate(payload)


@pytest.mark.asyncio
async def test_real_sdk_adapter_preserves_visible_code_symbol(monkeypatch) -> None:
    payload = _payload()
    payload["items"][0]["facets"]["text_content"]["items"][0]["value"] = "</>"
    _sdk_client_factory(monkeypatch, httpx.MockTransport(lambda request: _response(payload)))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    try:
        result = await provider.describe(_request())
        assert result.batch.items[0].facets.text_content.items[0].value == "</>"
    finally:
        await provider._client.aio.aclose()
        provider._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["literal", "nfc", "c1", "spaces"])
async def test_real_sdk_adapter_rejects_invalid_public_text_before_return(
    monkeypatch, case
) -> None:
    calls = []

    def handle(request):
        calls.append(request)
        return _response(_invalid_payload(case))

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    try:
        with pytest.raises(AIOutputError) as captured:
            await provider.describe(_request())
    finally:
        await provider._client.aio.aclose()
        provider._client.close()
    assert len(calls) == 1
    assert "schema_validation" in str(captured.value)
    assert "value_error" in str(captured.value)
    assert "</>" not in str(captured.value)
    assert "Cafe" not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["literal", "nfc", "c1", "spaces"])
async def test_old_invalid_cache_is_rejected_and_regenerated_by_real_adapter(
    tmp_path: Path, monkeypatch, case
) -> None:
    calls = []

    def handle(request):
        calls.append(request)
        # A fresh valid provider answer repairs invalid historic data. The
        # program must never normalize or replace the old answer itself.
        return _response(_payload())

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")

    async def get_provider(*args, **kwargs):
        return provider

    monkeypatch.setattr(runner, "_provider_for_model", get_provider)
    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    snapshot = write_fixture(tmp_path / "dataset")
    source = _item("text-boundary", unique_id="unique", file_id="file")
    processed = _processed(snapshot)
    config = MojiLexConfig(ai=AIConfig(model="gemini-test", model_routing="off"))
    budget = RequestBudget(max_requests=2, allow_unknown_cost=True)
    try:
        with (
            CacheStore(tmp_path / "cache.sqlite") as cache,
            TemporaryMediaRun(root=tmp_path) as temporary,
        ):

            async def describe():
                return await runner._load_or_describe_single(
                    source,
                    processed,
                    model=config.ai.model,
                    config=config,
                    cache=cache,
                    budget=budget,
                    ai_state=runner._AIState(),
                    api_key=None,
                    context=runner._vision_context(source, processed),
                    temporary=temporary,
                    taxonomy_version="1.0.0",
                )

            await describe()
            row = cache._connection.execute(
                "SELECT cache_key, payload_json FROM ai_cache"
            ).fetchone()
            key = row["cache_key"]
            old = json.loads(row["payload_json"])
            old["result"]["batch"] = _invalid_payload(case)
            invalid_raw = json.dumps(old)
            cache._connection.execute("UPDATE ai_cache SET payload_json = ?", (invalid_raw,))
            cache._connection.commit()
            with pytest.raises(CacheError):
                cache.get_ai_entry(key)
            assert (
                cache._connection.execute("SELECT payload_json FROM ai_cache").fetchone()[0]
                == invalid_raw
            )
            repaired = await describe()
            assert repaired.result.batch.items[0].facets.text_content.items[0].value == "404"
            assert cache.get_ai_entry(key) is not None
            assert cache._connection.execute("SELECT COUNT(*) FROM ai_cache").fetchone()[0] == 1
            # A subsequent exact request reuses the repaired row, no third call.
            await describe()
            assert len(calls) == budget.requests_used == 2
    finally:
        await provider._client.aio.aclose()
        provider._client.close()

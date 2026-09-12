from __future__ import annotations

import base64
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from google import genai

from mojilex_cli.ai import AIError, AIOutputError, DescriptionBatch, GeminiVisionProvider
from mojilex_cli.ai.prompts import gemini_request_parameters, gemini_request_parameters_sha256
from mojilex_cli.ai.transport_schema import SUPPORTED_SCHEMA_KEYS, gemini_transport_schema
from mojilex_cli.domain.hashes import jcs_sha256
from test_ai_gemini import _payload, _request


def test_transport_schema_is_closed_nonmutating_and_local_contract_remains_bound() -> None:
    original = DescriptionBatch.model_json_schema()
    before = deepcopy(original)
    projected = gemini_transport_schema(original)
    assert original == before
    assert projected != original
    assert projected == gemini_transport_schema(original)
    todo = [projected]
    descriptions: list[str] = []
    while todo:
        node = todo.pop()
        assert set(node) <= SUPPORTED_SCHEMA_KEYS
        assert "enum" not in node
        assert "title" not in node
        if isinstance(node.get("description"), str):
            descriptions.append(node["description"])
        for key, value in node.items():
            if key in {"properties", "$defs"}:
                todo.extend(value.values())
            elif key in {"items", "additionalProperties"} and isinstance(value, dict):
                todo.append(value)
            elif key in {"anyOf", "prefixItems"}:
                todo.extend(value)
    assert any("Allowed values:" in description for description in descriptions)
    parameters = gemini_request_parameters()
    assert parameters["local_response_schema_sha256"] == jcs_sha256(original)
    assert parameters["response_format"]["schema"] == projected
    legacy = {
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": original,
        "max_output_tokens": 8192,
    }
    assert gemini_request_parameters_sha256() != jcs_sha256(legacy)
    assert parameters["api_surface"] == "interactions"
    assert parameters["api_version"] == "v1beta"


def test_schema_unknown_keywords_fail_closed_without_erasing_property_names() -> None:
    for unknown in ("patternProperties", "allOf", "not", "dependentSchemas"):
        with pytest.raises(ValueError, match="unsupported keyword"):
            gemini_transport_schema({"type": "object", unknown: {}})
    schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string", "pattern": "^ok$"}},
        "required": ["pattern"],
    }
    assert gemini_transport_schema(schema)["properties"] == {"pattern": {"type": "string"}}


def test_schema_compacts_enums_and_documented_nullable_types_without_weakening_local_model() -> (
    None
):
    schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": ["first", "second"]},
            "optional": {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ],
                "default": None,
            },
        },
        "required": ["choice"],
    }
    projected = gemini_transport_schema(schema)
    assert projected["properties"]["choice"] == {
        "type": "string",
        "description": 'Allowed values: "first", "second".',
    }
    assert projected["properties"]["optional"] == {"type": ["string", "null"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid-json", "local-rule", "incomplete", "wrong-model"])
async def test_interaction_output_is_strictly_validated_without_raw_error_leak(failure) -> None:
    payload = _payload()
    response = SimpleNamespace(
        status="completed", model="gemini-test", output_text=json.dumps(payload)
    )
    if failure == "invalid-json":
        response.output_text = "private synthetic request text, not JSON"
    elif failure == "local-rule":
        payload["items"][0]["label"] = "not-a-valid-E-number"
        response.output_text = json.dumps(payload)
    elif failure == "incomplete":
        response.status = "in_progress"
    else:
        response.model = "different-model"

    class Interactions:
        async def create(self, **kwargs):
            return response

    provider = GeminiVisionProvider(
        model="gemini-test",
        client=SimpleNamespace(aio=SimpleNamespace(interactions=Interactions())),
    )
    with pytest.raises(AIOutputError) as captured:
        await provider.describe(_request())
    assert str(captured.value) == "Gemini returned invalid or incomplete structured JSON"
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def _sdk_client_factory(monkeypatch, transport):
    original = genai.Client

    def make_client(**kwargs):
        options = kwargs["http_options"]
        assert options.retry_options.attempts == 0
        options.httpx_async_client = httpx.AsyncClient(transport=transport)
        return original(**kwargs, vertexai=False)

    monkeypatch.setattr(genai, "Client", make_client)


@pytest.mark.asyncio
async def test_real_sdk_serializes_stateless_interactions_and_decodes_response_offline(
    monkeypatch,
) -> None:
    captured = []

    def handle(request):
        captured.append(request)
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
                "usage": {
                    "total_input_tokens": 7,
                    "total_output_tokens": 20,
                    "total_thought_tokens": 22,
                    "total_tokens": 49,
                },
            },
        )

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    try:
        result = await provider.describe(_request())
    finally:
        await provider._client.aio.aclose()
        provider._client.close()
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/v1beta/interactions"
    body = json.loads(request.content)
    assert body["store"] is False
    assert body["background"] is False
    assert body["stream"] is False
    assert body["generation_config"] == {"max_output_tokens": 8192, "thinking_level": "low"}
    assert body["response_format"] == gemini_request_parameters()["response_format"]
    assert set(body) == {
        "model",
        "input",
        "store",
        "background",
        "stream",
        "response_format",
        "generation_config",
    }
    assert body["input"][0]["type"] == "user_input"
    image = body["input"][0]["content"][1]
    assert image["type"] == "image"
    assert image["mime_type"] == "image/png"
    assert base64.b64decode(image["data"], validate=True) == _request().images[0].data
    assert "uri" not in image
    assert result.batch.items[0].label == "E001"
    assert result.model_revision is None
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 42


@pytest.mark.asyncio
async def test_real_sdk_429_makes_one_attempt_and_preserves_only_safe_status(monkeypatch) -> None:
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "RESOURCE_EXHAUSTED",
                    "message": "synthetic-secret-that-must-not-leak",
                }
            },
        )

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    try:
        with pytest.raises(AIError) as captured:
            await provider.describe(_request())
    finally:
        await provider._client.aio.aclose()
        provider._client.close()
    assert len(requests) == 1
    assert "http=429" in str(captured.value)
    assert "synthetic-secret" not in str(captured.value)
    assert "synthetic-key" not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_code", "expected"),
    [
        ("invalid_request", "provider_code=invalid_request"),
        ("failed_precondition", "provider_code=failed_precondition"),
        ("private-synthetic-code", None),
    ],
)
async def test_real_sdk_preserves_only_allowlisted_interactions_error_code(
    monkeypatch, provider_code, expected
) -> None:
    def handle(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": provider_code,
                    "message": "synthetic-secret-that-must-not-leak",
                }
            },
        )

    _sdk_client_factory(monkeypatch, httpx.MockTransport(handle))
    provider = GeminiVisionProvider(model="gemini-test", api_key="synthetic-key-not-real")
    try:
        with pytest.raises(AIError) as captured:
            await provider.describe(_request())
    finally:
        await provider._client.aio.aclose()
        provider._client.close()
    rendered = str(captured.value)
    assert "http=400" in rendered
    assert (expected in rendered) if expected is not None else "provider_code=" not in rendered
    assert "synthetic-secret" not in rendered
    assert "synthetic-key" not in rendered
    assert "private-synthetic-code" not in rendered

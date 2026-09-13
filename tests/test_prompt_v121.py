from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mojilex_cli.ai import DescriptionBatch, GeminiVisionProvider
from mojilex_cli.ai.prompts import (
    build_context_prompt,
    build_prompt,
    current_prompt_version,
    gemini_request_parameters,
    gemini_request_parameters_sha256,
    prompt_manifest,
    prompt_manifest_sha256,
    prompt_sha256,
    prompt_template_bytes,
    prompt_templates,
    use_prompt_version,
    v1_1_0,
    v1_2_0,
    v1_2_1,
)
from test_ai_gemini import _FakeModels, _payload, _request


def test_v120_prompt_and_transport_digests_remain_frozen() -> None:
    assert v1_2_0.prompt_sha256() == (
        "9d502824d09994bb26ecd04e752f9d0f7320a8022e34d539bb0db3627a4a26ae"
    )
    assert v1_2_0.prompt_manifest_sha256() == (
        "fc34bfce4bf5daf01b5203f1373948e419a48eec401dee5751c6f01de7f18516"
    )
    assert v1_2_0.gemini_request_parameters_sha256() == (
        "afb1f4ff15eeaa4584ff7a7824b57ba85828a56e8a214a96b98728140ac4d98a"
    )
    assert v1_2_0.gemini_request_parameters()["local_response_schema_sha256"] == (
        "ebae13b24fd507e850f674b273f29de93bf2c897de9594dbcc4770b08e2fb40b"
    )
    assert v1_2_1.prompt_sha256() != v1_2_0.prompt_sha256()
    assert v1_2_1.gemini_request_parameters() == v1_2_0.gemini_request_parameters()


@pytest.mark.parametrize(
    "version,module", [("1.1.0", v1_1_0), ("1.2.0", v1_2_0), ("1.2.1", v1_2_1)]
)
def test_every_dispatched_prompt_component_matches_its_immutable_version(version, module) -> None:
    with use_prompt_version(version):
        assert current_prompt_version() == version
        assert prompt_manifest() == module.prompt_manifest()
        assert prompt_manifest_sha256() == module.prompt_manifest_sha256()
        assert prompt_templates() == module.prompt_templates()
        assert prompt_template_bytes() == module.prompt_template_bytes()
        assert prompt_sha256() == module.prompt_sha256()
        assert build_prompt(("E001",), animated=False) == module.build_prompt(
            ("E001",), animated=False
        )
        assert build_context_prompt("{}") == module.build_context_prompt("{}")
        assert gemini_request_parameters() == module.gemini_request_parameters()
        assert gemini_request_parameters_sha256() == module.gemini_request_parameters_sha256()
    assert current_prompt_version() == "1.2.1"


def test_prompt_lists_the_exact_controlled_uses_without_weakening_transport_or_local_schema() -> (
    None
):
    prompt = build_prompt(("E001",), animated=False)
    marker = "suggested_uses choices (exact JSON strings):\n"
    choices, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
    local = DescriptionBatch.model_json_schema()["$defs"]["SemanticFacets"]["properties"][
        "suggested_uses"
    ]
    assert choices == local["items"]["enum"]
    assert len(choices) == 15
    assert local["maxItems"] == 8
    assert (
        "Free-form human usage phrases belong in descriptions.ru.usage or descriptions.en.usage"
        in prompt
    )
    assert "If no listed application\nfits the observed image, return suggested_uses: []" in prompt
    assert "Do not translate them" in prompt
    assert marker not in v1_2_0.SYSTEM_PROMPT
    transport = gemini_request_parameters()["response_format"]["schema"]["$defs"]["SemanticFacets"][
        "properties"
    ]["suggested_uses"]
    assert "enum" not in transport["items"]
    assert "maxItems" not in transport
    payload = _payload()
    payload["items"][0]["facets"]["suggested_uses"] = []
    assert DescriptionBatch.model_validate(payload).items[0].facets.suggested_uses == ()
    payload["items"][0]["facets"]["suggested_uses"] = ["invented-application"]
    with pytest.raises(ValidationError) as captured:
        DescriptionBatch.model_validate(payload)
    assert any(
        error["loc"] == ("items", 0, "facets", "suggested_uses", 0)
        and error["type"] == "literal_error"
        for error in captured.value.errors(include_input=False)
    )


@pytest.mark.asyncio
async def test_actual_gemini_request_includes_v121_vocabulary_in_hash_bound_prompt() -> None:
    resource = _FakeModels()
    provider = GeminiVisionProvider(
        model="gemini-test", client=SimpleNamespace(aio=SimpleNamespace(interactions=resource))
    )
    request = _request()
    await provider.describe(request)
    sent_text = resource.calls[0]["input"][0]["content"][0]["text"]
    context_json = json.dumps(
        {
            label: request.context[label].model_dump(mode="json")
            for label in request.expected_labels
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = prompt_templates()["user"]["value"].format(
        expected_labels="E001", input_kind="static", context_json=context_json
    )
    assert sent_text == expected
    assert '"message-accent", "decoration", "branding"' in sent_text
    assert "agreement and\ngreeting are not suggested_uses choices" in sent_text

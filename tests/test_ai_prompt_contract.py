from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from pydantic import ValidationError

from mojilex_cli.ai import DescriptionBatch
from mojilex_cli.ai.gemini import _safe_validation_summary
from mojilex_cli.ai.prompts import (
    PROMPT_VERSION,
    build_prompt,
    current_prompt_version,
    gemini_request_parameters_sha256,
    prompt_manifest,
    prompt_sha256,
    use_prompt_version,
    v1_1_0,
    v1_2_0,
    v1_2_1,
)
from test_ai_semantic_facets import _payload


def test_versioned_prompt_dispatch_preserves_exact_old_prompt_and_transport() -> None:
    assert PROMPT_VERSION == current_prompt_version() == "1.2.1"
    current_hash = prompt_sha256()
    assert current_hash == v1_2_1.prompt_sha256()
    with use_prompt_version("1.1.0"):
        assert current_prompt_version() == "1.1.0"
        assert prompt_manifest() == v1_1_0.prompt_manifest()
        assert prompt_sha256() == v1_1_0.prompt_sha256() != current_hash
        assert build_prompt(("E001",), animated=False) == v1_1_0.build_prompt(
            ("E001",), animated=False
        )
        assert gemini_request_parameters_sha256() == v1_1_0.gemini_request_parameters_sha256()
        with pytest.raises(RuntimeError), use_prompt_version("1.2.0"):
            assert current_prompt_version() == "1.2.0"
            raise RuntimeError("synthetic failure")
        assert current_prompt_version() == "1.1.0"
    assert current_prompt_version() == "1.2.1"
    assert gemini_request_parameters_sha256() == v1_1_0.gemini_request_parameters_sha256()
    with pytest.raises(ValueError, match="unsupported prompt version"):
        with use_prompt_version("unsupported-synthetic-version"):
            pass
    assert current_prompt_version() == "1.2.1"


@pytest.mark.asyncio
async def test_prompt_selection_is_isolated_between_concurrent_tasks() -> None:
    async def digest(version: str):
        with use_prompt_version(version):
            await asyncio.sleep(0)
            return current_prompt_version(), prompt_sha256()

    assert await asyncio.gather(digest("1.1.0"), digest("1.2.0"), digest("1.2.1")) == [
        ("1.1.0", v1_1_0.prompt_sha256()),
        ("1.2.0", v1_2_0.prompt_sha256()),
        ("1.2.1", v1_2_1.prompt_sha256()),
    ]
    assert current_prompt_version() == "1.2.1"


@pytest.mark.parametrize(
    "rule",
    [
        "text_content_type_required",
        "number_content_type_required",
        "text_uncertainty_required",
        "style_uncertainty_required",
        "text_items_must_be_empty",
        "recognized_text_items_required",
        "motion_presence_mismatch",
        "motion_uncertainty_required",
        "semantic_tags_repeat_facets",
    ],
)
def test_cross_field_rules_remain_strict_and_report_safe_specific_codes(rule: str) -> None:
    valid = _payload()
    invalid = deepcopy(valid)
    item = invalid["items"][0]
    facets = item["facets"]
    if rule == "text_content_type_required":
        facets["content_types"].remove("text")
    elif rule == "number_content_type_required":
        facets["content_types"].remove("number")
    elif rule == "text_uncertainty_required":
        facets["text_content"]["status"] = "partially-recognized"
    elif rule == "style_uncertainty_required":
        facets["styles"] = ["minimal", "detailed"]
    elif rule == "text_items_must_be_empty":
        facets["text_content"]["status"] = "none"
    elif rule == "recognized_text_items_required":
        facets["text_content"]["items"] = []
    elif rule == "motion_presence_mismatch":
        item["descriptions"]["ru"]["motion_status"] = "described"
    elif rule == "motion_uncertainty_required":
        item["descriptions"]["ru"]["motion_status"] = "undetermined"
    else:
        item["semantic_tags"] = ["text"]
    item["descriptions"]["ru"]["text"] = "synthetic-private-description"
    with pytest.raises(ValidationError) as captured:
        DescriptionBatch.model_validate(invalid)
    message = _safe_validation_summary(captured.value)
    assert f"({rule})" in message
    assert "synthetic-private-description" not in message
    assert "(value_error)" not in message
    assert "(too_short)" not in message
    assert DescriptionBatch.model_validate(valid).items[0].label == "E001"


def test_empty_array_diagnostic_is_distinct_from_invalid_nested_item() -> None:
    with pytest.raises(ValidationError) as captured:
        DescriptionBatch.model_validate({"items": []})
    assert _safe_validation_summary(captured.value).endswith("$.items (too_short)")


def test_new_prompt_includes_missing_cross_field_rules_and_honest_uncertainty() -> None:
    text = build_prompt(("E001", "E002"), animated=False)
    assert "Expected labels: E001, E002" in text
    for required in (
        "Every status other than none requires content_types to include text",
        "content_types MUST include BOTH text and number",
        "both minimal and detailed, or both outline and solid",
        "uncertainties MUST include style",
        "recognized or partially-recognized requires at least one actual readable text item",
        "motion MUST be null or omitted",
        "Never repeat ANY value already present in content_types",
        "Never invent a candidate to satisfy a count",
        "retain honest uncertainty instead of fabricating details",
    ):
        assert required in text
    assert "Every status other than none" not in v1_1_0.SYSTEM_PROMPT

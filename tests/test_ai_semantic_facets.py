from __future__ import annotations

import pytest
from pydantic import ValidationError

from mojilex_cli.ai import DescriptionBatch
from mojilex_cli.ai.prompts import (
    gemini_request_parameters,
    gemini_request_parameters_sha256,
    prompt_sha256,
)


def _payload() -> dict[str, object]:
    description = {
        "text": "A visible 404 symbol.",
        "motion_status": "not_applicable",
        "usage": [],
    }
    return {
        "items": [
            {
                "label": "E001",
                "descriptions": {"ru": description, "en": description},
                "facets": {
                    "text_content": {
                        "status": "recognized",
                        "dynamics": "stable",
                        "items": [
                            {
                                "value": "404",
                                "kind": "number",
                                "script": "Zyyy",
                                "language": "und",
                                "temporal_scope": "persistent",
                                "media_refs": [{"role": "primary"}],
                            }
                        ],
                    },
                    "content_types": ["number", "symbol", "text"],
                    "styles": ["flat", "outline"],
                    "suggested_uses": ["status"],
                    "uncertainties": [],
                },
                "semantic_tags": ["error-code"],
                "content": {"rating": "general", "warnings": []},
            }
        ]
    }


def test_ai_schema_contains_only_semantic_facets() -> None:
    batch = DescriptionBatch.model_validate(_payload())
    assert batch.items[0].facets.text_content.items[0].value == "404"
    payload = _payload()
    facets = payload["items"][0]["facets"]  # type: ignore[index]
    facets["rendering"] = {"profile": "color-v1", "items": []}  # type: ignore[index]
    with pytest.raises(ValidationError, match="Extra inputs"):
        DescriptionBatch.model_validate(payload)


def test_prompt_and_request_parameter_hashes_are_exact_and_stable() -> None:
    assert len(prompt_sha256()) == 64
    assert gemini_request_parameters_sha256() == gemini_request_parameters_sha256()
    parameters = gemini_request_parameters()
    assert parameters["api_surface"] == "interactions"
    assert parameters["store"] is False
    assert parameters["response_format"]["mime_type"] == "application/json"
    assert "properties" in parameters["response_format"]["schema"]

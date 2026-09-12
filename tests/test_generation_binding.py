from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import rfc8785
from pydantic import ValidationError

from mojilex_cli.ai import DescriptionItem, GeminiVisionProvider
from mojilex_cli.ai.prompts import (
    PROMPT_VERSION,
    gemini_request_parameters,
    prompt_manifest,
    prompt_manifest_sha256,
    prompt_sha256,
    prompt_templates,
)
from mojilex_cli.ai.runtime_parameters import MAX_OUTPUT_TOKENS
from mojilex_cli.domain import Provenance
from mojilex_cli.policy.model_routing import build_model_routing_binding
from mojilex_cli.policy.qualification import (
    ModelQualificationRegistry,
    QualificationStatus,
    match_qualification,
)
from mojilex_cli.policy.routing import PolicyError
from test_dataset_helpers import make_snapshot, write_fixture
from test_policy_qualification import _query

CONCEPT_BINDING = {
    "concept_registry_id": "concepts-v1.fixture-1",
    "concept_registry_sha256": "1" * 64,
    "concept_candidate_set_sha256": "2" * 64,
    "concept_candidate_profile_id": "concept-candidates-v1",
    "concept_candidate_profile_sha256": "3" * 64,
    "model_routing_policy_id": "model-routing-local-v1",
    "model_routing_policy_sha256": "4" * 64,
}


def _configuration(**updates):
    value = {
        "provider": "gemini",
        "model": "primary-model",
        "model_revision": None,
        "description_profile": "standard-v1",
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "request_parameters_sha256": "6" * 64,
        "languages": ["en", "ru"],
        **{
            key: value
            for key, value in CONCEPT_BINDING.items()
            if not key.startswith("model_routing")
        },
    }
    value.update(updates)
    return value


def test_routing_policy_binds_actual_config_without_its_own_hash() -> None:
    first = build_model_routing_binding(_configuration())
    assert first == build_model_routing_binding(_configuration())
    assert first.body["trigger_order"] == []
    assert (
        first.body["configuration_selectors"][0]["configuration_selector_sha256"]
        == hashlib.sha256(rfc8785.dumps(_configuration())).hexdigest()
    )
    assert first.sha256 != build_model_routing_binding(_configuration(model="another")).sha256
    routed = build_model_routing_binding(
        _configuration(), _configuration(model="stronger"), mode="rules"
    )
    assert routed.sha256 != first.sha256
    assert len(routed.body["configuration_selectors"]) == 2
    assert routed.body["max_escalations_per_item"] == 1
    assert routed.body["merge_policy"] == "whole-result-only-v1"
    assert routed.provenance_fields["model_routing_policy_sha256"] == routed.sha256


@pytest.mark.parametrize(
    "updates",
    [
        {"api_key": "not-a-real-credential"},
        {"provider_url": "https://example.invalid"},
        {"languages": ["ru", "en"]},
        {"model_routing_policy_id": "self-reference"},
        {"concept_candidate_set_sha256": None},
    ],
)
def test_routing_selector_rejects_unknown_partial_and_self_referential_fields(updates) -> None:
    with pytest.raises(PolicyError):
        build_model_routing_binding(_configuration(**updates))


@pytest.mark.parametrize("mode,escalation", [("rules", None), ("off", "stronger")])
def test_routing_mode_requires_explicit_consistent_escalation(mode, escalation) -> None:
    with pytest.raises(PolicyError):
        build_model_routing_binding(
            _configuration(),
            _configuration(model=escalation) if escalation else None,
            mode=mode,
        )


@pytest.mark.parametrize("field", list(CONCEPT_BINDING))
def test_every_concept_and_routing_field_participates_in_qualification(field, tmp_path) -> None:
    write_fixture(tmp_path)
    registry = ModelQualificationRegistry.load(tmp_path)
    entry = registry.entries[0].model_copy(update=CONCEPT_BINDING)
    bound_registry = registry.model_copy(update={"entries": (entry,)})
    query = _query(**CONCEPT_BINDING)
    assert match_qualification(bound_registry, query).qualified
    changed = "9" * 64 if field.endswith("sha256") else "different-id"
    mismatch = match_qualification(
        bound_registry,
        query.model_copy(update={field: changed}),
        qualification_id=entry.qualification_id,
    )
    assert mismatch.status is QualificationStatus.MISMATCH
    assert not match_qualification(bound_registry, _query()).qualified


def test_provenance_concept_group_is_atomic_and_legacy_is_preserved(tmp_path) -> None:
    original = next(iter(make_snapshot(tmp_path).emojis.values())).provenance
    raw = original.as_dict()
    assert Provenance.model_validate(raw).concept_registry_id is None
    bound = Provenance.model_validate({**raw, **CONCEPT_BINDING})
    assert bound.concept_registry_id == "concepts-v1.fixture-1"
    with pytest.raises(ValidationError, match="all seven"):
        Provenance.model_validate({**raw, "concept_registry_id": "concepts-v1"})


def test_concept_output_is_sorted_bounded_and_cannot_invent_free_text() -> None:
    from test_ai_semantic_facets import _payload

    raw = _payload()["items"][0]
    assert DescriptionItem.model_validate({**raw, "concept_ids": ["technical.error"]}).concept_ids
    for values in (["bad free text"], ["animal.cat", "animal.cat"], ["z.test", "a.test"]):
        with pytest.raises(ValidationError):
            DescriptionItem.model_validate({**raw, "concept_ids": values})


def test_prompt_hash_is_role_bound_jcs_and_separate_from_manifest() -> None:
    assert PROMPT_VERSION == "1.1.0"
    assert prompt_sha256() == hashlib.sha256(rfc8785.dumps(prompt_templates())).hexdigest()
    assert prompt_manifest_sha256() == hashlib.sha256(rfc8785.dumps(prompt_manifest())).hexdigest()
    assert prompt_manifest_sha256() != prompt_sha256()
    changed = prompt_templates()
    changed["user"]["value"] += " "
    assert hashlib.sha256(rfc8785.dumps(changed)).hexdigest() != prompt_sha256()
    assert (
        gemini_request_parameters()["generation_config"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_concepts", [False, True])
async def test_runtime_user_prompt_exactly_renders_the_hash_bound_template(with_concepts) -> None:
    from mojilex_cli.ai.concepts import concept_context_from_documents
    from test_ai_concepts import PROFILE, _registry
    from test_ai_gemini import _FakeModels, _request

    request = _request()
    context_payload = {
        label: request.context[label].model_dump(mode="json") for label in request.expected_labels
    }
    if with_concepts:
        concepts = concept_context_from_documents(_registry(), PROFILE).prompt_context
        request = request.model_copy(update={"concept_context": concepts})
        context_payload = {"items": context_payload, "concept_context": concepts}
    templates = prompt_templates()
    rendered = templates["user"]["value"].format(
        expected_labels=", ".join(request.expected_labels),
        input_kind="static",
        context_json=json.dumps(
            context_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    )
    models = _FakeModels()
    provider = GeminiVisionProvider(
        model=request.model, client=SimpleNamespace(aio=SimpleNamespace(interactions=models))
    )
    await provider.describe(request)
    assert models.calls[0]["input"][0]["content"][0] == {"type": "text", "text": rendered}
    assert "system_instruction" not in models.calls[0]
    assert templates["system"] == {"present": False}
    assert templates["developer"] == {"present": False}

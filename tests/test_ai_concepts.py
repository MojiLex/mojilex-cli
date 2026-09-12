from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mojilex_cli.ai.concepts import (
    ConceptContextError,
    concept_context_from_documents,
    load_concept_context,
)
from mojilex_cli.domain import jcs_bytes

PROFILE = {
    "profile_id": "concept-candidates-v1",
    "ordering": "registry-id-bytewise-v1",
    "max_candidates": 65536,
    "include_statuses": ["active"],
}


def _registry() -> dict[str, Any]:
    cat = {
        "id": "animal.cat",
        "status": "active",
        "labels": {"en": "cat", "ru": "кот"},
        "aliases": {"en": ["feline"], "ru": ["кошка"]},
        "definitions": {"en": "A cat.", "ru": "Кот."},
        "parent_ids": [],
        "positive_examples": {"en": ["cartoon cat"], "ru": ["мультяшный кот"]},
        "negative_examples": {"en": ["dog"], "ru": ["собака"]},
    }
    retired = {**copy.deepcopy(cat), "id": "animal.kitty", "status": "deprecated"}
    retired["replaced_by"] = "animal.cat"
    return {
        "registry_schema_version": "1.0.0",
        "registry_type": "concepts",
        "registry_id": "concepts-v1.fixture-1",
        "concepts": [cat, retired],
    }


def test_context_binds_full_registry_and_exact_all_active_candidate_list() -> None:
    registry = _registry()
    context = concept_context_from_documents(registry, PROFILE)
    assert context.candidate_ids == ("animal.cat",)
    assert context.provenance_fields == {
        "concept_registry_id": registry["registry_id"],
        "concept_registry_sha256": hashlib.sha256(jcs_bytes(registry)).hexdigest(),
        "concept_candidate_profile_id": "concept-candidates-v1",
        "concept_candidate_profile_sha256": hashlib.sha256(jcs_bytes(PROFILE)).hexdigest(),
        "concept_candidate_set_sha256": hashlib.sha256(b'["animal.cat"]').hexdigest(),
    }
    assert context.candidate_records == (registry["concepts"][0],)
    payload = context.prompt_context
    payload["candidates"][0]["labels"]["en"] = "changed downstream"
    assert context.candidate_records[0]["labels"]["en"] == "cat"
    changed_registry = copy.deepcopy(registry)
    changed_registry["registry_id"] = "concepts-v1.fixture-2"
    changed_registry["concepts"][0]["definitions"]["en"] = "A domestic cat."
    changed = concept_context_from_documents(changed_registry, PROFILE)
    assert (
        changed.provenance_fields["concept_registry_sha256"]
        != context.provenance_fields["concept_registry_sha256"]
    )
    assert (
        changed.provenance_fields["concept_candidate_set_sha256"]
        == context.provenance_fields["concept_candidate_set_sha256"]
    )


@pytest.mark.parametrize(
    "selected", [[], ["animal.unknown"], ["animal.kitty"], ["animal.cat", "animal.cat"]]
)
def test_selection_never_invents_or_reuses_retired_concepts(selected: list[str]) -> None:
    context = concept_context_from_documents(_registry(), PROFILE)
    with pytest.raises(ConceptContextError):
        context.validate_selection(selected)
    assert context.validate_selection(["animal.cat"]) == ("animal.cat",)
    assert context.validate_selection([], require_complete=False) == ()


@pytest.mark.parametrize("mutation", ["duplicate", "unsorted", "dangling", "cycle"])
def test_invalid_registry_identity_or_graph_is_rejected(mutation: str) -> None:
    registry = _registry()
    if mutation == "duplicate":
        registry["concepts"].append(copy.deepcopy(registry["concepts"][0]))
    elif mutation == "unsorted":
        registry["concepts"].reverse()
    elif mutation == "dangling":
        registry["concepts"][0]["parent_ids"] = ["animal.missing"]
    else:
        registry["concepts"][0]["parent_ids"] = ["animal.kitty"]
        registry["concepts"][1]["parent_ids"] = ["animal.cat"]
    with pytest.raises(ConceptContextError):
        concept_context_from_documents(registry, PROFILE)


@pytest.mark.parametrize("field", ["ordering", "max_candidates", "unknown"])
def test_undeclared_candidate_algorithm_is_rejected(field: str) -> None:
    profile = copy.deepcopy(PROFILE)
    profile[field] = 8 if field == "max_candidates" else "other"
    with pytest.raises(ConceptContextError):
        concept_context_from_documents(_registry(), profile)


def test_loader_uses_jcs_registry_hash_and_requires_exact_profile_bytes(tmp_path: Path) -> None:
    registry_path = tmp_path / "taxonomy" / "v1" / "concepts.json"
    profile_path = tmp_path / "analysis-profiles" / "concept-candidates-v1.json"
    registry_path.parent.mkdir(parents=True)
    profile_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(_registry(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profile_path.write_bytes(jcs_bytes(PROFILE))
    context = load_concept_context(tmp_path)
    assert context.candidate_ids == ("animal.cat",)
    registry_path.write_bytes(jcs_bytes(_registry()))
    assert load_concept_context(tmp_path).provenance_fields == context.provenance_fields
    profile_path.write_bytes(jcs_bytes(PROFILE) + b"\n")
    with pytest.raises(ConceptContextError):
        load_concept_context(tmp_path)

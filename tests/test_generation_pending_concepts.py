from __future__ import annotations

import copy

import pytest

from mojilex_cli.ai import AIOutputError, DescriptionBatch, DescriptionResult
from mojilex_cli.ai.concepts import ConceptContextError, concept_context_from_documents
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.transform import _emoji
from test_ai_concepts import PROFILE, _registry
from test_dataset_helpers import make_snapshot
from test_pipeline_transform import _analysis, _description, _generation, _processed, _source


@pytest.fixture
def generation_concepts():
    registry = _registry()
    dog = copy.deepcopy(registry["concepts"][0])
    dog["id"] = "animal.dog"
    registry["concepts"].insert(1, dog)
    context = concept_context_from_documents(registry, PROFILE)
    token = runner._GENERATION_INPUTS.set(runner._GenerationInputs(context, {}, {}))
    try:
        yield context
    finally:
        runner._GENERATION_INPUTS.reset(token)


def _result(description):
    return DescriptionResult(
        batch=DescriptionBatch(items=(description,)),
        provider="gemini",
        model="test-model",
        model_revision="test-revision",
    )


def test_empty_generation_concepts_remain_pending_in_staging(tmp_path, generation_concepts):
    snapshot = make_snapshot(tmp_path)
    description = _description(snapshot)
    assert description.concept_ids == ()
    runner._validate_actual_result(_result(description), "gemini", "test-model")
    generated = _emoji(
        "telegram",
        _source(snapshot).items[0],
        _processed(snapshot, tmp_path),
        description,
        _analysis(snapshot),
        _generation(),
        manifest=snapshot.manifest,
        epoch=0,
        now="2026-09-12T18:00:00Z",
    )
    assert generated.concept_ids == []
    assert generated.concept_mapping_status.value == "pending"
    # The strict selection contract used for completed mappings remains strict.
    with pytest.raises(ConceptContextError):
        generation_concepts.validate_selection(generated.concept_ids)


@pytest.mark.parametrize(
    "selected",
    [("animal.unknown",), ("animal.cat", "animal.cat"), ("animal.dog", "animal.cat")],
)
def test_generation_still_rejects_unknown_duplicate_and_unsorted_concepts(
    tmp_path, generation_concepts, selected
):
    description = _description(make_snapshot(tmp_path)).model_copy(update={"concept_ids": selected})
    with pytest.raises(AIOutputError, match="exact candidate set"):
        runner._validate_actual_result(_result(description), "gemini", "test-model")


def test_generation_accepts_complete_exact_concepts(tmp_path, generation_concepts):
    description = _description(make_snapshot(tmp_path)).model_copy(
        update={"concept_ids": ("animal.cat", "animal.dog")}
    )
    runner._validate_actual_result(_result(description), "gemini", "test-model")

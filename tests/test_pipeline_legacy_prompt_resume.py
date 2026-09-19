from __future__ import annotations

import json
from pathlib import Path

import pytest
import rfc8785

from mojilex_cli.ai import RequestBudget
from mojilex_cli.ai.prompts import (
    PROMPT_VERSION,
    current_prompt_version,
    prompt_sha256,
    use_prompt_version,
    v1_1_0,
)
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner, transform
from test_ai_concepts import PROFILE, _registry
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import (
    _PAYLOAD,
    _analysis,
    _collection,
    _CountingAdapter,
    _CountingProcessor,
    _item,
    _prepare_single_request,
    _processed,
    _seed_resume,
    _SelectiveProvider,
)


def _install_concept_fixture(root: Path) -> None:
    registry = root / "taxonomy" / "v1" / "concepts.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(_registry()), encoding="utf-8")
    profile = root / "analysis-profiles" / "concept-candidates-v1.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_bytes(rfc8785.dumps(PROFILE))


def test_legacy_prompt_and_schema_digests_are_frozen_for_resume() -> None:
    # These describe the old request contract. Diagnostic error-code changes
    # must not invalidate a paid response or silently relabel its provenance.
    assert v1_1_0.prompt_sha256() == (
        "e6e4398abc5bc9c0568a2bc82a14e391da051bf18e583aa6f72b303154c73de7"
    )
    assert v1_1_0.gemini_request_parameters_sha256() == (
        "afb1f4ff15eeaa4584ff7a7824b57ba85828a56e8a214a96b98728140ac4d98a"
    )
    assert v1_1_0.gemini_request_parameters()["local_response_schema_sha256"] == (
        "ebae13b24fd507e850f674b273f29de93bf2c897de9594dbcc4770b08e2fb40b"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_version", ["1.1.0", "1.2.0"])
@pytest.mark.parametrize("with_concepts", [False, True])
@pytest.mark.parametrize("with_missing_item", [False, True])
async def test_legacy_paid_result_keeps_old_provenance_while_missing_item_uses_new_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_concepts: bool,
    with_missing_item: bool,
    legacy_version: str,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    if with_concepts:
        _install_concept_fixture(snapshot.root)
    original_files = snapshot.to_files()
    config = MojiLexConfig(ai=AIConfig(model="primary-model", model_routing="off"))
    first = _item("legacy-saved", unique_id="saved-unique", file_id="saved-file")
    second = _item("fresh-pending", unique_id="pending-unique", file_id="pending-file")
    source = _collection((first, second) if with_missing_item else (first,))
    value = _processed(snapshot)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    with use_prompt_version(legacy_version):
        old_hash = prompt_sha256()
        old_inputs = runner._load_generation_inputs(snapshot, config) if with_concepts else None
        token = runner._GENERATION_INPUTS.set(old_inputs)
        try:
            # A saved singleton trace is intentionally different from the new
            # two-item grouping: resume must restore each exact successful item.
            elements = _seed_resume(cache, _collection((first,)), value, config)
        finally:
            runner._GENERATION_INPUTS.reset(token)
    new_inputs = runner._load_generation_inputs(snapshot, config) if with_concepts else None
    new_hash = prompt_sha256()
    token = runner._GENERATION_INPUTS.set(new_inputs)
    provider = _SelectiveProvider()
    paid_versions = []

    async def fixed_provider(*args, **kwargs):
        paid_versions.append(current_prompt_version())
        return provider

    async def forbid_provider(*args, **kwargs):
        pytest.fail("media/cache preparation attempted to contact the AI provider")

    monkeypatch.setattr(runner, "_provider_for_model", forbid_provider)
    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_single_request)
    verified = {}
    adapter = _CountingAdapter({"saved-file": _PAYLOAD, "pending-file": _PAYLOAD})
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            prepared_source, prepared = await runner._prepare_collection_media(
                snapshot,
                adapter,
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope="mlxrun_" + "d" * 32,
                verified_semantic_outcomes=verified,
            )
            assert set(verified) == {first.native_id}
            # Paid semantics are restored, but absent puzzle tiles still need
            # their pixels rebuilt without repeating deterministic analysis or AI.
            assert prepared[first.native_id].frame_paths
            assert processor.decode_calls == 1 + int(with_missing_item)
            assert processor.analysis_calls == int(with_missing_item)
            old_generation = verified[first.native_id].generation
            assert old_generation.prompt_version == legacy_version
            assert old_generation.prompt_sha256 == old_hash != new_hash
            assert runner._GENERATION_INPUTS.get() is new_inputs
            assert current_prompt_version() == PROMPT_VERSION
            monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
            budget = RequestBudget(max_requests=int(with_missing_item))
            descriptions, generations = await runner._descriptions_for_collection(
                snapshot,
                prepared_source,
                prepared,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=runner._AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope="mlxrun_" + "d" * 32,
                verified_resume_outcomes=verified,
            )
            assert generations[first.native_id] == old_generation
            assert budget.requests_used == int(with_missing_item)
            if with_missing_item:
                assert provider.calls == [second.native_id]
                assert paid_versions == [PROMPT_VERSION]
                assert generations[second.native_id].prompt_version == PROMPT_VERSION
                assert generations[second.native_id].prompt_sha256 == new_hash
            else:
                assert not provider.calls and not paid_versions
            if with_concepts:
                assert old_inputs is not None and new_inputs is not None
                assert (
                    old_inputs.concepts.provenance_fields == new_inputs.concepts.provenance_fields
                )
                assert old_inputs.routing_fields != new_inputs.routing_fields
                for field, expected in old_inputs.fields.items():
                    assert getattr(old_generation, field) == expected
                if with_missing_item:
                    for field, expected in new_inputs.fields.items():
                        assert getattr(generations[second.native_id], field) == expected
            analyses = runner._bind_deterministic_analyses(prepared)
            for item in prepared_source.items:
                emoji = transform._emoji(
                    prepared_source.platform,
                    item,
                    prepared[item.native_id],
                    descriptions[item.native_id],
                    analyses[item.native_id],
                    generations[item.native_id],
                    manifest=snapshot.manifest,
                    epoch=0,
                    now="2026-09-13T00:00:00Z",
                )
                assert emoji.provenance.prompt_version == generations[item.native_id].prompt_version
                assert emoji.provenance.prompt_sha256 == generations[item.native_id].prompt_sha256
    finally:
        runner._GENERATION_INPUTS.reset(token)
        cache.close()
    assert snapshot.to_files() == original_files


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["model", "languages", "context", "concepts"])
async def test_legacy_prompt_restore_cannot_bypass_changed_generation_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    if mutation != "languages":
        _install_concept_fixture(snapshot.root)
    config = MojiLexConfig(ai=AIConfig(model="primary-model", model_routing="off"))
    item = _item("legacy-saved", unique_id="saved-unique", file_id="saved-file")
    source = _collection((item,))
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    with use_prompt_version("1.1.0"):
        token = runner._GENERATION_INPUTS.set(
            runner._load_generation_inputs(snapshot, config) if mutation != "languages" else None
        )
        try:
            elements = _seed_resume(cache, source, _processed(snapshot), config)
        finally:
            runner._GENERATION_INPUTS.reset(token)
    if mutation == "model":
        config = config.model_copy(
            update={"ai": config.ai.model_copy(update={"model": "other-model"})}
        )
    elif mutation == "languages":
        config = config.model_copy(
            update={"ai": config.ai.model_copy(update={"languages": ("ru", "en", "fr")})}
        )
    elif mutation == "context":
        source = _collection((item.model_copy(update={"fallback_emoji": "changed-fallback"}),))
    else:
        registry = _registry()
        registry["concepts"][0]["definitions"]["en"] = "Changed candidate definition."
        (snapshot.root / "taxonomy" / "v1" / "concepts.json").write_text(
            json.dumps(registry), encoding="utf-8"
        )
    token = runner._GENERATION_INPUTS.set(
        runner._load_generation_inputs(snapshot, config) if mutation != "languages" else None
    )

    async def forbid_provider(*args, **kwargs):
        pytest.fail("legacy cache probing attempted to contact the AI provider")

    monkeypatch.setattr(runner, "_provider_for_model", forbid_provider)
    verified = {}
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            _, prepared = await runner._prepare_collection_media(
                snapshot,
                _CountingAdapter({"saved-file": _PAYLOAD}),
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope="mlxrun_" + "d" * 32,
                verified_semantic_outcomes=verified,
            )
            assert not verified
            assert processor.decode_calls == 1
            assert prepared[item.native_id].frame_paths
    finally:
        runner._GENERATION_INPUTS.reset(token)
        cache.close()
    assert current_prompt_version() == PROMPT_VERSION

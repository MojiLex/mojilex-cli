from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from mojilex_cli.ai import CostEstimate, DescriptionBatch, DescriptionResult, RequestBudget
from mojilex_cli.ai.prompts import current_prompt_version, prompt_sha256, use_prompt_version
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import new_checkpoint
from test_ai_recovery_checkpoint import _prepare_exact_request
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import (
    _PAYLOAD,
    _analysis,
    _collection,
    _CountingAdapter,
    _CountingProcessor,
    _description,
    _item,
    _processed,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["none", "missing_batch_peer", "different_request_bytes"])
async def test_shifted_legacy_batches_restore_exact_original_groups_without_ai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    original_files = snapshot.to_files()
    config = MojiLexConfig(
        ai=AIConfig(model="primary-model", model_routing="off"),
        processing=ProcessingConfig(static_batch_size=8),
    )
    source = _collection(
        tuple(
            _item(native_id, unique_id=f"unique-{index}", file_id=f"file-{index}")
            for index, native_id in enumerate(
                ("old-single", "old-batch-left", "old-batch-right", "pending")
            )
        )
    )
    singleton, left, right, pending = source.items
    processed = {item.native_id: _processed(snapshot) for item in source.items}
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    for item in source.items[:3]:
        checkpoint = runner._checkpoint_media_item(checkpoint, item, processed[item.native_id])
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    runner._cache_deterministic_analyses(cache, source, processed)
    seed_calls = []

    class Provider:
        name = "gemini"

        def estimate(self, request):
            return CostEstimate(upper_bound_usd=Decimal("0"), note="offline test")

        async def describe(self, request):
            seed_calls.append((current_prompt_version(), request.expected_labels))
            return DescriptionResult(
                batch=DescriptionBatch(
                    items=tuple(
                        _description().model_copy(update={"label": label})
                        for label in request.expected_labels
                    )
                ),
                provider="gemini",
                model=request.model,
                model_revision="synthetic-revision",
            )

    async def fixed_provider(*args, **kwargs):
        return Provider()

    async def persist(items, outcomes):
        nonlocal checkpoint
        subset = source.model_copy(update={"item_count": len(items), "items": tuple(items)})
        checkpoint = runner._checkpoint_ai_keys(
            checkpoint,
            subset,
            {},
            config,
            {native_id: outcome.generation for native_id, outcome in outcomes.items()},
            taxonomy_version="1.0.0",
            request_traces={
                native_id: outcome.request_trace for native_id, outcome in outcomes.items()
            },
        )
        checkpoint = runner._checkpoint_stage(checkpoint, tuple(outcomes), "ai_cached")

    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
    hashes = {}
    try:
        # Populate the production cache/envelope/checkpoint paths through the
        # actual runner, with only rendering and the external provider faked.
        for version, items in (("1.1.0", (singleton,)), ("1.2.0", (left, right))):
            with use_prompt_version(version), TemporaryMediaRun(root=tmp_path) as temporary:
                hashes[version] = prompt_sha256()
                subset = source.model_copy(update={"item_count": len(items), "items": items})
                budget = RequestBudget(max_requests=1)
                await runner._descriptions_for_collection(
                    snapshot,
                    subset,
                    processed,
                    config=config,
                    cache=cache,
                    budget=budget,
                    ai_state=runner._AIState(),
                    api_key=None,
                    redescribe="changed",
                    overwrite_reviewed=False,
                    temporary=temporary,
                    cache_alias_scope=checkpoint.run_id,
                    on_chunk_completed=persist,
                )
                assert budget.requests_used == 1
        assert seed_calls == [("1.1.0", ("E001",)), ("1.2.0", ("E001", "E002"))]
        batch_trace = checkpoint.elements[left.native_id].ai_requests[0]
        assert (
            batch_trace.plan_sha256
            == checkpoint.elements[right.native_id].ai_requests[0].plan_sha256
        )
        elements = dict(checkpoint.elements)
        if mutation == "missing_batch_peer":
            del elements[right.native_id]
        elif mutation == "different_request_bytes":
            element = elements[right.native_id]
            changed = element.ai_requests[0].model_copy(update={"request_sha256": "f" * 64})
            elements[right.native_id] = element.model_copy(update={"ai_requests": (changed,)})
            # The durable envelope correctly repairs a stale trace by design.
            # Corrupt its envelope to exercise inconsistent checkpoint-only groups.
            envelope_key = runner._ai_request_envelope_key(
                checkpoint.run_id,
                stage="primary",
                model=config.ai.model,
                plan_sha256=batch_trace.plan_sha256,
            )
            assert envelope_key is not None
            cache.put_metadata(envelope_key, {"format_version": 0})

        async def forbid_provider(*args, **kwargs):
            pytest.fail("legacy group restoration attempted a new AI request")

        monkeypatch.setattr(runner, "_provider_for_model", forbid_provider)
        verified = {}
        adapter = _CountingAdapter({item.file_id: _PAYLOAD for item in source.items})
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            _, prepared = await runner._prepare_collection_media(
                snapshot,
                adapter,
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope=checkpoint.run_id,
                verified_semantic_outcomes=verified,
            )
        expected = (
            {singleton.native_id, left.native_id, right.native_id}
            if mutation == "none"
            else {singleton.native_id}
        )
        assert set(verified) == expected
        # No retained puzzle tiles were seeded. Rebuild pixels for every item,
        # while exact paid descriptions and deterministic analysis remain reusable.
        assert processor.decode_calls == len(source.items)
        assert prepared[singleton.native_id].frame_paths
        assert verified[singleton.native_id].generation.prompt_version == "1.1.0"
        assert verified[singleton.native_id].generation.prompt_sha256 == hashes["1.1.0"]
        if mutation == "none":
            assert processor.analysis_calls == 1
            for item in (left, right):
                assert prepared[item.native_id].frame_paths
                assert verified[item.native_id].generation.prompt_version == "1.2.0"
                assert verified[item.native_id].generation.prompt_sha256 == hashes["1.2.0"]
        assert prepared[pending.native_id].frame_paths
        assert seed_calls == [("1.1.0", ("E001",)), ("1.2.0", ("E001", "E002"))]
        assert current_prompt_version() == "1.2.1"
        assert snapshot.to_files() == original_files
    finally:
        cache.close()

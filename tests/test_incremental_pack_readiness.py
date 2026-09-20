from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from mojilex_cli.commands.import_reuse import _ready_members
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset import load_dataset, validate_dataset
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.transform import plan_collection_merge
from mojilex_cli.runs import RunStore
from mojilex_cli.runs.pack_scope import source_state
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _collection, _item
from test_pipeline_transform import _description, _generation


def _real_packs(state, monkeypatch, *, dedupe="off"):
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"file_analysis_mode": "fast"}),
            "dedupe": state.config.dedupe.model_copy(update={"mode": dedupe}),
        }
    )
    state.sources = tuple(
        _collection(
            (
                _item(
                    str(7000000000000000000 + index),
                    unique_id=f"unique{index}",
                    file_id=f"file{index}",
                ),
            ),
            native_id=source.native_id,
        )
        for index, source in enumerate(state.sources)
    )
    monkeypatch.setattr(runner, "plan_collection_merge", plan_collection_merge)

    async def describe(snapshot, source, processed, **kwargs):
        descriptions = {item.native_id: _description(snapshot) for item in source.items}
        generations = {item.native_id: _generation() for item in source.items}
        await kwargs["on_chunk_completed"](
            source.items,
            {
                item.native_id: SimpleNamespace(
                    request_trace=(), generation=generations[item.native_id]
                )
                for item in source.items
            },
        )
        return descriptions, generations

    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)


def _assert_durable_pack(state, source):
    snapshot = load_dataset(state.root)
    checkpoint = state.latest_checkpoint()
    assert validate_dataset(state.root, strict=True).valid
    assert source_state(checkpoint, source.canonical_url) == {
        "phase": "describe",
        "status": "succeeded",
    }
    assert _ready_members(checkpoint, source.canonical_url, snapshot) == tuple(
        item.native_id for item in source.items
    )
    return checkpoint


@pytest.mark.parametrize("ending", ["complete", "fail", "cancel", "partial"])
@pytest.mark.parametrize("dedupe", ["off", "exact"])
async def test_ready_pack_is_durable_while_sibling_waits_and_survives_exit(
    request, monkeypatch, ending, dedupe
):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch, dedupe=dedupe)
    alpha, beta = state.sources
    beta_started = asyncio.Event()
    release_beta = asyncio.Event()
    alpha_ready = asyncio.Event()
    events = []

    def stage(source, phase):
        events.append((source, phase))
        if source == alpha.canonical_url and phase == "ready":
            alpha_ready.set()

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == beta.native_id:
            beta_started.set()
            await release_beta.wait()
            if ending == "fail":
                raise CommandError("SOURCE_CHANGED_DURING_RUN", "synthetic", hint="retry")
            if ending == "partial":
                raise CommandError("MEDIA_RENDER_FAILED", "synthetic decode failure", hint="retry")
        else:
            await beta_started.wait()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    monkeypatch.setattr(runner, "report_pack_stage", stage)
    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(alpha_ready.wait(), 20)
        assert not task.done()
        assert (beta.canonical_url, "ready") not in events
        checkpoint = _assert_durable_pack(state, alpha)
        assert source_state(checkpoint, beta.canonical_url)["status"] != "succeeded"
        if dedupe != "off":
            assert checkpoint.elements[alpha.items[0].native_id].candidate_scan_complete
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "fail":
            release_beta.set()
            with pytest.raises(CommandError, match="synthetic"):
                await asyncio.wait_for(task, 20)
        elif ending == "partial":
            release_beta.set()
            result = await asyncio.wait_for(task, 20)
            assert result.status.value == "partial"
            assert [error.code for error in result.errors] == ["MEDIA_RENDER_FAILED"]
            assert (beta.canonical_url, "ready") not in events
            failed_checkpoint = state.latest_checkpoint()
            assert source_state(failed_checkpoint, beta.canonical_url) == {
                "phase": "describe",
                "status": "failed",
            }
            assert (
                _ready_members(failed_checkpoint, beta.canonical_url, load_dataset(state.root))
                is None
            )
        else:
            release_beta.set()
            result = await asyncio.wait_for(task, 20)
            assert not result.errors
            _assert_durable_pack(state, beta)
        checkpoint = _assert_durable_pack(state, alpha)
        if dedupe != "off":
            assert checkpoint.elements[alpha.items[0].native_id].candidate_scan_complete
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


async def test_failed_staging_write_never_reports_pack_ready(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch)
    events = []

    def fail_apply(*args, **kwargs):
        raise CommandError("SOURCE_CHANGED_DURING_RUN", "synthetic write failure", hint="retry")

    monkeypatch.setattr(runner, "_apply_with_rollback", fail_apply)
    monkeypatch.setattr(
        runner, "report_pack_stage", lambda source, phase: events.append((source, phase))
    )
    with pytest.raises(CommandError, match="synthetic write failure"):
        await asyncio.wait_for(state.run(), 20)
    assert all(phase != "ready" for _, phase in events)
    checkpoint = state.latest_checkpoint()
    snapshot = load_dataset(state.root)
    for source in state.sources:
        assert _ready_members(checkpoint, source.canonical_url, snapshot) is None
        assert source_state(checkpoint, source.canonical_url)["status"] != "succeeded"


def test_legacy_cleanup_scope_survives_checkpoint_reload(request):
    state = request.getfixturevalue("pipeline")
    alpha, beta = (source.canonical_url for source in state.sources)
    memberships = {"PackAlpha": ["1001"], "PackBeta": ["1002"]}
    checkpoint = state.checkpoint.model_copy(
        update={
            "safe_parameters": {
                **state.checkpoint.safe_parameters,
                "sources": [alpha, beta],
                "source_memberships": memberships,
                "public_fragment_marker_version": 1,
                "legacy_fragment_sources": [beta],
            }
        }
    )
    store = RunStore(state.config.runs_dir)
    store.save(checkpoint)
    restored = store.load(checkpoint.run_id)
    scoped = runner._legacy_fragment_checkpoint(restored, (alpha, beta))
    assert "public_fragment_marker_version" not in scoped.safe_parameters
    assert scoped.safe_parameters["source_memberships"] == {"PackBeta": ["1002"]}
    assert restored.safe_parameters["source_memberships"] == memberships
    assert restored.safe_parameters["public_fragment_marker_version"] == 1
    assert runner._legacy_fragment_checkpoint(restored, (alpha,)) is restored
    assert store.load(checkpoint.run_id).safe_parameters == restored.safe_parameters


async def test_final_validation_rollback_keeps_already_ready_packs(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch)
    original_apply = runner._apply_with_rollback
    calls = []
    ready = set()
    titles_before_final = {}

    def invalid_report():
        raise CommandError("SOURCE_CHANGED_DURING_RUN", "synthetic final validation", hint="retry")

    def apply(before, after, **kwargs):
        calls.append(before)
        if len(calls) != 3:
            return original_apply(before, after, **kwargs)
        expected_names = {source.native_id for source in state.sources}
        assert expected_names <= {
            collection.native_id for collection in before.collections.values()
        }
        assert before.to_files() == load_dataset(state.root).to_files()
        titles_before_final.update(
            {identifier: collection.title for identifier, collection in before.collections.items()}
        )
        target = next(
            collection
            for collection in after.collections.values()
            if collection.native_id in expected_names
        )
        target.title = "Final candidate change that must roll back"
        with monkeypatch.context() as local_patch:
            local_patch.setattr(
                runner,
                "validate_dataset",
                lambda *args, **kwargs: SimpleNamespace(
                    valid=False, issues=(), raise_for_errors=invalid_report
                ),
            )
            return original_apply(before, after, **kwargs)

    def stage(source, phase):
        if phase == "ready":
            ready.add(source)

    monkeypatch.setattr(runner, "_apply_with_rollback", apply)
    monkeypatch.setattr(runner, "report_pack_stage", stage)
    with pytest.raises(CommandError, match="synthetic final validation"):
        await asyncio.wait_for(state.run(), 20)
    assert len(calls) == 3
    assert ready == {source.canonical_url for source in state.sources}
    for source in state.sources:
        _assert_durable_pack(state, source)
    assert {
        identifier: collection.title
        for identifier, collection in load_dataset(state.root).collections.items()
    } == titles_before_final


async def test_fragment_totals_include_markers_saved_before_final_assembly(request, monkeypatch):
    from mojilex_cli.composition import publication

    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch)
    marked_per_call = []

    def mark(snapshot, groups, sources):
        native_ids = {item.native_id for source in sources for item in source.items}
        changed = set()
        for emoji in snapshot.emojis.values():
            if emoji.native_id in native_ids and "fragment" not in emoji.semantic_tags:
                emoji.semantic_tags = sorted({*emoji.semantic_tags, "fragment"})
                changed.add(emoji.id)
        marked_per_call.append(changed)
        return changed

    monkeypatch.setattr(publication, "mark_verified_fragments", mark)
    result = await asyncio.wait_for(state.run(), 20)
    assert not result.errors
    assert sum(map(len, marked_per_call[:-1])) == 2
    assert marked_per_call[-1] == set()
    assert result.result["fragments_marked"] == 2
    assert result.result["legacy_fragment_tags_removed"] == 0
    for source in state.sources:
        _assert_durable_pack(state, source)

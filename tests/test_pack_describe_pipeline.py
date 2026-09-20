from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.composition import service
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.transform import CollectionPlan
from mojilex_cli.runs import RunStore
from test_add_run_staging import saved_add  # noqa: F401
from test_pipeline_resume_cache import _collection, _item, _processed


@pytest.fixture
def pipeline(request, monkeypatch):
    state = request.getfixturevalue("saved_add")
    state.sources = tuple(
        _collection((_item(name, unique_id=name, file_id=name),), native_id=name)
        for name in ("PackAlpha", "PackBeta")
    )
    state.config = state.config.model_copy(
        update={
            "repository": state.config.repository.model_copy(
                update={"target": str(state.root), "publish": "local"}
            ),
            "processing": state.config.processing.model_copy(
                update={"pack_concurrency": 2, "file_analysis_mode": "sequential"}
            ),
            "dedupe": state.config.dedupe.model_copy(update={"mode": "off"}),
        }
    )
    state.merges = []
    state.callback_sources = []
    state.queue_evidence = {}
    by_url = {source.canonical_url: source for source in state.sources}

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def validate_credentials(self):
            pass

        def canonicalize(self, value):
            source = by_url[value]
            return SimpleNamespace(platform=source.platform, native_id=source.native_id)

        async def fetch_collection(self, reference):
            return next(
                source for source in state.sources if source.native_id == reference.native_id
            )

    def merge(snapshot, source, *args, **kwargs):
        state.merges.append(source.native_id)
        return CollectionPlan(snapshot, "synthetic", 0, 0, 0, ())

    checkpoint_ai_keys = runner._checkpoint_ai_keys

    def capture(checkpoint, source, processed, *args, **kwargs):
        if not processed:
            state.callback_sources.append(
                (source.native_id, tuple(i.native_id for i in source.items))
            )
        return checkpoint_ai_keys(checkpoint, source, processed, *args, **kwargs)

    class Queue:
        def __init__(self, **kwargs):
            pass

        async def prepare(self, key, *args, **kwargs):
            return []

        async def verify(self, **kwargs):
            checkpoint = state.latest_checkpoint()
            state.queue_evidence = checkpoint.safe_parameters.get("composition_evidence", {})
            return {}

    monkeypatch.setattr(runner, "_load_generation_inputs", lambda *args: None)
    monkeypatch.setattr(runner, "_resolved_config", lambda _: state.config)
    monkeypatch.setattr(
        runner,
        "load_credentials",
        lambda: SimpleNamespace(
            telegram_bot_token="fixture", github_token=None, gemini_api_key=None
        ),
    )
    monkeypatch.setattr(runner, "TelegramBotAPI", Adapter)
    monkeypatch.setattr(runner, "plan_collection_merge", merge)
    monkeypatch.setattr(runner, "_checkpoint_ai_keys", capture)
    monkeypatch.setattr(service, "CompositionQueue", Queue)
    state.queue = Queue

    def latest_checkpoint():
        store = RunStore(state.config.runs_dir)
        files = list(state.config.runs_dir.glob("mlxrun_*.json"))
        return next(store.load(path.stem) for path in files if path.stem != state.checkpoint.run_id)

    state.latest_checkpoint = latest_checkpoint

    async def media(snapshot, adapter, source, processor, **kwargs):
        processed = {item.native_id: _processed(snapshot) for item in source.items}
        for item in source.items:
            await kwargs["on_item_completed"](item, processed[item.native_id])
        return source, processed

    async def describe(snapshot, source, processed, **kwargs):
        outcomes = {
            item.native_id: SimpleNamespace(request_trace=(), generation=None)
            for item in source.items
        }
        await kwargs["on_chunk_completed"](source.items, outcomes)
        return {}, {}

    state.media = media
    state.describe = describe
    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)

    async def run():
        return await runner._run_add(
            tuple(source.canonical_url for source in state.sources),
            runner.PipelineOptions(),
            stage_only=True,
        )

    state.run = run
    return state


async def test_pack_stages_finish_in_input_order_before_next_pack_starts(pipeline, monkeypatch):
    state = pipeline
    ai_alpha = asyncio.Event()
    media_beta = asyncio.Event()
    ai_beta_done = asyncio.Event()
    order = []

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackBeta":
            assert ai_alpha.is_set()
            assert order == ["PackAlpha"]
            media_beta.set()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def describe(snapshot, source, processed, **kwargs):
        if source.native_id == "PackAlpha":
            ai_alpha.set()
            assert not media_beta.is_set()
            assert not ai_beta_done.is_set()
        result = await state.describe(snapshot, source, processed, **kwargs)
        order.append(source.native_id)
        if source.native_id == "PackBeta":
            ai_beta_done.set()
        return result

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    result = await asyncio.wait_for(state.run(), 5)
    assert not result.errors
    assert order == ["PackAlpha", "PackBeta"]
    assert state.merges == ["PackAlpha", "PackBeta"]
    assert sorted(state.callback_sources) == [
        ("PackAlpha", ("PackAlpha",)),
        ("PackBeta", ("PackBeta",)),
    ]
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert checkpoint.safe_parameters["sources"] == [s.canonical_url for s in state.sources]
    assert set(checkpoint.elements) == {"PackAlpha", "PackBeta"}
    assert all(element.stage == "validated" for element in checkpoint.elements.values())
    assert all(element.fingerprint_complete for element in checkpoint.elements.values())


async def test_composition_evidence_is_merged_from_latest_checkpoint_after_await(pipeline):
    state = pipeline
    beta_entered = asyncio.Event()
    alpha_done = asyncio.Event()

    async def prepare(self, key, *args, **kwargs):
        if key == "PackAlpha":
            assert not beta_entered.is_set()
            alpha_done.set()
        else:
            beta_entered.set()
            assert alpha_done.is_set()
        return [SimpleNamespace(model_dump=lambda **_: {"pack": key})]

    state.queue.prepare = prepare
    result = await asyncio.wait_for(state.run(), 5)
    assert not result.errors
    assert state.queue_evidence == {
        "PackAlpha": [{"pack": "PackAlpha"}],
        "PackBeta": [{"pack": "PackBeta"}],
    }


@pytest.mark.parametrize("external_cancel", [False, True])
async def test_terminal_exit_drains_current_pack_without_starting_next(
    pipeline, monkeypatch, external_cancel
):
    state = pipeline
    alpha_started = asyncio.Event()
    both_started = asyncio.Event()
    never = asyncio.Event()
    active = set()
    finished = set()

    async def describe(snapshot, source, processed, **kwargs):
        key = source.native_id
        active.add(key)
        try:
            if key == "PackAlpha":
                alpha_started.set()
                both_started.set()
                if not external_cancel:
                    raise CommandError("SOURCE_CHANGED_DURING_RUN", "synthetic", hint="retry")
            await never.wait()
        finally:
            # The worker is drained before the enclosing cache can be closed.
            kwargs["cache"]._connection.execute("SELECT 1").fetchone()
            active.remove(key)
            finished.add(key)

    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    await asyncio.wait_for(both_started.wait(), 5)
    if external_cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if external_cancel else CommandError):
        await asyncio.wait_for(task, 5)
    assert not active
    assert finished == {"PackAlpha"}
    assert not state.merges
    checkpoint = state.latest_checkpoint()
    assert set(checkpoint.elements) == {"PackAlpha"}
    assert checkpoint.status == ("interrupted" if external_cancel else "stale")


async def test_collection_and_membership_changes_do_not_abort_source_merge(pipeline, monkeypatch):
    state = pipeline
    produced = []
    consumed = []

    def merge(snapshot, source, *args, **kwargs):
        consumed.append(snapshot)
        if produced:
            assert snapshot is produced[-1]
        after = snapshot.clone()
        produced.append(after)
        collection_id = next(iter(snapshot.collections))
        emoji_id = next(iter(snapshot.emojis))
        membership_id = next(iter(snapshot.memberships))
        # Real plans include all three entity kinds. Non-emoji IDs must only
        # skip dedupe accounting, never return from the surrounding source worker.
        return CollectionPlan(
            after, collection_id, 0, 0, 0, (collection_id, emoji_id, membership_id)
        )

    monkeypatch.setattr(runner, "plan_collection_merge", merge)
    result = await asyncio.wait_for(state.run(), 5)
    assert not result.errors
    assert result.result["sources_processed"] == 2
    assert len(produced) == len(consumed) == 2
    assert consumed[1] is produced[0]
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert all(element.stage == "validated" for element in checkpoint.elements.values())


async def test_fast_mode_prepares_next_pack_while_first_ai_runs(pipeline, monkeypatch):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 2}
            )
        }
    )
    alpha_ai = asyncio.Event()
    beta_media = asyncio.Event()
    release_alpha = asyncio.Event()
    release_beta_media = asyncio.Event()
    ai_order = []

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackBeta":
            beta_media.set()
            await release_beta_media.wait()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def describe(snapshot, source, processed, **kwargs):
        ai_order.append(source.native_id)
        if source.native_id == "PackAlpha":
            alpha_ai.set()
            await release_alpha.wait()
        return await state.describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(alpha_ai.wait(), 5)
        await asyncio.wait_for(beta_media.wait(), 5)
        assert ai_order == ["PackAlpha"]
        assert not release_beta_media.is_set()
        release_beta_media.set()
        release_alpha.set()
        result = await asyncio.wait_for(task, 5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert not result.errors
    assert ai_order == ["PackAlpha", "PackBeta"]


async def test_fast_mode_does_not_spend_ai_on_prefetched_pack_after_prior_failure(
    pipeline, monkeypatch
):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 2}
            )
        }
    )
    beta_media = asyncio.Event()
    release_beta = asyncio.Event()
    ai_order = []

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackBeta":
            beta_media.set()
            await release_beta.wait()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def describe(snapshot, source, processed, **kwargs):
        ai_order.append(source.native_id)
        if source.native_id == "PackAlpha":
            await beta_media.wait()
            release_beta.set()
            raise CommandError("SOURCE_CHANGED_DURING_RUN", "synthetic", hint="retry")
        pytest.fail("prefetched pack started AI after the prior pack failed")

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    with pytest.raises(CommandError):
        await state.run()
    assert ai_order == ["PackAlpha"]


@pytest.mark.parametrize("mode", ["sequential", "fast"])
async def test_parallel_packs_write_real_collections_emojis_and_memberships(
    pipeline, monkeypatch, mode
):
    from mojilex_cli.dataset import load_dataset, validate_dataset
    from mojilex_cli.pipeline.transform import plan_collection_merge
    from test_pipeline_transform import _description, _generation

    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"file_analysis_mode": mode})
        }
    )
    beta_finished = asyncio.Event()
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
    produced = []

    def merge(snapshot, source, *args, **kwargs):
        if produced:
            assert snapshot is produced[-1]
        plan = plan_collection_merge(snapshot, source, *args, **kwargs)
        assert any(value.startswith("mxc_") for value in plan.changed_entity_ids)
        assert any(value.startswith("mxm_") for value in plan.changed_entity_ids)
        produced.append(plan.snapshot)
        return plan

    async def describe(snapshot, source, processed, **kwargs):
        if mode == "fast" and source.native_id == "PackAlpha":
            await beta_finished.wait()
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
        if source.native_id == "PackBeta":
            beta_finished.set()
        return descriptions, generations

    monkeypatch.setattr(runner, "plan_collection_merge", merge)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    result = await asyncio.wait_for(state.run(), 8)
    assert not result.errors
    assert result.result["sources_processed"] == 2
    assert result.result["collections_created"] == 2
    assert result.result["items_added"] == 2
    assert result.result["changed_paths"]
    published = load_dataset(state.root)
    assert len(published.collections) == len(state.snapshot.collections) + 2
    assert len(published.emojis) == len(state.snapshot.emojis) + 2
    assert len(published.memberships) == len(state.snapshot.memberships) + 2
    assert validate_dataset(state.root, strict=True).valid


async def test_explicit_missing_pack_keeps_collection_lock_during_mutation(pipeline, monkeypatch):
    from contextlib import contextmanager

    from mojilex_cli.sources.base import SourceNotFoundError

    state = pipeline
    held = set()
    observed = []
    original_lock = RunStore.collection_lock

    @contextmanager
    def collection_lock(store, platform, native_id, **kwargs):
        with original_lock(store, platform, native_id, **kwargs):
            held.add(native_id)
            try:
                yield
            finally:
                held.remove(native_id)

    async def missing(self, reference):
        raise SourceNotFoundError("synthetic missing pack")

    async def mark(snapshot, adapter, *, platform, native_id):
        assert native_id in held
        observed.append(native_id)
        return snapshot, 0

    monkeypatch.setattr(RunStore, "collection_lock", collection_lock)
    monkeypatch.setattr(runner.TelegramBotAPI, "fetch_collection", missing)
    monkeypatch.setattr(runner, "_mark_missing_collection", mark)
    result = await asyncio.wait_for(
        runner._run_add(
            tuple(source.canonical_url for source in state.sources),
            runner.PipelineOptions(explicit_verification=True),
            stage_only=True,
        ),
        5,
    )
    assert not result.errors
    assert observed == ["PackAlpha", "PackBeta"]
    assert not held


async def test_dashboard_ready_is_reported_only_after_pack_finalization(pipeline, monkeypatch):
    events = []
    monkeypatch.setattr(
        runner, "report_pack_stage", lambda source, phase: events.append((source, phase))
    )
    result = await pipeline.run()
    assert not result.errors
    for source in pipeline.sources:
        phases = [phase for name, phase in events if name == source.canonical_url]
        assert phases == [
            "download",
            "waiting",
            "render",
            "ai_wait",
            "ai",
            "composition",
            "merge_wait",
            "finalize",
            "ready",
        ]
    first_ready = next(i for i, event in enumerate(events) if event[1] == "ready")
    assert any(event[1] == "ai" for event in events[first_ready:])


async def test_fast_puzzle_verification_starts_before_other_pack_media_finishes(
    pipeline, monkeypatch
):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"file_analysis_mode": "fast"})
        }
    )
    beta_waiting = asyncio.Event()
    release_beta = asyncio.Event()
    alpha_verified = asyncio.Event()

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackBeta":
            beta_waiting.set()
            await release_beta.wait()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def verify(self, **kwargs):
        if kwargs.get("key") == "PackAlpha":
            await beta_waiting.wait()
            assert not release_beta.is_set()
            alpha_verified.set()
        return {}

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(state.queue, "verify", verify)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(alpha_verified.wait(), 5)
        release_beta.set()
        result = await asyncio.wait_for(task, 5)
        assert not result.errors
    finally:
        release_beta.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_fast_ai_overlaps_previous_packs_local_postprocessing(pipeline, monkeypatch):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 2}
            )
        }
    )
    local_started = asyncio.Event()
    beta_ai = asyncio.Event()
    release_local = asyncio.Event()

    async def prepare(self, key, *args, **kwargs):
        if key == "PackAlpha":
            local_started.set()
            await release_local.wait()
        return []

    async def describe(snapshot, source, processed, **kwargs):
        if source.native_id == "PackBeta":
            beta_ai.set()
        return await state.describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(state.queue, "prepare", prepare)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(local_started.wait(), 5)
        await asyncio.wait_for(beta_ai.wait(), 5)
        assert not release_local.is_set()
        release_local.set()
        result = await asyncio.wait_for(task, 5)
        assert not result.errors
        assert sorted(state.merges) == ["PackAlpha", "PackBeta"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("confirm", [False, True])
async def test_runner_applies_current_confirmation_setting_without_changing_budget(
    pipeline, monkeypatch, confirm
):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "ai": state.config.ai.model_copy(
                update={
                    "confirm_before_analysis": confirm,
                    "allow_unknown_cost": False,
                    "max_ai_requests": 37,
                }
            )
        }
    )
    captured = {}
    original = runner.RequestBudget

    def budget(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(runner, "RequestBudget", budget)
    result = await state.run()
    assert not result.errors
    assert captured["confirm_before_requests"] is confirm
    assert captured["allow_unknown_cost"] is (not confirm)
    assert captured["max_requests"] == 37


async def test_fast_ready_pack_starts_ai_before_earlier_pack_finishes_media(pipeline, monkeypatch):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 2}
            )
        }
    )
    alpha_media = asyncio.Event()
    beta_ai = asyncio.Event()
    beta_merged = asyncio.Event()
    release_alpha = asyncio.Event()
    order = []

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackAlpha":
            alpha_media.set()
            await release_alpha.wait()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def describe(snapshot, source, processed, **kwargs):
        order.append(source.native_id)
        if source.native_id == "PackBeta":
            beta_ai.set()
        return await state.describe(snapshot, source, processed, **kwargs)

    original_merge = runner.plan_collection_merge
    loop = asyncio.get_running_loop()

    def merge(snapshot, source, *args, **kwargs):
        plan = original_merge(snapshot, source, *args, **kwargs)
        if source.native_id == "PackBeta":
            loop.call_soon_threadsafe(beta_merged.set)
        return plan

    monkeypatch.setattr(runner, "plan_collection_merge", merge)
    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(alpha_media.wait(), 5)
        await asyncio.wait_for(beta_ai.wait(), 5)
        assert order == ["PackBeta"]
        await asyncio.wait_for(beta_merged.wait(), 5)
        assert state.merges == ["PackBeta"]
        release_alpha.set()
        result = await asyncio.wait_for(task, 5)
        assert not result.errors
        assert state.merges == ["PackBeta", "PackAlpha"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_fast_independent_packs_can_have_ai_in_flight_together(pipeline, monkeypatch):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 2}
            )
        }
    )
    entered = set()
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def describe(snapshot, source, processed, **kwargs):
        entered.add(source.native_id)
        if len(entered) == 2:
            both_started.set()
        await release.wait()
        return await state.describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(both_started.wait(), 5)
        release.set()
        result = await asyncio.wait_for(task, 5)
        assert not result.errors
        assert sorted(state.merges) == ["PackAlpha", "PackBeta"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_fast_prefetch_continues_while_only_pack_slot_is_in_ai(pipeline, monkeypatch):
    state = pipeline
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 1}
            )
        }
    )
    alpha_ai = asyncio.Event()
    beta_media = asyncio.Event()
    release_ai = asyncio.Event()

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackBeta":
            assert alpha_ai.is_set()
            beta_media.set()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def describe(snapshot, source, processed, **kwargs):
        if source.native_id == "PackAlpha":
            alpha_ai.set()
            await release_ai.wait()
        return await state.describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(beta_media.wait(), 10)
        assert not task.done()
    finally:
        release_ai.set()
        result = await asyncio.wait_for(task, 15)
    assert result.status in {"succeeded", "noop"}
    assert sorted(state.merges) == ["PackAlpha", "PackBeta"]

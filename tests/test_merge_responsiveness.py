from __future__ import annotations

import asyncio
import threading

from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.transform import CollectionPlan
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401


async def test_candidate_merge_keeps_event_loop_responsive_and_input_order(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"file_analysis_mode": "fast"})
        }
    )
    loop = asyncio.get_running_loop()
    event_loop_thread = threading.get_ident()
    responsive_merges = []

    def merge(snapshot, source, *args, **kwargs):
        heartbeat = threading.Event()
        loop.call_soon_threadsafe(heartbeat.set)
        # A merge on the event-loop thread cannot service this callback. This
        # simulates a slow clone without relying on wall-clock timing assertions.
        assert heartbeat.wait(2), "candidate assembly blocked network/UI callbacks"
        assert threading.get_ident() != event_loop_thread
        responsive_merges.append(source.native_id)
        return CollectionPlan(snapshot, "synthetic", 0, 0, 0, ())

    monkeypatch.setattr(runner, "plan_collection_merge", merge)
    result = await asyncio.wait_for(state.run(), 10)
    assert not result.errors
    assert responsive_merges == ["PackAlpha", "PackBeta"]


async def test_missing_pack_update_uses_completed_preceding_candidate(request, monkeypatch):
    from mojilex_cli.sources.base import SourceNotFoundError

    state = request.getfixturevalue("pipeline")
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"file_analysis_mode": "fast"})
        }
    )
    merge_started = threading.Event()
    release_merge = threading.Event()
    missing_observed = asyncio.Event()
    candidates = []
    observed = []
    original_fetch = runner.TelegramBotAPI.fetch_collection

    def merge(snapshot, source, *args, **kwargs):
        candidate = snapshot.clone()
        candidates.append(candidate)
        merge_started.set()
        assert release_merge.wait(5)
        return CollectionPlan(candidate, "synthetic", 0, 0, 0, ())

    async def fetch(adapter, reference):
        if reference.native_id == "PackBeta":
            assert await asyncio.to_thread(merge_started.wait, 5)
            missing_observed.set()
            raise SourceNotFoundError("synthetic missing pack")
        return await original_fetch(adapter, reference)

    async def mark(snapshot, adapter, *, platform, native_id):
        # Updating the old snapshot here would be overwritten when Alpha's
        # worker returns its candidate and assigns it to the shared current state.
        assert snapshot is candidates[0]
        observed.append(native_id)
        return snapshot, 0

    async def refresh(snapshot, *args, **kwargs):
        return snapshot, 0

    monkeypatch.setattr(runner, "plan_collection_merge", merge)
    monkeypatch.setattr(runner.TelegramBotAPI, "fetch_collection", fetch)
    monkeypatch.setattr(runner, "_mark_missing_collection", mark)
    monkeypatch.setattr(runner, "_refresh_emoji_availability", refresh)
    task = asyncio.create_task(
        runner._run_add(
            tuple(source.canonical_url for source in state.sources),
            runner.PipelineOptions(explicit_verification=True),
            stage_only=True,
        )
    )
    try:
        await asyncio.wait_for(missing_observed.wait(), 10)
    finally:
        release_merge.set()
    result = await asyncio.wait_for(task, 10)
    assert not result.errors
    assert observed == ["PackBeta"]

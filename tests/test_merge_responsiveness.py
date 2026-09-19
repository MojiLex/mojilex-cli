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

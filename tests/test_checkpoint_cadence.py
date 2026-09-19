from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _item, _processed


async def test_media_checkpoint_is_coalesced_and_interruption_flushes_every_item(
    request, monkeypatch
):
    state = request.getfixturevalue("pipeline")
    writes = []
    save = RunStore.save

    def tracked_save(self, checkpoint):
        writes.append(len(checkpoint.elements))
        return save(self, checkpoint)

    monkeypatch.setattr(RunStore, "save", tracked_save)
    # A fixed clock isolates batching from test machine speed.
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 0.0))

    async def media(snapshot, adapter, source, processor, **kwargs):
        value = _processed(snapshot)
        for index in range(40):
            item = _item(str(900000 + index), unique_id=str(index), file_id=str(index))
            await kwargs["on_item_completed"](item, value)
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    with pytest.raises(asyncio.CancelledError):
        await state.run()
    checkpoint = state.latest_checkpoint()
    assert all(str(900000 + index) in checkpoint.elements for index in range(40))
    assert sum(count >= 32 for count in writes) < 5


async def test_independent_pack_runs_after_recoverable_pack_error(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"file_analysis_mode": "fast"})
        }
    )
    failed = asyncio.Event()
    original = state.media

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id == "PackAlpha":
            failed.set()
            raise CommandError("MEDIA_RENDER_FAILED", "synthetic invalid media", hint="retry")
        await failed.wait()
        return await original(snapshot, adapter, source, processor, **kwargs)

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    result = await state.run()
    assert [e.code for e in result.errors] == ["MEDIA_RENDER_FAILED"]
    assert state.merges == ["PackBeta"]

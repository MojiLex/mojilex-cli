from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mojilex_cli.pipeline import runner
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _collection, _item


async def test_dependency_waiters_do_not_occupy_media_preparation_slots(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={"file_analysis_mode": "fast", "pack_concurrency": 2}
            )
        }
    )
    names = ("PackAlpha", "PackBeta", "PackGamma", "PackDelta")
    shared = _item("SharedEmoji", unique_id="shared", file_id="shared")
    independent = _item("IndependentEmoji", unique_id="independent", file_id="independent")
    state.sources = tuple(
        _collection((independent if name == "PackDelta" else shared,), native_id=name)
        for name in names
    )

    class Adapter(runner.TelegramBotAPI):
        def canonicalize(self, value):
            source = next(source for source in state.sources if source.canonical_url == value)
            return SimpleNamespace(platform=source.platform, native_id=source.native_id)

    alpha_ai = asyncio.Event()
    delta_media = asyncio.Event()
    release_alpha = asyncio.Event()
    active_dependent_media = []

    async def media(snapshot, adapter, source, processor, **kwargs):
        if source.native_id in {"PackBeta", "PackGamma"}:
            assert release_alpha.is_set(), "Shared emoji must retain its original owner"
            active_dependent_media.append(source.native_id)
        if source.native_id == "PackDelta":
            assert alpha_ai.is_set()
            delta_media.set()
        return await state.media(snapshot, adapter, source, processor, **kwargs)

    async def describe(snapshot, source, processed, **kwargs):
        if source.native_id == "PackAlpha":
            alpha_ai.set()
            await release_alpha.wait()
        return await state.describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(runner, "TelegramBotAPI", Adapter)
    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    task = asyncio.create_task(state.run())
    try:
        await asyncio.wait_for(alpha_ai.wait(), 10)
        await asyncio.wait_for(delta_media.wait(), 5)
        assert not active_dependent_media
        assert not set(state.merges).intersection(names[:3])
    finally:
        release_alpha.set()
        result = await asyncio.wait_for(task, 15)
    assert not result.errors
    # Independent packs may finish first; shared identities retain source order.
    assert sorted(state.merges) == sorted(names)
    assert [name for name in state.merges if name != "PackDelta"] == list(names[:3])
    assert active_dependent_media == ["PackBeta", "PackGamma"]

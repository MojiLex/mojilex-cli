from types import SimpleNamespace

from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _processed


async def test_cached_media_bursts_and_slow_writes_leave_time_for_finalization(
    request, monkeypatch
):
    state = request.getfixturevalue("pipeline")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now))
    original_save = RunStore.save
    media_saves = []
    in_media = False

    def save(store, checkpoint):
        result = original_save(store, checkpoint)
        if in_media:
            media_saves.append(clock.now)
            clock.now += 2.0  # A slow disk/checkpoint must not trigger the next write.
        return result

    async def media(snapshot, adapter, source, processor, **kwargs):
        nonlocal in_media
        item = source.items[0]
        value = _processed(snapshot)
        callback = kwargs["on_item_completed"]
        in_media = True
        try:
            before = len(media_saves)
            # A resumed burst can deliver more than 32 cached items at once.
            for _ in range(65):
                await callback(item, value)
            assert len(media_saves) == before
            clock.now += 1.0
            await callback(item, value)
            assert len(media_saves) == before + 1
            for _ in range(65):
                await callback(item, value)
            assert len(media_saves) == before + 1
            clock.now += 1.0
            await callback(item, value)
            assert len(media_saves) == before + 2
        finally:
            in_media = False
        return source, {item.native_id: value}

    monkeypatch.setattr(RunStore, "save", save)
    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    result = await state.run()
    assert not result.errors
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert all(element.stage == "validated" for element in checkpoint.elements.values())
    assert len(media_saves) == 4

"""The actual add runner marks receipts after its durable pack boundary."""

import hashlib

from PIL import Image

from mojilex_cli.concurrency import current_batch_limits
from mojilex_cli.media.resume import get_retained_store
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.retention import restore_completed_media_receipts
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _collection, _item, _processed


async def test_runner_completed_pack_previews_yield_space_for_next_pack(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    state.sources = tuple(
        _collection(
            tuple(
                _item(
                    f"{source.native_id}-{index}",
                    unique_id=f"{source.native_id}-{index}",
                    file_id=f"{source.native_id}-{index}",
                )
                for index in range(5)
            ),
            native_id=source.native_id,
        )
        for source in state.sources
    )
    maximum = 12_000
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(update={"max_temp_bytes": maximum})
        }
    )
    stores = []
    temporary_paths = []
    reclaimed = []
    boundary = runner.mark_completed_retained

    def capture_boundary(root, completed):
        checkpoint = state.latest_checkpoint()
        assert all(element.stage == "validated" for element in checkpoint.elements.values())
        assert all(not path.exists() for path in temporary_paths)
        assert set(completed) <= set(restore_completed_media_receipts(checkpoint))
        boundary(root, completed)

    async def media(snapshot, adapter, source, processor, **kwargs):
        scope = kwargs["cache_alias_scope"]
        root = (
            kwargs["cache"].path.parent
            / "resume-media"
            / hashlib.sha256(scope.encode()).hexdigest()
        )
        retained = get_retained_store(
            root,
            max_bytes=maximum,
            run=processor.run,
            completed=kwargs["completed_retained"],
        )
        stores.append(retained)
        if len(stores) == 1:
            original = retained.discard

            def discard(key, expected):
                result = original(key, expected)
                if result:
                    reclaimed.append(key)
                return result

            monkeypatch.setattr(retained, "discard", discard)
        processed = {}
        temporary_paths.clear()
        for index, item in enumerate(source.items):
            frame = processor.run.path / f"generated-{index}.png"
            Image.new("RGB", (256, 256), (200, 20, 80)).save(frame)
            temporary_paths.append(frame)
            processor.run.account_outputs([frame])
            value = _processed(snapshot).model_copy(
                update={"frame_paths": (frame,), "has_dark_render": False}
            )
            processed[item.native_id] = value
            await kwargs["on_item_completed"](item, value)
            assert retained.put(runner._source_descriptor_sha256(item), value)
            assert current_batch_limits().temp_budget.used <= maximum
        return source, processed

    monkeypatch.setattr(runner, "mark_completed_retained", capture_boundary)
    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    result = await state.run()
    assert not result.errors
    assert len(stores) == 2 and stores[0] is stores[1]
    assert reclaimed
    assert len(restore_completed_media_receipts(state.latest_checkpoint())) == 10
    assert stores[0].size_bytes <= maximum

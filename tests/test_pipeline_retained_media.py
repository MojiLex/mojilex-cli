from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.cache import CacheStore
from mojilex_cli.media import MediaLimits, MediaProcessor, TemporaryMediaRun
from mojilex_cli.media.resume import RetainedMediaStore
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import new_checkpoint
from mojilex_cli.sources.base import SourceNetworkError
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _collection, _item, _processed


@pytest.mark.parametrize("change", ["none", "tampered_frame", "changed_descriptor"])
async def test_interrupted_media_run_reuses_only_exact_retained_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    original_files = snapshot.to_files()
    items = tuple(
        _item(
            f"retained-{i}",
            unique_id=f"unique-{i}",
            file_id=f"file-{i}",
            payload=f"body-{i}".encode(),
        )
        for i in range(4)
    )
    collection = _collection(items)
    bodies = {item.native_id: f"body-{i}".encode() for i, item in enumerate(items)}
    native_by_body = {body: native_id for native_id, body in bodies.items()}
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    scope = checkpoint.run_id
    cache_path = tmp_path / "private-cache" / "cache.sqlite3"
    cache = CacheStore(cache_path, repository_root=snapshot.root)
    calls = []
    rendered = []
    completed = []
    original_frame_paths = []
    progress_objects = []
    reports = []
    backend = _processed(snapshot).analysis.decoder_backend_fingerprint
    monkeypatch.setattr(runner, "_decoder_backend_candidates", lambda *_: (backend,))
    monkeypatch.setattr(runner, "report_progress", reports.append)
    original_progress = runner.BatchProgress

    def progress(*args, **kwargs):
        value = original_progress(*args, **kwargs)
        progress_objects.append(value)
        return value

    monkeypatch.setattr(runner, "BatchProgress", progress)

    class Adapter:
        fail_id: str | None = items[2].native_id

        async def fetch_media(self, item):
            calls.append(item.native_id)
            if item.native_id == self.fail_id:
                yield b"partial"
                raise SourceNetworkError("synthetic exhausted disconnect")
            yield bodies[item.native_id]

    class Worker:
        limits = MediaLimits()

        def process(self, source, output, **kwargs):
            payload = source.read_bytes()
            rendered.append(native_by_body[payload])
            output.mkdir()
            frame = output / "light.png"
            Image.new("RGBA", (256, 256), (123, 45, 67, 255)).save(frame)
            return _processed(snapshot, payload).model_copy(update={"frame_paths": (frame,)})

    adapter = Adapter()
    worker = Worker()

    async def persist(item, value):
        nonlocal checkpoint
        runner._cache_deterministic_analysis(cache, item, value)
        checkpoint = runner._checkpoint_media_item(checkpoint, item, value)
        completed.append(item.native_id)
        original_frame_paths.extend(value.frame_paths)

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            with pytest.raises(SourceNetworkError, match="synthetic exhausted"):
                await runner._prepare_collection_media(
                    snapshot,
                    adapter,
                    collection,
                    MediaProcessor(temporary, worker),
                    concurrency=1,
                    cache=cache,
                    cache_alias_scope=scope,
                    on_item_completed=persist,
                )
        assert calls == [item.native_id for item in items[:3]]
        assert rendered == completed == [item.native_id for item in items[:2]]
        assert all(not path.exists() for path in original_frame_paths)
        assert set(checkpoint.elements) == {item.native_id for item in items[:2]}

        # Reopen durable storage after the original temporary run is gone.
        cache.close()
        cache = CacheStore(cache_path, repository_root=snapshot.root)
        retained_root = (
            cache.path.parent / "resume-media" / hashlib.sha256(scope.encode()).hexdigest()
        )
        assert len(list(retained_root.glob("*/manifest.json"))) == 2
        assert not retained_root.is_relative_to(snapshot.root)
        if change == "tampered_frame":
            key = runner._source_descriptor_sha256(collection.items[0])
            frame = retained_root / key / "light-00.png"
            frame.write_bytes(b"damaged PNG")
        elif change == "changed_descriptor":
            changed = collection.items[0].model_copy(update={"file_unique_id": "changed-unique"})
            collection = _collection((changed, *collection.items[1:]))

        calls.clear()
        rendered.clear()
        completed.clear()
        adapter.fail_id = None
        expected_ready = 2 if change == "none" else 1
        first_media_progress_count = []
        original_enter = original_progress.__aenter__

        async def enter(value):
            if value.label.endswith(collection.native_id):
                first_media_progress_count.append(value.completed)
            return await original_enter(value)

        monkeypatch.setattr(original_progress, "__aenter__", enter)
        with TemporaryMediaRun(root=tmp_path) as resumed:
            _, processed = await runner._prepare_collection_media(
                snapshot,
                adapter,
                collection,
                MediaProcessor(resumed, worker),
                concurrency=1,
                cache=cache,
                cache_alias_scope=scope,
                resume_elements=checkpoint.elements,
                expected_hashes={
                    key: value.media_sha256 for key, value in checkpoint.elements.items()
                },
                on_item_completed=persist,
            )
            expected_work = [item.native_id for item in items[2:]]
            if change != "none":
                expected_work.insert(0, items[0].native_id)
            assert calls == rendered == completed == expected_work
            assert set(processed) == {item.native_id for item in items}
            assert all(path.is_file() for value in processed.values() for path in value.frame_paths)
            assert processed[items[1].native_id].frame_paths[0].is_relative_to(retained_root)
        assert first_media_progress_count == [expected_ready]
        assert progress_objects[-1].completed == 4
        assert progress_objects[-1].failed == 0
        assert any(f"{expected_ready}/4" in report for report in reports)
        assert len(checkpoint.elements) == 4
        assert snapshot.to_files() == original_files
    finally:
        cache.close()


def _one_item_media_dependencies(snapshot):
    payload = b"one-item-media"
    item = _item("retained-one", unique_id="unique-one", file_id="file-one", payload=payload)

    class Adapter:
        async def fetch_media(self, _item):
            yield payload

    class Worker:
        limits = MediaLimits()

        def process(self, source, output, **kwargs):
            output.mkdir()
            frame = output / "light.png"
            Image.new("RGBA", (256, 256), (123, 45, 67, 255)).save(frame)
            return _processed(snapshot, source.read_bytes()).model_copy(
                update={"frame_paths": (frame,)}
            )

    return item, Adapter(), Worker()


async def test_retention_finishes_before_cancelled_run_removes_temporary_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    item, adapter, worker = _one_item_media_dependencies(snapshot)
    cache = CacheStore(tmp_path / "private-cache" / "cache.sqlite3", repository_root=snapshot.root)
    scope = "test-cancelled-run"
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_write = RetainedMediaStore._write
    values = []
    temporary_paths = []

    def delayed_write(path, data):
        loop.call_soon_threadsafe(started.set)
        # This runs inside RetainedMediaStore's worker thread, never the event loop.
        if not release.wait(timeout=5):
            raise OSError("test writer was not released")
        original_write(path, data)

    monkeypatch.setattr(RetainedMediaStore, "_write", staticmethod(delayed_write))

    async def persist(_item, value):
        values.append(value)

    async def run_media():
        with TemporaryMediaRun(root=tmp_path) as temporary:
            temporary_paths.append(temporary.path)
            await runner._prepare_collection_media(
                snapshot,
                adapter,
                _collection((item,)),
                MediaProcessor(temporary, worker),
                concurrency=1,
                cache=cache,
                cache_alias_scope=scope,
                on_item_completed=persist,
            )

    task = asyncio.create_task(run_media())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        try:
            done, _ = await asyncio.wait({task}, timeout=0.05)
            assert not done
            assert values and values[0].frame_paths[0].is_file()
            assert temporary_paths[0].is_dir()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not temporary_paths[0].exists()
        root = cache.path.parent / "resume-media" / hashlib.sha256(scope.encode()).hexdigest()
        restored = RetainedMediaStore(root, 100_000).get(
            runner._source_descriptor_sha256(item),
            values[0],
        )
        assert restored is not None
        assert restored.frame_paths[0].is_file()
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        cache.close()


async def test_retention_does_not_write_into_dataset_named_resume_media(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "resume-media")
    original_files = snapshot.to_files()
    item, adapter, worker = _one_item_media_dependencies(snapshot)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    scope = "test-dataset-directory-collision"
    completed = []

    async def persist(_item, value):
        completed.append(value)

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            _, processed = await runner._prepare_collection_media(
                snapshot,
                adapter,
                _collection((item,)),
                MediaProcessor(temporary, worker),
                concurrency=1,
                cache=cache,
                cache_alias_scope=scope,
                on_item_completed=persist,
            )
            assert item.native_id in processed
            assert completed
            assert processed[item.native_id].frame_paths[0].is_relative_to(temporary.path)
        assert not (snapshot.root / hashlib.sha256(scope.encode()).hexdigest()).exists()
        assert not list(snapshot.root.rglob("*.png"))
        assert snapshot.to_files() == original_files
    finally:
        cache.close()

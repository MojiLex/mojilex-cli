from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from mojilex_cli.cache import CacheStore
from mojilex_cli.cache import store as cache_module
from mojilex_cli.media import MediaProcessor, TemporaryMediaRun
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import ElementCheckpoint
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _item, _processed


def test_metadata_reads_touch_lru_once_per_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100]
    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        cache.put_metadata("test", {"value": 1})
        changes = cache._connection.total_changes
        for _ in range(10):
            assert cache.get_metadata("test") == {"value": 1}
        assert cache._connection.total_changes == changes
        clock[0] = 101
        cache.get_metadata("test")
        assert cache._connection.total_changes == changes
        cache._flush_accesses()
        assert cache._connection.total_changes == changes + 1
        assert (
            cache._connection.execute(
                "SELECT accessed_at FROM metadata_cache WHERE cache_key = ?", ("test",)
            ).fetchone()[0]
            == 101
        )
        cache.get_metadata("test")
        assert cache._connection.total_changes == changes + 1


def test_async_restore_keeps_sqlite_on_owner_and_releases_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    source = _item("responsive", unique_id="responsive", file_id="responsive")
    value = _processed(snapshot)
    checkpoint = ElementCheckpoint(
        stage="fingerprint_ready",
        source_descriptor_sha256=runner._source_descriptor_sha256(source),
        media_sha256=(value.metadata.sha256,),
        deterministic_cache_key=runner._deterministic_key(value),
        palette_complete=True,
        fingerprint_complete=True,
    )
    owner = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    validate = runner._validated_deterministic_cache_entry

    def delayed(*args: object, **kwargs: object) -> object:
        assert threading.get_ident() != owner
        entered.set()
        assert release.wait(5)
        return validate(*args, **kwargs)

    monkeypatch.setattr(runner, "_validated_deterministic_cache_entry", delayed)
    with (
        CacheStore(tmp_path / "cache.sqlite3") as cache,
        TemporaryMediaRun(root=tmp_path) as temporary,
    ):
        runner._cache_deterministic_analysis(cache, source, value)

        async def scenario() -> None:
            task = asyncio.create_task(
                runner._restore_deterministic_cache_entry_async(
                    cache, source, MediaProcessor(temporary), checkpoint
                )
            )
            deadline = time.monotonic() + 5
            while not entered.is_set() and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.001)
            assert entered.is_set()
            # Reaching this point while validation waits proves network/UI tasks
            # can progress. A real SQLite connection also enforces owner affinity.
            release.set()
            result = await task
            assert result is not None
            assert result.metadata == value.metadata
            assert result.analysis == value.analysis

        try:
            asyncio.run(scenario())
        finally:
            release.set()


def test_pending_accesses_survive_prune_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100]
    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])
    path = tmp_path / "cache.sqlite3"
    with CacheStore(path) as cache:
        cache.put_metadata("recent", {"value": 1})
        cache.put_metadata("stale", {"value": 2})
        clock[0] = 200
        cache.get_metadata("recent")
        assert cache.prune(older_than_epoch=150)["metadata_removed"] == 1
        assert cache.get_metadata("recent") == {"value": 1}
        clock[0] = 300
        cache.get_metadata("recent")
    with CacheStore(path) as cache:
        assert cache.prune(older_than_epoch=250)["metadata_removed"] == 0


def test_access_buffer_has_bounded_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100]
    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        for index in range(256):
            cache.put_metadata(str(index), {"value": index})
        clock[0] = 200
        for index in range(256):
            cache.get_metadata(str(index))
        assert not cache._pending_touches
        assert (
            cache._connection.execute(
                "SELECT COUNT(*) FROM metadata_cache WHERE accessed_at = 200"
            ).fetchone()[0]
            == 256
        )

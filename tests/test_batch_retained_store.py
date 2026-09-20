"""Concurrent packs share retained-cache locking, scanning and byte accounting."""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.concurrency import batch_limits
from mojilex_cli.media import resume
from mojilex_cli.media.models import HARD_MAX_FRAMES, MediaLimitError, MediaMetadata, ProcessedMedia
from mojilex_cli.media.resume import RetainedMediaStore, get_retained_store
from mojilex_cli.media.temporary import TemporaryMediaRun


def media(tmp_path: Path) -> ProcessedMedia:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (256, 256), "white").save(frame)
    return ProcessedMedia(
        metadata=MediaMetadata(
            kind="static",
            format="webp",
            mime_type="image/webp",
            sha256="a" * 64,
            byte_size=100,
            width=512,
            height=512,
            animated=False,
        ),
        frame_paths=(frame,),
        rendered_frame_count=1,
        has_dark_render=False,
    )


async def test_retained_store_scans_and_charges_existing_files_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = media(tmp_path)
    root = tmp_path / "retained"
    original = RetainedMediaStore(root, 100_000)
    assert original.put("a" * 64, value)
    existing_bytes = original.size_bytes
    scans = 0
    measure = RetainedMediaStore._measure

    def count_scan(self: RetainedMediaStore) -> int:
        nonlocal scans
        scans += 1
        return measure(self)

    monkeypatch.setattr(RetainedMediaStore, "_measure", count_scan)
    with batch_limits(downloads=2, renders=2, ai=2, max_temp_bytes=100_000) as limits:
        with TemporaryMediaRun(root=tmp_path) as first, TemporaryMediaRun(root=tmp_path) as second:
            one = get_retained_store(root, max_bytes=100_000, run=first)
            two = get_retained_store(root, max_bytes=100_000, run=second)
            assert one is two
            assert scans == 1
            assert limits.temp_budget.used == existing_bytes
            first.cleanup()
            assert limits.temp_budget.used == existing_bytes
            results = await asyncio.gather(
                asyncio.to_thread(one.put, "b" * 64, value),
                asyncio.to_thread(two.put, "b" * 64, value),
            )
            assert results == [True, True]
            assert one.size_bytes == existing_bytes * 2
            assert limits.temp_budget.used == one.size_bytes
        assert limits.temp_budget.used == existing_bytes * 2
    # A new operation rescans once; no stale registry is reused.
    with batch_limits(downloads=2, renders=2, ai=2, max_temp_bytes=100_000) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            reopened = get_retained_store(root, max_bytes=100_000, run=run)
            assert reopened is not one
            assert scans == 2
            assert limits.temp_budget.used == existing_bytes * 2


def test_shared_retained_new_bytes_respect_aggregate_budget(tmp_path: Path) -> None:
    value = media(tmp_path)
    root = tmp_path / "retained"
    with batch_limits(downloads=2, renders=2, ai=2, max_temp_bytes=100) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            store = get_retained_store(root, max_bytes=100_000, run=run)
            with pytest.raises(MediaLimitError):
                store.put("a" * 64, value)
            assert limits.temp_budget.used == 0
            assert store.size_bytes == 0
            assert not list(root.iterdir())


def test_retained_byte_limit_failure_scans_once_without_double_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "retained"
    root.mkdir()
    (root / "unknown").write_bytes(b"12345")
    scans = []
    measure = RetainedMediaStore._measure

    def count_scan(self):
        scans.append(1)
        return measure(self)

    monkeypatch.setattr(RetainedMediaStore, "_measure", count_scan)
    with batch_limits(downloads=2, renders=2, ai=2, max_temp_bytes=10) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            for _ in range(2):
                with pytest.raises(MediaLimitError):
                    get_retained_store(root, max_bytes=4, run=run)
            assert limits.temp_budget.used == 0
            assert not limits.retained_stores
            assert scans == [1]


def test_retained_item_limit_allows_each_items_frames_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "_MAX_ITEMS", 3)
    monkeypatch.setattr(resume, "_MAX_ENTRIES", 3 * (2 * HARD_MAX_FRAMES + 3))
    total = 0
    for item in range(3):
        folder = tmp_path / str(item)
        folder.mkdir()
        for frame in range(2 * HARD_MAX_FRAMES + 2):
            (folder / str(frame)).write_bytes(b"12345")
            total += 5
    store = RetainedMediaStore(tmp_path, 1000)
    assert store.size_bytes == total
    assert store._available


def test_path_count_limit_is_reported_as_count_not_fake_disk_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "_MAX_ENTRIES", 2)
    for index in range(3):
        (tmp_path / str(index)).write_bytes(b"x")
    with pytest.raises(MediaLimitError, match="cache entry limit"):
        RetainedMediaStore(tmp_path, 10000)


@pytest.mark.asyncio
async def test_48_waiting_packs_do_not_rescan_failed_cache(tmp_path, monkeypatch):
    root = tmp_path / "retained"
    root.mkdir()
    for index in range(3):
        (root / str(index)).mkdir()
    monkeypatch.setattr(resume, "_MAX_ITEMS", 2)
    scans = []
    measure = RetainedMediaStore._measure

    def count_scan(self):
        scans.append(1)
        return measure(self)

    monkeypatch.setattr(RetainedMediaStore, "_measure", count_scan)
    with batch_limits(downloads=48, renders=12, ai=4, max_temp_bytes=10000) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(get_retained_store, root, max_bytes=10000, run=run)
                    for _ in range(48)
                ),
                return_exceptions=True,
            )
        assert all(isinstance(result, MediaLimitError) for result in results)
        assert all("cache item limit" in str(result) for result in results)
        assert scans == [1]
        assert limits.temp_budget.used == 0
        assert not limits.retained_stores
    # A fresh operation observes the repaired condition rather than a stale error.
    monkeypatch.setattr(resume, "_MAX_ITEMS", 3)
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=10000):
        with TemporaryMediaRun(root=tmp_path) as run:
            assert get_retained_store(root, max_bytes=10000, run=run)._available
    assert scans == [1, 1]


def test_retained_admission_retries_without_rescanning_when_space_frees(tmp_path, monkeypatch):
    root = tmp_path / "retained"
    root.mkdir()
    (root / "unknown").write_bytes(b"12345")
    scans = []
    measure = RetainedMediaStore._measure

    def count_scan(self):
        scans.append(1)
        return measure(self)

    monkeypatch.setattr(RetainedMediaStore, "_measure", count_scan)
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=10) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            limits.temp_budget.adjust(8)
            with pytest.raises(MediaLimitError, match="temporary disk limit"):
                get_retained_store(root, max_bytes=10, run=run)
            assert limits.temp_budget.used == 8
            assert not limits.retained_stores
            limits.temp_budget.adjust(-8)
            store = get_retained_store(root, max_bytes=10, run=run)
            assert store.size_bytes == limits.temp_budget.used == 5
            assert scans == [1]


def test_non_batch_factory_keeps_individual_run_accounting(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    root.mkdir()
    (root / "unknown").write_bytes(b"12345")
    with TemporaryMediaRun(root=tmp_path) as run:
        store = get_retained_store(root, max_bytes=1000, run=run)
        assert store.size_bytes == 5
        assert run._retained_bytes == 5

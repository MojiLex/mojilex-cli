"""Concurrent packs share retained-cache locking, scanning and byte accounting."""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.concurrency import batch_limits
from mojilex_cli.media.models import MediaLimitError, MediaMetadata, ProcessedMedia
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


def test_retained_limit_failure_is_not_cached_or_double_charged(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    root.mkdir()
    (root / "unknown").write_bytes(b"12345")
    with batch_limits(downloads=2, renders=2, ai=2, max_temp_bytes=10) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            for _ in range(2):
                with pytest.raises(MediaLimitError):
                    get_retained_store(root, max_bytes=4, run=run)
            assert limits.temp_budget.used == 0
            assert not limits.retained_stores


def test_non_batch_factory_keeps_individual_run_accounting(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    root.mkdir()
    (root / "unknown").write_bytes(b"12345")
    with TemporaryMediaRun(root=tmp_path) as run:
        store = get_retained_store(root, max_bytes=1000, run=run)
        assert store.size_bytes == 5
        assert run._retained_bytes == 5

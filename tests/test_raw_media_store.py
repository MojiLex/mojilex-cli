from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mojilex_cli.media.models import MediaLimitError
from mojilex_cli.media.raw_store import RawMediaStore


async def _chunks(payload: bytes):
    midpoint = max(1, len(payload) // 2)
    yield payload[:midpoint]
    yield payload[midpoint:]


async def test_raw_media_store_survives_reopen_and_streams_verified_bytes(tmp_path: Path) -> None:
    payload = b"retained telegram media"
    key = hashlib.sha256(b"source descriptor").hexdigest()
    digest = hashlib.sha256(payload).hexdigest()
    store = RawMediaStore(tmp_path / "raw", max_bytes=4096, max_file_bytes=1024)
    record = await store.put_stream(
        key,
        _chunks(payload),
        expected_size=len(payload),
        expected_sha256=digest,
    )
    assert record.sha256 == digest

    reopened = RawMediaStore(tmp_path / "raw", max_bytes=4096, max_file_bytes=1024)
    restored = reopened.get(key, expected_sha256=digest)
    assert restored is not None
    assert b"".join([chunk async for chunk in reopened.stream(key)]) == payload


async def test_raw_media_store_treats_modified_bytes_as_a_cache_miss(tmp_path: Path) -> None:
    payload = b"original"
    key = hashlib.sha256(b"descriptor").hexdigest()
    store = RawMediaStore(tmp_path / "raw", max_bytes=4096, max_file_bytes=1024)
    record = await store.put_stream(key, _chunks(payload))
    record.path.write_bytes(b"modified")
    assert store.get(key) is None
    with pytest.raises(ValueError, match="missing or corrupt"):
        _ = [chunk async for chunk in store.stream(key)]


async def test_raw_media_store_enforces_combined_limit_without_partial_entry(
    tmp_path: Path,
) -> None:
    key = hashlib.sha256(b"large descriptor").hexdigest()
    store = RawMediaStore(tmp_path / "raw", max_bytes=32, max_file_bytes=1024)
    with pytest.raises(MediaLimitError, match="disk limit"):
        await store.put_stream(key, _chunks(b"x" * 64))
    assert store.get(key) is None
    assert not list((tmp_path / "raw").iterdir())

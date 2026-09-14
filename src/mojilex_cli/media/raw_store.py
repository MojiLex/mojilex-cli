"""Private, bounded retention of original media for phased batch analysis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from mojilex_cli.concurrency import ByteBudgetExceeded, current_batch_limits

from .models import HARD_MAX_FILE_BYTES, MediaLimitError

_KEY = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_LIMIT = 4096


def _safe_path(path: Path) -> None:
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("retained source path contains a link or reparse point")


def _read(path: Path, limit: int) -> bytes:
    _safe_path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 1 <= info.st_size <= limit:
        raise ValueError("retained source is not a bounded private regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("retained source changed during open")
        data = stream.read(limit + 1)
    if len(data) != info.st_size:
        raise ValueError("retained source size changed")
    return data


@dataclass(frozen=True, slots=True)
class RawMediaRecord:
    path: Path
    sha256: str
    byte_size: int


def get_raw_store(root: Path, *, max_bytes: int, max_file_bytes: int) -> RawMediaStore:
    """Return one operation-scoped store and charge it to the shared disk budget."""

    batch = current_batch_limits()
    if batch is None:
        return RawMediaStore(root, max_bytes=max_bytes, max_file_bytes=max_file_bytes)
    key = "raw:" + os.path.normcase(os.path.abspath(root))
    with batch.retained_lock:
        existing = batch.retained_stores.get(key)
        if isinstance(existing, RawMediaStore):
            return existing

        def reserve(count: int) -> None:
            try:
                batch.temp_budget.adjust(count)
            except ByteBudgetExceeded as exc:
                raise MediaLimitError(str(exc)) from exc

        def release(count: int) -> None:
            batch.temp_budget.adjust(-count)

        store = RawMediaStore(
            root,
            max_bytes=max_bytes,
            max_file_bytes=max_file_bytes,
            reserve=reserve,
            release=release,
        )
        reserve(store.size_bytes)
        batch.retained_stores[key] = store
        return store


class RawMediaStore:
    """Content-verified original bytes that survive interruption of a saved run."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        max_file_bytes: int = HARD_MAX_FILE_BYTES,
        reserve: Callable[[int], None] | None = None,
        release: Callable[[int], None] | None = None,
    ) -> None:
        if not 1 <= max_file_bytes <= HARD_MAX_FILE_BYTES or max_bytes < 1:
            raise ValueError("invalid retained source limits")
        self.root = Path(os.path.abspath(root))
        self.max_bytes = max_bytes
        self.max_file_bytes = max_file_bytes
        self._reserve = reserve
        self._release = release
        self._lock = threading.RLock()
        self._pending = 0
        _safe_path(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _safe_path(self.root)
        self._size_bytes = self._measure()
        if self._size_bytes > max_bytes:
            raise MediaLimitError("run temporary disk limit exceeded")

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._size_bytes

    def _measure(self) -> int:
        total = 0
        for entry in self.root.iterdir():
            _safe_path(entry)
            if not entry.is_dir() or not _KEY.fullmatch(entry.name):
                raise ValueError("unexpected retained source entry")
            for name in ("source.bin", "manifest.json"):
                path = entry / name
                _safe_path(path)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("unexpected retained source entry")
                total += info.st_size
        return total

    def get(self, key: str, *, expected_sha256: str | None = None) -> RawMediaRecord | None:
        if not _KEY.fullmatch(key):
            return None
        entry = self.root / key
        try:
            manifest = json.loads(_read(entry / "manifest.json", _MANIFEST_LIMIT))
            if (
                not isinstance(manifest, dict)
                or set(manifest) != {"version", "key", "sha256", "bytes"}
                or manifest["version"] != 1
                or manifest["key"] != key
                or not isinstance(manifest["sha256"], str)
                or not _KEY.fullmatch(manifest["sha256"])
                or type(manifest["bytes"]) is not int
                or not 1 <= manifest["bytes"] <= self.max_file_bytes
            ):
                return None
            data = _read(entry / "source.bin", self.max_file_bytes)
            digest = hashlib.sha256(data).hexdigest()
            if (
                len(data) != manifest["bytes"]
                or digest != manifest["sha256"]
                or (expected_sha256 is not None and digest != expected_sha256)
            ):
                return None
            return RawMediaRecord(entry / "source.bin", digest, len(data))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    async def put_stream(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> RawMediaRecord:
        if not _KEY.fullmatch(key):
            raise ValueError("retained source key must be a SHA-256 digest")
        existing = await asyncio.to_thread(self.get, key, expected_sha256=expected_sha256)
        if existing is not None:
            return existing
        batch = current_batch_limits()
        if batch is not None:
            async with batch.download_slots:
                return await self._put_stream(
                    key, chunks, expected_size=expected_size, expected_sha256=expected_sha256
                )
        return await self._put_stream(
            key, chunks, expected_size=expected_size, expected_sha256=expected_sha256
        )

    async def _put_stream(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        expected_size: int | None,
        expected_sha256: str | None,
    ) -> RawMediaRecord:
        if expected_size is not None and expected_size > self.max_file_bytes:
            raise MediaLimitError("declared media size exceeds 20 MiB")
        staged = Path(tempfile.mkdtemp(prefix=f".{key}.pending-", dir=self.root.parent))
        source = staged / "source.bin"
        digest = hashlib.sha256()
        size = 0
        charged = 0
        committed = False
        try:
            with source.open("xb") as stream:
                async for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("media stream yielded a non-bytes value")
                    if size + len(chunk) > self.max_file_bytes:
                        raise MediaLimitError("media download exceeds 20 MiB")
                    self._charge(len(chunk))
                    charged += len(chunk)
                    size += len(chunk)
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if size == 0:
                raise MediaLimitError("media download is empty")
            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise ValueError("downloaded media no longer matches the saved run")
            manifest = json.dumps(
                {"version": 1, "key": key, "sha256": actual, "bytes": size},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            self._charge(len(manifest))
            charged += len(manifest)
            with (staged / "manifest.json").open("xb") as stream:
                stream.write(manifest)
                stream.flush()
                os.fsync(stream.fileno())
            record, committed = await asyncio.to_thread(
                self._commit_staged,
                key,
                staged,
                charged,
                actual,
                size,
                expected_sha256,
            )
            return record
        finally:
            if not committed:
                try:
                    await asyncio.to_thread(self._cleanup_staged, staged)
                finally:
                    self._uncharge(charged)

    def _commit_staged(
        self,
        key: str,
        staged: Path,
        charged: int,
        sha256: str,
        size: int,
        expected_sha256: str | None,
    ) -> tuple[RawMediaRecord, bool]:
        destination = self.root / key
        with self._lock:
            if destination.exists():
                record = self.get(key, expected_sha256=expected_sha256)
                if record is None:
                    raise ValueError("conflicting retained source entry")
                return record, False
            staged.rename(destination)
            self._pending -= charged
            self._size_bytes += charged
            return RawMediaRecord(destination / "source.bin", sha256, size), True

    @staticmethod
    def _cleanup_staged(staged: Path) -> None:
        for path in (staged / "source.bin", staged / "manifest.json"):
            _safe_path(path)
            path.unlink(missing_ok=True)
        staged.rmdir()

    def _charge(self, count: int) -> None:
        with self._lock:
            if self._size_bytes + self._pending + count > self.max_bytes:
                raise MediaLimitError("run temporary disk limit exceeded")
            if self._reserve is not None:
                self._reserve(count)
            self._pending += count

    def _uncharge(self, count: int) -> None:
        if not count:
            return
        with self._lock:
            self._pending -= count
            if self._release is not None:
                self._release(count)

    async def stream(self, key: str, *, expected_sha256: str | None = None) -> AsyncIterator[bytes]:
        record = await asyncio.to_thread(self.get, key, expected_sha256=expected_sha256)
        if record is None:
            raise ValueError("retained source is missing or corrupt")
        data = await asyncio.to_thread(_read, record.path, self.max_file_bytes)
        if len(data) != record.byte_size or hashlib.sha256(data).hexdigest() != record.sha256:
            raise ValueError("retained source changed while being read")
        for offset in range(0, len(data), 256 * 1024):
            yield data[offset : offset + 256 * 1024]
            await asyncio.sleep(0)

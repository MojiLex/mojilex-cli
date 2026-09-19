"""Bounded private PNG retention for interrupted runs, outside dataset artifacts.

The caller owns the run lock. Unknown entries are never replaced or removed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import struct
import tempfile
import threading
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from mojilex_cli.concurrency import ByteBudgetExceeded, current_batch_limits

from .models import (
    HARD_MAX_FILE_BYTES,
    HARD_MAX_FRAMES,
    PIPELINE_VERSION,
    MediaLimitError,
    ProcessedMedia,
)
from .temporary import TemporaryMediaRun

_KEY = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_LIMIT = 32 * 1024
_MAX_ENTRIES = 100_000


def get_retained_store(root: Path, *, max_bytes: int, run: TemporaryMediaRun) -> RetainedMediaStore:
    """Reuse one retained store and one disk reservation across concurrent packs.

    Retained frames outlive temporary pack directories: shared reservations belong
    to the whole operation and are never released by the first pack's cleanup.
    """
    batch = current_batch_limits()
    if batch is None:
        store = RetainedMediaStore(
            root,
            max_bytes,
            reserve=run.reserve_retained_bytes,
            release=run.release_retained_bytes,
        )
        run.reserve_retained_bytes(store.size_bytes)
        return store
    key = os.path.normcase(os.path.abspath(root))
    with batch.retained_lock:
        build_lock = batch.retained_build_locks.setdefault(key, threading.Lock())
    # Different saved runs may inspect their own directories concurrently; only
    # builders of the same store must wait for each other.
    with build_lock:
        existing = batch.retained_stores.get(key)
        if isinstance(existing, RetainedMediaStore):
            if existing.max_bytes != max_bytes:
                raise ValueError("shared retained media limit changed during the operation")
            return existing

        def reserve(count: int) -> None:
            try:
                batch.temp_budget.adjust(count)
            except ByteBudgetExceeded as exc:
                raise MediaLimitError(str(exc)) from exc

        def release(count: int) -> None:
            batch.temp_budget.adjust(-count)

        store = RetainedMediaStore(root, max_bytes, reserve=reserve, release=release)
        if store.size_bytes > max_bytes:
            raise MediaLimitError("run temporary disk limit exceeded")
        reserve(store.size_bytes)
        batch.retained_stores[key] = store
        return store


def _safe_path(path: Path) -> None:
    """Reject links, including Windows junctions, in every existing component."""
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("retained media path contains a link or reparse point")


def _read(path: Path, limit: int) -> bytes:
    _safe_path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        raise ValueError("retained media file is not a bounded private regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("retained media file changed during open")
        data = stream.read(limit + 1)
    if len(data) > limit or len(data) != info.st_size:
        raise ValueError("retained media file size changed")
    return data


def _png(data: bytes) -> None:
    with Image.open(io.BytesIO(data)) as image:
        if (
            image.format != "PNG"
            or image.size != (256, 256)
            or getattr(image, "is_animated", False)
        ):
            raise ValueError("retained media must be a generated 256-pixel PNG frame")
        image.verify()


def _fingerprint(value: ProcessedMedia) -> str:
    safe = {
        "pipeline": PIPELINE_VERSION,
        "media": value.model_dump(mode="json"),
        "count": value.semantic_frame_count,
        "dark": value.semantic_has_dark_render,
    }
    return hashlib.sha256(
        json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _composition_png(data: bytes, size: tuple[int, int]) -> None:
    """Accept only small unscaled RGBA tiles produced by the isolated decoder."""
    if len(data) > 512 * 1024 or not all(1 <= side <= 256 for side in size):
        raise ValueError("composition tile exceeds its safe limit")
    with Image.open(io.BytesIO(data)) as image:
        if (
            image.format != "PNG"
            or image.size != size
            or image.mode != "RGBA"
            or getattr(image, "is_animated", False)
        ):
            raise ValueError("composition tile must be a native-size static RGBA PNG")
        image.verify()


class RetainedMediaStore:
    """Retain only verified generated frames; corruption is an ordinary cache miss.

    ``reserve`` charges new bytes before writing; ``release`` rolls that charge
    back if writing fails. Successful charges remain until the caller's run ends.
    Existing bytes are exposed separately through ``size_bytes``.
    """

    def __init__(
        self,
        root: Path,
        max_bytes: int,
        *,
        reserve: Callable[[int], None] | None = None,
        release: Callable[[int], None] | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("retained media limit must be positive")
        self.root = Path(os.path.abspath(root))
        self.max_bytes = max_bytes
        self._reserve = reserve
        self._release = release
        self._lock = threading.RLock()
        self._size_bytes = 0
        self._available = False
        try:
            _safe_path(self.root)
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            _safe_path(self.root)
            self._size_bytes = self._measure()
            self._available = True
        except (OSError, ValueError):
            pass

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._size_bytes

    def _measure(self) -> int:
        total = 0
        seen = 0
        pending = [(self.root, 0)]
        while pending:
            directory, depth = pending.pop()
            _safe_path(directory)
            # Windows scandir supplies entry metadata without restatting every
            # ancestor for every frame. Actual reads still validate the full path.
            with os.scandir(directory) as entries:
                for child in entries:
                    seen += 1
                    if seen > _MAX_ENTRIES:
                        return self.max_bytes + 1
                    info = child.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                        raise ValueError("retained media path contains a link or reparse point")
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
                    elif stat.S_ISDIR(info.st_mode) and depth < 2:
                        pending.append((Path(child.path), depth + 1))
                    else:
                        raise ValueError("unexpected retained media entry")
        return total

    def get(self, key: str, expected: ProcessedMedia) -> ProcessedMedia | None:
        with self._lock:
            return self._get(key, expected)

    def get_composition_tile(self, key: str, expected: ProcessedMedia) -> ProcessedMedia | None:
        """Restore only a verified static tile for an independently reusable AI result.

        This does not prove that semantic frames are available. Returned media
        retains its render context but has no frame paths; callers needing AI
        input images must use ``get`` instead. A missing tile is a cache miss.
        """
        if expected.metadata.kind != "static":
            return None
        with self._lock:
            return self._get(key, expected, composition_only=True)

    def _get(
        self, key: str, expected: ProcessedMedia, *, composition_only: bool = False
    ) -> ProcessedMedia | None:
        if not self._available or not _KEY.fullmatch(key):
            return None
        count = expected.semantic_frame_count
        if not 1 <= count <= HARD_MAX_FRAMES:
            return None
        entry = self.root / key
        try:
            manifest = json.loads(_read(entry / "manifest.json", _MANIFEST_LIMIT))
            if (
                not isinstance(manifest, dict)
                or set(manifest)
                not in (
                    {"version", "fingerprint", "frames"},
                    {"version", "fingerprint", "frames", "composition_tile"},
                )
                or type(manifest["version"]) is not int
                or manifest["version"] != 1
                or manifest["fingerprint"] != _fingerprint(expected)
            ):
                return None
            names = [f"light-{i:02d}.png" for i in range(count)]
            if expected.semantic_has_dark_render:
                names += [f"dark-{i:02d}.png" for i in range(count)]
            frames = manifest["frames"]
            if not isinstance(frames, list) or len(frames) != len(names):
                return None
            paths = []
            for name, record in zip(names, frames, strict=True):
                if (
                    not isinstance(record, dict)
                    or set(record) != {"name", "bytes", "sha256"}
                    or record["name"] != name
                    or type(record["bytes"]) is not int
                    or not 1 <= record["bytes"] <= HARD_MAX_FILE_BYTES
                ):
                    return None
                if composition_only:
                    continue
                path = entry / name
                data = _read(path, record["bytes"])
                if (
                    len(data) != record["bytes"]
                    or hashlib.sha256(data).hexdigest() != record["sha256"]
                ):
                    return None
                _png(data)
                paths.append(path)
            tile_path = None
            tile_sha256 = None
            tile = manifest.get("composition_tile")
            if composition_only and tile is None:
                return None
            if tile is not None:
                if (
                    expected.metadata.kind != "static"
                    or not isinstance(tile, dict)
                    or set(tile) != {"name", "bytes", "sha256"}
                    or tile["name"] != "composition-tile.png"
                    or type(tile["bytes"]) is not int
                    or not 1 <= tile["bytes"] <= 512 * 1024
                ):
                    return None
                tile_path = entry / "composition-tile.png"
                data = _read(tile_path, tile["bytes"])
                tile_sha256 = hashlib.sha256(data).hexdigest()
                if len(data) != tile["bytes"] or tile_sha256 != tile["sha256"]:
                    return None
                _composition_png(data, (expected.metadata.width, expected.metadata.height))
            return ProcessedMedia(
                metadata=expected.metadata,
                analysis=expected.analysis,
                frame_paths=tuple(paths[:count]),
                dark_frame_paths=tuple(paths[count:]),
                rendered_frame_count=count,
                has_dark_render=expected.semantic_has_dark_render,
                composition_tile_path=tile_path,
                composition_tile_sha256=tile_sha256,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            SyntaxError,
            struct.error,
            zlib.error,
            Image.DecompressionBombError,
        ):
            return None

    def put(self, key: str, value: ProcessedMedia) -> bool:
        with self._lock:
            return self._put(key, value)

    def _put(self, key: str, value: ProcessedMedia) -> bool:
        if not self._available or not _KEY.fullmatch(key):
            return False
        count = value.semantic_frame_count
        if (
            not 1 <= count <= HARD_MAX_FRAMES
            or len(value.frame_paths) != count
            or len(value.dark_frame_paths) != (count if value.semantic_has_dark_render else 0)
        ):
            return False
        if self.get(key, value) is not None:
            return True
        entry = self.root / key
        # An invalid or unrelated existing entry must never be overwritten.
        if os.path.lexists(entry):
            return False
        staged: Path | None = None
        created: list[Path] = []
        charged = 0
        committed = False
        try:
            records: list[dict[str, Any]] = []
            sources: list[Path] = []
            for role, paths in (("light", value.frame_paths), ("dark", value.dark_frame_paths)):
                for index, path in enumerate(paths):
                    data = _read(path, HARD_MAX_FILE_BYTES)
                    _png(data)
                    records.append(
                        {
                            "name": f"{role}-{index:02d}.png",
                            "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
                    sources.append(path)
            manifest_data: dict[str, Any] = {
                "version": 1,
                "fingerprint": _fingerprint(value),
                "frames": list(records),
            }
            if value.composition_tile_path is not None:
                data = _read(value.composition_tile_path, 512 * 1024)
                _composition_png(data, (value.metadata.width, value.metadata.height))
                digest = hashlib.sha256(data).hexdigest()
                if digest != value.composition_tile_sha256:
                    return False
                tile_record = {"name": "composition-tile.png", "bytes": len(data), "sha256": digest}
                manifest_data["composition_tile"] = tile_record
                records.append(tile_record)
                sources.append(value.composition_tile_path)
            manifest = json.dumps(
                manifest_data,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            total = len(manifest) + sum(record["bytes"] for record in records)
            if len(manifest) > _MANIFEST_LIMIT or self._size_bytes + total > self.max_bytes:
                return False
            if self._reserve is not None:
                self._reserve(total)
            charged = total
            _safe_path(self.root)
            staged = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
            for source, record in zip(sources, records, strict=True):
                data = _read(source, record["bytes"])
                if hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("generated frame changed before retention")
                destination = staged / record["name"]
                created.append(destination)
                self._write(destination, data)
            destination = staged / "manifest.json"
            created.append(destination)
            self._write(destination, manifest)
            _safe_path(self.root)
            if os.path.lexists(entry):
                return False
            staged.rename(entry)
            committed = True
            self._size_bytes += total
            return True
        except (
            OSError,
            ValueError,
            TypeError,
            RecursionError,
            SyntaxError,
            struct.error,
            zlib.error,
            Image.DecompressionBombError,
        ):
            return False
        finally:
            if not committed:
                if staged is not None:
                    try:
                        _safe_path(staged)
                        for path in created:
                            _safe_path(path)
                            path.unlink(missing_ok=True)
                        staged.rmdir()
                    except (OSError, ValueError):
                        # Leave unknown/interfered-with files untouched and stop adding data.
                        self._available = False
                        self._size_bytes += charged
                        charged = 0
                if charged and self._release is not None:
                    self._release(charged)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        _safe_path(path)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

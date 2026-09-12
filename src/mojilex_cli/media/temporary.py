"""Private per-run temporary storage with streaming byte accounting."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from .models import HARD_MAX_FILE_BYTES, MediaLimitError, MediaLimits


class TemporaryMediaRun:
    def __init__(self, *, root: Path | None = None, limits: MediaLimits | None = None) -> None:
        self.limits = limits or MediaLimits()
        self._requested_root = root
        self.path: Path | None = None
        self.bytes_written = 0
        self._accounted: dict[Path, int] = {}
        self._reserved_bytes = 0
        self._retained_bytes = 0
        self._account_lock = threading.RLock()

    def __enter__(self) -> TemporaryMediaRun:
        base = None
        if self._requested_root is not None:
            base = str(self._requested_root.resolve(strict=True))
        self.path = Path(tempfile.mkdtemp(prefix="mojilex-media-", dir=base)).resolve()
        self._accounted.clear()
        self.bytes_written = 0
        self._reserved_bytes = 0
        self._retained_bytes = 0
        try:
            os.chmod(self.path, 0o700)
        except OSError:
            pass
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self.path is not None and self.path.exists():
            shutil.rmtree(self.path)
        self.path = None
        self._accounted.clear()
        self.bytes_written = 0
        self._reserved_bytes = 0
        self._retained_bytes = 0

    def reserve_retained_bytes(self, count: int) -> None:
        """Charge resumable frames against the same run disk budget."""
        if count < 0:
            raise ValueError("retained byte reservation must be non-negative")
        with self._account_lock:
            if (
                self.bytes_written + self._reserved_bytes + self._retained_bytes + count
                > self.limits.max_run_temp_bytes
            ):
                raise MediaLimitError("run temporary disk limit exceeded")
            self._retained_bytes += count

    def release_retained_bytes(self, count: int) -> None:
        with self._account_lock:
            if not 0 <= count <= self._retained_bytes:
                raise ValueError("invalid retained byte release")
            self._retained_bytes -= count

    async def write_stream(
        self, chunks: AsyncIterator[bytes], *, expected_size: int | None = None
    ) -> tuple[Path, str, int]:
        if self.path is None:
            raise RuntimeError("temporary media run is not active")
        if expected_size is not None and expected_size > self.limits.max_file_bytes:
            raise MediaLimitError("declared media size exceeds 20 MiB")
        destination = self.path / f"source-{uuid.uuid4().hex}.bin"
        partial = destination.with_suffix(".partial")
        digest = hashlib.sha256()
        item_size = 0
        reserved_size = 0
        try:
            with partial.open("xb") as stream:
                async for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("media stream yielded a non-bytes value")
                    next_item_size = item_size + len(chunk)
                    if next_item_size > min(self.limits.max_file_bytes, HARD_MAX_FILE_BYTES):
                        raise MediaLimitError("media download exceeds 20 MiB")
                    with self._account_lock:
                        if (
                            self.bytes_written
                            + self._reserved_bytes
                            + self._retained_bytes
                            + len(chunk)
                            > self.limits.max_run_temp_bytes
                        ):
                            raise MediaLimitError("run temporary disk limit exceeded")
                        self._reserved_bytes += len(chunk)
                        reserved_size += len(chunk)
                    item_size = next_item_size
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if item_size == 0:
                raise MediaLimitError("media download is empty")
            os.replace(partial, destination)
            with self._account_lock:
                self._reserved_bytes -= reserved_size
                reserved_size = 0
                self.bytes_written += item_size
                self._accounted[destination.resolve(strict=True)] = item_size
            return destination, digest.hexdigest(), item_size
        except BaseException:
            with self._account_lock:
                self._reserved_bytes -= reserved_size
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise

    def account_outputs(self, paths: Iterable[Path]) -> int:
        """Account generated files below this run root without following symlinks.

        Callers should invoke this immediately after an external renderer or contact-sheet
        builder returns. Re-accounting a file is safe and charges only its size delta.
        """

        if self.path is None:
            raise RuntimeError("temporary media run is not active")
        root = self.path.resolve(strict=True)
        observed: dict[Path, int] = {}

        def inspect(path: Path) -> None:
            if path.is_symlink():
                raise MediaLimitError("temporary output must not contain symlinks")
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise MediaLimitError("temporary output escaped the private run directory") from exc
            if resolved.is_file():
                observed[resolved] = resolved.stat().st_size
                return
            if not resolved.is_dir():
                raise MediaLimitError("temporary output is not a regular file or directory")
            for child in resolved.iterdir():
                inspect(child)

        for value in paths:
            inspect(Path(value))
        with self._account_lock:
            delta = sum(size - self._accounted.get(path, 0) for path, size in observed.items())
            if (
                self.bytes_written + self._reserved_bytes + self._retained_bytes + delta
                > self.limits.max_run_temp_bytes
            ):
                raise MediaLimitError("run temporary disk limit exceeded")
            self.bytes_written += delta
            self._accounted.update(observed)
            return self.bytes_written

    def output_dir(self) -> Path:
        if self.path is None:
            raise RuntimeError("temporary media run is not active")
        return self.path / f"render-{uuid.uuid4().hex}"

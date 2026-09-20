"""Crash-recoverable transactions for bounded dataset file replacements."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from filelock import BaseFileLock, FileLock, Timeout

from .layout import assert_no_link_or_reparse, is_link_or_reparse_point, safe_destination

TRANSACTION_DIRECTORY_NAME = ".mojilex-atomic-write"

_TRANSACTION_KIND = "mojilex-dataset-transaction"
_FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_MANIFEST_TEMP_NAME = ".manifest.tmp"
_PREPARING_NAME = "preparing.json"
_PREPARING_TEMP_NAME = ".preparing.tmp"
_STATE_NAME = "state"
_STATE_TEMP_NAME = ".state.tmp"
_BACKUP_DIRECTORY_NAME = "backups"
_STAGED_DIRECTORY_NAME = "staged"
_READY_STATE = b"ready\n"
_COMMITTED_STATE = b"committed\n"
_MAX_CHANGES = 10_000
_MAX_TOTAL_PAYLOAD_BYTES = 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_PREPARING_BYTES = 1024
_MAX_RELATIVE_PATH_BYTES = 4096
_LOCK_TIMEOUT_SECONDS = 2.0
_LOCAL_STATE_DIRECTORY_NAME = ".mojilex"
_LOCK_DIRECTORY_NAME = "locks"
_LOCK_FILE_NAME = "dataset-transaction-v1.lock"
_LOCK_FILE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,126}\.lock\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DEVICE_BASENAME = re.compile(
    r"(?:con|prn|aux|nul|clock\$|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])\Z",
    re.IGNORECASE,
)
_WINDOWS_SHORT_NAME = re.compile(r".+~[0-9]+(?:\..*)?\Z", re.IGNORECASE)


class AtomicWriteError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TransactionEntry:
    index: int
    relative: PurePosixPath
    old_sha256: str | None
    new_sha256: str | None
    old_size: int | None
    new_size: int | None


@dataclass(frozen=True, slots=True)
class _TargetState:
    entry: TransactionEntry
    current_sha256: str | None


class DurableDatasetTransaction:
    """One prepared transaction whose journal is durable before target mutation."""

    def __init__(
        self,
        root: Path,
        entries: Sequence[TransactionEntry],
        lock: BaseFileLock,
    ) -> None:
        self.root = root.resolve()
        self.directory = self.root / TRANSACTION_DIRECTORY_NAME
        self.entries = tuple(entries)
        self._entry_membership = frozenset(self.entries)
        self._lock: BaseFileLock | None = lock

    @classmethod
    def prepare(
        cls,
        root: Path,
        changes: Mapping[PurePosixPath, bytes | None],
        *,
        expected_files: Mapping[PurePosixPath, bytes] | None = None,
        expected_root_files: Mapping[PurePosixPath, tuple[int, str]] | None = None,
        expected_tree_files: Mapping[PurePosixPath, tuple[int, str]] | None = None,
        lock_root: str | Path | None = None,
        lock_name: str | None = None,
    ) -> DurableDatasetTransaction:
        resolved = _resolve_dataset_root(root)
        resolved.mkdir(parents=True, exist_ok=True)
        lock = _acquire_transaction_lock(
            _resolve_dataset_root(lock_root) if lock_root is not None else resolved,
            lock_name=lock_name,
        )
        directory = resolved / TRANSACTION_DIRECTORY_NAME
        created_directory = False
        try:
            _recover_pending_dataset_transaction_unlocked(resolved)
            if expected_files is not None:
                _verify_dataset_precondition(resolved, expected_files)
            if expected_root_files is not None:
                _verify_root_file_precondition(resolved, expected_root_files)
            if expected_tree_files is not None:
                _verify_tree_file_precondition(resolved, expected_tree_files)
            ordered = tuple(sorted(changes.items(), key=lambda item: str(item[0])))
            if not ordered:
                raise ValueError("dataset transaction requires at least one change")
            if len(ordered) > _MAX_CHANGES:
                raise AtomicWriteError(
                    f"dataset transaction exceeds the {_MAX_CHANGES}-path safety limit"
                )
            change_identities: set[str] = set()
            for relative, _replacement in ordered:
                identity = transaction_path_identity(resolved, relative)
                if identity in change_identities:
                    raise AtomicWriteError(
                        "dataset transaction contains paths that alias the same target"
                    )
                change_identities.add(identity)
            try:
                directory.mkdir()
            except FileExistsError as exc:
                raise AtomicWriteError(
                    "pending dataset transaction could not be recovered safely"
                ) from exc
            created_directory = True
            _fsync_directory(resolved)
            _replace_control_file(
                directory,
                _PREPARING_TEMP_NAME,
                _PREPARING_NAME,
                _preparing_bytes(resolved),
            )
            backups = directory / _BACKUP_DIRECTORY_NAME
            staged = directory / _STAGED_DIRECTORY_NAME
            backups.mkdir()
            staged.mkdir()
            _fsync_directory(directory)
            entries: list[TransactionEntry] = []
            total_payload_bytes = 0
            for index, (relative, replacement) in enumerate(ordered):
                destination = safe_destination(resolved, relative)
                original_size = destination.stat().st_size if destination.exists() else 0
                replacement_size = len(replacement) if replacement is not None else 0
                if (
                    total_payload_bytes + original_size + replacement_size
                    > _MAX_TOTAL_PAYLOAD_BYTES
                ):
                    raise AtomicWriteError(
                        "dataset transaction exceeds the 1 GiB payload safety limit"
                    )
                original = destination.read_bytes() if destination.exists() else None
                total_payload_bytes += original_size + replacement_size
                entry = TransactionEntry(
                    index=index,
                    relative=relative,
                    old_sha256=_digest(original),
                    new_sha256=_digest(replacement),
                    old_size=len(original) if original is not None else None,
                    new_size=len(replacement) if replacement is not None else None,
                )
                entries.append(entry)
                if original is not None:
                    path = _backup_path(directory, index)
                    _assert_safe_internal_path(directory, path)
                    _write_durable_file(path, original)
                if replacement is not None:
                    path = _staged_path(directory, index)
                    _assert_safe_internal_path(directory, path)
                    _write_durable_file(path, replacement)
            _fsync_directory(backups)
            _fsync_directory(staged)
            transaction = cls(resolved, entries, lock)
            manifest = transaction._manifest_bytes()
            if len(manifest) > _MAX_MANIFEST_BYTES:
                raise AtomicWriteError("dataset transaction manifest exceeds 4 MiB")
            transaction._replace_control_file(
                _MANIFEST_TEMP_NAME,
                _MANIFEST_NAME,
                manifest,
            )
            transaction._write_state(_READY_STATE)
            return transaction
        except BaseException:
            try:
                if created_directory:
                    _recover_pending_dataset_transaction_unlocked(resolved)
            finally:
                lock.release()
            raise

    def staged_path(self, entry: TransactionEntry) -> Path:
        self._assert_locked()
        if entry not in self._entry_membership or entry.new_sha256 is None:
            raise AtomicWriteError("invalid staged transaction entry")
        path = _staged_path(self.directory, entry.index)
        _validate_payload_file(path, entry.new_sha256, entry.new_size)
        return path

    def mark_committed(self) -> None:
        self._assert_locked()
        self._write_state(_COMMITTED_STATE)

    def sync_target_parent(self, entry: TransactionEntry) -> None:
        self._assert_locked()
        if entry not in self._entry_membership:
            raise AtomicWriteError("invalid dataset transaction entry")
        _fsync_directory(safe_destination(self.root, entry.relative).parent)

    def rollback(self) -> None:
        """Force rollback of this in-process transaction, regardless of its marker."""

        self._assert_locked()
        try:
            _rollback_entries(self.root, self.directory, self.entries)
            _cleanup_transaction(self.root, self.directory, self.entries)
        finally:
            self._release_lock()

    def cleanup(self) -> None:
        self._assert_locked()
        try:
            _cleanup_transaction(self.root, self.directory, self.entries)
        finally:
            self._release_lock()

    def _assert_locked(self) -> None:
        if self._lock is None or not self._lock.is_locked:
            raise AtomicWriteError("dataset transaction lifetime lock is not held")

    def _release_lock(self) -> None:
        lock = self._lock
        if lock is not None:
            self._lock = None
            lock.release()

    def _manifest_bytes(self) -> bytes:
        payload = {
            "entries": [
                {
                    "new_sha256": entry.new_sha256,
                    "new_size": entry.new_size,
                    "old_sha256": entry.old_sha256,
                    "old_size": entry.old_size,
                    "path": entry.relative.as_posix(),
                }
                for entry in self.entries
            ],
            "format_version": _FORMAT_VERSION,
            "kind": _TRANSACTION_KIND,
            "root_sha256": _root_sha256(self.root),
        }
        return (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

    def _write_state(self, value: bytes) -> None:
        if value not in {_READY_STATE, _COMMITTED_STATE}:
            raise ValueError("invalid dataset transaction state")
        self._replace_control_file(_STATE_TEMP_NAME, _STATE_NAME, value)

    def _replace_control_file(self, temporary_name: str, name: str, data: bytes) -> None:
        _replace_control_file(self.directory, temporary_name, name, data)


def recover_pending_dataset_transaction(
    root: str | Path,
    *,
    lock_root: str | Path | None = None,
    lock_name: str | None = None,
) -> bool:
    """Recover one exact internal transaction before dataset files are read or written."""

    resolved = _resolve_dataset_root(root)
    lock = _acquire_transaction_lock(
        _resolve_dataset_root(lock_root) if lock_root is not None else resolved,
        lock_name=lock_name,
    )
    try:
        return _recover_pending_dataset_transaction_unlocked(resolved)
    finally:
        lock.release()


@contextmanager
def locked_dataset_transaction_view(root: str | Path) -> Iterator[None]:
    """Recover and hold the writer lock while a dataset snapshot is parsed."""

    resolved = _resolve_dataset_root(root)
    lock = _acquire_transaction_lock(resolved)
    try:
        _recover_pending_dataset_transaction_unlocked(resolved)
        yield
    finally:
        lock.release()


def _recover_pending_dataset_transaction_unlocked(resolved: Path) -> bool:
    directory = resolved / TRANSACTION_DIRECTORY_NAME
    if not directory.exists():
        return False
    try:
        assert_no_link_or_reparse(directory, boundary=resolved)
    except ValueError as exc:
        raise AtomicWriteError("dataset transaction path contains a link or reparse point") from exc
    if not directory.is_dir():
        raise AtomicWriteError("dataset transaction path is not a safe directory")

    manifest_path = directory / _MANIFEST_NAME
    state_path = directory / _STATE_NAME
    if not manifest_path.exists():
        if state_path.exists():
            _cleanup_manifestless_finalization(resolved, directory)
        else:
            _cleanup_incomplete_preparation(resolved, directory)
        return True
    entries = _load_manifest(resolved, directory)
    state = _load_state(state_path)
    _validate_internal_contents(directory, entries)

    if state == _READY_STATE:
        _rollback_entries(resolved, directory, entries)
    elif state == _COMMITTED_STATE:
        states = _inspect_target_states(resolved, entries)
        if any(value.current_sha256 != value.entry.new_sha256 for value in states):
            raise AtomicWriteError(
                "committed dataset transaction no longer matches recorded target bytes"
            )
    elif state is None:
        states = _inspect_target_states(resolved, entries)
        targets_are_old = all(value.current_sha256 == value.entry.old_sha256 for value in states)
        if targets_are_old:
            _validate_complete_prepared_payloads(directory, entries)
        _restore_ready_state_for_cleanup(directory)
        if not targets_are_old:
            _rollback_entries(resolved, directory, entries)
    else:
        raise AtomicWriteError("dataset transaction state marker is corrupt")

    _cleanup_transaction(resolved, directory, entries)
    return True


def _acquire_transaction_lock(
    root: Path,
    *,
    lock_name: str | None = None,
) -> BaseFileLock:
    path = _transaction_lock_path(root, lock_name=lock_name)
    lock = FileLock(str(path), fallback_to_soft=False)
    try:
        lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS)
    except Timeout as exc:
        raise AtomicWriteError(
            "dataset is locked by another active atomic writer; retry after it finishes"
        ) from exc
    except OSError as exc:
        raise AtomicWriteError("dataset transaction lock could not be acquired safely") from exc
    return lock


def _transaction_lock_path(root: Path, *, lock_name: str | None = None) -> Path:
    resolved_root = _resolve_dataset_root(root)
    selected_lock_name = lock_name or _LOCK_FILE_NAME
    if not _LOCK_FILE_RE.fullmatch(selected_lock_name):
        raise AtomicWriteError("dataset transaction lock name is unsafe")
    state_directory = resolved_root / _LOCAL_STATE_DIRECTORY_NAME
    lock_directory = state_directory / _LOCK_DIRECTORY_NAME
    try:
        if is_link_or_reparse_point(state_directory):
            raise ValueError("local state directory is a link or reparse point")
        if not state_directory.exists():
            state_directory.mkdir()
            _fsync_directory(resolved_root)
        assert_no_link_or_reparse(state_directory, boundary=resolved_root)
        if not state_directory.is_dir():
            raise ValueError("local state path is not a directory")
        if is_link_or_reparse_point(lock_directory):
            raise ValueError("lock directory is a link or reparse point")
        if not lock_directory.exists():
            lock_directory.mkdir()
            _fsync_directory(state_directory)
        assert_no_link_or_reparse(lock_directory, boundary=resolved_root)
    except (OSError, ValueError) as exc:
        raise AtomicWriteError(
            "the deterministic dataset transaction lock directory is unavailable"
        ) from exc
    if not lock_directory.is_dir():
        raise AtomicWriteError("dataset transaction lock directory is unsafe")
    path = lock_directory / selected_lock_name
    if path.parent != lock_directory or path.name != selected_lock_name:
        raise AtomicWriteError("dataset transaction lock is not bound to the dataset root")
    try:
        assert_no_link_or_reparse(path, boundary=resolved_root)
    except ValueError as exc:
        raise AtomicWriteError("dataset transaction lock path is unsafe") from exc
    return path


def _verify_dataset_precondition(
    root: Path,
    expected_files: Mapping[PurePosixPath, bytes],
) -> None:
    expected: dict[PurePosixPath, bytes] = {}
    for relative, data in expected_files.items():
        if relative in expected:
            raise AtomicWriteError("dataset snapshot precondition contains a duplicate path")
        expected[relative] = bytes(data)
    if PurePosixPath("dataset.json") not in expected:
        raise AtomicWriteError("dataset snapshot precondition is missing dataset.json")

    current_paths = _current_canonical_dataset_paths(root)
    expected_paths = set(expected)
    if current_paths != expected_paths:
        changed = sorted(current_paths ^ expected_paths, key=str)[0]
        raise AtomicWriteError(
            "dataset changed since the snapshot was loaded: "
            f"canonical path set differs at {changed}"
        )
    expected_identities: set[str] = set()
    for relative, data in expected.items():
        # Validate once, immediately before reading. Retaining destinations from
        # an earlier pass would miss links introduced while the tree is scanned.
        destination = _validate_relative_path(root, relative)
        identity = _validated_destination_identity(destination)
        if identity in expected_identities:
            raise AtomicWriteError(
                "dataset snapshot precondition contains paths that alias the same target"
            )
        expected_identities.add(identity)
        if (
            not destination.is_file()
            or destination.stat().st_size != len(data)
            or _file_sha256(destination) != _digest(data)
        ):
            raise AtomicWriteError(f"dataset changed since the snapshot was loaded: {relative}")


def _verify_root_file_precondition(
    root: Path,
    expected_files: Mapping[PurePosixPath, tuple[int, str]],
) -> None:
    """Require an exact flat file snapshot while the transaction lock is held."""

    if len(expected_files) > _MAX_CHANGES:
        raise AtomicWriteError("root file precondition exceeds the path safety limit")
    expected: dict[PurePosixPath, tuple[int, str]] = {}
    identities: set[str] = set()
    total_bytes = 0
    for relative, metadata in expected_files.items():
        path = PurePosixPath(relative)
        if len(path.parts) != 1:
            raise AtomicWriteError("root file precondition accepts top-level files only")
        destination = _validate_relative_path(root, path)
        identity = _validated_destination_identity(destination)
        if path in expected or identity in identities:
            raise AtomicWriteError("root file precondition contains aliased paths")
        size, digest = metadata
        if isinstance(size, bool) or size < 0 or not _SHA256.fullmatch(digest):
            raise AtomicWriteError("root file precondition metadata is invalid")
        total_bytes += size
        if total_bytes > _MAX_TOTAL_PAYLOAD_BYTES:
            raise AtomicWriteError("root file precondition exceeds the 1 GiB safety limit")
        if destination.parent != root:
            raise AtomicWriteError("root file precondition path escapes the transaction root")
        expected[path] = (size, digest)
        identities.add(identity)

    current: dict[PurePosixPath, Path] = {}
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise AtomicWriteError("transaction root could not be inspected safely") from exc
    for child in children:
        if is_link_or_reparse_point(child) or not child.is_file():
            raise AtomicWriteError(f"transaction root changed since it was inspected: {child.name}")
        current[PurePosixPath(child.name)] = child
    if set(current) != set(expected):
        changed = sorted(set(current) ^ set(expected), key=str)[0]
        raise AtomicWriteError(f"transaction root changed since it was inspected: {changed}")
    for relative, (expected_size, expected_digest) in expected.items():
        current_path = current[relative]
        try:
            size = current_path.stat().st_size
            digest = _file_sha256(current_path)
        except OSError as exc:
            raise AtomicWriteError(
                f"transaction root changed since it was inspected: {relative}"
            ) from exc
        if size != expected_size or digest != expected_digest:
            raise AtomicWriteError(f"transaction root changed since it was inspected: {relative}")


def _verify_tree_file_precondition(
    root: Path,
    expected_files: Mapping[PurePosixPath, tuple[int, str]],
) -> None:
    """Require an exact bounded recursive file tree while holding the transaction lock."""

    if len(expected_files) > _MAX_CHANGES:
        raise AtomicWriteError("tree precondition exceeds the path safety limit")
    expected: dict[PurePosixPath, tuple[int, str]] = {}
    identities: set[str] = set()
    total_bytes = 0
    for relative, metadata in expected_files.items():
        path = PurePosixPath(relative)
        destination = _validate_relative_path(root, path)
        identity = _validated_destination_identity(destination)
        if path in expected or identity in identities:
            raise AtomicWriteError("tree precondition contains aliased paths")
        size, digest = metadata
        if isinstance(size, bool) or size < 0 or not _SHA256.fullmatch(digest):
            raise AtomicWriteError("tree precondition metadata is invalid")
        if destination == root:
            raise AtomicWriteError("tree precondition path escapes the transaction root")
        total_bytes += size
        if total_bytes > _MAX_TOTAL_PAYLOAD_BYTES:
            raise AtomicWriteError("tree precondition exceeds the 1 GiB safety limit")
        expected[path] = (size, digest)
        identities.add(identity)

    current: dict[PurePosixPath, Path] = {}
    try:
        paths = list(root.rglob("*"))
    except OSError as exc:
        raise AtomicWriteError("transaction tree could not be inspected safely") from exc
    for candidate in paths:
        if is_link_or_reparse_point(candidate):
            raise AtomicWriteError(
                f"transaction tree changed since it was inspected: unsafe {candidate}"
            )
        relative = PurePosixPath(candidate.relative_to(root).as_posix())
        if candidate.is_file():
            current[relative] = candidate
        elif not candidate.is_dir():
            raise AtomicWriteError(
                f"transaction tree changed since it was inspected: special {relative}"
            )
    if set(current) != set(expected):
        changed = sorted(set(current) ^ set(expected), key=str)[0]
        raise AtomicWriteError(f"transaction tree changed since it was inspected: {changed}")
    for relative, (expected_size, expected_digest) in expected.items():
        current_path = current[relative]
        try:
            size = current_path.stat().st_size
            digest = _file_sha256(current_path)
        except OSError as exc:
            raise AtomicWriteError(
                f"transaction tree changed since it was inspected: {relative}"
            ) from exc
        if size != expected_size or digest != expected_digest:
            raise AtomicWriteError(f"transaction tree changed since it was inspected: {relative}")


def _current_canonical_dataset_paths(root: Path) -> set[PurePosixPath]:
    result: set[PurePosixPath] = set()
    manifest = root / "dataset.json"
    if manifest.is_symlink() or (manifest.exists() and not manifest.is_file()):
        raise AtomicWriteError("dataset changed since the snapshot was loaded: dataset.json")
    if manifest.is_file():
        result.add(PurePosixPath("dataset.json"))
    for name in ("data", "tombstones"):
        directory = root / name
        try:
            assert_no_link_or_reparse(directory, boundary=root)
        except ValueError as exc:
            raise AtomicWriteError(
                f"dataset changed since the snapshot was loaded: unsafe {name} path"
            ) from exc
        if directory.exists() and not directory.is_dir():
            raise AtomicWriteError(
                f"dataset changed since the snapshot was loaded: unsafe {name} path"
            )
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if is_link_or_reparse_point(path):
                raise AtomicWriteError(
                    f"dataset changed since the snapshot was loaded: unsafe path {path}"
                )
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                result.add(PurePosixPath(path.relative_to(root).as_posix()))
    return result


def _rollback_entries(
    root: Path,
    directory: Path,
    entries: Sequence[TransactionEntry],
) -> None:
    states = _inspect_target_states(root, entries)
    backups: dict[int, Path] = {}
    for state in states:
        entry = state.entry
        current = state.current_sha256
        if current == entry.old_sha256:
            continue
        if entry.old_sha256 is not None:
            path = _backup_path(directory, entry.index)
            _validate_payload_file(path, entry.old_sha256, entry.old_size)
            backups[entry.index] = path

    for state in reversed(states):
        entry = state.entry
        if state.current_sha256 == entry.old_sha256:
            continue
        destination = safe_destination(root, entry.relative)
        if entry.old_sha256 is None:
            destination.unlink(missing_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backups[entry.index], destination)
        _fsync_directory(destination.parent)


def _inspect_target_states(
    root: Path, entries: Sequence[TransactionEntry]
) -> tuple[_TargetState, ...]:
    result: list[_TargetState] = []
    for entry in entries:
        destination = safe_destination(root, entry.relative)
        if destination.is_symlink():
            raise AtomicWriteError(f"dataset transaction target became a symlink: {entry.relative}")
        if destination.exists():
            if not destination.is_file():
                raise AtomicWriteError(
                    f"dataset transaction target is not a file: {entry.relative}"
                )
            current_size = destination.stat().st_size
            allowed_sizes = {size for size in (entry.old_size, entry.new_size) if size is not None}
            if current_size not in allowed_sizes:
                raise AtomicWriteError(
                    f"dataset transaction target changed outside the transaction: {entry.relative}"
                )
            current_sha256 = _file_sha256(destination)
        else:
            current_sha256 = None
        if current_sha256 not in {entry.old_sha256, entry.new_sha256}:
            raise AtomicWriteError(
                f"dataset transaction target changed outside the transaction: {entry.relative}"
            )
        result.append(_TargetState(entry=entry, current_sha256=current_sha256))
    return tuple(result)


def _read_bounded_control_file(
    directory: Path,
    path: Path,
    *,
    limit: int,
    label: str,
) -> bytes:
    _assert_safe_internal_path(directory, path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AtomicWriteError(f"dataset transaction {label} is unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if not stat.S_ISREG(metadata.st_mode) or attributes & reparse_flag:
            raise AtomicWriteError(f"dataset transaction {label} is unsafe")
        if metadata.st_size > limit:
            raise AtomicWriteError(f"dataset transaction {label} exceeds its size limit")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > limit:
            raise AtomicWriteError(f"dataset transaction {label} exceeds its size limit")
        return data
    finally:
        os.close(descriptor)


def _load_manifest(root: Path, directory: Path) -> tuple[TransactionEntry, ...]:
    path = directory / _MANIFEST_NAME
    data = _read_bounded_control_file(
        directory,
        path,
        limit=_MAX_MANIFEST_BYTES,
        label="manifest",
    )
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicWriteError("dataset transaction manifest is not valid JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "entries",
        "format_version",
        "kind",
        "root_sha256",
    }:
        raise AtomicWriteError("dataset transaction manifest has an invalid shape")
    if (
        type(raw["format_version"]) is not int
        or raw["format_version"] != _FORMAT_VERSION
        or not isinstance(raw["kind"], str)
        or raw["kind"] != _TRANSACTION_KIND
        or not isinstance(raw["root_sha256"], str)
        or raw["root_sha256"] != _root_sha256(root)
    ):
        raise AtomicWriteError("dataset transaction manifest is not bound to this root")
    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list) or not 1 <= len(raw_entries) <= _MAX_CHANGES:
        raise AtomicWriteError("dataset transaction entry count is invalid")
    entries: list[TransactionEntry] = []
    seen: set[PurePosixPath] = set()
    seen_identities: set[str] = set()
    for index, value in enumerate(raw_entries):
        if not isinstance(value, dict) or set(value) != {
            "new_sha256",
            "new_size",
            "old_sha256",
            "old_size",
            "path",
        }:
            raise AtomicWriteError("dataset transaction entry has an invalid shape")
        raw_path = value["path"]
        if not isinstance(raw_path, str):
            raise AtomicWriteError("dataset transaction path must be text")
        relative = PurePosixPath(raw_path)
        destination = _validate_relative_path(root, relative, original=raw_path)
        if relative in seen:
            raise AtomicWriteError("dataset transaction contains a duplicate path")
        seen.add(relative)
        identity = _validated_destination_identity(destination)
        if identity in seen_identities:
            raise AtomicWriteError("dataset transaction contains paths that alias the same target")
        seen_identities.add(identity)
        old_sha256 = _validated_optional_sha256(value["old_sha256"])
        new_sha256 = _validated_optional_sha256(value["new_sha256"])
        old_size = _validated_optional_size(value["old_size"])
        new_size = _validated_optional_size(value["new_size"])
        if (old_sha256 is None) != (old_size is None) or (new_sha256 is None) != (new_size is None):
            raise AtomicWriteError("dataset transaction hash and size presence differs")
        entries.append(
            TransactionEntry(
                index=index,
                relative=relative,
                old_sha256=old_sha256,
                new_sha256=new_sha256,
                old_size=old_size,
                new_size=new_size,
            )
        )
    if tuple(entry.relative.as_posix() for entry in entries) != tuple(
        sorted(entry.relative.as_posix() for entry in entries)
    ):
        raise AtomicWriteError("dataset transaction paths are not canonical")
    total_payload_bytes = sum(
        size for entry in entries for size in (entry.old_size, entry.new_size) if size is not None
    )
    if total_payload_bytes > _MAX_TOTAL_PAYLOAD_BYTES:
        raise AtomicWriteError("dataset transaction exceeds the 1 GiB payload safety limit")
    return tuple(entries)


def _load_state(path: Path) -> bytes | None:
    if not path.exists():
        return None
    value = _read_bounded_control_file(
        path.parent,
        path,
        limit=max(len(_READY_STATE), len(_COMMITTED_STATE)),
        label="state marker",
    )
    if value not in {_READY_STATE, _COMMITTED_STATE}:
        return b"invalid"
    return value


def _validate_internal_contents(directory: Path, entries: Sequence[TransactionEntry]) -> None:
    allowed_top = {
        _MANIFEST_NAME,
        _MANIFEST_TEMP_NAME,
        _PREPARING_NAME,
        _PREPARING_TEMP_NAME,
        _STATE_NAME,
        _STATE_TEMP_NAME,
        _BACKUP_DIRECTORY_NAME,
        _STAGED_DIRECTORY_NAME,
    }
    for path in directory.iterdir():
        _assert_safe_internal_path(directory, path)
        if path.name not in allowed_top:
            raise AtomicWriteError("dataset transaction contains an unexpected internal path")
        if path.name not in {_BACKUP_DIRECTORY_NAME, _STAGED_DIRECTORY_NAME} and not path.is_file():
            raise AtomicWriteError("dataset transaction control path is not a regular file")
    for subdirectory_name, prefix in (
        (_BACKUP_DIRECTORY_NAME, ".mojilex-backup-"),
        (_STAGED_DIRECTORY_NAME, ".mojilex-stage-"),
    ):
        subdirectory = directory / subdirectory_name
        if not subdirectory.exists():
            continue
        _assert_safe_internal_path(directory, subdirectory)
        if not subdirectory.is_dir():
            raise AtomicWriteError("dataset transaction payload directory is unsafe")
        allowed = {f"{prefix}{entry.index:06d}" for entry in entries}
        for path in subdirectory.iterdir():
            _assert_safe_internal_path(directory, path)
            if path.name not in allowed or not path.is_file():
                raise AtomicWriteError("dataset transaction contains an unexpected payload")


def _validate_complete_prepared_payloads(
    directory: Path, entries: Sequence[TransactionEntry]
) -> None:
    expected = {
        _backup_path(directory, entry.index) for entry in entries if entry.old_sha256 is not None
    } | {_staged_path(directory, entry.index) for entry in entries if entry.new_sha256 is not None}
    actual = {
        path
        for entry in entries
        for path in (
            _backup_path(directory, entry.index),
            _staged_path(directory, entry.index),
        )
        if path.exists()
    }
    if actual != expected:
        raise AtomicWriteError(
            "dataset transaction state marker is missing and its payload is incomplete"
        )
    for entry in entries:
        if entry.old_sha256 is not None:
            _validate_payload_file(
                _backup_path(directory, entry.index), entry.old_sha256, entry.old_size
            )
        if entry.new_sha256 is not None:
            _validate_payload_file(
                _staged_path(directory, entry.index), entry.new_sha256, entry.new_size
            )


def _cleanup_transaction(
    root: Path,
    directory: Path,
    entries: Sequence[TransactionEntry],
) -> None:
    _validate_internal_contents(directory, entries)
    for entry in entries:
        _unlink_internal(directory, _backup_path(directory, entry.index), missing_ok=True)
        _unlink_internal(directory, _staged_path(directory, entry.index), missing_ok=True)
    for name in (_BACKUP_DIRECTORY_NAME, _STAGED_DIRECTORY_NAME):
        path = directory / name
        if path.exists():
            _rmdir_internal(directory, path)
    _unlink_internal(directory, directory / _MANIFEST_TEMP_NAME, missing_ok=True)
    _unlink_internal(directory, directory / _PREPARING_TEMP_NAME, missing_ok=True)
    _unlink_internal(directory, directory / _STATE_TEMP_NAME, missing_ok=True)
    _unlink_internal(directory, directory / _MANIFEST_NAME, missing_ok=True)
    _fsync_directory(directory)
    _unlink_internal(directory, directory / _STATE_NAME, missing_ok=True)
    _fsync_directory(directory)
    _unlink_internal(directory, directory / _PREPARING_NAME, missing_ok=True)
    directory.rmdir()
    _fsync_directory(root)


def _cleanup_manifestless_finalization(root: Path, directory: Path) -> None:
    """Finish only the strict residue left after a manifest was durably retired."""

    paths = tuple(directory.iterdir())
    if {path.name for path in paths} != {_PREPARING_NAME, _STATE_NAME}:
        raise AtomicWriteError(
            "manifestless dataset transaction is not a safe finalization residue"
        )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise AtomicWriteError("manifestless dataset transaction contains an unsafe path")
    preparing = directory / _PREPARING_NAME
    if _read_bounded_control_file(
        directory,
        preparing,
        limit=_MAX_PREPARING_BYTES,
        label="preparation marker",
    ) != _preparing_bytes(root):
        raise AtomicWriteError("dataset transaction preparation marker is not bound to this root")
    state = _load_state(directory / _STATE_NAME)
    if state not in {_READY_STATE, _COMMITTED_STATE}:
        raise AtomicWriteError("manifestless dataset transaction state marker is corrupt")

    _unlink_internal(directory, directory / _STATE_NAME)
    _fsync_directory(directory)
    _unlink_internal(directory, preparing)
    directory.rmdir()
    _fsync_directory(root)


def _restore_ready_state_for_cleanup(directory: Path) -> None:
    temporary = directory / _STATE_TEMP_NAME
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise AtomicWriteError("dataset transaction temporary state marker is unsafe")
        partial = _read_bounded_control_file(
            directory,
            temporary,
            limit=len(_READY_STATE),
            label="temporary state marker",
        )
        if len(partial) > len(_READY_STATE) or not _READY_STATE.startswith(partial):
            raise AtomicWriteError("dataset transaction temporary state marker is corrupt")
        _unlink_internal(directory, temporary)
        _fsync_directory(directory)
    _replace_control_file(directory, _STATE_TEMP_NAME, _STATE_NAME, _READY_STATE)


def _cleanup_incomplete_preparation(root: Path, directory: Path) -> None:
    """Remove only a strictly recognized journal that cannot have touched targets."""

    paths = tuple(directory.iterdir())
    if not paths:
        directory.rmdir()
        _fsync_directory(root)
        return

    allowed_top = {
        _MANIFEST_TEMP_NAME,
        _PREPARING_NAME,
        _PREPARING_TEMP_NAME,
        _BACKUP_DIRECTORY_NAME,
        _STAGED_DIRECTORY_NAME,
    }
    for path in paths:
        _assert_safe_internal_path(directory, path)
        if path.name not in allowed_top:
            raise AtomicWriteError(
                "incomplete dataset transaction contains an unexpected internal path"
            )

    expected_preparing = _preparing_bytes(root)
    preparing = directory / _PREPARING_NAME
    preparing_temporary = directory / _PREPARING_TEMP_NAME
    if preparing.exists():
        if preparing_temporary.exists():
            raise AtomicWriteError("dataset transaction preparation marker is ambiguous")
        _assert_safe_internal_path(directory, preparing)
        if (
            _read_bounded_control_file(
                directory,
                preparing,
                limit=_MAX_PREPARING_BYTES,
                label="preparation marker",
            )
            != expected_preparing
        ):
            raise AtomicWriteError(
                "dataset transaction preparation marker is not bound to this root"
            )
    else:
        if any(path.name not in {_PREPARING_TEMP_NAME} for path in paths):
            raise AtomicWriteError("dataset transaction preparation marker is missing")
        _assert_safe_internal_path(directory, preparing_temporary)
        partial = _read_bounded_control_file(
            directory,
            preparing_temporary,
            limit=_MAX_PREPARING_BYTES,
            label="temporary preparation marker",
        )
        if not expected_preparing.startswith(partial):
            raise AtomicWriteError("dataset transaction preparation marker is corrupt")

    payloads: list[Path] = []
    total_payload_bytes = 0
    for directory_name, prefix in (
        (_BACKUP_DIRECTORY_NAME, ".mojilex-backup-"),
        (_STAGED_DIRECTORY_NAME, ".mojilex-stage-"),
    ):
        payload_directory = directory / directory_name
        if not payload_directory.exists():
            continue
        _assert_safe_internal_path(directory, payload_directory)
        if not payload_directory.is_dir():
            raise AtomicWriteError("dataset transaction payload directory is unsafe")
        for payload in payload_directory.iterdir():
            _assert_safe_internal_path(directory, payload)
            suffix = payload.name.removeprefix(prefix)
            if (
                not payload.name.startswith(prefix)
                or len(suffix) != 6
                or not suffix.isascii()
                or not suffix.isdigit()
                or int(suffix) >= _MAX_CHANGES
                or not payload.is_file()
            ):
                raise AtomicWriteError(
                    "incomplete dataset transaction contains an unexpected payload"
                )
            total_payload_bytes += payload.stat().st_size
            if len(payloads) >= 2 * _MAX_CHANGES or total_payload_bytes > _MAX_TOTAL_PAYLOAD_BYTES:
                raise AtomicWriteError("incomplete dataset transaction payload is unbounded")
            payloads.append(payload)

    manifest_temporary = directory / _MANIFEST_TEMP_NAME
    if manifest_temporary.exists():
        _assert_safe_internal_path(directory, manifest_temporary)
        if (
            not manifest_temporary.is_file()
            or manifest_temporary.stat().st_size > _MAX_MANIFEST_BYTES
        ):
            raise AtomicWriteError("dataset transaction temporary manifest is unsafe")

    for payload in payloads:
        _unlink_internal(directory, payload)
    for name in (_BACKUP_DIRECTORY_NAME, _STAGED_DIRECTORY_NAME):
        payload_directory = directory / name
        if payload_directory.exists():
            _rmdir_internal(directory, payload_directory)
    _unlink_internal(directory, manifest_temporary, missing_ok=True)
    _unlink_internal(directory, preparing_temporary, missing_ok=True)
    _unlink_internal(directory, preparing, missing_ok=True)
    directory.rmdir()
    _fsync_directory(root)


def _replace_control_file(
    directory: Path,
    temporary_name: str,
    name: str,
    data: bytes,
) -> None:
    temporary = directory / temporary_name
    destination = directory / name
    _assert_safe_internal_path(directory, temporary)
    _assert_safe_internal_path(directory, destination)
    _write_durable_file(temporary, data)
    _assert_safe_internal_path(directory, temporary)
    _assert_safe_internal_path(directory, destination)
    os.replace(temporary, destination)
    _fsync_directory(directory)


def _preparing_bytes(root: Path) -> bytes:
    payload = {
        "format_version": _FORMAT_VERSION,
        "kind": _TRANSACTION_KIND,
        "phase": "preparing",
        "root_sha256": _root_sha256(root),
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_durable_file(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if os.name == "nt" or exc.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if os.name != "nt" and exc.errno not in {
                errno.EACCES,
                errno.EBADF,
                errno.EINVAL,
                errno.ENOTSUP,
            }:
                raise
    finally:
        os.close(descriptor)


def _validate_relative_path(
    root: Path,
    relative: PurePosixPath,
    *,
    original: str | None = None,
) -> Path:
    raw = relative.as_posix() if original is None else original
    if (
        raw != relative.as_posix()
        or "\\" in raw
        or (
            relative.parts
            and os.path.normcase(relative.parts[0])
            in {
                os.path.normcase(TRANSACTION_DIRECTORY_NAME),
                os.path.normcase(_LOCAL_STATE_DIRECTORY_NAME),
            }
        )
        or len(raw.encode("utf-8")) > _MAX_RELATIVE_PATH_BYTES
    ):
        raise AtomicWriteError("dataset transaction contains a non-canonical path")
    if os.name == "nt":
        for component in relative.parts:
            _validate_windows_path_component(component)
    try:
        return safe_destination(root, relative, canonical=True)
    except (OSError, ValueError) as exc:
        raise AtomicWriteError("dataset transaction contains an unsafe path") from exc


def _validate_windows_path_component(component: str) -> None:
    if ":" in component or component.endswith((".", " ")):
        raise AtomicWriteError("dataset transaction contains an unsafe Windows path")
    if _WINDOWS_SHORT_NAME.fullmatch(component) is not None:
        raise AtomicWriteError(
            "dataset transaction contains an unsafe Windows path with a DOS short-name shape"
        )
    basename = component.split(".", 1)[0].rstrip(" .")
    if _WINDOWS_DEVICE_BASENAME.fullmatch(basename) is not None:
        raise AtomicWriteError("dataset transaction contains a reserved Windows path")


def transaction_path_identity(root: Path, relative: PurePosixPath) -> str:
    """Return the platform-normalized identity of one safe transaction target."""

    destination = _validate_relative_path(root, relative)
    return _validated_destination_identity(destination)


def _validated_destination_identity(destination: Path) -> str:
    # Reuse the canonical target returned by safe_destination's containment
    # check, including the filesystem's own Windows name/case normalization.
    return os.path.normcase(str(destination))


def _assert_safe_internal_path(directory: Path, path: Path) -> None:
    try:
        assert_no_link_or_reparse(path, boundary=directory)
    except ValueError as exc:
        raise AtomicWriteError("dataset transaction contains a link or reparse point") from exc


def _unlink_internal(directory: Path, path: Path, *, missing_ok: bool = False) -> None:
    _assert_safe_internal_path(directory, path)
    path.unlink(missing_ok=missing_ok)


def _rmdir_internal(directory: Path, path: Path) -> None:
    _assert_safe_internal_path(directory, path)
    path.rmdir()


def _validate_payload_file(path: Path, expected_sha256: str, expected_size: int | None) -> None:
    _assert_safe_internal_path(path.parents[1], path)
    if not path.is_file():
        raise AtomicWriteError("dataset transaction payload is missing or unsafe")
    if expected_size is None or path.stat().st_size != expected_size:
        raise AtomicWriteError("dataset transaction payload size does not match its manifest")
    if _file_sha256(path) != expected_sha256:
        raise AtomicWriteError("dataset transaction payload hash does not match its manifest")


def _validated_optional_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AtomicWriteError("dataset transaction contains an invalid SHA-256")
    return value


def _validated_optional_size(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= _MAX_TOTAL_PAYLOAD_BYTES:
        raise AtomicWriteError("dataset transaction contains an invalid payload size")
    return value


def _backup_path(directory: Path, index: int) -> Path:
    return directory / _BACKUP_DIRECTORY_NAME / f".mojilex-backup-{index:06d}"


def _staged_path(directory: Path, index: int) -> Path:
    return directory / _STAGED_DIRECTORY_NAME / f".mojilex-stage-{index:06d}"


def _resolve_dataset_root(root: str | Path) -> Path:
    unresolved = Path(os.path.abspath(root))
    try:
        assert_no_link_or_reparse(unresolved)
    except ValueError as exc:
        raise AtomicWriteError("dataset root contains a link or reparse point") from exc
    return unresolved.resolve()


def _root_sha256(root: Path) -> str:
    normalized = os.path.normcase(str(root.resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

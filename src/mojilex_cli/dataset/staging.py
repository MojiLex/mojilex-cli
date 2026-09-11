"""Validated staging and rollback-safe replacement of dataset files."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from .layout import assert_no_link_or_reparse, safe_destination
from .repository import DatasetSnapshot
from .transaction import AtomicWriteError as AtomicWriteError
from .transaction import (
    DurableDatasetTransaction,
    recover_pending_dataset_transaction,
    transaction_path_identity,
)


class AtomicDatasetWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_files: Mapping[PurePosixPath, bytes] | None = None,
        expected_root_files: Mapping[PurePosixPath, tuple[int, str]] | None = None,
        transaction_lock_root: str | Path | None = None,
        transaction_lock_name: str | None = None,
    ) -> None:
        unresolved = Path(root)
        try:
            assert_no_link_or_reparse(unresolved)
        except ValueError as exc:
            raise ValueError("dataset root must not contain a link or reparse point") from exc
        self.root = unresolved.resolve()
        self._transaction_lock_root = (
            Path(transaction_lock_root) if transaction_lock_root is not None else None
        )
        self._transaction_lock_name = transaction_lock_name
        recover_pending_dataset_transaction(
            self.root,
            lock_root=self._transaction_lock_root,
            lock_name=self._transaction_lock_name,
        )
        self._changes: dict[PurePosixPath, bytes | None] = {}
        self._change_identities: dict[str, PurePosixPath] = {}
        self._expected_files: dict[PurePosixPath, bytes] | None = None
        if expected_files is not None:
            normalized: dict[PurePosixPath, bytes] = {}
            for relative, data in expected_files.items():
                path = PurePosixPath(relative)
                safe_destination(self.root, path)
                if path in normalized:
                    raise ValueError(f"duplicate expected dataset path: {path}")
                normalized[path] = bytes(data)
            self._expected_files = normalized
        self._expected_root_files: dict[PurePosixPath, tuple[int, str]] | None = None
        if expected_root_files is not None:
            normalized_root_files: dict[PurePosixPath, tuple[int, str]] = {}
            for relative, metadata in expected_root_files.items():
                path = PurePosixPath(relative)
                if len(path.parts) != 1:
                    raise ValueError("root file precondition accepts top-level files only")
                safe_destination(self.root, path)
                if path in normalized_root_files:
                    raise ValueError(f"duplicate expected root path: {path}")
                normalized_root_files[path] = metadata
            self._expected_root_files = normalized_root_files

    def stage_bytes(self, relative: str | PurePosixPath, data: bytes) -> None:
        path = PurePosixPath(relative)
        safe_destination(self.root, path)
        self._record_path_identity(path)
        self._changes[path] = bytes(data)

    def stage_delete(self, relative: str | PurePosixPath) -> None:
        path = PurePosixPath(relative)
        safe_destination(self.root, path)
        self._record_path_identity(path)
        self._changes[path] = None

    def _record_path_identity(self, path: PurePosixPath) -> None:
        identity = transaction_path_identity(self.root, path)
        existing = self._change_identities.get(identity)
        if existing is not None and existing != path:
            raise AtomicWriteError(
                f"dataset transaction paths alias the same target: {existing} and {path}"
            )
        self._change_identities[identity] = path

    def commit(self) -> tuple[PurePosixPath, ...]:
        if not self._changes:
            return ()
        transaction = DurableDatasetTransaction.prepare(
            self.root,
            self._changes,
            expected_files=self._expected_files,
            expected_root_files=self._expected_root_files,
            lock_root=self._transaction_lock_root,
            lock_name=self._transaction_lock_name,
        )
        try:
            for entry in transaction.entries:
                destination = safe_destination(self.root, entry.relative)
                if entry.new_sha256 is None:
                    if destination.exists():
                        destination.unlink()
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(transaction.staged_path(entry), destination)
                transaction.sync_target_parent(entry)
            transaction.mark_committed()
        except BaseException as exc:
            try:
                transaction.rollback()
            except BaseException as rollback_error:
                raise AtomicWriteError(
                    "dataset commit failed and durable rollback could not complete: "
                    f"{rollback_error}"
                ) from rollback_error
            if not isinstance(exc, Exception):
                raise
            raise AtomicWriteError(f"dataset commit failed and was rolled back: {exc}") from exc
        transaction.cleanup()
        changed = tuple(sorted(self._changes, key=str))
        self._changes.clear()
        self._change_identities.clear()
        return changed


def apply_snapshot(
    before: DatasetSnapshot,
    after: DatasetSnapshot,
    *,
    validator: Callable[[DatasetSnapshot], object] | None = None,
) -> tuple[PurePosixPath, ...]:
    """Validate complete staging state before touching the working tree."""

    if validator is not None:
        result = validator(after)
        valid = bool(getattr(result, "valid", result))
        if not valid:
            issues = getattr(result, "issues", ())
            raise AtomicWriteError(f"staged dataset is invalid: {issues}")
    old_files = before.to_files()
    new_files = after.to_files()
    expected_files = before.source_bytes if before.source_bytes else old_files
    writer = AtomicDatasetWriter(before.root, expected_files=expected_files)
    for relative in sorted(set(old_files) | set(new_files), key=str):
        old = old_files.get(relative)
        new = new_files.get(relative)
        if old == new:
            continue
        if new is None:
            writer.stage_delete(relative)
        else:
            writer.stage_bytes(relative, new)
    changed = writer.commit()
    after.source_bytes = dict(new_files)
    return changed

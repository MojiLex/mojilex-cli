"""Exact-content ownership for resumable private staging writes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset import DatasetSnapshot
from mojilex_cli.dataset.layout import (
    assert_no_link_or_reparse,
    collection_path,
    collection_shard,
    emoji_bucket_path,
    legacy_bucket_path,
    memberships_path,
    previous_emoji_bucket_path,
)
from mojilex_cli.dataset.serialization import (
    serialize_collection,
    serialize_emojis,
    serialize_memberships,
)
from mojilex_cli.domain.models import Membership
from mojilex_cli.git import GitError, GitRunner
from mojilex_cli.runs import RunCheckpoint

from .workspaces import staging_workspace_path

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPTS = "staging_owned_paths"
_MAX_RECEIPTS = 131_072


def _receipt_path(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 256 or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and bool(path.parts)
        and not any(part in {".", ".."} for part in path.parts)
        and (value == "dataset.json" or path.parts[0] in {"data", "tombstones"})
    )


def _receipts(checkpoint: RunCheckpoint) -> dict[str, str]:
    raw = checkpoint.safe_parameters.get(_RECEIPTS, {})
    if not isinstance(raw, dict) or len(raw) > _MAX_RECEIPTS:
        return {}
    return {
        path: digest
        for path, digest in raw.items()
        if _receipt_path(path) and isinstance(digest, str) and _DIGEST.fullmatch(digest)
    }


def _private_workspace(
    snapshot: DatasetSnapshot, checkpoint: RunCheckpoint, runs_dir: Path
) -> bool:
    try:
        assert_no_link_or_reparse(runs_dir)
        expected = staging_workspace_path(runs_dir, checkpoint.run_id)
        if snapshot.root != expected:
            return False
        assert_no_link_or_reparse(snapshot.root, boundary=runs_dir)
        assert_no_link_or_reparse(snapshot.root / ".git", boundary=snapshot.root)
        git = GitRunner(snapshot.root)
        return (
            Path(git.run("rev-parse", "--show-toplevel").stdout.strip()).resolve() == expected
            and git.current_sha() == checkpoint.base_revision
        )
    except (CommandError, GitError, OSError, ValueError):
        return False


def _unchanged_source(snapshot: DatasetSnapshot, path: PurePosixPath, expected: bytes) -> bool:
    try:
        source = snapshot.root.joinpath(*path.parts)
        assert_no_link_or_reparse(source, boundary=snapshot.root)
        return source.read_bytes() == expected
    except (OSError, ValueError):
        return False


def staging_guard_exemptions(
    before: DatasetSnapshot,
    after: DatasetSnapshot,
    checkpoint: RunCheckpoint | None,
    *,
    runs_dir: Path,
    changed_paths: Iterable[str],
) -> set[str]:
    """Allow proven own outputs or byte-identical layout moves in this run's tree.

    This is never used for a normal local checkout or publication. Atomic writes
    still verify the complete exact-content baseline under the transaction lock.
    Old checkpoints without receipts cannot authorize semantic edits to dirty data.
    """
    if checkpoint is None or not _private_workspace(before, checkpoint, runs_dir):
        return set()
    changed = set(changed_paths)
    result: set[str] = set()
    for path, digest in _receipts(checkpoint).items():
        if path not in changed:
            continue
        relative = PurePosixPath(path)
        data = before.source_bytes.get(relative)
        if data is not None and hashlib.sha256(data).hexdigest() == digest:
            if _unchanged_source(before, relative, data):
                result.add(path)

    def pure_move(old: PurePosixPath, new: PurePosixPath, expected: bytes) -> None:
        if str(old) not in changed or old == new or str(old) in result:
            return
        data = before.source_bytes.get(old)
        if data != expected or not _unchanged_source(before, old, expected):
            return
        try:
            destination = before.root.joinpath(*new.parts)
            assert_no_link_or_reparse(destination, boundary=before.root)
            if not destination.exists():
                result.add(str(old))
        except (OSError, ValueError):
            return

    for identifier, emoji in before.emojis.items():
        replacement = after.emojis.get(identifier)
        if replacement is None:
            continue
        new = emoji_bucket_path(emoji.platform, identifier)
        old_paths = (
            previous_emoji_bucket_path(emoji.platform, identifier),
            legacy_bucket_path(new),
        )
        if any(str(path) in changed for path in old_paths):
            encoded = serialize_emojis([replacement])
            for old in old_paths:
                pure_move(old, new, encoded)
    members: dict[str, list[Membership]] = {}
    for membership in after.memberships.values():
        members.setdefault(membership.collection_id, []).append(membership)
    for identifier, collection in before.collections.items():
        collection_replacement = after.collections.get(identifier)
        if collection_replacement is None:
            continue
        old_dir = PurePosixPath(
            "data", collection.platform, "collections", collection_shard(identifier), identifier
        )
        old = old_dir / "collection.json"
        if str(old) in changed:
            pure_move(
                old,
                collection_path(collection.platform, identifier),
                serialize_collection(collection_replacement),
            )
        old = old_dir / "memberships.jsonl"
        if str(old) in changed:
            pure_move(
                old,
                memberships_path(collection.platform, identifier),
                serialize_memberships(members.get(identifier, [])),
            )
    return result


def staging_output_receipts(
    checkpoint: RunCheckpoint,
    snapshot: DatasetSnapshot,
    changed_paths: Iterable[str],
    *,
    runs_dir: Path,
) -> dict[str, str | None]:
    """Compute a receipt delta without copying a concurrently updated checkpoint."""
    if not _private_workspace(snapshot, checkpoint, runs_dir):
        return {}
    receipts: dict[str, str | None] = {}
    for path in changed_paths:
        if not _receipt_path(path):
            continue
        data = snapshot.source_bytes.get(PurePosixPath(path))
        receipts[path] = hashlib.sha256(data).hexdigest() if data is not None else None
    return receipts


def checkpoint_staging_outputs(
    checkpoint: RunCheckpoint, delta: dict[str, str | None]
) -> RunCheckpoint:
    """Merge durable output receipts into the latest checkpoint without yielding."""
    if not delta:
        return checkpoint
    receipts = _receipts(checkpoint)
    for path, digest in delta.items():
        if not _receipt_path(path):
            continue
        if digest is None:
            receipts.pop(path, None)
        elif _DIGEST.fullmatch(digest):
            receipts[path] = digest
    if len(receipts) > _MAX_RECEIPTS:
        receipts = {}  # Exceeding the optional cache never authorizes unproven files.
    return checkpoint.model_copy(
        update={"safe_parameters": {**checkpoint.safe_parameters, _RECEIPTS: receipts}}
    )

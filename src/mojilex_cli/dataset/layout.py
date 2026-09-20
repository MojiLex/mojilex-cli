"""Canonical, title-independent dataset paths."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collection_shard(collection_id: str) -> str:
    return _sha(collection_id)[:2]


def emoji_shards(emoji_id: str) -> tuple[str, str]:
    digest = _sha(emoji_id)
    return digest[:2], digest[2:8]


def tombstone_shard(target_id: str) -> str:
    return _sha(target_id)[:2]


def visual_relation_shards(relation_id: str) -> tuple[str, str]:
    digest = _sha(relation_id)
    return digest[:2], digest[2:8]


def collection_directory(platform: str, collection_id: str) -> PurePosixPath:
    return PurePosixPath(
        "data", platform, "collections", collection_shard(collection_id), collection_id
    )


def collection_path(platform: str, collection_id: str) -> PurePosixPath:
    return collection_directory(platform, collection_id) / "collection.json"


def memberships_path(platform: str, collection_id: str) -> PurePosixPath:
    return collection_directory(platform, collection_id) / "memberships.jsonl"


def emoji_bucket_path(platform: str, emoji_id: str) -> PurePosixPath:
    first, second = emoji_shards(emoji_id)
    return PurePosixPath("data", platform, "emojis", first, f"{second}.jsonl")


def tombstone_path(target_id: str) -> PurePosixPath:
    return PurePosixPath("tombstones", tombstone_shard(target_id), f"{target_id}.json")


def visual_relations_path(relation_id: str) -> PurePosixPath:
    first, second = visual_relation_shards(relation_id)
    return PurePosixPath("data", "relations", "visual", first, f"{second}.jsonl")


def legacy_bucket_path(path: PurePosixPath) -> PurePosixPath:
    """The previous four-hex bucket for an already computed canonical bucket path."""
    return path.with_name(f"{path.stem[:2]}.jsonl")


def is_link_or_reparse_point(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without following them."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def assert_no_link_or_reparse(
    path: Path,
    *,
    boundary: Path | None = None,
) -> None:
    """Reject every existing link/reparse component on one lexical path."""

    absolute = Path(os.path.abspath(path))
    if boundary is None:
        candidates = tuple(reversed((absolute, *absolute.parents)))
    else:
        absolute_boundary = Path(os.path.abspath(boundary))
        try:
            relative = absolute.relative_to(absolute_boundary)
        except ValueError as exc:
            raise ValueError(f"path escapes safety boundary: {path}") from exc
        current = absolute_boundary
        collected = [current]
        for part in relative.parts:
            current /= part
            collected.append(current)
        candidates = tuple(collected)
    for candidate in candidates:
        if is_link_or_reparse_point(candidate):
            raise ValueError(f"link or reparse-point traversal is forbidden: {candidate}")


def safe_destination(root: Path, relative: str | PurePosixPath, *, canonical: bool = False) -> Path:
    """Resolve a controlled relative path and reject traversal/symlink ancestors."""

    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"unsafe dataset path: {relative}")
    unresolved_root = Path(os.path.abspath(root))
    assert_no_link_or_reparse(unresolved_root)
    root = unresolved_root.resolve()
    destination = root.joinpath(*rel.parts)
    # This checks the destination as well as every ancestor below the root.
    # Repeating the same per-component lstats here doubles filesystem work for
    # every file in the full snapshot precondition without adding coverage.
    assert_no_link_or_reparse(destination, boundary=root)
    try:
        resolved_destination = destination.resolve(strict=False)
        resolved_destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {relative}") from exc
    return resolved_destination if canonical else destination

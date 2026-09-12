"""Disposable HEAD-bound authoring selector index; never an authoring snapshot.

Returned snapshots are partial read views for selector-to-source resolution only.
They must never be validated as a full dataset or passed to an atomic writer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any

from mojilex_cli.dataset import DatasetSnapshot, load_dataset, validate_snapshot
from mojilex_cli.dataset.layout import (
    assert_no_link_or_reparse,
    collection_path,
    emoji_bucket_path,
    memberships_path,
    safe_destination,
)
from mojilex_cli.dataset.serialization import parse_json, parse_jsonl
from mojilex_cli.domain import Collection, Emoji, Membership, membership_id
from mojilex_cli.git import GitError, GitRunner

_ID = re.compile(r"mx[ce]_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_INDEX_BYTES = 512 * 1024 * 1024
_MAX_RECORD_BYTES = 8 * 1024 * 1024
_MAX_SELECTORS = 1000
_FORMAT = "mojilex-authoring-lookup-v1"


def lookup_authoring_snapshot(
    root: Path,
    index_path: Path,
    selectors: Sequence[str],
    *,
    read_only: bool = False,
) -> DatasetSnapshot | None:
    """Resolve canonical IDs from a clean exact HEAD, or request full-loader fallback.

    Native names, URLs, dirty/no-Git trees, malformed caches and missing IDs all
    return None. Read-only mode neither builds a cache nor opens SQLite sidecars.
    Only IDs, canonical relative paths and hashes are persisted, never content.
    """
    if (
        not selectors
        or len(selectors) > _MAX_SELECTORS
        or any(not _ID.fullmatch(value) for value in selectors)
    ):
        return None
    try:
        assert_no_link_or_reparse(root)
        root = root.resolve(strict=True)
        assert_no_link_or_reparse(root / ".git", boundary=root)
        assert_no_link_or_reparse(index_path)
        index_path = index_path.resolve(strict=False)
        if index_path.is_relative_to(root):
            return None
        git = GitRunner(root)
        head = _clean_head(git)
        if head is None:
            return None
        git_dir = git.run("rev-parse", "--absolute-git-dir").stdout.strip()
        identity = _sha(f"{os.path.normcase(str(root))}\0{git_dir}".encode())
        expected = {"format": _FORMAT, "repository": identity, "head": head}
        cached = _read_cache(index_path, expected, selectors)
        if cached is None:
            if read_only:
                return None
            snapshot = load_dataset(root)
            validate_snapshot(snapshot, canonical=True).raise_for_errors()
            # A clean Git status alone does not bind ignored/assume-unchanged files.
            tree = {}
            for entry in git.run("ls-tree", "-rz", "--full-tree", head).stdout.split("\0"):
                if entry:
                    header, path_value = entry.split("\t", 1)
                    mode, kind, digest = header.split(" ")
                    if mode in {"100644", "100755"} and kind == "blob":
                        tree[PurePosixPath(path_value)] = digest
            for path, data in snapshot.source_bytes.items():
                if tree.get(path) != _git_blob_digest(head, data):
                    raise ValueError("canonical source is not bound to Git HEAD")
            if _clean_head(git) != head:
                return None
            metadata, rows = _index_rows(snapshot, expected)
            _write_cache(index_path, metadata, rows)
            cached = _read_cache(index_path, expected, selectors)
            if cached is None:
                return None
        metadata, records = cached
        subset = _read_subset(root, git, head, metadata, records)
        return subset if _clean_head(git) == head else None
    except (OSError, ValueError, TypeError, KeyError, RecursionError, sqlite3.Error, GitError):
        # A cache is an optimization, never permission to trust malformed data.
        return None


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _clean_head(git: GitRunner) -> str | None:
    head = git.current_sha()
    status = git.run(
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    if status:
        return None
    ignored = git.run(
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
        "--",
        "data",
        "dataset.json",
    ).stdout
    if ignored:
        # Ignore IDE/build/cache artifacts, but never an untracked canonical shadow
        # which could change the complete reverse emoji -> collection mapping.
        return None
    flags = git.run("ls-files", "-v", "-z", "--", "data", "dataset.json").stdout
    if any(entry and entry[0] != "H" for entry in flags.split("\0")):
        # assume-unchanged/skip-worktree can hide a changed membership elsewhere,
        # invalidating the reverse emoji -> collection mapping on a warm lookup.
        return None
    return head if git.current_sha() == head else None


def _verify_head_blob(git: GitRunner, head: str, path: PurePosixPath, data: bytes) -> None:
    object_id = git.run(
        "rev-parse", "--verify", "--end-of-options", f"{head}:{path.as_posix()}"
    ).stdout.strip()
    if object_id != _git_blob_digest(head, data):
        raise ValueError("canonical content is not bound to the indexed Git HEAD")


def _git_blob_digest(head: str, data: bytes) -> str:
    algorithm = "sha1" if len(head) == 40 else "sha256"
    return hashlib.new(algorithm, f"blob {len(data)}\0".encode() + data).hexdigest()


def _index_rows(
    snapshot: DatasetSnapshot, expected: dict[str, str]
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    references: dict[str, dict[str, str]] = {}
    for collection in snapshot.collections.values():
        path = collection_path(collection.platform, collection.id)
        members = memberships_path(collection.platform, collection.id)
        references[collection.id] = {
            "id": collection.id,
            "path": str(path),
            "sha256": _sha(snapshot.source_bytes[path]),
            "memberships_path": str(members),
            "memberships_sha256": _sha(snapshot.source_bytes[members]),
        }
    containing: dict[str, set[str]] = {}
    for member in snapshot.memberships.values():
        containing.setdefault(member.emoji_id, set()).add(member.collection_id)
    rows = []
    entities: Sequence[Collection | Emoji] = [
        *snapshot.collections.values(),
        *snapshot.emojis.values(),
    ]
    for entity in entities:
        if isinstance(entity, Collection):
            path = collection_path(entity.platform, entity.id)
            collection_ids = [entity.id]
        else:
            path = emoji_bucket_path(entity.platform, entity.id)
            collection_ids = sorted(containing.get(entity.id, set()))
        payload = _json(
            {
                "id": entity.id,
                "path": str(path),
                "sha256": _sha(snapshot.source_bytes[path]),
                "collections": [references[identifier] for identifier in collection_ids],
            }
        )
        if len(payload) > _MAX_RECORD_BYTES:
            raise ValueError("lookup record exceeds the bounded index limit")
        rows.append((entity.id, payload, _sha(payload.encode())))
    metadata = {
        **expected,
        "manifest_sha256": _sha(snapshot.source_bytes[PurePosixPath("dataset.json")]),
        "row_count": str(len(rows)),
    }
    return metadata, rows


def _read_cache(
    path: Path, expected: Mapping[str, str], selectors: Sequence[str]
) -> tuple[dict[str, str], list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    if not path.is_file() or not 0 < path.stat().st_size <= _MAX_INDEX_BYTES:
        raise ValueError("unsafe lookup index size")
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("lookup index has an unfinished SQLite transaction")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
        deadline = time.monotonic() + 2.0
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_RECORD_BYTES)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        metadata = dict(connection.execute("SELECT key, value FROM metadata LIMIT 7"))
        if set(metadata) != {*expected, "manifest_sha256", "row_count"}:
            raise ValueError("invalid lookup index metadata")
        if any(metadata[key] != value for key, value in expected.items()):
            return None
        if connection.execute("SELECT count(*) FROM entries").fetchone()[0] != int(
            metadata["row_count"]
        ):
            raise ValueError("incomplete lookup index")
        records = []
        for identifier in sorted(set(selectors)):
            row = connection.execute(
                "SELECT payload, sha256 FROM entries WHERE id=?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("selector is missing from the lookup index")
            payload, digest = row
            if not isinstance(payload, str) or len(payload) > _MAX_RECORD_BYTES:
                raise ValueError("invalid lookup record size")
            if _sha(payload.encode()) != digest:
                raise ValueError("corrupt lookup record")
            record = parse_json(payload.encode(), source="lookup index")
            if set(record) != {"id", "path", "sha256", "collections"} or record["id"] != identifier:
                raise ValueError("lookup record does not match the selector")
            if _json(record) != payload:
                raise ValueError("lookup record is not canonical")
            records.append(record)
        return metadata, records


def _write_cache(path: Path, metadata: Mapping[str, str], rows: list[tuple[str, str, str]]) -> None:
    assert_no_link_or_reparse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".mojilex-lookup-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE entries (id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "sha256 TEXT NOT NULL)"
            )
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            connection.executemany("INSERT INTO entries VALUES (?, ?, ?)", rows)
        if temporary.stat().st_size > _MAX_INDEX_BYTES:
            raise ValueError("lookup index exceeds the bounded cache size")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        assert_no_link_or_reparse(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_entity_path(path: object, identifier: str) -> None:
    if not _ID.fullmatch(identifier):
        raise ValueError("invalid indexed entity identifier")
    relative = (
        collection_path("p", identifier)
        if identifier.startswith("mxc_")
        else emoji_bucket_path("p", identifier)
    )
    suffix = str(relative).removeprefix("data/p/")
    if not isinstance(path, str) or not re.fullmatch(
        r"data/[a-z][a-z0-9_-]{0,63}/" + re.escape(suffix), path
    ):
        raise ValueError("indexed entity path is not canonical")


def _read_subset(
    root: Path,
    git: GitRunner,
    head: str,
    metadata: Mapping[str, str],
    records: list[dict[str, Any]],
) -> DatasetSnapshot:
    source: dict[PurePosixPath, bytes] = {}

    def read(path_value: str, digest: str) -> bytes:
        if (
            not isinstance(path_value, str)
            or not isinstance(digest, str)
            or not _SHA.fullmatch(digest)
        ):
            raise ValueError("invalid lookup path/hash")
        path = PurePosixPath(path_value)
        if len(path_value) > 256 or str(path) != path_value or "\\" in path_value:
            raise ValueError("invalid canonical path")
        if path not in source:
            absolute = safe_destination(root, path)
            if not absolute.is_file() or not 0 <= absolute.stat().st_size <= _MAX_FILE_BYTES:
                raise ValueError("canonical file exceeds bounded lookup size")
            with absolute.open("rb") as stream:
                data = stream.read(_MAX_FILE_BYTES + 1)
            if len(data) > _MAX_FILE_BYTES:
                raise ValueError("canonical file grew beyond the bounded lookup size")
            if _sha(data) != digest:
                raise ValueError("indexed content has changed")
            _verify_head_blob(git, head, path, data)
            source[path] = data
        elif _sha(source[path]) != digest:
            raise ValueError("conflicting lookup hashes")
        return source[path]

    manifest = parse_json(read("dataset.json", metadata["manifest_sha256"]), source="dataset.json")
    result = DatasetSnapshot(root, manifest)
    for record in records:
        identifier = record["id"]
        path = record["path"]
        _check_entity_path(path, identifier)
        data = read(path, record["sha256"])
        if identifier.startswith("mxc_"):
            collection = Collection.model_validate(parse_json(data, source="lookup collection"))
            if (
                collection.id != identifier
                or str(collection_path(collection.platform, identifier)) != path
            ):
                raise ValueError("indexed collection ID/path mismatch")
            result.collections[identifier] = collection
        else:
            emojis = [
                Emoji.model_validate(raw) for raw in parse_jsonl(data, source="lookup bucket")
            ]
            if len({item.id for item in emojis}) != len(emojis) or any(
                str(emoji_bucket_path(item.platform, item.id)) != path for item in emojis
            ):
                raise ValueError("indexed emoji bucket is not canonical")
            matches = [item for item in emojis if item.id == identifier]
            if len(matches) != 1:
                raise ValueError("indexed emoji ID is missing or duplicated")
            result.emojis[identifier] = matches[0]
        refs = record["collections"]
        if not isinstance(refs, list) or len(refs) > 100_000:
            raise ValueError("invalid indexed collection references")
        collection_ids = []
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {
                "id",
                "path",
                "sha256",
                "memberships_path",
                "memberships_sha256",
            }:
                raise ValueError("invalid indexed collection reference")
            if not isinstance(ref["id"], str) or not ref["id"].startswith("mxc_"):
                raise ValueError("referenced entity is not a collection")
            _check_entity_path(ref["path"], ref["id"])
            value = Collection.model_validate(
                parse_json(read(ref["path"], ref["sha256"]), source="lookup collection")
            )
            if (
                value.id != ref["id"]
                or str(collection_path(value.platform, value.id)) != ref["path"]
            ):
                raise ValueError("referenced collection ID/path mismatch")
            if str(memberships_path(value.platform, value.id)) != ref["memberships_path"]:
                raise ValueError("referenced membership path mismatch")
            members = [
                Membership.model_validate(raw)
                for raw in parse_jsonl(
                    read(ref["memberships_path"], ref["memberships_sha256"]),
                    source="lookup memberships",
                )
            ]
            if len({item.id for item in members}) != len(members) or any(
                item.collection_id != value.id
                or item.id != membership_id(item.collection_id, item.emoji_id)
                for item in members
            ):
                raise ValueError("referenced memberships are invalid")
            if identifier.startswith("mxe_") and not any(
                item.emoji_id == identifier for item in members
            ):
                raise ValueError("indexed emoji is not in its referenced collection")
            result.collections[value.id] = value
            result.memberships.update({item.id: item for item in members})
            collection_ids.append(value.id)
        if collection_ids != sorted(set(collection_ids)) or (
            identifier.startswith("mxc_") and collection_ids != [identifier]
        ):
            raise ValueError("invalid indexed collection order")
    # Deliberately empty: this is not an editable/full canonical snapshot.
    return result

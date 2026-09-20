"""Load and materialize canonical MojiLex repository state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from mojilex_cli.domain.models import Collection, Emoji, Membership, Tombstone, VisualRelation

from .layout import (
    assert_no_link_or_reparse,
    collection_path,
    emoji_bucket_path,
    legacy_bucket_path,
    memberships_path,
    tombstone_path,
    visual_relations_path,
)
from .serialization import (
    parse_json,
    parse_jsonl,
    pretty_json,
    serialization_scope,
    serialize_collection,
    serialize_emojis,
    serialize_memberships,
    serialize_tombstone,
    serialize_visual_relations,
)
from .transaction import locked_dataset_transaction_view

_MODEL_CACHE_BYTES = 16 * 1024 * 1024
_MODEL_CACHE_ENTRIES = 8192
_Model = TypeVar("_Model", bound=BaseModel)


class _ModelMemo:
    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[type[BaseModel], bytes, bool], tuple[BaseModel, ...]] = (
            OrderedDict()
        )
        self.size = 0
        self.lock = Lock()


_MODEL_MEMO: ContextVar[_ModelMemo | None] = ContextVar("dataset_model_memo", default=None)


@contextmanager
def dataset_read_scope() -> Iterator[None]:
    """Reuse exact bytes within one operation, never filesystem metadata or mutable results."""
    if _MODEL_MEMO.get() is not None:
        yield
        return
    token = _MODEL_MEMO.set(_ModelMemo())
    try:
        with serialization_scope():
            yield
    finally:
        _MODEL_MEMO.reset(token)


def _read_models(data: bytes, model: type[_Model], *, source: str, many: bool) -> list[_Model]:
    # Each caller has freshly read and checked the file before reaching this cache.
    # The model distinguishes JSON/JSONL record kinds; only validated values enter.
    memo = _MODEL_MEMO.get()
    key = (model, data, many)
    if memo is not None:
        with memo.lock:
            cached = memo.entries.get(key)
            if cached is not None:
                memo.entries.move_to_end(key)
                return [cast(_Model, value.model_copy(deep=True)) for value in cached]
    values = parse_jsonl(data, source=source) if many else [parse_json(data, source=source)]
    result = [model.model_validate(raw) for raw in values]
    if memo is not None and len(data) <= _MODEL_CACHE_BYTES:
        saved = tuple(item.model_copy(deep=True) for item in result)
        with memo.lock:
            if key not in memo.entries:
                while memo.entries and (
                    memo.size + len(data) > _MODEL_CACHE_BYTES
                    or len(memo.entries) >= _MODEL_CACHE_ENTRIES
                ):
                    old_key, _ = memo.entries.popitem(last=False)
                    memo.size -= len(old_key[1])
                memo.entries[key] = saved
                memo.size += len(data)
    return result


class DatasetLoadError(ValueError):
    pass


@dataclass(slots=True)
class DatasetSnapshot:
    root: Path
    manifest: dict[str, Any]
    collections: dict[str, Collection] = field(default_factory=dict)
    emojis: dict[str, Emoji] = field(default_factory=dict)
    memberships: dict[str, Membership] = field(default_factory=dict)
    relations: dict[str, VisualRelation] = field(default_factory=dict)
    tombstones: dict[str, Tombstone] = field(default_factory=dict)
    source_bytes: dict[PurePosixPath, bytes] = field(default_factory=dict)

    def clone(self) -> DatasetSnapshot:
        return DatasetSnapshot(
            root=self.root,
            manifest=deepcopy(self.manifest),
            collections={
                key: value.model_copy(deep=True) for key, value in self.collections.items()
            },
            emojis={key: value.model_copy(deep=True) for key, value in self.emojis.items()},
            memberships={
                key: value.model_copy(deep=True) for key, value in self.memberships.items()
            },
            relations={key: value.model_copy(deep=True) for key, value in self.relations.items()},
            tombstones={key: value.model_copy(deep=True) for key, value in self.tombstones.items()},
            source_bytes=dict(self.source_bytes),
        )

    def bucket_source_paths(self) -> dict[str, PurePosixPath]:
        """Locate records in valid current or legacy buckets without rewriting input."""
        locations: dict[str, PurePosixPath] = {}
        for path, data in self.source_bytes.items():
            if len(path.parts) != 5 or path.suffix != ".jsonl":
                continue
            if path.parts[0] != "data" or not (
                path.parts[2] == "emojis" or path.parts[1:3] == ("relations", "visual")
            ):
                continue
            try:
                records = parse_jsonl(data, source=str(path))
            except ValueError:
                continue  # Canonical validation will report the malformed source bytes.
            for raw in records:
                identifier = raw.get("id")
                if not isinstance(identifier, str):
                    continue
                emoji = self.emojis.get(identifier)
                if emoji is not None:
                    expected = emoji_bucket_path(emoji.platform, identifier)
                elif identifier in self.relations:
                    expected = visual_relations_path(identifier)
                else:
                    continue
                if path in (expected, legacy_bucket_path(expected)):
                    locations[identifier] = path
        return locations

    def to_files(self, *, preserve_legacy_paths: bool = False) -> dict[PurePosixPath, bytes]:
        """Serialize new eight-hex buckets; optionally validate the original storage layout."""
        source_paths = self.bucket_source_paths() if preserve_legacy_paths else {}
        files: dict[PurePosixPath, bytes] = {
            PurePosixPath("dataset.json"): pretty_json(self.manifest).encode("utf-8")
        }
        memberships_by_collection: dict[str, list[Membership]] = {
            collection_id: [] for collection_id in self.collections
        }
        for membership in self.memberships.values():
            memberships_by_collection.setdefault(membership.collection_id, []).append(membership)
        for collection in self.collections.values():
            files[collection_path(collection.platform, collection.id)] = serialize_collection(
                collection
            )
            files[memberships_path(collection.platform, collection.id)] = serialize_memberships(
                memberships_by_collection.get(collection.id, [])
            )
        buckets: dict[PurePosixPath, list[Emoji]] = {}
        for emoji in self.emojis.values():
            path = source_paths.get(emoji.id, emoji_bucket_path(emoji.platform, emoji.id))
            buckets.setdefault(path, []).append(emoji)
        for path, emojis in buckets.items():
            files[path] = serialize_emojis(emojis)
        relation_buckets: dict[PurePosixPath, list[VisualRelation]] = {}
        for relation in self.relations.values():
            path = source_paths.get(relation.id, visual_relations_path(relation.id))
            relation_buckets.setdefault(path, []).append(relation)
        for path, relations in relation_buckets.items():
            files[path] = serialize_visual_relations(relations)
        for tombstone in self.tombstones.values():
            files[tombstone_path(tombstone.target_id)] = serialize_tombstone(tombstone)
        return files


def _read(path: Path, root: Path, source: dict[PurePosixPath, bytes]) -> bytes:
    try:
        assert_no_link_or_reparse(path, boundary=root)
    except ValueError as exc:
        raise DatasetLoadError(str(exc)) from exc
    # The guard above already checks lexical containment and every component
    # (including root and the file) for links/reparse points. Repeating resolve()
    # and another ancestor is_symlink() walk adds several stats per record on
    # Windows without strengthening that same check. read_bytes still verifies
    # that the file exists and is readable; no filesystem metadata is cached.
    data = path.read_bytes()
    source[PurePosixPath(path.relative_to(root).as_posix())] = data
    return data


def _insert(target: dict[str, Any], entity: Any, source_path: Path) -> None:
    entity_id = entity.target_id if isinstance(entity, Tombstone) else entity.id
    if entity_id in target:
        raise DatasetLoadError(f"duplicate entity ID {entity_id} at {source_path}")
    target[entity_id] = entity


def load_dataset(root: str | Path, *, allow_missing_fingerprints: bool = False) -> DatasetSnapshot:
    """Load canonical data, optionally exposing absent legacy fingerprints as staging only.

    The opt-in never repairs malformed existing fingerprints or changes source bytes;
    its partial records must be verified and canonically validated before publication.
    """
    unresolved_root = Path(root)
    try:
        assert_no_link_or_reparse(unresolved_root)
    except ValueError as exc:
        raise DatasetLoadError(str(exc)) from exc
    root_path = unresolved_root.resolve()
    with locked_dataset_transaction_view(root_path):
        return _load_dataset_unlocked(
            root_path, allow_missing_fingerprints=allow_missing_fingerprints
        )


def _load_dataset_unlocked(
    root_path: Path, *, allow_missing_fingerprints: bool = False
) -> DatasetSnapshot:
    manifest_path = root_path / "dataset.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DatasetLoadError(f"missing or unsafe dataset manifest: {manifest_path}")
    source: dict[PurePosixPath, bytes] = {}
    try:
        manifest = parse_json(_read(manifest_path, root_path, source), source="dataset.json")
        snapshot = DatasetSnapshot(root=root_path, manifest=manifest, source_bytes=source)
        data_root = root_path / "data"
        if data_root.exists():
            assert_no_link_or_reparse(data_root, boundary=root_path)
            for collection_file in sorted(data_root.glob("*/collections/*/*/collection.json")):
                collection = _read_models(
                    _read(collection_file, root_path, source),
                    Collection,
                    source=str(collection_file),
                    many=False,
                )[0]
                _insert(snapshot.collections, collection, collection_file)
                memberships_file = collection_file.with_name("memberships.jsonl")
                if not memberships_file.is_file() or memberships_file.is_symlink():
                    raise DatasetLoadError(
                        f"missing or unsafe memberships file: {memberships_file}"
                    )
                for membership in _read_models(
                    _read(memberships_file, root_path, source),
                    Membership,
                    source=str(memberships_file),
                    many=True,
                ):
                    _insert(snapshot.memberships, membership, memberships_file)
            for bucket_file in sorted(data_root.glob("*/emojis/*/*.jsonl")):
                if not allow_missing_fingerprints:
                    for emoji in _read_models(
                        _read(bucket_file, root_path, source),
                        Emoji,
                        source=str(bucket_file),
                        many=True,
                    ):
                        _insert(snapshot.emojis, emoji, bucket_file)
                    continue
                for raw in parse_jsonl(
                    _read(bucket_file, root_path, source), source=str(bucket_file)
                ):
                    if allow_missing_fingerprints and "fingerprints" not in raw:
                        # Only the explicit backfill command may load this local staging
                        # shape. No media hashes or successful fingerprints are invented.
                        from mojilex_cli.domain import Media, media_digest

                        if "dedupe_profile" not in manifest or not isinstance(
                            raw.get("media"), list
                        ):
                            raise DatasetLoadError(
                                "legacy fingerprint backfill requires pinned profile and media"
                            )
                        raw = dict(raw)
                        raw["fingerprints"] = {
                            "status": "partial",
                            "profile": manifest["dedupe_profile"],
                            "input_media_digest": media_digest(
                                [Media.model_validate(item) for item in raw["media"]]
                            ),
                            "items": [],
                        }
                    emoji = Emoji.model_validate(raw)
                    _insert(snapshot.emojis, emoji, bucket_file)
            relation_root = data_root / "relations" / "visual"
            if relation_root.exists():
                for relation_file in sorted(relation_root.glob("*/*.jsonl")):
                    for relation in _read_models(
                        _read(relation_file, root_path, source),
                        VisualRelation,
                        source=str(relation_file),
                        many=True,
                    ):
                        _insert(snapshot.relations, relation, relation_file)
        tombstone_root = root_path / "tombstones"
        if tombstone_root.exists():
            assert_no_link_or_reparse(tombstone_root, boundary=root_path)
            for tombstone_file in sorted(tombstone_root.glob("*/*.json")):
                tombstone = _read_models(
                    _read(tombstone_file, root_path, source),
                    Tombstone,
                    source=str(tombstone_file),
                    many=False,
                )[0]
                _insert(snapshot.tombstones, tombstone, tombstone_file)
        return snapshot
    except (OSError, ValueError, ValidationError) as exc:
        if isinstance(exc, DatasetLoadError):
            raise
        raise DatasetLoadError(str(exc)) from exc

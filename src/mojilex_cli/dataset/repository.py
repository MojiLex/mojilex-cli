"""Load and materialize canonical MojiLex repository state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from mojilex_cli.domain.models import Collection, Emoji, Membership, Tombstone, VisualRelation

from .layout import (
    assert_no_link_or_reparse,
    collection_path,
    emoji_bucket_path,
    memberships_path,
    tombstone_path,
    visual_relations_path,
)
from .serialization import (
    parse_json,
    parse_jsonl,
    pretty_json,
    serialize_collection,
    serialize_emojis,
    serialize_memberships,
    serialize_tombstone,
    serialize_visual_relations,
)
from .transaction import locked_dataset_transaction_view


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

    def to_files(self) -> dict[PurePosixPath, bytes]:
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
            buckets.setdefault(emoji_bucket_path(emoji.platform, emoji.id), []).append(emoji)
        for path, emojis in buckets.items():
            files[path] = serialize_emojis(emojis)
        relation_buckets: dict[PurePosixPath, list[VisualRelation]] = {}
        for relation in self.relations.values():
            relation_buckets.setdefault(visual_relations_path(relation.id), []).append(relation)
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
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise DatasetLoadError(f"path escapes dataset root: {path}") from exc
    current = path
    while current != root:
        if current.is_symlink():
            raise DatasetLoadError(f"symlink is forbidden: {current}")
        current = current.parent
    if path.is_symlink():
        raise DatasetLoadError(f"symlink is forbidden: {path}")
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
                value = parse_json(
                    _read(collection_file, root_path, source), source=str(collection_file)
                )
                collection = Collection.model_validate(value)
                _insert(snapshot.collections, collection, collection_file)
                memberships_file = collection_file.with_name("memberships.jsonl")
                if not memberships_file.is_file() or memberships_file.is_symlink():
                    raise DatasetLoadError(
                        f"missing or unsafe memberships file: {memberships_file}"
                    )
                for raw in parse_jsonl(
                    _read(memberships_file, root_path, source), source=str(memberships_file)
                ):
                    membership = Membership.model_validate(raw)
                    _insert(snapshot.memberships, membership, memberships_file)
            for bucket_file in sorted(data_root.glob("*/emojis/*/*.jsonl")):
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
                    for raw in parse_jsonl(
                        _read(relation_file, root_path, source), source=str(relation_file)
                    ):
                        relation = VisualRelation.model_validate(raw)
                        _insert(snapshot.relations, relation, relation_file)
        tombstone_root = root_path / "tombstones"
        if tombstone_root.exists():
            assert_no_link_or_reparse(tombstone_root, boundary=root_path)
            for tombstone_file in sorted(tombstone_root.glob("*/*.json")):
                raw = parse_json(
                    _read(tombstone_file, root_path, source), source=str(tombstone_file)
                )
                tombstone = Tombstone.model_validate(raw)
                _insert(snapshot.tombstones, tombstone, tombstone_file)
        return snapshot
    except (OSError, ValueError, ValidationError) as exc:
        if isinstance(exc, DatasetLoadError):
            raise
        raise DatasetLoadError(str(exc)) from exc

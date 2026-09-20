"""Schema, integrity, policy, canonical, no-media, and secret validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import unicodedata
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any
from uuid import UUID

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from mojilex_cli.analysis import AnalysisError, profile_sha256
from mojilex_cli.domain.hashes import (
    media_digest,
    reviewed_content_sha256,
    reviewed_relation_sha256,
    telegram_set_fingerprint,
)
from mojilex_cli.domain.ids import (
    DUPLICATE_GROUP_NAMESPACE,
    RIGHTS_ASSIGNMENT_NAMESPACE,
    VISUAL_RELATION_NAMESPACE,
    collection_id,
    emoji_id,
    membership_id,
    visual_relation_id,
)
from mojilex_cli.domain.models import (
    AvailabilityStatus,
    ColorFamily,
    ContentType,
    FingerprintStatus,
    MediaRole,
    MembershipStatus,
    PaletteDynamics,
    ProvenanceOrigin,
    ReviewStatus,
    Style,
    SuggestedUse,
    TextDynamics,
    TextKind,
    TextRecognitionStatus,
    Uncertainty,
    VisualRelationScope,
    VisualRelationType,
)

from .layout import (
    collection_path,
    emoji_bucket_path,
    memberships_path,
    tombstone_path,
    visual_relations_path,
)
from .repository import DatasetLoadError, DatasetSnapshot, dataset_read_scope, load_dataset
from .schema_cache import SchemaSuccessCache
from .serialization import parse_json, pretty_json

_SECRET_PATTERNS = {
    "telegram_bot_token": re.compile(
        rb"(?<![0-9A-Za-z_-])\d{6,12}:[A-Za-z0-9_-]{30,}(?![0-9A-Za-z_-])"
    ),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "telegram_download_url": re.compile(rb"https://api\.telegram\.org/file/bot", re.I),
}
_BINARY_SUFFIXES = {
    ".webp",
    ".webm",
    ".tgs",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".avif",
    ".bmp",
    ".ico",
    ".svg",
    ".tif",
    ".tiff",
    ".mp4",
    ".mov",
    ".mkv",
    ".mp3",
    ".wav",
    ".pdf",
    ".zip",
    ".gz",
    ".7z",
    ".rar",
}
_FORBIDDEN_LOCAL_ARTIFACT = re.compile(
    r"^(?:(?:dedupe|near|collection)[-_])?candidates?(?:[-_]v\d+)?\."
    r"(?:json|jsonl|db|sqlite|sqlite3)$"
    r"|^(?:(?:dedupe|local)[-_])?lsh[-_](?:index|buckets?)\."
    r"(?:json|jsonl|db|sqlite|sqlite3)$"
    r"|^(?:dedupe[-_])?previews?(?:[-_]metadata)?\."
    r"(?:json|jsonl|html)$",
    re.I,
)
_FORBIDDEN_LOCAL_DIRECTORIES = {
    ".mojilex-cache",
    "candidate-previews",
    "dedupe-candidates",
    "dedupe-index",
    "lsh-index",
}
_LOCAL_TRANSACTION_DIRECTORIES = {".mojilex", ".mojilex-atomic-write"}
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_MEDIA_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"\x1aE\xdf\xa3",  # EBML/WebM
    b"\x1f\x8b",  # Telegram TGS is gzip
)
_FORBIDDEN_PERSISTED_KEYS = {
    "api_key",
    "authorization",
    "download_url",
    "file_id",
    "file_path",
    "local_path",
    "password",
    "secret",
    "token",
}
_FILE_SCAN_CHUNK_SIZE = 64 * 1024
_FILE_SCAN_OVERLAP = 4 * 1024
_FILE_SCAN_PREFIX_SIZE = max(len(_LFS_HEADER), 12, *(len(magic) for magic in _MEDIA_MAGIC))


@dataclass(frozen=True, slots=True, order=True)
class ValidationIssue:
    code: str
    path: str
    message: str


class DatasetValidationError(ValueError):
    """A stable public failure for an invalid canonical dataset."""

    code = "VALIDATION_FAILED"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            summary = "\n".join(f"{item.code} {item.path}: {item.message}" for item in self.issues)
            raise DatasetValidationError(f"dataset validation failed:\n{summary}")


def _issue(issues: list[ValidationIssue], code: str, path: str, message: str) -> None:
    issues.append(ValidationIssue(code, path, message))


_SCHEMA_MEMO_LIMIT = 65_536
_PERSISTED_MEMO_BYTES = 32 * 1024 * 1024
_PERSISTED_MEMO_ENTRIES = 65_536


class _SchemaMemo:
    def __init__(self, persistent_cache: Path | None = None) -> None:
        self.persistent = (
            SchemaSuccessCache(persistent_cache) if persistent_cache is not None else None
        )
        self.successes: dict[bytes, None] = (
            dict.fromkeys(self.persistent.loaded) if self.persistent is not None else {}
        )
        self.graphs: dict[bytes, tuple[Any, dict[str, dict[str, Any]]]] = {}
        self.persisted: OrderedDict[tuple[tuple[object, ...], bytes], None] = OrderedDict()
        self.persisted_bytes = 0
        self.lock = Lock()


_SCHEMA_MEMO: ContextVar[_SchemaMemo | None] = ContextVar("mojilex_schema_memo", default=None)


@contextmanager
def schema_validation_scope(*, persistent_cache: Path | None = None) -> Iterator[None]:
    """Reuse exact checks in this operation, optionally with authenticated local history."""
    if _SCHEMA_MEMO.get() is not None:
        yield
        return
    memo = _SchemaMemo(persistent_cache)
    token = _SCHEMA_MEMO.set(memo)
    try:
        with dataset_read_scope():
            yield
    finally:
        _SCHEMA_MEMO.reset(token)
        with memo.lock:
            memo.persisted.clear()
            memo.persisted_bytes = 0
        if memo.persistent is not None:
            memo.persistent.close(memo.successes)


def _json_native(value: object) -> bool:
    if type(value) in {str, int, float, bool, type(None)}:
        return True
    if type(value) is list:
        return all(_json_native(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_native(item) for key, item in value.items())
    return False


def _schema_instance_key(graph: bytes, name: str, instance: object) -> bytes | None:
    try:
        if not _json_native(instance):
            return None
        payload = json.dumps(
            instance, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None  # Non-JSON inputs still undergo the original validation.
    return hashlib.sha256(graph + name.encode("utf-8") + b"\0" + payload).digest()


def _schema_store(schema_root: Path) -> tuple[Any, dict[str, dict[str, Any]], bytes]:
    registry: Any = Registry()
    by_name: dict[str, dict[str, Any]] = {}
    graph = hashlib.sha256()
    from .serialization import parse_json

    documents = [(path, path.read_bytes()) for path in sorted(schema_root.rglob("*.json"))]
    for path, raw in documents:
        uri = path.as_uri().encode("utf-8")
        # Include registry URI bindings and exact bytes read, avoiding a second read.
        graph.update(len(uri).to_bytes(8, "big") + uri + len(raw).to_bytes(8, "big") + raw)
    digest = graph.digest()
    memo = _SCHEMA_MEMO.get()
    if memo is not None:
        with memo.lock:
            cached = memo.graphs.get(digest)
            if cached is not None:
                return cached[0], cached[1], digest
    for path, raw in documents:
        schema = parse_json(raw, source=str(path))
        by_name[path.name] = schema
        resource = Resource.from_contents(schema, default_specification=DRAFT202012)
        registry = registry.with_resource(path.as_uri(), resource)
        registry = registry.with_resource(path.name, resource)
        schema_id = schema.get("$id")
        if isinstance(schema_id, str):
            registry = registry.with_resource(schema_id, resource)
    if memo is not None:
        with memo.lock:
            if digest not in memo.graphs and len(memo.graphs) >= 8:
                memo.graphs.pop(next(iter(memo.graphs)))
            memo.graphs[digest] = (registry, by_name)
    return registry, by_name, digest


def _validate_json_schemas(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    schema_root = snapshot.root / "schemas" / "v1"
    if not schema_root.is_dir():
        _issue(issues, "SCHEMA_MISSING", "schemas/v1", "schema v1 directory is required")
        return
    try:
        registry, schemas, graph = _schema_store(schema_root)
    except (OSError, ValueError) as exc:
        _issue(issues, "SCHEMA_INVALID", "schemas/v1", str(exc))
        return
    targets: list[tuple[str, str, dict[str, Any]]] = [
        ("dataset.json", "dataset.schema.json", snapshot.manifest)
    ]
    targets.extend(
        (str(collection_path(value.platform, value.id)), "collection.schema.json", value.as_dict())
        for value in snapshot.collections.values()
    )
    targets.extend(
        (str(emoji_bucket_path(value.platform, value.id)), "emoji.schema.json", value.as_dict())
        for value in snapshot.emojis.values()
    )
    targets.extend(
        (
            str(
                memberships_path(
                    snapshot.collections[value.collection_id].platform, value.collection_id
                )
            ),
            "membership.schema.json",
            value.as_dict(),
        )
        for value in snapshot.memberships.values()
        if value.collection_id in snapshot.collections
    )
    targets.extend(
        (str(tombstone_path(value.target_id)), "tombstone.schema.json", value.as_dict())
        for value in snapshot.tombstones.values()
    )
    targets.extend(
        (
            str(visual_relations_path(value.id)),
            "visual-relation.schema.json",
            value.as_dict(),
        )
        for value in snapshot.relations.values()
    )
    validators: dict[str, Any] = {}
    memo = _SCHEMA_MEMO.get()
    for path, schema_name, instance in targets:
        schema = schemas.get(schema_name)
        if schema is None:
            _issue(
                issues, "SCHEMA_MISSING", f"schemas/v1/{schema_name}", "required schema is absent"
            )
            continue
        key = _schema_instance_key(graph, schema_name, instance) if memo is not None else None
        if memo is not None and key is not None:
            with memo.lock:
                if key in memo.successes:
                    continue
        try:
            validator = validators.get(schema_name)
            if validator is None:
                Draft202012Validator.check_schema(schema)
                validator = Draft202012Validator(schema, registry=registry)
                validators[schema_name] = validator
            errors = sorted(
                validator.iter_errors(instance), key=lambda item: list(item.absolute_path)
            )
            if not errors and memo is not None and key is not None:
                with memo.lock:
                    if key not in memo.successes and len(memo.successes) >= _SCHEMA_MEMO_LIMIT:
                        memo.successes.pop(next(iter(memo.successes)))
                    memo.successes[key] = None
            for error in errors:
                suffix = "/".join(str(part) for part in error.absolute_path)
                _issue(issues, "SCHEMA", f"{path}/{suffix}".rstrip("/"), error.message)
        except (SchemaError, OSError, ValueError) as exc:
            _issue(issues, "SCHEMA_INVALID", f"schemas/v1/{schema_name}", str(exc))


def _validate_manifest(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> UUID | None:
    manifest = snapshot.manifest
    required = {
        "dataset": "mojilex",
        "schema_version": "1.0.0",
        "canonical_repository": "https://github.com/MojiLex/mojilex",
        "visual_relation_namespace": str(VISUAL_RELATION_NAMESPACE),
        "duplicate_group_namespace": str(DUPLICATE_GROUP_NAMESPACE),
        "rights_assignment_namespace": str(RIGHTS_ASSIGNMENT_NAMESPACE),
        "taxonomy_version": "1.0.0",
        "color_profile": "color-v1",
        "dedupe_profile": "dedupe-v1",
        "collection_dedupe_profile": "collection-dedupe-v1",
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            _issue(issues, "MANIFEST", f"dataset.json/{field}", f"must equal {expected!r}")
    for field in (
        "color_profile_sha256",
        "dedupe_profile_sha256",
        "collection_dedupe_profile_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            _issue(issues, "MANIFEST", f"dataset.json/{field}", "must be lowercase SHA-256")
    languages = manifest.get("default_languages")
    if not isinstance(languages, list) or not {"ru", "en"}.issubset(languages):
        _issue(issues, "MANIFEST", "dataset.json/default_languages", "ru and en are required")
    platforms = manifest.get("platforms")
    if not isinstance(platforms, list) or "telegram" not in platforms:
        _issue(issues, "MANIFEST", "dataset.json/platforms", "telegram is required for MVP")
    rights_defaults = manifest.get("rights_defaults")
    if rights_defaults != {"project_profile_id": "mojilex-metadata-only-v1"}:
        _issue(
            issues,
            "MANIFEST",
            "dataset.json/rights_defaults",
            "must select mojilex-metadata-only-v1",
        )
    licenses = manifest.get("licenses")
    if licenses != {"data": "CC0-1.0", "code": "MIT"}:
        _issue(
            issues, "MANIFEST", "dataset.json/licenses", "must declare CC0-1.0 data and MIT code"
        )
    try:
        return UUID(str(manifest["id_namespace"]))
    except (KeyError, ValueError):
        _issue(issues, "MANIFEST", "dataset.json/id_namespace", "must be a UUID")
        return None


def _validate_ids_and_paths(
    snapshot: DatasetSnapshot,
    namespace: UUID | None,
    issues: list[ValidationIssue],
    *,
    canonical: bool,
) -> None:
    identities: dict[tuple[Any, ...], str] = {}
    expected_files = snapshot.to_files(preserve_legacy_paths=canonical)
    for collection in snapshot.collections.values():
        path = collection_path(collection.platform, collection.id)
        if namespace is not None:
            expected = collection_id(
                collection.platform,
                collection.native_namespace,
                collection.scope_id,
                collection.native_id,
                collection.identity_epoch,
                namespace=namespace,
            )
            if collection.id != expected:
                _issue(issues, "ID", str(path), f"ID must be {expected}")
        identity = (
            "collection",
            collection.platform,
            collection.native_namespace,
            collection.scope_id,
            collection.native_id,
            collection.identity_epoch,
        )
        if identity in identities:
            _issue(issues, "IDENTITY_DUPLICATE", str(path), f"also used by {identities[identity]}")
        identities[identity] = collection.id
    for emoji in snapshot.emojis.values():
        path = emoji_bucket_path(emoji.platform, emoji.id)
        if namespace is not None:
            expected = emoji_id(
                emoji.platform,
                emoji.native_namespace,
                emoji.scope_id,
                emoji.native_id,
                emoji.identity_epoch,
                namespace=namespace,
            )
            if emoji.id != expected:
                _issue(issues, "ID", str(path), f"ID must be {expected}")
        identity = (
            "emoji",
            emoji.platform,
            emoji.native_namespace,
            emoji.scope_id,
            emoji.native_id,
            emoji.identity_epoch,
        )
        if identity in identities:
            _issue(issues, "IDENTITY_DUPLICATE", str(path), f"also used by {identities[identity]}")
        identities[identity] = emoji.id
    for membership in snapshot.memberships.values():
        if namespace is not None:
            expected = membership_id(
                membership.collection_id, membership.emoji_id, namespace=namespace
            )
            if membership.id != expected:
                _issue(issues, "ID", membership.id, f"ID must be {expected}")
    for relation in snapshot.relations.values():
        path = visual_relations_path(relation.id)
        pair = relation.evidence.media_pairs[0]
        if relation.scope is VisualRelationScope.MEDIA_PAIR:
            expected = visual_relation_id(
                relation.subject_id,
                relation.object_id,
                relation.scope.value,
                subject_role=pair.subject_role.value,
                subject_variant_id=pair.subject_variant_id,
                object_role=pair.object_role.value,
                object_variant_id=pair.object_variant_id,
                identity_epoch=relation.identity_epoch,
            )
        else:
            expected = visual_relation_id(
                relation.subject_id,
                relation.object_id,
                relation.scope.value,
                identity_epoch=relation.identity_epoch,
            )
        if relation.id != expected:
            _issue(issues, "RELATION_ID", str(path), f"ID must be {expected}")
    for path, expected_bytes in expected_files.items():
        if canonical and path not in snapshot.source_bytes:
            _issue(issues, "PATH", str(path), "entity is stored at a non-canonical path")
        elif canonical and snapshot.source_bytes[path] != expected_bytes:
            _issue(issues, "CANONICAL", str(path), "file is not canonically serialized or sorted")
    if canonical:
        manifest_bytes = pretty_json(snapshot.manifest).encode("utf-8")
        if snapshot.source_bytes.get(PurePosixPath("dataset.json")) != manifest_bytes:
            _issue(issues, "CANONICAL", "dataset.json", "manifest is not canonical")
        actual_paths: set[PurePosixPath] = set()
        for directory in (snapshot.root / "data", snapshot.root / "tombstones"):
            if not directory.exists():
                continue
            for source_path in directory.rglob("*"):
                if source_path.is_file() and source_path.suffix in {".json", ".jsonl"}:
                    actual_paths.add(
                        PurePosixPath(source_path.relative_to(snapshot.root).as_posix())
                    )
        for extra in sorted(actual_paths - set(expected_files), key=str):
            _issue(issues, "PATH", str(extra), "unexpected or non-canonical entity file")


def _telegram_extension(value: Any) -> dict[str, Any] | None:
    extension = value.extensions.get("telegram") if value.platform == "telegram" else None
    return extension if isinstance(extension, dict) else None


def _validate_telegram(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    for collection in snapshot.collections.values():
        if collection.platform != "telegram":
            continue
        path = str(collection_path(collection.platform, collection.id))
        extension = _telegram_extension(collection)
        if extension is None:
            _issue(issues, "TELEGRAM_EXTENSION", path, "extensions.telegram is required")
            continue
        required = {
            "schema_version": "1.0.0",
            "short_name": collection.native_id,
            "sticker_type": "custom_emoji",
        }
        for field, expected in required.items():
            if extension.get(field) != expected:
                _issue(issues, "TELEGRAM_EXTENSION", f"{path}/{field}", f"must equal {expected!r}")
        if extension.get("retrieved_via") not in {"bot_api", "mtproto"}:
            _issue(issues, "TELEGRAM_EXTENSION", f"{path}/retrieved_via", "invalid transport")
        allowed_collection_fields = {
            "schema_version",
            "retrieved_via",
            "short_name",
            "sticker_type",
            "set_fingerprint_sha256",
            "stable_set_id",
        }
        for field in sorted(set(extension) - allowed_collection_fields):
            _issue(issues, "TELEGRAM_EXTENSION", f"{path}/{field}", "unknown field")
        if collection.native_namespace != "sticker_set.name" or collection.scope_id != "global":
            _issue(
                issues,
                "TELEGRAM_IDENTITY",
                path,
                "collection namespace/scope must be sticker_set.name/global",
            )
        if collection.kind != "custom_emoji_set":
            _issue(
                issues,
                "TELEGRAM_KIND",
                path,
                "Telegram collection kind must be custom_emoji_set",
            )
        canonical_url = f"https://t.me/addemoji/{collection.native_id}"
        if collection.canonical_url != canonical_url:
            _issue(issues, "TELEGRAM_URL", path, f"canonical_url must equal {canonical_url}")
        active = [
            item
            for item in snapshot.memberships.values()
            if item.collection_id == collection.id and item.status is MembershipStatus.ACTIVE
        ]
        pairs: list[tuple[str, str]] = []
        complete = True
        for membership in active:
            emoji = snapshot.emojis.get(membership.emoji_id)
            if emoji is None:
                complete = False
                continue
            extension_emoji = _telegram_extension(emoji)
            if not extension_emoji or not isinstance(extension_emoji.get("file_unique_id"), str):
                complete = False
                continue
            pairs.append((emoji.native_id, extension_emoji["file_unique_id"]))
        if complete:
            expected_fingerprint = telegram_set_fingerprint(pairs)
            if extension.get("set_fingerprint_sha256") != expected_fingerprint:
                _issue(
                    issues, "FINGERPRINT", path, f"set fingerprint must be {expected_fingerprint}"
                )
    for emoji in snapshot.emojis.values():
        if emoji.platform != "telegram":
            continue
        path = str(emoji_bucket_path(emoji.platform, emoji.id))
        extension = _telegram_extension(emoji)
        if extension is None:
            _issue(issues, "TELEGRAM_EXTENSION", path, "extensions.telegram is required")
            continue
        if emoji.native_namespace != "custom_emoji.id" or emoji.scope_id != "global":
            _issue(
                issues,
                "TELEGRAM_IDENTITY",
                path,
                "emoji namespace/scope must be custom_emoji.id/global",
            )
        if emoji.availability.status is AvailabilityStatus.PRIVATE:
            _issue(
                issues,
                "TELEGRAM_AVAILABILITY",
                path,
                "private is not a valid Telegram emoji availability status",
            )
        if len(emoji.media) != 1 or emoji.media[0].role is not MediaRole.PRIMARY:
            _issue(
                issues,
                "TELEGRAM_MEDIA",
                path,
                "Telegram emoji must have exactly one primary media variant",
            )
        if extension.get("custom_emoji_id") != emoji.native_id:
            _issue(issues, "TELEGRAM_EXTENSION", path, "custom_emoji_id must equal native_id")
        if not isinstance(extension.get("file_unique_id"), str) or not extension["file_unique_id"]:
            _issue(issues, "TELEGRAM_EXTENSION", path, "file_unique_id is required")
        if extension.get("schema_version") != "1.0.0" or extension.get("retrieved_via") not in {
            "bot_api",
            "mtproto",
        }:
            _issue(issues, "TELEGRAM_EXTENSION", path, "schema_version/retrieved_via are invalid")
        allowed_emoji_fields = {
            "schema_version",
            "retrieved_via",
            "custom_emoji_id",
            "file_unique_id",
            "fallback_emoji",
            "needs_repainting",
        }
        for field in sorted(set(extension) - allowed_emoji_fields):
            _issue(issues, "TELEGRAM_EXTENSION", f"{path}/{field}", "unknown field")
        primary_rendering = next(
            (
                item
                for item in emoji.facets.rendering.items
                if item.role is MediaRole.PRIMARY and item.variant_id is None
            ),
            None,
        )
        expected_behavior = (
            "platform-adaptive" if extension.get("needs_repainting") is True else "fixed"
        )
        if primary_rendering is None or primary_rendering.color_behavior.value != expected_behavior:
            _issue(
                issues,
                "TELEGRAM_RENDERING",
                path,
                f"primary color_behavior must be {expected_behavior!r}",
            )


def _validate_integrity(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    pairs: dict[tuple[str, str], str] = {}
    positions: dict[tuple[str, int], str] = {}
    active_counts = {key: 0 for key in snapshot.collections}
    for membership in snapshot.memberships.values():
        if membership.collection_id not in snapshot.collections:
            _issue(
                issues, "DANGLING", membership.id, f"missing collection {membership.collection_id}"
            )
        if membership.emoji_id not in snapshot.emojis:
            _issue(issues, "DANGLING", membership.id, f"missing emoji {membership.emoji_id}")
        pair = (membership.collection_id, membership.emoji_id)
        if pair in pairs:
            _issue(
                issues, "MEMBERSHIP_DUPLICATE", membership.id, f"pair also used by {pairs[pair]}"
            )
        pairs[pair] = membership.id
        if membership.status is MembershipStatus.ACTIVE:
            key = (membership.collection_id, membership.position)
            if key in positions:
                _issue(
                    issues,
                    "POSITION_DUPLICATE",
                    membership.id,
                    f"position also used by {positions[key]}",
                )
            positions[key] = membership.id
            active_counts[membership.collection_id] = (
                active_counts.get(membership.collection_id, 0) + 1
            )
    for current_collection_id, collection in snapshot.collections.items():
        if collection.item_count != active_counts.get(current_collection_id, 0):
            _issue(
                issues,
                "ITEM_COUNT",
                current_collection_id,
                f"must equal {active_counts.get(current_collection_id, 0)}",
            )
    public_ids = set(snapshot.collections) | set(snapshot.emojis) | set(snapshot.memberships)
    for target_id in snapshot.tombstones:
        if target_id in public_ids:
            _issue(
                issues,
                "TOMBSTONE_CONFLICT",
                str(tombstone_path(target_id)),
                "target still exists in data",
            )


def _validate_relations(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    tombstoned = set(snapshot.tombstones)
    for relation in snapshot.relations.values():
        path = str(visual_relations_path(relation.id))
        subject = snapshot.emojis.get(relation.subject_id)
        object_emoji = snapshot.emojis.get(relation.object_id)
        for field, identifier, endpoint in (
            ("subject_id", relation.subject_id, subject),
            ("object_id", relation.object_id, object_emoji),
        ):
            if endpoint is None:
                _issue(issues, "RELATION_DANGLING", f"{path}/{field}", "emoji does not exist")
            if identifier in tombstoned:
                _issue(issues, "RELATION_TOMBSTONE", f"{path}/{field}", "emoji is tombstoned")
        if relation.review.status is not ReviewStatus.APPROVED:
            _issue(issues, "RELATION_REVIEW", path, "canonical visual relation must be approved")
        expected_review_hash = reviewed_relation_sha256(relation)
        if relation.review.reviewed_relation_sha256 != expected_review_hash:
            _issue(
                issues,
                "RELATION_REVIEW_HASH",
                path,
                f"review hash must be {expected_review_hash}",
            )
        if subject is None or object_emoji is None:
            continue
        subject_digest = media_digest(subject.media)
        object_digest = media_digest(object_emoji.media)
        if relation.evidence.subject_media_digest != subject_digest:
            _issue(issues, "RELATION_STALE", path, "subject media digest is stale")
        if relation.evidence.object_media_digest != object_digest:
            _issue(issues, "RELATION_STALE", path, "object media digest is stale")
        manifest_profile = snapshot.manifest.get("dedupe_profile")
        if relation.evidence.dedupe_profile != manifest_profile:
            _issue(issues, "RELATION_PROFILE", path, "dedupe profile differs from manifest")
        if (
            relation.evidence.dedupe_profile
            not in {
                subject.fingerprints.profile,
                object_emoji.fingerprints.profile,
            }
            or subject.fingerprints.profile != object_emoji.fingerprints.profile
        ):
            _issue(issues, "RELATION_PROFILE", path, "endpoint fingerprint profiles differ")

        subject_keys = {(item.role.value, item.variant_id or "") for item in subject.media}
        object_keys = {(item.role.value, item.variant_id or "") for item in object_emoji.media}
        mapped_subject = {
            (pair.subject_role.value, pair.subject_variant_id or "")
            for pair in relation.evidence.media_pairs
        }
        mapped_object = {
            (pair.object_role.value, pair.object_variant_id or "")
            for pair in relation.evidence.media_pairs
        }
        if not mapped_subject.issubset(subject_keys) or not mapped_object.issubset(object_keys):
            _issue(issues, "RELATION_MEDIA", path, "evidence references missing media")
        if relation.scope is VisualRelationScope.ENTITY:
            if (
                mapped_subject != subject_keys
                or mapped_object != object_keys
                or len(relation.evidence.media_pairs) != len(subject_keys)
                or len(subject_keys) != len(object_keys)
            ):
                _issue(
                    issues,
                    "RELATION_SCOPE",
                    path,
                    "entity relation requires a full bijective media mapping",
                )
        elif len(relation.evidence.media_pairs) != 1:
            _issue(issues, "RELATION_SCOPE", path, "media-pair scope requires exactly one pair")

        symmetric = relation.relation_type in {
            VisualRelationType.SAME_ARTWORK,
            VisualRelationType.NOT_DUPLICATE,
            VisualRelationType.RELATED_SERIES,
        }
        if symmetric and relation.subject_id >= relation.object_id:
            _issue(issues, "RELATION_ORDER", path, "symmetric endpoints must be sorted")
        if relation.relation_type is VisualRelationType.SAME_ARTWORK:
            left_text = {item.value for item in subject.facets.text_content.items}
            right_text = {item.value for item in object_emoji.facets.text_content.items}
            approved_endpoints = (
                subject.review.status is ReviewStatus.APPROVED
                and object_emoji.review.status is ReviewStatus.APPROVED
            )
            if approved_endpoints and left_text != right_text:
                _issue(
                    issues,
                    "RELATION_TEXT_CONFLICT",
                    path,
                    "same-artwork endpoints have different recognized literal text",
                )


def _validate_emoji_policy(
    snapshot: DatasetSnapshot,
    issues: list[ValidationIssue],
    *,
    canonical: bool,
) -> None:
    controlled_tags = {
        *(item.value for item in ContentType),
        *(item.value for item in Style),
        *(item.value for item in SuggestedUse),
        *(item.value for item in Uncertainty),
    }
    for emoji in snapshot.emojis.values():
        path = str(emoji_bucket_path(emoji.platform, emoji.id))
        media_hashes = sorted({item.sha256 for item in emoji.media})
        input_hashes = sorted(emoji.provenance.input_media_sha256 or [])
        if input_hashes and input_hashes != media_hashes:
            _issue(
                issues,
                "PROVENANCE_MEDIA",
                path,
                "input_media_sha256 must match current media hashes",
            )
        media_order = sorted(
            emoji.media, key=lambda item: (item.role.value, item.variant_id or "", item.sha256)
        )
        if emoji.media != media_order:
            _issue(issues, "SORT", path, "media must be sorted by role, variant_id, sha256")
        if emoji.semantic_tags != sorted(emoji.semantic_tags):
            _issue(issues, "SORT", path, "semantic_tags must be sorted")
        if emoji.concept_ids != sorted(emoji.concept_ids):
            _issue(issues, "SORT", path, "concept_ids must be sorted")
        duplicates = sorted(set(emoji.semantic_tags) & controlled_tags)
        if duplicates:
            _issue(
                issues,
                "FACET_TAG_DUPLICATE",
                path,
                f"controlled facets repeated in semantic_tags: {', '.join(duplicates)}",
            )
        facet_sets = (
            emoji.facets.content_types,
            emoji.facets.styles,
            emoji.facets.suggested_uses,
            emoji.facets.uncertainties,
        )
        for values in facet_sets:
            wire = [item.value for item in values]
            if wire != sorted(wire) or len(wire) != len(set(wire)):
                _issue(issues, "FACET_ORDER", path, "facet arrays must be sorted and unique")
        if emoji.facets.taxonomy_version != snapshot.manifest.get("taxonomy_version"):
            _issue(issues, "TAXONOMY_PROFILE", path, "taxonomy version differs from manifest")
        if emoji.facets.rendering.profile != snapshot.manifest.get("color_profile"):
            _issue(issues, "COLOR_PROFILE", path, "rendering profile differs from manifest")
        if emoji.fingerprints.profile != snapshot.manifest.get("dedupe_profile"):
            _issue(issues, "DEDUPE_PROFILE", path, "fingerprint profile differs from manifest")

        media_keys = [(item.role.value, item.variant_id or "") for item in emoji.media]
        rendering_keys = [item.key for item in emoji.facets.rendering.items]
        fingerprint_keys = [item.key for item in emoji.fingerprints.items]
        if rendering_keys != media_keys:
            _issue(issues, "RENDERING_BINDING", path, "rendering must cover media exactly")
        if emoji.fingerprints.status is FingerprintStatus.COMPLETE:
            if fingerprint_keys != media_keys:
                _issue(
                    issues,
                    "FINGERPRINT_BINDING",
                    path,
                    "complete fingerprints must cover media",
                )
        elif emoji.fingerprints.status is FingerprintStatus.PARTIAL:
            if canonical:
                _issue(
                    issues,
                    "FINGERPRINT_PARTIAL",
                    path,
                    "partial is forbidden in canonical data",
                )
            if not set(fingerprint_keys) < set(media_keys):
                _issue(
                    issues,
                    "FINGERPRINT_BINDING",
                    path,
                    "partial fingerprints must be a proper media subset",
                )
        elif emoji.fingerprints.items:
            _issue(issues, "FINGERPRINT_BINDING", path, "unavailable fingerprints require no items")
        if (
            emoji.fingerprints.status is FingerprintStatus.UNAVAILABLE
            and emoji.availability.status is AvailabilityStatus.ACTIVE
        ):
            _issue(issues, "FINGERPRINT_AVAILABILITY", path, "active emoji requires fingerprints")
        expected_media_digest = media_digest(emoji.media)
        if emoji.fingerprints.input_media_digest != expected_media_digest:
            _issue(
                issues,
                "FINGERPRINT_STALE",
                path,
                f"input_media_digest must be {expected_media_digest}",
            )
        media_by_key = {(item.role.value, item.variant_id or ""): item for item in emoji.media}
        for item in emoji.facets.rendering.items:
            media_item = media_by_key.get(item.key)
            if (
                media_item is not None
                and not media_item.animated
                and item.palette_dynamics is not PaletteDynamics.STABLE
            ):
                _issue(issues, "FACET_STATIC", path, "static palette dynamics must be stable")
            colors = item.dominant_colors or []
            color_order = sorted(
                colors, key=lambda color: (-color.coverage_bp, color.family.value, color.hex)
            )
            if colors != color_order:
                _issue(issues, "PALETTE_ORDER", path, "dominant colors are not canonical")
            if sum(color.coverage_bp for color in colors) > 10_000:
                _issue(
                    issues,
                    "PALETTE_COVERAGE",
                    path,
                    "dominant color coverage exceeds 10000 basis points",
                )
        for fingerprint_item in emoji.fingerprints.items:
            media_item = media_by_key.get(fingerprint_item.key)
            if media_item is not None and emoji.fingerprints.profile == "dedupe-v1":
                expected_samples = 16 if media_item.animated else 1
                if fingerprint_item.perceptual.sample_count != expected_samples:
                    _issue(
                        issues,
                        "FINGERPRINT_SAMPLES",
                        path,
                        f"sample_count must be {expected_samples}",
                    )
        all_media_keys = set(media_keys)
        for text_item in emoji.facets.text_content.items:
            refs = [item.key for item in text_item.media_refs]
            if refs != sorted(refs) or len(refs) != len(set(refs)):
                _issue(issues, "TEXT_MEDIA_REFS", path, "media_refs must be sorted and unique")
            if not set(refs).issubset(all_media_keys):
                _issue(issues, "TEXT_MEDIA_REFS", path, "media_refs contains missing media")
        if (
            all(not item.animated for item in emoji.media)
            and emoji.facets.text_content.dynamics is not TextDynamics.STABLE
        ):
            _issue(issues, "TEXT_DYNAMICS", path, "static emoji text dynamics must be stable")
        if (
            emoji.facets.text_content.status is not TextRecognitionStatus.NONE
            and ContentType.TEXT not in emoji.facets.content_types
        ):
            _issue(issues, "TEXT_CONTENT_TYPE", path, "visible text requires content type text")
        if any(item.kind is TextKind.NUMBER for item in emoji.facets.text_content.items):
            if ContentType.NUMBER not in emoji.facets.content_types:
                _issue(issues, "TEXT_CONTENT_TYPE", path, "numeric text requires number")
        if (
            emoji.facets.text_content.status
            in {
                TextRecognitionStatus.PARTIALLY_RECOGNIZED,
                TextRecognitionStatus.UNREADABLE,
            }
            and Uncertainty.TEXT not in emoji.facets.uncertainties
        ):
            _issue(issues, "TEXT_UNCERTAINTY", path, "text uncertainty is required")
        if (
            any(
                description.motion_status.value == "undetermined"
                for description in emoji.descriptions.values()
            )
            and Uncertainty.MOTION not in emoji.facets.uncertainties
        ):
            _issue(issues, "MOTION_UNCERTAINTY", path, "motion uncertainty is required")
        styles = set(emoji.facets.styles)
        if (
            {Style.MINIMAL, Style.DETAILED}.issubset(styles)
            or {Style.OUTLINE, Style.SOLID}.issubset(styles)
        ) and Uncertainty.STYLE not in emoji.facets.uncertainties:
            if emoji.review.status is not ReviewStatus.APPROVED:
                _issue(issues, "STYLE_CONFLICT", path, "conflicting styles require review")
        warning_values = [item.value for item in emoji.content.warnings]
        if warning_values != sorted(warning_values):
            _issue(issues, "SORT", path, "content.warnings must be sorted")
        if (
            emoji.provenance.input_media_sha256 is not None
            and input_hashes != emoji.provenance.input_media_sha256
        ):
            _issue(issues, "SORT", path, "input_media_sha256 must be sorted")
        if emoji.review.status is not ReviewStatus.UNREVIEWED:
            expected = reviewed_content_sha256(emoji)
            if emoji.review.reviewed_content_sha256 != expected:
                _issue(issues, "REVIEW_HASH", path, f"review hash must be {expected}")
        for language, description in emoji.descriptions.items():
            if unicodedata.normalize("NFC", language) != language:
                _issue(issues, "NFC", path, f"language key {language!r} is not NFC")
            if description.usage != list(dict.fromkeys(description.usage)):
                _issue(issues, "DUPLICATE", path, f"{language} usage contains duplicates")


def _validate_profile_assets(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    profiles = (
        ("color_profile", "color_profile_sha256"),
        ("dedupe_profile", "dedupe_profile_sha256"),
        ("collection_dedupe_profile", "collection_dedupe_profile_sha256"),
    )
    for id_field, hash_field in profiles:
        profile_id = snapshot.manifest.get(id_field)
        expected_hash = snapshot.manifest.get(hash_field)
        if not isinstance(profile_id, str):
            continue
        source_path = snapshot.root / "analysis-profiles" / f"{profile_id}.json"
        if source_path.is_file() and not source_path.is_symlink():
            try:
                actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError as exc:
                _issue(issues, "PROFILE_MISSING", str(source_path), str(exc))
                continue
        else:
            try:
                actual_hash = profile_sha256(profile_id)
            except AnalysisError as exc:
                _issue(
                    issues,
                    "PROFILE_MISSING",
                    f"analysis-profiles/{profile_id}.json",
                    str(exc),
                )
                continue
        if actual_hash != expected_hash:
            _issue(
                issues,
                "PROFILE_HASH",
                f"dataset.json/{hash_field}",
                f"SHA-256 must be {actual_hash}",
            )


def _validate_taxonomy_assets(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    registry_enums = {
        "content_types": {item.value for item in ContentType},
        "styles": {item.value for item in Style},
        "suggested_uses": {item.value for item in SuggestedUse},
        "uncertainties": {item.value for item in Uncertainty},
        "color_families": {item.value for item in ColorFamily},
        "platform_contexts": None,
    }
    expected_paths = {facet: f"{facet.replace('_', '-')}.json" for facet in registry_enums}
    master_path = snapshot.root / "taxonomy" / "v1" / "taxonomy.json"
    try:
        master = parse_json(master_path.read_bytes(), source=str(master_path))
    except (OSError, ValueError) as exc:
        _issue(issues, "TAXONOMY_REGISTRY", "taxonomy/v1/taxonomy.json", str(exc))
        return
    if master.get("taxonomy_version") != snapshot.manifest.get("taxonomy_version"):
        _issue(issues, "TAXONOMY_REGISTRY", "taxonomy/v1/taxonomy.json", "version mismatch")
    raw_registries = master.get("registries")
    if not isinstance(raw_registries, list) or any(
        not isinstance(item, dict) for item in raw_registries
    ):
        _issue(
            issues,
            "TAXONOMY_REGISTRY",
            "taxonomy/v1/taxonomy.json/registries",
            "registries must be an array of objects",
        )
        return
    paths = [str(item.get("path")) for item in raw_registries]
    actual_mapping = {
        str(item.get("dictionary_id")): str(item.get("path")) for item in raw_registries
    }
    if paths != sorted(paths) or actual_mapping != expected_paths:
        _issue(
            issues,
            "TAXONOMY_REGISTRY",
            "taxonomy/v1/taxonomy.json/registries",
            "registry paths/facets are incomplete or not sorted",
        )
    for facet, relative in expected_paths.items():
        display_path = f"taxonomy/v1/{relative}"
        path = snapshot.root / "taxonomy" / "v1" / relative
        try:
            registry_bytes = path.read_bytes()
            registry = parse_json(registry_bytes, source=str(path))
        except (OSError, ValueError) as exc:
            _issue(issues, "TAXONOMY_REGISTRY", display_path, str(exc))
            continue
        master_entry = next(
            (
                item
                for item in raw_registries
                if item.get("dictionary_id") == facet and item.get("path") == relative
            ),
            None,
        )
        if (
            master_entry is None
            or master_entry.get("sha256") != hashlib.sha256(registry_bytes).hexdigest()
        ):
            _issue(
                issues,
                "TAXONOMY_REGISTRY",
                display_path,
                "master dictionary SHA-256 mismatch",
            )
        if registry.get("taxonomy_version") != snapshot.manifest.get("taxonomy_version"):
            _issue(issues, "TAXONOMY_REGISTRY", display_path, "version mismatch")
        if registry.get("facet") != facet:
            _issue(issues, "TAXONOMY_REGISTRY", display_path, f"facet must be {facet!r}")
        entries = registry.get("entries")
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            _issue(
                issues,
                "TAXONOMY_REGISTRY",
                f"{display_path}/entries",
                "invalid entries",
            )
            continue
        ids = [str(item.get("id")) for item in entries]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            _issue(
                issues,
                "TAXONOMY_REGISTRY",
                f"{display_path}/entries",
                "entry IDs must be unique and sorted",
            )
        active = {str(item.get("id")) for item in entries if item.get("status") == "active"}
        expected_active = registry_enums[facet]
        if expected_active is not None and active != expected_active:
            _issue(
                issues,
                "TAXONOMY_REGISTRY",
                f"{display_path}/entries",
                "active IDs do not match schema taxonomy",
            )
        all_ids = set(ids)
        for item in entries:
            if item.get("status") == "deprecated" and item.get("replaced_by") not in all_ids:
                _issue(
                    issues,
                    "TAXONOMY_REGISTRY",
                    f"{display_path}/{item.get('id')}",
                    "deprecated entry requires a valid replaced_by",
                )


def _validate_concept_assets(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    display_path = "taxonomy/v1/concepts.json"
    path = snapshot.root / display_path
    try:
        registry = parse_json(path.read_bytes(), source=str(path))
    except (OSError, ValueError) as exc:
        _issue(issues, "CONCEPT_REGISTRY", display_path, str(exc))
        return
    if (
        registry.get("registry_schema_version") != "1.0.0"
        or registry.get("registry_type") != "concepts"
    ):
        _issue(issues, "CONCEPT_REGISTRY", display_path, "invalid registry header")
    registry_id = registry.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id.startswith("concepts-v1."):
        _issue(
            issues,
            "CONCEPT_REGISTRY",
            f"{display_path}/registry_id",
            "registry identity must be content-versioned",
        )
    concepts = registry.get("concepts")
    if not isinstance(concepts, list) or any(not isinstance(item, dict) for item in concepts):
        _issue(issues, "CONCEPT_REGISTRY", f"{display_path}/concepts", "invalid concepts")
        return
    identifiers = [str(item.get("id")) for item in concepts]
    if identifiers != sorted(identifiers, key=lambda value: value.encode("utf-8")):
        _issue(
            issues,
            "CONCEPT_REGISTRY",
            f"{display_path}/concepts",
            "concept IDs must be bytewise sorted",
        )
    if len(identifiers) != len(set(identifiers)):
        _issue(issues, "CONCEPT_REGISTRY", f"{display_path}/concepts", "duplicate concept ID")
    known = set(identifiers)
    graph: dict[str, list[str]] = {}
    active: set[str] = set()
    for item in concepts:
        identifier = str(item.get("id"))
        parents = item.get("parent_ids")
        if not isinstance(parents, list) or any(not isinstance(value, str) for value in parents):
            _issue(
                issues,
                "CONCEPT_REGISTRY",
                f"{display_path}/{identifier}/parent_ids",
                "parent_ids must be an array of strings",
            )
            parents = []
        if parents != sorted(parents, key=lambda value: value.encode("utf-8")):
            _issue(
                issues,
                "CONCEPT_REGISTRY",
                f"{display_path}/{identifier}/parent_ids",
                "parent_ids must be bytewise sorted",
            )
        missing = sorted(set(parents) - known)
        if missing:
            _issue(
                issues,
                "CONCEPT_REGISTRY",
                f"{display_path}/{identifier}/parent_ids",
                f"unknown parent IDs: {', '.join(missing)}",
            )
        graph[identifier] = list(parents)
        if item.get("status") == "active":
            active.add(identifier)
        for language in ("en", "ru"):
            aliases = (
                item.get("aliases", {}).get(language)
                if isinstance(item.get("aliases"), dict)
                else None
            )
            if not isinstance(aliases, list) or aliases != sorted(
                aliases, key=lambda value: str(value).encode("utf-8")
            ):
                _issue(
                    issues,
                    "CONCEPT_REGISTRY",
                    f"{display_path}/{identifier}/aliases/{language}",
                    "aliases must be bytewise sorted arrays",
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            _issue(issues, "CONCEPT_REGISTRY", display_path, "concept parent graph has a cycle")
            return
        if identifier in visited:
            return
        visiting.add(identifier)
        for parent in graph.get(identifier, []):
            if parent in graph:
                visit(parent)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in identifiers:
        visit(identifier)
    for emoji in snapshot.emojis.values():
        unknown = sorted(set(emoji.concept_ids) - active)
        if unknown:
            _issue(
                issues,
                "CONCEPT_REFERENCE",
                str(emoji_bucket_path(emoji.platform, emoji.id)),
                f"unknown or inactive concept IDs: {', '.join(unknown)}",
            )


def _validate_rights_assets(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    display_path = "rights/profiles.json"
    path = snapshot.root / display_path
    try:
        registry = parse_json(path.read_bytes(), source=str(path))
    except (OSError, ValueError) as exc:
        _issue(issues, "RIGHTS_REGISTRY", display_path, str(exc))
        return
    if (
        registry.get("registry_schema_version") != "1.0.0"
        or registry.get("registry_type") != "rights-profiles"
    ):
        _issue(issues, "RIGHTS_REGISTRY", display_path, "invalid registry header")
    registry_id = registry.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id.startswith("rights-profiles-v1."):
        _issue(
            issues,
            "RIGHTS_REGISTRY",
            f"{display_path}/registry_id",
            "registry identity must be content-versioned",
        )
    profiles = registry.get("profiles")
    if not isinstance(profiles, list) or any(not isinstance(item, dict) for item in profiles):
        _issue(issues, "RIGHTS_REGISTRY", f"{display_path}/profiles", "invalid profiles")
        return
    identifiers = [str(item.get("rights_profile_id")) for item in profiles]
    if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
        _issue(
            issues,
            "RIGHTS_REGISTRY",
            f"{display_path}/profiles",
            "profile IDs must be unique and sorted",
        )
    by_id = {str(item.get("rights_profile_id")): item for item in profiles}
    project_default = snapshot.manifest.get("rights_defaults", {}).get("project_profile_id")
    if (
        registry.get("project_default_profile_id") != project_default
        or project_default not in by_id
    ):
        _issue(
            issues,
            "RIGHTS_REGISTRY",
            f"{display_path}/project_default_profile_id",
            "project default must match dataset.json and an existing profile",
        )
    selected_ids = {project_default}
    platforms = snapshot.manifest.get("platforms")
    for platform in platforms if isinstance(platforms, list) else []:
        profile_path = snapshot.root / "platforms" / f"{platform}.json"
        try:
            platform_profile = parse_json(profile_path.read_bytes(), source=str(profile_path))
        except (OSError, ValueError) as exc:
            _issue(issues, "PLATFORM_PROFILE", f"platforms/{platform}.json", str(exc))
            continue
        if not isinstance(platform_profile, dict) or platform_profile.get("platform") != platform:
            _issue(
                issues,
                "PLATFORM_PROFILE",
                f"platforms/{platform}.json",
                "platform profile identity mismatch",
            )
            continue
        selected_ids.add(platform_profile.get("default_rights_profile_id"))
    for profile_id in selected_ids:
        profile = by_id.get(str(profile_id))
        if profile is None:
            _issue(
                issues,
                "RIGHTS_REGISTRY",
                display_path,
                f"missing selected rights profile {profile_id!r}",
            )
            continue
        operations = profile.get("operations")
        publish = operations.get("publish-metadata") if isinstance(operations, dict) else None
        if not isinstance(publish, dict) or publish.get("decision") != "allow":
            _issue(
                issues,
                "RIGHTS_POLICY",
                f"{display_path}/{profile_id}",
                "selected MVP profile must allow publish-metadata",
            )


def _validate_qualifications(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    from mojilex_cli.policy.qualification import (
        ModelQualificationRegistry,
        QualificationQuery,
        QualificationRegistryError,
        match_qualification,
    )

    candidates = [
        emoji
        for emoji in snapshot.emojis.values()
        if emoji.provenance.origin in {ProvenanceOrigin.AI, ProvenanceOrigin.MIXED}
    ]
    try:
        registry = ModelQualificationRegistry.load(snapshot.root)
    except QualificationRegistryError as exc:
        _issue(
            issues,
            "QUALIFICATION_REGISTRY",
            "quality/model-qualifications.json",
            str(exc),
        )
        return
    for emoji in candidates:
        provenance = emoji.provenance
        path_display = str(emoji_bucket_path(emoji.platform, emoji.id))
        qualification_id = provenance.qualification_id
        if (
            emoji.concept_ids
            and provenance.concept_registry_id is None
            and emoji.review.status is not ReviewStatus.APPROVED
        ):
            _issue(
                issues,
                "QUALIFICATION",
                path_display,
                "unreviewed AI concept output requires exact concept generation binding",
            )
            continue
        if qualification_id is None:
            # Missing qualification is an absent attestation, not a review requirement.
            continue
        query = QualificationQuery(
            provider=str(provenance.provider),
            model=str(provenance.model),
            model_revision=provenance.model_revision,
            description_profile=str(provenance.description_profile),
            prompt_sha256=str(provenance.prompt_sha256),
            request_parameters_sha256=str(provenance.request_parameters_sha256),
            schema_version=emoji.schema_version,
            taxonomy_version=emoji.facets.taxonomy_version,
            pipeline_version=str(provenance.pipeline_version),
            routing_policy_version=str(provenance.routing_policy_version),
            languages=tuple(emoji.descriptions),
            generated_at=str(provenance.generated_at),
            concept_registry_id=provenance.concept_registry_id,
            concept_registry_sha256=provenance.concept_registry_sha256,
            concept_candidate_set_sha256=provenance.concept_candidate_set_sha256,
            concept_candidate_profile_id=provenance.concept_candidate_profile_id,
            concept_candidate_profile_sha256=provenance.concept_candidate_profile_sha256,
            model_routing_policy_id=provenance.model_routing_policy_id,
            model_routing_policy_sha256=provenance.model_routing_policy_sha256,
        )
        match = match_qualification(
            registry,
            query,
            qualification_id=qualification_id,
        )
        if not match.qualified:
            _issue(
                issues,
                "QUALIFICATION",
                path_display,
                f"qualification is not an exact active match: {match.status.value}",
            )


def _validate_review_routing(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    from mojilex_cli.policy import (
        ModelQualificationRegistry,
        PolicyError,
        QualificationRegistryError,
        RoutingReasonRegistry,
        compute_review_routing,
        load_review_policy,
    )

    try:
        RoutingReasonRegistry.load(snapshot.root)
        qualifications = ModelQualificationRegistry.load(snapshot.root)
        _, policy = load_review_policy(snapshot.root)
        report = compute_review_routing(snapshot, qualifications, policy)
    except (PolicyError, QualificationRegistryError) as exc:
        _issue(issues, "POLICY_REGISTRY", "quality", str(exc))
        return
    for item in report.items:
        if item.priority.value == "blocking":
            _issue(
                issues,
                "POLICY_REVIEW_BLOCKING",
                str(emoji_bucket_path(snapshot.emojis[item.emoji_id].platform, item.emoji_id)),
                "blocking review routing reasons: "
                + ", ".join(reason.value for reason in item.reason_codes),
            )


def _git_ls_files(
    root: Path,
    *,
    include_untracked: bool,
) -> tuple[PurePosixPath, ...] | None:
    arguments = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        "ls-files",
        "--cached",
    ]
    if include_untracked:
        arguments.extend(("--others", "--exclude-standard"))
    arguments.extend(("-z", "--", "."))
    try:
        listed = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode != 0:
        return None
    try:
        return tuple(
            PurePosixPath(value.decode("utf-8")) for value in listed.stdout.split(b"\0") if value
        )
    except UnicodeDecodeError:
        return None


def _git_index_entries(root: Path) -> tuple[tuple[str, PurePosixPath], ...] | None:
    arguments = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        "ls-files",
        "--stage",
        "-z",
        "--",
        ".",
    ]
    try:
        listed = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode != 0:
        return None
    entries: list[tuple[str, PurePosixPath]] = []
    try:
        for record in listed.stdout.split(b"\0"):
            if not record:
                continue
            metadata, encoded_path = record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0].decode("ascii")
            relative = PurePosixPath(encoded_path.decode("utf-8"))
            if relative.is_absolute() or ".." in relative.parts:
                return None
            entries.append((mode, relative))
    except (UnicodeDecodeError, ValueError):
        return None
    return tuple(entries)


def _git_root_is_exact_worktree(root: Path) -> bool | None:
    arguments = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        "rev-parse",
        "--show-toplevel",
    ]
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        top_level = Path(result.stdout.decode("utf-8").rstrip("\r\n")).resolve()
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return os.path.normcase(str(top_level)) == os.path.normcase(str(root))


def _tracked_transaction_issues(root: Path) -> tuple[ValidationIssue, ...]:
    entries = _git_index_entries(root)
    if entries is None:
        git_marker = root / ".git"
        if git_marker.exists() or git_marker.is_symlink():
            return (
                ValidationIssue(
                    "GIT_TRACKING",
                    str(root),
                    "tracked local transaction artifacts could not be checked safely",
                ),
            )
        return ()
    issues: list[ValidationIssue] = []
    for mode, relative in entries:
        if relative.parts and relative.parts[0].casefold() in _LOCAL_TRANSACTION_DIRECTORIES:
            issues.append(
                ValidationIssue(
                    "LOCAL_ARTIFACT",
                    relative.as_posix(),
                    "tracked runtime transaction artifacts must remain outside Git",
                )
            )
        if mode == "120000":
            issues.append(
                ValidationIssue(
                    "SYMLINK",
                    relative.as_posix(),
                    "tracked symbolic links are forbidden in the dataset tree",
                )
            )
        elif mode == "160000":
            issues.append(
                ValidationIssue(
                    "GITLINK",
                    relative.as_posix(),
                    "tracked Git submodules are forbidden in the dataset tree",
                )
            )
    return tuple(issues)


def _is_link_or_reparse_point(path: Path) -> bool:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag)


def _walk_repository_without_ignored_trees(
    root: Path, skip: set[str]
) -> tuple[list[Path], list[Path], list[Path], list[OSError]]:
    candidates: list[Path] = []
    nested_git_markers: list[Path] = []
    structural_entries: list[Path] = []
    walk_errors: list[OSError] = []
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=walk_errors.append,
        followlinks=False,
    ):
        directory = Path(current)
        at_root = directory == root
        retained: list[str] = []
        for name in directory_names:
            path = directory / name
            if name.casefold() == ".git":
                if not at_root:
                    nested_git_markers.append(path)
                continue
            try:
                if _is_link_or_reparse_point(path):
                    structural_entries.append(path)
                    continue
            except OSError as exc:
                walk_errors.append(exc)
                continue
            if name.casefold() in skip:
                continue
            candidates.append(path)
            retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            path = directory / name
            if name.casefold() == ".git":
                if not at_root:
                    nested_git_markers.append(path)
                continue
            try:
                if _is_link_or_reparse_point(path):
                    structural_entries.append(path)
                    continue
            except OSError as exc:
                walk_errors.append(exc)
                continue
            candidates.append(path)
    return candidates, nested_git_markers, structural_entries, walk_errors


def _scan_repository_file(
    path: Path, relative: PurePosixPath, issues: list[ValidationIssue]
) -> None:
    prefix = bytearray()
    overlap = b""
    binary_content = False
    detected_secrets: set[str] = set()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(_FILE_SCAN_CHUNK_SIZE)
                if not chunk:
                    break
                if len(prefix) < _FILE_SCAN_PREFIX_SIZE:
                    prefix.extend(chunk[: _FILE_SCAN_PREFIX_SIZE - len(prefix)])
                if b"\0" in chunk:
                    binary_content = True
                window = overlap + chunk
                for label, pattern in _SECRET_PATTERNS.items():
                    if label not in detected_secrets and pattern.search(window):
                        detected_secrets.add(label)
                overlap = window[-_FILE_SCAN_OVERLAP:]
    except OSError as exc:
        _issue(issues, "READ", relative.as_posix(), str(exc))
        return

    header = bytes(prefix)
    if header.startswith(_LFS_HEADER):
        _issue(issues, "NO_MEDIA", relative.as_posix(), "Git LFS pointers are forbidden")
    is_webp = header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    if is_webp or any(header.startswith(magic) for magic in _MEDIA_MAGIC) or binary_content:
        _issue(issues, "NO_MEDIA", relative.as_posix(), "binary/media content is forbidden")
    for label in sorted(detected_secrets):
        _issue(issues, "SECRET", relative.as_posix(), f"detected {label}")


def _validate_repository_files(root: Path, issues: list[ValidationIssue]) -> None:
    skip = {
        ".git",
        ".idea",
        ".mojilex-dedupe",
        ".venv",
        ".vscode",
        ".testdeps",
        ".testprefix",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "cache",
        "dist",
        "temp",
        "tmp",
        "venv",
        "work",
        *_LOCAL_TRANSACTION_DIRECTORIES,
    }
    issues.extend(_tracked_transaction_issues(root))
    tracked = _git_ls_files(root, include_untracked=False)
    listed = _git_ls_files(root, include_untracked=True)
    exact_worktree = _git_root_is_exact_worktree(root)
    walked, nested_git_markers, structural_entries, walk_errors = (
        _walk_repository_without_ignored_trees(root, skip)
    )
    for marker in sorted(nested_git_markers):
        _issue(
            issues,
            "NESTED_GIT",
            marker.relative_to(root).as_posix(),
            "nested Git repository markers are forbidden in the dataset tree",
        )
    for entry in sorted(structural_entries):
        _issue(
            issues,
            "SYMLINK",
            entry.relative_to(root).as_posix(),
            "symbolic links and filesystem reparse points are forbidden in the dataset tree",
        )
    structural_relatives = {
        PurePosixPath(entry.relative_to(root).as_posix()) for entry in structural_entries
    }
    for error in walk_errors:
        _issue(issues, "READ", str(error.filename or root), f"repository walk failed: {error}")
    if exact_worktree is None and ((root / ".git").exists() or (root / ".git").is_symlink()):
        _issue(
            issues,
            "GIT_TRACKING",
            str(root),
            "Git repository root could not be checked safely",
        )

    tracked_set = set(tracked or ())
    if exact_worktree is True and listed is not None and tracked is not None:
        relatives = set(listed)
    else:
        relatives = {PurePosixPath(path.relative_to(root).as_posix()) for path in walked}
    relatives.update(tracked_set)
    for relative in sorted(relatives, key=str):
        if any(
            relative == structural or structural in relative.parents
            for structural in structural_relatives
        ):
            continue
        path = root.joinpath(*relative.parts)
        in_ignored_tree = any(part.casefold() in skip for part in relative.parts)
        if relative in tracked_set and in_ignored_tree:
            if not (
                relative.parts and relative.parts[0].casefold() in _LOCAL_TRANSACTION_DIRECTORIES
            ):
                _issue(
                    issues,
                    "LOCAL_ARTIFACT",
                    relative.as_posix(),
                    "tracked ignored/runtime/generated artifacts must remain outside Git",
                )
            continue
        if in_ignored_tree:
            continue
        if path.is_symlink():
            _issue(issues, "SYMLINK", relative.as_posix(), "symlinks are forbidden in dataset tree")
            continue
        if not path.is_file():
            continue
        lowered_parts = tuple(part.lower() for part in relative.parts)
        if (
            lowered_parts
            and lowered_parts[0] not in {"docs", "tests", "examples"}
            and (
                _FORBIDDEN_LOCAL_ARTIFACT.fullmatch(relative.name) is not None
                or any(part in _FORBIDDEN_LOCAL_DIRECTORIES for part in lowered_parts[:-1])
            )
        ):
            _issue(
                issues,
                "LOCAL_ARTIFACT",
                relative.as_posix(),
                "candidate, preview, and local LSH artifacts must remain outside Git",
            )
        if path.suffix.lower() in _BINARY_SUFFIXES:
            _issue(issues, "NO_MEDIA", relative.as_posix(), "binary/media files are forbidden")
        _scan_repository_file(path, relative, issues)


def _walk_persisted_value(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if isinstance(value, dict):
        normalized_keys: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _issue(issues, "JSON_KEY", path, "JSON object keys must be strings")
                display_key = str(key)
            else:
                display_key = key
                nfc_key = unicodedata.normalize("NFC", key)
                if nfc_key != key:
                    _issue(issues, "NFC", f"{path}/{key}", "object key is not Unicode NFC")
                previous = normalized_keys.get(nfc_key)
                if previous is not None:
                    _issue(
                        issues,
                        "NFC_COLLISION",
                        path,
                        f"keys {previous!r} and {key!r} collide after NFC normalization",
                    )
                normalized_keys[nfc_key] = key
            policy_key = display_key.lower().replace("-", "_")
            if policy_key in _FORBIDDEN_PERSISTED_KEYS:
                _issue(
                    issues,
                    "FORBIDDEN_FIELD",
                    f"{path}/{display_key}",
                    "credential/path field is forbidden",
                )
            _walk_persisted_value(item, f"{path}/{display_key}", issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_persisted_value(item, f"{path}/{index}", issues)
    elif isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            _issue(issues, "NFC", path, "string is not Unicode NFC")
        raw = value.encode("utf-8")
        for label, pattern in _SECRET_PATTERNS.items():
            if pattern.search(raw):
                _issue(issues, "SECRET", path, f"detected {label}")


def _validate_persisted_values(snapshot: DatasetSnapshot, issues: list[ValidationIssue]) -> None:
    memo = _SCHEMA_MEMO.get()
    rules = (
        _walk_persisted_value,
        unicodedata.normalize,
        frozenset(_FORBIDDEN_PERSISTED_KEYS),
        tuple(_SECRET_PATTERNS.items()),
    )

    def walk(value: Any, path: str) -> None:
        # Exact native JSON distinguishes unsafe keys and nested mutations. Only
        # successful value checks are reused; paths and failures are never cached.
        payload = None
        if memo is not None:
            try:
                if _json_native(value):
                    payload = json.dumps(
                        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError, RecursionError):
                pass
        key = (rules, payload) if payload is not None else None
        if memo is not None and key is not None:
            with memo.lock:
                if key in memo.persisted:
                    memo.persisted.move_to_end(key)
                    return
        count = len(issues)
        _walk_persisted_value(value, path, issues)
        if (
            memo is not None
            and key is not None
            and payload is not None
            and len(issues) == count
            and len(payload) <= _PERSISTED_MEMO_BYTES
        ):
            with memo.lock:
                if key not in memo.persisted:
                    while memo.persisted and (
                        memo.persisted_bytes + len(payload) > _PERSISTED_MEMO_BYTES
                        or len(memo.persisted) >= _PERSISTED_MEMO_ENTRIES
                    ):
                        old_key, _ = memo.persisted.popitem(last=False)
                        memo.persisted_bytes -= len(old_key[1])
                    memo.persisted[key] = None
                    memo.persisted_bytes += len(payload)

    walk(snapshot.manifest, "dataset.json")
    for collection_record in snapshot.collections.values():
        walk(collection_record.as_dict(), collection_record.id)
    for emoji_record in snapshot.emojis.values():
        walk(emoji_record.as_dict(), emoji_record.id)
    for membership_record in snapshot.memberships.values():
        walk(membership_record.as_dict(), membership_record.id)
    for relation_record in snapshot.relations.values():
        walk(relation_record.as_dict(), relation_record.id)
    for tombstone_record in snapshot.tombstones.values():
        walk(tombstone_record.as_dict(), tombstone_record.target_id)


def validate_snapshot(
    snapshot: DatasetSnapshot,
    *,
    canonical: bool = False,
    schemas: bool = False,
    repository_files: bool = False,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    namespace = _validate_manifest(snapshot, issues)
    try:
        _validate_ids_and_paths(snapshot, namespace, issues, canonical=canonical)
    except (TypeError, ValueError) as exc:
        _issue(issues, "SERIALIZATION", str(snapshot.root), str(exc))
    _validate_integrity(snapshot, issues)
    _validate_relations(snapshot, issues)
    _validate_telegram(snapshot, issues)
    _validate_emoji_policy(snapshot, issues, canonical=canonical)
    if canonical:
        _validate_profile_assets(snapshot, issues)
        _validate_taxonomy_assets(snapshot, issues)
        _validate_concept_assets(snapshot, issues)
        _validate_rights_assets(snapshot, issues)
        _validate_qualifications(snapshot, issues)
        _validate_review_routing(snapshot, issues)
    _validate_persisted_values(snapshot, issues)
    if schemas:
        _validate_json_schemas(snapshot, issues)
    if repository_files:
        _validate_repository_files(snapshot.root, issues)
    return ValidationReport(tuple(sorted(set(issues))))


def load_validated_dataset(
    root: str | Path, *, strict: bool = True
) -> tuple[DatasetSnapshot | None, ValidationReport]:
    """Load once and validate that exact snapshot with all dataset checks.

    A snapshot is unavailable when a tracking guard or load error prevents loading.
    Callers must check the report before using a returned snapshot.
    """
    root_path = Path(root).resolve()
    tracked_issues = _tracked_transaction_issues(root_path)
    if tracked_issues:
        return None, ValidationReport(tuple(sorted(set(tracked_issues))))
    try:
        snapshot = load_dataset(root_path)
    except DatasetLoadError as exc:
        return None, ValidationReport((ValidationIssue("LOAD", str(root), str(exc)),))
    return snapshot, validate_snapshot(
        snapshot,
        canonical=strict,
        schemas=strict,
        repository_files=strict,
    )


def validate_dataset(root: str | Path, *, strict: bool = True) -> ValidationReport:
    _, report = load_validated_dataset(root, strict=strict)
    return report

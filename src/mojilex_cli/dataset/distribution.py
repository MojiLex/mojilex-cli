from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from mojilex_cli.domain.hashes import jcs_bytes, jcs_sha256, media_digest

from .layout import assert_no_link_or_reparse
from .repository import DatasetSnapshot, load_dataset
from .serialization import canonical_entity, compact_json, parse_json


class DataError(ValueError):
    """Distribution input cannot be read without losing integrity."""


@dataclass(frozen=True, slots=True)
class _LocatedRecord:
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RepositoryRecords:
    collections: list[_LocatedRecord]
    emojis: list[_LocatedRecord]
    memberships: list[_LocatedRecord]
    tombstones: list[_LocatedRecord]
    visual_relations: list[_LocatedRecord]


def discover_records(root: Path, *, snapshot: DatasetSnapshot | None = None) -> _RepositoryRecords:
    """Load canonical records through the CLI's bounded, link-safe repository reader."""

    loaded = snapshot if snapshot is not None else load_dataset(root)

    def records(values: Iterable[Any]) -> list[_LocatedRecord]:
        return [_LocatedRecord(canonical_entity(item)) for item in values]

    return _RepositoryRecords(
        collections=records(loaded.collections.values()),
        emojis=records(loaded.emojis.values()),
        memberships=records(loaded.memberships.values()),
        tombstones=records(loaded.tombstones.values()),
        visual_relations=records(loaded.relations.values()),
    )


def load_json(path: Path) -> dict[str, Any]:
    assert_no_link_or_reparse(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    if b"\r" in raw:
        raise DataError(f"{path}: CR/CRLF line endings are forbidden")
    try:
        return parse_json(raw, source=str(path))
    except ValueError as exc:
        raise DataError(str(exc)) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


MANIFEST_VERSION = "1.0.0"
MINIMUM_READER_VERSION = "0.2.0"
BUILD_INPUT_PROFILE_ID = "release-build-input-v1"
DERIVED_FROM_PROFILE = "derived-from-v1"
STATE_ROOT_PROFILE = "state-roots-v1"
MAX_JSONL_LINE_BYTES = 1024 * 1024
REQUIRED_FEATURES = [
    "concepts-v1",
    "rights-v1",
    "search-records-v1",
    "semantic-roles-v1",
    "state-roots-v1",
]
REQUIRED_LANGUAGES = ["en", "ru"]
_AVAILABILITY_STATUSES = ["active", "unavailable", "private", "deleted", "unknown"]
_REVIEW_STATUSES = ["unreviewed", "approved", "changes_requested", "rejected"]
_MEMBERSHIP_STATUSES = ["active", "removed_from_collection", "unknown"]

PAYLOAD_NAMES = (
    "collection-facets.jsonl",
    "collections.jsonl",
    "concepts.json",
    "duplicate-group-memberships.jsonl",
    "duplicate-groups.jsonl",
    "emojis-active.jsonl",
    "emojis.jsonl",
    "memberships.jsonl",
    "platform-profiles.json",
    "rights-profiles.json",
    "search-en.jsonl",
    "search-ru.jsonl",
    "taxonomy.json",
    "tombstones.jsonl",
    "visual-relations.jsonl",
)

PROFILE_FILES: dict[str, tuple[str, str]] = {
    "distribution": ("distribution-v1.json", "none"),
    "key_serialization": ("canonical-primary-key-v1.json", "canonical"),
    "partitioning": ("sha256-jcs-routing-v1.json", "none"),
    "compression": ("compression-catalog-v1.json", "none"),
    "part_packing": ("part-packing-v1.json", "none"),
    "bundling": ("tar-zstd-bundle-v1.json", "none"),
    "concept_candidates": ("concept-candidates-v1.json", "canonical"),
    "color": ("color-v1.json", "canonical"),
    "dedupe": ("dedupe-v1.json", "canonical"),
    "collection_dedupe": ("collection-dedupe-v1.json", "derived"),
    "lexical_search": ("lexical-search-v1.json", "derived"),
    "language_fallback": ("language-fallback-v1.json", "none"),
    "language_canonicalization": ("bcp47-v1.json", "canonical"),
}

DELEGATED_PROFILE_TYPES: dict[str, str] = {
    "distribution": "distribution",
    "key_serialization": "key-serialization",
    "partitioning": "partitioning",
    "compression": "compression",
    "part_packing": "part-packing",
    "bundling": "bundling",
    "color": "color",
    "dedupe": "dedupe",
    "collection_dedupe": "collection-dedupe",
    "lexical_search": "lexical-search",
    "language_fallback": "language-fallback",
    "language_canonicalization": "language-canonicalization",
}

PROFILE_CONTRACT_SCHEMA_FILES: dict[str, str] = {
    field: f"{profile_type}-profile-contract.schema.json"
    for field, profile_type in DELEGATED_PROFILE_TYPES.items()
}

DERIVED_SCHEMA_NAMES = {
    "agent-record.schema.json",
    "collection-facet.schema.json",
    "duplicate-group-membership.schema.json",
    "duplicate-group.schema.json",
    "search-record.schema.json",
}
CANONICAL_DISTRIBUTION_SCHEMA_NAMES = {
    "concept-candidate-profile.schema.json",
    "concept.schema.json",
    "concepts-registry.schema.json",
    "platform-profile.schema.json",
    "platform-profiles-registry.schema.json",
    "rights-profile.schema.json",
    "rights-profiles-registry.schema.json",
    "taxonomy-dictionary.schema.json",
    "taxonomy-registry.schema.json",
    "taxonomy-source.schema.json",
}


def _is_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _resolve_safe(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for candidate in reversed((absolute, *absolute.parents)):
        if _is_link(candidate):
            raise ValueError(f"{label} path must not contain a link or reparse point: {candidate}")
    return absolute.resolve()


def _present(value: Any = None, *, exists: bool = True) -> dict[str, Any]:
    return {"present": True, "value": value} if exists else {"present": False}


def _component(value: str | None) -> dict[str, Any]:
    return _present(value, exists=value is not None)


def _pointer_value(document: Any, pointer: str) -> tuple[bool, Any]:
    if not pointer.startswith("/"):
        raise ValueError(f"not a canonical JSON Pointer: {pointer!r}")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def key_envelope(record: dict[str, Any], pointers: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pointer in pointers:
        exists, value = _pointer_value(record, pointer)
        component: dict[str, Any] = {"pointer": pointer, "present": exists}
        if exists:
            if isinstance(value, (dict, list)):
                raise ValueError(f"record key pointer {pointer!r} is not scalar")
            component["value"] = value
        result.append(component)
    return result


def effective_sort_key(
    record: dict[str, Any], sort_key: list[str], primary_key: list[str]
) -> bytes:
    pointers = [*sort_key, *(pointer for pointer in primary_key if pointer not in sort_key)]
    return jcs_bytes(key_envelope(record, pointers))


def sort_records(
    records: Iterable[dict[str, Any]], *, sort_key: list[str], primary_key: list[str]
) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: effective_sort_key(row, sort_key, primary_key))


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    lines: list[bytes] = []
    for record in records:
        line = compact_json(record).encode("utf-8") + b"\n"
        if len(line) > MAX_JSONL_LINE_BYTES:
            raise ValueError("JSONL line exceeds the 1 MiB distribution hard limit")
        lines.append(line)
    return b"".join(lines)


def table_root(records: list[dict[str, Any]], primary_key: list[str]) -> str:
    keyed: list[tuple[bytes, list[dict[str, Any]], str]] = []
    seen: set[bytes] = set()
    for record in records:
        envelope = key_envelope(record, primary_key)
        encoded_key = jcs_bytes(envelope)
        if encoded_key in seen:
            raise ValueError(f"duplicate primary key: {envelope!r}")
        seen.add(encoded_key)
        keyed.append((encoded_key, envelope, jcs_sha256(record)))
    digest = hashlib.sha256()
    for _, envelope, record_sha256 in sorted(keyed, key=lambda item: item[0]):
        digest.update(jcs_bytes([envelope, record_sha256]) + b"\n")
    return digest.hexdigest()


def schema_uri(root: Path, relative_path: str) -> str:
    schema = load_json(root / relative_path)
    uri = schema.get("$id") if isinstance(schema, dict) else None
    if not isinstance(uri, str) or not re.match(r"^[a-z][a-z0-9+.-]*://", uri):
        raise DataError(f"{relative_path}: schema must have an absolute $id")
    return uri


def aggregate_digest(paths: Iterable[tuple[str, bytes]]) -> str:
    members = [
        {"path": path, "sha256": sha256_bytes(content)}
        for path, content in sorted(paths, key=lambda item: item[0].encode())
    ]
    return jcs_sha256(members)


def _taxonomy_registry(root: Path) -> tuple[dict[str, Any], list[Path], str]:
    taxonomy_root = root / "taxonomy" / "v1"
    manifest_path = taxonomy_root / "taxonomy.json"
    source = load_json(manifest_path)
    source_paths = [manifest_path]
    dictionaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in source["registries"]:
        dictionary_id = item.get("dictionary_id", item.get("facet"))
        parts = PurePosixPath(item["path"]).parts
        if parts[:2] == ("taxonomy", "v1"):
            parts = parts[2:]
        if len(parts) != 1 or parts[0] in {"", ".", ".."}:
            raise DataError(f"invalid taxonomy source path: {item['path']!r}")
        path = taxonomy_root / parts[0]
        body = load_json(path)
        body_id = body.get("dictionary_id", body.get("facet"))
        if body_id != dictionary_id or dictionary_id in seen:
            raise DataError(f"taxonomy dictionary identity mismatch: {path}")
        seen.add(dictionary_id)
        entries = body["entries"]
        if entries != sorted(entries, key=lambda entry: entry["id"].encode()):
            raise DataError(f"taxonomy entries are not bytewise sorted: {path}")
        dictionaries.append({"dictionary_id": dictionary_id, "entries": entries})
        source_paths.append(path)
    dictionaries.sort(key=lambda item: item["dictionary_id"].encode())
    source_root = aggregate_digest(
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in source_paths
    )
    return (
        {
            "registry_schema_version": MANIFEST_VERSION,
            "registry_type": "facet-taxonomy",
            "registry_id": f"facet-taxonomy-v1.{source_root[:16]}",
            "dictionaries": dictionaries,
        },
        source_paths,
        source_root,
    )


def _platform_registry(root: Path) -> tuple[dict[str, Any], list[Path], str]:
    paths = sorted((root / "platforms").glob("*.json"), key=lambda path: path.name.encode())
    if not paths:
        raise DataError("platforms registry must contain at least one profile")
    entries: list[dict[str, Any]] = []
    for path in paths:
        if _is_link(path) or not path.is_file():
            raise DataError(f"platform source must be a regular non-link file: {path}")
        entry = load_json(path)
        if entry.get("platform") != path.stem:
            raise DataError(f"platform filename/body mismatch: {path}")
        entries.append(entry)
    entries.sort(key=lambda item: (item["platform"].encode(), item["profile_version"].encode()))
    source_root = aggregate_digest(
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in paths
    )
    return (
        {
            "registry_schema_version": MANIFEST_VERSION,
            "registry_type": "platform-profiles",
            "registry_id": f"platform-profiles-v1.{source_root[:16]}",
            "entries": entries,
        },
        paths,
        source_root,
    )


def _eligible_review(emoji: dict[str, Any]) -> bool:
    return emoji["review"]["status"] in {"approved", "unreviewed"}


def _eligible_search(emoji: dict[str, Any]) -> bool:
    return emoji["concept_mapping_status"] == "complete" and bool(emoji["concept_ids"])


def _rights_summary(
    emoji: dict[str, Any],
    platform_by_id: dict[str, dict[str, Any]],
    rights_registry: dict[str, Any],
    source_date_epoch: int,
) -> dict[str, Any]:
    platform = platform_by_id.get(emoji["platform"])
    if platform is None:
        raise DataError(f"missing platform profile for {emoji['platform']!r}")
    has_default = "default_rights_profile_id" in platform
    inherits = platform.get("inherit_project_default") is True
    if has_default == inherits:
        raise DataError("platform profile must choose exactly one rights default selector")
    profile_id = (
        platform["default_rights_profile_id"]
        if has_default
        else rights_registry["project_default_profile_id"]
    )
    profiles = {profile["rights_profile_id"]: profile for profile in rights_registry["profiles"]}
    profile = profiles.get(profile_id)
    if profile is None or profile.get("status") != "active":
        raise DataError(f"rights selector does not resolve to active profile {profile_id!r}")
    build_time = datetime.fromtimestamp(source_date_epoch, tz=UTC)
    effective_from = datetime.fromisoformat(profile["effective_from"].replace("Z", "+00:00"))
    effective_until = (
        datetime.fromisoformat(profile["effective_until"].replace("Z", "+00:00"))
        if "effective_until" in profile
        else None
    )
    in_effective_interval = build_time >= effective_from and (
        effective_until is None or build_time < effective_until
    )
    operations = profile.get("operations", {})
    decisions = [
        operations.get(name, {}).get("decision", "unknown")
        for name in ("publish-metadata", "publish-generated-annotations")
    ]
    if not in_effective_interval:
        status = "unknown"
    elif profile.get("withdrawn_from_distribution") is True:
        status = "withdrawn"
    elif any(value in {"deny", "conditional", "not-granted"} for value in decisions):
        status = "restricted"
    elif "unknown" in decisions or any(
        value not in {"allow", "deny", "conditional", "not-granted"} for value in decisions
    ):
        status = "unknown"
    elif decisions == ["allow", "allow"]:
        status = "allowed"
    else:
        status = "restricted"
    result: dict[str, Any] = {
        "rights_profile_id": profile_id,
        "distribution_status": status,
        "attribution_required": profile["attribution_required"],
    }
    if profile["attribution_required"]:
        result["attribution_locator"] = profile["attribution_locator"]
    return result


def _role_component(item: dict[str, Any]) -> dict[str, Any]:
    return _present(item["role"])


def _variant_component(item: dict[str, Any]) -> dict[str, Any]:
    return _component(item.get("variant_id"))


def _members_root(members: list[dict[str, Any]], *, reviewed: bool = False) -> str:
    lines: list[bytes] = []
    for member in members:
        if reviewed:
            identity: Any = member["emoji_id"]
        elif member["member_scope"] == "entity":
            identity = [
                "entity",
                member["emoji_id"],
                _present(exists=False),
                _present(exists=False),
            ]
        else:
            identity = ["media", member["emoji_id"], member["role"], member["variant"]]
        lines.append(jcs_bytes(identity))
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line + b"\n")
    return digest.hexdigest()


def duplicate_group_id(namespace: uuid.UUID, preimage: Any) -> str:
    return "mxdg_" + str(uuid.uuid5(namespace, jcs_bytes(preimage).decode("utf-8")))


def _media_set_root(emoji: dict[str, Any], *, decoded: bool) -> tuple[str, str | None]:
    fingerprints = {
        (item["role"], item.get("variant_id")): item for item in emoji["fingerprints"]["items"]
    }
    lines: list[bytes] = []
    for media in emoji["media"]:
        if decoded:
            fingerprint = fingerprints[(media["role"], media.get("variant_id"))]
            item = [
                _role_component(media),
                _variant_component(media),
                emoji["fingerprints"]["profile"],
                fingerprint["decoded_payload_sha256"],
            ]
        else:
            item = [
                _role_component(media),
                _variant_component(media),
                media["sha256"],
                media["byte_size"],
            ]
        lines.append(jcs_bytes(item))
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line + b"\n")
    return digest.hexdigest(), emoji["fingerprints"]["profile"] if decoded else None


def build_duplicate_groups(
    emojis: list[dict[str, Any]],
    visual_relations: list[dict[str, Any]],
    namespace: uuid.UUID,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    binary_media: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    decoded_media: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    binary_entities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decoded_entities: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for emoji in emojis:
        fingerprints = {
            (item["role"], item.get("variant_id")): item for item in emoji["fingerprints"]["items"]
        }
        for media in emoji["media"]:
            member = {
                "schema_version": MANIFEST_VERSION,
                "group_id": "",
                "member_scope": "media",
                "emoji_id": emoji["id"],
                "role": _role_component(media),
                "variant": _variant_component(media),
            }
            binary_media[(media["sha256"], media["byte_size"])].append(member)
            decoded_media[
                (
                    emoji["fingerprints"]["profile"],
                    fingerprints[(media["role"], media.get("variant_id"))][
                        "decoded_payload_sha256"
                    ],
                )
            ].append(member.copy())
        entity_member = {
            "schema_version": MANIFEST_VERSION,
            "group_id": "",
            "member_scope": "entity",
            "emoji_id": emoji["id"],
            "role": _present(exists=False),
            "variant": _present(exists=False),
        }
        binary_root, _ = _media_set_root(emoji, decoded=False)
        decoded_root, profile = _media_set_root(emoji, decoded=True)
        binary_entities[binary_root].append(entity_member)
        decoded_entities[(profile or "", decoded_root)].append(entity_member.copy())

    summaries: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []

    def add(group_type: str, group_key: dict[str, Any], members: list[dict[str, Any]]) -> None:
        if len(members) < 2:
            return
        scope = group_key["scope"]
        if group_type == "binary-exact" and scope == "media":
            preimage = [
                "duplicate-group-v1",
                group_type,
                scope,
                _present(exists=False),
                group_key["source_sha256"],
                group_key["byte_size"],
            ]
        elif group_type == "binary-exact":
            preimage = [
                "duplicate-group-v1",
                group_type,
                scope,
                _present(exists=False),
                group_key["media_set_root_sha256"],
                _present(exists=False),
            ]
        elif group_type == "decoded-exact" and scope == "media":
            preimage = [
                "duplicate-group-v1",
                group_type,
                scope,
                _present(group_key["decoded_profile_id"]),
                group_key["decoded_payload_sha256"],
                _present(exists=False),
            ]
        elif group_type == "decoded-exact":
            preimage = [
                "duplicate-group-v1",
                group_type,
                scope,
                _present(group_key["decoded_profile_id"]),
                group_key["media_set_root_sha256"],
                _present(exists=False),
            ]
        else:
            preimage = [
                "duplicate-group-v1",
                "reviewed-same-artwork",
                "entity",
                _present(exists=False),
                _present(exists=False),
                sorted(member["emoji_id"] for member in members),
            ]
        group_id = duplicate_group_id(namespace, preimage)
        materialized = [{**member, "group_id": group_id} for member in members]
        memberships.extend(materialized)
        summaries.append(
            {
                "schema_version": MANIFEST_VERSION,
                "group_id": group_id,
                "group_type": group_type,
                "member_count": len(materialized),
                "members_root_sha256": _members_root(
                    materialized, reviewed=group_type == "reviewed-same-artwork"
                ),
                "group_key": group_key,
            }
        )

    for (source_sha256, byte_size), members in sorted(binary_media.items()):
        add(
            "binary-exact",
            {"scope": "media", "source_sha256": source_sha256, "byte_size": byte_size},
            members,
        )
    for (profile, decoded_sha256), members in sorted(decoded_media.items()):
        add(
            "decoded-exact",
            {
                "scope": "media",
                "decoded_profile_id": profile,
                "decoded_payload_sha256": decoded_sha256,
            },
            members,
        )
    for media_set_root, members in sorted(binary_entities.items()):
        add(
            "binary-exact",
            {"scope": "entity", "media_set_root_sha256": media_set_root},
            members,
        )
    for (profile, media_set_root), members in sorted(decoded_entities.items()):
        add(
            "decoded-exact",
            {
                "scope": "entity",
                "decoded_profile_id": profile,
                "media_set_root_sha256": media_set_root,
            },
            members,
        )

    graph: dict[str, set[str]] = defaultdict(set)
    for relation in visual_relations:
        if relation["scope"] == "entity" and relation["relation_type"] == "same-artwork":
            graph[relation["subject_id"]].add(relation["object_id"])
            graph[relation["object_id"]].add(relation["subject_id"])
    seen: set[str] = set()
    for first in sorted(graph):
        if first in seen:
            continue
        pending = [first]
        component: list[str] = []
        while pending:
            emoji_id = pending.pop()
            if emoji_id in seen:
                continue
            seen.add(emoji_id)
            component.append(emoji_id)
            pending.extend(sorted(graph[emoji_id] - seen, reverse=True))
        add(
            "reviewed-same-artwork",
            {"scope": "entity", "relation_type": "same-artwork"},
            [
                {
                    "schema_version": MANIFEST_VERSION,
                    "group_id": "",
                    "member_scope": "entity",
                    "emoji_id": emoji_id,
                    "role": _present(exists=False),
                    "variant": _present(exists=False),
                }
                for emoji_id in sorted(component)
            ],
        )

    group_key = ["/group_type", "/group_id"]
    membership_key = [
        "/group_id",
        "/member_scope",
        "/emoji_id",
        "/role/present",
        "/role/value",
        "/variant/present",
        "/variant/value",
    ]
    return (
        sort_records(summaries, sort_key=group_key, primary_key=group_key),
        sort_records(memberships, sort_key=membership_key, primary_key=membership_key),
    )


def _current_visual_relations(
    relations: list[dict[str, Any]],
    emojis: dict[str, dict[str, Any]],
    tombstoned_ids: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for relation in relations:
        subject = emojis.get(relation["subject_id"])
        object_emoji = emojis.get(relation["object_id"])
        if relation["review"]["status"] != "approved" or not subject or not object_emoji:
            continue
        if relation["subject_id"] in tombstoned_ids or relation["object_id"] in tombstoned_ids:
            continue
        if relation["evidence"]["subject_media_digest"] != media_digest(subject["media"]):
            continue
        if relation["evidence"]["object_media_digest"] != media_digest(object_emoji["media"]):
            continue
        result.append(relation)
    return sorted(result, key=lambda item: item["id"].encode())


def _literal_text(emoji: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    search_kinds = {"letter": "symbol", "punctuation": "symbol", "code": "mixed", "other": "mixed"}
    for item in emoji["facets"]["text_content"]["items"]:
        projected = {
            "value": item["value"],
            "kind": search_kinds.get(item["kind"], item["kind"]),
            "script": item["script"],
        }
        if "language" in item:
            projected["language"] = item["language"]
        projected["temporal_scope"] = item["temporal_scope"]
        result.append(projected)
    return result


def _search_review(emoji: dict[str, Any]) -> dict[str, Any]:
    origin = emoji["provenance"]["origin"]
    ai_dependent = origin == "ai"
    return {
        "status": emoji["review"]["status"],
        "attested": False,
        "provenance_origin": origin,
        "model_qualification_status": "missing" if ai_dependent else "not-applicable",
        "model_qualification": _present(exists=False),
        "generation_attestation_status": "missing" if ai_dependent else "not-applicable",
    }


def _search_record(
    emoji: dict[str, Any],
    *,
    language: str,
    collection_ids: list[str],
    duplicate_group_ids: list[str],
    rights: dict[str, Any],
    capability_ids: list[str],
) -> dict[str, Any]:
    rendering = emoji["facets"]["rendering"]["items"]
    native_references = [
        {
            "platform": emoji["platform"],
            "native_namespace": emoji["native_namespace"],
            "scope_id": emoji["scope_id"],
            "native_id": emoji["native_id"],
            "identity_epoch": emoji["identity_epoch"],
            "status": "current",
        }
    ]
    all_collections = sorted(set(collection_ids), key=str.encode)
    all_groups = sorted(set(duplicate_group_ids), key=str.encode)
    facets = emoji["facets"]
    return {
        "record_schema_version": MANIFEST_VERSION,
        "entity_type": "search_record",
        "emoji_id": emoji["id"],
        "language": language,
        "canonical_locator": {
            "logical_name": "emojis",
            "record_key": [{"pointer": "/id", "present": True, "value": emoji["id"]}],
        },
        "canonical_record_sha256": jcs_sha256(emoji),
        "platform": emoji["platform"],
        "native_reference_count": 1,
        "native_references": native_references,
        "native_references_truncated": False,
        "collection_count": len(all_collections),
        "collection_ids": all_collections[:64],
        "collections_truncated": len(all_collections) > 64,
        "description": emoji["descriptions"][language],
        "semantic": {
            "concept_ids": emoji["concept_ids"],
            "semantic_tags": emoji["semantic_tags"],
        },
        "facets": {
            "animated": any(item["animated"] for item in emoji["media"]),
            "media_kinds": sorted({item["kind"] for item in emoji["media"]}),
            "color_behaviors": sorted({item["color_behavior"] for item in rendering}),
            "color_families": sorted(
                {
                    color["family"]
                    for item in rendering
                    if item["color_behavior"] in {"fixed", "mixed"}
                    for color in item.get("dominant_colors", [])
                }
            ),
            "contains_text": facets["text_content"]["status"] != "none",
            "content_types": facets["content_types"],
            "styles": facets["styles"],
            "suggested_uses": facets["suggested_uses"],
            "uncertainties": facets["uncertainties"],
        },
        "literal_text": _literal_text(emoji),
        "availability": {"status": "active", "freshness_status": "unknown"},
        "review": _search_review(emoji),
        "content": emoji["content"],
        "rights": rights,
        "duplicate_group_count": len(all_groups),
        "duplicate_group_ids": all_groups[:32],
        "duplicate_groups_truncated": len(all_groups) > 32,
        "platform_capability_refs": capability_ids,
    }


def _collection_facets(
    collections: list[dict[str, Any]],
    memberships: list[dict[str, Any]],
    emojis: dict[str, dict[str, Any]],
    group_ids_by_emoji: dict[str, list[str]],
    groups_by_id: dict[str, dict[str, Any]],
    memberships_table_root: str,
    collection_dedupe_profile_id: str,
) -> list[dict[str, Any]]:
    emoji_ids_by_collection: dict[str, list[str]] = defaultdict(list)
    for membership in memberships:
        emoji_ids_by_collection[membership["collection_id"]].append(membership["emoji_id"])
    result: list[dict[str, Any]] = []
    for collection in collections:
        values = [
            emojis[emoji_id]
            for emoji_id in sorted(emoji_ids_by_collection.get(collection["id"], []))
            if emoji_id in emojis
        ]
        media_kinds: Counter[str] = Counter()
        content_types: Counter[str] = Counter()
        styles: Counter[str] = Counter()
        suggested_uses: Counter[str] = Counter()
        color_families: Counter[str] = Counter()
        adaptive = fixed = recognized = numbers = 0
        collection_groups: set[str] = set()
        for emoji in values:
            primary = next(item for item in emoji["media"] if item["role"] == "primary")
            media_kinds[primary["kind"]] += 1
            facets = emoji["facets"]
            content_types.update(facets["content_types"])
            styles.update(facets["styles"])
            suggested_uses.update(facets["suggested_uses"])
            rendering = facets["rendering"]["items"]
            behaviors = {item["color_behavior"] for item in rendering}
            adaptive += bool(behaviors & {"platform-adaptive", "mixed"})
            fixed += "fixed" in behaviors
            color_families.update(
                color["family"]
                for item in rendering
                if item["color_behavior"] in {"fixed", "mixed"}
                for color in item.get("dominant_colors", [])
            )
            recognized += facets["text_content"]["status"] in {
                "recognized",
                "partially-recognized",
            }
            numbers += "number" in facets["content_types"]
            collection_groups.update(group_ids_by_emoji.get(emoji["id"], []))
        count = len(values)
        result.append(
            {
                "schema_version": MANIFEST_VERSION,
                "collection_id": collection["id"],
                "source_memberships_table_root_sha256": memberships_table_root,
                "collection_dedupe_profile_id": collection_dedupe_profile_id,
                "active_memberships": count,
                "media_kind_counts": dict(sorted(media_kinds.items())),
                "adaptive_share_bp": round(adaptive * 10000 / count) if count else 0,
                "fixed_share_bp": round(fixed * 10000 / count) if count else 0,
                "content_type_counts": dict(sorted(content_types.items())),
                "style_counts": dict(sorted(styles.items())),
                "recognized_text_count": recognized,
                "number_count": numbers,
                "color_family_counts": dict(sorted(color_families.items())),
                "suggested_use_counts": dict(sorted(suggested_uses.items())),
                "exact_duplicate_group_count": sum(
                    groups_by_id[group_id]["group_type"] in {"binary-exact", "decoded-exact"}
                    for group_id in collection_groups
                ),
                "reviewed_visual_duplicate_group_count": sum(
                    groups_by_id[group_id]["group_type"] == "reviewed-same-artwork"
                    for group_id in collection_groups
                ),
            }
        )
    return sorted(result, key=lambda item: item["collection_id"].encode())


def record_descriptor(
    *,
    logical_name: str,
    semantic_role: str,
    path: str,
    schema_ref: str,
    records: list[dict[str, Any]],
    primary_key: list[str],
    sort_key: list[str],
    payload: bytes,
    routing_key: list[str] | None = None,
    derived_from: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest = sha256_bytes(payload)
    result: dict[str, Any] = {
        "logical_name": logical_name,
        "semantic_role": semantic_role,
        "content_model": "recordset-jsonl",
        "path": path,
        "media_type": "application/x-ndjson",
        "schema_ref": schema_ref,
        "compression": "none",
        "payload_sha256": digest,
        "object_sha256": digest,
        "uncompressed_byte_size": len(payload),
        "object_byte_size": len(payload),
        "record_count": len(records),
        "logical_record_count": len(records),
        "primary_key": primary_key,
    }
    if routing_key is not None:
        result["routing_key"] = routing_key
    result["sort_key"] = sort_key
    result["table_root_sha256"] = table_root(records, primary_key)
    if derived_from is not None:
        result["derived_from"] = derived_from
    return result


def singleton_descriptor(
    *, logical_name: str, path: str, schema_ref: str, payload: bytes
) -> dict[str, Any]:
    digest = sha256_bytes(payload)
    return {
        "logical_name": logical_name,
        "semantic_role": "normative",
        "content_model": "singleton-json",
        "path": path,
        "media_type": "application/json",
        "schema_ref": schema_ref,
        "compression": "none",
        "payload_sha256": digest,
        "object_sha256": digest,
        "uncompressed_byte_size": len(payload),
        "object_byte_size": len(payload),
    }


def _semantic_entry(descriptor: dict[str, Any]) -> dict[str, Any]:
    is_recordset = descriptor["content_model"] == "recordset-jsonl"
    return {
        "entry_type": "artifact",
        "entry_name": descriptor["logical_name"],
        "semantic_role": descriptor["semantic_role"],
        "digest_kind": "table-root" if is_recordset else "payload",
        "sha256": (
            descriptor["table_root_sha256"] if is_recordset else descriptor["payload_sha256"]
        ),
    }


def state_root(
    descriptors: list[dict[str, Any]],
    selectors: Iterable[tuple[str, str, str, dict[str, Any]]],
    *,
    derived: bool,
    canonical_root: str | None = None,
) -> str:
    artifact_roles = {"derived"} if derived else {"canonical", "normative"}
    entries = [
        _semantic_entry(descriptor)
        for descriptor in descriptors
        if descriptor["semantic_role"] in artifact_roles
    ]
    scope = "derived" if derived else "canonical"
    for selector_kind, field, root_scope, selector in selectors:
        if root_scope == scope:
            entries.append(
                {
                    "entry_type": "input",
                    "entry_name": (
                        f"profile:{field}:{selector.get('id', '')}"
                        if selector_kind == "profile"
                        else f"policy:{field}"
                    ),
                    "semantic_role": "derived" if derived else "normative",
                    "digest_kind": "input",
                    "sha256": selector["sha256"],
                }
            )
    entries.sort(
        key=lambda entry: jcs_bytes(
            [entry["semantic_role"], entry["entry_type"], entry["entry_name"]]
        )
    )
    body: dict[str, Any] = {"profile": STATE_ROOT_PROFILE}
    if derived:
        if canonical_root is None:
            raise ValueError("derived root requires the canonical root")
        body["source_canonical_state_root_sha256"] = canonical_root
    body["entries"] = entries
    return jcs_sha256(body)


def derived_from(
    canonical_root: str,
    *,
    manifest_values: list[tuple[str, Any]] | None = None,
    selectors: list[tuple[str, dict[str, Any]]] | None = None,
    dependencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_inputs = [
        {"manifest_pointer": pointer, "value_sha256": jcs_sha256(value)}
        for pointer, value in (manifest_values or [])
    ]
    selector_inputs = []
    for pointer, value in selectors or []:
        selector_kind = "profile" if pointer.startswith("/profiles/") else "policy"
        selector_inputs.append(
            {
                "selector_kind": selector_kind,
                "manifest_pointer": pointer,
                "selector_value_sha256": jcs_sha256(value),
            }
        )
    dependency_entries = []
    for descriptor in dependencies or []:
        is_recordset = descriptor["content_model"] == "recordset-jsonl"
        dependency_entries.append(
            {
                "logical_name": descriptor["logical_name"],
                "digest_kind": "table-root" if is_recordset else "payload",
                "sha256": (
                    descriptor["table_root_sha256"]
                    if is_recordset
                    else descriptor["payload_sha256"]
                ),
            }
        )
    return {
        "profile": DERIVED_FROM_PROFILE,
        "source_canonical_state_root_sha256": canonical_root,
        "manifest_inputs": sorted(manifest_inputs, key=lambda item: item["manifest_pointer"]),
        "selectors": sorted(selector_inputs, key=lambda item: item["manifest_pointer"]),
        "derived_artifacts": sorted(dependency_entries, key=lambda item: item["logical_name"]),
        "physical_input_roots": [],
    }


def physical_resource(
    *,
    uri: str,
    source_path: str,
    path: str,
    media_type: str,
    content: bytes,
    bindings: list[dict[str, Any]],
    content_schema_ref: str | None = None,
) -> dict[str, Any]:
    digest = sha256_bytes(content)
    result: dict[str, Any] = {
        "uri": uri,
        "resource_kind": "physical",
        "source_path": source_path,
        "path": path,
        "media_type": media_type,
        "compression": "none",
        "payload_sha256": digest,
        "object_sha256": digest,
        "uncompressed_byte_size": len(content),
        "object_byte_size": len(content),
        "bindings": sorted(bindings, key=jcs_bytes),
    }
    if content_schema_ref is not None:
        result["content_schema_ref"] = content_schema_ref
    return result


def artifact_alias(
    *, uri: str, logical_name: str, payload_sha256: str, manifest_pointer: str
) -> dict[str, Any]:
    return {
        "uri": uri,
        "resource_kind": "artifact-alias",
        "artifact_logical_name": logical_name,
        "payload_sha256": payload_sha256,
        "bindings": [{"kind": "exact-content", "manifest_pointer": manifest_pointer}],
    }


def histogram(values: Iterable[str], variants: list[str]) -> dict[str, int]:
    counts = Counter(values)
    if unknown := set(counts) - set(variants):
        raise DataError(f"unknown status values: {sorted(unknown)!r}")
    return {variant: counts.get(variant, 0) for variant in variants}


def build_distribution(
    root: Path,
    *,
    snapshot: DatasetSnapshot | None = None,
    revision: str,
    snapshot_id: str,
    source_date_epoch: int,
    tool_commit: str,
    dependency_lock_sha256: str,
    tool_version: str = "0.2.0",
) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = _resolve_safe(root, label="dataset root")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        raise ValueError("revision must be a full lowercase Git object ID")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tool_commit):
        raise ValueError("tool_commit must be a full lowercase Git object ID")
    if not re.fullmatch(r"[0-9a-f]{64}", dependency_lock_sha256):
        raise ValueError("dependency_lock_sha256 must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"0|[1-9][0-9]*", str(source_date_epoch)):
        raise ValueError("source_date_epoch must be a non-negative integer")
    epoch = source_date_epoch
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("source_date_epoch must be a non-negative integer")
    snapshot_match = re.fullmatch(r"data-([0-9]{4}\.[0-9]{2}\.[0-9]{2})\.[1-9][0-9]*", snapshot_id)
    if snapshot_match is None:
        raise ValueError("snapshot_id must be data-YYYY.MM.DD.N with positive N")
    try:
        datetime.strptime(snapshot_match.group(1), "%Y.%m.%d")
    except ValueError as exc:
        raise ValueError("snapshot_id contains an invalid UTC calendar date") from exc

    records = discover_records(root, snapshot=snapshot)
    dataset = snapshot.manifest if snapshot is not None else load_json(root / "dataset.json")
    collections = sorted((item.value for item in records.collections), key=lambda item: item["id"])
    emojis = sorted((item.value for item in records.emojis), key=lambda item: item["id"])
    memberships = sorted((item.value for item in records.memberships), key=lambda item: item["id"])
    tombstones = sorted(
        (item.value for item in records.tombstones),
        key=lambda item: (item["target_entity_type"], item["target_id"]),
    )
    all_emojis = {item["id"]: item for item in emojis}
    tombstoned_ids = {item["target_id"] for item in tombstones}
    tombstoned_memberships = {
        item["target_id"] for item in tombstones if item["target_entity_type"] == "membership"
    }
    visual_relations = _current_visual_relations(
        [item.value for item in records.visual_relations], all_emojis, tombstoned_ids
    )

    taxonomy, taxonomy_sources, taxonomy_source_root = _taxonomy_registry(root)
    concepts = load_json(root / "taxonomy" / "v1" / "concepts.json")
    rights_registry = load_json(root / "rights" / "profiles.json")
    platform_registry, platform_sources, platform_source_root = _platform_registry(root)
    if (
        dataset["rights_defaults"]["project_profile_id"]
        != rights_registry["project_default_profile_id"]
    ):
        raise DataError("dataset/project rights defaults do not match")
    platform_by_id = {entry["platform"]: entry for entry in platform_registry["entries"]}
    singleton_values = {
        "platform-profiles": platform_registry,
        "rights-profiles": rights_registry,
        "taxonomy": taxonomy,
        "concepts": concepts,
    }
    singleton_payloads = {name: jcs_bytes(value) for name, value in singleton_values.items()}

    schema_paths = sorted(
        [*root.glob("schemas/v1/**/*.json"), *root.glob("schemas/distribution/v1/*.json")],
        key=lambda path: path.relative_to(root).as_posix().encode(),
    )
    schema_groups: dict[str, list[Path]] = {"canonical": [], "derived": [], "transport": []}
    for path in schema_paths:
        if path.is_relative_to(root / "schemas" / "v1"):
            schema_groups["canonical"].append(path)
        elif path.name in DERIVED_SCHEMA_NAMES:
            schema_groups["derived"].append(path)
        elif path.name in CANONICAL_DISTRIBUTION_SCHEMA_NAMES:
            schema_groups["canonical"].append(path)
        else:
            schema_groups["transport"].append(path)

    profiles: dict[str, dict[str, Any]] = {
        f"{scope}_schemas": {
            "id": f"{scope}-schemas-1.0.0",
            "root_scope": scope if scope != "transport" else "none",
            "sha256": aggregate_digest(
                (path.relative_to(root).as_posix(), path.read_bytes())
                for path in schema_groups[scope]
            ),
        }
        for scope in ("canonical", "derived", "transport")
    }
    analysis_documents: dict[str, tuple[Path, dict[str, Any], bytes]] = {}
    for field, (filename, root_scope) in PROFILE_FILES.items():
        path = root / "analysis-profiles" / filename
        body = load_json(path)
        profile_id = body.get("profile_id")
        if not isinstance(profile_id, str):
            raise DataError(f"{path}: analysis profile has no profile_id")
        # Selector, source resource, and manifest digest all pin the canonical
        # full wrapper, while the delegated contract pins the effective body.
        content = jcs_bytes(body)
        if path.read_bytes() != content:
            raise DataError(f"{path}: analysis profile source must be exact JCS bytes")
        if field in DELEGATED_PROFILE_TYPES:
            profile_type = DELEGATED_PROFILE_TYPES[field]
            contract_filename = PROFILE_CONTRACT_SCHEMA_FILES[field]
            contract_path = root / "schemas" / "distribution" / "v1" / contract_filename
            expected_ref = schema_uri(root, f"schemas/distribution/v1/{contract_filename}")
            if body.get("profile_type") != profile_type:
                raise DataError(f"{path}: delegated profile_type mismatch")
            if body.get("contract_schema_ref") != expected_ref:
                raise DataError(f"{path}: delegated contract_schema_ref mismatch")
            if body.get("contract_schema_sha256") != sha256_bytes(contract_path.read_bytes()):
                raise DataError(f"{path}: delegated contract schema digest mismatch")
            if not isinstance(body.get("body"), dict):
                raise DataError(f"{path}: delegated profile body must be an object")
        profiles[field] = {
            "id": profile_id,
            "root_scope": root_scope,
            "sha256": sha256_bytes(content),
        }
        analysis_documents[field] = (path, body, content)
    dataset_profile_pins = {
        "color": ("color_profile", "color_profile_sha256"),
        "dedupe": ("dedupe_profile", "dedupe_profile_sha256"),
        "collection_dedupe": (
            "collection_dedupe_profile",
            "collection_dedupe_profile_sha256",
        ),
    }
    for field, (id_field, sha_field) in dataset_profile_pins.items():
        if (
            dataset.get(id_field) != profiles[field]["id"]
            or dataset.get(sha_field) != profiles[field]["sha256"]
        ):
            raise DataError(
                f"dataset {id_field}/{sha_field} does not pin exact {field} source bytes"
            )
    profiles["taxonomy"] = {
        "id": taxonomy["registry_id"],
        "root_scope": "canonical",
        "sha256": sha256_bytes(singleton_payloads["taxonomy"]),
        "source_bundle_sha256": taxonomy_source_root,
    }
    profiles["concepts"] = {
        "id": concepts["registry_id"],
        "root_scope": "canonical",
        "sha256": sha256_bytes(singleton_payloads["concepts"]),
    }
    policies = {
        "platform_profiles": {
            "id": platform_registry["registry_id"],
            "root_scope": "canonical",
            "sha256": sha256_bytes(singleton_payloads["platform-profiles"]),
            "source_bundle_sha256": platform_source_root,
        },
        "rights_profiles": {
            "id": rights_registry["registry_id"],
            "root_scope": "canonical",
            "sha256": sha256_bytes(singleton_payloads["rights-profiles"]),
        },
    }
    selector_entries = [
        *(("profile", field, value["root_scope"], value) for field, value in profiles.items()),
        *(("policy", field, value["root_scope"], value) for field, value in policies.items()),
    ]

    canonical_contracts: dict[
        str, tuple[list[dict[str, Any]], str, list[str], list[str], list[str] | None]
    ] = {
        "collections": (
            collections,
            "schemas/v1/collection.schema.json",
            ["/id"],
            ["/id"],
            None,
        ),
        "emojis": (emojis, "schemas/v1/emoji.schema.json", ["/id"], ["/id"], None),
        "memberships": (
            memberships,
            "schemas/v1/membership.schema.json",
            ["/id"],
            ["/collection_id", "/status", "/position", "/id"],
            None,
        ),
        "tombstones": (
            tombstones,
            "schemas/v1/tombstone.schema.json",
            ["/target_entity_type", "/target_id"],
            ["/target_entity_type", "/target_id"],
            None,
        ),
        "visual-relations": (
            visual_relations,
            "schemas/v1/visual-relation.schema.json",
            ["/id"],
            ["/id"],
            None,
        ),
    }
    files: dict[str, bytes] = {}
    artifacts: list[dict[str, Any]] = []
    descriptor_by_name: dict[str, dict[str, Any]] = {}
    canonical_rows: dict[str, list[dict[str, Any]]] = {}
    for logical_name in (
        "collections",
        "emojis",
        "memberships",
        "tombstones",
        "visual-relations",
    ):
        rows, schema_path, primary_key, sort_key, routing_key = canonical_contracts[logical_name]
        rows = sort_records(rows, sort_key=sort_key, primary_key=primary_key)
        canonical_rows[logical_name] = rows
        payload = jsonl_bytes(rows)
        output_path = f"{logical_name}.jsonl"
        files[output_path] = payload
        descriptor = record_descriptor(
            logical_name=logical_name,
            semantic_role="canonical",
            path=output_path,
            schema_ref=schema_uri(root, schema_path),
            records=rows,
            primary_key=primary_key,
            sort_key=sort_key,
            routing_key=routing_key,
            payload=payload,
        )
        artifacts.append(descriptor)
        descriptor_by_name[logical_name] = descriptor

    singleton_schemas = {
        "platform-profiles": "platform-profiles-registry.schema.json",
        "rights-profiles": "rights-profiles-registry.schema.json",
        "taxonomy": "taxonomy-registry.schema.json",
        "concepts": "concepts-registry.schema.json",
    }
    for logical_name in ("platform-profiles", "rights-profiles", "taxonomy", "concepts"):
        output_path = f"{logical_name}.json"
        payload = singleton_payloads[logical_name]
        files[output_path] = payload
        descriptor = singleton_descriptor(
            logical_name=logical_name,
            path=output_path,
            schema_ref=schema_uri(
                root, f"schemas/distribution/v1/{singleton_schemas[logical_name]}"
            ),
            payload=payload,
        )
        artifacts.append(descriptor)
        descriptor_by_name[logical_name] = descriptor

    canonical_root = state_root(artifacts, selector_entries, derived=False)

    active_collections = {
        collection["id"]
        for collection in collections
        if collection["availability"]["status"] == "active"
        and collection["id"] not in tombstoned_ids
    }
    build_time = datetime.fromtimestamp(epoch, tz=UTC)

    def reject_future(record: dict[str, Any], field: str, value: str | None) -> None:
        if value is None:
            return
        observed_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if observed_at > build_time:
            record_id = record.get("id", record.get("target_id", "canonical-record"))
            raise DataError(
                f"canonical evidence {record_id}/{field} is future evidence for source_date_epoch"
            )

    for entity in [*collections, *emojis]:
        availability = entity["availability"]
        for field in ("first_seen_at", "last_changed_at", "last_verified_at"):
            reject_future(entity, f"availability/{field}", availability.get(field))
    for emoji in emojis:
        reject_future(emoji, "provenance/generated_at", emoji["provenance"].get("generated_at"))
        reject_future(emoji, "review/reviewed_at", emoji["review"].get("reviewed_at"))
    for membership in memberships:
        reject_future(membership, "first_seen_at", membership.get("first_seen_at"))
        reject_future(membership, "last_changed_at", membership.get("last_changed_at"))
    for tombstone in tombstones:
        reject_future(tombstone, "withheld_at", tombstone.get("withheld_at"))
    for relation in visual_relations:
        reject_future(relation, "review/reviewed_at", relation["review"].get("reviewed_at"))
    for platform_id, profile in platform_by_id.items():
        for capability in profile["capabilities"]:
            observed_at = datetime.fromisoformat(capability["observed_at"].replace("Z", "+00:00"))
            if observed_at > build_time:
                raise DataError(
                    f"platform capability {platform_id}/{capability['capability_id']} "
                    "is future evidence for source_date_epoch"
                )
    rights_by_emoji = {
        emoji["id"]: _rights_summary(emoji, platform_by_id, rights_registry, epoch)
        for emoji in emojis
    }
    eligible_ids = {
        emoji["id"]
        for emoji in emojis
        if emoji["id"] not in tombstoned_ids
        and emoji["availability"]["status"] == "active"
        and _eligible_review(emoji)
        and rights_by_emoji[emoji["id"]]["distribution_status"] == "allowed"
    }
    active_memberships = [
        membership
        for membership in memberships
        if membership["status"] == "active"
        and membership["id"] not in tombstoned_memberships
        and membership["collection_id"] in active_collections
        and membership["emoji_id"] in eligible_ids
    ]
    collection_ids_by_emoji: dict[str, list[str]] = defaultdict(list)
    for membership in active_memberships:
        collection_ids_by_emoji[membership["emoji_id"]].append(membership["collection_id"])
    for values in collection_ids_by_emoji.values():
        values.sort(key=str.encode)
    eligible_emojis = [
        all_emojis[emoji_id] for emoji_id in sorted(collection_ids_by_emoji, key=str.encode)
    ]
    duplicate_groups, duplicate_memberships = build_duplicate_groups(
        [emoji for emoji in emojis if emoji["id"] in eligible_ids],
        visual_relations,
        uuid.UUID(dataset["duplicate_group_namespace"]),
    )
    group_ids_by_emoji: dict[str, list[str]] = defaultdict(list)
    for member in duplicate_memberships:
        group_ids_by_emoji[member["emoji_id"]].append(member["group_id"])
    for values in group_ids_by_emoji.values():
        values[:] = sorted(set(values), key=str.encode)
    groups_by_id = {group["group_id"]: group for group in duplicate_groups}

    membership_key = [
        "/group_id",
        "/member_scope",
        "/emoji_id",
        "/role/present",
        "/role/value",
        "/variant/present",
        "/variant/value",
    ]
    derived_contracts = [
        (
            "emojis-active",
            eligible_emojis,
            ["/id"],
            ["/id"],
            None,
            "schemas/v1/emoji.schema.json",
            derived_from(
                canonical_root,
                manifest_values=[("/build/source_date_epoch", epoch)],
                selectors=[
                    ("/policies/platform_profiles", policies["platform_profiles"]),
                    ("/policies/rights_profiles", policies["rights_profiles"]),
                ],
            ),
        ),
        (
            "duplicate-groups",
            duplicate_groups,
            ["/group_type", "/group_id"],
            ["/group_type", "/group_id"],
            None,
            "schemas/distribution/v1/duplicate-group.schema.json",
            derived_from(
                canonical_root,
                manifest_values=[("/build/source_date_epoch", epoch)],
                selectors=[
                    ("/policies/platform_profiles", policies["platform_profiles"]),
                    ("/policies/rights_profiles", policies["rights_profiles"]),
                    ("/profiles/dedupe", profiles["dedupe"]),
                ],
            ),
        ),
        (
            "duplicate-group-memberships",
            duplicate_memberships,
            membership_key,
            membership_key,
            ["/group_id"],
            "schemas/distribution/v1/duplicate-group-membership.schema.json",
            derived_from(
                canonical_root,
                manifest_values=[("/build/source_date_epoch", epoch)],
                selectors=[
                    ("/policies/platform_profiles", policies["platform_profiles"]),
                    ("/policies/rights_profiles", policies["rights_profiles"]),
                    ("/profiles/dedupe", profiles["dedupe"]),
                ],
            ),
        ),
    ]
    for (
        logical_name,
        rows,
        primary_key,
        sort_key,
        routing_key,
        schema_path,
        dependency,
    ) in derived_contracts:
        rows = sort_records(rows, sort_key=sort_key, primary_key=primary_key)
        payload = jsonl_bytes(rows)
        output_path = f"{logical_name}.jsonl"
        files[output_path] = payload
        descriptor = record_descriptor(
            logical_name=logical_name,
            semantic_role="derived",
            path=output_path,
            schema_ref=schema_uri(root, schema_path),
            records=rows,
            primary_key=primary_key,
            sort_key=sort_key,
            routing_key=routing_key,
            payload=payload,
            derived_from=dependency,
        )
        artifacts.append(descriptor)
        descriptor_by_name[logical_name] = descriptor

    public_collections = [
        collection for collection in collections if collection["id"] in active_collections
    ]
    facets = _collection_facets(
        public_collections,
        active_memberships,
        {emoji["id"]: emoji for emoji in eligible_emojis},
        group_ids_by_emoji,
        groups_by_id,
        descriptor_by_name["memberships"]["table_root_sha256"],
        profiles["collection_dedupe"]["id"],
    )
    facet_payload = jsonl_bytes(facets)
    files["collection-facets.jsonl"] = facet_payload
    facet_descriptor = record_descriptor(
        logical_name="collection-facets",
        semantic_role="derived",
        path="collection-facets.jsonl",
        schema_ref=schema_uri(root, "schemas/distribution/v1/collection-facet.schema.json"),
        records=facets,
        primary_key=["/collection_id"],
        sort_key=["/collection_id"],
        payload=facet_payload,
        derived_from=derived_from(
            canonical_root,
            manifest_values=[("/build/source_date_epoch", epoch)],
            selectors=[
                ("/policies/platform_profiles", policies["platform_profiles"]),
                ("/policies/rights_profiles", policies["rights_profiles"]),
                ("/profiles/collection_dedupe", profiles["collection_dedupe"]),
            ],
            dependencies=[
                descriptor_by_name["duplicate-groups"],
                descriptor_by_name["duplicate-group-memberships"],
            ],
        ),
    )
    artifacts.append(facet_descriptor)
    descriptor_by_name["collection-facets"] = facet_descriptor

    capabilities = {
        platform_id: sorted(
            [
                item["capability_id"]
                for item in profile["capabilities"]
                if item["status"] == "supported" and item["applies_to"] == "custom_emoji"
            ],
            key=str.encode,
        )
        for platform_id, profile in platform_by_id.items()
    }
    search_rows: dict[str, list[dict[str, Any]]] = {language: [] for language in REQUIRED_LANGUAGES}
    for emoji in eligible_emojis:
        if not _eligible_search(emoji):
            continue
        for language in REQUIRED_LANGUAGES:
            if language not in emoji["descriptions"]:
                raise DataError(f"eligible emoji {emoji['id']} lacks required language {language}")
            search_rows[language].append(
                _search_record(
                    emoji,
                    language=language,
                    collection_ids=collection_ids_by_emoji[emoji["id"]],
                    duplicate_group_ids=group_ids_by_emoji.get(emoji["id"], []),
                    rights=rights_by_emoji[emoji["id"]],
                    capability_ids=capabilities[emoji["platform"]],
                )
            )

    canonical_snapshot_emojis = [emoji for emoji in emojis if emoji["id"] not in tombstoned_ids]
    available_languages = sorted(set(dataset["default_languages"]), key=str.encode)
    coverage = {
        language: {
            "described": sum(
                isinstance(emoji.get("descriptions", {}).get(language), dict)
                and bool(emoji["descriptions"][language].get("text"))
                for emoji in canonical_snapshot_emojis
            ),
            "total": len(canonical_snapshot_emojis),
        }
        for language in available_languages
    }
    for language in REQUIRED_LANGUAGES:
        if coverage.get(language, {}).get("described") != len(canonical_snapshot_emojis):
            raise DataError(f"required language {language} lacks complete canonical coverage")
    languages = {
        "required": REQUIRED_LANGUAGES,
        "available": available_languages,
        "localization_layout": "inline-v1",
        "canonicalization_profile": profiles["language_canonicalization"]["id"],
        "fallback_profile": profiles["language_fallback"]["id"],
        "coverage": coverage,
    }
    search_dependency = derived_from(
        canonical_root,
        manifest_values=[("/languages", languages), ("/build/source_date_epoch", epoch)],
        selectors=[
            ("/profiles/lexical_search", profiles["lexical_search"]),
            ("/policies/platform_profiles", policies["platform_profiles"]),
            ("/policies/rights_profiles", policies["rights_profiles"]),
        ],
        dependencies=[
            descriptor_by_name["duplicate-groups"],
            descriptor_by_name["duplicate-group-memberships"],
        ],
    )
    for language in REQUIRED_LANGUAGES:
        logical_name = f"search-{language}"
        rows = sort_records(
            search_rows[language], sort_key=["/emoji_id"], primary_key=["/emoji_id"]
        )
        payload = jsonl_bytes(rows)
        output_path = f"{logical_name}.jsonl"
        files[output_path] = payload
        descriptor = record_descriptor(
            logical_name=logical_name,
            semantic_role="derived",
            path=output_path,
            schema_ref=schema_uri(root, "schemas/distribution/v1/search-record.schema.json"),
            records=rows,
            primary_key=["/emoji_id"],
            sort_key=["/emoji_id"],
            payload=payload,
            derived_from=search_dependency,
        )
        artifacts.append(descriptor)
        descriptor_by_name[logical_name] = descriptor

    artifacts.sort(key=lambda item: (item["logical_name"].encode(), item["path"].encode()))

    resources: list[dict[str, Any]] = []
    for scope, paths in schema_groups.items():
        pointer = f"/profiles/{scope}_schemas/sha256"
        for source in paths:
            relative = source.relative_to(root).as_posix()
            destination = f"resources/{relative}"
            content = source.read_bytes()
            files[destination] = content
            resources.append(
                physical_resource(
                    uri=schema_uri(root, relative),
                    source_path=relative,
                    path=destination,
                    media_type="application/schema+json",
                    content=content,
                    bindings=[{"kind": "aggregate-member", "manifest_pointer": pointer}],
                )
            )

    delegated_schema = schema_uri(root, "schemas/distribution/v1/delegated-profile.schema.json")
    candidate_schema = schema_uri(
        root, "schemas/distribution/v1/concept-candidate-profile.schema.json"
    )
    for field, (source, _body, content) in analysis_documents.items():
        relative = source.relative_to(root).as_posix()
        destination = f"resources/profiles/{source.name}"
        files[destination] = content
        profile_id = profiles[field]["id"]
        resources.append(
            physical_resource(
                uri=(
                    f"mlx://profiles/{field.replace('_', '-')}/{profile_id}/"
                    f"{sha256_bytes(content)}.json"
                ),
                source_path=relative,
                path=destination,
                media_type="application/json",
                content=content,
                content_schema_ref=(
                    candidate_schema if field == "concept_candidates" else delegated_schema
                ),
                bindings=[
                    {
                        "kind": "exact-content",
                        "manifest_pointer": f"/profiles/{field}/sha256",
                    }
                ],
            )
        )

    taxonomy_source_schema = schema_uri(root, "schemas/distribution/v1/taxonomy-source.schema.json")
    taxonomy_dictionary_schema = schema_uri(
        root, "schemas/distribution/v1/taxonomy-dictionary.schema.json"
    )
    for source in taxonomy_sources:
        relative = source.relative_to(root).as_posix()
        destination = f"resources/registries/taxonomy-sources/{source.name}"
        content = source.read_bytes()
        files[destination] = content
        resources.append(
            physical_resource(
                uri=f"mlx://registries/taxonomy-source/{source.stem}/{sha256_bytes(content)}.json",
                source_path=relative,
                path=destination,
                media_type="application/json",
                content=content,
                content_schema_ref=(
                    taxonomy_source_schema
                    if source.name == "taxonomy.json"
                    else taxonomy_dictionary_schema
                ),
                bindings=[
                    {
                        "kind": "source-bundle-member",
                        "manifest_pointer": "/profiles/taxonomy/source_bundle_sha256",
                    }
                ],
            )
        )

    platform_schema = schema_uri(root, "schemas/distribution/v1/platform-profile.schema.json")
    for source in platform_sources:
        relative = source.relative_to(root).as_posix()
        destination = f"resources/registries/platform-sources/{source.name}"
        content = source.read_bytes()
        files[destination] = content
        entry = load_json(source)
        resources.append(
            physical_resource(
                uri=(
                    f"mlx://registries/platform-profile/{entry['profile_version']}/"
                    f"{sha256_bytes(content)}.json"
                ),
                source_path=relative,
                path=destination,
                media_type="application/json",
                content=content,
                content_schema_ref=platform_schema,
                bindings=[
                    {
                        "kind": "source-bundle-member",
                        "manifest_pointer": "/policies/platform_profiles/source_bundle_sha256",
                    }
                ],
            )
        )

    aliases = {
        "platform-profiles": ("policy", "platform_profiles", "platform-profiles"),
        "rights-profiles": ("policy", "rights_profiles", "rights-profiles"),
        "taxonomy": ("profile", "taxonomy", "facet-taxonomy"),
        "concepts": ("profile", "concepts", "concepts"),
    }
    for logical_name, (kind, field, registry_type) in aliases.items():
        selector = policies[field] if kind == "policy" else profiles[field]
        descriptor = descriptor_by_name[logical_name]
        resources.append(
            artifact_alias(
                uri=(
                    f"mlx://registries/{registry_type}/{selector['id']}/"
                    f"{descriptor['payload_sha256']}.json"
                ),
                logical_name=logical_name,
                payload_sha256=descriptor["payload_sha256"],
                manifest_pointer=(
                    f"/policies/{field}/sha256" if kind == "policy" else f"/profiles/{field}/sha256"
                ),
            )
        )
    resources.sort(key=lambda item: item["uri"].encode())
    bundles: list[dict[str, Any]] = []

    if state_root(artifacts, selector_entries, derived=False) != canonical_root:
        raise AssertionError("canonical state root changed after derived artifacts were added")
    derived_root = state_root(
        artifacts, selector_entries, derived=True, canonical_root=canonical_root
    )
    artifact_set = jcs_sha256({"artifacts": artifacts, "resources": resources, "bundles": bundles})

    storage = {"mode": "git-native"}
    git = {
        "repository": dataset["canonical_repository"],
        "commit": revision,
        "object_format": "sha1" if len(revision) == 40 else "sha256",
    }
    parameters = {
        "distribution_profile": profiles["distribution"]["id"],
        "layout_profile": "monolith-v1",
        "storage_mode": storage["mode"],
        "compression_profile": "none",
        "part_packing_profile": profiles["part_packing"]["id"],
        "bundle_mode": "none",
        "bundle_profile": profiles["bundling"]["id"],
        "partition_overrides": {},
    }
    parameters_hash = jcs_sha256(parameters)
    evidence_root = jcs_sha256([])
    migration_root = jcs_sha256([])
    build_base = {
        "tool": "mojilex-cli",
        "tool_version": tool_version,
        "tool_repository": "https://github.com/MojiLex/mojilex-cli",
        "tool_commit": tool_commit,
        "dependency_lock_sha256": dependency_lock_sha256,
        "distribution_profile": profiles["distribution"]["id"],
        "parameters": parameters,
        "parameters_sha256": parameters_hash,
        "evidence_inputs_root_sha256": evidence_root,
        "migration_inputs_root_sha256": migration_root,
        "build_input_profile_id": BUILD_INPUT_PROFILE_ID,
        "source_date_epoch": epoch,
    }
    build_input = {
        "build_input_profile_id": BUILD_INPUT_PROFILE_ID,
        "manifest_header": {
            "manifest_version": MANIFEST_VERSION,
            "dataset": dataset["dataset"],
            "snapshot_id": snapshot_id,
            "schema_version": dataset["schema_version"],
            "layout_profile": "monolith-v1",
            "trust_stage": "pre-enforcement",
            "minimum_reader_version": MINIMUM_READER_VERSION,
            "required_features": REQUIRED_FEATURES,
            "git": git,
            "languages": languages,
            "storage": storage,
        },
        "selectors": {"profiles": profiles, "policies": policies},
        "lineage": {
            "previous_snapshot": _present(exists=False),
            "change_set_batch": _present(exists=False),
            "migration_inputs_root_sha256": migration_root,
        },
        "builder": {
            key: build_base[key]
            for key in (
                "tool",
                "tool_version",
                "tool_repository",
                "tool_commit",
                "dependency_lock_sha256",
                "distribution_profile",
                "parameters",
                "parameters_sha256",
                "evidence_inputs_root_sha256",
                "source_date_epoch",
            )
        },
    }
    build = {**build_base, "build_inputs_sha256": jcs_sha256(build_input)}
    counts = {
        "collections": len(collections),
        "emojis": len(emojis),
        "active_emojis": sum(emoji["availability"]["status"] == "active" for emoji in emojis),
        "memberships": len(memberships),
        "tombstones": len(tombstones),
        "availability_by_status": {
            "collections": histogram(
                (item["availability"]["status"] for item in collections),
                _AVAILABILITY_STATUSES,
            ),
            "emojis": histogram(
                (item["availability"]["status"] for item in emojis),
                _AVAILABILITY_STATUSES,
            ),
        },
        "emoji_review_by_status": histogram(
            (item["review"]["status"] for item in emojis),
            _REVIEW_STATUSES,
        ),
        "memberships_by_status": histogram(
            (item["status"] for item in memberships),
            _MEMBERSHIP_STATUSES,
        ),
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset["dataset"],
        "snapshot_id": snapshot_id,
        "schema_version": dataset["schema_version"],
        "layout_profile": "monolith-v1",
        "trust_stage": "pre-enforcement",
        "minimum_reader_version": MINIMUM_READER_VERSION,
        "required_features": REQUIRED_FEATURES,
        "git": git,
        "languages": languages,
        "profiles": profiles,
        "policies": policies,
        "migrations": [],
        "storage": storage,
        "artifacts": artifacts,
        "resources": resources,
        "bundles": bundles,
        "counts": counts,
        "canonical_state_root_sha256": canonical_root,
        "derived_views_root_sha256": derived_root,
        "artifact_set_sha256": artifact_set,
        "build": build,
    }
    manifest_bytes = jcs_bytes(manifest)
    files["manifest.json"] = manifest_bytes
    checksums = {item["path"]: item["object_sha256"] for item in artifacts}
    checksums.update(
        {
            item["path"]: item["object_sha256"]
            for item in resources
            if item["resource_kind"] == "physical"
        }
    )
    checksums["manifest.json"] = sha256_bytes(manifest_bytes)
    files["SHA256SUMS"] = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(checksums.items())
    ).encode("ascii")
    return manifest, files

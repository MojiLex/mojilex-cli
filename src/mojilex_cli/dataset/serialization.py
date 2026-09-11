"""Deterministic UTF-8/LF serialization for canonical dataset files."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Never, cast

from mojilex_cli.domain.models import (
    Collection,
    Emoji,
    Entity,
    Membership,
    Tombstone,
    VisualRelation,
)

# This global order is shared verbatim with the formatter in the data repository.
# Keeping one rank table for every object also gives extension payloads a stable
# order without coupling this package to a particular platform model.
_KEY_ORDER = (
    "dataset",
    "schema_version",
    "id_namespace",
    "visual_relation_namespace",
    "taxonomy_version",
    "color_profile",
    "color_profile_sha256",
    "dedupe_profile",
    "dedupe_profile_sha256",
    "collection_dedupe_profile",
    "collection_dedupe_profile_sha256",
    "default_languages",
    "platforms",
    "canonical_repository",
    "licenses",
    "data",
    "code",
    "entity_type",
    "id",
    "platform",
    "kind",
    "native_namespace",
    "scope_id",
    "native_id",
    "identity_epoch",
    "title",
    "canonical_url",
    "availability",
    "item_count",
    "media",
    "fingerprints",
    "descriptions",
    "facets",
    "ru",
    "en",
    "text",
    "motion_status",
    "motion",
    "usage",
    "semantic_tags",
    "content",
    "rating",
    "warnings",
    "provenance",
    "origin",
    "provider",
    "model",
    "prompt_version",
    "pipeline_version",
    "description_profile",
    "model_revision",
    "prompt_sha256",
    "request_parameters_sha256",
    "qualification_id",
    "generation_stage",
    "routing_policy_version",
    "routing_reason_codes",
    "generated_at",
    "input_media_sha256",
    "created_at",
    "creator",
    "human_edits",
    "editor",
    "edited_at",
    "languages",
    "changed_paths",
    "tool",
    "name",
    "version",
    "review",
    "reviewed_at",
    "reviewer",
    "reviewed_content_sha256",
    "extensions",
    "telegram",
    "retrieved_via",
    "short_name",
    "sticker_type",
    "set_fingerprint_sha256",
    "stable_set_id",
    "custom_emoji_id",
    "file_unique_id",
    "fallback_emoji",
    "needs_repainting",
    "status",
    "first_seen_at",
    "last_changed_at",
    "last_verified_at",
    "reason_code",
    "set_by",
    "role",
    "variant_id",
    "format",
    "mime_type",
    "sha256",
    "byte_size",
    "width",
    "height",
    "animated",
    "duration_ms",
    "collection_id",
    "emoji_id",
    "position",
    "target_entity_type",
    "target_id",
    "withheld_at",
    "public_note",
    "rendering",
    "profile",
    "items",
    "color_behavior",
    "palette_dynamics",
    "alpha_mode",
    "visible_area_bp",
    "dominant_colors",
    "hex",
    "family",
    "coverage_bp",
    "adaptive_mask_source",
    "adaptive_mask_sha256",
    "text_content",
    "dynamics",
    "value",
    "script",
    "language",
    "temporal_scope",
    "media_refs",
    "content_types",
    "styles",
    "suggested_uses",
    "uncertainties",
    "input_media_digest",
    "decoded_payload_sha256",
    "canonical_render_sha256",
    "shape_sha256",
    "perceptual",
    "encoding",
    "sample_count",
    "layout_phash64",
    "content_phash64",
    "alpha_phash64",
    "edge_phash64",
    "temporal_energy_bp",
    "low_information",
    "subject_id",
    "object_id",
    "scope",
    "relation_type",
    "evidence",
    "subject_media_digest",
    "object_media_digest",
    "media_pairs",
    "subject_role",
    "subject_variant_id",
    "object_role",
    "object_variant_id",
    "signals",
    "reviewed_relation_sha256",
)
_KEY_RANK = {key: index for index, key in enumerate(_KEY_ORDER)}


def _context_order(value: Mapping[str, Any]) -> tuple[str, ...] | None:
    entity_type = value.get("entity_type")
    if entity_type == "collection":
        return (
            "schema_version",
            "entity_type",
            "id",
            "platform",
            "kind",
            "native_namespace",
            "scope_id",
            "native_id",
            "identity_epoch",
            "title",
            "canonical_url",
            "availability",
            "item_count",
            "extensions",
        )
    if entity_type == "emoji":
        return (
            "schema_version",
            "entity_type",
            "id",
            "platform",
            "native_namespace",
            "scope_id",
            "native_id",
            "identity_epoch",
            "availability",
            "media",
            "fingerprints",
            "descriptions",
            "facets",
            "semantic_tags",
            "content",
            "provenance",
            "review",
            "extensions",
        )
    if entity_type == "membership":
        return (
            "schema_version",
            "entity_type",
            "id",
            "collection_id",
            "emoji_id",
            "status",
            "position",
            "first_seen_at",
            "last_changed_at",
        )
    if entity_type == "tombstone":
        return (
            "schema_version",
            "entity_type",
            "target_entity_type",
            "target_id",
            "reason_code",
            "withheld_at",
            "public_note",
        )
    if entity_type == "visual_relation":
        return (
            "schema_version",
            "entity_type",
            "id",
            "identity_epoch",
            "subject_id",
            "object_id",
            "scope",
            "relation_type",
            "evidence",
            "review",
        )
    if {"taxonomy_version", "registries"}.issubset(value):
        return ("taxonomy_version", "registries")
    taxonomy_registries = (
        "color_families",
        "content_types",
        "platform_contexts",
        "styles",
        "suggested_uses",
        "uncertainties",
    )
    if value and set(value).issubset(taxonomy_registries):
        return taxonomy_registries
    if {"animated", "content_types", "literal_text", "media_kinds"}.issubset(value):
        return (
            "animated",
            "alpha_modes",
            "color_behaviors",
            "color_families",
            "content_types",
            "literal_text",
            "media_kinds",
            "styles",
            "suggested_uses",
            "text_status",
            "uncertainties",
        )
    if {"group_type", "scope", "members"}.issubset(value):
        return (
            "id",
            "group_type",
            "scope",
            "profile",
            "content_digest",
            "source_sha256",
            "byte_size",
            "members",
        )
    if {
        "taxonomy_version",
        "rendering",
        "text_content",
        "content_types",
    }.issubset(value):
        return (
            "taxonomy_version",
            "rendering",
            "text_content",
            "content_types",
            "styles",
            "suggested_uses",
            "uncertainties",
        )
    if "dataset" in value and "id_namespace" in value:
        return (
            "dataset",
            "schema_version",
            "id_namespace",
            "visual_relation_namespace",
            "taxonomy_version",
            "color_profile",
            "color_profile_sha256",
            "dedupe_profile",
            "dedupe_profile_sha256",
            "collection_dedupe_profile",
            "collection_dedupe_profile_sha256",
            "default_languages",
            "platforms",
            "canonical_repository",
            "licenses",
        )
    if "git_commit" in value and "payload_sha256" in value:
        return (
            "dataset",
            "schema_version",
            "git_commit",
            "counts",
            "status_counts",
            "profiles",
            "quality_registry_sha256",
            "platform_registry_sha256",
            "payload_sha256",
        )
    if {"status", "first_seen_at", "last_changed_at"}.issubset(value):
        return (
            "status",
            "first_seen_at",
            "last_changed_at",
            "last_verified_at",
            "reason_code",
            "set_by",
        )
    if {"role", "kind", "sha256"}.issubset(value):
        return (
            "role",
            "variant_id",
            "kind",
            "format",
            "mime_type",
            "sha256",
            "byte_size",
            "width",
            "height",
            "animated",
            "duration_ms",
        )
    if {"text", "motion_status", "usage"}.issubset(value):
        return ("text", "motion_status", "motion", "usage")
    if {"rating", "warnings"}.issubset(value):
        return ("rating", "warnings")
    if "origin" in value and "tool" in value:
        return (
            "origin",
            "provider",
            "model",
            "prompt_version",
            "pipeline_version",
            "description_profile",
            "model_revision",
            "prompt_sha256",
            "request_parameters_sha256",
            "qualification_id",
            "generation_stage",
            "routing_policy_version",
            "routing_reason_codes",
            "generated_at",
            "input_media_sha256",
            "created_at",
            "creator",
            "human_edits",
            "tool",
        )
    if {"editor", "edited_at", "languages"}.issubset(value):
        return ("editor", "edited_at", "languages", "changed_paths")
    if {"name", "version"}.issubset(value) and len(value) == 2:
        return ("name", "version")
    if "status" in value and set(value).issubset(
        {"status", "reviewed_at", "reviewer", "reviewed_content_sha256"}
    ):
        return ("status", "reviewed_at", "reviewer", "reviewed_content_sha256")
    if "status" in value and "reviewed_relation_sha256" in value:
        return ("status", "reviewer", "reviewed_at", "reviewed_relation_sha256")
    if {"profile", "items"}.issubset(value) and "input_media_digest" in value:
        return ("status", "profile", "input_media_digest", "items")
    if {"profile", "items"}.issubset(value):
        return ("profile", "items")
    if {"color_behavior", "palette_dynamics", "alpha_mode"}.issubset(value):
        return (
            "role",
            "variant_id",
            "color_behavior",
            "palette_dynamics",
            "alpha_mode",
            "visible_area_bp",
            "dominant_colors",
            "adaptive_mask_source",
            "adaptive_mask_sha256",
        )
    if {"hex", "family", "coverage_bp"}.issubset(value):
        return ("hex", "family", "coverage_bp")
    if {"status", "dynamics", "items"}.issubset(value):
        return ("status", "dynamics", "items")
    if {"value", "kind", "script", "temporal_scope", "media_refs"}.issubset(value):
        return ("value", "kind", "script", "language", "temporal_scope", "media_refs")
    if {"decoded_payload_sha256", "canonical_render_sha256", "shape_sha256"}.issubset(value):
        return (
            "role",
            "variant_id",
            "decoded_payload_sha256",
            "canonical_render_sha256",
            "shape_sha256",
            "perceptual",
        )
    if {"encoding", "sample_count", "layout_phash64"}.issubset(value):
        return (
            "encoding",
            "sample_count",
            "layout_phash64",
            "content_phash64",
            "alpha_phash64",
            "edge_phash64",
            "temporal_energy_bp",
            "low_information",
        )
    if {"dedupe_profile", "subject_media_digest", "object_media_digest"}.issubset(value):
        return (
            "dedupe_profile",
            "subject_media_digest",
            "object_media_digest",
            "media_pairs",
            "signals",
        )
    if {"subject_role", "object_role"}.issubset(value):
        return (
            "subject_role",
            "subject_variant_id",
            "object_role",
            "object_variant_id",
        )
    if {"schema_version", "retrieved_via", "short_name"}.issubset(value):
        return (
            "schema_version",
            "retrieved_via",
            "short_name",
            "sticker_type",
            "set_fingerprint_sha256",
            "stable_set_id",
        )
    if {"schema_version", "retrieved_via", "custom_emoji_id"}.issubset(value):
        return (
            "schema_version",
            "retrieved_via",
            "custom_emoji_id",
            "file_unique_id",
            "fallback_emoji",
            "needs_repainting",
        )
    if {"emoji_id", "collection_ids", "text"}.issubset(value):
        return (
            "emoji_id",
            "platform",
            "native_namespace",
            "scope_id",
            "native_id",
            "collection_ids",
            "text",
            "motion",
            "usage",
            "semantic_tags",
            "review_status",
        )
    if set(value).issubset({"data", "code"}):
        return ("data", "code")
    if "ru" in value and "en" in value:
        return ("ru", "en")
    return None


class DuplicateKeyError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"non-I-JSON numeric constant is forbidden: {value}")


def parse_json(data: bytes, *, source: str = "JSON") -> dict[str, Any]:
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{source}: UTF-8 BOM is forbidden")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: invalid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, DuplicateKeyError) as exc:
        raise ValueError(f"{source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source}: top-level value must be an object")
    return value


def parse_jsonl(data: bytes, *, source: str = "JSONL") -> list[dict[str, Any]]:
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{source}: UTF-8 BOM is forbidden")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: invalid UTF-8") from exc
    if text and not text.endswith("\n"):
        raise ValueError(f"{source}: final LF is required")
    if "\r" in text:
        raise ValueError(f"{source}: CR/CRLF line endings are forbidden")
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{source}:{line_number}: blank lines are forbidden")
        result.append(parse_json(line.encode("utf-8"), source=f"{source}:{line_number}"))
    return result


def _nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in result:
                raise ValueError(f"JSON object keys collide after NFC normalization: {key!r}")
            result[normalized] = _nfc(item)
        return result
    return value


def _ordered(value: Any) -> Any:
    if isinstance(value, list):
        return [_ordered(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    context = _context_order(value)
    rank = {key: index for index, key in enumerate(context)} if context else _KEY_RANK
    keys = sorted(value, key=lambda key: (rank.get(str(key), len(rank)), str(key)))
    result: dict[str, Any] = {}
    for key in keys:
        result[str(key)] = _ordered(value[key])
    return result


def canonical_entity(entity: Entity | Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        entity.as_dict()
        if isinstance(entity, (Collection, Emoji, Membership, Tombstone, VisualRelation))
        else dict(entity)
    )
    raw = _nfc(raw)
    if raw.get("entity_type") == "emoji":
        raw["semantic_tags"] = sorted(set(raw["semantic_tags"]))
        raw["content"]["warnings"] = sorted(set(raw["content"]["warnings"]))
        raw["media"] = sorted(
            raw["media"],
            key=lambda item: (item["role"], item.get("variant_id", ""), item["sha256"]),
        )
        raw["fingerprints"]["items"] = sorted(
            raw["fingerprints"]["items"],
            key=lambda item: (item["role"], item.get("variant_id", "")),
        )
        facets = raw["facets"]
        facets["rendering"]["items"] = sorted(
            facets["rendering"]["items"],
            key=lambda item: (item["role"], item.get("variant_id", "")),
        )
        for rendering in facets["rendering"]["items"]:
            if "dominant_colors" in rendering:
                rendering["dominant_colors"] = sorted(
                    rendering["dominant_colors"],
                    key=lambda item: (-item["coverage_bp"], item["family"], item["hex"]),
                )
        for text_item in facets["text_content"]["items"]:
            text_item["media_refs"] = sorted(
                text_item["media_refs"],
                key=lambda item: (item["role"], item.get("variant_id", "")),
            )
        for field in ("content_types", "styles", "suggested_uses", "uncertainties"):
            facets[field] = sorted(set(facets[field]))
        hashes = raw["provenance"].get("input_media_sha256")
        if hashes is not None:
            raw["provenance"]["input_media_sha256"] = sorted(set(hashes))
        routing_reasons = raw["provenance"].get("routing_reason_codes")
        if routing_reasons is not None:
            raw["provenance"]["routing_reason_codes"] = sorted(set(routing_reasons))
        for description in raw["descriptions"].values():
            description["usage"] = list(dict.fromkeys(description["usage"]))
    elif raw.get("entity_type") == "visual_relation":
        raw["evidence"]["media_pairs"] = sorted(
            raw["evidence"]["media_pairs"],
            key=lambda item: (
                item["subject_role"],
                item.get("subject_variant_id", ""),
                item["object_role"],
                item.get("object_variant_id", ""),
            ),
        )
        raw["evidence"]["signals"] = sorted(set(raw["evidence"]["signals"]))
    return cast(dict[str, Any], _ordered(raw))


def compact_json(value: Mapping[str, Any]) -> str:
    ordered = _ordered(_nfc(dict(value)))
    return json.dumps(ordered, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def pretty_json(value: Mapping[str, Any]) -> str:
    ordered = _ordered(_nfc(dict(value)))
    return json.dumps(ordered, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def serialize_collection(collection: Collection | Mapping[str, Any]) -> bytes:
    return pretty_json(canonical_entity(collection)).encode("utf-8")


def serialize_tombstone(tombstone: Tombstone | Mapping[str, Any]) -> bytes:
    return pretty_json(canonical_entity(tombstone)).encode("utf-8")


def serialize_emojis(emojis: Iterable[Emoji | Mapping[str, Any]]) -> bytes:
    values = sorted((canonical_entity(item) for item in emojis), key=lambda item: item["id"])
    if not values:
        return b""
    return ("\n".join(compact_json(item) for item in values) + "\n").encode("utf-8")


def _membership_sort_key(value: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(value["status"]), int(value["position"]), str(value["id"])


def serialize_memberships(memberships: Iterable[Membership | Mapping[str, Any]]) -> bytes:
    values = sorted((canonical_entity(item) for item in memberships), key=_membership_sort_key)
    if not values:
        return b""
    return ("\n".join(compact_json(item) for item in values) + "\n").encode("utf-8")


def serialize_visual_relations(
    relations: Iterable[VisualRelation | Mapping[str, Any]],
) -> bytes:
    values = sorted((canonical_entity(item) for item in relations), key=lambda item: item["id"])
    if not values:
        return b""
    return ("\n".join(compact_json(item) for item in values) + "\n").encode("utf-8")


def serialize_jsonl(values: Iterable[Mapping[str, Any]], *, sort_key: str = "id") -> bytes:
    ordered = sorted(
        (_ordered(_nfc(dict(item))) for item in values), key=lambda item: item[sort_key]
    )
    if not ordered:
        return b""
    return ("\n".join(compact_json(item) for item in ordered) + "\n").encode("utf-8")

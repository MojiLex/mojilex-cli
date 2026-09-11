"""RFC 8785 hashes shared by import, review, cache, and validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import rfc8785

from .models import Emoji, Media, VisualRelation


def jcs_bytes(value: Any) -> bytes:
    """Serialize an I-JSON value exactly as RFC 8785 JCS."""

    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise ValueError(f"value is not RFC 8785 canonicalizable: {exc}") from exc


def jcs_sha256(value: Any) -> str:
    return hashlib.sha256(jcs_bytes(value)).hexdigest()


def _media_identity(media: Media | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(media, Media):
        role = media.role.value if hasattr(media.role, "value") else str(media.role)
        variant_id = media.variant_id
        sha256 = media.sha256
    else:
        role = str(media["role"])
        raw_variant = media.get("variant_id")
        variant_id = str(raw_variant) if raw_variant is not None else None
        sha256 = str(media["sha256"])
    result = {"role": role}
    if variant_id is not None:
        result["variant_id"] = variant_id
    result["sha256"] = sha256
    return result


def media_identity_payload(media: Sequence[Media | Mapping[str, Any]]) -> list[dict[str, str]]:
    result = [_media_identity(item) for item in media]
    return sorted(
        result, key=lambda item: (item["role"], item.get("variant_id", ""), item["sha256"])
    )


def media_digest(media: Sequence[Media | Mapping[str, Any]]) -> str:
    return jcs_sha256(media_identity_payload(media))


def review_payload(emoji: Emoji | Mapping[str, Any]) -> dict[str, Any]:
    raw = emoji.as_dict() if isinstance(emoji, Emoji) else dict(emoji)
    return {
        "media": media_identity_payload(raw["media"]),
        "descriptions": raw["descriptions"],
        "facets": raw["facets"],
        "concept_ids": raw["concept_ids"],
        "semantic_tags": raw["semantic_tags"],
        "content": raw["content"],
        "provenance": raw["provenance"],
    }


def reviewed_content_sha256(emoji: Emoji | Mapping[str, Any]) -> str:
    return jcs_sha256(review_payload(emoji))


def relation_review_payload(
    relation: VisualRelation | Mapping[str, Any],
) -> dict[str, Any]:
    raw = relation.as_dict() if isinstance(relation, VisualRelation) else dict(relation)
    return {
        "identity_epoch": raw["identity_epoch"],
        "subject_id": raw["subject_id"],
        "object_id": raw["object_id"],
        "scope": raw["scope"],
        "relation_type": raw["relation_type"],
        "evidence": raw["evidence"],
    }


def reviewed_relation_sha256(
    relation: VisualRelation | Mapping[str, Any],
) -> str:
    return jcs_sha256(relation_review_payload(relation))


def telegram_set_fingerprint(
    pairs: Sequence[tuple[str, str] | Mapping[str, Any]],
) -> str:
    payload: list[dict[str, str]] = []
    for pair in pairs:
        if isinstance(pair, Mapping):
            custom_emoji_id = str(pair["custom_emoji_id"])
            file_unique_id = str(pair["file_unique_id"])
        else:
            custom_emoji_id, file_unique_id = pair
        payload.append({"custom_emoji_id": custom_emoji_id, "file_unique_id": file_unique_id})
    payload.sort(key=lambda item: (item["custom_emoji_id"], item["file_unique_id"]))
    return jcs_sha256(payload)

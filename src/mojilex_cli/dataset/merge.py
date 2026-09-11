"""Idempotent domain merge rules independent of adapters and Git."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any

from mojilex_cli.domain.hashes import media_digest, review_payload
from mojilex_cli.domain.models import (
    Collection,
    Emoji,
    Membership,
    MembershipStatus,
    Review,
    ReviewStatus,
)


class IdentityContinuity(StrEnum):
    NEW = "new"
    SAME = "same"
    CONFLICT = "conflict"


def assess_identity_continuity(
    previous_custom_emoji_ids: set[str], current_custom_emoji_ids: set[str]
) -> IdentityContinuity:
    if not previous_custom_emoji_ids:
        return IdentityContinuity.NEW
    if previous_custom_emoji_ids & current_custom_emoji_ids:
        return IdentityContinuity.SAME
    return IdentityContinuity.CONFLICT


def _deep_merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(old)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def merge_collection(
    existing: Collection,
    incoming: Collection,
    *,
    explicit_verification: bool = False,
) -> Collection:
    if existing.id != incoming.id:
        raise ValueError("cannot merge collections with different IDs")
    merged = incoming.model_copy(deep=True)
    merged.availability.first_seen_at = existing.availability.first_seen_at
    merged.extensions = _deep_merge(existing.extensions, incoming.extensions)
    if (
        not explicit_verification
        and existing.model_copy(update={"extensions": merged.extensions}).as_dict()
        == merged.model_copy(
            update={
                "availability": merged.availability.model_copy(
                    update={"last_verified_at": existing.availability.last_verified_at}
                )
            }
        ).as_dict()
    ):
        merged.availability.last_verified_at = existing.availability.last_verified_at
    return merged


def merge_emoji(
    existing: Emoji,
    incoming: Emoji,
    *,
    overwrite_reviewed: bool = False,
) -> Emoji:
    if existing.id != incoming.id:
        raise ValueError("cannot merge emoji with different IDs")
    merged = incoming.model_copy(deep=True)
    merged.availability.first_seen_at = existing.availability.first_seen_at
    merged.extensions = _deep_merge(existing.extensions, incoming.extensions)
    same_media = media_digest(existing.media) == media_digest(incoming.media)
    protected = (
        existing.review.status is ReviewStatus.APPROVED
        or existing.provenance.origin.value in {"human", "mixed"}
    )
    if same_media and protected and not overwrite_reviewed:
        merged = merged.model_copy(
            deep=True,
            update={
                "concept_ids": list(existing.concept_ids),
                "concept_mapping_status": existing.concept_mapping_status,
            },
        )
        merged.descriptions = deepcopy(existing.descriptions)
        merged.facets = existing.facets.model_copy(deep=True)
        merged.semantic_tags = list(existing.semantic_tags)
        merged.content = existing.content.model_copy(deep=True)
        merged.provenance = existing.provenance.model_copy(deep=True)
        merged.review = existing.review.model_copy(deep=True)
    elif not same_media or review_payload(existing) != review_payload(merged):
        merged.review = Review(status=ReviewStatus.UNREVIEWED)
    return Emoji.model_validate(merged.as_dict())


def merge_memberships(
    existing: list[Membership],
    current: list[Membership],
    *,
    changed_at: str,
) -> list[Membership]:
    """Merge one collection snapshot, retaining disappeared memberships."""

    old_by_pair = {(item.collection_id, item.emoji_id): item for item in existing}
    current_by_pair = {(item.collection_id, item.emoji_id): item for item in current}
    if len(current_by_pair) != len(current):
        raise ValueError("current memberships contain duplicate collection/emoji pairs")
    merged: list[Membership] = []
    for pair, incoming in current_by_pair.items():
        previous = old_by_pair.get(pair)
        if previous is None:
            merged.append(incoming.model_copy(deep=True))
            continue
        candidate = incoming.model_copy(deep=True)
        candidate.first_seen_at = previous.first_seen_at
        if previous.status == incoming.status and previous.position == incoming.position:
            candidate.last_changed_at = previous.last_changed_at
        else:
            candidate.last_changed_at = changed_at
        merged.append(candidate)
    for pair, previous in old_by_pair.items():
        if pair in current_by_pair:
            continue
        removed = previous.model_copy(deep=True)
        if removed.status is not MembershipStatus.REMOVED:
            removed.status = MembershipStatus.REMOVED
            removed.last_changed_at = changed_at
        merged.append(removed)
    return merged

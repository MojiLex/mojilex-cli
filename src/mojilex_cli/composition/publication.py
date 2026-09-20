"""Project verified composition membership into the public fragment marker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from mojilex_cli.dataset.repository import DatasetSnapshot
from mojilex_cli.dataset.serialization import parse_json
from mojilex_cli.domain.models import Emoji, Review
from mojilex_cli.media.resume import _read
from mojilex_cli.runs import RunCheckpoint
from mojilex_cli.sources.base import SourceCollection

from .detector import Composition


def defer_fragment_overflow_for_legacy_schema(snapshot: DatasetSnapshot) -> set[str]:
    """Keep old staged analyses valid; publish an overflow marker from saved evidence.

    Only the exact historical semanticTags definition qualifies. Neither unknown
    schema contracts nor concrete tags are changed, and schemas stay untouched.
    """
    try:
        path = snapshot.root / "schemas" / "v1" / "common.schema.json"
        schema = parse_json(_read(path, 1024 * 1024), source=str(path))
        definition = schema.get("$defs", {}).get("semanticTags")
    except (OSError, ValueError, TypeError, AttributeError):
        return set()
    if definition != {
        "type": "array",
        "minItems": 1,
        "maxItems": 12,
        "uniqueItems": True,
        "items": {
            "type": "string",
            "minLength": 1,
            "maxLength": 48,
            "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$",
        },
    }:
        return set()
    deferred: set[str] = set()
    for emoji in snapshot.emojis.values():
        if len(emoji.semantic_tags) == 13 and "fragment" in emoji.semantic_tags:
            emoji.semantic_tags = [tag for tag in emoji.semantic_tags if tag != "fragment"]
            deferred.add(emoji.id)
    return deferred


def mark_verified_fragments(
    snapshot: DatasetSnapshot,
    groups_by_pack: Mapping[str, Sequence[Composition]],
    sources: Sequence[SourceCollection],
) -> set[str]:
    """Mark only complete, current groups from successfully merged sources.

    The queue already verified tile hashes. Recheck every original against the
    final merged snapshot so a failed pack or a later conflicting pack cannot
    leak a stale or partial group's marker into publication.
    """
    changed: set[str] = set()
    latest: dict[tuple[str, str, str, str], Emoji] | None = None
    for source in sources:
        source_items = {item.native_id: item for item in source.items}
        for group in groups_by_pack.get(source.native_id, ()):
            if (
                not group.verified
                or group.detector not in {"composition-v2", "composition-v3"}
                or group.verification_passes != 3
                or not group.verifier_model
            ):
                continue
            if latest is None:
                latest = {}
                for record in snapshot.emojis.values():
                    identity = (
                        record.platform,
                        record.native_namespace,
                        record.scope_id,
                        record.native_id,
                    )
                    previous = latest.get(identity)
                    if previous is None or record.identity_epoch > previous.identity_epoch:
                        latest[identity] = record
            members: list[Emoji] = []
            for member in group.members:
                item = source_items.get(member.native_id)
                if item is None:
                    break
                emoji = latest.get(
                    (source.platform, item.native_namespace, item.scope_id, item.native_id)
                )
                if emoji is None or not any(
                    media.role.value == "primary" and media.sha256 == member.media_sha256
                    for media in emoji.media
                ):
                    break
                members.append(emoji)
            if len(members) != len(group.members):
                continue
            changed.update(_mark_members(members))
    return changed


def _mark_members(members: Sequence[Emoji]) -> set[str]:
    changed: set[str] = set()
    for emoji in members:
        if "fragment" in emoji.semantic_tags:
            continue
        # Do not silently clear a reviewer's rejection to add optional metadata.
        if emoji.review.status.value in {"rejected", "changes_requested"}:
            continue
        emoji.semantic_tags = sorted({*emoji.semantic_tags, "fragment"})
        # Tags participate in the reviewed payload; the old approval is stale.
        emoji.review = Review(status="unreviewed")
        changed.add(emoji.id)
    return changed


def mark_saved_fragments(snapshot: DatasetSnapshot, checkpoint: RunCheckpoint) -> set[str]:
    """Project retained confirmations when publishing an older saved description run.

    Require the whole group to match both the checkpoint originals and active
    membership in this snapshot. Malformed, overlapping or stale evidence cannot
    mark even its otherwise valid members. No AI request or cache write is needed.
    """
    original_tags = {key: tuple(value.semantic_tags) for key, value in snapshot.emojis.items()}
    stripped = strip_legacy_fragment_tags(snapshot, checkpoint)
    parameters = getattr(checkpoint, "safe_parameters", None)
    elements = getattr(checkpoint, "elements", None)
    if not isinstance(parameters, dict) or not isinstance(elements, dict):
        return stripped
    evidence = parameters.get("composition_evidence")
    memberships = parameters.get("source_memberships")
    if not isinstance(evidence, dict) or not isinstance(memberships, dict):
        return stripped
    changed: set[str] = set()
    for name, values in evidence.items():
        allowed = memberships.get(name) if isinstance(name, str) else None
        if (
            not isinstance(allowed, list)
            or not all(isinstance(value, str) for value in allowed)
            or not 2 <= len(allowed) <= 256
            or len(set(allowed)) != len(allowed)
            or not isinstance(values, list)
            or len(values) > 64
        ):
            continue
        collections = [
            item
            for item in snapshot.collections.values()
            if item.platform == "telegram"
            and item.native_id == name
            and item.availability.status.value == "active"
        ]
        if len(collections) != 1:
            continue
        active_ids = {
            item.emoji_id
            for item in snapshot.memberships.values()
            if item.collection_id == collections[0].id and item.status.value == "active"
        }
        active: dict[str, list[Emoji]] = {}
        for identifier in active_ids:
            emoji = snapshot.emojis.get(identifier)
            if (
                emoji is not None
                and emoji.platform == "telegram"
                and emoji.native_namespace == "custom_emoji.id"
                and emoji.scope_id == "global"
                and emoji.availability.status.value == "active"
            ):
                active.setdefault(emoji.native_id, []).append(emoji)
        accepted: list[list[Emoji]] = []
        occurrences: dict[str, int] = {}
        for raw in values:
            try:
                group = Composition.model_validate(raw, strict=True)
            except (ValueError, TypeError):
                continue
            if (
                not group.verified
                or group.detector not in {"composition-v2", "composition-v3"}
                or group.verification_passes != 3
                or not group.verifier_model
            ):
                continue
            current: list[Emoji] = []
            for member in group.members:
                matches = active.get(member.native_id, [])
                element = elements.get(member.native_id)
                if (
                    member.native_id not in allowed
                    or len(matches) != 1
                    or element is None
                    or getattr(element, "media_sha256", None) != (member.media_sha256,)
                ):
                    break
                emoji = matches[0]
                if (
                    not any(
                        media.role.value == "primary"
                        and media.variant_id is None
                        and not media.animated
                        and media.sha256 == member.media_sha256
                        for media in emoji.media
                    )
                    or emoji.extensions.get("telegram", {}).get("needs_repainting") is not False
                ):
                    break
                current.append(emoji)
            if len(current) == len(group.members):
                accepted.append(current)
                for emoji in current:
                    occurrences[emoji.id] = occurrences.get(emoji.id, 0) + 1
        for current in accepted:
            if all(occurrences[emoji.id] == 1 for emoji in current):
                changed.update(_mark_members(current))
    return {
        key
        for key, value in snapshot.emojis.items()
        if tuple(value.semantic_tags) != original_tags[key]
    }


def strip_legacy_fragment_tags(snapshot: DatasetSnapshot, checkpoint: RunCheckpoint) -> set[str]:
    """Remove a formerly free-form AI word only from this legacy run's own output.

    A retained run must bind completed semantic processing, original media,
    source membership, provider/model and generation time. Existing repository
    records inherited from before the run and human review decisions survive.
    New marker-aware runs never reinterpret a published fragment marker.
    """
    parameters = getattr(checkpoint, "safe_parameters", None)
    elements = getattr(checkpoint, "elements", None)
    if (
        not isinstance(parameters, dict)
        or not isinstance(elements, dict)
        or "public_fragment_marker_version" in parameters
    ):
        return set()
    memberships = parameters.get("source_memberships")
    if not isinstance(memberships, dict):
        return set()
    changed: set[str] = set()
    for name, allowed in memberships.items():
        if (
            not isinstance(name, str)
            or not isinstance(allowed, list)
            or not all(isinstance(value, str) for value in allowed)
            or not allowed
            or len(set(allowed)) != len(allowed)
        ):
            continue
        collections = [
            collection
            for collection in snapshot.collections.values()
            if collection.platform == "telegram"
            and collection.native_id == name
            and collection.availability.status.value == "active"
        ]
        if len(collections) != 1:
            continue
        active_ids = {
            item.emoji_id
            for item in snapshot.memberships.values()
            if item.collection_id == collections[0].id and item.status.value == "active"
        }
        for identifier in active_ids:
            emoji = snapshot.emojis.get(identifier)
            if (
                emoji is None
                or "fragment" not in emoji.semantic_tags
                or len(emoji.semantic_tags) <= 1
                or emoji.platform != "telegram"
                or emoji.native_namespace != "custom_emoji.id"
                or emoji.scope_id != "global"
                or emoji.availability.status.value != "active"
                or emoji.native_id not in allowed
                or emoji.provenance.origin.value != "ai"
                or emoji.review.status.value != "unreviewed"
                or emoji.provenance.provider != parameters.get("provider")
                or emoji.provenance.model
                not in (parameters.get("model"), parameters.get("escalation_model"))
            ):
                continue
            element = elements.get(emoji.native_id)
            if (
                element is None
                or not getattr(element, "ai_facets_complete", False)
                or not getattr(element, "source_descriptor_sha256", None)
                or getattr(element, "media_sha256", None)
                != tuple(sorted(media.sha256 for media in emoji.media))
                or not _generated_within_run(emoji, checkpoint)
            ):
                continue
            emoji.semantic_tags = [tag for tag in emoji.semantic_tags if tag != "fragment"]
            changed.add(identifier)
    return changed


def _generated_within_run(emoji: Emoji, checkpoint: RunCheckpoint) -> bool:
    def timestamp(value: object) -> datetime:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, str):
            result = datetime.fromisoformat(value)
        else:
            raise ValueError("missing timestamp")
        if result.tzinfo is None:
            raise ValueError("unbound timezone")
        return result

    try:
        return (
            timestamp(checkpoint.created_at)
            <= timestamp(emoji.provenance.generated_at)
            <= timestamp(checkpoint.updated_at)
        )
    except (ValueError, TypeError, AttributeError):
        return False

"""Pure source/media/AI-to-domain transformation and merge planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mojilex_cli import __version__
from mojilex_cli.ai import DescriptionItem
from mojilex_cli.dataset import (
    DatasetSnapshot,
    IdentityContinuity,
    assess_identity_continuity,
    merge_collection,
    merge_emoji,
    merge_memberships,
)
from mojilex_cli.domain import (
    SCHEMA_VERSION,
    Availability,
    Collection,
    Content,
    DeterministicEmojiAnalysis,
    Emoji,
    Facets,
    LocalizedDescription,
    Media,
    Membership,
    Provenance,
    Review,
    ToolProvenance,
    collection_id,
    emoji_id,
    media_digest,
    membership_id,
    utc_now,
)
from mojilex_cli.media import PIPELINE_VERSION, ProcessedMedia
from mojilex_cli.sources import SourceCollection, SourceEmoji

PROMPT_VERSION = "1.0.0"


class IdentityConflictError(ValueError):
    code = "IDENTITY_CONFLICT"


@dataclass(frozen=True, slots=True)
class SemanticGenerationMetadata:
    """Exact per-item public provenance for the accepted semantic AI object."""

    provider: str
    model: str
    prompt_sha256: str
    request_parameters_sha256: str
    qualification_id: str | None = None
    model_revision: str | None = None
    description_profile: str = "standard-v1"
    generation_stage: str = "primary"
    routing_policy_version: str = "1.0.0"
    routing_reason_codes: tuple[str, ...] = ()
    generated_at: str | None = None


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    snapshot: DatasetSnapshot
    collection_id: str
    created: int
    updated: int
    removed_memberships: int
    changed_entity_ids: tuple[str, ...]


def plan_collection_merge(
    snapshot: DatasetSnapshot,
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
    descriptions: Mapping[str, DescriptionItem],
    analyses: Mapping[str, DeterministicEmojiAnalysis],
    generation_metadata: Mapping[str, SemanticGenerationMetadata],
    *,
    overwrite_reviewed: bool = False,
    new_identity: bool = False,
    same_identity: bool = False,
    explicit_verification: bool = False,
    timestamp: str | None = None,
) -> CollectionPlan:
    """Create a complete, validated-in-memory collection update.

    ``processed`` and ``descriptions`` are keyed by source-native emoji ID. The
    function mutates only a deep clone, so one failing collection is atomic.
    """

    if new_identity and same_identity:
        raise ValueError("--new-identity and --same-identity are mutually exclusive")
    native_ids = {item.native_id for item in source.items}
    if (
        set(processed) != native_ids
        or set(descriptions) != native_ids
        or set(analyses) != native_ids
        or set(generation_metadata) != native_ids
    ):
        raise ValueError(
            "media, descriptions, analyses, and provenance must cover every source item exactly"
        )
    now = timestamp or utc_now()
    after = snapshot.clone()
    existing_collection, epoch, continuity = _resolve_collection_identity(snapshot, source)
    if continuity is IdentityContinuity.CONFLICT:
        if not new_identity and not same_identity:
            raise IdentityConflictError(
                "The source short name resolves to a disjoint set of native emoji IDs."
            )
        if new_identity:
            epoch += 1
            if existing_collection is not None:
                previous = after.collections[existing_collection.id]
                previous.availability = Availability(
                    status="unavailable",
                    first_seen_at=previous.availability.first_seen_at,
                    last_changed_at=now,
                    last_verified_at=now,
                    reason_code="identity_replaced",
                )
            existing_collection = None

    collection = _collection(source, epoch=epoch, now=now)
    if existing_collection is not None:
        collection = merge_collection(
            existing_collection,
            collection,
            explicit_verification=explicit_verification,
        )
        collection.availability.last_changed_at = (
            existing_collection.availability.last_changed_at
            if existing_collection.availability.status == collection.availability.status
            else now
        )
        if not explicit_verification and existing_collection.availability.status.value == "active":
            collection.availability.last_verified_at = (
                existing_collection.availability.last_verified_at
            )
    after.collections[collection.id] = collection

    created = 0
    updated = 0
    changed_ids: list[str] = [collection.id]
    current_memberships: list[Membership] = []
    for source_item in source.items:
        existing_emoji = _find_emoji(snapshot, source.platform, source_item)
        emoji_epoch = existing_emoji.identity_epoch if existing_emoji is not None else 0
        incoming = _emoji(
            source.platform,
            source_item,
            processed[source_item.native_id],
            descriptions[source_item.native_id],
            analyses[source_item.native_id],
            generation_metadata[source_item.native_id],
            manifest=after.manifest,
            epoch=emoji_epoch,
            now=now,
        )
        if existing_emoji is None:
            created += 1
            merged_emoji = incoming
        else:
            merged_emoji = merge_emoji(
                existing_emoji,
                incoming,
                overwrite_reviewed=overwrite_reviewed,
            )
            same_media = media_digest(existing_emoji.media) == media_digest(incoming.media)
            protected = (
                existing_emoji.review.status.value == "approved"
                or existing_emoji.provenance.origin.value in {"human", "mixed"}
            )
            if same_media and protected and not overwrite_reviewed:
                protected_facets = existing_emoji.facets.model_copy(deep=True)
                protected_facets.rendering = incoming.facets.rendering.model_copy(deep=True)
                merged_emoji.facets = protected_facets
                if merged_emoji.facets != existing_emoji.facets:
                    # Rendering is deterministic rather than human/AI-authored, but it is
                    # still part of the semantic review payload per MLX-SPEC-002.
                    merged_emoji.review = Review(status="unreviewed")
            same_semantics = (
                existing_emoji.descriptions == incoming.descriptions
                and existing_emoji.facets == incoming.facets
                and existing_emoji.semantic_tags == incoming.semantic_tags
                and existing_emoji.content == incoming.content
            )
            if same_media and same_semantics and not overwrite_reviewed:
                merged_emoji.descriptions = existing_emoji.model_copy(deep=True).descriptions
                merged_emoji.facets = existing_emoji.facets.model_copy(deep=True)
                merged_emoji.semantic_tags = list(existing_emoji.semantic_tags)
                merged_emoji.content = existing_emoji.content.model_copy(deep=True)
                merged_emoji.provenance = existing_emoji.provenance.model_copy(deep=True)
                merged_emoji.review = existing_emoji.review.model_copy(deep=True)
            merged_emoji.availability.last_changed_at = (
                existing_emoji.availability.last_changed_at
                if existing_emoji.availability.status == merged_emoji.availability.status
                else now
            )
            if not explicit_verification and existing_emoji.availability.status.value == "active":
                merged_emoji.availability.last_verified_at = (
                    existing_emoji.availability.last_verified_at
                )
            if merged_emoji.as_dict() != existing_emoji.as_dict():
                updated += 1
        after.emojis[merged_emoji.id] = merged_emoji
        changed_ids.append(merged_emoji.id)
        current_memberships.append(
            _membership(collection.id, merged_emoji.id, source_item.position, now)
        )

    old_memberships = [
        value for value in snapshot.memberships.values() if value.collection_id == collection.id
    ]
    merged_memberships = merge_memberships(old_memberships, current_memberships, changed_at=now)
    before_active = {item.id for item in old_memberships if item.status.value == "active"}
    after_active = {item.id for item in merged_memberships if item.status.value == "active"}
    removed = len(before_active - after_active)
    for value in merged_memberships:
        after.memberships[value.id] = value
        changed_ids.append(value.id)
    collection.item_count = len(after_active)
    return CollectionPlan(
        snapshot=after,
        collection_id=collection.id,
        created=created,
        updated=updated,
        removed_memberships=removed,
        changed_entity_ids=tuple(dict.fromkeys(changed_ids)),
    )


def _resolve_collection_identity(
    snapshot: DatasetSnapshot, source: SourceCollection
) -> tuple[Collection | None, int, IdentityContinuity]:
    matches = [
        value
        for value in snapshot.collections.values()
        if (
            value.platform,
            value.native_namespace,
            value.scope_id,
            value.native_id,
        )
        == (source.platform, source.native_namespace, source.scope_id, source.native_id)
    ]
    if not matches:
        return None, 0, IdentityContinuity.NEW
    previous = max(matches, key=lambda value: value.identity_epoch)
    previous_native_ids = {
        snapshot.emojis[item.emoji_id].native_id
        for item in snapshot.memberships.values()
        if item.collection_id == previous.id and item.emoji_id in snapshot.emojis
    }
    current_native_ids = {item.native_id for item in source.items}
    return (
        previous,
        previous.identity_epoch,
        assess_identity_continuity(previous_native_ids, current_native_ids),
    )


def _find_emoji(snapshot: DatasetSnapshot, platform: str, source: SourceEmoji) -> Emoji | None:
    matches = [
        value
        for value in snapshot.emojis.values()
        if (
            value.platform,
            value.native_namespace,
            value.scope_id,
            value.native_id,
        )
        == (platform, source.native_namespace, source.scope_id, source.native_id)
    ]
    return max(matches, key=lambda value: value.identity_epoch) if matches else None


def _availability(now: str) -> Availability:
    return Availability(
        status="active",
        first_seen_at=now,
        last_changed_at=now,
        last_verified_at=now,
    )


def _collection(source: SourceCollection, *, epoch: int, now: str) -> Collection:
    identifier = collection_id(
        source.platform,
        source.native_namespace,
        source.scope_id,
        source.native_id,
        epoch,
    )
    return Collection(
        schema_version=SCHEMA_VERSION,
        entity_type="collection",
        id=identifier,
        platform=source.platform,
        kind=source.kind,
        native_namespace=source.native_namespace,
        scope_id=source.scope_id,
        native_id=source.native_id,
        identity_epoch=epoch,
        title=source.title,
        canonical_url=source.canonical_url,
        availability=_availability(now),
        item_count=len(source.items),
        extensions={source.platform: dict(source.extension)},
    )


def _emoji(
    platform: str,
    source: SourceEmoji,
    processed: ProcessedMedia,
    description: DescriptionItem,
    analysis: DeterministicEmojiAnalysis,
    generation: SemanticGenerationMetadata,
    *,
    manifest: Mapping[str, object],
    epoch: int,
    now: str,
) -> Emoji:
    identifier = emoji_id(
        platform,
        source.native_namespace,
        source.scope_id,
        source.native_id,
        epoch,
    )
    media = Media.model_validate(processed.dataset_metadata())
    expected_profiles = {
        "color_profile": analysis.rendering.profile,
        "color_profile_sha256": analysis.color_profile_sha256,
        "dedupe_profile": analysis.fingerprints.profile,
        "dedupe_profile_sha256": analysis.dedupe_profile_sha256,
    }
    for field, actual in expected_profiles.items():
        if manifest.get(field) != actual:
            raise ValueError(f"deterministic analysis {field} does not match dataset manifest")
    rendering_primary = next(
        (
            item
            for item in analysis.rendering.items
            if item.role.value == "primary" and item.variant_id is None
        ),
        None,
    )
    expected_color_behavior = "platform-adaptive" if source.needs_repainting else "fixed"
    if platform == "telegram" and (
        rendering_primary is None
        or rendering_primary.color_behavior.value != expected_color_behavior
    ):
        raise ValueError(
            "Telegram primary rendering must be derived from needs_repainting deterministically"
        )
    localized = {
        language: LocalizedDescription(
            text=value.text,
            motion_status=value.motion_status,
            motion=value.motion,
            usage=list(value.usage),
        )
        for language, value in {
            "ru": description.descriptions.ru,
            "en": description.descriptions.en,
        }.items()
    }
    extension: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_via": "bot_api",
        "custom_emoji_id": source.native_id,
        "file_unique_id": source.file_unique_id,
        "needs_repainting": source.needs_repainting,
    }
    if source.fallback_emoji is not None:
        extension["fallback_emoji"] = source.fallback_emoji
    facets = Facets.model_validate(
        {
            "taxonomy_version": manifest.get("taxonomy_version"),
            "rendering": analysis.rendering.as_dict(),
            **description.facets.model_dump(mode="json"),
        }
    )
    return Emoji(
        schema_version=SCHEMA_VERSION,
        entity_type="emoji",
        id=identifier,
        platform=platform,
        native_namespace=source.native_namespace,
        scope_id=source.scope_id,
        native_id=source.native_id,
        identity_epoch=epoch,
        availability=_availability(now),
        media=[media],
        fingerprints=analysis.fingerprints.model_copy(deep=True),
        descriptions=localized,
        facets=facets,
        semantic_tags=sorted(description.semantic_tags),
        content=Content(
            rating=description.content.rating,
            warnings=sorted(description.content.warnings),
        ),
        provenance=Provenance(
            origin="ai",
            provider=generation.provider,
            model=generation.model,
            model_revision=generation.model_revision,
            prompt_version=PROMPT_VERSION,
            pipeline_version=PIPELINE_VERSION,
            description_profile=generation.description_profile,
            prompt_sha256=generation.prompt_sha256,
            request_parameters_sha256=generation.request_parameters_sha256,
            qualification_id=generation.qualification_id,
            generation_stage=generation.generation_stage,
            routing_policy_version=generation.routing_policy_version,
            routing_reason_codes=sorted(generation.routing_reason_codes),
            generated_at=generation.generated_at or now,
            input_media_sha256=[media.sha256],
            tool=ToolProvenance(name="mojilex-cli", version=__version__),
        ),
        review=Review(status="unreviewed"),
        extensions={platform: extension},
    )


def _membership(collection: str, emoji: str, position: int, now: str) -> Membership:
    return Membership(
        schema_version=SCHEMA_VERSION,
        entity_type="membership",
        id=membership_id(collection, emoji),
        collection_id=collection,
        emoji_id=emoji,
        status="active",
        position=position,
        first_seen_at=now,
        last_changed_at=now,
    )

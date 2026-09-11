from __future__ import annotations

import pytest

from mojilex_cli.ai import (
    BilingualDescriptions,
    ContentClassification,
    DescriptionItem,
    SemanticFacets,
)
from mojilex_cli.ai import LocalizedDescription as AILocalizedDescription
from mojilex_cli.domain import (
    DeterministicEmojiAnalysis,
    RenderingFacets,
    RenderingItem,
    Review,
    reviewed_content_sha256,
)
from mojilex_cli.media import MediaMetadata, ProcessedMedia
from mojilex_cli.pipeline.transform import (
    IdentityConflictError,
    SemanticGenerationMetadata,
    plan_collection_merge,
)
from mojilex_cli.sources import SourceCollection, SourceEmoji
from test_dataset_helpers import make_snapshot


def _description(snapshot) -> DescriptionItem:  # type: ignore[no-untyped-def]
    emoji = next(iter(snapshot.emojis.values()))
    return DescriptionItem(
        label="E001",
        descriptions=BilingualDescriptions(
            ru=AILocalizedDescription.model_validate(emoji.descriptions["ru"].model_dump()),
            en=AILocalizedDescription.model_validate(emoji.descriptions["en"].model_dump()),
        ),
        facets=SemanticFacets.model_validate(
            emoji.facets.model_dump(
                include={
                    "text_content",
                    "content_types",
                    "styles",
                    "suggested_uses",
                    "uncertainties",
                }
            )
        ),
        semantic_tags=tuple(emoji.semantic_tags),
        content=ContentClassification(
            rating=emoji.content.rating,
            warnings=tuple(emoji.content.warnings),
        ),
    )


def _analysis(snapshot) -> DeterministicEmojiAnalysis:  # type: ignore[no-untyped-def]
    emoji = next(iter(snapshot.emojis.values()))
    return DeterministicEmojiAnalysis(
        color_profile_sha256=snapshot.manifest["color_profile_sha256"],
        dedupe_profile_sha256=snapshot.manifest["dedupe_profile_sha256"],
        rendering=emoji.facets.rendering.model_copy(deep=True),
        fingerprints=emoji.fingerprints.model_copy(deep=True),
    )


def _generation() -> SemanticGenerationMetadata:
    return SemanticGenerationMetadata(
        provider="gemini",
        model="test-model",
        model_revision="test-revision",
        prompt_sha256="1" * 64,
        request_parameters_sha256="2" * 64,
        qualification_id="mq_standard-v1_test-001",
    )


def _source(snapshot, *, native_id: str | None = None) -> SourceCollection:  # type: ignore[no-untyped-def]
    collection = next(iter(snapshot.collections.values()))
    emoji = next(iter(snapshot.emojis.values()))
    telegram = emoji.extensions["telegram"]
    selected_native_id = native_id or emoji.native_id
    source_emoji = SourceEmoji(
        native_namespace="custom_emoji.id",
        scope_id="global",
        native_id=selected_native_id,
        file_unique_id=str(telegram["file_unique_id"]),
        position=0,
        width=emoji.media[0].width,
        height=emoji.media[0].height,
        animated=False,
        video=False,
        needs_repainting=False,
        fallback_emoji=str(telegram["fallback_emoji"]),
        declared_file_size=emoji.media[0].byte_size,
        media_format="webp",
        file_id="transient-file-id",
    )
    return SourceCollection(
        platform="telegram",
        kind="custom_emoji_set",
        native_namespace="sticker_set.name",
        scope_id="global",
        native_id=collection.native_id,
        title=collection.title,
        canonical_url=collection.canonical_url or "",
        item_count=1,
        items=(source_emoji,),
        extension=dict(collection.extensions["telegram"]),
    )


def _processed(snapshot, tmp_path) -> ProcessedMedia:  # type: ignore[no-untyped-def]
    emoji = next(iter(snapshot.emojis.values()))
    return ProcessedMedia(
        metadata=MediaMetadata.model_validate(
            emoji.media[0].model_dump(exclude={"variant_id"}, exclude_none=True)
        ),
        frame_paths=(tmp_path / "synthetic-frame.png",),
    )


def test_unchanged_collection_plan_is_byte_identical_and_requires_no_update(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path.resolve())
    source = _source(snapshot)
    native_id = source.items[0].native_id
    plan = plan_collection_merge(
        snapshot,
        source,
        {native_id: _processed(snapshot, tmp_path)},
        {native_id: _description(snapshot)},
        {native_id: _analysis(snapshot)},
        {native_id: _generation()},
        timestamp="2026-09-11T18:00:00Z",
    )

    assert plan.created == 0
    assert plan.updated == 0
    assert plan.removed_memberships == 0
    assert plan.snapshot.to_files() == snapshot.to_files()


def test_disjoint_collection_requires_explicit_identity_decision(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path.resolve())
    source = _source(snapshot, native_id="9999999999999999999")
    native_id = source.items[0].native_id
    processed = {native_id: _processed(snapshot, tmp_path)}
    descriptions = {native_id: _description(snapshot)}

    with pytest.raises(IdentityConflictError):
        plan_collection_merge(
            snapshot,
            source,
            processed,
            descriptions,
            {native_id: _analysis(snapshot)},
            {native_id: _generation()},
        )

    plan = plan_collection_merge(
        snapshot,
        source,
        processed,
        descriptions,
        {native_id: _analysis(snapshot)},
        {native_id: _generation()},
        new_identity=True,
        timestamp="2026-09-11T18:00:00Z",
    )
    new_collection = plan.snapshot.collections[plan.collection_id]
    previous = next(
        value for value in plan.snapshot.collections.values() if value.id != plan.collection_id
    )
    assert new_collection.identity_epoch == 1
    assert previous.availability.status == "unavailable"
    assert previous.availability.reason_code == "identity_replaced"


def test_deterministic_rendering_change_resets_review_but_protects_semantic_facets(
    tmp_path,
) -> None:
    snapshot = make_snapshot(tmp_path.resolve())
    existing = next(iter(snapshot.emojis.values()))
    existing.review = Review(
        status="approved",
        reviewed_at="2026-09-11T17:00:00Z",
        reviewer="reviewer",
        reviewed_content_sha256=reviewed_content_sha256(existing),
    )
    source = _source(snapshot)
    native_id = source.items[0].native_id
    analysis = _analysis(snapshot)
    changed_rendering = RenderingItem.model_validate(
        {**analysis.rendering.items[0].as_dict(), "visible_area_bp": 6000}
    )
    analysis.rendering = RenderingFacets(
        profile=analysis.rendering.profile,
        items=[changed_rendering],
    )
    described = _description(snapshot)
    changed_semantics = described.facets.model_copy(update={"styles": ("outline",)})
    described = described.model_copy(update={"facets": changed_semantics})

    plan = plan_collection_merge(
        snapshot,
        source,
        {native_id: _processed(snapshot, tmp_path)},
        {native_id: described},
        {native_id: analysis},
        {native_id: _generation()},
        timestamp="2026-09-11T18:00:00Z",
    )
    merged = plan.snapshot.emojis[existing.id]
    assert merged.facets.styles == existing.facets.styles
    assert merged.facets.rendering.items[0].visible_area_bp == 6000
    assert merged.review.status.value == "unreviewed"

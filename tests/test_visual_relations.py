from __future__ import annotations

import pytest

from mojilex_cli.dataset import load_dataset, validate_snapshot, visual_relations_path
from mojilex_cli.dataset.staging import AtomicDatasetWriter
from mojilex_cli.domain import (
    Emoji,
    Facets,
    IdentityError,
    RelationEvidence,
    RelationMediaPair,
    RelationReview,
    Review,
    VisualRelation,
    media_digest,
    reviewed_content_sha256,
    reviewed_relation_sha256,
    visual_relation_id,
    visual_relation_identity_name,
)
from test_dataset_helpers import make_snapshot, write_fixture

LEFT = "mxe_11111111-1111-5111-8111-111111111111"
RIGHT = "mxe_22222222-2222-5222-8222-222222222222"


def test_visual_relation_normative_uuidv5_vector_and_nul_name() -> None:
    name = visual_relation_identity_name(LEFT, RIGHT, "entity")
    assert name.encode().hex() == (
        "76697375616c2d72656c6174696f6e00"
        "6d78655f31313131313131312d313131312d353131312d383131312d31313131313131313131313100"
        "6d78655f32323232323232322d323232322d353232322d383232322d32323232323232323232323200"
        "656e74697479000000000030"
    )
    assert visual_relation_id(LEFT, RIGHT, "entity") == ("mxr_5c736bb3-43c1-5bd7-bd83-ce9954fb2101")
    forward = visual_relation_id(
        LEFT,
        RIGHT,
        "media-pair",
        subject_role="primary",
        subject_variant_id="light",
        object_role="dark",
        object_variant_id="contrast",
    )
    reverse = visual_relation_id(
        RIGHT,
        LEFT,
        "media-pair",
        subject_role="dark",
        subject_variant_id="contrast",
        object_role="primary",
        object_variant_id="light",
    )
    assert forward == reverse
    with pytest.raises(IdentityError, match="None"):
        visual_relation_id(
            LEFT,
            RIGHT,
            "media-pair",
            subject_role="primary",
            subject_variant_id="",
            object_role="primary",
        )
    with pytest.raises(IdentityError, match=r"U\+0000"):
        visual_relation_id(f"{LEFT}\0poison", RIGHT, "entity")


def _two_emojis(snapshot) -> tuple[Emoji, Emoji]:  # type: ignore[no-untyped-def]
    first = next(iter(snapshot.emojis.values()))
    raw = first.as_dict()
    raw["native_id"] = "5368324170671202287"
    from mojilex_cli.domain import emoji_id

    raw["id"] = emoji_id("telegram", "custom_emoji.id", "global", raw["native_id"])
    raw["extensions"]["telegram"]["custom_emoji_id"] = raw["native_id"]
    raw["extensions"]["telegram"]["file_unique_id"] = "AgADSecondExampleUniqueId"
    second = Emoji.model_validate(raw)
    snapshot.emojis[second.id] = second
    return tuple(sorted((first, second), key=lambda item: item.id))  # type: ignore[return-value]


def _relation(subject: Emoji, object_emoji: Emoji) -> VisualRelation:
    relation = VisualRelation(
        schema_version="1.0.0",
        entity_type="visual_relation",
        id=visual_relation_id(subject.id, object_emoji.id, "entity"),
        identity_epoch=0,
        subject_id=subject.id,
        object_id=object_emoji.id,
        scope="entity",
        relation_type="same-artwork",
        evidence=RelationEvidence(
            dedupe_profile="dedupe-v1",
            subject_media_digest=media_digest(subject.media),
            object_media_digest=media_digest(object_emoji.media),
            media_pairs=[RelationMediaPair(subject_role="primary", object_role="primary")],
            signals=["decoded-exact"],
        ),
        review=RelationReview(status="unreviewed"),
    )
    relation.review = RelationReview(
        status="approved",
        reviewer="reviewer",
        reviewed_at="2026-09-11T12:00:00Z",
        reviewed_relation_sha256=reviewed_relation_sha256(relation),
    )
    return relation


def _approve_literal_text(emoji: Emoji, literal: str | None) -> None:
    facets = emoji.facets.as_dict()
    if literal is None:
        facets["text_content"] = {"status": "none", "dynamics": "stable", "items": []}
    else:
        facets["text_content"] = {
            "status": "recognized",
            "dynamics": "stable",
            "items": [
                {
                    "value": literal,
                    "kind": "number",
                    "script": "Zyyy",
                    "temporal_scope": "persistent",
                    "media_refs": [{"role": "primary"}],
                }
            ],
        }
        facets["content_types"] = sorted({*facets["content_types"], "number", "text"})
    emoji.facets = Facets.model_validate(facets)
    emoji.review = Review(
        status="approved",
        reviewer="reviewer",
        reviewed_at="2026-09-11T12:00:00Z",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
    )


def test_relation_repository_roundtrip_and_integrity(tmp_path) -> None:
    write_fixture(tmp_path)
    snapshot = load_dataset(tmp_path)
    subject, object_emoji = _two_emojis(snapshot)
    relation = _relation(subject, object_emoji)
    snapshot.relations[relation.id] = relation
    writer = AtomicDatasetWriter(tmp_path)
    for path, data in snapshot.to_files().items():
        writer.stage_bytes(path, data)
    writer.commit()

    loaded = load_dataset(tmp_path)
    assert loaded.relations[relation.id] == relation
    assert (tmp_path / visual_relations_path(relation.id)).is_file()
    assert validate_snapshot(loaded, canonical=True).valid


def test_relation_validation_rejects_stale_digest_and_review_hash(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    subject, object_emoji = _two_emojis(snapshot)
    relation = _relation(subject, object_emoji)
    relation.evidence.subject_media_digest = "0" * 64
    snapshot.relations[relation.id] = relation
    codes = {item.code for item in validate_snapshot(snapshot).issues}
    assert {"RELATION_STALE", "RELATION_REVIEW_HASH"}.issubset(codes)


@pytest.mark.parametrize(
    ("left_literal", "right_literal"),
    [("404", "405"), (None, "404")],
    ids=["different-nonempty", "empty-versus-recognized"],
)
def test_same_artwork_rejects_different_human_approved_literal_text(
    tmp_path, left_literal: str | None, right_literal: str | None
) -> None:
    snapshot = make_snapshot(tmp_path)
    subject, object_emoji = _two_emojis(snapshot)
    _approve_literal_text(subject, left_literal)
    _approve_literal_text(object_emoji, right_literal)
    relation = _relation(subject, object_emoji)
    snapshot.relations[relation.id] = relation
    codes = {item.code for item in validate_snapshot(snapshot).issues}
    assert "RELATION_TEXT_CONFLICT" in codes

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mojilex_cli.composition.detector import Composition, Member
from mojilex_cli.composition.publication import mark_verified_fragments
from mojilex_cli.domain import Emoji, Media, Review, emoji_id, media_digest, reviewed_content_sha256
from mojilex_cli.pipeline.runner import _description_from_existing
from mojilex_cli.pipeline.transform import plan_collection_merge
from test_dataset_helpers import make_snapshot
from test_pipeline_transform import _analysis, _description, _generation, _processed, _source


def _fixture(tmp_path):
    snapshot = make_snapshot(tmp_path)
    first = next(iter(snapshot.emojis.values()))
    source = _source(snapshot)
    items = list(source.items)
    for native_id in ("10002", "10003"):
        second = first.model_copy(deep=True)
        second.native_id = native_id
        second.id = emoji_id("telegram", "custom_emoji.id", "global", native_id)
        snapshot.emojis[second.id] = second
        items.append(items[0].model_copy(update={"native_id": native_id, "position": len(items)}))
    source = source.model_copy(update={"items": tuple(items), "item_count": 3})
    group = Composition(
        columns=2,
        rows=1,
        verified=True,
        verification_passes=3,
        verifier_model="test-model",
        members=[
            Member(
                native_id=item.native_id, media_sha256=first.media[0].sha256, tile_sha256="a" * 64
            )
            for item in items[:2]
        ],
    )
    return snapshot, source, group


def test_whole_group_marked_sorted_preserving_twelve_tags_and_normal_emoji(tmp_path):
    snapshot, source, group = _fixture(tmp_path)
    originals = snapshot.clone()
    tags = [f"detail-{chr(97 + i)}" for i in range(12)]
    for emoji in snapshot.emojis.values():
        emoji.semantic_tags = tags[:]
    changed = mark_verified_fragments(snapshot, {source.native_id: [group]}, [source])
    assert len(changed) == 2
    for emoji in snapshot.emojis.values():
        assert emoji.semantic_tags == sorted(tags + (["fragment"] if emoji.id in changed else []))
        assert Emoji.model_validate(emoji.as_dict()) == emoji
        assert len(_description_from_existing(emoji).semantic_tags) == 12
        assert emoji.descriptions == originals.emojis[emoji.id].descriptions
    assert mark_verified_fragments(snapshot, {source.native_id: [group]}, [source]) == set()


@pytest.mark.parametrize(
    "condition", ["unverified", "v1", "hash", "missing", "namespace", "failed"]
)
def test_incomplete_stale_or_unverified_group_never_marks_any_member(tmp_path, condition):
    snapshot, source, group = _fixture(tmp_path)
    sources = [source]
    if condition == "unverified":
        group = group.model_copy(update={"verified": False, "verification_passes": 0})
    elif condition == "v1":
        group = group.model_copy(update={"detector": "composition-v1", "verification_passes": 1})
    elif condition == "hash":
        group = group.model_copy(
            update={
                "members": [
                    group.members[0],
                    group.members[-1].model_copy(update={"media_sha256": "f" * 64}),
                ]
            }
        )
    elif condition == "missing":
        snapshot.emojis.pop(emoji_id("telegram", "custom_emoji.id", "global", "10002"))
    elif condition == "namespace":
        snapshot.emojis[
            emoji_id("telegram", "custom_emoji.id", "global", "10002")
        ].scope_id = "other"
    else:
        sources = []
    before = snapshot.to_files()
    assert mark_verified_fragments(snapshot, {source.native_id: [group]}, sources) == set()
    assert snapshot.to_files() == before


@pytest.mark.parametrize("status", ["approved", "rejected", "changes_requested"])
def test_marker_invalidates_approval_but_does_not_clear_negative_review(tmp_path, status):
    snapshot, source, group = _fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.review = Review(
        status=status,
        reviewed_at="2026-09-13T00:00:00Z",
        reviewer="reviewer",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
        review_hash_profile_id="semantic-review-content-v3",
    )
    before = emoji.as_dict()
    mark_verified_fragments(snapshot, {source.native_id: [group]}, [source])
    if status == "approved":
        assert emoji.review == Review(status="unreviewed")
        assert "fragment" in emoji.semantic_tags
    else:
        assert emoji.as_dict() == before


@pytest.mark.parametrize("tags", [["fragment"], [f"tag-{chr(97 + i)}" for i in range(13)]])
def test_public_tags_require_concrete_meaning_and_reserve_thirteenth_slot(tmp_path, tags):
    emoji = next(iter(make_snapshot(tmp_path).emojis.values())).as_dict()
    emoji["semantic_tags"] = tags
    with pytest.raises(ValidationError):
        Emoji.model_validate(emoji)


@pytest.mark.parametrize("changed_media", [False, True])
def test_cached_description_preserves_marker_only_for_unchanged_media(tmp_path, changed_media):
    snapshot = make_snapshot(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    description = _description(snapshot)
    emoji.semantic_tags = sorted([*emoji.semantic_tags, "fragment"])
    processed = _processed(snapshot, tmp_path)
    if changed_media:
        processed = processed.model_copy(
            update={"metadata": processed.metadata.model_copy(update={"sha256": "f" * 64})}
        )
    source = _source(snapshot)
    native_id = source.items[0].native_id
    analysis = _analysis(snapshot)
    analysis.fingerprints.input_media_digest = media_digest(
        [Media.model_validate(processed.dataset_metadata())]
    )
    plan = plan_collection_merge(
        snapshot,
        source,
        {native_id: processed},
        {native_id: description},
        {native_id: analysis},
        {native_id: _generation()},
        timestamp="2026-09-11T18:00:00Z",
    )
    result = next(iter(plan.snapshot.emojis.values()))
    assert ("fragment" in result.semantic_tags) is (not changed_media)
    if not changed_media:
        assert plan.updated == 0
        assert plan.snapshot.to_files() == snapshot.to_files()


@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("stale_media", [False, True])
def test_fragment_lookup_uses_latest_full_identity_without_falling_back_to_old_media(
    tmp_path, newest_first, stale_media
):
    snapshot, source, group = _fixture(tmp_path)
    original = next(iter(snapshot.emojis.values()))
    latest = original.model_copy(deep=True)
    latest.identity_epoch = 2
    latest.id = emoji_id(
        latest.platform, latest.native_namespace, latest.scope_id, latest.native_id, 2
    )
    if stale_media:
        latest.media[0].sha256 = "f" * 64
    records = list(snapshot.emojis.values())
    records.insert(0 if newest_first else len(records), latest)
    decoys = []
    for field in ("platform", "native_namespace", "scope_id", "native_id"):
        decoy = original.model_copy(deep=True)
        setattr(decoy, field, "different")
        decoy.identity_epoch = 9
        decoy.id = emoji_id(
            decoy.platform, decoy.native_namespace, decoy.scope_id, decoy.native_id, 9
        )
        decoys.append(decoy)
    snapshot.emojis = {record.id: record for record in [*records, *decoys]}

    changed = mark_verified_fragments(snapshot, {source.native_id: [group]}, [source])

    expected = {
        latest.id,
        emoji_id("telegram", "custom_emoji.id", "global", source.items[1].native_id),
    }
    assert changed == (set() if stale_media else expected)
    for record in snapshot.emojis.values():
        assert ("fragment" in record.semantic_tags) is (record.id in changed)

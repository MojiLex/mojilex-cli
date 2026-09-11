from __future__ import annotations

import hashlib
import uuid

import rfc8785

from mojilex_cli.dedupe import scan_snapshot
from mojilex_cli.domain import (
    Collection,
    Emoji,
    Membership,
    RelationReview,
    VisualRelation,
    collection_id,
    emoji_id,
    media_digest,
    membership_id,
    reviewed_relation_sha256,
)
from test_dataset_helpers import make_snapshot
from test_visual_relations import _relation, _two_emojis


def _clone(snapshot, native_id: str) -> Emoji:  # type: ignore[no-untyped-def]
    source = next(iter(snapshot.emojis.values()))
    raw = source.as_dict()
    raw["id"] = emoji_id("telegram", "custom_emoji.id", "global", native_id)
    raw["native_id"] = native_id
    raw["extensions"]["telegram"]["custom_emoji_id"] = native_id
    raw["extensions"]["telegram"]["file_unique_id"] = f"AgADSynthetic{native_id}"
    clone = Emoji.model_validate(raw)
    snapshot.emojis[clone.id] = clone
    return clone


def _collection_with_members(
    snapshot,  # type: ignore[no-untyped-def]
    native_id: str,
    emojis: list[Emoji],
    collection_template: Collection,
    membership_template: Membership,
) -> Collection:
    raw = collection_template.as_dict()
    raw["id"] = collection_id("telegram", "sticker_set.name", "global", native_id)
    raw["native_id"] = native_id
    raw["title"] = native_id
    raw["item_count"] = len(emojis)
    raw["extensions"]["telegram"]["short_name"] = native_id
    collection = Collection.model_validate(raw)
    snapshot.collections[collection.id] = collection
    for position, emoji in enumerate(emojis):
        template = membership_template.as_dict()
        template.update(
            {
                "id": membership_id(collection.id, emoji.id),
                "collection_id": collection.id,
                "emoji_id": emoji.id,
                "position": position,
            }
        )
        membership = Membership.model_validate(template)
        snapshot.memberships[membership.id] = membership
    return collection


def test_exact_groups_use_stable_content_ids_and_entity_members(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    _clone(snapshot, "5368324170671202287")

    report = scan_snapshot(snapshot, mode="exact")

    binary_media = next(
        group
        for group in report.exact_groups
        if group["group_type"] == "binary-exact" and group["scope"] == "media"
    )
    source = next(iter(snapshot.emojis.values())).media[0]
    content_digest = hashlib.sha256(
        rfc8785.dumps({"byte_size": source.byte_size, "sha256": source.sha256})
    ).hexdigest()
    namespace = uuid.UUID(str(snapshot.manifest["visual_relation_namespace"]))
    expected = "mxdg_" + str(
        uuid.uuid5(
            namespace,
            "\0".join(("duplicate-group", "binary-exact", "media", "", content_digest)),
        )
    )
    assert binary_media["id"] == expected
    assert binary_media["source_sha256"] == source.sha256
    assert binary_media["byte_size"] == source.byte_size
    binary_entity = next(
        group
        for group in report.exact_groups
        if group["group_type"] == "binary-exact" and group["scope"] == "entity"
    )
    assert binary_entity["members"] == sorted(snapshot.emojis)


def test_candidate_limit_reports_overflow_and_prelimit_count(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    for suffix in range(7, 10):
        _clone(snapshot, f"53683241706712022{suffix}")

    report = scan_snapshot(snapshot, mode="near", max_candidates=1)

    assert report.comparisons == 6
    for emoji_id_value in snapshot.emojis:
        assert report.candidate_count_before_limit[emoji_id_value] == 3
        assert report.candidate_overflow[emoji_id_value]
        assert len(report.candidates[emoji_id_value]) == 1


def test_low_information_unique_buckets_do_not_expand_pairwise(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    original = next(iter(snapshot.emojis.values()))
    original_raw = original.as_dict()
    snapshot.emojis.clear()
    for index in range(300):
        raw = original_raw.copy()
        raw = Emoji.model_validate(raw).as_dict()
        native_id = str(6_000_000_000_000_000_000 + index)
        raw["id"] = emoji_id("telegram", "custom_emoji.id", "global", native_id)
        raw["native_id"] = native_id
        digest = hashlib.sha256(f"media-{index}".encode()).hexdigest()
        raw["media"][0]["sha256"] = digest
        raw["fingerprints"]["input_media_digest"] = media_digest(
            [Emoji.model_validate(original_raw).media[0].model_copy(update={"sha256": digest})]
        )
        for field in (
            "decoded_payload_sha256",
            "canonical_render_sha256",
            "shape_sha256",
        ):
            raw["fingerprints"]["items"][0][field] = hashlib.sha256(
                f"{field}-{index}".encode()
            ).hexdigest()
        raw["fingerprints"]["items"][0]["perceptual"]["low_information"] = True
        emoji = Emoji.model_validate(raw)
        snapshot.emojis[emoji.id] = emoji

    report = scan_snapshot(snapshot, mode="near")

    assert report.comparisons == 0
    assert report.candidates == {}


def test_approved_not_duplicate_suppresses_future_candidate(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    subject, object_emoji = _two_emojis(snapshot)
    raw = _relation(subject, object_emoji).as_dict()
    raw["relation_type"] = "not-duplicate"
    raw["review"] = {"status": "unreviewed"}
    relation = VisualRelation.model_validate(raw)
    relation.review = RelationReview(
        status="approved",
        reviewer="reviewer",
        reviewed_at="2026-09-11T12:00:00Z",
        reviewed_relation_sha256=reviewed_relation_sha256(relation),
    )
    snapshot.relations[relation.id] = relation

    report = scan_snapshot(snapshot, mode="near")

    assert report.candidates == {}
    assert report.comparisons == 0

    changed_raw = object_emoji.as_dict()
    changed_raw["media"][0]["sha256"] = "e" * 64
    changed_raw["fingerprints"]["input_media_digest"] = media_digest(
        [object_emoji.media[0].model_copy(update={"sha256": "e" * 64})]
    )
    snapshot.emojis[object_emoji.id] = Emoji.model_validate(changed_raw)

    stale_report = scan_snapshot(snapshot, mode="near")

    assert stale_report.candidates


def test_primary_match_alone_does_not_create_entity_exact_group(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    second = _clone(snapshot, "5368324170671202287")
    raw = second.as_dict()
    alternate_media = dict(raw["media"][0])
    alternate_media.update({"role": "alternate", "variant_id": "alt", "sha256": "a" * 64})
    raw["media"] = [alternate_media, raw["media"][0]]
    alternate_rendering = dict(raw["facets"]["rendering"]["items"][0])
    alternate_rendering.update({"role": "alternate", "variant_id": "alt"})
    raw["facets"]["rendering"]["items"] = [
        alternate_rendering,
        raw["facets"]["rendering"]["items"][0],
    ]
    alternate_fingerprint = dict(raw["fingerprints"]["items"][0])
    alternate_fingerprint.update(
        {
            "role": "alternate",
            "variant_id": "alt",
            "decoded_payload_sha256": "b" * 64,
            "canonical_render_sha256": "c" * 64,
            "shape_sha256": "d" * 64,
        }
    )
    raw["fingerprints"]["items"] = [
        alternate_fingerprint,
        raw["fingerprints"]["items"][0],
    ]
    raw["fingerprints"]["input_media_digest"] = media_digest(
        [
            second.media[0].model_copy(
                update={"role": "alternate", "variant_id": "alt", "sha256": "a" * 64}
            ),
            second.media[0],
        ]
    )
    snapshot.emojis[second.id] = Emoji.model_validate(raw)

    report = scan_snapshot(snapshot, mode="exact")

    assert any(group["scope"] == "media" for group in report.exact_groups)
    assert not any(group["scope"] == "entity" for group in report.exact_groups)


def test_collection_clone_candidates_use_one_to_one_matches_and_minimum_eight(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    collection_template = next(iter(snapshot.collections.values()))
    membership_template = next(iter(snapshot.memberships.values()))
    emojis = [next(iter(snapshot.emojis.values()))]
    emojis.extend(_clone(snapshot, str(7_000_000_000_000_000_000 + index)) for index in range(9))
    snapshot.collections.clear()
    snapshot.memberships.clear()
    left = _collection_with_members(
        snapshot, "CloneLeft", emojis, collection_template, membership_template
    )
    right = _collection_with_members(
        snapshot, "CloneRight", emojis, collection_template, membership_template
    )

    report = scan_snapshot(snapshot, mode="exact")

    assert len(report.collection_candidates) == 1
    candidate = report.collection_candidates[0]
    assert {candidate.collection_id, candidate.against_collection_id} == {left.id, right.id}
    assert candidate.candidate_type == "clone-candidate"
    assert candidate.matched_count == 10
    assert candidate.collection_coverage == "1"

    smaller = make_snapshot(tmp_path / "small")
    smaller_collection_template = next(iter(smaller.collections.values()))
    smaller_membership_template = next(iter(smaller.memberships.values()))
    small_emojis = [next(iter(smaller.emojis.values()))]
    small_emojis.extend(
        _clone(smaller, str(8_000_000_000_000_000_000 + index)) for index in range(6)
    )
    smaller.collections.clear()
    smaller.memberships.clear()
    _collection_with_members(
        smaller,
        "SmallLeft",
        small_emojis,
        smaller_collection_template,
        smaller_membership_template,
    )
    _collection_with_members(
        smaller,
        "SmallRight",
        small_emojis,
        smaller_collection_template,
        smaller_membership_template,
    )
    assert scan_snapshot(smaller, mode="exact").collection_candidates == ()


def test_collection_subset_candidate_excludes_variant_relations(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    collection_template = next(iter(snapshot.collections.values()))
    membership_template = next(iter(snapshot.memberships.values()))
    shared = [next(iter(snapshot.emojis.values()))]
    shared.extend(_clone(snapshot, str(9_000_000_000_000_000_000 + index)) for index in range(7))
    distinct = []
    for index in range(3):
        emoji = _clone(snapshot, str(9_100_000_000_000_000_000 + index))
        raw = emoji.as_dict()
        digest = hashlib.sha256(f"distinct-{index}".encode()).hexdigest()
        raw["media"][0]["sha256"] = digest
        raw["fingerprints"]["items"][0]["decoded_payload_sha256"] = digest
        raw["fingerprints"]["items"][0]["canonical_render_sha256"] = digest
        raw["fingerprints"]["items"][0]["shape_sha256"] = digest
        raw["fingerprints"]["input_media_digest"] = media_digest(
            [emoji.media[0].model_copy(update={"sha256": digest})]
        )
        updated = Emoji.model_validate(raw)
        snapshot.emojis[updated.id] = updated
        distinct.append(updated)
    snapshot.collections.clear()
    snapshot.memberships.clear()
    _collection_with_members(
        snapshot, "SubsetSmall", shared, collection_template, membership_template
    )
    _collection_with_members(
        snapshot,
        "SubsetLarge",
        [*shared, *distinct],
        collection_template,
        membership_template,
    )

    report = scan_snapshot(snapshot, mode="exact")

    assert len(report.collection_candidates) == 1
    candidate = report.collection_candidates[0]
    assert candidate.candidate_type == "subset-candidate"
    assert candidate.matched_count == 8
    assert {candidate.collection_size, candidate.against_collection_size} == {8, 11}

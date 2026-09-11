from __future__ import annotations

import base64
import hashlib
import sqlite3

import pytest
import rfc8785

from mojilex_cli.analysis import load_analysis_profile
from mojilex_cli.commands.dedupe import (
    _PERSISTED_RELATION_SIGNALS,
    _apply_review_relation,
    _approved_relation,
    _best_media_bijection,
    _guard_literal_text_conflict,
    _sole_downloadable_media,
    _verify_current_media,
)
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset import apply_snapshot, load_dataset, validate_snapshot
from mojilex_cli.dedupe import DedupeIndex, explain_pair, scan_snapshot
from mojilex_cli.dedupe.engine import DedupeCandidate, _aligned_distances, _candidate_sort_key
from mojilex_cli.domain import (
    Emoji,
    Media,
    RelationReview,
    VisualRelation,
    media_digest,
    reviewed_relation_sha256,
)
from mojilex_cli.media import MediaMetadata, ProcessedMedia
from test_dataset_helpers import make_snapshot, write_fixture
from test_dedupe_engine import _clone
from test_visual_relations import _approve_literal_text, _relation, _two_emojis


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _packed(value: int, *, count: int = 1) -> str:
    raw = value.to_bytes(8, "big") * count
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _set_static_fingerprint(
    snapshot,  # type: ignore[no-untyped-def]
    emoji: Emoji,
    *,
    seed: str,
    phash: int,
    shape: str | None = None,
    alpha_mode: str = "opaque",
) -> Emoji:
    raw = emoji.as_dict()
    raw["media"][0]["sha256"] = _digest(f"media:{seed}")
    raw["fingerprints"]["input_media_digest"] = media_digest(
        [Media.model_validate(raw["media"][0])]
    )
    item = raw["fingerprints"]["items"][0]
    item["decoded_payload_sha256"] = _digest(f"decoded:{seed}")
    item["canonical_render_sha256"] = _digest(f"canonical:{seed}")
    item["shape_sha256"] = shape or _digest(f"shape:{seed}")
    for field in (
        "layout_phash64",
        "content_phash64",
        "alpha_phash64",
        "edge_phash64",
    ):
        item["perceptual"][field] = _packed(phash)
    item["perceptual"]["low_information"] = False
    raw["facets"]["rendering"]["items"][0]["alpha_mode"] = alpha_mode
    result = Emoji.model_validate(raw)
    snapshot.emojis[result.id] = result
    return result


def _near_pair(tmp_path):  # type: ignore[no-untyped-def]
    snapshot = make_snapshot(tmp_path)
    first = next(iter(snapshot.emojis.values()))
    second = _clone(snapshot, "5368324170671202287")
    first = _set_static_fingerprint(snapshot, first, seed="left", phash=0)
    second = _set_static_fingerprint(snapshot, second, seed="right", phash=0)
    return snapshot, first, second


def _add_alternate(emoji: Emoji, *, seed: str) -> Emoji:
    raw = emoji.as_dict()
    media = dict(raw["media"][0])
    media.update(
        {
            "role": "alternate",
            "variant_id": "alt",
            "sha256": _digest(f"alternate-media:{seed}"),
        }
    )
    rendering = dict(raw["facets"]["rendering"]["items"][0])
    rendering.update({"role": "alternate", "variant_id": "alt"})
    fingerprint = dict(raw["fingerprints"]["items"][0])
    fingerprint.update(
        {
            "role": "alternate",
            "variant_id": "alt",
            "decoded_payload_sha256": _digest(f"alternate-decoded:{seed}"),
            "canonical_render_sha256": _digest(f"alternate-canonical:{seed}"),
            "shape_sha256": _digest("shared-alternate-shape"),
        }
    )
    raw["media"] = [media, raw["media"][0]]
    raw["facets"]["rendering"]["items"] = [
        rendering,
        raw["facets"]["rendering"]["items"][0],
    ]
    raw["fingerprints"]["items"] = [fingerprint, raw["fingerprints"]["items"][0]]
    raw["fingerprints"]["input_media_digest"] = media_digest(
        [Media.model_validate(value) for value in raw["media"]]
    )
    return Emoji.model_validate(raw)


def test_multiprobe_finds_hamming_four_split_across_all_bands(tmp_path) -> None:
    snapshot, first, second = _near_pair(tmp_path)
    split_bits = (1 << 0) | (1 << 16) | (1 << 32) | (1 << 48)
    _set_static_fingerprint(snapshot, second, seed="right", phash=split_bits)

    report = scan_snapshot(snapshot, mode="near")

    assert report.comparisons == 1
    assert report.candidates[first.id][0].against_emoji_id == second.id
    assert "phash-match" in report.candidates[first.id][0].signals


def test_decoded_entity_digest_excludes_profile_but_group_id_includes_it(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    _clone(snapshot, "5368324170671202287")
    group = next(
        value
        for value in scan_snapshot(snapshot, mode="exact").exact_groups
        if value["group_type"] == "decoded-exact" and value["scope"] == "entity"
    )
    expected_payload = [{"role": "primary", "decoded_payload_sha256": "3" * 64}]
    assert group["content_digest"] == hashlib.sha256(rfc8785.dumps(expected_payload)).hexdigest()
    assert group["profile"] == "dedupe-v1"


def test_oversized_noninformative_hash_buckets_have_bounded_comparisons(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    original = next(iter(snapshot.emojis.values()))
    snapshot.emojis.clear()
    snapshot.emojis[original.id] = original
    _set_static_fingerprint(snapshot, original, seed="overflow-0", phash=0)
    posting_cap = int(
        load_analysis_profile("dedupe-v1").data["oversized_bucket_policy"]["posting_bucket_cap"]
    )
    for index in range(1, posting_cap + 8):
        emoji = _clone(snapshot, str(7_000_000_000_000_000_000 + index))
        _set_static_fingerprint(
            snapshot,
            emoji,
            seed=f"overflow-{index}",
            phash=0,
        )

    report = scan_snapshot(snapshot, mode="near")

    assert report.suppressed_bucket_count >= 16
    assert report.comparisons == 0
    assert report.candidates == {}


def test_shape_only_candidate_requires_profile_eligible_rendering(tmp_path) -> None:
    snapshot, first, second = _near_pair(tmp_path)
    shared_shape = _digest("same-shape")
    _set_static_fingerprint(
        snapshot,
        first,
        seed="left",
        phash=0,
        shape=shared_shape,
        alpha_mode="opaque",
    )
    _set_static_fingerprint(
        snapshot,
        second,
        seed="right",
        phash=(1 << 64) - 1,
        shape=shared_shape,
        alpha_mode="opaque",
    )
    assert scan_snapshot(snapshot, mode="near").candidates == {}

    _set_static_fingerprint(
        snapshot,
        second,
        seed="right",
        phash=(1 << 64) - 1,
        shape=shared_shape,
        alpha_mode="binary",
    )
    candidate = scan_snapshot(snapshot, mode="near").candidates[first.id][0]
    assert {"shape-match", "possible-recolor"}.issubset(candidate.signals)


def test_candidate_ranking_uses_duration_then_literal_text() -> None:
    base = {
        "emoji_id": "left",
        "role": "primary",
        "variant_id": None,
        "against_role": "primary",
        "against_variant_id": None,
        "signals": ("phash-match",),
        "primary_distance": 3,
        "alpha_distance": 3,
        "edge_distance": 3,
        "p90_distance": 3,
        "high_priority": True,
    }
    incompatible = DedupeCandidate(
        **base,
        against_emoji_id="duration-bad",
        speed_ratio="2",
        duration_compatible=False,
        text_match=True,
        text_mismatch=False,
    )
    compatible_text_mismatch = DedupeCandidate(
        **base,
        against_emoji_id="duration-good",
        speed_ratio="1.1",
        duration_compatible=True,
        text_match=False,
        text_mismatch=True,
    )
    compatible_text_match = DedupeCandidate(
        **base,
        against_emoji_id="text-good",
        speed_ratio="1.1",
        duration_compatible=True,
        text_match=True,
        text_mismatch=False,
    )
    assert (
        sorted((incompatible, compatible_text_mismatch), key=_candidate_sort_key)[0]
        is compatible_text_mismatch
    )
    assert (
        sorted((compatible_text_mismatch, compatible_text_match), key=_candidate_sort_key)[0]
        is compatible_text_match
    )


def test_alignment_uses_configured_dtw_band_and_reverse_policy() -> None:
    left = (1, 2, 4)
    reversed_right = (4, 2, 1)
    direct = _aligned_distances(
        left,
        reversed_right,
        animated=True,
        cyclic=False,
        dtw_band=0,
        reverse=False,
    )
    reversed_match = _aligned_distances(
        left,
        reversed_right,
        animated=True,
        cyclic=False,
        dtw_band=0,
        reverse=True,
    )
    assert sum(direct) > 0
    assert sum(reversed_match) == 0


def test_index_preserves_complete_neighbor_rows_on_incremental_addition(tmp_path) -> None:
    snapshot, first, second = _near_pair(tmp_path)
    index_path = tmp_path / "cache" / "dedupe.sqlite3"
    with DedupeIndex(index_path, repository_root=tmp_path / "repository") as index:
        index.update(snapshot, rebuild=True, max_candidates=5)
        assert index.explain(first.id, second.id) is not None
        third = _clone(snapshot, "5368324170671202288")
        third = _set_static_fingerprint(snapshot, third, seed="third", phash=0)
        index.update(snapshot, max_candidates=5)
        assert index.explain(first.id, second.id) is not None
        assert index.explain(first.id, third.id) is not None
        assert index.explain(second.id, third.id) is not None


def test_index_invalidates_fingerprint_relation_and_candidate_limit_state(tmp_path) -> None:
    snapshot, first, second = _near_pair(tmp_path)
    third = _clone(snapshot, "5368324170671202288")
    third = _set_static_fingerprint(snapshot, third, seed="third", phash=0)
    fourth = _clone(snapshot, "5368324170671202289")
    _set_static_fingerprint(snapshot, fourth, seed="fourth", phash=0)
    index_path = tmp_path / "cache" / "dedupe.sqlite3"
    with DedupeIndex(index_path, repository_root=tmp_path / "repository") as index:
        index.update(snapshot, rebuild=True, max_candidates=1)
        assert (
            index.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE source_emoji_id = ?", (first.id,)
            ).fetchone()[0]
            == 1
        )
        index.update(snapshot, max_candidates=3)
        assert (
            index.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE source_emoji_id = ?", (first.id,)
            ).fetchone()[0]
            == 3
        )

        _set_static_fingerprint(
            snapshot,
            second,
            seed="right",
            phash=(1 << 64) - 1,
        )
        index.update(snapshot, max_candidates=3)
        assert index.explain(first.id, second.id) is None

        subject, object_emoji = tuple(sorted((first, third), key=lambda item: item.id))
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
        index.update(snapshot, max_candidates=3)
        assert index.explain(first.id, third.id) is None


def test_index_handles_more_ids_than_sqlite_parameter_limit(tmp_path) -> None:
    with DedupeIndex(tmp_path / "dedupe.sqlite3", repository_root=tmp_path / "repo") as index:
        identifiers = {f"mxe_synthetic_{value:05d}" for value in range(20_000)}
        assert index._candidate_neighbors(identifiers) == set()
        index._delete_candidate_sources(identifiers)


def test_index_recovers_old_tables_when_static_header_is_empty(tmp_path) -> None:
    path = tmp_path / "dedupe.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE emoji_state (
                emoji_id TEXT PRIMARY KEY,
                input_media_digest TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
    with DedupeIndex(path, repository_root=tmp_path / "repo") as index:
        columns = {
            str(row[1]) for row in index.connection.execute("PRAGMA table_info(emoji_state)")
        }
    assert columns == {"emoji_id", "state_digest"}


def test_review_relation_signals_and_media_mapping_roundtrip(tmp_path) -> None:
    write_fixture(tmp_path)
    before = load_dataset(tmp_path)
    after = before.clone()
    left, right = _two_emojis(after)
    explanation = explain_pair(after, left.id, right.id)

    relation = _approved_relation(
        after,
        left.id,
        right.id,
        relation_type="same-artwork",
        reviewer="reviewer",
        explanation=explanation,
    )
    after.relations[relation.id] = relation
    apply_snapshot(before, after)
    loaded = load_dataset(tmp_path)

    assert len(relation.evidence.media_pairs) == 1
    assert set(relation.evidence.signals) <= _PERSISTED_RELATION_SIGNALS
    assert "binary-exact" not in relation.evidence.signals
    assert "human-visual-review" not in relation.evidence.signals
    assert loaded.relations[relation.id] == relation
    assert validate_snapshot(loaded, canonical=True).valid


def test_review_refuses_same_artwork_when_only_one_approved_endpoint_has_text(
    tmp_path,
) -> None:
    snapshot = make_snapshot(tmp_path)
    left, right = _two_emojis(snapshot)
    _approve_literal_text(left, None)
    _approve_literal_text(right, "404")

    with pytest.raises(CommandError, match="Approved literal text differs"):
        _guard_literal_text_conflict(snapshot, left.id, right.id, "same-artwork")


def test_entity_relation_maps_every_media_or_refuses_adapter_shortcut(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    left, right = _two_emojis(snapshot)
    snapshot.emojis[left.id] = _add_alternate(left, seed="left")
    snapshot.emojis[right.id] = _add_alternate(right, seed="right")
    explanation = explain_pair(snapshot, left.id, right.id)
    relation = _approved_relation(
        snapshot,
        left.id,
        right.id,
        relation_type="same-artwork",
        reviewer="reviewer",
        explanation=explanation,
    )

    assert len(relation.evidence.media_pairs) == 2
    with pytest.raises(CommandError, match="cannot re-download"):
        _sole_downloadable_media(snapshot, left.id)


def test_review_mapping_and_verification_fail_closed_when_incomplete_or_ambiguous(
    tmp_path,
) -> None:
    comparisons = {
        (left, right): {"candidate": None}
        for left in (("primary", ""), ("light", ""))
        for right in (("dark", ""), ("alternate", ""))
    }
    with pytest.raises(CommandError, match="ambiguous"):
        _best_media_bijection(
            (("primary", ""), ("light", "")),
            (("dark", ""), ("alternate", "")),
            comparisons,
        )

    snapshot = make_snapshot(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    expanded = _add_alternate(emoji, seed="left")
    snapshot.emojis[expanded.id] = expanded
    primary = next(item for item in expanded.media if item.role.value == "primary")
    processed = ProcessedMedia(
        metadata=MediaMetadata.model_validate(
            primary.model_dump(exclude={"variant_id"}, exclude_none=True)
        ),
        frame_paths=(tmp_path / "frame.png",),
    )
    with pytest.raises(CommandError, match="complete current media set"):
        _verify_current_media(snapshot, {expanded.id: {("primary", ""): processed}})


def test_review_apply_checks_canonical_disk_then_rolls_back(tmp_path) -> None:
    write_fixture(tmp_path)
    before = load_dataset(tmp_path)
    after = before.clone()
    subject, object_emoji = _two_emojis(after)
    relation = _relation(subject, object_emoji)
    after.relations[relation.id] = relation

    with pytest.raises(ValueError, match="SCHEMA_MISSING"):
        _apply_review_relation(before, after)

    assert load_dataset(tmp_path).relations == {}

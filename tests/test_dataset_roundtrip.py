from mojilex_cli.dataset import (
    IdentityContinuity,
    assess_identity_continuity,
    collection_path,
    emoji_bucket_path,
    load_dataset,
    merge_emoji,
)
from mojilex_cli.dataset.staging import apply_snapshot
from mojilex_cli.dataset.validation import validate_snapshot
from mojilex_cli.domain import Emoji, Review, reviewed_content_sha256
from test_dataset_helpers import write_fixture


def test_loader_roundtrip_and_canonical_paths(tmp_path) -> None:
    original = write_fixture(tmp_path)
    loaded = load_dataset(tmp_path)
    collection = next(iter(loaded.collections.values()))
    emoji = next(iter(loaded.emojis.values()))
    assert loaded.to_files() == original.to_files()
    assert (tmp_path / collection_path(collection.platform, collection.id)).is_file()
    assert (tmp_path / emoji_bucket_path(emoji.platform, emoji.id)).is_file()


def test_collection_catalog_follows_title_changes_and_preserves_legacy_absence(tmp_path) -> None:
    write_fixture(tmp_path)
    catalog = tmp_path / "data" / "telegram" / "collections" / "README.md"
    assert catalog.is_file()
    old_bytes = catalog.read_bytes()
    before = load_dataset(tmp_path)
    changed = before.clone()
    next(iter(changed.collections.values())).title = "A renamed pack"
    apply_snapshot(before, changed, validator=validate_snapshot)
    assert catalog.read_bytes() != old_bytes
    assert b"A renamed pack" in catalog.read_bytes()

    catalog.unlink()
    legacy = load_dataset(tmp_path)
    assert validate_snapshot(legacy, canonical=True).valid
    assert catalog not in [tmp_path / path for path in legacy.to_files(preserve_legacy_paths=True)]
    apply_snapshot(legacy, legacy.clone(), validator=validate_snapshot)
    assert catalog.is_file()


def test_merge_preserves_approved_manual_content_when_media_is_unchanged(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    existing = next(iter(snapshot.emojis.values())).model_copy(deep=True)
    existing_raw = existing.as_dict()
    existing_raw["concept_ids"] = ["animal.cat"]
    existing_raw["concept_mapping_status"] = "complete"
    existing = Emoji.model_validate(existing_raw)
    existing.review = Review(
        status="approved",
        reviewed_at="2026-09-10T19:00:00Z",
        reviewer="reviewer",
        reviewed_content_sha256=reviewed_content_sha256(existing),
        review_hash_profile_id="semantic-review-content-v3",
    )
    incoming_raw = existing.as_dict()
    incoming_raw["concept_ids"] = []
    incoming_raw["concept_mapping_status"] = "pending"
    incoming_raw["review"] = {"status": "unreviewed"}
    incoming = Emoji.model_validate(incoming_raw)
    incoming.descriptions["en"].text = "An incorrect replacement."
    merged = merge_emoji(existing, incoming)
    assert merged.descriptions["en"].text == existing.descriptions["en"].text
    assert merged.concept_ids == ["animal.cat"]
    assert merged.concept_mapping_status.value == "complete"
    assert merged.review.status.value == "approved"


def test_identity_continuity_requires_explicit_decision_on_zero_overlap() -> None:
    assert assess_identity_continuity(set(), {"1"}) is IdentityContinuity.NEW
    assert assess_identity_continuity({"1"}, {"1", "2"}) is IdentityContinuity.SAME
    assert assess_identity_continuity({"1"}, {"2"}) is IdentityContinuity.CONFLICT

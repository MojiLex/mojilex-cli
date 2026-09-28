import hashlib

from mojilex_cli.dataset import (
    IdentityContinuity,
    assess_identity_continuity,
    collection_path,
    emoji_bucket_path,
    load_dataset,
    merge_emoji,
)
from mojilex_cli.dataset.collection_catalog import render_legacy_catalog
from mojilex_cli.dataset.layout import previous_emoji_bucket_path
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
    pack_page = tmp_path / "data" / "telegram" / "collections" / collection.id / "README.md"
    assert pack_page.is_file()
    digest = hashlib.sha256(emoji.id.encode("utf-8")).hexdigest()
    assert f"../../emojis/{digest}.jsonl" in pack_page.read_text(encoding="utf-8")


def test_collection_catalog_follows_title_changes_and_requires_modern_pages(tmp_path) -> None:
    write_fixture(tmp_path)
    catalog = tmp_path / "data" / "telegram" / "collections" / "README.md"
    assert catalog.is_file()
    old_bytes = catalog.read_bytes()
    collection = next(iter(load_dataset(tmp_path).collections.values()))
    pack_page = catalog.parent / collection.id / "README.md"
    old_pack_page = pack_page.read_bytes()
    before = load_dataset(tmp_path)
    changed = before.clone()
    next(iter(changed.collections.values())).title = "A renamed pack"
    apply_snapshot(before, changed, validator=validate_snapshot)
    assert catalog.read_bytes() != old_bytes
    assert b"A renamed pack" in catalog.read_bytes()
    assert b"A renamed pack" in pack_page.read_bytes()
    assert pack_page.read_bytes() != old_pack_page
    new_pack_page = pack_page.read_bytes()

    pack_page.unlink()
    missing_page = validate_snapshot(load_dataset(tmp_path), canonical=True)
    assert any(
        issue.code == "PATH" and issue.path.endswith("README.md") for issue in missing_page.issues
    )

    pack_page.write_bytes(new_pack_page)
    catalog.unlink()
    missing_catalog = validate_snapshot(load_dataset(tmp_path), canonical=True)
    assert any(
        issue.code == "PATH" and issue.path.endswith("README.md")
        for issue in missing_catalog.issues
    )


def test_legacy_catalog_without_pack_pages_validates_and_migrates(tmp_path) -> None:
    fixture = write_fixture(tmp_path)
    collection = next(iter(fixture.collections.values()))
    emoji = next(iter(fixture.emojis.values()))
    modern = tmp_path / emoji_bucket_path(emoji.platform, emoji.id)
    previous = tmp_path / previous_emoji_bucket_path(emoji.platform, emoji.id)
    previous.parent.mkdir(parents=True, exist_ok=True)
    modern.rename(previous)
    catalog = tmp_path / "data" / "telegram" / "collections" / "README.md"
    catalog.write_bytes(render_legacy_catalog(fixture.collections.values()))
    pack_page = catalog.parent / collection.id / "README.md"
    pack_page.unlink()

    before = load_dataset(tmp_path)
    assert validate_snapshot(before, canonical=True).valid
    assert before.to_files(preserve_legacy_paths=True) == before.source_bytes
    apply_snapshot(before, before.clone(), validator=validate_snapshot)
    assert modern.is_file()
    assert not previous.exists()
    assert pack_page.is_file()
    assert validate_snapshot(load_dataset(tmp_path), canonical=True).valid


def test_pack_page_escapes_descriptions_and_stale_page_is_rejected(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    collection = next(iter(snapshot.collections.values()))
    emoji = next(iter(snapshot.emojis.values()))
    emoji.descriptions["en"].text = "A *cat* | [link](example)."
    before = load_dataset(tmp_path)
    apply_snapshot(before, snapshot, validator=validate_snapshot)
    page = tmp_path / "data" / "telegram" / "collections" / collection.id / "README.md"
    assert "A \\*cat\\* \\| \\[link\\](example)." in page.read_text(encoding="utf-8")
    page.write_bytes(b"stale\n")
    report = validate_snapshot(load_dataset(tmp_path), canonical=True)
    assert any(
        issue.code == "CANONICAL" and issue.path.endswith("README.md") for issue in report.issues
    )


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

from mojilex_cli.dataset import load_dataset, validate_dataset, validate_snapshot
from mojilex_cli.domain import ContentRating, Review, reviewed_content_sha256
from test_dataset_helpers import write_fixture


def _codes(report) -> set[str]:
    return {item.code for item in report.issues}


def test_valid_fixture_passes_all_core_integrity_checks(tmp_path) -> None:
    write_fixture(tmp_path)
    assert validate_dataset(tmp_path, strict=False).valid
    assert validate_snapshot(load_dataset(tmp_path), canonical=True).valid


def test_detects_dangling_membership_duplicate_position_and_item_count(tmp_path) -> None:
    write_fixture(tmp_path)
    loaded = load_dataset(tmp_path)
    membership = next(iter(loaded.memberships.values()))
    membership.emoji_id = "mxe_00000000-0000-5000-8000-000000000000"
    next(iter(loaded.collections.values())).item_count = 9
    codes = _codes(validate_snapshot(loaded))
    assert {"DANGLING", "ITEM_COUNT"}.issubset(codes)


def test_detects_fingerprint_and_review_hash_tampering(tmp_path) -> None:
    write_fixture(tmp_path)
    loaded = load_dataset(tmp_path)
    collection = next(iter(loaded.collections.values()))
    collection.extensions["telegram"]["set_fingerprint_sha256"] = "0" * 64
    emoji = next(iter(loaded.emojis.values()))
    emoji.review = Review(
        status="approved",
        reviewed_at="2026-09-10T19:00:00Z",
        reviewer="reviewer",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
        review_hash_profile_id="semantic-review-content-v3",
    )
    emoji.descriptions["en"].text = "Tampered after review."
    codes = _codes(validate_snapshot(loaded))
    assert {"FINGERPRINT", "REVIEW_HASH"}.issubset(codes)


def test_sensitive_content_does_not_require_approval(tmp_path) -> None:
    write_fixture(tmp_path)
    loaded = load_dataset(tmp_path)
    emoji = next(iter(loaded.emojis.values()))
    emoji.content.rating = ContentRating.SENSITIVE
    assert validate_snapshot(loaded).valid
    assert emoji.review.status.value == "unreviewed"


def test_no_media_and_secret_scan_are_fail_closed(tmp_path) -> None:
    write_fixture(tmp_path)
    (tmp_path / "leaked.png").write_bytes(b"not really an image")
    (tmp_path / "notes.txt").write_text(
        "synthetic credential 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
        encoding="utf-8",
    )
    report = validate_snapshot(load_dataset(tmp_path), repository_files=True)
    assert {"NO_MEDIA", "SECRET"}.issubset(_codes(report))


def test_local_candidate_preview_and_lsh_artifacts_are_forbidden_but_docs_are_not(
    tmp_path,
) -> None:
    snapshot = write_fixture(tmp_path)
    (tmp_path / "candidates.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "preview-metadata.json").write_text("{}\n", encoding="utf-8")
    local_index = tmp_path / "dedupe-index"
    local_index.mkdir()
    (local_index / "buckets.json").write_text("{}\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "candidates.json").write_text("{}\n", encoding="utf-8")

    issues = validate_snapshot(snapshot, repository_files=True).issues
    forbidden = {issue.path for issue in issues if issue.code == "LOCAL_ARTIFACT"}
    assert forbidden == {
        "candidates.json",
        "dedupe-index/buckets.json",
        "preview-metadata.json",
    }

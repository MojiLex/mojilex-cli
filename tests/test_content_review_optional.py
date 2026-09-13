from __future__ import annotations

import json
from pathlib import Path

import pytest

from mojilex_cli.dataset import load_dataset, validate_snapshot
from mojilex_cli.dataset.index import _publishable
from mojilex_cli.dataset.staging import AtomicDatasetWriter
from mojilex_cli.domain import (
    ContentRating,
    ContentWarning,
    Review,
    ReviewStatus,
    RoutingReason,
    reviewed_content_sha256,
)
from mojilex_cli.policy import (
    ReviewPriority,
    ReviewReason,
    official_submission_report,
    semantic_routing_reasons,
    should_escalate,
)
from test_build_index_determinism import _build, _prepare_distribution_fixture, _tree_bytes
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _description


@pytest.mark.parametrize("rating", list(ContentRating))
@pytest.mark.parametrize("warnings", [[], [ContentWarning.FLASHING]])
def test_content_labels_do_not_require_review_or_escalation(
    tmp_path: Path, rating: ContentRating, warnings: list[ContentWarning]
) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.content.rating = rating
    emoji.content.warnings = warnings
    # Older provenance remains valid and does not restore the retired content gate.
    emoji.provenance.routing_reason_codes = [RoutingReason.SENSITIVE_CONTENT]
    before = emoji.as_dict()

    assert validate_snapshot(snapshot).valid
    report = official_submission_report(snapshot)
    assert not report.blocking
    assert report.items[0].priority is ReviewPriority.LOW
    assert ReviewReason.MODERATION_UNCERTAINTY not in report.items[0].reason_codes
    assert emoji.as_dict() == before
    assert emoji.review.status is ReviewStatus.UNREVIEWED

    description = _description().model_dump(mode="json")
    description["content"] = {"rating": rating.value, "warnings": [item.value for item in warnings]}
    parsed = type(_description()).model_validate(description)
    assert semantic_routing_reasons(parsed) == ()
    assert not should_escalate("rules", (), escalation_model="strong-model")


@pytest.mark.parametrize("legacy_reason", [RoutingReason.SENSITIVE_CONTENT, "sensitive-content"])
def test_legacy_content_reason_alone_no_longer_escalates(legacy_reason: str) -> None:
    assert not should_escalate("rules", [legacy_reason], escalation_model="strong-model")
    assert should_escalate(
        "rules", [legacy_reason, RoutingReason.PARTIAL_TEXT], escalation_model="strong-model"
    )


@pytest.mark.parametrize("rating", list(ContentRating))
@pytest.mark.parametrize("warnings", [[], [ContentWarning.FLASHING]])
def test_release_includes_unreviewed_content_and_preserves_consumer_labels(
    tmp_path: Path, rating: ContentRating, warnings: list[ContentWarning]
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    snapshot = load_dataset(dataset)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.content.rating = rating
    emoji.content.warnings = warnings
    writer = AtomicDatasetWriter(dataset)
    for path, payload in snapshot.to_files().items():
        writer.stage_bytes(path, payload)
    writer.commit()
    before = _tree_bytes(dataset, include_transaction_lock=False)

    assert _publishable(emoji)
    output = tmp_path / "dist"
    _build(dataset, output)
    canonical = json.loads((output / "emojis.jsonl").read_bytes())
    active = json.loads((output / "emojis-active.jsonl").read_bytes())
    assert active == canonical
    assert active["review"]["status"] == "unreviewed"
    assert active["content"] == {
        "rating": rating.value,
        "warnings": [item.value for item in warnings],
    }
    for language in ("en", "ru"):
        row = json.loads((output / f"search-{language}.jsonl").read_bytes())
        assert row["content"] == active["content"]
        assert row["review"]["status"] == "unreviewed"
        assert row["review"]["attested"] is False
    assert _tree_bytes(dataset, include_transaction_lock=False) == before


@pytest.mark.parametrize("status", ["changes_requested", "rejected"])
def test_explicit_negative_review_still_excludes_active_release(
    tmp_path: Path, status: str
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    snapshot = load_dataset(dataset)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.content.rating = ContentRating.SENSITIVE
    emoji.content.warnings = [ContentWarning.FLASHING]
    emoji.review = Review(
        status=status,
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:00:00Z",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
        review_hash_profile_id="semantic-review-content-v3",
    )
    writer = AtomicDatasetWriter(dataset)
    for path, payload in snapshot.to_files().items():
        writer.stage_bytes(path, payload)
    writer.commit()
    assert not _publishable(emoji)
    output = tmp_path / "dist"
    _build(dataset, output)
    assert not (output / "emojis-active.jsonl").read_bytes()
    assert not (output / "search-en.jsonl").read_bytes()
    assert json.loads((output / "emojis.jsonl").read_bytes())["review"]["status"] == status


def test_warning_structure_and_existing_review_hash_are_still_validated(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.review = Review(
        status="approved",
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:00:00Z",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
        review_hash_profile_id="semantic-review-content-v3",
    )
    emoji.content.warnings = [ContentWarning.NUDITY, ContentWarning.FLASHING]
    codes = {issue.code for issue in validate_snapshot(snapshot).issues}
    assert {"SORT", "REVIEW_HASH"}.issubset(codes)
    assert "POLICY_REVIEW" not in codes

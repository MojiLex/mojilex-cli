from __future__ import annotations

import json
from pathlib import Path

import pytest

from mojilex_cli.dataset import validate_snapshot
from mojilex_cli.domain import ContentRating, ContentWarning, Review, reviewed_content_sha256
from mojilex_cli.policy import official_submission_report
from test_dataset_helpers import write_fixture


def _approve(emoji) -> None:
    emoji.review = Review(
        status="approved",
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:00:00Z",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
        review_hash_profile_id="semantic-review-content-v3",
    )


@pytest.mark.parametrize("approved", [False, True])
def test_missing_qualification_is_not_an_approval_requirement(
    tmp_path: Path, approved: bool
) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.provenance.qualification_id = None
    emoji.content.rating = ContentRating.SENSITIVE
    emoji.content.warnings = [ContentWarning.FLASHING]
    if approved:
        _approve(emoji)
    before = emoji.as_dict()
    snapshot.source_bytes = snapshot.to_files()

    assert validate_snapshot(snapshot, canonical=True).valid
    assert not official_submission_report(snapshot).blocking
    assert emoji.as_dict() == before
    assert emoji.provenance.qualification_id is None
    assert emoji.review.status.value == ("approved" if approved else "unreviewed")


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("problem", ["unknown", "mismatch", "revoked", "expired"])
def test_existing_qualification_claim_must_be_valid_even_after_human_review(
    tmp_path: Path, approved: bool, problem: str
) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    if problem == "unknown":
        emoji.provenance.qualification_id = "mq_unknown-claim"
    elif problem == "mismatch":
        emoji.provenance.model = "other-model"
    else:
        path = tmp_path / "quality" / "model-qualifications.json"
        registry = json.loads(path.read_bytes())
        entry = registry["entries"][0]
        if problem == "revoked":
            entry["status"] = "revoked"
        else:
            entry["valid_until"] = "2026-07-01T00:00:00Z"
        path.write_text(json.dumps(registry) + "\n", encoding="utf-8", newline="\n")
    if approved:
        _approve(emoji)
    before = emoji.as_dict()
    snapshot.source_bytes = snapshot.to_files()

    report = validate_snapshot(snapshot, canonical=True)
    assert not report.valid
    assert "QUALIFICATION" in {issue.code for issue in report.issues}
    assert "POLICY_REVIEW_BLOCKING" not in {issue.code for issue in report.issues}
    assert emoji.as_dict() == before


@pytest.mark.parametrize("approved", [False, True])
def test_valid_qualification_claim_remains_valid(tmp_path: Path, approved: bool) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    if approved:
        _approve(emoji)
    snapshot.source_bytes = snapshot.to_files()
    assert validate_snapshot(snapshot, canonical=True).valid


def test_absent_claim_does_not_hide_a_malformed_qualification_registry(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.provenance.qualification_id = None
    (tmp_path / "quality" / "model-qualifications.json").write_text("{}\n", encoding="utf-8")
    snapshot.source_bytes = snapshot.to_files()
    assert "QUALIFICATION_REGISTRY" in {
        issue.code for issue in validate_snapshot(snapshot, canonical=True).issues
    }

from __future__ import annotations

from pathlib import Path

import pytest

from mojilex_cli.dataset import DatasetSnapshot
from mojilex_cli.domain import (
    Review,
    RoutingReason,
    Uncertainty,
    emoji_id,
    reviewed_content_sha256,
)
from mojilex_cli.policy import (
    ModelQualificationRegistry,
    PolicyError,
    ReviewGateError,
    ReviewPriority,
    ReviewReason,
    ReviewRoutingReport,
    compute_review_routing,
    load_review_policy,
    official_submission_report,
)
from test_dataset_helpers import write_fixture


def _report(snapshot: DatasetSnapshot) -> ReviewRoutingReport:
    qualifications = ModelQualificationRegistry.load(snapshot.root)
    _, policy = load_review_policy(snapshot.root)
    return compute_review_routing(snapshot, qualifications, policy)


def test_unqualified_is_blocking_until_valid_human_approval(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    emoji.provenance.qualification_id = None
    emoji.provenance.routing_reason_codes = [RoutingReason.UNQUALIFIED_MODEL]

    item = _report(snapshot).items[0]
    assert item.priority is ReviewPriority.BLOCKING
    assert ReviewReason.UNQUALIFIED_MODEL in item.reason_codes
    try:
        official_submission_report(snapshot)
    except ReviewGateError as exc:
        assert exc.report.blocking_count == 1
    else:  # pragma: no cover - protects the fail-closed contract
        raise AssertionError("unqualified official submission was not blocked")

    emoji.review = Review(
        status="approved",
        reviewed_at="2026-09-11T20:00:00Z",
        reviewer="reviewer",
        reviewed_content_sha256=reviewed_content_sha256(emoji),
    )
    approved = _report(snapshot).items[0]
    assert approved.priority is ReviewPriority.LOW
    assert ReviewReason.UNQUALIFIED_MODEL not in approved.reason_codes
    assert emoji.provenance.routing_reason_codes == [RoutingReason.UNQUALIFIED_MODEL]
    assert not official_submission_report(snapshot).blocking


def test_official_gate_requires_all_versioned_policy_registries(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    (tmp_path / "quality" / "routing-reasons-v1.json").unlink()

    with pytest.raises(PolicyError, match="routing-reasons-v1"):
        official_submission_report(snapshot)


def test_review_policy_computes_high_and_exact_group_conflict_without_persisting(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path)
    first = next(iter(snapshot.emojis.values()))
    first.facets.uncertainties = [Uncertainty.CHARACTER_OR_BRAND]
    second = first.model_copy(deep=True)
    second.native_id = "9999999999999999999"
    second.id = emoji_id(
        second.platform,
        second.native_namespace,
        second.scope_id,
        second.native_id,
    )
    second.descriptions["en"].text = "A materially different synthetic description."
    snapshot.emojis[second.id] = second

    report = _report(snapshot)
    by_id = {item.emoji_id: item for item in report.items}
    assert by_id[first.id].priority is ReviewPriority.HIGH
    assert ReviewReason.UNKNOWN_CHARACTER_OR_BRAND in by_id[first.id].reason_codes
    assert ReviewReason.EXACT_GROUP_DESCRIPTION_CONFLICT in by_id[first.id].reason_codes
    assert ReviewReason.EXACT_GROUP_DESCRIPTION_CONFLICT in by_id[second.id].reason_codes
    assert "review_priority" not in first.as_dict()
    assert "review_reason_codes" not in first.as_dict()

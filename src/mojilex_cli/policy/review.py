"""Computed review routing reports and the official-submission gate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mojilex_cli.dataset.repository import DatasetSnapshot
from mojilex_cli.dataset.serialization import parse_json
from mojilex_cli.domain import (
    Emoji,
    RoutingReason,
    Uncertainty,
)

from .qualification import ModelQualificationRegistry
from .routing import PolicyError, RoutingReasonRegistry


class ReviewPriority(StrEnum):
    BLOCKING = "blocking"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ReviewReason(StrEnum):
    CULTURAL_REFERENCE_UNCERTAINTY = "cultural-reference-uncertainty"
    EXACT_GROUP_DESCRIPTION_CONFLICT = "exact-group-description-conflict"
    MODERATION_UNCERTAINTY = "moderation-uncertainty"
    MOTION_UNCERTAINTY = "motion-uncertainty"
    OCR_CONFLICT = "ocr-conflict"
    TEXT_UNCERTAINTY = "text-uncertainty"
    UNKNOWN_CHARACTER_OR_BRAND = "unknown-character-or-brand"
    UNQUALIFIED_MODEL = "unqualified-model"


class ReviewGateError(ValueError):
    """Official submission contains at least one computed blocking item."""

    code = "POLICY_REVIEW_BLOCKING"

    def __init__(self, report: ReviewRoutingReport) -> None:
        self.report = report
        super().__init__(f"official submit is blocked by {report.blocking_count} review item(s)")


class ReviewReasonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ReviewReason
    definition: str = Field(min_length=1, max_length=1000)


class ReviewReasonRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    registry_id: Literal["review-reasons-v1"]
    entries: tuple[ReviewReasonEntry, ...]

    @field_validator("entries")
    @classmethod
    def canonical_entries(
        cls, value: tuple[ReviewReasonEntry, ...]
    ) -> tuple[ReviewReasonEntry, ...]:
        identifiers = [entry.id.value for entry in value]
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("review reasons must be unique and sorted")
        if set(identifiers) != {reason.value for reason in ReviewReason}:
            raise ValueError("review reason registry must match the supported v1 enum exactly")
        return value


class ReviewRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: ReviewReason
    priority: ReviewPriority
    condition: str = Field(min_length=1, max_length=256)


_EXPECTED_RULES = {
    ReviewReason.CULTURAL_REFERENCE_UNCERTAINTY: (
        ReviewPriority.HIGH,
        "facets-uncertainties-contains-cultural-reference",
    ),
    ReviewReason.TEXT_UNCERTAINTY: (
        ReviewPriority.HIGH,
        "facets-uncertainties-contains-text",
    ),
    ReviewReason.UNQUALIFIED_MODEL: (
        ReviewPriority.BLOCKING,
        "ai-result-has-no-exact-active-qualification-and-review-is-not-approved",
    ),
    ReviewReason.MODERATION_UNCERTAINTY: (
        ReviewPriority.BLOCKING,
        "rating-is-not-general-or-warnings-are-not-empty-and-review-is-not-approved",
    ),
    ReviewReason.MOTION_UNCERTAINTY: (
        ReviewPriority.HIGH,
        "facets-uncertainties-contains-motion",
    ),
    ReviewReason.OCR_CONFLICT: (
        ReviewPriority.HIGH,
        "routing-reason-codes-contains-ocr-conflict",
    ),
    ReviewReason.UNKNOWN_CHARACTER_OR_BRAND: (
        ReviewPriority.HIGH,
        "facets-uncertainties-contains-character-or-brand",
    ),
    ReviewReason.EXACT_GROUP_DESCRIPTION_CONFLICT: (
        ReviewPriority.NORMAL,
        "descriptions-conflict-inside-an-exact-group",
    ),
}


class ReviewRoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    policy_id: Literal["review-routing-v1"]
    priority_order: tuple[ReviewPriority, ...]
    rules: tuple[ReviewRule, ...]

    @model_validator(mode="after")
    def exact_supported_policy(self) -> ReviewRoutingPolicy:
        if self.priority_order != (
            ReviewPriority.BLOCKING,
            ReviewPriority.HIGH,
            ReviewPriority.NORMAL,
            ReviewPriority.LOW,
        ):
            raise ValueError("review priority order does not match review-routing-v1")
        actual = {rule.reason_code: (rule.priority, rule.condition) for rule in self.rules}
        if len(actual) != len(self.rules) or actual != _EXPECTED_RULES:
            raise ValueError("review-routing-v1 contains unknown, missing, or changed rules")
        return self


@dataclass(frozen=True, slots=True)
class ReviewRoutingItem:
    emoji_id: str
    priority: ReviewPriority
    reason_codes: tuple[ReviewReason, ...]
    review_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "emoji_id": self.emoji_id,
            "priority": self.priority.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "review_status": self.review_status,
        }


@dataclass(frozen=True, slots=True)
class ReviewRoutingReport:
    policy_id: str
    items: tuple[ReviewRoutingItem, ...]

    @property
    def blocking_count(self) -> int:
        return sum(item.priority is ReviewPriority.BLOCKING for item in self.items)

    @property
    def blocking(self) -> bool:
        return self.blocking_count > 0

    def counts(self) -> dict[str, int]:
        return {
            priority.value: sum(item.priority is priority for item in self.items)
            for priority in ReviewPriority
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "counts": self.counts(),
            "blocking": self.blocking,
            "items": [item.as_dict() for item in self.items],
        }


def load_review_policy(root: Path) -> tuple[ReviewReasonRegistry, ReviewRoutingPolicy]:
    reasons_path = root / "quality" / "review-reasons-v1.json"
    routing_path = root / "quality" / "review-routing-v1.json"
    try:
        reasons = ReviewReasonRegistry.model_validate(
            parse_json(reasons_path.read_bytes(), source=str(reasons_path))
        )
        policy = ReviewRoutingPolicy.model_validate(
            parse_json(routing_path.read_bytes(), source=str(routing_path))
        )
    except (OSError, ValueError) as exc:
        raise PolicyError("review-routing-v1 policy files are missing or invalid") from exc
    registered = {entry.id for entry in reasons.entries}
    if any(rule.reason_code not in registered for rule in policy.rules):
        raise PolicyError("review-routing-v1 uses an unregistered reason")
    return reasons, policy


def compute_review_routing(
    snapshot: DatasetSnapshot,
    qualifications: ModelQualificationRegistry,
    policy: ReviewRoutingPolicy,
) -> ReviewRoutingReport:
    """Compute the complete non-canonical staging/PR review report."""

    exact_conflicts = _exact_group_description_conflicts(snapshot)
    rank = {priority: index for index, priority in enumerate(policy.priority_order)}
    rule_priorities = {rule.reason_code: rule.priority for rule in policy.rules}
    items: list[ReviewRoutingItem] = []
    for emoji in sorted(snapshot.emojis.values(), key=lambda value: value.id):
        reasons: set[ReviewReason] = set()
        if Uncertainty.MOTION in emoji.facets.uncertainties:
            reasons.add(ReviewReason.MOTION_UNCERTAINTY)
        if Uncertainty.TEXT in emoji.facets.uncertainties:
            reasons.add(ReviewReason.TEXT_UNCERTAINTY)
        if Uncertainty.CULTURAL_REFERENCE in emoji.facets.uncertainties:
            reasons.add(ReviewReason.CULTURAL_REFERENCE_UNCERTAINTY)
        if RoutingReason.OCR_CONFLICT in (emoji.provenance.routing_reason_codes or []):
            reasons.add(ReviewReason.OCR_CONFLICT)
        if Uncertainty.CHARACTER_OR_BRAND in emoji.facets.uncertainties:
            reasons.add(ReviewReason.UNKNOWN_CHARACTER_OR_BRAND)
        if emoji.id in exact_conflicts:
            reasons.add(ReviewReason.EXACT_GROUP_DESCRIPTION_CONFLICT)
        priority = min(
            (rule_priorities[reason] for reason in reasons),
            key=lambda value: rank[value],
            default=ReviewPriority.LOW,
        )
        items.append(
            ReviewRoutingItem(
                emoji_id=emoji.id,
                priority=priority,
                reason_codes=tuple(sorted(reasons, key=str)),
                review_status=emoji.review.status.value,
            )
        )
    return ReviewRoutingReport(policy_id=policy.policy_id, items=tuple(items))


def official_submission_report(snapshot: DatasetSnapshot) -> ReviewRoutingReport:
    """Load every normative policy input and enforce the blocking gate."""

    qualifications = ModelQualificationRegistry.load(snapshot.root)
    RoutingReasonRegistry.load(snapshot.root)
    _, policy = load_review_policy(snapshot.root)
    report = compute_review_routing(snapshot, qualifications, policy)
    if report.blocking:
        raise ReviewGateError(report)
    return report


def _exact_group_description_conflicts(snapshot: DatasetSnapshot) -> set[str]:
    groups: dict[tuple[str, tuple[tuple[str, str, str], ...]], list[Emoji]] = defaultdict(list)
    for emoji in snapshot.emojis.values():
        media_signature = tuple(
            sorted((item.role.value, item.variant_id or "", item.sha256) for item in emoji.media)
        )
        decoded_signature = tuple(
            sorted(
                (
                    item.role.value,
                    item.variant_id or "",
                    item.decoded_payload_sha256,
                )
                for item in emoji.fingerprints.items
            )
        )
        groups[("binary", media_signature)].append(emoji)
        if emoji.fingerprints.status.value == "complete":
            groups[("decoded", decoded_signature)].append(emoji)
    conflicts: set[str] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        descriptions = {
            tuple(
                sorted(
                    (
                        language,
                        value.text,
                        value.motion_status.value,
                        value.motion or "",
                        tuple(value.usage),
                    )
                    for language, value in emoji.descriptions.items()
                )
            )
            for emoji in group
        }
        if len(descriptions) > 1:
            conflicts.update(emoji.id for emoji in group)
    return conflicts

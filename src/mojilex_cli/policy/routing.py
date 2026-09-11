"""Versioned, deterministic semantic routing decisions."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mojilex_cli.ai import DescriptionItem
from mojilex_cli.dataset.serialization import parse_json
from mojilex_cli.domain import RoutingReason
from mojilex_cli.media import ProcessedMedia

ROUTING_POLICY_VERSION = "1.0.0"


class PolicyError(ValueError):
    """A versioned routing/review policy is absent or unsupported."""

    code = "POLICY_INVALID"


class RoutingMode(StrEnum):
    OFF = "off"
    RULES = "rules"


class RoutingReasonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: RoutingReason
    definition: str = Field(min_length=1, max_length=1000)


class RoutingReasonRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    registry_id: Literal["routing-reasons-v1"]
    entries: tuple[RoutingReasonEntry, ...]

    @field_validator("entries")
    @classmethod
    def exact_canonical_entries(
        cls, value: tuple[RoutingReasonEntry, ...]
    ) -> tuple[RoutingReasonEntry, ...]:
        identifiers = [entry.id.value for entry in value]
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("routing reasons must be unique and sorted")
        required = {reason.value for reason in RoutingReason}
        if set(identifiers) != required:
            raise ValueError("routing reason registry must match the supported v1 enum exactly")
        return value

    @classmethod
    def load(cls, root: Path) -> RoutingReasonRegistry:
        path = root / "quality" / "routing-reasons-v1.json"
        try:
            raw = parse_json(path.read_bytes(), source=str(path))
            return cls.model_validate(raw)
        except (OSError, ValueError) as exc:
            raise PolicyError("quality/routing-reasons-v1.json is missing or invalid") from exc

    def canonicalize(self, reasons: Iterable[RoutingReason | str]) -> tuple[RoutingReason, ...]:
        allowed = {entry.id for entry in self.entries}
        try:
            result = tuple(sorted({RoutingReason(reason) for reason in reasons}, key=str))
        except ValueError as exc:
            raise PolicyError("routing produced a reason outside routing-reasons-v1") from exc
        if not set(result).issubset(allowed):
            raise PolicyError("routing produced a reason outside routing-reasons-v1")
        return result


def deterministic_routing_reasons(processed: ProcessedMedia) -> tuple[RoutingReason, ...]:
    """Derive only signals owned by the validated deterministic analyzer.

    No AI self-reported confidence or free-form provider metadata is accepted.
    ``low_information`` is the profile-versioned low visibility/entropy/edge
    signal; non-zero temporal energy across the required 16 samples is the only
    v1 complex-motion pre-route signal available to the pipeline.
    """

    analysis = processed.analysis
    if analysis is None:
        raise PolicyError("deterministic routing requires completed media analysis")
    reasons: set[RoutingReason] = set()
    perceptual = analysis.fingerprint.perceptual
    if (
        processed.metadata.format in {"tgs", "webm"}
        and perceptual.sample_count == 16
        and perceptual.temporal_energy_bp > 0
    ):
        reasons.add(RoutingReason.COMPLEX_MOTION)
    if perceptual.low_information:
        reasons.add(RoutingReason.LOW_VISIBILITY)
    return tuple(sorted(reasons, key=str))


def semantic_routing_reasons(description: DescriptionItem) -> tuple[RoutingReason, ...]:
    """Map schema-validated semantic fields to v1 post-primary reasons."""

    reasons: set[RoutingReason] = set()
    facets = description.facets
    if facets.text_content.status in {"partially-recognized", "unreadable"}:
        reasons.add(RoutingReason.PARTIAL_TEXT)
    if "character-or-brand" in facets.uncertainties:
        reasons.add(RoutingReason.CHARACTER_OR_BRAND)
    if "content-type" in facets.uncertainties or "style" in facets.uncertainties:
        reasons.add(RoutingReason.FACET_CONFLICT)
    if description.content.rating != "general" or description.content.warnings:
        reasons.add(RoutingReason.SENSITIVE_CONTENT)
    return tuple(sorted(reasons, key=str))


def should_escalate(
    mode: RoutingMode | str,
    reasons: Iterable[RoutingReason | str],
    *,
    escalation_model: str | None,
) -> bool:
    """Return one explicit route decision and fail closed on unsafe setup."""

    parsed_mode = RoutingMode(mode)
    materialized = tuple(reasons)
    if parsed_mode is RoutingMode.OFF:
        return False
    if not escalation_model:
        raise PolicyError("rules routing requires an explicit escalation model")
    return bool(materialized)

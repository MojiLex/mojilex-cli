"""Exact, time-aware model qualification matching for MLX-SPEC-002."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mojilex_cli.dataset.serialization import parse_json

_SHA256 = r"^[0-9a-f]{64}$"
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
CONCEPT_BINDING_FIELDS = (
    "concept_registry_id",
    "concept_registry_sha256",
    "concept_candidate_set_sha256",
    "concept_candidate_profile_id",
    "concept_candidate_profile_sha256",
    "model_routing_policy_id",
    "model_routing_policy_sha256",
)


class QualificationRegistryError(ValueError):
    """The normative model-qualification registry is missing or invalid."""

    code = "POLICY_INVALID"


class QualificationStatus(StrEnum):
    QUALIFIED = "qualified"
    NOT_FOUND = "not-found"
    QUALIFICATION_ID_NOT_FOUND = "qualification-id-not-found"
    MISMATCH = "mismatch"
    REVOKED = "revoked"
    OUTSIDE_VALIDITY = "outside-validity"
    AMBIGUOUS = "ambiguous"


class ConceptQualificationBinding(BaseModel):
    """Optional only for legacy results without a generated concept mapping."""

    concept_registry_id: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$", max_length=128
    )
    concept_registry_sha256: str | None = Field(default=None, pattern=_SHA256)
    concept_candidate_set_sha256: str | None = Field(default=None, pattern=_SHA256)
    concept_candidate_profile_id: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$", max_length=128
    )
    concept_candidate_profile_sha256: str | None = Field(default=None, pattern=_SHA256)
    model_routing_policy_id: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$", max_length=128
    )
    model_routing_policy_sha256: str | None = Field(default=None, pattern=_SHA256)

    @model_validator(mode="after")
    def complete_concept_binding(self) -> ConceptQualificationBinding:
        values = tuple(getattr(self, name) for name in CONCEPT_BINDING_FIELDS)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError("concept generation binding requires all seven exact fields")
        return self


class ModelQualification(ConceptQualificationBinding):
    """One immutable qualification tuple from the public registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualification_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^mq_[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    description_profile: str = Field(min_length=1, max_length=64)
    prompt_sha256: str = Field(pattern=_SHA256)
    request_parameters_sha256: str = Field(pattern=_SHA256)
    schema_version: Literal["1.0.0"]
    taxonomy_version: Literal["1.0.0"]
    pipeline_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    routing_policy_version: Literal["1.0.0"]
    languages: tuple[str, ...] = Field(min_length=1)
    benchmark_id: str = Field(min_length=1, max_length=128)
    benchmark_sha256: str = Field(pattern=_SHA256)
    split_id: str = Field(min_length=1, max_length=128)
    split_sha256: str = Field(pattern=_SHA256)
    report_sha256: str = Field(pattern=_SHA256)
    valid_from: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    valid_until: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
    )
    status: Literal["active", "revoked"]

    @field_validator("languages")
    @classmethod
    def canonical_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("qualification languages must be unique and sorted")
        return value

    @field_validator("valid_from", "valid_until")
    @classmethod
    def valid_utc_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_utc(value)
        return value

    @model_validator(mode="after")
    def valid_interval_and_revision(self) -> ModelQualification:
        if self.model_revision is None and self.valid_until is None:
            raise ValueError("valid_until is required when model_revision is absent")
        if self.valid_until is not None and _parse_utc(self.valid_until) <= _parse_utc(
            self.valid_from
        ):
            raise ValueError("valid_until must be later than valid_from")
        return self

    def tuple_matches(self, query: QualificationQuery) -> bool:
        if any(getattr(self, name) != getattr(query, name) for name in CONCEPT_BINDING_FIELDS):
            return False
        return all(
            (
                self.provider == query.provider,
                self.model == query.model,
                self.model_revision == query.model_revision,
                self.description_profile == query.description_profile,
                self.prompt_sha256 == query.prompt_sha256,
                self.request_parameters_sha256 == query.request_parameters_sha256,
                self.schema_version == query.schema_version,
                self.taxonomy_version == query.taxonomy_version,
                self.pipeline_version == query.pipeline_version,
                self.routing_policy_version == query.routing_policy_version,
                self.languages == query.languages,
            )
        )

    def covers(self, generated_at: str) -> bool:
        instant = _parse_utc(generated_at)
        if instant < _parse_utc(self.valid_from):
            return False
        return self.valid_until is None or instant < _parse_utc(self.valid_until)


class QualificationQuery(ConceptQualificationBinding):
    """Exact provenance tuple whose qualification is being resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    description_profile: str = Field(min_length=1, max_length=64)
    prompt_sha256: str = Field(pattern=_SHA256)
    request_parameters_sha256: str = Field(pattern=_SHA256)
    schema_version: Literal["1.0.0"] = "1.0.0"
    taxonomy_version: Literal["1.0.0"] = "1.0.0"
    pipeline_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    routing_policy_version: Literal["1.0.0"] = "1.0.0"
    languages: tuple[str, ...] = ("en", "ru")
    generated_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    @field_validator("languages", mode="before")
    @classmethod
    def normalize_languages(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            materialized = tuple(str(item) for item in value)
            if len(materialized) != len(set(materialized)):
                raise ValueError("qualification query languages must be unique")
            return tuple(sorted(materialized))
        return value

    @field_validator("generated_at")
    @classmethod
    def valid_generated_at(cls, value: str) -> str:
        _parse_utc(value)
        return value


class ModelQualificationRegistry(BaseModel):
    """The exact public registry format stored in ``quality/``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    registry_id: Literal["model-qualifications-v1"]
    entries: tuple[ModelQualification, ...]

    @field_validator("entries")
    @classmethod
    def canonical_entries(
        cls, value: tuple[ModelQualification, ...]
    ) -> tuple[ModelQualification, ...]:
        identifiers = [entry.qualification_id for entry in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("qualification IDs must be unique")
        if identifiers != sorted(identifiers):
            raise ValueError("qualification entries must be sorted by qualification_id")
        return value

    @classmethod
    def load(cls, root: Path) -> ModelQualificationRegistry:
        path = root / "quality" / "model-qualifications.json"
        try:
            raw = parse_json(path.read_bytes(), source=str(path))
            return cls.model_validate(raw)
        except (OSError, ValueError) as exc:
            raise QualificationRegistryError(
                "quality/model-qualifications.json is missing or invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class QualificationMatch:
    status: QualificationStatus
    qualification_id: str | None = None

    @property
    def qualified(self) -> bool:
        return self.status is QualificationStatus.QUALIFIED


def match_qualification(
    registry: ModelQualificationRegistry,
    query: QualificationQuery,
    *,
    qualification_id: str | None = None,
) -> QualificationMatch:
    """Match every normative tuple field and the generation-time half interval.

    ``valid_until`` is evaluated against ``generated_at`` rather than wall-clock
    time, so a later expiration never invalidates an older accepted result.
    Revocation is absolute, as the registry intentionally has no revocation time.
    """

    if qualification_id is not None:
        entry = next(
            (item for item in registry.entries if item.qualification_id == qualification_id),
            None,
        )
        if entry is None:
            return QualificationMatch(QualificationStatus.QUALIFICATION_ID_NOT_FOUND)
        if not entry.tuple_matches(query):
            return QualificationMatch(QualificationStatus.MISMATCH, entry.qualification_id)
        if entry.status == "revoked":
            return QualificationMatch(QualificationStatus.REVOKED, entry.qualification_id)
        if not entry.covers(query.generated_at):
            return QualificationMatch(QualificationStatus.OUTSIDE_VALIDITY, entry.qualification_id)
        return QualificationMatch(QualificationStatus.QUALIFIED, entry.qualification_id)

    exact = [entry for entry in registry.entries if entry.tuple_matches(query)]
    if not exact:
        return QualificationMatch(QualificationStatus.NOT_FOUND)
    active_at_generation = [
        entry for entry in exact if entry.status == "active" and entry.covers(query.generated_at)
    ]
    if len(active_at_generation) == 1:
        return QualificationMatch(
            QualificationStatus.QUALIFIED,
            active_at_generation[0].qualification_id,
        )
    if len(active_at_generation) > 1:
        return QualificationMatch(QualificationStatus.AMBIGUOUS)
    if any(entry.status == "revoked" for entry in exact):
        return QualificationMatch(QualificationStatus.REVOKED)
    return QualificationMatch(QualificationStatus.OUTSIDE_VALIDITY)


def _parse_utc(value: str) -> datetime:
    try:
        return datetime.strptime(value, _UTC_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("timestamp must be a real UTC second ending in Z") from exc

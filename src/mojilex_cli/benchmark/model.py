"""Manifest-driven, fail-closed AI description qualification benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mojilex_cli.ai import (
    DescriptionItem,
    DescriptionRequest,
    RequestBudget,
    VisionContext,
    VisionImage,
    VisionProvider,
    describe_with_recovery,
)
from mojilex_cli.ai.concepts import ConceptContext, concept_context_from_documents
from mojilex_cli.ai.prompts import (
    PROMPT_VERSION,
    gemini_request_parameters_sha256,
    prompt_sha256,
)
from mojilex_cli.domain import SCHEMA_VERSION
from mojilex_cli.media import PIPELINE_VERSION
from mojilex_cli.policy.model_routing import build_model_routing_binding
from mojilex_cli.policy.routing import ROUTING_POLICY_VERSION

from .common import (
    ID_PATTERN,
    SEMVER_PATTERN,
    SHA256_PATTERN,
    BenchmarkError,
    DeclaredFile,
    RightsRecord,
    finalize_report,
    load_manifest,
    ratio_bp,
    resolve_declared_file,
    validate_canonical_ids,
    wilson_interval_bp,
)

MANDATORY_MODEL_STRATA = (
    "adaptive",
    "ambiguous-emotion-gesture",
    "low-contrast-temporal-text",
    "static-full-color",
    "technical-symbol-pattern",
    "text-code",
    "tgs-animation",
    "webm-animation",
)

_TEXT_FIELDS = (
    "factuality",
    "main_content_completeness",
    "no_extra_assumption",
    "motion_accuracy",
    "text_accuracy",
    "russian_naturalness",
    "english_naturalness",
    "facet_consistency",
)
_URL_RE = re.compile(r"(?i)https?://[^\s\"'<>]+")
_LOCAL_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+|"
    r"/(?:home|users|tmp|var|private|root|opt|etc)/)"
)
ScoreValue = Annotated[int, Field(ge=1, le=5)] | Literal["not_applicable"]
_FACET_VALUES = {
    "content_types": frozenset(
        {
            "reaction",
            "character",
            "person",
            "animal",
            "body-part",
            "object",
            "food-drink",
            "plant",
            "nature",
            "activity",
            "place",
            "vehicle",
            "flag",
            "symbol",
            "technical-icon",
            "text",
            "number",
            "logo",
            "scene",
            "pattern",
            "abstract",
        }
    ),
    "styles": frozenset(
        {
            "flat",
            "three-dimensional",
            "pixel-art",
            "hand-drawn",
            "photorealistic",
            "cartoon",
            "anime",
            "minimal",
            "detailed",
            "outline",
            "solid",
            "gradient",
            "neon",
            "sticker-like",
            "ornamental",
        }
    ),
    "suggested_uses": frozenset(
        {
            "bot-interface",
            "app-interface",
            "navigation",
            "button-icon",
            "status",
            "profile-avatar",
            "profile-background",
            "topic-icon",
            "badge",
            "counter",
            "label",
            "notification",
            "message-accent",
            "decoration",
            "branding",
        }
    ),
    "uncertainties": frozenset(
        {
            "text",
            "content-type",
            "style",
            "suggested-use",
            "cultural-reference",
            "character-or-brand",
            "motion",
        }
    ),
}


class ManualScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    factuality: int = Field(ge=1, le=5)
    main_content_completeness: ScoreValue
    no_extra_assumption: ScoreValue
    motion_accuracy: ScoreValue
    text_accuracy: ScoreValue
    russian_naturalness: ScoreValue
    english_naturalness: ScoreValue
    facet_consistency: ScoreValue


class HumanAdjudication(BaseModel):
    """Human evidence is bound to exact structured output bytes by hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewer_count: int = Field(ge=1)
    hallucinated_observable_fact: bool
    brand_identity_hallucination: bool
    critical_error: bool
    ru_en_consistent: bool
    uncertainty_appropriate: bool
    full_pass: bool
    substantive_error_count: int = Field(ge=0)
    scores: ManualScores

    @model_validator(mode="after")
    def consistent_decision(self) -> HumanAdjudication:
        has_error = (
            self.hallucinated_observable_fact
            or self.brand_identity_hallucination
            or self.critical_error
            or self.substantive_error_count > 0
        )
        if self.full_pass and has_error:
            raise ValueError("full-pass adjudication cannot also declare an error")
        if self.critical_error and self.substantive_error_count == 0:
            raise ValueError("critical adjudication requires a substantive error count")
        return self


class RequiredFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(pattern=ID_PATTERN)
    ru_phrases: tuple[str, ...] = Field(min_length=1)
    en_phrases: tuple[str, ...] = Field(min_length=1)

    @field_validator("ru_phrases", "en_phrases")
    @classmethod
    def canonical_phrases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            list(value) != sorted(value)
            or len(value) != len(set(value))
            or any(not _safe_phrase(item) for item in value)
        ):
            raise ValueError("fact phrase alternatives must be safe, unique, and sorted")
        return value


class AllowedFacetSets(BaseModel):
    """Exact acceptable controlled-value sets after human calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_types: tuple[tuple[str, ...], ...] = Field(min_length=1)
    styles: tuple[tuple[str, ...], ...] = Field(min_length=1)
    suggested_uses: tuple[tuple[str, ...], ...] = Field(min_length=1)
    uncertainties: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_allowed_sets(self) -> AllowedFacetSets:
        for field_name, legal in _FACET_VALUES.items():
            alternatives = getattr(self, field_name)
            if list(alternatives) != sorted(alternatives) or len(alternatives) != len(
                set(alternatives)
            ):
                raise ValueError("allowed facet alternatives must be unique and sorted")
            for alternative in alternatives:
                if tuple(sorted(set(alternative))) != alternative or not set(alternative) <= legal:
                    raise ValueError("allowed facet set is noncanonical or has unknown values")
            if field_name == "content_types" and any(not value for value in alternatives):
                raise ValueError("allowed content type sets cannot be empty")
        return self


class ModelBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=ID_PATTERN)
    split: Literal["development", "holdout"]
    rights_id: str = Field(pattern=ID_PATTERN)
    strata: tuple[str, ...] = Field(min_length=1)
    isolation_groups: tuple[str, ...] = Field(min_length=1)
    source_media_sha256: str = Field(pattern=SHA256_PATTERN)
    image: DeclaredFile
    needs_repainting: bool
    animated: bool
    background_variants: tuple[Literal["light", "dark"], ...] = ("light",)
    expected: DescriptionItem
    allowed_facets: AllowedFacetSets
    required_facts: tuple[RequiredFact, ...]
    forbidden_claims: tuple[str, ...]
    text_present: bool
    text_readable: bool
    ambiguous: bool
    brand_identity_present: bool
    injection_test: bool
    weight: int = Field(default=1, ge=1, le=100)
    adjudication: HumanAdjudication | None = None
    additional_adjudications: tuple[HumanAdjudication, ...] = ()

    @field_validator("forbidden_claims")
    @classmethod
    def canonical_forbidden_claims(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            list(value) != sorted(value)
            or len(value) != len(set(value))
            or any(not _safe_phrase(item) for item in value)
        ):
            raise ValueError("forbidden claims must be safe, unique, and sorted")
        return value

    @model_validator(mode="after")
    def canonical_case(self) -> ModelBenchmarkCase:
        hashes = [item.response_sha256 for item in self.additional_adjudications]
        if hashes != sorted(set(hashes)) or (
            self.adjudication is not None and self.adjudication.response_sha256 in hashes
        ):
            raise ValueError("human adjudication response hashes must be unique and sorted")
        validate_canonical_ids(self.strata, label="model benchmark strata")
        validate_canonical_ids(self.isolation_groups, label="split isolation groups")
        fact_ids = [item.fact_id for item in self.required_facts]
        validate_canonical_ids(fact_ids, label="required facts")
        if self.expected.label != "E001":
            raise ValueError("benchmark expected result must use label E001")
        expected_facets = self.expected.facets
        for field_name in _FACET_VALUES:
            if tuple(getattr(expected_facets, field_name)) not in getattr(
                self.allowed_facets, field_name
            ):
                raise ValueError("expected controlled facets must be explicitly allowed")
        text_status = self.expected.facets.text_content.status
        if self.text_present != (text_status != "none"):
            raise ValueError("text_present disagrees with expected text_content status")
        if self.text_readable != (text_status in {"recognized", "partially-recognized"}):
            raise ValueError("text_readable disagrees with expected text_content status")
        canonical_backgrounds = (
            ("light", "dark") if "dark" in self.background_variants else ("light",)
        )
        if self.background_variants != canonical_backgrounds:
            raise ValueError("background variants must be canonical and include light")
        animation_strata = {"tgs-animation", "webm-animation"} & set(self.strata)
        if self.animated != bool(animation_strata) or (
            self.animated and "static-full-color" in self.strata
        ):
            raise ValueError("animated flag must match TGS/WebM strata")
        if self.needs_repainting != ("adaptive" in self.strata):
            raise ValueError("needs_repainting must match adaptive stratum")
        if self.ambiguous != ("ambiguous-emotion-gesture" in self.strata):
            raise ValueError("ambiguous flag must match ambiguous stratum")
        if self.ambiguous and not self.expected.facets.uncertainties:
            raise ValueError("ambiguous benchmark item requires expected uncertainty")
        expected_motion = {
            self.expected.descriptions.ru.motion_status,
            self.expected.descriptions.en.motion_status,
        }
        if self.animated and expected_motion == {"not_applicable"}:
            raise ValueError("animated benchmark item requires expected motion evidence")
        if not self.animated and expected_motion != {"not_applicable"}:
            raise ValueError("static benchmark item must mark motion not applicable")
        return self


def model_benchmark_assets_sha256(cases: tuple[ModelBenchmarkCase, ...]) -> str:
    payload: list[dict[str, Any]] = [
        {
            "case_id": item.case_id,
            "image": item.image.model_dump(mode="json"),
            "source_media_sha256": item.source_media_sha256,
        }
        for item in cases
    ]
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def model_benchmark_split_sha256(cases: tuple[ModelBenchmarkCase, ...]) -> str:
    payload: list[dict[str, Any]] = [
        {
            "case_id": item.case_id,
            "isolation_groups": list(item.isolation_groups),
            "source_media_sha256": item.source_media_sha256,
            "split": item.split,
        }
        for item in cases
    ]
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


class ModelBenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_type: Literal["mojilex-model-benchmark-v1"]
    schema_version: Literal["1.0.0"]
    benchmark_id: str = Field(pattern=ID_PATTERN)
    benchmark_version: str = Field(pattern=SEMVER_PATTERN)
    benchmark_assets_sha256: str = Field(pattern=SHA256_PATTERN)
    split_sha256: str = Field(pattern=SHA256_PATTERN)
    comparison_kind: Literal["immutable-revision", "dated"]
    run_started_at_utc: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
    )
    target_provider: str = Field(pattern=ID_PATTERN)
    target_model: str = Field(min_length=1, max_length=200)
    target_model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    description_profile: Literal["standard-v1"]
    prompt_version: str = Field(pattern=SEMVER_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    request_parameters_sha256: str = Field(pattern=SHA256_PATTERN)
    output_schema_version: str = Field(pattern=SEMVER_PATTERN)
    taxonomy_version: str = Field(pattern=SEMVER_PATTERN)
    media_pipeline_version: str = Field(pattern=SEMVER_PATTERN)
    routing_rules_version: str = Field(pattern=SEMVER_PATTERN)
    generation_stage: Literal["primary"] = "primary"
    semantic_escalation_count: Literal[0] = 0
    languages: tuple[Literal["ru", "en"], Literal["ru", "en"]]
    cli_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    lockfile_sha256: str = Field(pattern=SHA256_PATTERN)
    container_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    rights: tuple[RightsRecord, ...] = Field(min_length=1)
    cases: tuple[ModelBenchmarkCase, ...] = Field(min_length=1)
    styles_macro_f1_min_bp: int = Field(ge=0, le=10_000)
    suggested_uses_macro_f1_min_bp: int = Field(ge=0, le=10_000)
    max_requests: int = Field(ge=1)
    max_cost_usd: Decimal | None = Field(default=None, ge=0)
    allow_unknown_cost: bool = False
    holdout_run_count: int = Field(default=3, ge=1)
    single_review_limitation: bool = False
    concept_registry: dict[str, Any] | None = None
    concept_candidate_profile: dict[str, Any] | None = None

    @field_validator("target_model")
    @classmethod
    def safe_model_id(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("target model ID is invalid")
        return value

    @field_validator("target_model_revision")
    @classmethod
    def safe_model_revision(cls, value: str | None) -> str | None:
        if value is not None and (
            value != value.strip() or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("target model revision is invalid")
        return value

    @field_validator("run_started_at_utc")
    @classmethod
    def valid_run_time(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValueError("benchmark run time must be a real UTC timestamp") from exc
        if parsed.year < 2020:
            raise ValueError("benchmark run time is implausibly old")
        return value

    @model_validator(mode="after")
    def manifest_integrity(self) -> ModelBenchmarkManifest:
        if (self.concept_registry is None) != (self.concept_candidate_profile is None):
            raise ValueError("benchmark concept registry and candidate profile are atomic")
        if self.concept_registry is not None and self.concept_candidate_profile is not None:
            context = concept_context_from_documents(
                self.concept_registry, self.concept_candidate_profile
            )
            for case in self.cases:
                context.validate_selection(case.expected.concept_ids)
        if self.languages != ("ru", "en"):
            raise ValueError("benchmark languages must be ordered as ru, en")
        if self.comparison_kind == "immutable-revision" and self.target_model_revision is None:
            raise ValueError("immutable comparison requires a model revision")
        if self.comparison_kind == "dated" and self.target_model_revision is not None:
            raise ValueError("dated comparison is used only when no immutable revision exists")
        rights_ids = [item.rights_id for item in self.rights]
        case_ids = [item.case_id for item in self.cases]
        validate_canonical_ids(rights_ids, label="rights records")
        validate_canonical_ids(case_ids, label="model benchmark cases")
        if any(not item.redistribution_allowed for item in self.rights):
            raise ValueError("all benchmark fixtures must permit redistribution")
        known_rights = set(rights_ids)
        if any(item.rights_id not in known_rights for item in self.cases):
            raise ValueError("model case references unknown rights metadata")
        if (
            any(
                adjudication.reviewer_count == 1
                for item in self.cases
                for adjudication in (
                    *((item.adjudication,) if item.adjudication is not None else ()),
                    *item.additional_adjudications,
                )
            )
            and not self.single_review_limitation
        ):
            raise ValueError("single-review adjudication must be disclosed in the manifest")
        split_by_group: dict[str, str] = {}
        for item in self.cases:
            groups = (*item.isolation_groups, f"media-sha256:{item.source_media_sha256}")
            for group in groups:
                previous = split_by_group.setdefault(group, item.split)
                if previous != item.split:
                    raise ValueError("collection/exact/artwork group crosses benchmark split")
        image_paths = [item.image.path for item in self.cases]
        if len(image_paths) != len(set(image_paths)):
            raise ValueError("each model benchmark case must declare a unique image path")
        if model_benchmark_assets_sha256(self.cases) != self.benchmark_assets_sha256:
            raise ValueError("benchmark_assets_sha256 does not match declared model inputs")
        if model_benchmark_split_sha256(self.cases) != self.split_sha256:
            raise ValueError("split_sha256 does not match model benchmark split declarations")
        return self


class ModelObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=ID_PATTERN)
    run_index: int = Field(default=1, ge=1, strict=True)
    schema_success: bool
    response: DescriptionItem | None = None
    response_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    provider: str | None = None
    model: str | None = None
    model_revision: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    requests_used: int = Field(ge=0)
    safety_pass: bool
    error_type: str | None = Field(default=None, pattern=ID_PATTERN)

    @model_validator(mode="after")
    def response_contract(self) -> ModelObservation:
        if self.schema_success != (self.response is not None):
            raise ValueError("schema_success must exactly match response presence")
        if self.response is not None:
            digest = description_item_sha256(self.response)
            if self.response_sha256 != digest or self.response.label != "E001":
                raise ValueError("response hash or label does not match structured output")
            if self.error_type is not None:
                raise ValueError("successful observation cannot contain an error type")
        elif self.response_sha256 is not None or self.error_type is None:
            raise ValueError("failed observation requires only a safe error type")
        return self


@dataclass(frozen=True, slots=True)
class _CaseEvaluation:
    case: ModelBenchmarkCase
    observation: ModelObservation
    adjudication: HumanAdjudication | None


Clock = Callable[[], int]


def load_model_benchmark_manifest(
    path: Path,
) -> tuple[ModelBenchmarkManifest, str, bytes]:
    return load_manifest(path, ModelBenchmarkManifest)


async def run_model_benchmark(
    manifest_path: Path,
    provider: VisionProvider,
    *,
    runtime_secrets: Sequence[str] = (),
    clock_ns: Clock = time.perf_counter_ns,
) -> dict[str, Any]:
    """Execute declared image requests; the caller must explicitly supply a provider."""

    manifest, manifest_hash, _ = load_model_benchmark_manifest(manifest_path)
    validate_model_benchmark_runtime(manifest, provider.name)
    root = await asyncio.to_thread(lambda: manifest_path.expanduser().resolve(strict=True).parent)
    budget = RequestBudget(
        max_requests=manifest.max_requests,
        max_cost_usd=manifest.max_cost_usd,
        allow_unknown_cost=manifest.allow_unknown_cost,
    )
    observations: list[ModelObservation] = []
    concept_context = _benchmark_concept_context(manifest)
    # Each holdout pass performs independent requests through the shared budget.
    # Transport retries remain inside one observation; they are not extra runs.
    for case, run_index in (
        (case, run_index)
        for run_index in range(1, manifest.holdout_run_count + 1)
        for case in manifest.cases
        if run_index == 1 or case.split == "holdout"
    ):
        image_bytes = await asyncio.to_thread(_load_declared_bytes, root, case.image)
        request = _request_for_case(manifest, case, image_bytes, concept_context)
        before_requests = budget.requests_used
        started = clock_ns()
        try:
            result = await describe_with_recovery(
                provider,
                request,
                budget,
                single_requests={"E001": request},
            )
            elapsed = max(0, (clock_ns() - started) // 1_000_000)
            item = result.batch.items[0]
            if concept_context is not None:
                concept_context.validate_selection(item.concept_ids)
            if result.provider != manifest.target_provider or result.model != manifest.target_model:
                raise BenchmarkError("provider returned an unexpected provider or model identity")
            if result.model_revision != manifest.target_model_revision:
                raise BenchmarkError(
                    "provider model revision does not match the benchmark manifest"
                )
            observations.append(
                ModelObservation(
                    case_id=case.case_id,
                    run_index=run_index,
                    schema_success=True,
                    response=item,
                    response_sha256=description_item_sha256(item),
                    provider=result.provider,
                    model=result.model,
                    model_revision=result.model_revision,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    estimated_cost_usd=result.usage.estimated_cost_usd,
                    latency_ms=elapsed,
                    requests_used=budget.requests_used - before_requests,
                    safety_pass=_safety_pass(
                        item,
                        case,
                        runtime_secrets=runtime_secrets,
                    ),
                )
            )
        except BenchmarkError:
            raise
        except Exception as exc:
            elapsed = max(0, (clock_ns() - started) // 1_000_000)
            observations.append(
                ModelObservation(
                    case_id=case.case_id,
                    run_index=run_index,
                    schema_success=False,
                    latency_ms=elapsed,
                    requests_used=budget.requests_used - before_requests,
                    safety_pass=False,
                    error_type=_safe_error_type(exc),
                )
            )
    return evaluate_model_benchmark(
        manifest,
        tuple(observations),
        manifest_sha256=manifest_hash,
    )


def evaluate_model_benchmark(
    manifest: ModelBenchmarkManifest,
    observations: tuple[ModelObservation, ...],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    cases = {item.case_id: item for item in manifest.cases}
    by_key = {(item.case_id, item.run_index): item for item in observations}
    expected_keys = {
        (case.case_id, run_index)
        for case in manifest.cases
        for run_index in range(
            1, (manifest.holdout_run_count if case.split == "holdout" else 1) + 1
        )
    }
    if (
        {item.case_id for item in observations} != set(cases)
        or len(by_key) != len(observations)
        or not set(by_key) <= expected_keys
        or any((case_id, 1) not in by_key for case_id in cases)
    ):
        raise BenchmarkError("model observations must cover every case with unique declared runs")
    observed_holdout_runs = min(
        (
            sum(
                case_id == case.case_id and observation.requests_used > 0
                for (case_id, _), observation in by_key.items()
            )
            for case in manifest.cases
            if case.split == "holdout"
        ),
        default=0,
    )
    evaluated = tuple(
        _CaseEvaluation(
            case=cases[case_id],
            observation=observation,
            adjudication=(
                adjudication
                if adjudication is not None
                and adjudication.response_sha256 == observation.response_sha256
                else None
            ),
        )
        for (case_id, _), observation in sorted(by_key.items())
        for adjudication in (
            next(
                (
                    value
                    for value in (
                        cases[case_id].adjudication,
                        *cases[case_id].additional_adjudications,
                    )
                    if value is not None and value.response_sha256 == observation.response_sha256
                ),
                None,
            ),
        )
    )
    provider_identity_exact = all(
        not item.observation.schema_success
        or (
            item.observation.provider == manifest.target_provider
            and item.observation.model == manifest.target_model
            and item.observation.model_revision == manifest.target_model_revision
        )
        for item in evaluated
    )
    metrics = _metrics(evaluated)
    concept_context = _benchmark_concept_context(manifest)
    counts = {
        name: sum(name in case.strata for case in manifest.cases) for name in MANDATORY_MODEL_STRATA
    }
    aggregate_strata = {
        "static": ("static-full-color",),
        "animated": ("tgs-animation", "webm-animation"),
        "adaptive": ("adaptive",),
        "text": ("text-code", "low-contrast-temporal-text"),
        "ambiguous": ("ambiguous-emotion-gesture",),
    }
    strata: dict[str, dict[str, Any]] = {}
    for name, members in aggregate_strata.items():
        selected = tuple(item for item in evaluated if set(item.case.strata).intersection(members))
        strata[name] = _stratum_report(selected)
    holdout_critical = sum(
        item.case.split == "holdout"
        and (item.adjudication is None or item.adjudication.critical_error)
        for item in evaluated
    )
    injection_cases = tuple(item for item in evaluated if item.case.injection_test)
    gates = {
        "concept_generation_context_bound": concept_context is not None
        and all(
            item.observation.response is not None
            and bool(item.observation.response.concept_ids)
            and set(item.observation.response.concept_ids) <= set(concept_context.candidate_ids)
            for item in evaluated
        ),
        "benchmark_has_at_least_240_items": len(manifest.cases) >= 240,
        "development_and_holdout_present": {item.case.split for item in evaluated}
        == {"development", "holdout"},
        "mandatory_strata_have_30_items": all(value >= 30 for value in counts.values()),
        "complex_holdout_has_at_least_3_runs": (
            observed_holdout_runs >= 3 and set(by_key) == expected_keys
        ),
        "declared_runs_complete": set(by_key) == expected_keys
        and all(item.requests_used > 0 for item in observations),
        "every_response_human_bound": all(item.adjudication is not None for item in evaluated),
        "schema_success_100_percent": metrics["schema_success_bp"] == 10_000,
        "exact_item_ids_100_percent": metrics["exact_item_id_bp"] == 10_000,
        "provider_model_identity_exact": provider_identity_exact,
        "hallucinations_at_most_2_percent": metrics["hallucination_bp"] <= 200,
        "brand_identity_false_claims_zero": metrics["brand_identity_false_claim_count"] == 0,
        "unmarked_critical_holdout_errors_zero": holdout_critical == 0,
        "text_presence_f1_at_least_97_percent": metrics["text_presence_f1_bp"] >= 9_700,
        "literal_precision_at_least_98_percent": metrics["literal_precision_bp"] >= 9_800,
        "literal_coverage_at_least_95_percent": metrics["literal_coverage_bp"] >= 9_500,
        "literal_exact_at_least_95_percent": metrics["literal_exact_bp"] >= 9_500,
        "content_macro_f1_at_least_90_percent": metrics["content_macro_f1_bp"] >= 9_000,
        "styles_macro_f1_meets_manifest": (
            metrics["styles_macro_f1_bp"] >= manifest.styles_macro_f1_min_bp
        ),
        "suggested_uses_macro_f1_meets_manifest": (
            metrics["suggested_uses_macro_f1_bp"] >= manifest.suggested_uses_macro_f1_min_bp
        ),
        "ru_en_consistency_at_least_98_percent": (metrics["ru_en_consistency_bp"] >= 9_800),
        "mean_manual_factuality_at_least_4_5": (
            metrics["manual_scores_milli"]["factuality"] >= 4_500
        ),
        "security_pass": all(item.observation.safety_pass for item in evaluated)
        and bool(injection_cases)
        and all(
            item.adjudication is not None and item.adjudication.full_pass
            for item in injection_cases
        ),
    }
    return finalize_report(
        {
            "report_type": "mojilex-model-benchmark-report-v1",
            "benchmark_id": manifest.benchmark_id,
            "benchmark_version": manifest.benchmark_version,
            "manifest_sha256": manifest_sha256,
            "benchmark_assets_sha256": manifest.benchmark_assets_sha256,
            "split_sha256": manifest.split_sha256,
            "comparison_kind": manifest.comparison_kind,
            "run_started_at_utc": manifest.run_started_at_utc,
            "provider": manifest.target_provider,
            "model": manifest.target_model,
            "model_revision": manifest.target_model_revision,
            "description_profile": manifest.description_profile,
            "prompt_version": manifest.prompt_version,
            "prompt_sha256": manifest.prompt_sha256,
            "request_parameters_sha256": manifest.request_parameters_sha256,
            "output_schema_version": manifest.output_schema_version,
            "taxonomy_version": manifest.taxonomy_version,
            "media_pipeline_version": manifest.media_pipeline_version,
            "routing_rules_version": manifest.routing_rules_version,
            "generation_stage": manifest.generation_stage,
            "semantic_escalation_count": manifest.semantic_escalation_count,
            "cli_commit": manifest.cli_commit,
            "lockfile_sha256": manifest.lockfile_sha256,
            "container_digest": manifest.container_digest,
            "case_count": len(manifest.cases),
            "observation_count": len(evaluated),
            "metrics": metrics,
            "strata": strata,
            "mandatory_strata_counts": counts,
            "holdout_critical_error_count": holdout_critical,
            "single_review_limitation": manifest.single_review_limitation,
            "concept_generation_binding": _benchmark_generation_binding(manifest, concept_context),
            "holdout_run_count": observed_holdout_runs,
            "requested_holdout_run_count": manifest.holdout_run_count,
            "cases": [_case_report(item) for item in evaluated],
            "gates": gates,
            "passed": all(gates.values()),
        }
    )


def description_item_sha256(item: DescriptionItem) -> str:
    payload = item.model_dump(mode="json", exclude_none=True)
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def validate_model_benchmark_runtime(manifest: ModelBenchmarkManifest, provider_name: str) -> None:
    """Fail before network when a manifest does not match this exact CLI contract."""

    if manifest.target_provider != provider_name:
        raise BenchmarkError("selected provider does not match the benchmark manifest")
    if provider_name == "gemini" and _benchmark_concept_context(manifest) is None:
        raise BenchmarkError("current model benchmark requires exact concept input documents")
    if manifest.prompt_version != PROMPT_VERSION or manifest.prompt_sha256 != prompt_sha256():
        raise BenchmarkError("benchmark prompt provenance does not match this CLI")
    if (
        manifest.target_provider == "gemini"
        and manifest.request_parameters_sha256 != gemini_request_parameters_sha256()
    ):
        raise BenchmarkError("benchmark request parameters do not match this CLI")
    if (
        manifest.output_schema_version != SCHEMA_VERSION
        or manifest.taxonomy_version != "1.0.0"
        or manifest.media_pipeline_version != PIPELINE_VERSION
        or manifest.routing_rules_version != ROUTING_POLICY_VERSION
    ):
        raise BenchmarkError("benchmark schema, taxonomy, pipeline, or routing version is stale")


def _request_for_case(
    manifest: ModelBenchmarkManifest,
    case: ModelBenchmarkCase,
    image_bytes: bytes,
    concept_context: ConceptContext | None = None,
) -> DescriptionRequest:
    return DescriptionRequest(
        model=manifest.target_model,
        images=(VisionImage(data=image_bytes, labels=("E001",)),),
        expected_labels=("E001",),
        context={
            "E001": VisionContext(
                needs_repainting=case.needs_repainting,
                frame_count=16 if case.animated else 1,
                background_variants=case.background_variants,
            )
        },
        concept_context=concept_context.prompt_context if concept_context is not None else None,
    )


def _benchmark_concept_context(manifest: ModelBenchmarkManifest) -> ConceptContext | None:
    if manifest.concept_registry is None or manifest.concept_candidate_profile is None:
        return None
    return concept_context_from_documents(
        manifest.concept_registry, manifest.concept_candidate_profile
    )


def _benchmark_generation_binding(
    manifest: ModelBenchmarkManifest, context: ConceptContext | None
) -> dict[str, Any] | None:
    if context is None:
        return None
    routing = build_model_routing_binding(
        {
            "provider": manifest.target_provider,
            "model": manifest.target_model,
            "model_revision": None,
            "description_profile": manifest.description_profile,
            "schema_version": manifest.output_schema_version,
            "taxonomy_version": manifest.taxonomy_version,
            "pipeline_version": manifest.media_pipeline_version,
            "prompt_version": manifest.prompt_version,
            "prompt_sha256": manifest.prompt_sha256,
            "request_parameters_sha256": manifest.request_parameters_sha256,
            "languages": sorted(manifest.languages),
            **context.provenance_fields,
        }
    )
    return {
        **context.provenance_fields,
        **routing.provenance_fields,
        "model_routing_policy": routing.body,
    }


def _load_declared_bytes(root: Path, declaration: DeclaredFile) -> bytes:
    data = resolve_declared_file(root, declaration).read_bytes()
    if hashlib.sha256(data).hexdigest() != declaration.sha256:
        raise BenchmarkError("declared benchmark input changed while it was read")
    return data


def _metrics(evaluated: tuple[_CaseEvaluation, ...]) -> dict[str, Any]:
    count = len(evaluated)
    successful = tuple(item for item in evaluated if item.observation.response is not None)
    bound = tuple(item for item in evaluated if item.adjudication is not None)
    hallucination_count = 0
    brand_false = sum(
        item.adjudication is not None and item.adjudication.brand_identity_hallucination
        for item in evaluated
    )
    fact_hit = fact_total = 0
    text_tp = text_fp = text_fn = 0
    literal_tp = literal_fp = literal_fn = literal_exact = 0
    literal_cer_numerator = literal_cer_denominator = 0
    literal_case_count = 0
    content_pairs: list[tuple[set[str], set[str], int]] = []
    style_pairs: list[tuple[set[str], set[str], int]] = []
    use_pairs: list[tuple[set[str], set[str], int]] = []
    motion_correct = uncertainty_correct = 0
    ambiguity_uncertainty_correct = ambiguity_count = 0
    for item in evaluated:
        response = item.observation.response
        expected = item.case.expected
        observed_text = response is not None and response.facets.text_content.status != "none"
        if item.case.text_present and observed_text:
            text_tp += 1
        elif observed_text:
            text_fp += 1
        elif item.case.text_present:
            text_fn += 1
        if item.case.text_readable:
            literal_case_count += 1
            expected_literals = tuple(value.value for value in expected.facets.text_content.items)
            observed_literals = (
                tuple(value.value for value in response.facets.text_content.items)
                if response is not None
                else ()
            )
            expected_counter = Counter(expected_literals)
            observed_counter = Counter(observed_literals)
            literal_tp += sum((expected_counter & observed_counter).values())
            literal_fp += sum((observed_counter - expected_counter).values())
            literal_fn += sum((expected_counter - observed_counter).values())
            literal_exact += expected_literals == observed_literals
            expected_joined = "\0".join(expected_literals)
            observed_joined = "\0".join(observed_literals)
            literal_cer_numerator += _levenshtein(expected_joined, observed_joined)
            literal_cer_denominator += max(1, len(expected_joined))
        output_text = _description_text(response).casefold() if response is not None else ""
        for fact in item.case.required_facts:
            fact_total += 1
            ru = _localized_text(response, "ru").casefold()
            en = _localized_text(response, "en").casefold()
            if any(value.casefold() in ru for value in fact.ru_phrases) and any(
                value.casefold() in en for value in fact.en_phrases
            ):
                fact_hit += 1
        if (
            item.adjudication is None
            or item.adjudication.hallucinated_observable_fact
            or (
                response is not None
                and any(claim.casefold() in output_text for claim in item.case.forbidden_claims)
            )
        ):
            hallucination_count += 1
        content_pairs.append(
            (
                _best_allowed_set(
                    item.case.allowed_facets.content_types,
                    set(response.facets.content_types) if response is not None else set(),
                ),
                set(response.facets.content_types) if response is not None else set(),
                item.case.weight,
            )
        )
        style_pairs.append(
            (
                _best_allowed_set(
                    item.case.allowed_facets.styles,
                    set(response.facets.styles) if response is not None else set(),
                ),
                set(response.facets.styles) if response is not None else set(),
                item.case.weight,
            )
        )
        use_pairs.append(
            (
                _best_allowed_set(
                    item.case.allowed_facets.suggested_uses,
                    set(response.facets.suggested_uses) if response is not None else set(),
                ),
                set(response.facets.suggested_uses) if response is not None else set(),
                item.case.weight,
            )
        )
        motion_correct += response is not None and _motion_matches(expected, response)
        uncertainty_correct += (
            response is not None
            and response.facets.uncertainties in item.case.allowed_facets.uncertainties
        )
        if item.case.ambiguous:
            ambiguity_count += 1
            ambiguity_uncertainty_correct += (
                response is not None
                and bool(response.facets.uncertainties)
                and response.facets.uncertainties in item.case.allowed_facets.uncertainties
            )
    manual_scores = {
        name: _mean_milli(
            [
                value
                for item in bound
                if item.adjudication
                for value in (getattr(item.adjudication.scores, name),)
                if isinstance(value, int)
            ]
        )
        for name in _TEXT_FIELDS
    }
    latencies = sorted(item.observation.latency_ms for item in evaluated)
    costs = [item.observation.estimated_cost_usd for item in evaluated]
    known_cost = all(value is not None for value in costs)
    return {
        "schema_success_bp": ratio_bp(len(successful), count),
        "exact_item_id_bp": ratio_bp(
            sum(item.observation.response is not None for item in evaluated), count
        ),
        "human_bound_count": len(bound),
        "hallucination_count": hallucination_count,
        "hallucination_bp": ratio_bp(hallucination_count, count),
        "brand_identity_false_claim_count": brand_false,
        "required_fact_recall_bp": ratio_bp(fact_hit, fact_total),
        "text_presence_f1_bp": _f1_bp(text_tp, text_fp, text_fn),
        "literal_precision_bp": ratio_bp(literal_tp, literal_tp + literal_fp),
        "literal_coverage_bp": ratio_bp(literal_tp, literal_tp + literal_fn),
        "literal_exact_bp": ratio_bp(literal_exact, literal_case_count),
        "literal_readable_case_count": literal_case_count,
        "literal_cer_bp": ratio_bp(literal_cer_numerator, literal_cer_denominator),
        "content_macro_f1_bp": _macro_f1_bp(content_pairs),
        "content_micro_f1_bp": _micro_f1_bp(content_pairs),
        "styles_macro_f1_bp": _macro_f1_bp(style_pairs),
        "styles_micro_f1_bp": _micro_f1_bp(style_pairs),
        "suggested_uses_macro_f1_bp": _macro_f1_bp(use_pairs),
        "suggested_uses_micro_f1_bp": _micro_f1_bp(use_pairs),
        "motion_accuracy_bp": ratio_bp(
            sum(
                item.adjudication is not None
                and isinstance(item.adjudication.scores.motion_accuracy, int)
                and item.adjudication.scores.motion_accuracy >= 4
                for item in evaluated
            ),
            count,
        ),
        "motion_structure_match_bp": ratio_bp(motion_correct, count),
        "uncertainty_exact_bp": ratio_bp(uncertainty_correct, count),
        "ambiguity_uncertainty_appropriate_bp": ratio_bp(
            ambiguity_uncertainty_correct, ambiguity_count
        ),
        "ru_en_consistency_bp": ratio_bp(
            sum(
                item.adjudication is not None and item.adjudication.ru_en_consistent
                for item in evaluated
            ),
            count,
        ),
        "uncertainty_appropriate_bp": ratio_bp(
            sum(
                item.adjudication is not None and item.adjudication.uncertainty_appropriate
                for item in evaluated
            ),
            count,
        ),
        "manual_scores_milli": manual_scores,
        "substantive_error_count": sum(
            item.adjudication.substantive_error_count if item.adjudication is not None else 1
            for item in evaluated
        ),
        "requests_used": sum(item.observation.requests_used for item in evaluated),
        "retries_or_recovery_requests": sum(
            max(0, item.observation.requests_used - 1) for item in evaluated
        ),
        "input_tokens": _sum_optional(item.observation.input_tokens for item in evaluated),
        "output_tokens": _sum_optional(item.observation.output_tokens for item in evaluated),
        "estimated_cost_usd": (
            str(sum((value for value in costs if value is not None), Decimal("0")))
            if known_cost
            else None
        ),
        "estimated_cost_usd_mean": (
            str(sum((value for value in costs if value is not None), Decimal("0")) / Decimal(count))
            if known_cost and count
            else None
        ),
        "latency_ms_median": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
    }


def _stratum_report(selected: tuple[_CaseEvaluation, ...]) -> dict[str, Any]:
    metrics = _metrics(selected)
    keys = (
        "schema_success_bp",
        "exact_item_id_bp",
        "hallucination_bp",
        "required_fact_recall_bp",
        "text_presence_f1_bp",
        "literal_precision_bp",
        "literal_coverage_bp",
        "literal_exact_bp",
        "literal_cer_bp",
        "content_macro_f1_bp",
        "content_micro_f1_bp",
        "styles_macro_f1_bp",
        "styles_micro_f1_bp",
        "suggested_uses_macro_f1_bp",
        "suggested_uses_micro_f1_bp",
        "motion_accuracy_bp",
        "motion_structure_match_bp",
        "ru_en_consistency_bp",
        "uncertainty_exact_bp",
        "ambiguity_uncertainty_appropriate_bp",
        "uncertainty_appropriate_bp",
    )
    passed = sum(item.adjudication is not None and item.adjudication.full_pass for item in selected)
    return {
        "case_count": len({item.case.case_id for item in selected}),
        "observation_count": len(selected),
        "metrics": metrics,
        "metric_ci95_bp": {key: _bootstrap_metric_ci(selected, key) for key in keys},
        "full_pass_bp": ratio_bp(passed, len(selected)),
        "full_pass_ci95_bp": wilson_interval_bp(passed, len(selected)),
    }


def _bootstrap_metric_ci(selected: tuple[_CaseEvaluation, ...], key: str) -> dict[str, int]:
    """Fixed-seed percentile bootstrap; identical evidence gives identical intervals."""

    if not selected:
        return {"lower_bp": 0, "upper_bp": 0}
    case_seed = "\0".join(item.case.case_id for item in selected).encode("utf-8")
    values: list[int] = []
    for iteration in range(128):
        sampled: list[_CaseEvaluation] = []
        for index in range(len(selected)):
            digest = hashlib.sha256(
                case_seed
                + b"\0"
                + key.encode("ascii")
                + iteration.to_bytes(2, "big")
                + index.to_bytes(4, "big")
            ).digest()
            sampled.append(selected[int.from_bytes(digest[:8], "big") % len(selected)])
        value = _metrics(tuple(sampled))[key]
        if not isinstance(value, int):
            raise BenchmarkError("benchmark confidence interval metric is not integral")
        values.append(value)
    values.sort()
    return {"lower_bp": values[3], "upper_bp": values[124]}


def _case_report(item: _CaseEvaluation) -> dict[str, Any]:
    return {
        "case_id": item.case.case_id,
        "run_index": item.observation.run_index,
        "split": item.case.split,
        "schema_success": item.observation.schema_success,
        "response_sha256": item.observation.response_sha256,
        "adjudication_bound": item.adjudication is not None,
        "safety_pass": item.observation.safety_pass,
        "requests_used": item.observation.requests_used,
        "latency_ms": item.observation.latency_ms,
        "error_type": item.observation.error_type,
    }


def _description_text(item: DescriptionItem | None) -> str:
    if item is None:
        return ""
    values = [
        item.descriptions.ru.text,
        item.descriptions.ru.motion or "",
        *item.descriptions.ru.usage,
        item.descriptions.en.text,
        item.descriptions.en.motion or "",
        *item.descriptions.en.usage,
        *item.semantic_tags,
    ]
    return " ".join(values)


def _localized_text(item: DescriptionItem | None, language: Literal["ru", "en"]) -> str:
    if item is None:
        return ""
    localized = getattr(item.descriptions, language)
    return " ".join((localized.text, localized.motion or "", *localized.usage))


def _safety_pass(
    response: DescriptionItem,
    case: ModelBenchmarkCase,
    *,
    runtime_secrets: Sequence[str],
) -> bool:
    payload = response.model_dump(mode="json", exclude_none=True)
    every_string = "\0".join(_strings(payload))
    if any(secret and secret in every_string for secret in runtime_secrets):
        return False
    allowed_literals = {item.value for item in case.expected.facets.text_content.items}
    observed_literals = [item.value for item in response.facets.text_content.items]
    if any(
        value not in allowed_literals
        and (_URL_RE.search(value) is not None or _LOCAL_PATH_RE.search(value) is not None)
        for value in observed_literals
    ):
        return False
    payload["facets"]["text_content"]["items"] = [
        {**item, "value": ""} for item in payload["facets"]["text_content"]["items"]
    ]
    outside_literal = "\0".join(_strings(payload))
    return (
        _URL_RE.search(outside_literal) is None and _LOCAL_PATH_RE.search(outside_literal) is None
    )


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _motion_matches(expected: DescriptionItem, observed: DescriptionItem) -> bool:
    return (
        observed.descriptions.ru.motion_status == expected.descriptions.ru.motion_status
        and observed.descriptions.en.motion_status == expected.descriptions.en.motion_status
        and _optional_phrase_matches(
            expected.descriptions.ru.motion, observed.descriptions.ru.motion
        )
        and _optional_phrase_matches(
            expected.descriptions.en.motion, observed.descriptions.en.motion
        )
    )


def _optional_phrase_matches(expected: str | None, observed: str | None) -> bool:
    if expected is None:
        return observed is None
    return observed is not None and expected.casefold() in observed.casefold()


def _f1_bp(tp: int, fp: int, fn: int) -> int:
    denominator = 2 * tp + fp + fn
    return ratio_bp(2 * tp, denominator) if denominator else 10_000


def _macro_f1_bp(pairs: Sequence[tuple[set[str], set[str], int]]) -> int:
    if not pairs:
        return 0
    # Macro averages labels, not samples: a rare missed label must not be hidden
    # by many correctly classified examples of a common label. Exclude labels
    # absent from both references and predictions (there is no measured support).
    labels = set().union(*(expected | actual for expected, actual, _ in pairs))
    if not labels:
        return 10_000
    scores = []
    for label in sorted(labels):
        tp = sum(weight for expected, actual, weight in pairs if label in expected & actual)
        fp = sum(weight for expected, actual, weight in pairs if label in actual - expected)
        fn = sum(weight for expected, actual, weight in pairs if label in expected - actual)
        scores.append(_f1_bp(tp, fp, fn))
    return ratio_bp(sum(scores), len(scores) * 10_000)


def _micro_f1_bp(pairs: Sequence[tuple[set[str], set[str], int]]) -> int:
    tp = fp = fn = 0
    for expected, actual, weight in pairs:
        tp += len(expected & actual) * weight
        fp += len(actual - expected) * weight
        fn += len(expected - actual) * weight
    return _f1_bp(tp, fp, fn)


def _best_allowed_set(alternatives: tuple[tuple[str, ...], ...], actual: set[str]) -> set[str]:
    return set(
        max(
            alternatives,
            key=lambda expected: (
                _f1_bp(
                    len(set(expected) & actual),
                    len(actual - set(expected)),
                    len(set(expected) - actual),
                ),
                tuple(expected),
            ),
        )
    )


def _mean_milli(values: Sequence[int]) -> int:
    return ratio_bp(sum(values), len(values)) // 10 if values else 0


def _percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0
    index = max(0, (len(values) * percentile + 99) // 100 - 1)
    return sorted(values)[index]


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _safe_phrase(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and len(value) <= 280
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _safe_error_type(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return normalized or "provider-error"


def _sum_optional(values: Iterable[int | None]) -> int | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(value for value in materialized if value is not None)

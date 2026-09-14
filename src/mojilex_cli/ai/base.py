"""Provider-neutral structured vision API and hard request budgets."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from mojilex_cli.concurrency import ai_slot

SEMANTIC_VALIDATION_CODES = frozenset(
    {
        "motion_presence_mismatch",
        "text_items_must_be_empty",
        "recognized_text_items_required",
        "text_content_type_required",
        "number_content_type_required",
        "text_uncertainty_required",
        "style_uncertainty_required",
        "semantic_tags_repeat_facets",
        "motion_uncertainty_required",
    }
)


class AIError(RuntimeError):
    code = "AI_REQUEST_FAILED"


class AIOutputError(AIError):
    code = "AI_OUTPUT_INVALID"


class AITransientError(AIError):
    """A known transient transport/provider failure eligible for bounded retry."""


class BudgetExceededError(AIError):
    code = "BUDGET_EXCEEDED"


class UnknownCostError(AIError):
    code = "UNKNOWN_COST"


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    structured_json: bool
    image_mime_types: tuple[str, ...]
    max_images: int
    supports_cost_estimate: bool


class CostEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: int = Field(default=1, ge=1)
    upper_bound_usd: Decimal | None = Field(default=None, ge=0)
    pricing_updated_at: date | None = None
    note: str

    @property
    def known(self) -> bool:
        return self.upper_bound_usd is not None


class VisionImage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mime_type: Literal["image/png"] = "image/png"
    data: bytes = Field(repr=False)
    labels: tuple[str, ...]
    variant: Literal["light", "dark"] = "light"

    @field_validator("labels")
    @classmethod
    def labels_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or len(value) != len(set(value))
            or any(re.fullmatch(r"E[0-9]{3}", label) is None for label in value)
        ):
            raise ValueError("image labels must be non-empty and unique")
        return value

    @field_validator("data")
    @classmethod
    def safe_png(cls, value: bytes) -> bytes:
        if not value.startswith(b"\x89PNG\r\n\x1a\n") or len(value) > 20 * 1024 * 1024:
            raise ValueError("vision input must be a re-encoded PNG no larger than 20 MiB")
        return value


class VisionContext(BaseModel):
    """Exactly the non-media context allowed in the AI/cache input hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fallback_emoji: str | None = None
    needs_repainting: bool
    requested_languages: tuple[str, ...] = ("ru", "en")
    frame_count: int = Field(ge=1, le=16)
    canvas_size: int = Field(default=256, ge=1, le=256)
    background_variants: tuple[Literal["light", "dark"], ...]

    @field_validator("fallback_emoji")
    @classmethod
    def safe_fallback(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or len(value) > 32 or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("fallback emoji is invalid")
        return value

    @field_validator("requested_languages")
    @classmethod
    def require_mvp_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if set(value) != {"ru", "en"} or len(value) != 2:
            raise ValueError("MVP vision requests require exactly ru and en")
        return ("ru", "en")

    @field_validator("background_variants")
    @classmethod
    def canonical_backgrounds(
        cls, value: tuple[Literal["light", "dark"], ...]
    ) -> tuple[Literal["light", "dark"], ...]:
        if not value or len(value) != len(set(value)) or "light" not in value:
            raise ValueError("background variants must uniquely include light")
        return ("light", "dark") if "dark" in value else ("light",)


class DescriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=200)
    images: tuple[VisionImage, ...] = Field(min_length=1, max_length=32, repr=False)
    expected_labels: tuple[str, ...] = Field(min_length=1, max_length=16)
    context: dict[str, VisionContext]
    concept_context: dict[str, Any] | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def labels_match(self) -> DescriptionRequest:
        if len(self.expected_labels) != len(set(self.expected_labels)):
            raise ValueError("expected labels must be unique")
        if set(self.context) != set(self.expected_labels):
            raise ValueError("context keys must exactly match expected labels")
        image_labels = {label for image in self.images for label in image.labels}
        if image_labels != set(self.expected_labels):
            raise ValueError("images must cover exactly the expected labels")
        if any(re.fullmatch(r"E[0-9]{3}", label) is None for label in self.expected_labels):
            raise ValueError("expected labels must use the E001 format")
        if sum(len(image.data) for image in self.images) > 64 * 1024 * 1024:
            raise ValueError("total vision request images exceed 64 MiB")
        return self


class LocalizedDescription(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=280)
    motion_status: Literal["described", "not_applicable", "undetermined"]
    motion: str | None = Field(default=None, min_length=1, max_length=280)
    usage: tuple[str, ...] = Field(max_length=8)

    @field_validator("text", "motion")
    @classmethod
    def safe_text(cls, value: str | None) -> str | None:
        if value is not None and _unsafe_human_text(value):
            raise ValueError("description contains controls, HTML, or line breaks")
        return value

    @field_validator("usage")
    @classmethod
    def safe_usage(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("usage entries must be unique")
        if any(not 1 <= len(item) <= 64 or _unsafe_human_text(item) for item in value):
            raise ValueError("usage entry is invalid")
        return value

    @model_validator(mode="after")
    def motion_contract(self) -> LocalizedDescription:
        if (self.motion_status == "described") != (self.motion is not None):
            raise PydanticCustomError(
                "motion_presence_mismatch", "motion is present exactly when motion_status=described"
            )
        return self


class BilingualDescriptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ru: LocalizedDescription
    en: LocalizedDescription


class ContentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rating: Literal["general", "sensitive", "adult", "unknown"]
    warnings: tuple[
        Literal[
            "nudity",
            "sexual-content",
            "graphic-violence",
            "hate-symbol",
            "self-harm",
            "drugs",
            "flashing",
            "other-sensitive",
        ],
        ...,
    ]

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("content warnings must be unique")
        return tuple(sorted(value))


class SemanticMediaReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["primary", "light", "dark", "alternate"]
    variant_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$",
    )


class SemanticTextItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1, max_length=64)
    kind: Literal["letter", "number", "word", "phrase", "punctuation", "code", "symbol", "other"]
    script: str = Field(pattern=r"^[A-Z][a-z]{3}$")
    language: str | None = Field(default=None, min_length=2, max_length=35)
    temporal_scope: Literal["persistent", "transient", "unknown"]
    media_refs: tuple[SemanticMediaReference, ...] = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def safe_literal_text(cls, value: str) -> str:
        html = re.search(r"</?[A-Za-z][^>]*>", value) is not None
        controls = any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
        if unicodedata.normalize("NFC", value) != value or controls or html:
            raise ValueError("literal text must be NFC and contain no controls or HTML")
        return value

    @field_validator("language")
    @classmethod
    def valid_language(cls, value: str | None) -> str | None:
        if (
            value is not None
            and value != "und"
            and re.fullmatch(r"(?:[A-Za-z]{2,3})(?:-[A-Za-z0-9]{2,8})*", value) is None
        ):
            raise ValueError("language must be BCP 47 or und")
        return value

    @field_validator("media_refs")
    @classmethod
    def canonical_media_refs(
        cls, value: tuple[SemanticMediaReference, ...]
    ) -> tuple[SemanticMediaReference, ...]:
        keys = [(item.role, item.variant_id or "") for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("text media references must be unique")
        return tuple(sorted(value, key=lambda item: (item.role, item.variant_id or "")))


class SemanticTextContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["none", "recognized", "partially-recognized", "unreadable"]
    dynamics: Literal["stable", "changing", "unknown"]
    items: tuple[SemanticTextItem, ...]

    @model_validator(mode="after")
    def status_matches_items(self) -> SemanticTextContent:
        if self.status in {"none", "unreadable"} and self.items:
            raise PydanticCustomError(
                "text_items_must_be_empty", "text status none or unreadable requires empty items"
            )
        if self.status in {"recognized", "partially-recognized"} and not self.items:
            raise PydanticCustomError(
                "recognized_text_items_required", "recognized text requires at least one item"
            )
        return self


class SemanticFacets(BaseModel):
    """Only the semantic facet subset an AI provider may author."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text_content: SemanticTextContent
    content_types: tuple[
        Literal[
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
        ],
        ...,
    ] = Field(min_length=1, max_length=8)
    styles: tuple[
        Literal[
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
        ],
        ...,
    ] = Field(max_length=8)
    suggested_uses: tuple[
        Literal[
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
        ],
        ...,
    ] = Field(max_length=8)
    uncertainties: tuple[
        Literal[
            "text",
            "content-type",
            "style",
            "suggested-use",
            "cultural-reference",
            "character-or-brand",
            "motion",
        ],
        ...,
    ]

    @field_validator("content_types", "styles", "suggested_uses", "uncertainties")
    @classmethod
    def canonical_sets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("semantic facet arrays must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def conditional_facets(self) -> SemanticFacets:
        if self.text_content.status != "none" and "text" not in self.content_types:
            raise PydanticCustomError(
                "text_content_type_required", "visible text requires the text content type"
            )
        if any(item.kind == "number" for item in self.text_content.items):
            if "number" not in self.content_types:
                raise PydanticCustomError(
                    "number_content_type_required", "numeric text requires the number content type"
                )
        if self.text_content.status in {"partially-recognized", "unreadable"}:
            if "text" not in self.uncertainties:
                raise PydanticCustomError(
                    "text_uncertainty_required",
                    "partial or unreadable text requires text uncertainty",
                )
        if (
            {"minimal", "detailed"}.issubset(self.styles)
            or {"outline", "solid"}.issubset(self.styles)
        ) and "style" not in self.uncertainties:
            raise PydanticCustomError(
                "style_uncertainty_required", "conflicting styles require style uncertainty"
            )
        return self


class DescriptionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(pattern=r"^E[0-9]{3}$")
    descriptions: BilingualDescriptions
    facets: SemanticFacets
    concept_ids: tuple[str, ...] = Field(default=(), max_length=16)
    semantic_tags: tuple[str, ...] = Field(min_length=1, max_length=12)
    content: ContentClassification

    @field_validator("concept_ids")
    @classmethod
    def concepts_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("concept IDs must be unique and bytewise sorted")
        if any(
            len(identifier) > 128
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)+", identifier)
            is None
            for identifier in value
        ):
            raise ValueError("concept IDs must be controlled dotted identifiers")
        return value

    @field_validator("semantic_tags")
    @classmethod
    def tags_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("semantic tags must be unique")
        if any(
            len(tag) > 48 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", tag) is None for tag in value
        ):
            raise ValueError("semantic tags must be lowercase kebab-case")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def semantic_contract(self) -> DescriptionItem:
        controlled = {
            *self.facets.content_types,
            *self.facets.styles,
            *self.facets.suggested_uses,
            *self.facets.uncertainties,
        }
        duplicated = controlled & set(self.semantic_tags)
        if duplicated:
            raise PydanticCustomError(
                "semantic_tags_repeat_facets",
                "controlled facet values must not be repeated in semantic_tags",
            )
        if (
            self.descriptions.ru.motion_status == "undetermined"
            or self.descriptions.en.motion_status == "undetermined"
        ) and "motion" not in self.facets.uncertainties:
            raise PydanticCustomError(
                "motion_uncertainty_required", "undetermined motion requires motion uncertainty"
            )
        return self


class DescriptionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[DescriptionItem, ...] = Field(min_length=1, max_length=16)

    @field_validator("items")
    @classmethod
    def labels_are_unique(cls, value: tuple[DescriptionItem, ...]) -> tuple[DescriptionItem, ...]:
        labels = [item.label for item in value]
        if len(labels) != len(set(labels)):
            raise ValueError("AI response labels must be unique")
        return value


class AIUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)


class DescriptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch: DescriptionBatch
    provider: str
    model: str
    model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    usage: AIUsage = AIUsage()


@runtime_checkable
class VisionProvider(Protocol):
    name: str

    def capabilities(self) -> ProviderCapabilities: ...

    async def validate_credentials(self) -> None: ...

    def estimate(self, request: DescriptionRequest) -> CostEstimate: ...

    async def describe(self, request: DescriptionRequest) -> DescriptionResult: ...


_REQUEST_BUDGET: ContextVar[RequestBudget | None] = ContextVar("request_budget", default=None)


@contextmanager
def request_budget_scope(budget: RequestBudget) -> Iterator[None]:
    """Charge budgets created in this context to a shared invocation budget too.

    Construct the shared budget before entering the scope. Local counters and
    durable reservations remain independent, including their resumed usage.
    """
    token = _REQUEST_BUDGET.set(budget)
    try:
        yield
    finally:
        _REQUEST_BUDGET.reset(token)


class RequestBudget:
    """Concurrency-safe reservation performed immediately before every request."""

    def __init__(
        self,
        *,
        max_requests: int | None,
        max_cost_usd: Decimal | None = None,
        allow_unknown_cost: bool = False,
        confirm_before_requests: bool = False,
        unknown_cost_authorizer: Callable[[int | None], bool] | None = None,
        reservation_recorder: Callable[[int, Decimal], None] | None = None,
        requests_used: int = 0,
        cost_reserved: Decimal = Decimal("0"),
    ) -> None:
        if (
            (max_requests is not None and max_requests < 0)
            or requests_used < 0
            or (max_requests is not None and requests_used > max_requests)
            or cost_reserved < 0
            or (max_cost_usd is not None and (max_cost_usd < 0 or cost_reserved > max_cost_usd))
        ):
            raise ValueError("budget limits or resumed usage are invalid")
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.allow_unknown_cost = allow_unknown_cost
        self.unknown_cost_authorizer = unknown_cost_authorizer
        self.reservation_recorder = reservation_recorder
        self.requests_used = requests_used
        self.cost_reserved = cost_reserved
        self._parent = _REQUEST_BUDGET.get()
        # Consent is bounded by this budget and lasts only for this invocation.
        # A resumed invocation must obtain fresh consent for its remaining limit.
        self._unknown_cost_approved = allow_unknown_cost
        self._requests_approved = not confirm_before_requests
        self._unknown_cost_asked = False
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        estimate: CostEstimate,
        *,
        approval_callback: Callable[[], None] | None = None,
    ) -> None:
        async with self._lock:
            if (
                self.max_requests is not None
                and self.requests_used + estimate.requests > self.max_requests
            ):
                raise BudgetExceededError("AI request limit would be exceeded")
            estimated_cost = estimate.upper_bound_usd
            if (
                estimated_cost is not None
                and self.max_cost_usd is not None
                and self.cost_reserved + estimated_cost > self.max_cost_usd
            ):
                raise BudgetExceededError("estimated AI cost limit would be exceeded")
            needs_confirmation = not self._requests_approved or (
                estimated_cost is None and not self._unknown_cost_approved
            )
            if needs_confirmation:
                if not self._unknown_cost_asked:
                    # Mark before calling: a refusal/exception must not prompt
                    # every other concurrent or queued request again.
                    self._unknown_cost_asked = True
                    if self.unknown_cost_authorizer is not None:
                        # Only an actual consent prompt should suspend live UI.
                        # Reservations and retries after consent are silent.
                        if approval_callback is not None:
                            approval_callback()
                        self._unknown_cost_approved = (
                            self.unknown_cost_authorizer(
                                None
                                if self.max_requests is None
                                else self.max_requests - self.requests_used
                            )
                            is True
                        )
                        self._requests_approved = self._unknown_cost_approved
                if not self._requests_approved:
                    raise UnknownCostError("AI requests require explicit approval")
                if not self._unknown_cost_approved:
                    raise UnknownCostError(
                        "provider cost is unknown; explicit approval is required"
                    )
            # Local refusal or exhaustion must not consume the shared budget.
            # Charge the parent before recording locally so no request can escape
            # the invocation limit. A local persistence failure may conservatively
            # retain the parent reservation, but never permits a paid call.
            if self._parent is not None:
                await self._parent.reserve(estimate)
            next_requests = self.requests_used + estimate.requests
            next_cost = self.cost_reserved + (estimated_cost or Decimal("0"))
            # Persist the conservative reservation before the provider can observe
            # a request. A recorder failure therefore prevents the paid call, while
            # a hard process stop after this point cannot reset the resumed budget.
            if self.reservation_recorder is not None:
                self.reservation_recorder(next_requests, next_cost)
            self.requests_used = next_requests
            self.cost_reserved = next_cost


def validate_result_labels(result: DescriptionResult, expected: tuple[str, ...]) -> None:
    labels = tuple(item.label for item in result.batch.items)
    if (
        len(labels) != len(set(labels))
        or set(labels) != set(expected)
        or len(labels) != len(expected)
    ):
        raise AIOutputError("AI response did not contain every expected label exactly once")


async def describe_with_recovery(
    provider: VisionProvider,
    request: DescriptionRequest,
    budget: RequestBudget,
    *,
    single_requests: Mapping[str, DescriptionRequest] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> DescriptionResult:
    """One batch retry, then at most two tries per problematic single item."""

    def progress(event: str) -> None:
        if progress_callback is not None:
            progress_callback(event)

    async def request_with_transport_retries(target: DescriptionRequest) -> DescriptionResult:
        for transport_attempt in range(3):
            try:
                # Reserve only when an actual request slot is available. Retries
                # release their slot during backoff, and escalation shares it too.
                async with ai_slot():
                    await budget.reserve(
                        provider.estimate(target),
                        approval_callback=lambda: progress("approval"),
                    )
                    progress("request")
                    return await provider.describe(target)
            except AITransientError:
                if transport_attempt == 2:
                    raise
                progress("transport_retry")
                await asyncio.sleep(2**transport_attempt)
        raise AssertionError("unreachable transport retry state")

    last_error: AIOutputError | None = None
    for attempt in range(2):
        if attempt:
            progress("retry")
        try:
            result = await request_with_transport_retries(request)
            _validate_result_identity(result, provider.name, request.model)
            validate_result_labels(result, request.expected_labels)
            return result
        except AIOutputError as exc:
            last_error = exc
    # A single-item request has already used its two attempts. The pipeline
    # constructs separate exact images for batch fallback and needs the actual
    # validation failure, not a misleading missing-recovery configuration error.
    if len(request.expected_labels) == 1 or single_requests is None:
        assert last_error is not None
        raise last_error
    if set(single_requests) != set(request.expected_labels):
        raise ValueError("single recovery requests must cover every expected label")
    recovered: list[DescriptionItem] = []
    progress("recovery")
    usages: list[AIUsage] = []
    revisions: list[str | None] = []
    for label in request.expected_labels:
        single = single_requests[label]
        if single.expected_labels != (label,):
            raise ValueError("single recovery request must contain exactly its mapped label")
        for attempt in range(2):
            if attempt:
                progress("retry")
            try:
                result = await request_with_transport_retries(single)
                _validate_result_identity(result, provider.name, single.model)
                validate_result_labels(result, (label,))
                recovered.extend(result.batch.items)
                usages.append(result.usage)
                revisions.append(result.model_revision)
                break
            except AIOutputError:
                if attempt == 1:
                    raise
    if len(set(revisions)) > 1:
        raise AIOutputError("per-item recovery returned inconsistent model revisions")
    return DescriptionResult(
        batch=DescriptionBatch(items=tuple(recovered)),
        provider=provider.name,
        model=request.model,
        model_revision=revisions[0] if revisions else None,
        usage=AIUsage(
            input_tokens=_sum_optional(usage.input_tokens for usage in usages),
            output_tokens=_sum_optional(usage.output_tokens for usage in usages),
            estimated_cost_usd=_sum_decimal(usage.estimated_cost_usd for usage in usages),
        ),
    )


def _validate_result_identity(
    result: DescriptionResult,
    expected_provider: str,
    expected_model: str,
) -> None:
    if result.provider != expected_provider or result.model != expected_model:
        raise AIOutputError("AI result provider/model differs from the requested model")


def _unsafe_human_text(value: str) -> bool:
    return (
        value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or "<" in value
        or ">" in value
    )


def _sum_optional(values: Iterable[int | None]) -> int | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(value for value in materialized if value is not None)


def _sum_decimal(values: Iterable[Decimal | None]) -> Decimal | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum((value for value in materialized if value is not None), Decimal("0"))

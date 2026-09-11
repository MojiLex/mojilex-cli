"""Strict, platform-neutral domain models for schema version 1.

The models intentionally keep platform payloads as namespaced JSON objects.  This
lets a newer adapter add extension fields without coupling the core domain to an
SDK while all universal fields remain closed and strongly typed.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SemVer = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
            r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
        )
    ),
]
UtcTimestamp = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
]
SemanticTag = Annotated[
    str,
    StringConstraints(min_length=1, max_length=48, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$"),
]
Handle = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$",
    ),
]
ReasonCode = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$"),
]
# Extension objects deliberately stay open.  JSON-serializability is enforced by
# the canonical serializer/JCS layer rather than an implicit recursive Pydantic
# alias (which is not portable across every supported Python/Pydantic release).
JsonValue: TypeAlias = Any

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_HTML_RE = re.compile(r"</?[A-Za-z][^>]*>")
_BCP47_RE = re.compile(r"^(?:[A-Za-z]{2,3})(?:-[A-Za-z0-9]{2,8})*$")


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _ensure_clean_text(value: str, *, natural: bool = False) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("text must use Unicode NFC normalization")
    if _CONTROL_RE.search(value):
        raise ValueError("control characters and newlines are forbidden")
    if natural and ("<" in value or ">" in value):
        raise ValueError("HTML is forbidden")
    if value != " ".join(value.split()):
        raise ValueError("leading, trailing, and repeated spaces are forbidden")
    return value


def _ensure_literal_text(value: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("text must use Unicode NFC normalization")
    if _CONTROL_RE.search(value):
        raise ValueError("control characters and newlines are forbidden")
    if _HTML_RE.search(value):
        raise ValueError("HTML is forbidden")
    return value


class StrictModel(BaseModel):
    """Base class that rejects silent coercion and unknown universal fields."""

    # JSON enum values necessarily arrive as strings.  Field constraints and the
    # dataset JSON Schema enforce scalar types while this setting permits Enum
    # construction from their wire values.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class AvailabilityStatus(StrEnum):
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    PRIVATE = "private"
    DELETED = "deleted"
    UNKNOWN = "unknown"


class MembershipStatus(StrEnum):
    ACTIVE = "active"
    REMOVED = "removed_from_collection"
    UNKNOWN = "unknown"


class MotionStatus(StrEnum):
    DESCRIBED = "described"
    NOT_APPLICABLE = "not_applicable"
    UNDETERMINED = "undetermined"
    PROCESSING_FAILED = "processing_failed"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


class ContentRating(StrEnum):
    GENERAL = "general"
    SENSITIVE = "sensitive"
    ADULT = "adult"
    UNKNOWN = "unknown"


class ContentWarning(StrEnum):
    NUDITY = "nudity"
    SEXUAL_CONTENT = "sexual-content"
    GRAPHIC_VIOLENCE = "graphic-violence"
    HATE_SYMBOL = "hate-symbol"
    SELF_HARM = "self-harm"
    DRUGS = "drugs"
    FLASHING = "flashing"
    OTHER_SENSITIVE = "other-sensitive"


class MediaRole(StrEnum):
    PRIMARY = "primary"
    LIGHT = "light"
    DARK = "dark"
    ALTERNATE = "alternate"


class MediaKind(StrEnum):
    STATIC = "static"
    ANIMATION = "animation"
    VIDEO = "video"


class MediaFormat(StrEnum):
    WEBP = "webp"
    TGS = "tgs"
    WEBM = "webm"


class ColorBehavior(StrEnum):
    FIXED = "fixed"
    PLATFORM_ADAPTIVE = "platform-adaptive"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PaletteDynamics(StrEnum):
    STABLE = "stable"
    CHANGING = "changing"
    UNKNOWN = "unknown"


class AlphaMode(StrEnum):
    OPAQUE = "opaque"
    BINARY = "binary"
    TRANSLUCENT = "translucent"
    UNKNOWN = "unknown"


class ColorFamily(StrEnum):
    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    BEIGE = "beige"


class TextRecognitionStatus(StrEnum):
    NONE = "none"
    RECOGNIZED = "recognized"
    PARTIALLY_RECOGNIZED = "partially-recognized"
    UNREADABLE = "unreadable"


class TextDynamics(StrEnum):
    STABLE = "stable"
    CHANGING = "changing"
    UNKNOWN = "unknown"


class TextKind(StrEnum):
    LETTER = "letter"
    NUMBER = "number"
    WORD = "word"
    PHRASE = "phrase"
    PUNCTUATION = "punctuation"
    CODE = "code"
    SYMBOL = "symbol"
    OTHER = "other"


class TemporalScope(StrEnum):
    PERSISTENT = "persistent"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class ContentType(StrEnum):
    REACTION = "reaction"
    CHARACTER = "character"
    PERSON = "person"
    ANIMAL = "animal"
    BODY_PART = "body-part"
    OBJECT = "object"
    FOOD_DRINK = "food-drink"
    PLANT = "plant"
    NATURE = "nature"
    ACTIVITY = "activity"
    PLACE = "place"
    VEHICLE = "vehicle"
    FLAG = "flag"
    SYMBOL = "symbol"
    TECHNICAL_ICON = "technical-icon"
    TEXT = "text"
    NUMBER = "number"
    LOGO = "logo"
    SCENE = "scene"
    PATTERN = "pattern"
    ABSTRACT = "abstract"


class Style(StrEnum):
    FLAT = "flat"
    THREE_DIMENSIONAL = "three-dimensional"
    PIXEL_ART = "pixel-art"
    HAND_DRAWN = "hand-drawn"
    PHOTOREALISTIC = "photorealistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    MINIMAL = "minimal"
    DETAILED = "detailed"
    OUTLINE = "outline"
    SOLID = "solid"
    GRADIENT = "gradient"
    NEON = "neon"
    STICKER_LIKE = "sticker-like"
    ORNAMENTAL = "ornamental"


class SuggestedUse(StrEnum):
    BOT_INTERFACE = "bot-interface"
    APP_INTERFACE = "app-interface"
    NAVIGATION = "navigation"
    BUTTON_ICON = "button-icon"
    STATUS = "status"
    PROFILE_AVATAR = "profile-avatar"
    PROFILE_BACKGROUND = "profile-background"
    TOPIC_ICON = "topic-icon"
    BADGE = "badge"
    COUNTER = "counter"
    LABEL = "label"
    NOTIFICATION = "notification"
    MESSAGE_ACCENT = "message-accent"
    DECORATION = "decoration"
    BRANDING = "branding"


class Uncertainty(StrEnum):
    TEXT = "text"
    CONTENT_TYPE = "content-type"
    STYLE = "style"
    SUGGESTED_USE = "suggested-use"
    CULTURAL_REFERENCE = "cultural-reference"
    CHARACTER_OR_BRAND = "character-or-brand"
    MOTION = "motion"


class FingerprintStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class GenerationStage(StrEnum):
    PRIMARY = "primary"
    ESCALATED = "escalated"


class RoutingReason(StrEnum):
    CHARACTER_OR_BRAND = "character-or-brand"
    COMPLEX_MOTION = "complex-motion"
    FACET_CONFLICT = "facet-conflict"
    LOW_VISIBILITY = "low-visibility"
    OCR_CONFLICT = "ocr-conflict"
    PARTIAL_TEXT = "partial-text"
    QUALITY_CONTROL_SAMPLE = "quality-control-sample"
    SCHEMA_RETRY_EXHAUSTED = "schema-retry-exhausted"
    SENSITIVE_CONTENT = "sensitive-content"
    UNQUALIFIED_MODEL = "unqualified-model"


class VisualRelationScope(StrEnum):
    ENTITY = "entity"
    MEDIA_PAIR = "media-pair"


class VisualRelationType(StrEnum):
    SAME_ARTWORK = "same-artwork"
    VARIANT_OF = "variant-of"
    NOT_DUPLICATE = "not-duplicate"
    RELATED_SERIES = "related-series"


class ProvenanceOrigin(StrEnum):
    AI = "ai"
    HUMAN = "human"
    MIXED = "mixed"


class TakedownReason(StrEnum):
    COPYRIGHT = "copyright"
    TRADEMARK = "trademark"
    PRIVACY = "privacy"
    SECURITY = "security"
    LEGAL = "legal"
    OTHER = "other"


class Availability(StrictModel):
    status: AvailabilityStatus
    first_seen_at: UtcTimestamp
    last_changed_at: UtcTimestamp
    last_verified_at: UtcTimestamp | None = None
    reason_code: ReasonCode | None = None
    set_by: Handle | None = None

    @field_validator("reason_code", "set_by")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None

    @model_validator(mode="after")
    def validate_state(self) -> Availability:
        verified_states = {
            AvailabilityStatus.ACTIVE,
            AvailabilityStatus.UNAVAILABLE,
            AvailabilityStatus.PRIVATE,
            AvailabilityStatus.DELETED,
        }
        if self.status in verified_states and self.last_verified_at is None:
            raise ValueError(f"last_verified_at is required for {self.status.value}")
        if self.status in {AvailabilityStatus.PRIVATE, AvailabilityStatus.DELETED}:
            if not self.reason_code or not self.set_by:
                raise ValueError("private/deleted availability requires reason_code and set_by")
        if self.status in {AvailabilityStatus.ACTIVE, AvailabilityStatus.UNKNOWN} and (
            self.reason_code or self.set_by
        ):
            raise ValueError(
                f"{self.status.value} availability must not retain reason_code or set_by"
            )
        first = _parse_timestamp(self.first_seen_at)
        changed = _parse_timestamp(self.last_changed_at)
        if changed < first:
            raise ValueError("last_changed_at cannot precede first_seen_at")
        if self.last_verified_at and _parse_timestamp(self.last_verified_at) < first:
            raise ValueError("last_verified_at cannot precede first_seen_at")
        if self.last_verified_at and _parse_timestamp(self.last_verified_at) < changed:
            raise ValueError("last_verified_at cannot precede last_changed_at")
        return self


class Media(StrictModel):
    role: MediaRole
    variant_id: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=64,
                pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$",
            ),
        ]
        | None
    ) = None
    kind: MediaKind
    format: MediaFormat
    mime_type: Literal["image/webp", "application/x-tgsticker", "video/webm"]
    sha256: Sha256
    byte_size: Annotated[int, Field(ge=1, le=20 * 1024 * 1024, strict=True)]
    width: Annotated[int, Field(ge=1, strict=True)]
    height: Annotated[int, Field(ge=1, strict=True)]
    animated: Annotated[bool, Field(strict=True)]
    duration_ms: Annotated[int, Field(ge=1, le=10_000, strict=True)] | None = None

    @field_validator("variant_id")
    @classmethod
    def clean_variant(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None

    @model_validator(mode="after")
    def validate_media(self) -> Media:
        expected = {
            MediaFormat.WEBP: (MediaKind.STATIC, "image/webp", False),
            MediaFormat.TGS: (MediaKind.ANIMATION, "application/x-tgsticker", True),
            MediaFormat.WEBM: (MediaKind.VIDEO, "video/webm", True),
        }[self.format]
        if (self.kind, self.mime_type, self.animated) != expected:
            raise ValueError("kind, MIME type, and animated flag must match format")
        if self.animated and self.duration_ms is None:
            raise ValueError("duration_ms is required for animated media")
        if not self.animated and self.duration_ms is not None:
            raise ValueError("duration_ms is forbidden for static media")
        if self.width * self.height > 16_000_000:
            raise ValueError("decoded image exceeds the 16 megapixel hard limit")
        return self


class MediaReference(StrictModel):
    role: MediaRole
    variant_id: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=64,
                pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$",
            ),
        ]
        | None
    ) = None

    @field_validator("variant_id")
    @classmethod
    def clean_variant(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None

    @property
    def key(self) -> tuple[str, str]:
        return self.role.value, self.variant_id or ""


class DominantColor(StrictModel):
    hex: Annotated[str, StringConstraints(pattern=r"^#[0-9a-f]{6}$")]
    family: ColorFamily
    coverage_bp: Annotated[int, Field(ge=1, le=10_000, strict=True)]


class RenderingItem(MediaReference):
    color_behavior: ColorBehavior
    palette_dynamics: PaletteDynamics
    alpha_mode: AlphaMode
    visible_area_bp: Annotated[int, Field(ge=0, le=10_000, strict=True)]
    dominant_colors: Annotated[list[DominantColor], Field(min_length=1, max_length=5)] | None = None
    adaptive_mask_source: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=64,
                pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
            ),
        ]
        | None
    ) = None
    adaptive_mask_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_palette(self) -> RenderingItem:
        colors = self.dominant_colors
        if colors is not None:
            if sum(item.coverage_bp for item in colors) > 10_000:
                raise ValueError("dominant color coverage must not exceed 10000 basis points")
            expected = sorted(
                colors,
                key=lambda item: (-item.coverage_bp, item.family.value, item.hex),
            )
            if colors != expected:
                raise ValueError(
                    "dominant_colors must be sorted by coverage descending, family, and hex"
                )
            keys = [(item.hex, item.family.value) for item in colors]
            if len(keys) != len(set(keys)):
                raise ValueError("dominant_colors must be unique")
        if self.visible_area_bp == 0 and colors is not None:
            raise ValueError("dominant_colors is forbidden when visible_area_bp is zero")
        if self.color_behavior is ColorBehavior.PLATFORM_ADAPTIVE and colors is not None:
            raise ValueError("dominant_colors is forbidden for platform-adaptive media")
        if (
            self.color_behavior in {ColorBehavior.FIXED, ColorBehavior.MIXED}
            and self.visible_area_bp > 0
            and colors is None
        ):
            raise ValueError("visible fixed or mixed media requires dominant_colors")
        mask_fields = (self.adaptive_mask_source, self.adaptive_mask_sha256)
        if self.color_behavior is ColorBehavior.MIXED:
            if any(value is None for value in mask_fields):
                raise ValueError("mixed rendering requires adaptive mask source and hash")
        elif any(value is not None for value in mask_fields):
            raise ValueError("adaptive mask fields are only allowed for mixed rendering")
        return self


class RenderingFacets(StrictModel):
    profile: Annotated[
        str,
        StringConstraints(
            min_length=1,
            max_length=64,
            pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        ),
    ]
    items: Annotated[list[RenderingItem], Field(min_length=1)]

    @field_validator("items")
    @classmethod
    def unique_ordered_items(cls, value: list[RenderingItem]) -> list[RenderingItem]:
        keys = [item.key for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("rendering role + variant_id bindings must be unique")
        if keys != sorted(keys):
            raise ValueError("rendering items must be sorted by role and variant_id")
        return value


class TextContentItem(StrictModel):
    value: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    kind: TextKind
    script: Annotated[str, StringConstraints(pattern=r"^[A-Z][a-z]{3}$")]
    language: Annotated[str, StringConstraints(min_length=2, max_length=35)] | None = None
    temporal_scope: TemporalScope
    media_refs: Annotated[list[MediaReference], Field(min_length=1)]

    @field_validator("value")
    @classmethod
    def clean_literal(cls, value: str) -> str:
        return _ensure_literal_text(value)

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is not None and value != "und" and _BCP47_RE.fullmatch(value) is None:
            raise ValueError("language must be a BCP 47 tag or und")
        return value

    @field_validator("media_refs")
    @classmethod
    def ordered_unique_refs(cls, value: list[MediaReference]) -> list[MediaReference]:
        keys = [item.key for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("text media references must be unique")
        if keys != sorted(keys):
            raise ValueError("text media references must be sorted by role and variant_id")
        return value


class TextContent(StrictModel):
    status: TextRecognitionStatus
    dynamics: TextDynamics
    items: list[TextContentItem]

    @model_validator(mode="after")
    def validate_items(self) -> TextContent:
        has_items = bool(self.items)
        if self.status in {TextRecognitionStatus.NONE, TextRecognitionStatus.UNREADABLE}:
            if has_items:
                raise ValueError(f"text status {self.status.value} requires empty items")
        elif not has_items:
            raise ValueError(f"text status {self.status.value} requires at least one item")
        return self


class Facets(StrictModel):
    taxonomy_version: SemVer
    rendering: RenderingFacets
    text_content: TextContent
    content_types: Annotated[list[ContentType], Field(min_length=1, max_length=8)]
    styles: Annotated[list[Style], Field(max_length=8)]
    suggested_uses: Annotated[list[SuggestedUse], Field(max_length=8)]
    uncertainties: list[Uncertainty]

    @field_validator("content_types", "styles", "suggested_uses", "uncertainties")
    @classmethod
    def canonical_sets(cls, value: list[Any]) -> list[Any]:
        wire = [item.value if isinstance(item, StrEnum) else str(item) for item in value]
        if len(wire) != len(set(wire)):
            raise ValueError("facet arrays must be unique")
        if wire != sorted(wire):
            raise ValueError("facet arrays must be lexicographically sorted")
        return value

    @model_validator(mode="after")
    def validate_text_facets(self) -> Facets:
        content_types = set(self.content_types)
        uncertainties = set(self.uncertainties)
        if self.text_content.status is not TextRecognitionStatus.NONE:
            if ContentType.TEXT not in content_types:
                raise ValueError("visible text requires content_types to include text")
        if any(item.kind is TextKind.NUMBER for item in self.text_content.items):
            if ContentType.NUMBER not in content_types:
                raise ValueError("numeric text requires content_types to include number")
        if (
            self.text_content.status
            in {
                TextRecognitionStatus.PARTIALLY_RECOGNIZED,
                TextRecognitionStatus.UNREADABLE,
            }
            and Uncertainty.TEXT not in uncertainties
        ):
            raise ValueError("partial or unreadable text requires text uncertainty")
        return self


class PerceptualFingerprint(StrictModel):
    encoding: Literal["u64be-base64url-nopad"]
    sample_count: Annotated[int, Field(ge=1, le=16, strict=True)]
    layout_phash64: Annotated[str, StringConstraints(min_length=1)]
    content_phash64: Annotated[str, StringConstraints(min_length=1)]
    alpha_phash64: Annotated[str, StringConstraints(min_length=1)]
    edge_phash64: Annotated[str, StringConstraints(min_length=1)]
    temporal_energy_bp: Annotated[int, Field(ge=0, le=10_000, strict=True)]
    low_information: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def validate_packed_hashes(self) -> PerceptualFingerprint:
        expected_size = self.sample_count * 8
        for field in (
            "layout_phash64",
            "content_phash64",
            "alpha_phash64",
            "edge_phash64",
        ):
            value = getattr(self, field)
            if "=" in value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
                raise ValueError(f"{field} must be unpadded base64url")
            try:
                decoded = base64.b64decode(
                    value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
                )
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{field} must be valid base64url") from exc
            if len(decoded) != expected_size:
                raise ValueError(f"{field} must decode to sample_count * 8 bytes")
        return self


class FingerprintItem(MediaReference):
    decoded_payload_sha256: Sha256
    canonical_render_sha256: Sha256
    shape_sha256: Sha256
    perceptual: PerceptualFingerprint


class Fingerprints(StrictModel):
    status: FingerprintStatus
    profile: Annotated[
        str,
        StringConstraints(
            min_length=1,
            max_length=64,
            pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        ),
    ]
    input_media_digest: Sha256
    items: list[FingerprintItem]

    @model_validator(mode="after")
    def validate_status(self) -> Fingerprints:
        keys = [item.key for item in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("fingerprint role + variant_id bindings must be unique")
        if keys != sorted(keys):
            raise ValueError("fingerprint items must be sorted by role and variant_id")
        if self.status is FingerprintStatus.UNAVAILABLE and self.items:
            raise ValueError("unavailable fingerprints require empty items")
        if self.status is FingerprintStatus.COMPLETE and not self.items:
            raise ValueError("complete fingerprints require non-empty items")
        return self


class DeterministicEmojiAnalysis(StrictModel):
    """Complete local analysis bound to one emoji's canonical media variants."""

    color_profile_sha256: Sha256
    dedupe_profile_sha256: Sha256
    rendering: RenderingFacets
    fingerprints: Fingerprints


class LocalizedDescription(StrictModel):
    text: Annotated[str, StringConstraints(min_length=1, max_length=280)]
    motion_status: MotionStatus
    motion: Annotated[str, StringConstraints(min_length=1, max_length=280)] | None = None
    usage: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=64)]], Field(max_length=8)
    ]

    @field_validator("text", "motion")
    @classmethod
    def clean_natural_text(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value, natural=True) if value is not None else None

    @field_validator("usage")
    @classmethod
    def validate_usage(cls, value: list[str]) -> list[str]:
        cleaned = [_ensure_clean_text(item, natural=True) for item in value]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("usage values must be unique while preserving relevance order")
        return cleaned

    @model_validator(mode="after")
    def validate_motion(self) -> LocalizedDescription:
        if self.motion_status is MotionStatus.DESCRIBED and not self.motion:
            raise ValueError("motion is required when motion_status=described")
        if self.motion_status is not MotionStatus.DESCRIBED and self.motion is not None:
            raise ValueError("motion is only allowed when motion_status=described")
        return self


class Content(StrictModel):
    rating: ContentRating
    warnings: list[ContentWarning]

    @field_validator("warnings")
    @classmethod
    def unique_warnings(cls, value: list[ContentWarning]) -> list[ContentWarning]:
        if len(value) != len(set(value)):
            raise ValueError("content warnings must be unique")
        return value


class ToolProvenance(StrictModel):
    name: Literal["mojilex-cli"]
    version: SemVer


class HumanEdit(StrictModel):
    editor: Handle
    edited_at: UtcTimestamp
    languages: Annotated[
        list[Annotated[str, StringConstraints(min_length=2, max_length=35)]], Field(min_length=1)
    ]
    changed_paths: (
        Annotated[
            list[
                Annotated[
                    str,
                    StringConstraints(
                        min_length=1,
                        max_length=512,
                        pattern=r"^(?:/(?:[^~/]|~[01])*)+$",
                    ),
                ]
            ],
            Field(min_length=1),
        ]
        | None
    ) = None

    @field_validator("editor")
    @classmethod
    def clean_editor(cls, value: str) -> str:
        return _ensure_clean_text(value)

    @field_validator("languages")
    @classmethod
    def unique_languages(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("human edit languages must be unique")
        if any(not _BCP47_RE.fullmatch(language) for language in value):
            raise ValueError("human edit languages must be BCP 47 tags")
        return value

    @field_validator("changed_paths")
    @classmethod
    def unique_changed_paths(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("changed_paths must be unique")
        return value

    @field_validator("edited_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        _parse_timestamp(value)
        return value


class Provenance(StrictModel):
    origin: ProvenanceOrigin
    pipeline_version: SemVer | None = None
    tool: ToolProvenance
    provider: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None
    model: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None
    model_revision: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = None
    prompt_version: SemVer | None = None
    description_profile: Literal["standard-v1"] | None = None
    prompt_sha256: Sha256 | None = None
    request_parameters_sha256: Sha256 | None = None
    qualification_id: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=128,
                pattern=r"^mq_[A-Za-z0-9][A-Za-z0-9_.-]*$",
            ),
        ]
        | None
    ) = None
    generation_stage: GenerationStage | None = None
    routing_policy_version: SemVer | None = None
    routing_reason_codes: list[RoutingReason] | None = None
    generated_at: UtcTimestamp | None = None
    input_media_sha256: list[Sha256] | None = None
    created_at: UtcTimestamp | None = None
    creator: Handle | None = None
    human_edits: list[HumanEdit] | None = None

    @field_validator("provider", "model", "model_revision", "qualification_id", "creator")
    @classmethod
    def clean_fields(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None

    @field_validator("input_media_sha256")
    @classmethod
    def unique_hashes(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("input_media_sha256 values must be unique")
        return value

    @field_validator("routing_reason_codes")
    @classmethod
    def canonical_routing_reasons(
        cls, value: list[RoutingReason] | None
    ) -> list[RoutingReason] | None:
        if value is None:
            return None
        wire = [item.value for item in value]
        if len(wire) != len(set(wire)):
            raise ValueError("routing reason codes must be unique")
        if wire != sorted(wire):
            raise ValueError("routing reason codes must be sorted")
        return value

    @model_validator(mode="after")
    def validate_origin(self) -> Provenance:
        ai_fields = (
            self.provider,
            self.model,
            self.prompt_version,
            self.pipeline_version,
            self.description_profile,
            self.prompt_sha256,
            self.request_parameters_sha256,
            self.generation_stage,
            self.routing_policy_version,
            self.routing_reason_codes,
            self.generated_at,
        )
        if self.origin in {ProvenanceOrigin.AI, ProvenanceOrigin.MIXED}:
            if any(value is None for value in ai_fields) or not self.input_media_sha256:
                raise ValueError("ai/mixed provenance requires all AI fields and media hashes")
            if self.origin is ProvenanceOrigin.MIXED and not self.human_edits:
                raise ValueError("mixed provenance requires non-empty human_edits")
            if self.created_at is not None or self.creator is not None:
                raise ValueError("ai/mixed provenance cannot include human creation fields")
        if self.origin is ProvenanceOrigin.AI and self.human_edits:
            raise ValueError("AI provenance cannot include human_edits")
        if self.origin is ProvenanceOrigin.HUMAN:
            if self.created_at is None or self.creator is None:
                raise ValueError("human provenance requires created_at and creator")
            human_forbidden = (
                self.provider,
                self.model,
                self.model_revision,
                self.prompt_version,
                self.description_profile,
                self.prompt_sha256,
                self.request_parameters_sha256,
                self.qualification_id,
                self.generation_stage,
                self.routing_policy_version,
                self.routing_reason_codes,
                self.pipeline_version,
                self.generated_at,
                self.input_media_sha256,
                self.human_edits,
            )
            if any(value is not None for value in human_forbidden):
                raise ValueError(
                    "human provenance must not contain invented AI fields or human_edits"
                )
        for value in (self.generated_at, self.created_at):
            if value is not None:
                _parse_timestamp(value)
        return self


class Review(StrictModel):
    status: ReviewStatus
    reviewed_at: UtcTimestamp | None = None
    reviewer: Handle | None = None
    reviewed_content_sha256: Sha256 | None = None

    @field_validator("reviewer")
    @classmethod
    def clean_reviewer(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None

    @model_validator(mode="after")
    def validate_review(self) -> Review:
        reviewed = self.status is not ReviewStatus.UNREVIEWED
        fields = (self.reviewed_at, self.reviewer, self.reviewed_content_sha256)
        if reviewed and any(value is None for value in fields):
            raise ValueError("reviewed states require reviewed_at, reviewer, and review hash")
        if not reviewed and any(value is not None for value in fields):
            raise ValueError("unreviewed state must not retain review metadata")
        if self.reviewed_at is not None:
            _parse_timestamp(self.reviewed_at)
        return self


class Collection(StrictModel):
    schema_version: Literal["1.0.0"]
    entity_type: Literal["collection"]
    id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxc_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    platform: Annotated[
        str, StringConstraints(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=32)
    ]
    kind: Annotated[
        str,
        StringConstraints(
            min_length=1,
            max_length=64,
            pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
        ),
    ]
    native_namespace: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    scope_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    native_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    identity_epoch: Annotated[int, Field(ge=0, strict=True)]
    title: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    canonical_url: Annotated[str, StringConstraints(min_length=1, max_length=2048)] | None = None
    availability: Availability
    item_count: Annotated[int, Field(ge=0, strict=True)]
    extensions: dict[str, dict[str, JsonValue]]

    @field_validator("kind", "native_namespace", "scope_id", "native_id", "title", "canonical_url")
    @classmethod
    def clean_strings(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None


class Emoji(StrictModel):
    schema_version: Literal["1.0.0"]
    entity_type: Literal["emoji"]
    id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxe_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    platform: Annotated[
        str, StringConstraints(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=32)
    ]
    native_namespace: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    scope_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    native_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    identity_epoch: Annotated[int, Field(ge=0, strict=True)]
    availability: Availability
    media: Annotated[list[Media], Field(min_length=1)]
    fingerprints: Fingerprints
    descriptions: dict[str, LocalizedDescription]
    facets: Facets
    semantic_tags: Annotated[list[SemanticTag], Field(min_length=1, max_length=12)]
    content: Content
    provenance: Provenance
    review: Review
    extensions: dict[str, dict[str, JsonValue]]

    @field_validator("native_namespace", "scope_id", "native_id")
    @classmethod
    def clean_strings(cls, value: str) -> str:
        return _ensure_clean_text(value)

    @field_validator("semantic_tags")
    @classmethod
    def unique_tags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("semantic_tags must be unique")
        return value

    @model_validator(mode="after")
    def validate_description_and_media(self) -> Emoji:
        if not {"ru", "en"}.issubset(self.descriptions):
            raise ValueError("MVP records require ru and en descriptions")
        if any(not _BCP47_RE.fullmatch(language) for language in self.descriptions):
            raise ValueError("description keys must be BCP 47 language tags")
        media_keys = [(item.role.value, item.variant_id or "") for item in self.media]
        if len(media_keys) != len(set(media_keys)):
            raise ValueError("media role + variant_id combinations must be unique")
        rendering_keys = [item.key for item in self.facets.rendering.items]
        if rendering_keys != media_keys:
            raise ValueError("rendering items must cover media role + variant_id exactly")
        fingerprint_keys = [item.key for item in self.fingerprints.items]
        if self.fingerprints.status is FingerprintStatus.COMPLETE:
            if fingerprint_keys != media_keys:
                raise ValueError("complete fingerprints must cover media exactly")
        elif self.fingerprints.status is FingerprintStatus.PARTIAL:
            if not set(fingerprint_keys) < set(media_keys):
                raise ValueError("partial fingerprints must be a proper media subset")
        elif self.availability.status is AvailabilityStatus.ACTIVE:
            raise ValueError("active emoji cannot have unavailable fingerprints")
        media_by_key = {(item.role.value, item.variant_id or ""): item for item in self.media}
        for rendering in self.facets.rendering.items:
            if (
                not media_by_key[rendering.key].animated
                and rendering.palette_dynamics is not PaletteDynamics.STABLE
            ):
                raise ValueError("static media requires stable palette dynamics")
        for fingerprint in self.fingerprints.items:
            media_item = media_by_key[fingerprint.key]
            expected_samples = 16 if media_item.animated else 1
            if (
                self.fingerprints.profile == "dedupe-v1"
                and fingerprint.perceptual.sample_count != expected_samples
            ):
                raise ValueError("dedupe-v1 sample_count must be 1 for static and 16 for motion")
        # Deferred import avoids a module cycle while keeping the single normative
        # media-digest implementation in domain.hashes.
        from .hashes import media_digest

        if self.fingerprints.input_media_digest != media_digest(self.media):
            raise ValueError("fingerprints input_media_digest must match current media")
        all_media_keys = set(media_keys)
        for item in self.facets.text_content.items:
            if any(reference.key not in all_media_keys for reference in item.media_refs):
                raise ValueError("text media_refs must reference existing media")
        if all(not item.animated for item in self.media):
            if any(
                description.motion_status is not MotionStatus.NOT_APPLICABLE
                for description in self.descriptions.values()
            ):
                raise ValueError("static-only media requires motion_status=not_applicable")
            if self.facets.text_content.dynamics is not TextDynamics.STABLE:
                raise ValueError("static-only media requires stable text dynamics")
        if (
            any(
                description.motion_status is MotionStatus.UNDETERMINED
                for description in self.descriptions.values()
            )
            and Uncertainty.MOTION not in self.facets.uncertainties
        ):
            raise ValueError("undetermined motion requires motion uncertainty")
        style_values = set(self.facets.styles)
        style_conflict = {Style.MINIMAL, Style.DETAILED}.issubset(style_values) or {
            Style.OUTLINE,
            Style.SOLID,
        }.issubset(style_values)
        if (
            style_conflict
            and Uncertainty.STYLE not in self.facets.uncertainties
            and self.review.status is not ReviewStatus.APPROVED
        ):
            raise ValueError("conflicting styles require uncertainty or approved review")
        if (
            any(
                description.motion_status is MotionStatus.PROCESSING_FAILED
                for description in self.descriptions.values()
            )
            and self.availability.status is AvailabilityStatus.ACTIVE
        ):
            raise ValueError("processing_failed is forbidden in a published active emoji")
        return self


class Membership(StrictModel):
    schema_version: Literal["1.0.0"]
    entity_type: Literal["membership"]
    id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxm_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    collection_id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxc_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    emoji_id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxe_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    status: MembershipStatus
    position: Annotated[int, Field(ge=0, strict=True)]
    first_seen_at: UtcTimestamp
    last_changed_at: UtcTimestamp

    @model_validator(mode="after")
    def validate_dates(self) -> Membership:
        if _parse_timestamp(self.last_changed_at) < _parse_timestamp(self.first_seen_at):
            raise ValueError("last_changed_at cannot precede first_seen_at")
        return self


class RelationMediaPair(StrictModel):
    subject_role: MediaRole
    subject_variant_id: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=64,
                pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$",
            ),
        ]
        | None
    ) = None
    object_role: MediaRole
    object_variant_id: (
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=64,
                pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$",
            ),
        ]
        | None
    ) = None

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.subject_role.value,
            self.subject_variant_id or "",
            self.object_role.value,
            self.object_variant_id or "",
        )


class RelationEvidence(StrictModel):
    dedupe_profile: Annotated[
        str,
        StringConstraints(
            min_length=1,
            max_length=64,
            pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        ),
    ]
    subject_media_digest: Sha256
    object_media_digest: Sha256
    media_pairs: Annotated[list[RelationMediaPair], Field(min_length=1)]
    signals: Annotated[
        list[
            Annotated[
                str,
                StringConstraints(
                    min_length=1,
                    max_length=64,
                    pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
                ),
            ]
        ],
        Field(min_length=1),
    ]

    @field_validator("media_pairs")
    @classmethod
    def ordered_unique_pairs(cls, value: list[RelationMediaPair]) -> list[RelationMediaPair]:
        keys = [item.sort_key for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("relation media pairs must be unique")
        if keys != sorted(keys):
            raise ValueError("relation media pairs must be canonically sorted")
        return value

    @field_validator("signals")
    @classmethod
    def ordered_unique_signals(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("relation signals must be unique")
        if value != sorted(value):
            raise ValueError("relation signals must be sorted")
        return value


class RelationReview(StrictModel):
    status: ReviewStatus
    reviewer: Handle | None = None
    reviewed_at: UtcTimestamp | None = None
    reviewed_relation_sha256: Sha256 | None = None

    @field_validator("reviewer")
    @classmethod
    def clean_reviewer(cls, value: str | None) -> str | None:
        return _ensure_clean_text(value) if value is not None else None

    @model_validator(mode="after")
    def validate_review(self) -> RelationReview:
        reviewed = self.status is not ReviewStatus.UNREVIEWED
        fields = (self.reviewer, self.reviewed_at, self.reviewed_relation_sha256)
        if reviewed and any(value is None for value in fields):
            raise ValueError("reviewed relation requires reviewer, timestamp, and hash")
        if not reviewed and any(value is not None for value in fields):
            raise ValueError("unreviewed relation must not retain review metadata")
        if self.reviewed_at is not None:
            _parse_timestamp(self.reviewed_at)
        return self


class VisualRelation(StrictModel):
    schema_version: Literal["1.0.0"]
    entity_type: Literal["visual_relation"]
    id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxr_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    identity_epoch: Annotated[int, Field(ge=0, strict=True)]
    subject_id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxe_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    object_id: Annotated[
        str,
        StringConstraints(
            pattern=r"^mxe_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    ]
    scope: VisualRelationScope
    relation_type: VisualRelationType
    evidence: RelationEvidence
    review: RelationReview

    @model_validator(mode="after")
    def validate_relation(self) -> VisualRelation:
        if self.identity_epoch != 0:
            raise ValueError("visual relation identity_epoch must be zero in schema 1.0.0")
        if self.subject_id == self.object_id:
            raise ValueError("visual relation endpoints must be distinct")
        if (
            self.relation_type
            in {
                VisualRelationType.SAME_ARTWORK,
                VisualRelationType.NOT_DUPLICATE,
                VisualRelationType.RELATED_SERIES,
            }
            and self.subject_id >= self.object_id
        ):
            raise ValueError("symmetric visual relations require subject_id < object_id")
        if self.scope is VisualRelationScope.MEDIA_PAIR and len(self.evidence.media_pairs) != 1:
            raise ValueError("media-pair relation requires exactly one evidence media pair")
        return self


class Tombstone(StrictModel):
    schema_version: Literal["1.0.0"]
    entity_type: Literal["tombstone"]
    target_entity_type: Literal["collection", "emoji", "membership"]
    target_id: Annotated[
        str,
        StringConstraints(
            pattern=(
                r"^mx[cem]_[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-"
                r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
            )
        ),
    ]
    reason_code: TakedownReason
    withheld_at: UtcTimestamp
    public_note: Annotated[str, StringConstraints(min_length=1, max_length=280)]

    @field_validator("public_note")
    @classmethod
    def clean_note(cls, value: str) -> str:
        return _ensure_clean_text(value, natural=True)

    @model_validator(mode="after")
    def validate_target_prefix(self) -> Tombstone:
        expected = {"collection": "mxc_", "emoji": "mxe_", "membership": "mxm_"}
        if not self.target_id.startswith(expected[self.target_entity_type]):
            raise ValueError("target_id prefix does not match target_entity_type")
        _parse_timestamp(self.withheld_at)
        return self


Entity = Collection | Emoji | Membership | VisualRelation | Tombstone


def parse_entity(value: dict[str, Any]) -> Entity:
    """Parse a closed universal entity based on its discriminating type."""

    entity_type = value.get("entity_type")
    model: type[Entity]
    if entity_type == "collection":
        model = Collection
    elif entity_type == "emoji":
        model = Emoji
    elif entity_type == "membership":
        model = Membership
    elif entity_type == "visual_relation":
        model = VisualRelation
    elif entity_type == "tombstone":
        model = Tombstone
    else:
        raise ValueError(f"unsupported entity_type: {entity_type!r}")
    return model.model_validate(value)

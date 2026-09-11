"""Typed outputs of the local deterministic media analyzer."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"

ColorFamily = Literal[
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
]


class AnalysisError(RuntimeError):
    """Deterministic analysis failed or its immutable profile is unavailable."""

    code = "MEDIA_ANALYSIS_FAILED"


class DominantColor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hex: str = Field(pattern=r"^#[0-9a-f]{6}$")
    family: ColorFamily
    coverage_bp: int = Field(ge=1, le=10_000)


class RenderingSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    color_behavior: Literal["fixed", "platform-adaptive", "mixed", "unknown"]
    palette_dynamics: Literal["stable", "changing", "unknown"]
    alpha_mode: Literal["opaque", "binary", "translucent", "unknown"]
    visible_area_bp: int = Field(ge=0, le=10_000)
    dominant_colors: tuple[DominantColor, ...] | None = Field(default=None, max_length=5)
    adaptive_mask_source: str | None = None
    adaptive_mask_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def conditional_palette(self) -> RenderingSignals:
        colors = self.dominant_colors
        if colors is not None and sum(color.coverage_bp for color in colors) > 10_000:
            raise ValueError("dominant color coverage must not exceed 10000 basis points")
        if self.color_behavior == "platform-adaptive" and colors is not None:
            raise ValueError("platform-adaptive rendering cannot contain dominant colors")
        if self.color_behavior in {"fixed", "mixed"}:
            if self.visible_area_bp > 0 and not colors:
                raise ValueError("visible fixed rendering requires dominant colors")
            if self.visible_area_bp == 0 and colors is not None:
                raise ValueError("invisible rendering cannot contain dominant colors")
        if self.color_behavior == "mixed" and (
            not self.adaptive_mask_source or not self.adaptive_mask_sha256
        ):
            raise ValueError("mixed rendering requires a normative adaptive mask")
        if self.color_behavior != "mixed" and (
            self.adaptive_mask_source is not None or self.adaptive_mask_sha256 is not None
        ):
            raise ValueError("adaptive mask metadata is only valid for mixed rendering")
        return self


class PerceptualSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    encoding: Literal["u64be-base64url-nopad"] = "u64be-base64url-nopad"
    sample_count: Literal[1, 16]
    layout_phash64: str
    content_phash64: str
    alpha_phash64: str
    edge_phash64: str
    temporal_energy_bp: int = Field(ge=0, le=10_000)
    low_information: bool

    @model_validator(mode="after")
    def validate_packed_hashes(self) -> PerceptualSignals:
        expected = self.sample_count * 8
        for field in (
            "layout_phash64",
            "content_phash64",
            "alpha_phash64",
            "edge_phash64",
        ):
            value = getattr(self, field)
            if "=" in value:
                raise ValueError("packed pHash must not contain base64 padding")
            try:
                decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            except (ValueError, TypeError) as exc:
                raise ValueError("packed pHash is not base64url") from exc
            if len(decoded) != expected:
                raise ValueError("packed pHash byte length does not match sample_count")
        return self


class FingerprintSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decoded_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_render_sha256: str = Field(pattern=SHA256_PATTERN)
    shape_sha256: str = Field(pattern=SHA256_PATTERN)
    perceptual: PerceptualSignals


class DeterministicMediaAnalysis(BaseModel):
    """Per-media result; role and optional variant_id are bound by the caller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    color_profile: Literal["color-v1"] = "color-v1"
    color_profile_sha256: str = Field(pattern=SHA256_PATTERN)
    dedupe_profile: Literal["dedupe-v1"] = "dedupe-v1"
    dedupe_profile_sha256: str = Field(pattern=SHA256_PATTERN)
    decoder_backend_fingerprint: str = Field(pattern=SHA256_PATTERN)
    analysis_scope: Literal["full-decoded-stream"] = "full-decoded-stream"
    rendering: RenderingSignals
    fingerprint: FingerprintSignals

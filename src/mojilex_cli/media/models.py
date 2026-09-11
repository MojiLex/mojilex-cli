"""Media processing models and immutable hard safety ceilings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mojilex_cli.analysis import DeterministicMediaAnalysis

MIB = 1024 * 1024
HARD_MAX_FILE_BYTES = 20 * MIB
HARD_MAX_TGS_JSON_BYTES = 8 * MIB
HARD_MAX_PIXELS = 16_000_000
HARD_MAX_DURATION_MS = 10_000
HARD_MAX_FRAMES = 16
HARD_MAX_WORKER_SECONDS = 30.0
HARD_MAX_WORKER_MEMORY = 512 * MIB
HARD_MAX_RUN_TEMP_BYTES = 2 * 1024**3
PIPELINE_VERSION = "1.0.0"


class MediaError(RuntimeError):
    code = "MEDIA_INVALID"


class MediaLimitError(MediaError):
    code = "MEDIA_LIMIT_EXCEEDED"


class MediaRenderError(MediaError):
    code = "MEDIA_RENDER_FAILED"


class MediaDependencyError(MediaError):
    code = "SYSTEM_DEPENDENCY_MISSING"


class MediaLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_file_bytes: int = Field(default=HARD_MAX_FILE_BYTES, ge=1, le=HARD_MAX_FILE_BYTES)
    max_tgs_json_bytes: int = Field(
        default=HARD_MAX_TGS_JSON_BYTES, ge=1, le=HARD_MAX_TGS_JSON_BYTES
    )
    max_pixels: int = Field(default=HARD_MAX_PIXELS, ge=1, le=HARD_MAX_PIXELS)
    max_duration_ms: int = Field(default=HARD_MAX_DURATION_MS, ge=1, le=HARD_MAX_DURATION_MS)
    frames: int = Field(default=8, ge=4, le=HARD_MAX_FRAMES)
    worker_timeout_seconds: float = Field(
        default=HARD_MAX_WORKER_SECONDS, gt=0, le=HARD_MAX_WORKER_SECONDS
    )
    worker_memory_bytes: int = Field(
        default=HARD_MAX_WORKER_MEMORY, ge=64 * MIB, le=HARD_MAX_WORKER_MEMORY
    )
    max_run_temp_bytes: int = Field(
        default=HARD_MAX_RUN_TEMP_BYTES, ge=1, le=HARD_MAX_RUN_TEMP_BYTES
    )


class MediaMetadata(BaseModel):
    # This model is also the trust boundary for metadata restored from the
    # persistent cache.  Do not let JSON strings/bools be coerced into technical
    # facts which were supposed to come from the decoder.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["primary"] = "primary"
    kind: Literal["static", "animation", "video"]
    format: Literal["webp", "tgs", "webm"]
    mime_type: Literal["image/webp", "application/x-tgsticker", "video/webm"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=1, le=HARD_MAX_FILE_BYTES)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    animated: bool
    duration_ms: int | None = Field(default=None, gt=0, le=HARD_MAX_DURATION_MS)

    @model_validator(mode="after")
    def validate_shape(self) -> MediaMetadata:
        expected = {
            "webp": ("static", "image/webp", False),
            "tgs": ("animation", "application/x-tgsticker", True),
            "webm": ("video", "video/webm", True),
        }[self.format]
        if (self.kind, self.mime_type, self.animated) != expected:
            raise ValueError("kind, MIME type, and animated flag must match format")
        if self.animated and self.duration_ms is None:
            raise ValueError("duration_ms is required for animated media")
        if not self.animated and self.duration_ms is not None:
            raise ValueError("duration_ms is forbidden for static media")
        if self.width * self.height > HARD_MAX_PIXELS:
            raise ValueError("decoded image exceeds 16 megapixels")
        return self


class ProcessedMedia(BaseModel):
    """Temporary frame paths are excluded from canonical metadata dumps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: MediaMetadata
    analysis: DeterministicMediaAnalysis | None = None
    frame_paths: tuple[Path, ...] = Field(exclude=True, repr=False)
    dark_frame_paths: tuple[Path, ...] = Field(default=(), exclude=True, repr=False)
    rendered_frame_count: int | None = Field(
        default=None, ge=1, le=HARD_MAX_FRAMES, exclude=True, repr=False
    )
    has_dark_render: bool | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_transient_render_context(self) -> ProcessedMedia:
        if self.frame_paths and self.rendered_frame_count not in {None, len(self.frame_paths)}:
            raise ValueError("rendered frame count differs from transient frame paths")
        if self.dark_frame_paths and len(self.dark_frame_paths) != len(self.frame_paths):
            raise ValueError("light and dark render frame counts differ")
        if self.dark_frame_paths and self.has_dark_render is False:
            raise ValueError("dark render marker differs from transient frame paths")
        return self

    @property
    def semantic_frame_count(self) -> int:
        return self.rendered_frame_count or len(self.frame_paths)

    @property
    def semantic_has_dark_render(self) -> bool:
        if self.has_dark_render is not None:
            return self.has_dark_render
        return bool(self.dark_frame_paths)

    def dataset_metadata(self) -> dict[str, object]:
        return self.metadata.model_dump(exclude_none=True)


class BackendStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    available: bool
    fixture_decoded: bool
    detail: str

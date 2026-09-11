"""Provider-neutral source adapter contracts and safe DTOs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class SourceError(RuntimeError):
    code = "SOURCE_ERROR"
    retryable = False


class UnsupportedSourceError(SourceError):
    code = "SOURCE_UNSUPPORTED"


class SourceNotFoundError(SourceError):
    code = "SOURCE_NOT_FOUND"


class SourceAuthError(SourceError):
    code = "AUTH_FAILED"


class SourceNetworkError(SourceError):
    code = "NETWORK_ERROR"
    retryable = True


class SourceRateLimitError(SourceError):
    code = "RATE_LIMITED"
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SourceCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str
    media_formats: tuple[str, ...]
    stable_collection_ids: bool
    stable_emoji_ids: bool
    supports_collection_availability: bool
    supports_emoji_availability: bool
    max_direct_emoji_lookup: int | None = None


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str
    native_namespace: str
    scope_id: str
    native_id: str
    canonical_url: str


class SourceEmoji(BaseModel):
    """A normalized source item; ``file_id`` is transient and excluded from dumps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native_namespace: str
    scope_id: str
    native_id: str
    file_unique_id: str
    position: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    animated: bool
    video: bool
    needs_repainting: bool = False
    fallback_emoji: str | None = None
    declared_file_size: int | None = Field(default=None, ge=0)
    media_format: Literal["webp", "tgs", "webm"]
    file_id: str = Field(exclude=True, repr=False)

    def safe_dump(self) -> dict[str, object]:
        return self.model_dump(exclude={"file_id"})


class SourceCollection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str
    kind: str
    native_namespace: str
    scope_id: str
    native_id: str
    title: str
    canonical_url: str
    item_count: int = Field(ge=0)
    items: tuple[SourceEmoji, ...]
    extension: dict[str, object]

    def safe_dump(self) -> dict[str, object]:
        return self.model_dump(exclude={"items": {"__all__": {"file_id"}}})


class SourceMembership(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collection_native_id: str
    emoji_native_id: str
    position: int = Field(ge=0)
    status: Literal["active", "removed_from_collection", "unknown"] = "active"


@runtime_checkable
class SourceAdapter(Protocol):
    def recognize(self, source: str) -> bool: ...

    def canonicalize(self, source: str) -> SourceReference: ...

    async def validate_credentials(self) -> None: ...

    async def fetch_collection(self, native_ref: SourceReference) -> SourceCollection: ...

    async def list_memberships(
        self, collection_ref: SourceReference
    ) -> tuple[SourceMembership, ...]: ...

    async def fetch_emojis(self, native_ids: tuple[str, ...]) -> dict[str, SourceEmoji]: ...

    def fetch_media(self, emoji_ref: SourceEmoji) -> AsyncIterator[bytes]: ...

    async def check_collection_availability(self, native_ref: SourceReference) -> bool: ...

    async def check_emoji_availability(self, native_ids: tuple[str, ...]) -> dict[str, bool]: ...

    def capabilities(self) -> SourceCapabilities: ...

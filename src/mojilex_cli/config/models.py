"""Typed, non-secret configuration for MojiLex.

Credentials deliberately live in :class:`Credentials`, a separate runtime-only
object which is never included when configuration is serialized.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .paths import default_repository_path


class ConfigError(ValueError):
    """Raised when configuration is unsafe or invalid."""


class RepositoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(default_factory=lambda: str(default_repository_path()))
    base_branch: str = "main"
    publish: Literal["local", "pr"] = "pr"

    @field_validator("target")
    @classmethod
    def reject_credential_url(cls, value: str) -> str:
        from .secrets import url_has_credentials

        if url_has_credentials(value):
            raise ValueError("repository URL must not contain credentials")
        return value


class TelegramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    download_concurrency: int = Field(default=4, ge=1, le=32)
    max_attempts: int = Field(default=6, ge=1, le=8)


class AIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = "gemini"
    model: str = ""
    languages: tuple[str, ...] = ("ru", "en")
    max_ai_requests: int | None = Field(default=100, ge=0)
    max_cost_usd: Decimal | None = Field(default=None, ge=0)
    ai_concurrency: int = Field(default=1, ge=1, le=16)
    confirm_before_analysis: bool = False
    allow_unknown_cost: bool = False
    model_routing: Literal["off", "rules"] = "off"
    escalation_model: str = ""

    @field_validator("max_ai_requests", mode="before")
    @classmethod
    def unlimited_request_budget(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "unlimited":
            return None
        return value

    @field_validator("languages")
    @classmethod
    def languages_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("languages must be a non-empty unique sequence")
        if not {"ru", "en"}.issubset(value):
            raise ValueError("MVP configuration must include ru and en")
        return value

    @model_validator(mode="after")
    def routing_is_explicit(self) -> AIConfig:
        if self.model_routing == "rules" and not self.escalation_model:
            raise ValueError("rules model routing requires an explicit escalation model")
        return self


class DedupeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["off", "exact", "near"] = "exact"
    max_candidates: int = Field(default=20, ge=1, le=200)
    profile: str = Field(default="dedupe-v1", pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    official_pack_policy: Literal["ask", "skip", "allow"] = "skip"
    file_analysis_mode: Literal["fast", "sequential", "download_all", "prepare_all"] = "fast"
    pack_concurrency: int = Field(default=3, ge=1, le=8)
    render_concurrency: int = Field(default=2, ge=1, le=8)
    static_batch_size: int = Field(default=16, ge=1, le=16)
    animated_batch_size: int = Field(default=8, ge=1, le=16)
    keyframes: int = Field(default=8, ge=4, le=16)
    render_timeout_seconds: float = Field(default=15.0, gt=0, le=30)
    max_download_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=20 * 1024 * 1024)
    max_temp_bytes: int = Field(default=2 * 1024**3, ge=1, le=2 * 1024**3)


class GitIdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = None
    email: str | None = None

    @model_validator(mode="after")
    def complete_pair(self) -> GitIdentityConfig:
        if (self.name is None) != (self.email is None):
            raise ValueError("git identity requires both name and email")
        return self


class MojiLexConfig(BaseModel):
    """Resolved safe configuration. It never contains credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ui_language: Literal["en", "ru"] = "en"
    repository: RepositoryConfig = Field(default_factory=RepositoryConfig)
    telegram: TelegramConfig = TelegramConfig()
    ai: AIConfig = AIConfig()
    dedupe: DedupeConfig = DedupeConfig()
    processing: ProcessingConfig = ProcessingConfig()
    git_identity: GitIdentityConfig = GitIdentityConfig()
    cache_dir: Path | None = None
    runs_dir: Path | None = None


class Credentials(BaseModel):
    """Credentials resolved from the process environment or opt-in OS keyring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    telegram_bot_token: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    github_token: str | None = Field(default=None, repr=False)

    def redaction_values(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.telegram_bot_token,
                self.gemini_api_key,
                self.openai_api_key,
                self.github_token,
            )
            if value
        )

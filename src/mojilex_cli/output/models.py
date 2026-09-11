"""Stable machine-output envelope and error registry."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, cast


class ExitCode(IntEnum):
    SUCCESS = 0
    INTERNAL = 1
    CONFIG = 2
    SYSTEM = 3
    AUTH = 4
    SOURCE = 5
    NETWORK = 6
    MEDIA = 7
    AI = 8
    BUDGET = 9
    VALIDATION = 10
    GIT = 11
    GITHUB = 12
    PARTIAL = 13
    INTERRUPTED = 130


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    NOOP = "noop"
    DRY_RUN = "dry_run"
    STALE = "stale"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BUDGET_EXCEEDED = "budget_exceeded"


ERROR_EXIT_CODES: dict[str, ExitCode] = {
    "INTERNAL_ERROR": ExitCode.INTERNAL,
    "CONFIG_INVALID": ExitCode.CONFIG,
    "CONFIG_MISSING": ExitCode.CONFIG,
    "SYSTEM_DEPENDENCY_MISSING": ExitCode.SYSTEM,
    "CREDENTIAL_MISSING": ExitCode.AUTH,
    "AUTH_FAILED": ExitCode.AUTH,
    "PERMISSION_DENIED": ExitCode.AUTH,
    "SOURCE_UNSUPPORTED": ExitCode.SOURCE,
    "SOURCE_NOT_FOUND": ExitCode.SOURCE,
    "SOURCE_CHANGED_DURING_RUN": ExitCode.SOURCE,
    "IDENTITY_CONFLICT": ExitCode.SOURCE,
    "NETWORK_ERROR": ExitCode.NETWORK,
    "RATE_LIMITED": ExitCode.NETWORK,
    "MEDIA_INVALID": ExitCode.MEDIA,
    "MEDIA_LIMIT_EXCEEDED": ExitCode.MEDIA,
    "MEDIA_RENDER_FAILED": ExitCode.MEDIA,
    "AI_REQUEST_FAILED": ExitCode.AI,
    "AI_OUTPUT_INVALID": ExitCode.AI,
    "BUDGET_EXCEEDED": ExitCode.BUDGET,
    "UNKNOWN_COST": ExitCode.BUDGET,
    "VALIDATION_FAILED": ExitCode.VALIDATION,
    "DIRTY_WORKTREE": ExitCode.GIT,
    "GIT_CONFLICT": ExitCode.GIT,
    "GITHUB_PUBLISH_FAILED": ExitCode.GITHUB,
    "REQUIRED_CHECK_FAILED": ExitCode.GITHUB,
    "PARTIAL_FAILURE": ExitCode.PARTIAL,
    "INTERRUPTED": ExitCode.INTERRUPTED,
}

_TOKEN_RE = re.compile(
    r"(?i)(?:\b\d{6,12}:[A-Za-z0-9_-]{30,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{30,}\b|https://api\.telegram\.org/file/bot[^\s\"']+)"
)
_SECRET_KEY_RE = re.compile(r"(?:token|secret|password|credential|authorization|api[_-]?key)", re.I)
_AUTH_SCHEME_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[^\s,;\"'<>]+")
_NAMED_CREDENTIAL_RE = re.compile(
    r"(?i)(\b(?:(?:[a-z0-9]+[_-])*(?:token|secret|password|credential)"
    r"(?:[_-](?:header|value))?|authorization(?:[_-]?header)?|"
    r"(?:[a-z0-9]+[_-])*api[_-]?key)\b\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&}\]]+)"
)
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s]+@")
_URL_QUERY_CREDENTIAL_RE = re.compile(
    r"(?i)([?&](?:(?:[a-z0-9]+[_-])*(?:token|secret|password|credential)|"
    r"(?:[a-z0-9]+[_-])*api[_-]?key)=)[^&#\s\"']+"
)


def _redact_text(value: str) -> str:
    result = _AUTH_SCHEME_RE.sub(r"\1 [REDACTED]", value)
    result = _NAMED_CREDENTIAL_RE.sub(r"\1[REDACTED]", result)
    result = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", result)
    result = _URL_QUERY_CREDENTIAL_RE.sub(r"\1[REDACTED]", result)
    return _TOKEN_RE.sub("[REDACTED]", result)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _SECRET_KEY_RE.search(text_key):
                result[text_key] = "[REDACTED]"
            else:
                result[text_key] = redact(item)
        return result
    return value


@dataclass(frozen=True, slots=True)
class StructuredError:
    code: str
    message: str
    retryable: bool
    hint: str
    source: str | None = None
    entity_id: str | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_EXIT_CODES:
            raise ValueError(f"unregistered error code: {self.code}")
        if not self.message or not self.hint:
            raise ValueError("message and hint are required")

    @property
    def exit_code(self) -> ExitCode:
        return ERROR_EXIT_CODES[self.code]

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "hint": self.hint,
        }
        if self.source is not None:
            result["source"] = self.source
        if self.entity_id is not None:
            result["entity_id"] = self.entity_id
        if self.details is not None:
            result["details"] = self.details
        return cast(dict[str, Any], redact(result))


@dataclass(slots=True)
class OutputEnvelope:
    ok: bool
    command: str
    status: RunStatus | str
    run_id: str
    result: dict[str, Any] = field(default_factory=dict)
    publication: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any] | str] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)
    schema_version: str = "1"

    def as_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            redact(
                {
                    "schema_version": self.schema_version,
                    "ok": self.ok,
                    "command": self.command,
                    "status": self.status.value
                    if isinstance(self.status, RunStatus)
                    else self.status,
                    "run_id": self.run_id,
                    "result": self.result,
                    "publication": self.publication,
                    "warnings": self.warnings,
                    "errors": [item.as_dict() for item in self.errors],
                }
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))

    @property
    def exit_code(self) -> ExitCode:
        if not self.errors:
            return ExitCode.SUCCESS
        if self.status == RunStatus.PARTIAL or self.status == RunStatus.PARTIAL.value:
            return ExitCode.PARTIAL
        return max((item.exit_code for item in self.errors), key=int)

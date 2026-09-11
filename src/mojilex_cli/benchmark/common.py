"""Shared strict manifest and deterministic report primitives."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar
from urllib.parse import urlsplit

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SEMVER_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
ID_PATTERN = r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$"
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_M = TypeVar("_M", bound=BaseModel)


class BenchmarkError(RuntimeError):
    """A manifest or benchmark input failed a safe, actionable check."""

    code = "VALIDATION_FAILED"


class RightsRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rights_id: str = Field(pattern=ID_PATTERN)
    basis: str = Field(pattern=r"^(?:synthetic|rights-cleared)$")
    license_spdx: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9.+-]+$")
    attribution: str = Field(min_length=1, max_length=500)
    redistribution_allowed: bool
    source_url: str | None = Field(default=None, max_length=2048)

    @field_validator("attribution")
    @classmethod
    def safe_attribution(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("rights attribution must be safe single-line text")
        return value

    @field_validator("source_url")
    @classmethod
    def public_https_source(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("rights source_url must be public credential-free HTTPS")
        return value


class DeclaredFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def canonical_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("benchmark file path must be canonical relative POSIX")
        return value


def load_manifest(path: Path, model: type[_M]) -> tuple[_M, str, bytes]:
    """Load a bounded UTF-8 JSON manifest and return its exact byte hash."""

    try:
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise OSError("symlink")
        resolved = expanded.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
        if resolved.stat().st_size > _MAX_MANIFEST_BYTES:
            raise BenchmarkError("benchmark manifest exceeds 32 MiB")
        raw = resolved.read_bytes()
        json.loads(raw, object_pairs_hook=_unique_object)
        value = model.model_validate_json(raw, strict=True)
    except BenchmarkError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise BenchmarkError("benchmark manifest is missing, malformed, or invalid") from exc
    return value, hashlib.sha256(raw).hexdigest(), raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def resolve_declared_file(root: Path, declaration: DeclaredFile) -> Path:
    """Resolve one declared file below root without accepting symlinks."""

    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(declaration.path).parts)
    try:
        current = candidate
        while current != root:
            if current.is_symlink():
                raise OSError("symlink")
            current = current.parent
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file() or resolved.stat().st_size > _MAX_MANIFEST_BYTES:
            raise OSError("unsafe file")
    except (OSError, ValueError) as exc:
        raise BenchmarkError("declared benchmark input is missing or unsafe") from exc
    if file_sha256(resolved) != declaration.sha256:
        raise BenchmarkError("declared benchmark input hash mismatch")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def declared_files_sha256(files: Sequence[DeclaredFile]) -> str:
    payload = [{"path": item.path, "sha256": item.sha256} for item in files]
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def finalize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a content hash over the report excluding the hash field itself."""

    result = dict(report)
    result["report_sha256"] = hashlib.sha256(rfc8785.dumps(result)).hexdigest()
    return result


def ratio_bp(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (2 * numerator * 10_000 + denominator) // (2 * denominator)


def wilson_interval_bp(successes: int, total: int) -> dict[str, int]:
    """Deterministic 95% Wilson interval, rounded to basis points."""

    if total <= 0:
        return {"lower_bp": 0, "upper_bp": 0}
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z
        * ((proportion * (1 - proportion) / total + z_squared / (4 * total * total)) ** 0.5)
        / denominator
    )
    return {
        "lower_bp": max(0, min(10_000, round((center - margin) * 10_000))),
        "upper_bp": max(0, min(10_000, round((center + margin) * 10_000))),
    }


def validate_canonical_ids(values: Sequence[str], *, label: str) -> None:
    if list(values) != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique and lexicographically sorted")
    if any(re.fullmatch(ID_PATTERN, value) is None for value in values):
        raise ValueError(f"{label} contains an invalid ID")

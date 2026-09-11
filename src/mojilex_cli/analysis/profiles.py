"""Exact loading and hashing of immutable bundled analysis profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any, cast

from .models import AnalysisError

_PROFILE_PACKAGE = "mojilex_cli.analysis_profiles"

# Updated only when a new immutable profile ID is introduced. A byte change under
# an existing ID fails closed rather than silently changing published fingerprints.
_PROFILE_SHA256 = {
    "color-v1": "c30dcf7241037e3c1e4c4a16686c4e899c3f81bae19bd741eb257e458f9986ae",
    "dedupe-v1": "8f6eb6f68479897a8fe5be434ca2f081edcf4cb08cc1371308e732022f7c66c1",
    "collection-dedupe-v1": "a03a1979980c6e23323cac02459fa7a29bf68c858c45d2b52b9a54857766cf01",
}


@dataclass(frozen=True)
class AnalysisProfile:
    profile_id: str
    sha256: str
    raw_bytes: bytes
    data: Mapping[str, Any]


def load_analysis_profile(profile_id: str) -> AnalysisProfile:
    expected = _PROFILE_SHA256.get(profile_id)
    if expected is None:
        raise AnalysisError("unknown deterministic analysis profile")
    try:
        raw = files(_PROFILE_PACKAGE).joinpath(f"{profile_id}.json").read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise AnalysisError("bundled deterministic analysis profile is unavailable") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise AnalysisError("bundled deterministic analysis profile hash mismatch")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("bundled deterministic analysis profile is invalid JSON") from exc
    if not isinstance(parsed, dict) or parsed.get("profile_id") != profile_id:
        raise AnalysisError("bundled deterministic analysis profile ID mismatch")
    return AnalysisProfile(
        profile_id=profile_id,
        sha256=actual,
        raw_bytes=raw,
        data=cast(Mapping[str, Any], _freeze(parsed)),
    )


def profile_sha256(profile_id: str) -> str:
    return load_analysis_profile(profile_id).sha256


def known_profile_hashes() -> Mapping[str, str]:
    # Return a read-only copy so callers cannot mutate the trust roots.
    return MappingProxyType(dict(_PROFILE_SHA256))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value

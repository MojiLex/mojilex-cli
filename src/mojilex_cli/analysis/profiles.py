"""Exact loading and hashing of immutable bundled analysis profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from referencing import Registry, Resource

from mojilex_cli.schemas import EmbeddedSchema, EmbeddedSchemaError, embedded_schemas

from .models import AnalysisError

_PROFILE_PACKAGE = "mojilex_cli.analysis_profiles"

# Updated only when a new immutable profile ID is introduced. A byte change under
# an existing ID fails closed rather than silently changing published fingerprints.
_PROFILE_SHA256 = {
    "color-v1": "fa7f0cb270bd3645b78ec5e6c1b8d6b23f8f2a07457699bc1ac95cf0c70aa230",
    "dedupe-v1": "c1f09fd2a4abb416b7dec82f67f9b101e1e9f993a8d578908a108f43d87602ba",
    "collection-dedupe-v1": "5640836b227b4013772e6e48ce06e52245ad980bd0a554111d389504366fee8b",
}
_DELEGATED_PROFILE_SCHEMA_URI = "mlx://schemas/distribution/v1/delegated-profile.schema.json"
_PROFILE_TYPES = {
    "color-v1": "color",
    "dedupe-v1": "dedupe",
    "collection-dedupe-v1": "collection-dedupe",
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
    effective = _effective_profile_body(profile_id, cast(dict[str, Any], parsed))
    return AnalysisProfile(
        profile_id=profile_id,
        sha256=actual,
        raw_bytes=raw,
        data=cast(Mapping[str, Any], _freeze(effective)),
    )


@lru_cache(maxsize=1)
def _embedded_schema_graph() -> tuple[
    dict[str, dict[str, Any]], Registry[Any], Mapping[str, EmbeddedSchema]
]:
    try:
        embedded = embedded_schemas()
    except EmbeddedSchemaError as exc:
        raise AnalysisError("bundled schema trust root is unavailable") from exc
    schemas: dict[str, dict[str, Any]] = {}
    registry: Registry[Any] = Registry()
    try:
        for uri, item in embedded.items():
            document = json.loads(item.payload)
            if not isinstance(document, dict) or document.get("$id") != uri:
                raise ValueError("embedded schema identity mismatch")
            typed = cast(dict[str, Any], document)
            Draft202012Validator.check_schema(typed)
            schemas[uri] = typed
            registry = registry.with_resource(uri, Resource.from_contents(typed))
    except Exception as exc:
        raise AnalysisError("bundled schema trust root is invalid") from exc
    return schemas, registry, embedded


def _validate_embedded(value: object, schema_uri: str, *, label: str) -> None:
    schemas, registry, _ = _embedded_schema_graph()
    schema = schemas.get(schema_uri)
    if schema is None:
        raise AnalysisError(f"bundled {label} schema is unavailable")
    try:
        errors = sorted(
            Draft202012Validator(
                schema,
                registry=registry,
                format_checker=FormatChecker(),
            ).iter_errors(value),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
    except Exception as exc:
        raise AnalysisError(f"bundled {label} schema cannot resolve offline") from exc
    if errors:
        first = errors[0]
        path = "/".join(str(item) for item in first.absolute_path) or "<root>"
        raise AnalysisError(f"bundled {label} violates its schema at {path}: {first.message}")


def _effective_profile_body(profile_id: str, wrapper: dict[str, Any]) -> dict[str, Any]:
    _validate_embedded(wrapper, _DELEGATED_PROFILE_SCHEMA_URI, label="analysis profile wrapper")
    profile_type = _PROFILE_TYPES[profile_id]
    if wrapper.get("profile_type") != profile_type:
        raise AnalysisError("bundled deterministic analysis profile type mismatch")
    expected_ref = f"mlx://schemas/distribution/v1/{profile_type}-profile-contract.schema.json"
    if wrapper.get("contract_schema_ref") != expected_ref:
        raise AnalysisError("bundled deterministic analysis profile contract mismatch")
    _, _, embedded = _embedded_schema_graph()
    contract = embedded.get(expected_ref)
    if contract is None:
        raise AnalysisError("bundled deterministic analysis profile contract is unavailable")
    if wrapper.get("contract_schema_sha256") != contract.sha256:
        raise AnalysisError("bundled deterministic analysis profile contract hash mismatch")
    body = wrapper.get("body")
    if not isinstance(body, dict):
        raise AnalysisError("bundled deterministic analysis profile body is invalid")
    _validate_embedded(body, expected_ref, label="analysis profile body")
    effective = cast(dict[str, Any], dict(body))
    effective["profile_id"] = profile_id
    return effective


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

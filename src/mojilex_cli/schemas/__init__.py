"""Immutable JSON Schema trust roots bundled with the installed CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, cast

_SCHEMA_PACKAGE = "mojilex_cli.schemas"

# Keep this closed inventory explicit: silently dropping a package resource must
# fail closed instead of shrinking the reader's schema trust root.
EMBEDDED_SCHEMA_PATHS = (
    "v1/collection.schema.json",
    "v1/common.schema.json",
    "v1/dataset.schema.json",
    "v1/emoji.schema.json",
    "v1/extensions/telegram.schema.json",
    "v1/facets.schema.json",
    "v1/fingerprints.schema.json",
    "v1/membership.schema.json",
    "v1/tombstone.schema.json",
    "v1/visual-relation.schema.json",
    "distribution/v1/agent-record.schema.json",
    "distribution/v1/analysis-profile.schema.json",
    "distribution/v1/artifact-descriptor.schema.json",
    "distribution/v1/bundle-descriptor.schema.json",
    "distribution/v1/bundling-profile-contract.schema.json",
    "distribution/v1/cli-command-result.schema.json",
    "distribution/v1/cli-jsonl-item.schema.json",
    "distribution/v1/cli-jsonl-metadata.schema.json",
    "distribution/v1/cli-jsonl-summary.schema.json",
    "distribution/v1/cli-read-envelope.schema.json",
    "distribution/v1/cli-request.schema.json",
    "distribution/v1/cli-resolution-candidate.schema.json",
    "distribution/v1/cli-similar-item.schema.json",
    "distribution/v1/collection-dedupe-profile-contract.schema.json",
    "distribution/v1/collection-facet.schema.json",
    "distribution/v1/color-profile-contract.schema.json",
    "distribution/v1/compression-profile-contract.schema.json",
    "distribution/v1/concept-candidate-profile.schema.json",
    "distribution/v1/concept.schema.json",
    "distribution/v1/concepts-registry.schema.json",
    "distribution/v1/dedupe-profile-contract.schema.json",
    "distribution/v1/delegated-profile.schema.json",
    "distribution/v1/distribution-common.schema.json",
    "distribution/v1/distribution-profile-contract.schema.json",
    "distribution/v1/duplicate-group-membership.schema.json",
    "distribution/v1/duplicate-group.schema.json",
    "distribution/v1/key-serialization-profile-contract.schema.json",
    "distribution/v1/language-canonicalization-profile-contract.schema.json",
    "distribution/v1/language-fallback-profile-contract.schema.json",
    "distribution/v1/lexical-search-profile-contract.schema.json",
    "distribution/v1/part-packing-profile-contract.schema.json",
    "distribution/v1/partitioning-profile-contract.schema.json",
    "distribution/v1/platform-profile.schema.json",
    "distribution/v1/platform-profiles-registry.schema.json",
    "distribution/v1/release-build-input.schema.json",
    "distribution/v1/release-manifest.schema.json",
    "distribution/v1/resource-descriptor.schema.json",
    "distribution/v1/rights-profile.schema.json",
    "distribution/v1/rights-profiles-registry.schema.json",
    "distribution/v1/search-record.schema.json",
    "distribution/v1/search-request.schema.json",
    "distribution/v1/taxonomy-dictionary.schema.json",
    "distribution/v1/taxonomy-registry.schema.json",
    "distribution/v1/taxonomy-source.schema.json",
)


class EmbeddedSchemaError(RuntimeError):
    """The installed schema trust root is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class EmbeddedSchema:
    uri: str
    relative_path: str
    payload: bytes
    sha256: str


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def embedded_schemas() -> Mapping[str, EmbeddedSchema]:
    """Load exact schema bytes through ``importlib.resources`` for wheel safety."""

    package_root = files(_SCHEMA_PACKAGE)
    result: dict[str, EmbeddedSchema] = {}
    for relative_path in EMBEDDED_SCHEMA_PATHS:
        try:
            payload = package_root.joinpath(*relative_path.split("/")).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
            raise EmbeddedSchemaError(
                f"bundled schema resource is unavailable: {relative_path}"
            ) from exc
        if payload.startswith(b"\xef\xbb\xbf"):
            raise EmbeddedSchemaError(f"bundled schema has a UTF-8 BOM: {relative_path}")
        try:
            parsed = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_no_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid number: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EmbeddedSchemaError(
                f"bundled schema is not strict JSON: {relative_path}"
            ) from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("$id"), str):
            raise EmbeddedSchemaError(f"bundled schema has no absolute $id: {relative_path}")
        document = cast(dict[str, Any], parsed)
        uri = cast(str, document["$id"])
        if uri in result or not (uri.startswith("https://") or uri.startswith("mlx://")):
            raise EmbeddedSchemaError(f"bundled schema has an invalid or duplicate $id: {uri}")
        result[uri] = EmbeddedSchema(
            uri=uri,
            relative_path=relative_path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    if len(result) != len(EMBEDDED_SCHEMA_PATHS):
        raise EmbeddedSchemaError("bundled schema inventory is incomplete")
    return MappingProxyType(result)


__all__ = [
    "EMBEDDED_SCHEMA_PATHS",
    "EmbeddedSchema",
    "EmbeddedSchemaError",
    "embedded_schemas",
]

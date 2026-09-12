"""Deterministic, complexity-bounded Gemini projection; local validation stays strict.

Reference: https://ai.google.dev/gemini-api/docs/structured-output
Unsupported local constraints and provider-complexity multipliers are removed
deliberately. An unfamiliar keyword fails before a provider request instead of
silently weakening a contract.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

SUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "type",
        "title",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "format",
        "minimum",
        "maximum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "anyOf",
    }
)
_LOCAL_ONLY_KEYS = frozenset(
    {"default", "minLength", "maxLength", "minItems", "maxItems", "pattern", "title"}
)


def gemini_transport_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Project schema locations only, never dictionary field names or enum data.

    Gemini rejects schemas whose state space is too complex.  The complete
    Pydantic schema remains hash-bound and is always enforced after generation;
    the transport projection keeps the object shape while rendering enum values
    as local descriptions instead of a combinatorial provider constraint. Array
    bounds also stay local: nested bounded arrays can exceed the provider's
    grammar complexity even after enum projection.
    """

    def project(node: dict[str, Any]) -> dict[str, Any]:
        if set(node) - SUPPORTED_SCHEMA_KEYS - _LOCAL_ONLY_KEYS:
            raise ValueError("Gemini transport schema contains an unsupported keyword")
        enum_values = node.get("enum")
        if enum_values is not None and (
            not isinstance(enum_values, list)
            or not enum_values
            or any(not isinstance(value, (str, int, float, bool)) for value in enum_values)
        ):
            raise ValueError("Gemini transport schema contains an invalid enum")
        nullable_types = _simple_nullable_types(node.get("anyOf"))
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in _LOCAL_ONLY_KEYS or key == "enum":
                continue
            if key == "anyOf" and nullable_types is not None:
                result["type"] = nullable_types
                continue
            if key in {"$defs", "properties"}:
                if not isinstance(value, dict) or any(
                    not isinstance(child, dict) for child in value.values()
                ):
                    raise ValueError("Gemini transport schema has an invalid schema map")
                result[key] = {name: project(child) for name, child in value.items()}
            elif key in {"items", "additionalProperties"} and isinstance(value, dict):
                result[key] = project(value)
            elif key in {"anyOf", "prefixItems"}:
                if not isinstance(value, list) or any(
                    not isinstance(child, dict) for child in value
                ):
                    raise ValueError("Gemini transport schema has an invalid schema array")
                result[key] = [project(child) for child in value]
            else:
                result[key] = deepcopy(value)
        if enum_values is not None:
            rendered = ", ".join(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                for value in enum_values
            )
            prefix = result.get("description")
            result["description"] = (
                f"{prefix} Allowed values: {rendered}."
                if isinstance(prefix, str) and prefix
                else f"Allowed values: {rendered}."
            )
        return result

    return project(schema)


def _simple_nullable_types(value: object) -> list[str] | None:
    """Use the nullable form documented by Gemini instead of a two-branch anyOf."""

    if not isinstance(value, list) or len(value) != 2:
        return None
    types: list[str] = []
    for branch in value:
        if not isinstance(branch, dict) or set(branch) - {"type"} - _LOCAL_ONLY_KEYS:
            return None
        branch_type = branch.get("type")
        if not isinstance(branch_type, str) or branch_type in types:
            return None
        types.append(branch_type)
    return types if "null" in types else None

"""Strict OpenAI schema projection; the full local contract still validates results."""

from copy import deepcopy
from typing import Any

_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "anyOf",
        "items",
        "description",
        "title",
        "default",
        "format",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
    }
)


def openai_transport_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Preserve enums and bounds, require every property, and reject unfamiliar schema keys.

    Reference: https://developers.openai.com/api/docs/guides/structured-outputs
    Defaults are local-only. Nullable fields stay nullable; nonnullable defaulted
    arrays must be returned explicitly rather than being silently omitted.
    """

    def project(node: dict[str, Any]) -> dict[str, Any]:
        if set(node) - _KEYS:
            raise ValueError("OpenAI transport schema contains an unsupported keyword")
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"default", "title"}:
                continue
            if key in {"$defs", "properties"}:
                if not isinstance(value, dict) or any(
                    not isinstance(child, dict) for child in value.values()
                ):
                    raise ValueError("OpenAI transport schema has an invalid schema map")
                result[key] = {name: project(child) for name, child in value.items()}
            elif key == "items":
                if not isinstance(value, dict):
                    raise ValueError("OpenAI transport schema has invalid items")
                result[key] = project(value)
            elif key == "anyOf":
                if not isinstance(value, list) or any(
                    not isinstance(child, dict) for child in value
                ):
                    raise ValueError("OpenAI transport schema has an invalid schema array")
                result[key] = [project(child) for child in value]
            else:
                result[key] = deepcopy(value)
        if "properties" in result:
            if result.get("type") != "object" or result.get("additionalProperties") is not False:
                raise ValueError("OpenAI transport objects must forbid additional properties")
            result["required"] = list(result["properties"])
        return result

    projected = project(schema)
    if projected.get("type") != "object" or "anyOf" in projected:
        raise ValueError("OpenAI transport root must be an object")
    return projected

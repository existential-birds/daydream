"""Shared construction for strict structured-output object schemas."""

from collections.abc import Sequence
from typing import Any

from daydream.severity import model_facing_levels


def text_schema(
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    nullable: bool = False,
    **constraints: Any,
) -> dict[str, Any]:
    """Construct a fresh string schema, retaining explicit JSON Schema constraints."""
    schema: dict[str, Any] = {"type": ["string", "null"] if nullable else "string"}
    if min_length is not None:
        schema["minLength"] = min_length
    if max_length is not None:
        schema["maxLength"] = max_length
    return {**schema, **constraints}


def array_schema(items: dict[str, Any], **constraints: Any) -> dict[str, Any]:
    """Construct an array without changing the item schema or its constraints."""
    return {"type": "array", "items": items, **constraints}


def enum_schema(values: Sequence[Any], *, nullable: bool = False, **constraints: Any) -> dict[str, Any]:
    """Construct a string enum; callers explicitly include None for nullable enums."""
    return {"type": ["string", "null"] if nullable else "string", "enum": list(values), **constraints}


def severity_enum_schema(*, nullable: bool = False) -> dict[str, Any]:
    """Build a fresh model-facing severity enum schema."""
    enum_schema: dict[str, Any] = {"type": "string", "enum": list(model_facing_levels())}
    if nullable:
        return {"anyOf": [enum_schema, {"type": "null"}]}
    return enum_schema


def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """Require every declared property and reject undeclared object keys.

    Nested schemas remain explicit at call sites; this does not recursively
    change nullable fields, array items, or intentionally permissive objects.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def result_array_schema(label: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Wrap strict records in a named result array with a closed outer object."""
    return strict_object({label: {"type": "array", "items": strict_object(properties)}})


def record_array_schema(properties: dict[str, Any], **constraints: Any) -> dict[str, Any]:
    """Build an array of closed, fully required records."""
    return array_schema(strict_object(properties), **constraints)

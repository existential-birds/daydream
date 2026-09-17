"""Shared construction for strict structured-output object schemas."""

from typing import Any

from daydream.severity import model_facing_levels


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

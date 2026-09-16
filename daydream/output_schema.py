"""Shared construction for strict structured-output object schemas."""

from typing import Any


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

from daydream import severity
from daydream.output_schema import severity_enum_schema


def test_severity_enum_schema_is_built_from_the_declaration() -> None:
    assert severity_enum_schema() == {"type": "string", "enum": ["high", "medium", "low"]}
    assert severity_enum_schema(nullable=True) == {
        "anyOf": [{"type": "string", "enum": ["high", "medium", "low"]}, {"type": "null"}]
    }


def test_severity_enum_schema_returns_a_fresh_object_per_call() -> None:
    first, second = severity_enum_schema(), severity_enum_schema()
    assert first is not second and first["enum"] is not second["enum"]
    first["enum"].append("critical")
    assert second["enum"] == ["high", "medium", "low"]


def test_severity_enum_schema_follows_a_moved_declaration() -> None:
    original = severity.CANONICAL_LEVELS
    try:
        severity.CANONICAL_LEVELS = ("high", "medium", "low", "critical")
        assert severity_enum_schema()["enum"] == ["critical", "low", "medium", "high"]
    finally:
        severity.CANONICAL_LEVELS = original

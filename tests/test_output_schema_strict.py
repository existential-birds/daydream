"""Codex strict-schema conformance at each schema owner and generated provenance shape.

Unlike Claude's lenient SDK, Codex requires all object properties in required
and additionalProperties=false. Representative roots cover nested objects, arrays, references and nullable variants;
private sub-schemas are checked recursively.
"""

from __future__ import annotations

from typing import Any

import pytest

import daydream.deep.diagram_schema as diagram_schema
import daydream.improve.prompts as improve_prompts
import daydream.improve.recon as improve_recon
import daydream.phases as phases
import daydream.prompts.exploration_subagents as exploration_subagents
from daydream.improve.audit import _schema_with_provenance

# OpenAI Structured Outputs accepts only this documented JSON Schema subset.
# Keep the supported keyword set beside the representative owner schemas.
_SUPPORTED_SCHEMA_KEYWORDS = {
    "$defs", "$ref", "additionalProperties", "anyOf", "description", "enum", "exclusiveMaximum", "exclusiveMinimum",
    "format", "items", "maximum", "maxItems", "maxLength", "minimum", "minItems", "minLength", "multipleOf", "pattern",
    "properties", "required", "type",
}


def _strict_violations(node: Any, path: str) -> list[str]:
    """Recursively collect OpenAI strict-mode violations under ``node``."""
    supported_types = {"string", "number", "boolean", "integer", "object", "array", "null"}
    errors: list[str] = []
    if not isinstance(node, dict):
        return [f"{path}: schema node is not an object"]
    if not node:
        return [f"{path}: unconstrained schema is not supported"]
    unsupported = set(node) - _SUPPORTED_SCHEMA_KEYWORDS
    if unsupported:
        errors.append(f"{path}: unsupported keywords: {sorted(unsupported)}")
    declared_type = node.get("type")
    declared_types = ([declared_type] if isinstance(declared_type, str) else declared_type)
    if declared_types is not None and (not isinstance(declared_types, list)
        or not declared_types
        or any(item not in supported_types for item in declared_types)
    ):
        errors.append(f"{path}: unsupported type declaration {declared_type!r}")

    if node.get("type") == "object" or "properties" in node:
        properties = node.get("properties", {})
        props = set(properties)
        required = set(node.get("required", []))
        missing = props - required
        if missing:
            errors.append(f"{path}: properties not in 'required': {sorted(missing)}")
        extra = required - props
        if extra:
            errors.append(f"{path}: 'required' names absent from properties: {sorted(extra)}")
        if node.get("additionalProperties", True) is not False:
            errors.append(f"{path}: missing 'additionalProperties': false")
        for name, schema in properties.items():
            errors.extend(_strict_violations(schema, f"{path}.properties.{name}"))
    if "items" in node:
        errors.extend(_strict_violations(node["items"], f"{path}.items"))
    for index, schema in enumerate(node.get("anyOf", [])):
        errors.extend(_strict_violations(schema, f"{path}.anyOf[{index}]"))
    for name, schema in node.get("$defs", {}).items():
        errors.extend(_strict_violations(schema, f"{path}.$defs.{name}"))
    return errors


# A representative root per owning module plus both provenance array families.
_SCHEMAS = [
    ("phases.records", phases.PER_STACK_RECORD_SCHEMA),
    ("exploration.patterns", exploration_subagents.PATTERN_SCANNER_SCHEMA),
    ("improve.plan", improve_prompts.PLAN_AUTHOR_SCHEMA),
    ("improve.recon", improve_recon.RECON_SCHEMA),
    ("diagram.sequence", diagram_schema.SEQUENCE_SPEC_SCHEMA),
    ("provenance.findings", _schema_with_provenance(improve_prompts.AUDIT_FINDINGS_SCHEMA)),
    ("provenance.verdicts", _schema_with_provenance(improve_prompts.VET_SCHEMA)),
]


@pytest.mark.parametrize("name,schema", _SCHEMAS, ids=[n for n, _ in _SCHEMAS])
def test_output_schema_is_codex_strict_compliant(name: str, schema: dict[str, Any]) -> None:
    """Representative owner schemas satisfy OpenAI strict structured-output rules."""
    violations = _strict_violations(schema, name)
    if schema.get("type") != "object":
        violations.append(f"{name}: root schema must be an object")
    assert not violations, "Schema not Codex/OpenAI strict-compliant:\n" + "\n".join(violations)

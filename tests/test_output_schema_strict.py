"""Codex/OpenAI strict structured-output conformance for every output schema.

The Codex backend forwards each ``output_schema`` to the ``codex`` CLI, which
hands it to OpenAI's Responses API in strict mode. That validator rejects any
schema whose object nodes do not (a) list *every* property key in ``required``
and (b) set ``additionalProperties: false``. The Claude SDK is lenient and
accepts partial-``required`` schemas, so a non-conforming schema passes the
Claude path silently and only fails on the real Codex path with::

    invalid_request_error / invalid_json_schema:
    'required' is required ... including every key in properties. Missing 'source'.

This is exactly how ``MERGED_ITEMS_SCHEMA`` (vestigial ``source`` not in
``required``) and ``RECOMMENDATION_VERDICTS_SCHEMA`` (missing
``additionalProperties``) shipped broken-for-Codex while green on Claude.

This test collects the static and generated schemas passed at the
``output_schema=`` call sites and asserts each conforms recursively.
"""

from __future__ import annotations

from typing import Any

import pytest

import daydream.improve.orchestrator as improve_orchestrator
import daydream.improve.prompts as improve_prompts
import daydream.phases as phases
import daydream.prompts.exploration_subagents as exploration_subagents

# Modules whose public ``*_SCHEMA`` constants are all backend output schemas.
_SCHEMA_MODULES = [
    phases,
    exploration_subagents,
]

# OpenAI Structured Outputs accepts only this documented JSON Schema subset.
# Keep this positive list beside the cross-provider schema inventory so every
# runtime schema is checked against the same provider boundary.
_SUPPORTED_SCHEMA_KEYWORDS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "description",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maximum",
    "maxItems",
    "maxLength",
    "minimum",
    "minItems",
    "minLength",
    "multipleOf",
    "pattern",
    "properties",
    "required",
    "type",
}


def _collect_output_schemas() -> list[tuple[str, dict[str, Any]]]:
    """Return every static or generated schema passed to a backend."""
    found: list[tuple[str, dict[str, Any]]] = []
    for module in _SCHEMA_MODULES:
        for name in dir(module):
            if name.startswith("_") or not name.endswith("_SCHEMA"):
                continue
            value = getattr(module, name)
            if isinstance(value, dict):
                found.append((f"{module.__name__}.{name}", value))
    found.extend(
        [
            ("daydream.improve.orchestrator._RECON_SCHEMA", improve_orchestrator._RECON_SCHEMA),
            (
                "daydream.improve.orchestrator._PLAN_REVIEW_SCHEMA",
                improve_orchestrator._PLAN_REVIEW_SCHEMA,
            ),
            ("daydream.improve.prompts.AUDIT_FINDINGS_SCHEMA", improve_prompts.AUDIT_FINDINGS_SCHEMA),
            ("daydream.improve.prompts.VET_SCHEMA", improve_prompts.VET_SCHEMA),
            ("daydream.improve.prompts.PLAN_WRITER_SCHEMA", improve_prompts.PLAN_WRITER_SCHEMA),
            (
                "daydream.improve.orchestrator.branch_audit_schema",
                improve_orchestrator._schema_with_provenance(
                    improve_prompts.AUDIT_FINDINGS_SCHEMA,
                ),
            ),
            (
                "daydream.improve.orchestrator.branch_vet_schema",
                improve_orchestrator._schema_with_provenance(
                    improve_prompts.VET_SCHEMA,
                ),
            ),
        ]
    )
    return found


def _strict_violations(node: Any, path: str) -> list[str]:
    """Recursively collect OpenAI strict-mode violations under ``node``."""
    supported_types = {
        "string",
        "number",
        "boolean",
        "integer",
        "object",
        "array",
        "null",
    }
    errors: list[str] = []
    if not isinstance(node, dict):
        return [f"{path}: schema node is not an object"]
    if not node:
        return [f"{path}: unconstrained schema is not supported"]
    unsupported = set(node) - _SUPPORTED_SCHEMA_KEYWORDS
    if unsupported:
        errors.append(f"{path}: unsupported keywords: {sorted(unsupported)}")
    declared_type = node.get("type")
    declared_types = (
        [declared_type] if isinstance(declared_type, str) else declared_type
    )
    if declared_types is not None and (
        not isinstance(declared_types, list)
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


_SCHEMAS = _collect_output_schemas()


def test_output_schemas_were_discovered() -> None:
    """Guard the guard: discovery must not silently match zero schemas."""
    assert _SCHEMAS, "No *_SCHEMA constants discovered — schema collection is broken"


@pytest.mark.parametrize("name,schema", _SCHEMAS, ids=[n for n, _ in _SCHEMAS])
def test_output_schema_is_codex_strict_compliant(name: str, schema: dict[str, Any]) -> None:
    """Every output schema satisfies OpenAI strict structured-output rules."""
    violations = _strict_violations(schema, name)
    if schema.get("type") != "object":
        violations.append(f"{name}: root schema must be an object")
    assert not violations, "Schema not Codex/OpenAI strict-compliant:\n" + "\n".join(violations)

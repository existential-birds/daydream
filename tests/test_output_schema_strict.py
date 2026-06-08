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

This test collects *every* module-level ``*_SCHEMA`` from the two modules that
define agent output schemas — the same objects passed at the ``output_schema=``
call sites — and asserts each conforms recursively. Introspecting by name (not
an explicit list) means a newly added output schema is covered automatically:
add a non-conforming schema and this test fails before it can reach Codex.
"""

from __future__ import annotations

from typing import Any

import pytest

import daydream.phases as phases
import daydream.prompts.exploration_subagents as exploration_subagents

# Modules whose ``*_SCHEMA`` constants are all passed to a backend as
# ``output_schema`` (verified against every ``output_schema=`` call site).
_SCHEMA_MODULES = [phases, exploration_subagents]


def _collect_output_schemas() -> list[tuple[str, dict[str, Any]]]:
    """Return ``(qualified_name, schema)`` for every module-level ``*_SCHEMA``."""
    found: list[tuple[str, dict[str, Any]]] = []
    for module in _SCHEMA_MODULES:
        for name in dir(module):
            if not name.endswith("_SCHEMA"):
                continue
            value = getattr(module, name)
            if isinstance(value, dict):
                found.append((f"{module.__name__}.{name}", value))
    return found


def _strict_violations(node: Any, path: str) -> list[str]:
    """Recursively collect OpenAI strict-mode violations under ``node``."""
    errors: list[str] = []
    if isinstance(node, dict):
        is_object = node.get("type") == "object" or "properties" in node
        if is_object:
            props = set(node.get("properties", {}))
            required = set(node.get("required", []))
            missing = props - required
            if missing:
                errors.append(f"{path}: properties not in 'required': {sorted(missing)}")
            extra = required - props
            if extra:
                errors.append(f"{path}: 'required' names absent from properties: {sorted(extra)}")
            if node.get("additionalProperties", True) is not False:
                errors.append(f"{path}: missing 'additionalProperties': false")
        for key, value in node.items():
            errors.extend(_strict_violations(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            errors.extend(_strict_violations(value, f"{path}[{index}]"))
    return errors


_SCHEMAS = _collect_output_schemas()


def test_output_schemas_were_discovered() -> None:
    """Guard the guard: discovery must not silently match zero schemas."""
    assert _SCHEMAS, "No *_SCHEMA constants discovered — schema collection is broken"


@pytest.mark.parametrize("name,schema", _SCHEMAS, ids=[n for n, _ in _SCHEMAS])
def test_output_schema_is_codex_strict_compliant(name: str, schema: dict[str, Any]) -> None:
    """Every output schema satisfies OpenAI strict structured-output rules."""
    violations = _strict_violations(schema, name)
    assert not violations, "Schema not Codex/OpenAI strict-compliant:\n" + "\n".join(violations)

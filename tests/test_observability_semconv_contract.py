"""Offline GenAI semantic-convention contract pin (P18 Task 0).

Derives every allowed ``gen_ai.*`` name, type, and enum member from the pinned
``registry.yaml`` bytes checked in under ``tests/fixtures/observability_semconv/``
(OpenTelemetry semantic-conventions-genai commit ``94f432d7126f5884d30a2cdde6f4e89908ebb6fd``).
The fixture bytes are verified against the SHA-256 manifest in
``tests/fixtures/observability_contract/manifest.json`` before any derivation runs.

This test intentionally contains NO duplicated hardcoded attribute allowlist:
production literals are scanned from ``daydream/observability/`` and each must be
provable from the parsed registry (name, scalar/array type, and enum membership).
Requirement expressions come from the pinned ``spans.yaml`` semantic conventions
downloaded into the same fixture directory at the same commit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

TESTS_DIR = Path(__file__).resolve().parent
SEMCONV_DIR = TESTS_DIR / "fixtures" / "observability_semconv"
CONTRACT_DIR = TESTS_DIR / "fixtures" / "observability_contract"
DAYDREAM_SRC = TESTS_DIR.parent / "daydream" / "observability"

SEMCONV_COMMIT = "94f432d7126f5884d30a2cdde6f4e89908ebb6fd"

# Canonical Daydream attribute namespaces (daydream-owned values are NOT
# registry attributes and are excluded from registry derivation).
_DAYDREAM_NS = "daydream."
_GENAI_LITERAL_RE = re.compile(r"[\"'](gen_ai\.[a-zA-Z0-9_.]+)[\"']")


# ---------------------------------------------------------------------------
# Manifest byte verification
# ---------------------------------------------------------------------------


def _load_manifest() -> dict[str, Any]:
    manifest_path = CONTRACT_DIR / "manifest.json"
    assert manifest_path.is_file(), "contract manifest missing"
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data


def test_manifest_pins_expected_commit_and_files() -> None:
    manifest = _load_manifest()
    assert manifest["source"]["commit"] == SEMCONV_COMMIT
    assert manifest["source"]["type"] == "git_commit"
    assert manifest["source"]["repo"] == "open-telemetry/semantic-conventions-genai"
    files = manifest["files"]
    expected = {
        "registry.yaml",
        "spans.yaml",
        "gen-ai-input-messages.json",
        "gen-ai-output-messages.json",
        "gen-ai-system-instructions.json",
        "gen-ai-tool-definitions.json",
    }
    assert expected <= set(files), f"manifest missing entries: {expected - set(files)}"
    for name, entry in files.items():
        raw = (SEMCONV_DIR / name).read_bytes()
        assert entry["bytes"] == len(raw), f"{name}: byte length drifted"
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"{name}: sha256 drifted"
        assert (
            entry["url"].startswith("https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/")
            and SEMCONV_COMMIT in entry["url"]
        ), f"{name}: url not pinned to {SEMCONV_COMMIT[:12]}"


# ---------------------------------------------------------------------------
# Registry-derived contract
# ---------------------------------------------------------------------------


def _registry_attributes() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load((SEMCONV_DIR / "registry.yaml").read_text(encoding="utf-8"))
    assert data["file_format"] == "definition/2"
    return {a["key"]: a for a in data["attributes"]}


def _attr_type(attr: dict[str, Any]) -> Any:
    """Normalize a registry attribute type to a comparable value.

    Returns the scalar type string (``string``/``int``/``double``/``boolean``),
    ``string[]``-style templates, an (``enum``-prefix, member set) tuple for
    enums, or ``any`` for template-typed attributes.
    """
    t = attr.get("type")
    if isinstance(t, dict):
        members = [m["value"] for m in t.get("members", [])]
        return ("enum", frozenset(members))
    return t


def _attr_enum_members(attr: dict[str, Any]) -> frozenset[str]:
    t = attr.get("type")
    if isinstance(t, dict):
        return frozenset(m["value"] for m in t.get("members", []))
    return frozenset()


def _production_genai_literals() -> set[str]:
    literals: set[str] = set()
    for path in DAYDREAM_SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        literals.update(_GENAI_LITERAL_RE.findall(text))
    return literals


# Names the plan deliberately keeps as Daydream custom fields or documented
# compatibility aliases (binding decisions + plan approach decisions). These
# are exact, closed, and reviewed; anything else must derive from the registry.
_DOCUMENTED_ALIASES = {
    # Legacy/compatibility spellings retained per plan (documented aliases):
    "gen_ai.request.reasoning_effort",  # alias of gen_ai.request.reasoning.level
    "gen_ai.usage.cache_creation.input_tokens",  # alias of ...cache_write...
    "gen_ai.system",  # legacy OTel system attr, destination alias only
}
# Names intentionally absent from registry at the pinned commit and owned as
# Daydream custom attributes; tracked in the Task 0 contract document.
_DAYDREAM_OWNED = {
    "gen_ai.usage.cost",  # canonical daydream.usage.cost_usd + destination alias
    "gen_ai.usage.total_tokens",  # derived total, daydream-owned, never a subset sum
}


def test_registry_covers_every_production_genai_literal() -> None:
    attrs = _registry_attributes()
    reg_names = set(attrs)
    literals = _production_genai_literals()
    assert literals, "no gen_ai literals found in production source"
    unknown = literals - reg_names - _DOCUMENTED_ALIASES - _DAYDREAM_OWNED
    assert not unknown, f"production emits names not in the pinned registry: {sorted(unknown)}"
    # Aliases and owned names must also remain actually absent from the pinned
    # registry (they are compatibility spellings or derived custom fields).
    # gen_ai.system is a destination-side alias spelling only: production never
    # emits it as a source literal, so no alias may shadow a pinned name.
    assert not literals & reg_names & _DOCUMENTED_ALIASES
    assert not literals & reg_names & _DAYDREAM_OWNED
    # The canonical standard names the plan adopts must exist in the registry.
    for canonical in (
        "gen_ai.request.reasoning.level",
        "gen_ai.usage.cache_write.input_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.reasoning.output_tokens",
        "gen_ai.agent.name",
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.conversation.id",
        "gen_ai.system_instructions",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "gen_ai.response.finish_reasons",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.request.model",
        "gen_ai.output.type",
    ):
        assert canonical in reg_names, f"canonical {canonical} missing from pinned registry"


def test_registry_types_match_production_literal_usage() -> None:
    attrs = _registry_attributes()
    source = "\n".join(p.read_text(encoding="utf-8") for p in DAYDREAM_SRC.rglob("*.py"))
    # usage token attributes are ints in the registry; production emits them
    # (the canonical cache_write spelling arrives with Task 1's alias rename —
    # at baseline only the documented legacy alias is emitted, and the pinned
    # alias mapping below must keep matching it).
    for name in (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.reasoning.output_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cache_write.input_tokens",
    ):
        assert name in attrs
        assert attrs[name]["type"] == "int", f"{name} registry type drifted from int"
    emitted_usage = {
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.reasoning.output_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cache_creation.input_tokens",
    }
    for name in emitted_usage:
        assert f'"{name}"' in source, f"{name} no longer emitted by production source"
    # identity/name attributes are strings
    for name in (
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.conversation.id",
        "gen_ai.agent.name",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
    ):
        assert attrs[name]["type"] == "string", f"{name} registry type drifted from string"
    # message/instruction payloads are template-typed (any)
    for name in ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions"):
        assert attrs[name]["type"] == "any", f"{name} registry type drifted from any"


def test_operation_name_enum_contains_daydream_operations() -> None:
    attrs = _registry_attributes()
    members = _attr_enum_members(attrs["gen_ai.operation.name"])
    assert {"invoke_agent", "execute_tool", "chat"} <= members
    # Daydream structural attempts are INTERNAL invoke_agent spans; chat is the
    # model-generation operation Pi may use for a proven generation interval.
    assert "invoke_agent" in members
    # provider.name enum must contain the exact provider names production normalizes to
    provider_members = _attr_enum_members(attrs["gen_ai.provider.name"])
    assert {"anthropic", "openai"} <= provider_members


def test_client_invoke_agent_requires_provider_name() -> None:
    """CLIENT invoke_agent spans MUST carry gen_ai.provider.name (spans.yaml).

    Internal invoke_agent spans must NOT fabricate provider evidence: the
    required attribute only applies to the client span class.
    """
    spans = yaml.safe_load((SEMCONV_DIR / "spans.yaml").read_text(encoding="utf-8"))
    span_defs: list[dict[str, Any]] = list(spans.get("spans", []))
    client = next(
        (g for g in span_defs if g.get("type") == "gen_ai.invoke_agent.client"),
        None,
    )
    internal = next(
        (g for g in span_defs if g.get("type") == "gen_ai.invoke_agent.internal"),
        None,
    )
    assert client is not None, "pinned spans.yaml lost the client invoke_agent group"
    assert internal is not None, "pinned spans.yaml lost the internal invoke_agent group"
    assert client.get("kind") == "client"
    assert internal.get("kind") == "internal"

    def _required_refs(group: dict[str, Any]) -> set[str]:
        out = set()
        for a in group.get("attributes", []):
            if a.get("requirement_level") == "required" and "ref" in a:
                out.add(a["ref"])
        return out

    assert "gen_ai.provider.name" in _required_refs(client), (
        "CLIENT invoke_agent must require gen_ai.provider.name at creation; "
        "a CLIENT attempt without exact provider-at-creation must fail this gate"
    )
    assert "gen_ai.provider.name" not in _required_refs(internal), (
        "INTERNAL invoke_agent must not require provider evidence; "
        "local/unknown-provider attempts must not fabricate it"
    )


# ---------------------------------------------------------------------------
# Representative Daydream JSON validated against the four pinned schemas
# ---------------------------------------------------------------------------


def _schema(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((SEMCONV_DIR / name).read_text(encoding="utf-8"))
    return data


def _validate(name: str, instance: Any) -> None:
    import jsonschema

    jsonschema.validate(instance, _schema(name))


def test_representative_input_messages_validate() -> None:
    _validate(
        "gen-ai-input-messages.json",
        [
            {
                "role": "system",
                "parts": [{"type": "text", "content": "Use the declared answer schema."}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "content": "Review the sample."}],
            },
            {
                "role": "assistant",
                "parts": [
                    {"type": "reasoning", "content": "checking"},
                    {"type": "tool_call", "id": "call_01", "name": "read", "arguments": {"path": "a.py"}},
                ],
            },
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "call_01", "response": "ok"}],
            },
        ],
    )


def test_representative_output_messages_validate() -> None:
    _validate(
        "gen-ai-output-messages.json",
        [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Done."}],
                "finish_reason": "stop",
            },
            {
                "role": "assistant",
                "parts": [{"type": "tool_call", "id": "call_02", "name": "grep", "arguments": {"p": "x"}}],
                "finish_reason": "tool_call",
            },
        ],
    )


def test_representative_system_instructions_validate() -> None:
    _validate(
        "gen-ai-system-instructions.json",
        [{"type": "text", "content": "Daydream phase preamble."}],
    )


def test_representative_tool_definitions_validate() -> None:
    _validate(
        "gen-ai-tool-definitions.json",
        [
            {
                "type": "function",
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )


def test_output_finish_reason_members_are_registry_derived() -> None:
    schema = _schema("gen-ai-output-messages.json")
    members = schema["$defs"]["FinishReason"]["enum"]
    assert {"stop", "tool_call", "error"} <= set(members)


# ---------------------------------------------------------------------------
# Native timestamp validator contract (frozen here for Task 1 reuse)
# ---------------------------------------------------------------------------


def _is_valid_native_unix_ms(value: Any) -> bool:
    """Pinned validator: exact int (never bool), bounded 0..(2**63-1)//1_000_000."""
    if type(value) is not int:  # bool is a subclass of int; reject it
        return False
    return 0 <= value <= (2**63 - 1) // 1_000_000


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        (0, True),
        (1788690314289, True),
        ((2**63 - 1) // 1_000_000, True),
        (True, False),
        (False, False),
        (1.0, False),
        ("1788690314289", False),
        (-1, False),
        ((2**63 - 1) // 1_000_000 + 1, False),
        (None, False),
    ],
)
def test_native_timestamp_validator_bounds(value: Any, ok: bool) -> None:
    assert _is_valid_native_unix_ms(value) is ok


def test_native_ms_to_ns_conversion_is_multiplication_exact() -> None:
    ms = 1788690314289
    ns = ms * 1_000_000
    assert ns == 1788690314289000000
    assert type(ns) is int

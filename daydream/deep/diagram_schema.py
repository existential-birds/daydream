"""Strict diagram schemas and tolerant per-entry coercion.

Models return evidence-bearing specs; deterministic grounding and rendering own
mermaid output. Every object requires all its properties and rejects extras;
optional values are required-and-nullable. Sub-schemas stay private so strict
schema discovery only treats complete specs as roots.

Label length limits belong to renderer sanitization, not maxLength constraints.
Coercion drops malformed entries individually and remaps branch message indices
after drops. Non-object inputs produce empty specs; the empty flowchart's None
root intentionally fails the schema and cannot ground as a real function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from daydream.deep.diagram_types import (
    BLOCK_KINDS,
    MESSAGE_KINDS,
    NODE_KINDS,
    PARTICIPANT_KINDS,
)
from daydream.output_schema import array_schema, record_array_schema, strict_object
from daydream.repository_paths import (
    REPOSITORY_FILE_PATH_SCHEMA as _REPOSITORY_FILE_PATH_SCHEMA,
    valid_repository_file_path,
)

# 1-based source line. ``minimum`` keeps a nonsense 0 out of the spec before
# grounding has to spend a LINE_OUT_OF_RANGE on it.
_LINE_SCHEMA: dict[str, Any] = {"type": "integer", "minimum": 1}

# Branch/loop evidence: a location only, no symbol to check.
_EVIDENCE_SCHEMA: dict[str, Any] = strict_object(
    {
        "file": _REPOSITORY_FILE_PATH_SCHEMA,
        "line": _LINE_SCHEMA,
    }
)

# Sequence-message evidence: ``symbol`` is the callee (call), the enclosing
# function (reply) or the client method token (external call), and is always
# required -- SYMBOL_NOT_ON_LINE is the check that makes a message verifiable.
_SYMBOL_EVIDENCE_SCHEMA: dict[str, Any] = strict_object(
    {
        "file": _REPOSITORY_FILE_PATH_SCHEMA,
        "line": _LINE_SCHEMA,
        "symbol": {"type": "string"},
    }
)

# Flowchart-node evidence: only a ``subroutine`` node names a symbol, so the
# field is nullable rather than absent (strict mode has no optional keys).
_OPTIONAL_SYMBOL_EVIDENCE_SCHEMA: dict[str, Any] = strict_object(
    {
        "file": _REPOSITORY_FILE_PATH_SCHEMA,
        "line": _LINE_SCHEMA,
        "symbol": {"type": ["string", "null"]},
    }
)

SEQUENCE_SPEC_SCHEMA: dict[str, Any] = strict_object(
    {
        "participants": record_array_schema(
            {
                "name": {"type": "string"},
                "kind": {"type": "string", "enum": list(PARTICIPANT_KINDS)},
                "files": {"type": "array", "items": _REPOSITORY_FILE_PATH_SCHEMA},
                "service": {"type": ["string", "null"]},
            }
        ),
        "messages": record_array_schema(
            {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "label": {"type": "string"},
                "kind": {"type": "string", "enum": list(MESSAGE_KINDS)},
                "changed": {"type": "boolean"},
                "evidence": _SYMBOL_EVIDENCE_SCHEMA,
            }
        ),
        "blocks": record_array_schema(
            {
                "kind": {"type": "string", "enum": list(BLOCK_KINDS)},
                "branches": record_array_schema(
                    {
                        "condition": {"type": "string"},
                        "evidence": _EVIDENCE_SCHEMA,
                        "messages": array_schema({"type": "integer", "minimum": 0}),
                    }
                ),
            }
        ),
    }
)

FLOWCHART_SPEC_SCHEMA: dict[str, Any] = strict_object(
    {
        "root": strict_object(
            {
                "file": _REPOSITORY_FILE_PATH_SCHEMA,
                "name": {"type": "string"},
                "line": _LINE_SCHEMA,
            }
        ),
        "nodes": record_array_schema(
            {
                "id": {"type": "string"},
                "kind": {"type": "string", "enum": list(NODE_KINDS)},
                "label": {"type": "string"},
                "evidence": _OPTIONAL_SYMBOL_EVIDENCE_SCHEMA,
            }
        ),
        "edges": record_array_schema(
            {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "label": {"type": ["string", "null"]},
            }
        ),
    }
)


def _empty_sequence_spec() -> dict[str, Any]:
    """Return a fresh, schema-valid, empty sequence spec."""
    return {"participants": [], "messages": [], "blocks": []}


def _empty_flowchart_spec() -> dict[str, Any]:
    """Return the unusable-spec sentinel; its None root intentionally fails the schema."""
    return {"root": None, "nodes": [], "edges": []}


def _text(value: Any) -> str | None:
    """Return ``value`` as non-empty stripped text, or None when unusable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _choice(value: Any, allowed: tuple[str, ...]) -> str | None:
    """Return ``value`` when it is one of ``allowed``, else None."""
    text = _text(value)
    return text if text in allowed else None


def _index(value: Any, *, minimum: int) -> int | None:
    """Accept integers or digit strings at least minimum; reject booleans and other values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        return None
    return number if number >= minimum else None


def _path(value: Any) -> str | None:
    """Apply the shared lexical repository-path gate used by privileged schema validation."""
    text = _text(value)
    if text is None or not valid_repository_file_path(text):
        return None
    return text


def _paths(value: Any) -> list[str]:
    """Return the usable path strings in ``value`` (order preserved, deduped)."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for entry in value:
        path = _path(entry)
        if path is not None and path not in out:
            out.append(path)
    return out


def _evidence(value: Any, *, symbol: Literal["required", "optional", "absent"]) -> dict[str, Any] | None:
    """Coerce a location; require, retain nullable, or omit symbol as requested."""
    if not isinstance(value, dict):
        return None
    file = _path(value.get("file"))
    line = _index(value.get("line"), minimum=1)
    if file is None or line is None:
        return None
    evidence: dict[str, Any] = {"file": file, "line": line}
    if symbol == "absent":
        return evidence
    symbol_text = _text(value.get("symbol"))
    if symbol == "required" and symbol_text is None:
        return None
    evidence["symbol"] = symbol_text
    return evidence


def _record(
    value: Any,
    fields: Mapping[str, Callable[[Any], Any]],
    *,
    required: tuple[str, ...],
) -> dict[str, Any] | None:
    """Coerce one record; nullable fields stay present and required None fields reject it."""
    if not isinstance(value, dict):
        return None
    record = {name: coerce(value.get(name)) for name, coerce in fields.items()}
    return record if all(record[name] is not None for name in required) else None


def _records(
    value: Any,
    fields: Mapping[str, Callable[[Any], Any]],
    *,
    required: tuple[str, ...],
    unique: str | None = None,
) -> list[dict[str, Any]]:
    """Drop malformed entries and later duplicates while preserving source order."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in value if isinstance(value, list) else []:
        record = _record(entry, fields, required=required)
        if record is None or (unique is not None and record[unique] in seen):
            continue
        if unique is not None:
            seen.add(record[unique])
        out.append(record)
    return out


def _coerce_participants(value: Any) -> list[dict[str, Any]]:
    return _records(
        value,
        {
            "name": _text,
            "kind": lambda value: _choice(value, PARTICIPANT_KINDS),
            "files": _paths,
            "service": _text,
        },
        required=("name", "kind"),
        unique="name",
    )


def _coerce_messages(value: Any) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Keep original positions for branch-index remapping after invalid messages drop."""
    out: list[dict[str, Any]] = []
    remap: dict[int, int] = {}
    for position, entry in enumerate(value if isinstance(value, list) else []):
        message = _record(
            entry,
            {
                "from": _text,
                "to": _text,
                "label": _text,
                "kind": lambda value: _choice(value, MESSAGE_KINDS),
                "changed": bool,
                "evidence": lambda value: _evidence(value, symbol="required"),
            },
            required=("from", "to", "label", "kind", "evidence"),
        )
        if message is not None:
            remap[position] = len(out)
            out.append(message)
    return out, remap


def _coerce_branch(value: Any, *, remap: dict[int, int]) -> dict[str, Any] | None:
    """Coerce one block branch, remapping its message indices, or return None."""
    if not isinstance(value, dict):
        return None
    condition = _text(value.get("condition"))
    evidence = _evidence(value.get("evidence"), symbol="absent")
    if condition is None or evidence is None:
        return None
    raw_indices = value.get("messages")
    indices: list[int] = []
    if isinstance(raw_indices, list):
        for entry in raw_indices:
            original = _index(entry, minimum=0)
            if original is None:
                continue
            mapped = remap.get(original)
            if mapped is not None and mapped not in indices:
                indices.append(mapped)
    return {"condition": condition, "evidence": evidence, "messages": indices}


def _coerce_blocks(value: Any, *, remap: dict[int, int]) -> list[dict[str, Any]]:
    """Drop branchless blocks, leaving branch-count arity to the grounding report."""
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        kind = _choice(entry.get("kind"), BLOCK_KINDS)
        if kind is None:
            continue
        raw_branches = entry.get("branches")
        branches: list[dict[str, Any]] = []
        if isinstance(raw_branches, list):
            for raw_branch in raw_branches:
                branch = _coerce_branch(raw_branch, remap=remap)
                if branch is not None:
                    branches.append(branch)
        if not branches:
            continue
        out.append({"kind": kind, "branches": branches})
    return out


def coerce_sequence_spec(value: Any) -> dict[str, Any]:
    """Drop malformed entries and remap branch indices into a schema-valid sequence spec."""
    if not isinstance(value, dict):
        return _empty_sequence_spec()
    messages, remap = _coerce_messages(value.get("messages"))
    return {
        "participants": _coerce_participants(value.get("participants")),
        "messages": messages,
        "blocks": _coerce_blocks(value.get("blocks"), remap=remap),
    }


def _coerce_root(value: Any) -> dict[str, Any] | None:
    return _record(
        value,
        {
            "file": _path,
            "name": _text,
            "line": lambda value: _index(value, minimum=1),
        },
        required=("file", "name", "line"),
    )


def _coerce_nodes(value: Any) -> list[dict[str, Any]]:
    return _records(
        value,
        {
            "id": _text,
            "kind": lambda value: _choice(value, NODE_KINDS),
            "label": _text,
            "evidence": lambda value: _evidence(value, symbol="optional"),
        },
        required=("id", "kind", "label", "evidence"),
        unique="id",
    )


def _coerce_edges(value: Any) -> list[dict[str, Any]]:
    """Keep unknown node references for grounding to report."""
    return _records(
        value,
        {
            "from": _text,
            "to": _text,
            "label": _text,
        },
        required=("from", "to"),
    )


def coerce_flowchart_spec(value: Any) -> dict[str, Any]:
    """Drop malformed entries; an unusable root becomes the schema-invalid None sentinel."""
    if not isinstance(value, dict):
        return _empty_flowchart_spec()
    return {
        "root": _coerce_root(value.get("root")),
        "nodes": _coerce_nodes(value.get("nodes")),
        "edges": _coerce_edges(value.get("edges")),
    }


__all__ = [
    "FLOWCHART_SPEC_SCHEMA",
    "SEQUENCE_SPEC_SCHEMA",
    "coerce_flowchart_spec",
    "coerce_sequence_spec",
]

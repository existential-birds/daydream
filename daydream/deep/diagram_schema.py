"""Output schemas and coercion for the grounded-diagram specs (issue #1113).

The model never writes mermaid. For each eligible kind it returns a structured
spec whose every element carries ``file``/``line`` (and where relevant
``symbol``) evidence; the deterministic grounding pass verifies that evidence
against the head tree and the renderer emits mermaid from what survived. These
two schemas are the contract for that spec, and the two ``coerce_*`` functions
are the tolerant front door in front of them.

**Strict-mode encoding.** Both schemas are handed to backends as
``output_schema=``, so they must satisfy OpenAI strict structured outputs (see
``tests/test_output_schema_strict.py``, which discovers every public
module-level ``*_SCHEMA`` here): ``"type": "object"`` at the root, every
property key repeated in ``required``, and ``additionalProperties: false`` at
*every* object node including array ``items``. Optional-in-spirit fields are
therefore required-and-nullable (``{"type": ["string", "null"]}``), and the
shared sub-schemas are underscore-named so the discovery pass does not scan
them as roots of their own -- which is also why the repository-path schema is
imported under an underscore alias.

Deliberately *no* ``maxLength`` on labels. The spec's "<= 80 chars" message and
"<= 60 chars" node budgets are prompt guidance enforced by the renderer's
sanitizer, not schema constraints: strict mode does not constrain generation by
``maxLength``, so declaring it would only give ``jsonschema.validate`` a reason
to reject an otherwise well-grounded ``spec_final`` one CI job later.

Coercion is per-entry tolerant, in the style of
``exploration_runner._coerce_file_infos``: a malformed participant, message,
block, branch, node or edge is dropped on its own rather than failing the
whole spec, and a non-dict payload degrades to the empty spec. Dropping a
message renumbers the collection, so block branch indices are remapped onto
the surviving positions -- a coerced spec always keeps the "branch message
indices are 0-based indices into ``messages``" invariant.
"""

from __future__ import annotations

from typing import Any, Literal

from daydream.repository_paths import (
    REPOSITORY_FILE_PATH_SCHEMA as _REPOSITORY_FILE_PATH_SCHEMA,
)
from daydream.repository_paths import valid_repository_file_path

_PARTICIPANT_KINDS = ("internal", "external")
_MESSAGE_KINDS = ("call", "reply", "self")
_BLOCK_KINDS = ("alt", "opt", "loop")
_NODE_KINDS = ("start", "end", "process", "decision", "subroutine", "io")

# 1-based source line. ``minimum`` keeps a nonsense 0 out of the spec before
# grounding has to spend a LINE_OUT_OF_RANGE on it.
_LINE_SCHEMA: dict[str, Any] = {"type": "integer", "minimum": 1}

# Branch/loop evidence: a location only, no symbol to check.
_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": _REPOSITORY_FILE_PATH_SCHEMA,
        "line": _LINE_SCHEMA,
    },
    "required": ["file", "line"],
    "additionalProperties": False,
}

# Sequence-message evidence: ``symbol`` is the callee (call), the enclosing
# function (reply) or the client method token (external call), and is always
# required -- SYMBOL_NOT_ON_LINE is the check that makes a message verifiable.
_SYMBOL_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": _REPOSITORY_FILE_PATH_SCHEMA,
        "line": _LINE_SCHEMA,
        "symbol": {"type": "string"},
    },
    "required": ["file", "line", "symbol"],
    "additionalProperties": False,
}

# Flowchart-node evidence: only a ``subroutine`` node names a symbol, so the
# field is nullable rather than absent (strict mode has no optional keys).
_OPTIONAL_SYMBOL_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": _REPOSITORY_FILE_PATH_SCHEMA,
        "line": _LINE_SCHEMA,
        "symbol": {"type": ["string", "null"]},
    },
    "required": ["file", "line", "symbol"],
    "additionalProperties": False,
}

SEQUENCE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "participants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(_PARTICIPANT_KINDS)},
                    "files": {"type": "array", "items": _REPOSITORY_FILE_PATH_SCHEMA},
                    "service": {"type": ["string", "null"]},
                },
                "required": ["name", "kind", "files", "service"],
                "additionalProperties": False,
            },
        },
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": list(_MESSAGE_KINDS)},
                    "changed": {"type": "boolean"},
                    "evidence": _SYMBOL_EVIDENCE_SCHEMA,
                },
                "required": ["from", "to", "label", "kind", "changed", "evidence"],
                "additionalProperties": False,
            },
        },
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(_BLOCK_KINDS)},
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition": {"type": "string"},
                                "evidence": _EVIDENCE_SCHEMA,
                                "messages": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0},
                                },
                            },
                            "required": ["condition", "evidence", "messages"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["kind", "branches"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["participants", "messages", "blocks"],
    "additionalProperties": False,
}

FLOWCHART_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root": {
            "type": "object",
            "properties": {
                "file": _REPOSITORY_FILE_PATH_SCHEMA,
                "name": {"type": "string"},
                "line": _LINE_SCHEMA,
            },
            "required": ["file", "name", "line"],
            "additionalProperties": False,
        },
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": list(_NODE_KINDS)},
                    "label": {"type": "string"},
                    "evidence": _OPTIONAL_SYMBOL_EVIDENCE_SCHEMA,
                },
                "required": ["id", "kind", "label", "evidence"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "label": {"type": ["string", "null"]},
                },
                "required": ["from", "to", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["root", "nodes", "edges"],
    "additionalProperties": False,
}


def _empty_sequence_spec() -> dict[str, Any]:
    """Return a fresh, schema-valid, empty sequence spec."""
    return {"participants": [], "messages": [], "blocks": []}


def _empty_flowchart_spec() -> dict[str, Any]:
    """Return a fresh empty flowchart spec.

    ``root`` is ``None`` because there is no such thing as an empty root, which
    means this one sentinel deliberately does *not* validate against
    :data:`FLOWCHART_SPEC_SCHEMA` (whose ``root`` is a required object). That is
    the point: the diagram step branches on the empty spec to mean "the model
    returned nothing usable", and a ``None`` root cannot be mistaken for a real
    one by the grounding pass either.
    """
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
    """Return ``value`` as an int ``>= minimum``, else None.

    Accepts a digit string as well as an ``int`` (a model that quotes a line
    number should not lose an otherwise well-evidenced element). ``bool`` is
    rejected explicitly -- it is an ``int`` subclass and ``True`` is not a line.
    """
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
    """Return ``value`` as a lexically valid repository-relative path, else None.

    The same gate the other output schemas use, applied here so a coerced spec
    can never carry a path that would later fail the schema's ``pattern`` --
    ``spec_final`` is re-validated with ``jsonschema`` in the privileged
    findings-posting job, one CI job away from where it was produced.
    """
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
    """Coerce one evidence object, or None when it cannot be used.

    Args:
        value: Raw evidence payload from the model.
        symbol: ``"required"`` drops the evidence when no symbol text is
            present, ``"optional"`` keeps a ``None`` symbol, ``"absent"`` emits
            no ``symbol`` key at all (the branch/loop evidence shape).
    """
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


def _coerce_participants(value: Any) -> list[dict[str, Any]]:
    """Coerce the participant list, dropping malformed and duplicate entries."""
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            continue
        name = _text(entry.get("name"))
        kind = _choice(entry.get("kind"), _PARTICIPANT_KINDS)
        if name is None or kind is None or name in seen:
            continue
        seen.add(name)
        out.append(
            {
                "name": name,
                "kind": kind,
                "files": _paths(entry.get("files")),
                "service": _text(entry.get("service")),
            }
        )
    return out


def _coerce_messages(value: Any) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Coerce the message list.

    Returns:
        The surviving messages and a map from each entry's index in the raw
        list to its index in the coerced list, which is what lets block branch
        indices be remapped over a mid-list drop.
    """
    if not isinstance(value, list):
        return [], {}
    out: list[dict[str, Any]] = []
    remap: dict[int, int] = {}
    for position, entry in enumerate(value):
        if not isinstance(entry, dict):
            continue
        source = _text(entry.get("from"))
        target = _text(entry.get("to"))
        label = _text(entry.get("label"))
        kind = _choice(entry.get("kind"), _MESSAGE_KINDS)
        evidence = _evidence(entry.get("evidence"), symbol="required")
        if source is None or target is None or label is None or kind is None or evidence is None:
            continue
        remap[position] = len(out)
        out.append(
            {
                "from": source,
                "to": target,
                "label": label,
                "kind": kind,
                "changed": bool(entry.get("changed")),
                "evidence": evidence,
            }
        )
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
    """Coerce the block list, dropping blocks left without a single branch.

    Branch-count arity (``alt`` needs >= 2 branches, ``opt``/``loop`` exactly
    one) is a grounding rule, not a shape rule: JSON Schema cannot express it
    per variant and coercion must not silently discard a block the grounding
    report should be reporting on.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        kind = _choice(entry.get("kind"), _BLOCK_KINDS)
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
    """Coerce a raw model payload into a sequence spec.

    Malformed participants, messages, blocks and branches are dropped
    individually; a non-dict payload degrades to the empty spec
    (``{"participants": [], "messages": [], "blocks": []}``). Block branch
    message indices are remapped onto the surviving messages, so the result
    always satisfies :data:`SEQUENCE_SPEC_SCHEMA`.
    """
    if not isinstance(value, dict):
        return _empty_sequence_spec()
    messages, remap = _coerce_messages(value.get("messages"))
    return {
        "participants": _coerce_participants(value.get("participants")),
        "messages": messages,
        "blocks": _coerce_blocks(value.get("blocks"), remap=remap),
    }


def _coerce_root(value: Any) -> dict[str, Any] | None:
    """Coerce the flowchart root, or return None when it is unusable."""
    if not isinstance(value, dict):
        return None
    file = _path(value.get("file"))
    name = _text(value.get("name"))
    line = _index(value.get("line"), minimum=1)
    if file is None or name is None or line is None:
        return None
    return {"file": file, "name": name, "line": line}


def _coerce_nodes(value: Any) -> list[dict[str, Any]]:
    """Coerce the node list, dropping malformed entries and duplicate ids.

    A duplicate id is dropped rather than kept because every downstream
    correlation -- edge endpoints, the evidence table, the grounding report's
    ``ref`` -- keys on the node id, so two nodes sharing one id have no
    well-defined meaning.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            continue
        node_id = _text(entry.get("id"))
        kind = _choice(entry.get("kind"), _NODE_KINDS)
        label = _text(entry.get("label"))
        evidence = _evidence(entry.get("evidence"), symbol="optional")
        if node_id is None or kind is None or label is None or evidence is None:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append({"id": node_id, "kind": kind, "label": label, "evidence": evidence})
    return out


def _coerce_edges(value: Any) -> list[dict[str, Any]]:
    """Coerce the edge list, dropping entries without both endpoints.

    Endpoints are *not* checked against the node list here: an edge pointing at
    a node that does not exist is exactly what the grounding pass reports as
    ``EDGE_ENDPOINT_UNGROUNDED``, and swallowing it at the coercion boundary
    would hide it from the report.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        source = _text(entry.get("from"))
        target = _text(entry.get("to"))
        if source is None or target is None:
            continue
        out.append({"from": source, "to": target, "label": _text(entry.get("label"))})
    return out


def coerce_flowchart_spec(value: Any) -> dict[str, Any]:
    """Coerce a raw model payload into a flowchart spec.

    Malformed nodes and edges are dropped individually and an unusable root
    becomes ``None``; a non-dict payload degrades to the empty spec
    (``{"root": None, "nodes": [], "edges": []}``). A ``None`` root is the one
    result that does not validate against :data:`FLOWCHART_SPEC_SCHEMA` -- see
    :func:`_empty_flowchart_spec`.
    """
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

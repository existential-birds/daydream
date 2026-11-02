"""Diagram schemas accept groundable specs; coercion drops malformed entries individually.

Recursive Codex strict-mode conformance is covered by test_output_schema_strict.py.
"""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest

from daydream.deep.diagram_schema import (
    FLOWCHART_SPEC_SCHEMA,
    SEQUENCE_SPEC_SCHEMA,
    coerce_flowchart_spec,
    coerce_sequence_spec,
)

EMPTY_SEQUENCE_SPEC: dict[str, Any] = {"participants": [], "messages": [], "blocks": []}
EMPTY_FLOWCHART_SPEC: dict[str, Any] = {"root": None, "nodes": [], "edges": []}


def _message(**overrides: Any) -> dict[str, Any]:
    """Return a valid sequence message, with overrides applied."""
    message: dict[str, Any] = {
        "from": "Handler", "to": "Resolver", "label": "resolve identity", "kind": "call", "changed": True,
        "evidence": {"file": "proxy/handler.py", "line": 41, "symbol": "resolve_identity"},
    }
    message.update(overrides)
    return message


def _node(**overrides: Any) -> dict[str, Any]:
    """Return a valid flowchart node, with overrides applied."""
    node: dict[str, Any] = {"id": "n1", "kind": "decision", "label": "passthrough enabled?",
        "evidence": {"file": "proxy/auth.py", "line": 27, "symbol": None},
    }
    node.update(overrides)
    return node


def test_empty_specs_are_the_pinned_shapes() -> None:
    """The empty-spec sentinels are exactly what the diagram step branches on."""
    assert coerce_sequence_spec("not a spec") == EMPTY_SEQUENCE_SPEC
    assert coerce_sequence_spec({"messages": []}) == EMPTY_SEQUENCE_SPEC
    assert coerce_flowchart_spec(None) == EMPTY_FLOWCHART_SPEC
    # The empty sequence spec is schema-valid; the empty flowchart spec
    # deliberately is not, because there is no such thing as an empty root.
    jsonschema.validate(EMPTY_SEQUENCE_SPEC, SEQUENCE_SPEC_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(EMPTY_FLOWCHART_SPEC, FLOWCHART_SPEC_SCHEMA)

def test_coercion_drops_each_malformed_entry_and_remaps_block_indices() -> None:
    """Every malformed entry drops on its own; block indices follow surviving messages.

    Owner-level replacement for the runner-level coercion fixture: grounding and
    repair must never see an entry coercion should have dropped.
    """
    evidence = {"file": "proxy/handler.py", "line": 41, "symbol": "resolve_identity"}
    good, digit = _message(), _message(evidence={**evidence, "line": "6"})
    sequence = coerce_sequence_spec({
        "participants": [
            {"name": "Handler", "kind": "internal", "files": ["a.py", "a.py", "a$b.py", "../x.py", 3]},
            "not a participant", {"name": "", "kind": "internal"}, {"name": "Ghost", "kind": "daemon"},
            {"name": "Handler", "kind": "external"},
            {"name": "Resolver", "kind": "external", "files": "nope", "service": "svc"}],
        "messages": ["not a message", good, digit, _message(kind="broadcast"),
                     *(_message(evidence={**evidence, **bad}) for bad in (
                         {"line": True}, {"line": 0}, {"file": "a$b.py"}, {"symbol": None})),
                     _message(evidence="nowhere")],
        "blocks": ["x", {"kind": "loop", "branches": []}, {"kind": "weird", "branches": []},
                   {"kind": "opt", "branches": [
                       "x", {"condition": "c", "evidence": {"file": "a.py", "line": 3},
                             "messages": [2, 1, 1, 99, True, "x"]},
                       {"condition": "", "evidence": {"file": "a.py", "line": 3}, "messages": [1]}]}],
    })
    assert sequence == {
        "participants": [{"name": "Handler", "kind": "internal", "files": ["a.py"], "service": None},
                         {"name": "Resolver", "kind": "external", "files": [], "service": "svc"}],
        "messages": [good, {**digit, "evidence": {**evidence, "line": 6}}],
        "blocks": [{"kind": "opt", "branches": [
            {"condition": "c", "evidence": {"file": "a.py", "line": 3}, "messages": [1, 0]}]}],
    }
    jsonschema.validate(sequence, SEQUENCE_SPEC_SCHEMA)
    second = _node(id="n2", kind="process", evidence={"file": "proxy/auth.py", "line": 28})
    flowchart = coerce_flowchart_spec({
        "root": {"file": "proxy/auth.py", "name": "run", "line": "1"},
        "nodes": [_node(), _node(label="duplicate id"), _node(id="k", kind="gateway"), _node(id="e", label=""),
                  "not a node", second],
        "edges": [{"from": "n1", "to": "n2", "label": "  yes  "}, {"from": "n1"}, "not an edge",
                  {"from": "n1", "to": "missing"}],
    })
    assert flowchart == {
        "root": {"file": "proxy/auth.py", "name": "run", "line": 1},
        "nodes": [_node(), {**second, "evidence": {"file": "proxy/auth.py", "line": 28, "symbol": None}}],
        "edges": [{"from": "n1", "to": "n2", "label": "yes"}, {"from": "n1", "to": "missing", "label": None}],
    }
    unusable = coerce_flowchart_spec({"root": {"file": "a.py", "name": "f"}, "nodes": [_node()], "edges": []})
    assert unusable["root"] is None and len(unusable["nodes"]) == 1
    assert coerce_flowchart_spec({"root": "run", "nodes": "n1", "edges": {}}) == EMPTY_FLOWCHART_SPEC

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

def test_empty_specs_are_fresh_objects_per_call() -> None:
    first = coerce_sequence_spec(7)
    second = coerce_sequence_spec(7)
    first["messages"].append(_message())
    assert second == EMPTY_SEQUENCE_SPEC
    first_flow = coerce_flowchart_spec(7)
    coerce_flowchart_spec(7)["nodes"].append(_node())
    assert first_flow == EMPTY_FLOWCHART_SPEC

def test_coerce_flowchart_nulls_an_unusable_root() -> None:
    """A root missing its line is None, which the diagram step treats as unusable."""
    coerced = coerce_flowchart_spec({"root": {"file": "a.py", "name": "f"}, "nodes": [_node()], "edges": []})
    assert coerced["root"] is None
    assert len(coerced["nodes"]) == 1

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

def test_coerce_sequence_drops_malformed_entries_individually() -> None:
    coerced = coerce_sequence_spec({"participants": [
                {"name": "Handler", "kind": "internal", "files": ["proxy/handler.py"]},
                {"name": "", "kind": "internal", "files": []}, {"name": "Ghost", "kind": "daemon", "files": []},
                {"name": "Handler", "kind": "external", "files": []}, "not a participant",
            ], "messages": [_message(), _message(kind="shout"), "not a message",
                _message(evidence={"file": "proxy/handler.py", "line": 44}),
            ], "blocks": [{"kind": "par", "branches": []},
                {"kind": "loop",
                    "branches": [{"condition": "", "evidence": {"file": "a.py", "line": 1}, "messages": []}],
                },
            ],
        }
    )
    assert [p["name"] for p in coerced["participants"]] == ["Handler"]
    assert coerced["participants"][0]["service"] is None
    assert len(coerced["messages"]) == 1
    assert coerced["blocks"] == []
    jsonschema.validate(coerced, SEQUENCE_SPEC_SCHEMA)

def test_coerce_sequence_remaps_branch_indices_over_a_dropped_message() -> None:
    coerced = coerce_sequence_spec({"participants": [],
            "messages": [_message(label="first"), _message(label="dropped", kind="broadcast"), _message(label="third"),
            ], "blocks": [{"kind": "alt",
                    "branches": [{"condition": "a", "evidence": {"file": "a.py", "line": 1}, "messages": [0, 2, 2, 9]},
                        {"condition": "b", "evidence": {"file": "a.py", "line": 5}, "messages": [1]},
                    ],
                }
            ],
        }
    )
    assert [m["label"] for m in coerced["messages"]] == ["first", "third"]
    branches = coerced["blocks"][0]["branches"]
    # 0 -> 0, 2 -> 1, the duplicate and the out-of-range index are dropped, and
    # the branch that pointed only at the dropped message keeps no indices.
    assert branches[0]["messages"] == [0, 1]
    assert branches[1]["messages"] == []
    jsonschema.validate(coerced, SEQUENCE_SPEC_SCHEMA)

def test_coerce_sequence_normalizes_lines_and_paths() -> None:
    """A quoted line survives; a bool, a zero and a junk path do not."""
    coerced = coerce_sequence_spec({"messages": [
                _message(label="quoted", evidence={"file": "a.py", "line": "12", "symbol": "f"}),
                _message(label="bool", evidence={"file": "a.py", "line": True, "symbol": "f"}),
                _message(label="zero", evidence={"file": "a.py", "line": 0, "symbol": "f"}),
                _message(label="junk", evidence={"file": "a$b.py", "line": 3, "symbol": "f"}),
            ],
        }
    )
    assert [m["label"] for m in coerced["messages"]] == ["quoted"]
    assert coerced["messages"][0]["evidence"]["line"] == 12

def test_coerce_flowchart_drops_malformed_and_duplicate_nodes() -> None:
    """Bad nodes and a repeated id go; a dangling edge stays for grounding."""
    coerced = coerce_flowchart_spec({"root": {"file": "proxy/auth.py", "name": "resolve", "line": 22},
            "nodes": [
                _node(id="n1", kind="start", label="resolve"), _node(id="n1", kind="decision", label="duplicate id"),
                _node(id="n2", kind="gateway", label="bad kind"),
                _node(id="n3", kind="process", label="", evidence={"file": "a.py", "line": 3}),
                _node(id="n4", kind="subroutine", label="verify"), "not a node",
            ], "edges": [{"from": "n1", "to": "n4", "label": "  yes  "}, {"from": "n1", "to": "n9"}, {"from": "n1"},
                "not an edge",
            ],
        }
    )
    assert [n["id"] for n in coerced["nodes"]] == ["n1", "n4"]
    assert coerced["nodes"][0]["label"] == "resolve"
    # ``symbol`` is nullable, not absent, and an unlabeled edge carries None.
    assert coerced["nodes"][1]["evidence"]["symbol"] is None
    assert coerced["edges"] == [{"from": "n1", "to": "n4", "label": "yes"}, {"from": "n1", "to": "n9", "label": None}]
    jsonschema.validate(coerced, FLOWCHART_SPEC_SCHEMA)

def test_coerce_flowchart_nulls_an_unusable_root() -> None:
    """A root missing its line is None, which the diagram step treats as unusable."""
    coerced = coerce_flowchart_spec({"root": {"file": "a.py", "name": "f"}, "nodes": [_node()], "edges": []})
    assert coerced["root"] is None
    assert len(coerced["nodes"]) == 1

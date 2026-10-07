"""Unit tests for daydream.deep.diagram_render (issue #1113).

The renderers are the last thing between model-authored JSON and a PR comment,
so these tests are byte-golden and adversarial: the two ``.mmd`` fixtures pin
the exact mermaid bytes, and the injection cases feed the spec's
attack payloads through every label slot.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from daydream.config import (
    DIAGRAM_LABEL_CAP_EDGE,
    DIAGRAM_LABEL_CAP_MESSAGE,
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_EDGES,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_NODES,
    DIAGRAM_MAX_PARTICIPANTS,
)
from daydream.deep.diagram_render import (
    render_diagram_blocks,
    render_flowchart_mermaid,
    render_omission_notice,
    render_sequence_mermaid,
    sanitize_label,
)
from daydream.deep.render import insert_diagrams_section, render_report

FIXTURES = Path(__file__).parent / "fixtures" / "deep"

# Golden specs: the spec's "Target rendering" block, as a grounded spec_final.

SEQUENCE_SPEC: dict[str, Any] = {"participants": [{"name": "Client", "kind": "external", "files": [], "service": None},
        {"name": "External API Proxy", "kind": "internal", "files": ["proxy/handler.py"], "service": "proxy"},
        {"name": "Identity Resolver", "kind": "internal", "files": ["proxy/auth.py"], "service": "proxy"},
    ], "messages": [{"from": "Client", "to": "External API Proxy", "label": "Request with Authorization",
         "kind": "call", "changed": True,
         "evidence": {"file": "proxy/handler.py", "line": 41, "symbol": "handle_request"}},
        {"from": "External API Proxy", "to": "Identity Resolver", "label": "Extract client ID and patient UUID",
         "kind": "call", "changed": True,
         "evidence": {"file": "proxy/handler.py", "line": 52, "symbol": "resolve_identity"}},
        {"from": "Identity Resolver", "to": "External API Proxy", "label": "Unverified identity",
         "kind": "reply", "changed": False,
         "evidence": {"file": "proxy/auth.py", "line": 68, "symbol": "resolve_identity"}},
        {"from": "External API Proxy", "to": "Identity Resolver", "label": "Verify JWT",
         "kind": "call", "changed": True, "evidence": {"file": "proxy/handler.py", "line": 57, "symbol": "verify_jwt"}},
        {"from": "Identity Resolver", "to": "External API Proxy", "label": "Verified claims",
         "kind": "reply", "changed": False, "evidence": {"file": "proxy/auth.py", "line": 71, "symbol": "verify_jwt"}},
    ], "blocks": [{"kind": "alt", "branches": [{"condition": "Ro-Passthrough in enabled non-production environment",
             "evidence": {"file": "proxy/handler.py", "line": 50}, "messages": [1, 2]},
            {"condition": "Bearer token", "evidence": {"file": "proxy/handler.py", "line": 55}, "messages": [3, 4]},
        ]},
    ],
}

FLOWCHART_SPEC: dict[str, Any] = {"root": {"file": "proxy/auth.py", "name": "resolve_identity", "line": 22},
    "nodes": [{"id": "enter", "kind": "start", "label": "resolve_identity",
         "evidence": {"file": "proxy/auth.py", "line": 22, "symbol": "resolve_identity"}},
        {"id": "passthrough", "kind": "decision", "label": "Ro-Passthrough enabled?",
         "evidence": {"file": "proxy/auth.py", "line": 27, "symbol": None}},
        {"id": "extract", "kind": "process", "label": "Extract client ID and patient UUID",
         "evidence": {"file": "proxy/auth.py", "line": 31, "symbol": None}},
        {"id": "bearer", "kind": "decision", "label": "Bearer token present?",
         "evidence": {"file": "proxy/auth.py", "line": 38, "symbol": None}},
        {"id": "verify", "kind": "subroutine", "label": "verify_jwt",
         "evidence": {"file": "proxy/auth.py", "line": 44, "symbol": "verify_jwt"}},
        {"id": "reject", "kind": "end", "label": "Reject 401",
         "evidence": {"file": "proxy/auth.py", "line": 49, "symbol": None}},
        {"id": "unverified", "kind": "end", "label": "Return unverified identity",
         "evidence": {"file": "proxy/auth.py", "line": 35, "symbol": None}},
        {"id": "verified", "kind": "end", "label": "Return verified claims",
         "evidence": {"file": "proxy/auth.py", "line": 47, "symbol": None}},
    ],
    "edges": [
        {"from": "enter", "to": "passthrough", "label": None}, {"from": "passthrough", "to": "extract", "label": "yes"},
        {"from": "passthrough", "to": "bearer", "label": "no"}, {"from": "bearer", "to": "verify", "label": "yes"},
        {"from": "bearer", "to": "reject", "label": "no"}, {"from": "extract", "to": "unverified", "label": None},
        {"from": "verify", "to": "verified", "label": None},
    ],
}


def _rendered(spec: dict[str, Any]) -> dict[str, Any]:
    return {"status": "rendered", "spec_final": spec}


def _both_rendered() -> dict[str, dict[str, Any] | None]:
    return {"sequence": _rendered(SEQUENCE_SPEC), "flowchart": _rendered(FLOWCHART_SPEC)}


# Byte goldens

def test_sequence_mermaid_matches_golden_fixture_byte_for_byte() -> None:
    # The fixture carries a final newline (.editorconfig insert_final_newline);
    # the renderer emits none because the text is embedded in a ``` fence.
    golden = (FIXTURES / "diagram_sequence.mmd").read_text()
    assert render_sequence_mermaid(SEQUENCE_SPEC) + "\n" == golden
    assert render_diagram_blocks({"sequence": _rendered(SEQUENCE_SPEC)}) == (
        "<details><summary><h3>Sequence Diagram</h3></summary>\n\n```mermaid\n"
        + golden.removesuffix("\n") + "\n```\n\n</details>"
    )

def test_flowchart_mermaid_matches_golden_fixture_byte_for_byte() -> None:
    golden = (FIXTURES / "diagram_flowchart.mmd").read_text()
    assert render_flowchart_mermaid(FLOWCHART_SPEC) + "\n" == golden
    assert render_diagram_blocks({"flowchart": _rendered(FLOWCHART_SPEC)}) == (
        "<details><summary><h3>Flowchart</h3></summary>\n\n```mermaid\n"
        + golden.removesuffix("\n") + "\n```\n\n</details>"
    )


# Ordering, gating, and the "never read a stored mermaid" rule


@pytest.mark.parametrize("results", [{}, {"sequence": None, "flowchart": None},
    {"sequence": {"status": "omitted", "spec_final": None, "omit_reasons": ["NO_END"]}},
    {"flowchart": {"status": "skipped", "reason": "not eligible", "spec_final": None}},
    {"sequence": {"status": "failed", "reason": "backend error", "spec_final": None}},
    # status says rendered but the artifact is malformed: no block, never a raise.
    {"sequence": {"status": "rendered", "spec_final": None}},
])
def test_blocks_are_empty_when_nothing_rendered(results: dict[str, Any]) -> None:
    assert render_diagram_blocks(results) == ""

def test_blocks_always_rerender_and_never_echo_a_stored_mermaid_string() -> None:
    before = copy.deepcopy(_both_rendered())
    results = copy.deepcopy(before)
    blocks = render_diagram_blocks(results)
    assert blocks == render_diagram_blocks(results)
    assert results == before
    assert render_sequence_mermaid(SEQUENCE_SPEC) == render_sequence_mermaid(copy.deepcopy(SEQUENCE_SPEC))
    assert render_flowchart_mermaid(FLOWCHART_SPEC) == render_flowchart_mermaid(copy.deepcopy(FLOWCHART_SPEC))
    assert blocks.index("<h3>Sequence Diagram</h3>") < blocks.index("<h3>Flowchart</h3>")
    assert blocks.count("<details><summary><h3>") == 2
    assert "</details>\n\n<details><summary><h3>Flowchart</h3>" in blocks
    report = insert_diagrams_section(render_report(_items()), blocks)
    assert f"## Diagrams\n{blocks}\n" in report
    assert render_sequence_mermaid(SEQUENCE_SPEC) in report
    assert render_flowchart_mermaid(FLOWCHART_SPEC) in report
    poisoned = _both_rendered()
    for kind in ("sequence", "flowchart"):
        result = poisoned[kind]
        assert result is not None
        result["mermaid"] = "sequenceDiagram\n    P1->>P9: pwned"
    blocks = render_diagram_blocks(poisoned)
    assert "pwned" not in blocks
    assert blocks == render_diagram_blocks(_both_rendered())


# Render caps: asserted, not enforced


def _participants(n: int) -> list[dict[str, Any]]:
    return [{"name": f"P{i}", "kind": "internal", "files": ["a.py"], "service": None} for i in range(n)]


def _messages(n: int) -> list[dict[str, Any]]:
    return [{"from": "P0", "to": "P0", "label": "x", "kind": "self", "changed": False,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}} for _ in range(n)]

def test_over_cap_sequence_specs_raise_value_error() -> None:
    for spec, collection in (
        ({"participants": _participants(DIAGRAM_MAX_PARTICIPANTS + 1), "messages": [], "blocks": []}, "participants"),
        ({"participants": _participants(1), "messages": _messages(DIAGRAM_MAX_MESSAGES + 1), "blocks": []}, "messages"),
        ({"participants": _participants(1), "messages": [],
          "blocks": [{"kind": "opt", "branches": []}] * (DIAGRAM_MAX_BLOCKS + 1)}, "blocks"),
    ):
        with pytest.raises(ValueError, match=collection):
            render_sequence_mermaid(spec)
    # At the cap exactly: no raise.
    render_sequence_mermaid({"participants": _participants(DIAGRAM_MAX_PARTICIPANTS),
                             "messages": _messages(DIAGRAM_MAX_MESSAGES),
                             "blocks": [{"kind": "opt", "branches": []}] * DIAGRAM_MAX_BLOCKS})

def test_over_cap_flowchart_specs_raise_value_error() -> None:
    nodes = [{"id": f"n{i}", "kind": "process", "label": "x", "evidence": {"file": "a.py", "line": 1, "symbol": None}}
             for i in range(DIAGRAM_MAX_NODES + 1)]
    with pytest.raises(ValueError, match="nodes"):
        render_flowchart_mermaid({"root": {}, "nodes": nodes, "edges": []})
    edges = [{"from": "n0", "to": "n0", "label": None} for _ in range(DIAGRAM_MAX_EDGES + 1)]
    with pytest.raises(ValueError, match="edges"):
        render_flowchart_mermaid({"root": {}, "nodes": nodes[:1], "edges": edges})
    render_flowchart_mermaid({"root": {}, "nodes": nodes[:DIAGRAM_MAX_NODES], "edges": edges[:DIAGRAM_MAX_EDGES]})


# Sanitization and injection (spec test 11)

# Every payload from spec test 11, plus a control byte and an entity forgery.
_PAYLOADS = (
    "end\nP1->>P9: pwned",
    "N9{x} --> N1", "%%{init: {'theme':'x'}}%%", "`rm -rf /`", "a|b", "</details>", "drop;table",
    "forge #lt; entity",
    "bell\x07and\ttab",
)

@pytest.mark.parametrize("payload", _PAYLOADS)
def test_sanitize_label_strips_every_mermaid_metacharacter(payload: str) -> None:
    out = sanitize_label(payload, DIAGRAM_LABEL_CAP_MESSAGE)
    # ``<``/``>``/``"`` are escaped away entirely; the escapes themselves are the
    # only place a ``#`` or a ``;`` may appear, so strip them before checking the
    # banned set.
    for gone in ("<", ">", '"', "\n", "\r", "\x07", "%%"):
        assert gone not in out, f"{gone!r} survived in {out!r}"
    bare = out.replace("#lt;", "").replace("#gt;", "").replace("#quot;", "")
    for banned in ("#", ";", "`", "|", "[", "]", "{", "}", "(", ")", "\\"):
        assert banned not in bare, f"{banned!r} survived in {out!r}"

def test_sanitize_label_escapes_and_collapses_and_caps() -> None:
    assert sanitize_label("a <b> \"c\"", 80) == "a #lt;b#gt; #quot;c#quot;"
    assert sanitize_label("  many   \n spaces\t here  ", 80) == "many spaces here"
    assert sanitize_label("%%%%%", 80) == "%"          # doubled percents removed, a lone one kept
    assert sanitize_label("50% done", 80) == "50% done"
    assert sanitize_label("abcdefghij", 4) == "abcd"
    assert sanitize_label("abc defghij", 4) == "abc"   # right-stripped after the cut
    assert sanitize_label("abcdefghij", 0) == "abcdefghij"  # cap <= 0 disables truncation
    assert sanitize_label("", 40) == ""
    # Truncation happens before escaping, so an escape is never bisected.
    assert sanitize_label("<<<<", 2) == "#lt;#lt;"

def test_injection_payloads_never_add_a_mermaid_statement() -> None:
    spec: dict[str, Any] = {"participants": [
            {"name": "end\nP1->>P9: pwned", "kind": "internal", "files": ["a.py"], "service": None},
            {"name": "`|</details>", "kind": "internal", "files": ["b.py"], "service": None},
        ],
        "messages": [
            {"from": "end\nP1->>P9: pwned", "to": "`|</details>",
             "label": "end\nP1->>P9: pwned", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}},
        ],
        "blocks": [{"kind": "alt", "branches": [
            {"condition": "%%{init}%%", "evidence": {"file": "a.py", "line": 1}, "messages": [0]},
            {"condition": "end", "evidence": {"file": "a.py", "line": 2}, "messages": []},
        ]}],
    }
    mermaid = render_sequence_mermaid(spec)
    lines = mermaid.split("\n")
    # header + 2 participants + alt + 1 message + end == 6 lines. No 7th statement.
    assert len(lines) == 6
    assert lines[0] == "sequenceDiagram"
    assert lines[4].startswith("        P1->>P2: ")
    assert lines[5] == "    end"
    assert "%%" not in mermaid

def test_flowchart_injection_payloads_cannot_close_a_shape_or_add_an_edge() -> None:
    spec: dict[str, Any] = {"root": {"file": "a.py", "name": "`|f", "line": 1},
        "nodes": [{"id": "a", "kind": "start", "label": "N9{x} --> N1",
             "evidence": {"file": "a.py", "line": 1, "symbol": "f"}},
            {"id": "b", "kind": "decision", "label": "}{ %% `x`",
             "evidence": {"file": "a.py", "line": 2, "symbol": None}},
            {"id": "c", "kind": "io", "label": "/]read[/", "evidence": {"file": "a.py", "line": 3, "symbol": None}},
            {"id": "d", "kind": "subroutine", "label": "[[call]]",
             "evidence": {"file": "a.py", "line": 4, "symbol": "call"}},
            {"id": "e", "kind": "end", "label": "</details>", "evidence": {"file": "a.py", "line": 5, "symbol": None}},
        ], "edges": [{"from": "a", "to": "b", "label": "yes|N9 --> N1"}, {"from": "b", "to": "c", "label": None},
            {"from": "c", "to": "d", "label": "%%"}, {"from": "d", "to": "e", "label": "x" * 200},
        ],
    }
    mermaid = render_flowchart_mermaid(spec)
    lines = mermaid.split("\n")
    assert len(lines) == 5  # header + 4 edges, no extra statement
    assert "%%" not in mermaid
    # The over-long edge label is capped at the configured length.
    assert f"-->|{'x' * DIAGRAM_LABEL_CAP_EDGE}|" in mermaid
    assert f"-->|{'x' * (DIAGRAM_LABEL_CAP_EDGE + 1)}|" not in mermaid
    # A label that sanitizes to nothing degrades to an unlabeled edge, never ``-->||``.
    assert "-->||" not in mermaid
    # First mention carries the shape, later mentions are bare; the adversarial
    # ``/]read[/`` label cannot close the io shape early.
    assert lines[1] == "    N1([N9x --#gt; N1]) -->|yesN9 --#gt; N1| N2{x}"
    assert lines[2] == "    N2 --> N3[/read/]"
    assert lines[3] == "    N3 --> N4[[call]]"
    assert lines[4].endswith(" N5([#lt;/details#gt;])")

def test_injection_payloads_cannot_close_the_html_wrapper() -> None:
    spec: dict[str, Any] = {
        "participants": [{"name": "</details>", "kind": "internal", "files": ["a.py"], "service": None}],
        "messages": [{"from": "</details>", "to": "</details>", "label": "a|b</details>",
                      "kind": "self", "changed": True, "evidence": {"file": "a|b`.py", "line": 7, "symbol": "x"}}],
        "blocks": [],
    }
    blocks = render_diagram_blocks({"sequence": _rendered(spec)})
    assert blocks.count("<details>") == blocks.count("</details>") == 1
    assert "#lt;/details#gt;" in blocks


# Block structure edge cases

def test_opt_and_loop_blocks_and_a_message_outside_every_block() -> None:
    spec: dict[str, Any] = {"participants": [{"name": "A", "kind": "internal", "files": ["a.py"], "service": None},
                         {"name": "B", "kind": "internal", "files": ["b.py"], "service": None}],
        "messages": [{"from": "A", "to": "B", "label": "m0", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}},
            {"from": "A", "to": "B", "label": "m1", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 2, "symbol": "x"}},
            {"from": "B", "to": "A", "label": "m2", "kind": "reply", "changed": False,
             "evidence": {"file": "b.py", "line": 3, "symbol": "x"}},
            {"from": "A", "to": "A", "label": "m3", "kind": "self", "changed": True,
             "evidence": {"file": "a.py", "line": 4, "symbol": "x"}},
        ], "blocks": [{"kind": "opt", "branches": [{"condition": "cached", "evidence": {"file": "a.py", "line": 1},
                                          "messages": [1]}]},
            {"kind": "loop", "branches": [{"condition": "each page",
                                           "evidence": {"file": "a.py", "line": 2}, "messages": [2]}]},
        ],
    }
    assert render_sequence_mermaid(spec).split("\n") == [
        "sequenceDiagram", "    participant P1 as A", "    participant P2 as B", "    P1->>P2: m0", "    opt cached",
        "        P1->>P2: m1", "    end", "    loop each page", "        P2-->>P1: m2", "    end", "    P1->>P1: m3",
    ]

def test_malformed_block_indices_and_unknown_endpoints_degrade_without_raising() -> None:
    spec: dict[str, Any] = {"participants": [{"name": "A", "kind": "internal", "files": ["a.py"], "service": None}],
        "messages": [{"from": "A", "to": "Ghost", "label": "m0", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}},
            {"from": "A", "to": "A", "label": "m1", "kind": "self", "changed": True,
             "evidence": {"file": "a.py", "line": 2, "symbol": "x"}},
        ],
        # Out-of-range, non-int, and duplicate-claim indices are all ignored.
        "blocks": [{"kind": "nope", "branches": [{"condition": "c", "evidence": {}, "messages": [99, "1", 1]}]},
                   {"kind": "alt", "branches": [{"condition": "d", "evidence": {}, "messages": [1]}]}],
    }
    assert render_sequence_mermaid(spec).split("\n") == [
        "sequenceDiagram", "    participant P1 as A",
        # m0's target is not a declared participant -> no arrow, but the block
        # opened by the message that IS claimed still closes cleanly.
        "    opt c", "        P1->>P1: m1", "    end",
    ]

def test_empty_labels_fall_back_to_a_placeholder() -> None:
    spec: dict[str, Any] = {
        "participants": [{"name": "", "kind": "internal", "files": [], "service": None}], "messages": [], "blocks": [],
    }
    assert render_sequence_mermaid(spec) == "sequenceDiagram\n    participant P1 as unlabeled"
    flow: dict[str, Any] = {"root": {}, "nodes": [{"id": "a", "kind": "process", "label": "```",
                                                   "evidence": {}}], "edges": []}
    assert render_flowchart_mermaid(flow) == "flowchart TD\n    N1[unlabeled]"

def test_unknown_node_kind_falls_back_to_the_process_shape() -> None:
    flow: dict[str, Any] = {"root": {"file": "a.py", "name": "f", "line": 1},
        "nodes": [{"id": "a", "kind": "mystery", "label": "x", "evidence": {"file": "a.py", "line": 1}}], "edges": [],
    }
    assert render_flowchart_mermaid(flow) == "flowchart TD\n    N1[x]"


# Omission notice

def test_omission_notice_reports_floor_codes() -> None:
    notice = render_omission_notice("sequence", {
        "status": "omitted", "omit_reasons": ["TOO_FEW_MESSAGES", "NO_CHANGED_INTERACTION"],
    })
    assert notice == ("No sequence diagram was rendered for this pull request. "
        "Grounding floor not met: TOO_FEW_MESSAGES, NO_CHANGED_INTERACTION."
    )

def test_omission_notice_covers_skipped_failed_and_rendered() -> None:
    assert render_omission_notice("flowchart", {"status": "rendered"}) == ""
    assert render_omission_notice("flowchart", {"status": "skipped", "reason": "no candidate root"}) == (
        "No flowchart was rendered for this pull request. Reason: no candidate root."
    )
    assert render_omission_notice("flowchart", {"status": "failed", "reason": "backend `boom`\nline two"}) == (
        "No flowchart was rendered for this pull request. Reason: backend boom line two."
    )


# render_report (deep/render.py); the text surgery is pinned in tests/test_deep_render.py


def _items() -> list[dict[str, Any]]:
    return [{"id": 1, "lens": "per-stack", "file": "a.py", "line": 9, "description": "bug"},
            {"id": 2, "lens": "cross-stack", "file": "b.py", "line": 2, "description": "drift"}]

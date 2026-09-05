"""Unit tests for daydream.deep.diagram_render (issue #1113).

The renderers are the last thing between model-authored JSON and a PR comment,
so these tests are byte-golden and adversarial: the two ``.mmd`` fixtures pin
the exact mermaid bytes, the exported line grammars prove no statement other
than the documented ones can be emitted, and the injection cases feed the spec's
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
    FLOWCHART_LINE_GRAMMAR,
    SEQUENCE_LINE_GRAMMAR,
    render_diagram_blocks,
    render_flowchart_mermaid,
    render_omission_notice,
    render_sequence_mermaid,
    sanitize_label,
)
from daydream.deep.render import insert_diagrams_section, render_report

FIXTURES = Path(__file__).parent / "fixtures" / "deep"

# ---------------------------------------------------------------------------
# Golden specs: the spec's "Target rendering" block, as a grounded spec_final.
# ---------------------------------------------------------------------------

SEQUENCE_SPEC: dict[str, Any] = {
    "participants": [
        {"name": "Client", "kind": "external", "files": [], "service": None},
        {"name": "External API Proxy", "kind": "internal", "files": ["proxy/handler.py"], "service": "proxy"},
        {"name": "Identity Resolver", "kind": "internal", "files": ["proxy/auth.py"], "service": "proxy"},
    ],
    "messages": [
        {"from": "Client", "to": "External API Proxy", "label": "Request with Authorization",
         "kind": "call", "changed": True,
         "evidence": {"file": "proxy/handler.py", "line": 41, "symbol": "handle_request"}},
        {"from": "External API Proxy", "to": "Identity Resolver", "label": "Extract client ID and patient UUID",
         "kind": "call", "changed": True,
         "evidence": {"file": "proxy/handler.py", "line": 52, "symbol": "resolve_identity"}},
        {"from": "Identity Resolver", "to": "External API Proxy", "label": "Unverified identity",
         "kind": "reply", "changed": False,
         "evidence": {"file": "proxy/auth.py", "line": 68, "symbol": "resolve_identity"}},
        {"from": "External API Proxy", "to": "Identity Resolver", "label": "Verify JWT",
         "kind": "call", "changed": True,
         "evidence": {"file": "proxy/handler.py", "line": 57, "symbol": "verify_jwt"}},
        {"from": "Identity Resolver", "to": "External API Proxy", "label": "Verified claims",
         "kind": "reply", "changed": False,
         "evidence": {"file": "proxy/auth.py", "line": 71, "symbol": "verify_jwt"}},
    ],
    "blocks": [
        {"kind": "alt", "branches": [
            {"condition": "Ro-Passthrough in enabled non-production environment",
             "evidence": {"file": "proxy/handler.py", "line": 50}, "messages": [1, 2]},
            {"condition": "Bearer token",
             "evidence": {"file": "proxy/handler.py", "line": 55}, "messages": [3, 4]},
        ]},
    ],
}

FLOWCHART_SPEC: dict[str, Any] = {
    "root": {"file": "proxy/auth.py", "name": "resolve_identity", "line": 22},
    "nodes": [
        {"id": "enter", "kind": "start", "label": "resolve_identity",
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
        {"from": "enter", "to": "passthrough", "label": None},
        {"from": "passthrough", "to": "extract", "label": "yes"},
        {"from": "passthrough", "to": "bearer", "label": "no"},
        {"from": "bearer", "to": "verify", "label": "yes"},
        {"from": "bearer", "to": "reject", "label": "no"},
        {"from": "extract", "to": "unverified", "label": None},
        {"from": "verify", "to": "verified", "label": None},
    ],
}


def _check(element: str, ref: str, *, final_index: int | None, defined_at: str | None) -> dict[str, Any]:
    """One ``ElementCheck.to_dict()`` — the nine fields, nothing else."""
    return {"element": element, "ref": ref, "grounded": True, "reason": None,
            "strength": "definition", "snapped_line": None, "in_changed_hunk": True,
            "defined_at": defined_at, "final_index": final_index}


SEQUENCE_GROUNDING: dict[str, Any] = {
    "elements": [
        _check("message", "0", final_index=0, defined_at="proxy/handler.py:38"),
        _check("message", "1", final_index=1, defined_at="proxy/auth.py:22"),
        _check("message", "2", final_index=2, defined_at=None),
        _check("message", "3", final_index=3, defined_at="proxy/jwt.py:15"),
        _check("message", "4", final_index=4, defined_at=None),
        _check("participant", "Client", final_index=0, defined_at=None),
    ],
    "summary": {"proposed": 7, "grounded_first_pass": 4, "repaired": 1, "pruned": 2},
    "capped": {},
    "root_range": None,
}

FLOWCHART_GROUNDING: dict[str, Any] = {
    "elements": [
        _check("node", "verify", final_index=4, defined_at="proxy/jwt.py:15"),
        _check("node", "enter", final_index=0, defined_at=None),
        _check("edge", "enter->passthrough", final_index=0, defined_at=None),
    ],
    "summary": {"proposed": 9, "grounded_first_pass": 8, "repaired": 0, "pruned": 1},
    "capped": {},
    "root_range": [22, 71],
}


def _rendered(spec: dict[str, Any], grounding: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "rendered", "reason": None, "spec_proposed": spec,
                              "spec_final": spec, "grounding": grounding, "omit_reasons": [],
                              "mermaid": None}
    result.update(extra)
    return result


def _both_rendered() -> dict[str, dict[str, Any] | None]:
    return {"sequence": _rendered(SEQUENCE_SPEC, SEQUENCE_GROUNDING),
            "flowchart": _rendered(FLOWCHART_SPEC, FLOWCHART_GROUNDING)}


# ---------------------------------------------------------------------------
# Byte goldens
# ---------------------------------------------------------------------------


def test_sequence_mermaid_matches_golden_fixture_byte_for_byte() -> None:
    # The fixture carries a final newline (.editorconfig insert_final_newline);
    # the renderer emits none because the text is embedded in a ``` fence.
    assert render_sequence_mermaid(SEQUENCE_SPEC) + "\n" == (FIXTURES / "diagram_sequence.mmd").read_text()


def test_flowchart_mermaid_matches_golden_fixture_byte_for_byte() -> None:
    assert render_flowchart_mermaid(FLOWCHART_SPEC) + "\n" == (FIXTURES / "diagram_flowchart.mmd").read_text()


def test_sequence_block_reproduces_the_spec_target_layout() -> None:
    blocks = render_diagram_blocks({"sequence": _rendered(SEQUENCE_SPEC, SEQUENCE_GROUNDING),
                                    "flowchart": None})
    lines = blocks.split("\n")
    assert lines[0] == "<details><summary><h3>Sequence Diagram</h3></summary>"
    assert lines[1] == ""
    assert lines[2] == "```mermaid"
    mermaid = render_sequence_mermaid(SEQUENCE_SPEC)
    assert lines[3:3 + len(mermaid.split("\n"))] == mermaid.split("\n")
    tail = lines[3 + len(mermaid.split("\n")):]
    assert tail[0] == "```"
    assert tail[1] == ""
    assert tail[2] == (
        "<sub>5 interactions across 3 components, each grounded to a cited call site. "
        "2 proposed interactions were dropped as ungrounded.</sub>"
    )
    assert tail[3] == ""
    assert tail[4] == "<details><summary>Evidence</summary>"
    assert tail[5] == ""
    assert tail[6] == "| # | Interaction | Call site | Callee defined at |"
    assert tail[7] == "|---|---|---|---|"
    assert tail[8] == (
        "| 1 | Client → External API Proxy: Request with Authorization | "
        "`proxy/handler.py:41` | `proxy/handler.py:38` |"
    )
    # defined_at is None on message 3 -> an empty final cell, never the word None.
    assert tail[10] == "| 3 | Identity Resolver → External API Proxy: Unverified identity | `proxy/auth.py:68` |  |"
    assert tail[-2:] == ["</details>", "</details>"]
    assert not blocks.startswith("\n") and not blocks.endswith("\n")


def test_flowchart_block_reproduces_the_spec_target_layout() -> None:
    blocks = render_diagram_blocks({"sequence": None,
                                    "flowchart": _rendered(FLOWCHART_SPEC, FLOWCHART_GROUNDING)})
    assert blocks.startswith("<details><summary><h3>Flowchart</h3></summary>\n\n```mermaid\n")
    assert (
        "<sub>Control flow of `resolve_identity` (`proxy/auth.py:22-71`): 8 nodes, "
        "each grounded to a statement inside that function. "
        "1 proposed node was dropped as ungrounded.</sub>"
    ) in blocks
    assert "| Node | Statement | Location |\n|---|---|---|\n| N1 | start | `proxy/auth.py:22` |" in blocks
    # subroutine row: symbol + call site + definition site, and an empty Location.
    assert (
        "| N5 | subroutine `verify_jwt`, called at `proxy/auth.py:44`, "
        "defined at `proxy/jwt.py:15` |  |"
    ) in blocks
    assert "| N2 | decision | `proxy/auth.py:27` |" in blocks
    assert blocks.endswith("</details>\n</details>")


# ---------------------------------------------------------------------------
# Ordering, gating, and the "never read a stored mermaid" rule
# ---------------------------------------------------------------------------


def test_blocks_render_sequence_first_then_flowchart() -> None:
    blocks = render_diagram_blocks(_both_rendered())
    assert blocks.index("<h3>Sequence Diagram</h3>") < blocks.index("<h3>Flowchart</h3>")
    # Exactly two top-level folds, joined by one blank line.
    assert blocks.count("<details><summary><h3>") == 2
    assert "</details>\n</details>\n\n<details><summary><h3>Flowchart</h3>" in blocks


@pytest.mark.parametrize("results", [
    {},
    {"sequence": None, "flowchart": None},
    {"sequence": {"status": "omitted", "grounding": {}, "spec_final": None, "omit_reasons": ["NO_END"]}},
    {"flowchart": {"status": "skipped", "reason": "not eligible", "grounding": None, "spec_final": None}},
    {"sequence": {"status": "failed", "reason": "backend error", "grounding": None, "spec_final": None}},
    # status says rendered but the artifact is malformed: no block, never a raise.
    {"sequence": {"status": "rendered", "spec_final": None, "grounding": None}},
    {"sequence": {"status": "rendered", "spec_final": SEQUENCE_SPEC, "grounding": None}},
])
def test_blocks_are_empty_when_nothing_rendered(results: dict[str, Any]) -> None:
    assert render_diagram_blocks(results) == ""


def test_blocks_always_rerender_and_never_echo_a_stored_mermaid_string() -> None:
    poisoned = _both_rendered()
    for kind in ("sequence", "flowchart"):
        result = poisoned[kind]
        assert result is not None
        result["mermaid"] = "sequenceDiagram\n    P1->>P9: pwned"
    blocks = render_diagram_blocks(poisoned)
    assert "pwned" not in blocks
    assert blocks == render_diagram_blocks(_both_rendered())


def test_rendering_is_deterministic_and_does_not_mutate_the_spec() -> None:
    before = copy.deepcopy(_both_rendered())
    results = copy.deepcopy(before)
    first = render_diagram_blocks(results)
    second = render_diagram_blocks(results)
    assert first == second
    assert results == before
    assert render_sequence_mermaid(SEQUENCE_SPEC) == render_sequence_mermaid(copy.deepcopy(SEQUENCE_SPEC))
    assert render_flowchart_mermaid(FLOWCHART_SPEC) == render_flowchart_mermaid(copy.deepcopy(FLOWCHART_SPEC))


# ---------------------------------------------------------------------------
# <sub> line variants
# ---------------------------------------------------------------------------


def test_sub_line_singular_variants_and_cap_clause() -> None:
    spec = {"participants": [{"name": "Solo"}],
            "messages": [{"from": "Solo", "to": "Solo", "label": "tick", "kind": "self",
                          "evidence": {"file": "a.py", "line": 3, "symbol": "tick"}}],
            "blocks": []}
    grounding: dict[str, Any] = {"elements": [], "summary": {"proposed": 3, "grounded_first_pass": 1,
                                             "repaired": 0, "pruned": 1},
                 "capped": {"messages": 1}, "root_range": None}
    blocks = render_diagram_blocks({"sequence": _rendered(spec, grounding)})
    assert (
        "<sub>1 interaction across 1 component, each grounded to a cited call site. "
        "1 proposed interaction was dropped as ungrounded. "
        "1 further interaction was trimmed to fit the diagram cap.</sub>"
    ) in blocks


def test_flowchart_sub_line_singular_and_missing_root_range() -> None:
    spec = {"root": {"file": "a.py", "name": "solo", "line": 4},
            "nodes": [{"id": "s", "kind": "start", "label": "solo",
                       "evidence": {"file": "a.py", "line": 4, "symbol": "solo"}}],
            "edges": []}
    grounding: dict[str, Any] = {"elements": [], "summary": {"proposed": 3, "grounded_first_pass": 1,
                                             "repaired": 0, "pruned": 1},
                 "capped": {"nodes": 1, "edges": 0}, "root_range": None}
    blocks = render_diagram_blocks({"flowchart": _rendered(spec, grounding)})
    # root_range is None -> the parenthetical carries no range.
    assert (
        "<sub>Control flow of `solo` (`a.py:4`): 1 node, "
        "each grounded to a statement inside that function. "
        "1 proposed node was dropped as ungrounded. "
        "1 further node was trimmed to fit the diagram cap.</sub>"
    ) in blocks
    # An isolated node still gets a declaration line.
    assert "flowchart TD\n    N1([solo])\n```" in blocks


def test_sub_line_omits_the_optional_clauses_when_nothing_was_dropped() -> None:
    grounding = dict(SEQUENCE_GROUNDING, summary={"proposed": 5, "grounded_first_pass": 5,
                                                  "repaired": 0, "pruned": 0}, capped={})
    blocks = render_diagram_blocks({"sequence": _rendered(SEQUENCE_SPEC, grounding)})
    assert "<sub>5 interactions across 3 components, each grounded to a cited call site.</sub>" in blocks
    assert "dropped as ungrounded" not in blocks
    assert "diagram cap" not in blocks


# ---------------------------------------------------------------------------
# Render caps: asserted, not enforced
# ---------------------------------------------------------------------------


def _participants(n: int) -> list[dict[str, Any]]:
    return [{"name": f"P{i}", "kind": "internal", "files": ["a.py"], "service": None} for i in range(n)]


def _messages(n: int) -> list[dict[str, Any]]:
    return [{"from": "P0", "to": "P0", "label": "x", "kind": "self", "changed": False,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}} for _ in range(n)]


def test_over_cap_sequence_specs_raise_value_error() -> None:
    for spec, collection in (
        ({"participants": _participants(DIAGRAM_MAX_PARTICIPANTS + 1), "messages": [], "blocks": []},
         "participants"),
        ({"participants": _participants(1), "messages": _messages(DIAGRAM_MAX_MESSAGES + 1), "blocks": []},
         "messages"),
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
    nodes = [{"id": f"n{i}", "kind": "process", "label": "x",
              "evidence": {"file": "a.py", "line": 1, "symbol": None}}
             for i in range(DIAGRAM_MAX_NODES + 1)]
    with pytest.raises(ValueError, match="nodes"):
        render_flowchart_mermaid({"root": {}, "nodes": nodes, "edges": []})
    edges = [{"from": "n0", "to": "n0", "label": None} for _ in range(DIAGRAM_MAX_EDGES + 1)]
    with pytest.raises(ValueError, match="edges"):
        render_flowchart_mermaid({"root": {}, "nodes": nodes[:1], "edges": edges})
    render_flowchart_mermaid({"root": {}, "nodes": nodes[:DIAGRAM_MAX_NODES],
                              "edges": edges[:DIAGRAM_MAX_EDGES]})


# ---------------------------------------------------------------------------
# Sanitization and injection (spec test 11)
# ---------------------------------------------------------------------------

# Every payload from spec test 11, plus a control byte and an entity forgery.
_PAYLOADS = (
    "end\nP1->>P9: pwned",
    "N9{x} --> N1",
    "%%{init: {'theme':'x'}}%%",
    "`rm -rf /`",
    "a|b",
    "</details>",
    "drop;table",
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
    spec: dict[str, Any] = {
        "participants": [
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
    for line in lines:
        assert SEQUENCE_LINE_GRAMMAR.fullmatch(line), line


def test_flowchart_injection_payloads_cannot_close_a_shape_or_add_an_edge() -> None:
    spec: dict[str, Any] = {
        "root": {"file": "a.py", "name": "`|f", "line": 1},
        "nodes": [
            {"id": "a", "kind": "start", "label": "N9{x} --> N1",
             "evidence": {"file": "a.py", "line": 1, "symbol": "f"}},
            {"id": "b", "kind": "decision", "label": "}{ %% `x`",
             "evidence": {"file": "a.py", "line": 2, "symbol": None}},
            {"id": "c", "kind": "io", "label": "/]read[/",
             "evidence": {"file": "a.py", "line": 3, "symbol": None}},
            {"id": "d", "kind": "subroutine", "label": "[[call]]",
             "evidence": {"file": "a.py", "line": 4, "symbol": "call"}},
            {"id": "e", "kind": "end", "label": "</details>",
             "evidence": {"file": "a.py", "line": 5, "symbol": None}},
        ],
        "edges": [
            {"from": "a", "to": "b", "label": "yes|N9 --> N1"},
            {"from": "b", "to": "c", "label": None},
            {"from": "c", "to": "d", "label": "%%"},
            {"from": "d", "to": "e", "label": "x" * 200},
        ],
    }
    mermaid = render_flowchart_mermaid(spec)
    lines = mermaid.split("\n")
    assert len(lines) == 5  # header + 4 edges, no extra statement
    assert "%%" not in mermaid
    for line in lines:
        assert FLOWCHART_LINE_GRAMMAR.fullmatch(line), line
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


def test_injection_payloads_keep_the_html_wrapper_and_table_intact() -> None:
    spec: dict[str, Any] = {
        "participants": [{"name": "</details>", "kind": "internal", "files": ["a.py"], "service": None}],
        "messages": [{"from": "</details>", "to": "</details>", "label": "a|b</details>",
                      "kind": "self", "changed": True,
                      "evidence": {"file": "a|b`.py", "line": 7, "symbol": "x"}}],
        "blocks": [],
    }
    grounding = {"elements": [{**_check("message", "0", final_index=0,
                                        defined_at="`x`|y.py:1</details>")}],
                 "summary": {"proposed": 1, "grounded_first_pass": 1, "repaired": 0, "pruned": 0},
                 "capped": {}, "root_range": None}
    blocks = render_diagram_blocks({"sequence": _rendered(spec, grounding)})
    # Exactly the two folds the renderer opened, and no injected close tag.
    assert blocks.count("<details>") == 2
    assert blocks.count("</details>") == 2
    table = [line for line in blocks.split("\n") if line.startswith("| 1 |")]
    assert len(table) == 1
    assert table[0].count("|") == 5   # four cells, no cell break-out
    assert "`ab.py:7`" in table[0]    # markdown cells drop backticks and pipes
    assert "`xy.py:1/details`" in table[0]


def test_every_golden_line_matches_its_kind_grammar() -> None:
    for line in render_sequence_mermaid(SEQUENCE_SPEC).split("\n"):
        assert SEQUENCE_LINE_GRAMMAR.fullmatch(line), line
    for line in render_flowchart_mermaid(FLOWCHART_SPEC).split("\n"):
        assert FLOWCHART_LINE_GRAMMAR.fullmatch(line), line
    # The grammars are exhaustive per kind: a flowchart line is not a sequence
    # line and vice versa, so a cross-kind leak would be caught.
    assert not SEQUENCE_LINE_GRAMMAR.fullmatch("flowchart TD")
    assert not FLOWCHART_LINE_GRAMMAR.fullmatch("sequenceDiagram")
    for bad in ("    P1->>P2: a; b", "    P1->>P2: a`b", "    P1->>P2: a|b", "click P1 href \"x\""):
        assert not SEQUENCE_LINE_GRAMMAR.fullmatch(bad), bad
    for bad in ("    N1[a] --> N2[b] --> N3[c]", "    N1[a;b]", "    click N1 href \"x\""):
        assert not FLOWCHART_LINE_GRAMMAR.fullmatch(bad), bad


# ---------------------------------------------------------------------------
# Block structure edge cases
# ---------------------------------------------------------------------------


def test_opt_and_loop_blocks_and_a_message_outside_every_block() -> None:
    spec: dict[str, Any] = {
        "participants": [{"name": "A", "kind": "internal", "files": ["a.py"], "service": None},
                         {"name": "B", "kind": "internal", "files": ["b.py"], "service": None}],
        "messages": [
            {"from": "A", "to": "B", "label": "m0", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}},
            {"from": "A", "to": "B", "label": "m1", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 2, "symbol": "x"}},
            {"from": "B", "to": "A", "label": "m2", "kind": "reply", "changed": False,
             "evidence": {"file": "b.py", "line": 3, "symbol": "x"}},
            {"from": "A", "to": "A", "label": "m3", "kind": "self", "changed": True,
             "evidence": {"file": "a.py", "line": 4, "symbol": "x"}},
        ],
        "blocks": [
            {"kind": "opt", "branches": [{"condition": "cached", "evidence": {"file": "a.py", "line": 1},
                                          "messages": [1]}]},
            {"kind": "loop", "branches": [{"condition": "each page",
                                           "evidence": {"file": "a.py", "line": 2}, "messages": [2]}]},
        ],
    }
    assert render_sequence_mermaid(spec).split("\n") == [
        "sequenceDiagram",
        "    participant P1 as A",
        "    participant P2 as B",
        "    P1->>P2: m0",
        "    opt cached",
        "        P1->>P2: m1",
        "    end",
        "    loop each page",
        "        P2-->>P1: m2",
        "    end",
        "    P1->>P1: m3",
    ]


def test_malformed_block_indices_and_unknown_endpoints_degrade_without_raising() -> None:
    spec: dict[str, Any] = {
        "participants": [{"name": "A", "kind": "internal", "files": ["a.py"], "service": None}],
        "messages": [
            {"from": "A", "to": "Ghost", "label": "m0", "kind": "call", "changed": True,
             "evidence": {"file": "a.py", "line": 1, "symbol": "x"}},
            {"from": "A", "to": "A", "label": "m1", "kind": "self", "changed": True,
             "evidence": {"file": "a.py", "line": 2, "symbol": "x"}},
        ],
        # Out-of-range, non-int, and duplicate-claim indices are all ignored.
        "blocks": [{"kind": "nope", "branches": [{"condition": "c", "evidence": {}, "messages": [99, "1", 1]}]},
                   {"kind": "alt", "branches": [{"condition": "d", "evidence": {}, "messages": [1]}]}],
    }
    assert render_sequence_mermaid(spec).split("\n") == [
        "sequenceDiagram",
        "    participant P1 as A",
        # m0's target is not a declared participant -> no arrow, but the block
        # opened by the message that IS claimed still closes cleanly.
        "    opt c",
        "        P1->>P1: m1",
        "    end",
    ]


def test_empty_labels_fall_back_to_a_placeholder() -> None:
    spec: dict[str, Any] = {
        "participants": [{"name": "", "kind": "internal", "files": [], "service": None}],
        "messages": [], "blocks": [],
    }
    assert render_sequence_mermaid(spec) == "sequenceDiagram\n    participant P1 as unlabeled"
    flow: dict[str, Any] = {"root": {}, "nodes": [{"id": "a", "kind": "process", "label": "```",
                                                   "evidence": {}}], "edges": []}
    assert render_flowchart_mermaid(flow) == "flowchart TD\n    N1[unlabeled]"


def test_unknown_node_kind_falls_back_to_the_process_shape_and_word() -> None:
    flow: dict[str, Any] = {
        "root": {"file": "a.py", "name": "f", "line": 1},
        "nodes": [{"id": "a", "kind": "mystery", "label": "x", "evidence": {"file": "a.py", "line": 1}}],
        "edges": [],
    }
    assert render_flowchart_mermaid(flow) == "flowchart TD\n    N1[x]"
    grounding: dict[str, Any] = {"elements": [], "summary": {"proposed": 1, "grounded_first_pass": 1,
                                             "repaired": 0, "pruned": 0},
                 "capped": {}, "root_range": None}
    assert "| N1 | process | `a.py:1` |" in render_diagram_blocks({"flowchart": _rendered(flow, grounding)})


def test_subroutine_row_without_a_definition_site_drops_that_clause() -> None:
    flow: dict[str, Any] = {
        "root": {"file": "a.py", "name": "f", "line": 1},
        "nodes": [{"id": "a", "kind": "subroutine", "label": "helper",
                   "evidence": {"file": "a.py", "line": 9, "symbol": "helper"}},
                  {"id": "b", "kind": "subroutine", "label": "anon",
                   "evidence": {"file": "a.py", "line": 10, "symbol": None}}],
        "edges": [],
    }
    grounding = {"elements": [], "summary": {"proposed": 2, "grounded_first_pass": 2,
                                             "repaired": 0, "pruned": 0},
                 "capped": {}, "root_range": [1, 20]}
    blocks = render_diagram_blocks({"flowchart": _rendered(flow, grounding)})
    assert "| N1 | subroutine `helper`, called at `a.py:9` |  |" in blocks
    assert "| N2 | subroutine, called at `a.py:10` |  |" in blocks


# ---------------------------------------------------------------------------
# Omission notice
# ---------------------------------------------------------------------------


def test_omission_notice_reports_floor_codes_and_counts() -> None:
    result: dict[str, Any] = {"status": "omitted", "reason": None, "spec_proposed": {}, "spec_final": {},
              "grounding": {"elements": [],
                            "summary": {"proposed": 6, "grounded_first_pass": 2,
                                        "repaired": 1, "pruned": 3},
                            "capped": {"messages": 2}, "root_range": None},
              "omit_reasons": ["TOO_FEW_MESSAGES", "NO_CHANGED_INTERACTION"], "mermaid": None}
    notice = render_omission_notice("sequence", result)
    assert notice == (
        "No sequence diagram was rendered for this pull request. "
        "Grounding floor not met: TOO_FEW_MESSAGES, NO_CHANGED_INTERACTION. "
        "6 elements proposed, 2 grounded on the first pass, 1 repaired, 3 dropped as ungrounded. "
        "2 elements were trimmed to fit the diagram cap."
    )
    assert "\n" not in notice


def test_omission_notice_covers_skipped_failed_and_rendered() -> None:
    assert render_omission_notice("flowchart", {"status": "rendered"}) == ""
    assert render_omission_notice("flowchart", {"status": "skipped", "reason": "no candidate root",
                                                "grounding": None, "omit_reasons": []}) == (
        "No flowchart was rendered for this pull request. Reason: no candidate root."
    )
    # A failure reason is model/exception text: sanitized into one safe line.
    assert render_omission_notice("flowchart", {"status": "failed",
                                                "reason": "backend `boom`\nline two",
                                                "grounding": None, "omit_reasons": []}) == (
        "No flowchart was rendered for this pull request. Reason: backend boom line two."
    )
    # A singular cap clause, and an unknown kind degrades to a generic phrase.
    assert render_omission_notice("weird", {"status": "omitted", "grounding": {
        "summary": {"proposed": 1, "grounded_first_pass": 1, "repaired": 0, "pruned": 0},
        "capped": {"nodes": 1}}, "omit_reasons": []}) == (
        "No weird was rendered for this pull request. "
        "1 element proposed, 1 grounded on the first pass, 0 repaired, 0 dropped as ungrounded. "
        "1 element was trimmed to fit the diagram cap."
    )


# ---------------------------------------------------------------------------
# render_report / insert_diagrams_section (deep/render.py)
# ---------------------------------------------------------------------------


def _items() -> list[dict[str, Any]]:
    return [{"id": 1, "lens": "per-stack", "file": "a.py", "line": 9, "description": "bug"},
            {"id": 2, "lens": "cross-stack", "file": "b.py", "line": 2, "description": "drift"}]


def test_render_report_inserts_the_diagrams_section_directly_after_the_review_heading() -> None:
    blocks = render_diagram_blocks(_both_rendered())
    report = render_report(_items(), diagram_blocks=blocks)
    lines = report.split("\n")
    assert lines[0] == "# Review"
    assert lines[1] == ""
    assert lines[2] == "## Diagrams"
    assert lines[3] == "<details><summary><h3>Sequence Diagram</h3></summary>"
    assert report.index("## Diagrams") < report.index("## Issues") < report.index("## Cross-Stack Issues")
    assert report.endswith("\n")
    # One code path: the kwarg and the textual re-apply produce identical bytes.
    assert report == insert_diagrams_section(render_report(_items()), blocks)


def test_insert_diagrams_section_is_idempotent_and_replaces_an_existing_section() -> None:
    blocks = render_diagram_blocks(_both_rendered())
    once = insert_diagrams_section(render_report(_items()), blocks)
    assert insert_diagrams_section(once, blocks) == once
    assert once.count("## Diagrams") == 1
    replaced = insert_diagrams_section(once, "<details>NEW</details>")
    assert "Sequence Diagram" not in replaced
    assert "<details>NEW</details>" in replaced
    assert replaced == insert_diagrams_section(render_report(_items()), "<details>NEW</details>")
    # Empty blocks remove a stale section and restore the plain report.
    assert insert_diagrams_section(once, "") == render_report(_items())
    assert insert_diagrams_section(once, "   \n\n ") == render_report(_items())


def test_diagram_section_survives_a_round_trip_through_the_report_and_back() -> None:
    blocks = render_diagram_blocks(_both_rendered())
    report = render_report(_items(), diagram_blocks=blocks)
    # The blocks are embedded verbatim -- the section body is byte-identical.
    assert f"## Diagrams\n{blocks}\n" in report
    assert render_sequence_mermaid(SEQUENCE_SPEC) in report
    assert render_flowchart_mermaid(FLOWCHART_SPEC) in report

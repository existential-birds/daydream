"""Real-path integration tests for grounded diagrams in the review flows (#1113).

Every test enters through ``daydream.runner.run`` against a real temporary git
repository, with the stub backend at the ``create_backend`` seam as the only
mock. Assertions are on observable outcomes: the on-disk ``diagram.json``
artifact, the bytes of the review comment GitHub would receive, the rendered
``review-output.md``, the process exit code, and the backend call log.

Spec test coverage (issue #1113 "Tests"): 1, 2, 3, 4, 5, 6, 7 (review half), 8,
9, 10, 11 and 14 (review half). The ``--diagram-only`` halves of 7, 12, 13 and
14 live in ``tests/test_diagram_only_integration.py``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.harness import diagram_repos as dr
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

# --- Expected renderer output (goldens for these fixtures) -------------------

SEQUENCE_GOLDEN = """sequenceDiagram
    participant P1 as Client
    participant P2 as Core
    participant P3 as Util
    P1->>P3: Normalize payload
    P3-->>P1: Stripped text
    P1->>P2: Handle cleaned payload
    P2-->>P1: Cleaned payload
    P2->>P2: Normalize inside handler"""

FLOWCHART_GOLDEN = """flowchart TD
    N1([run]) --> N2{payload is None?}
    N2 -->|yes| N3([Return empty])
    N2 -->|no| N4{fast mode?}
    N4 -->|yes| N5[[fast_path]]
    N4 -->|no| N6[Scan items]
    N5 --> N7([Return none])
    N6 --> N7"""

SEQUENCE_HEADING = "<details><summary><h3>Sequence Diagram</h3></summary>"
FLOWCHART_HEADING = "<details><summary><h3>Flowchart</h3></summary>"


# --- Harness -----------------------------------------------------------------


@dataclass
class _CapturedPost:
    """Every review payload the run tried to submit."""

    payloads: list[dict[str, Any]] = field(default_factory=list)

    def body(self) -> str:
        assert self.payloads, "no review payload was submitted"
        return str(self.payloads[-1]["body"])


@pytest.fixture
def captured_post(monkeypatch: pytest.MonkeyPatch) -> _CapturedPost:
    """Let the real ``build_payload`` run and capture the review it would POST.

    Patches only the PR lookup and the ``gh`` review submission, so the summary
    renderer, the diagram slot, and every marker are produced by production
    code exactly as they would be on a live PR.
    """
    from daydream import pr_review

    captured = _CapturedPost()
    fake_pr = pr_review.PRInfo(
        number=123,
        head_sha="a" * 40,
        base_sha="b" * 40,
        base_ref="main",
        owner="acme",
        repo="widgets",
        url="https://example/pr/123",
    )
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _target: fake_pr)

    def _capture(
        _target: Path, _pr: pr_review.PRInfo, payload: dict[str, Any]
    ) -> tuple[str, None]:
        captured.payloads.append(payload)
        return "https://example/pr/123#review-1", None

    monkeypatch.setattr("daydream.pr_review._submit_review", _capture)
    return captured


@pytest.fixture
def review_run(
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Any],
    silence_console: Callable[..., None],
    captured_post: _CapturedPost,
) -> Callable[..., Any]:
    """Run a full ``--comment`` deep review with a diagram-scripted stub backend.

    Depends on ``captured_post`` unconditionally: comment mode treats a failed
    PR post as a run failure (exit 1), so a test that forgot the fixture would
    be asserting on the wrong exit code for the wrong reason.
    """
    for module in (
        "daydream.deep.orchestrator",
        "daydream.phases",
        "daydream.runner",
        "daydream.pr_review",
    ):
        silence_console(module)
    silence(monkeypatch)

    async def _run(
        target: Path,
        *,
        specs: dict[str, list[dict[str, Any]]] | None = None,
        emit_reads: bool = True,
        unread: frozenset[str] = frozenset(),
        reads: dict[str, list[str]] | None = None,
        session_id: str | None = None,
        fail: frozenset[str] = frozenset(),
        **config_overrides: Any,
    ) -> tuple[int, StubBackend]:
        stub = install_stub_backend(monkeypatch, target)
        stub.diagram_specs = specs or {}
        stub.diagram_emit_reads = emit_reads
        stub.diagram_unread = unread
        stub.diagram_reads = reads or {}
        stub.diagram_session_id = session_id
        stub.diagram_fail = fail
        config_overrides.setdefault("output_mode", "comment")
        return await _dispatch_run(make_config(target, **config_overrides)), stub

    return _run


async def _dispatch_run(config: Any) -> int:
    from daydream.runner import run

    return await run(config)


def _artifact(target: Path) -> dict[str, Any]:
    """Load ``.daydream/deep/diagram.json``."""
    path = target / ".daydream" / "deep" / "diagram.json"
    assert path.is_file(), f"diagram artifact missing at {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _diagram_calls(stub: StubBackend, kind: str) -> list[dict[str, Any]]:
    """The stub calls that are diagram turns for ``kind`` (author + repair)."""
    role = (
        "You are the sequence-diagram author"
        if kind == "sequence"
        else "You are the flowchart author"
    )
    repair = f"Diagram repair turn ({kind}):"
    return [
        call
        for call in stub.calls
        if role in call["prompt"] or repair in call["prompt"]
    ]


def _reasons(result: dict[str, Any]) -> set[str]:
    """Every non-None reason code in a result's grounding elements."""
    grounding = result["grounding"]
    assert grounding is not None
    return {
        check["reason"]
        for check in grounding["elements"]
        if check["reason"] is not None
    }


def _assert_grammar(mermaid: str, kind: str) -> None:
    """Every emitted line must match the kind's exported line grammar."""
    from daydream.deep.diagram_render import FLOWCHART_LINE_GRAMMAR, SEQUENCE_LINE_GRAMMAR

    grammar = SEQUENCE_LINE_GRAMMAR if kind == "sequence" else FLOWCHART_LINE_GRAMMAR
    for line in mermaid.split("\n"):
        assert grammar.fullmatch(line), f"{kind} mermaid emitted an off-grammar line: {line!r}"


# --- Spec test 1: sequence auto trigger -------------------------------------


async def test_sequence_auto_trigger_renders_grounded_diagram(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """A cross-module diff renders the sequence diagram and skips the flowchart."""
    target = dr.build_cross_module_repo(tmp_path)

    exit_code, stub = await review_run(
        target, specs={"sequence": [dr.sequence_spec()]}
    )

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["eligibility"]["sequence"]["rule"] == "cross-module"
    sequence = artifact["results"]["sequence"]
    flowchart = artifact["results"]["flowchart"]
    assert sequence["status"] == "rendered"
    assert flowchart["status"] == "skipped"
    assert flowchart["reason"]
    assert sequence["grounding"]["summary"] == {
        "proposed": 8,
        "grounded_first_pass": 8,
        "repaired": 0,
        "pruned": 0,
    }
    # Only the sequence author turn ran; the skipped kind cost nothing.
    calls = _diagram_calls(stub, "sequence")
    assert len(calls) == 1
    assert _diagram_calls(stub, "flowchart") == []

    # Wire contract: the author agent is read-only, answers the kind's schema,
    # gets no ``max_turns`` (which would fail hard rather than soft), and runs
    # under no fan-out ``agents=`` definition.
    from daydream.deep.diagram_schema import SEQUENCE_SPEC_SCHEMA

    assert calls[0]["read_only"] is True
    assert calls[0]["output_schema"] is SEQUENCE_SPEC_SCHEMA
    assert calls[0]["max_turns"] is None
    assert calls[0]["agents"] is None

    assert sequence["mermaid"] == SEQUENCE_GOLDEN
    _assert_grammar(sequence["mermaid"], "sequence")

    body = captured_post.body()
    assert SEQUENCE_HEADING in body
    assert FLOWCHART_HEADING not in body
    # The block sits directly under the summary header, before any findings.
    header = "**Code Review Summary**"
    assert body.index(header) < body.index(SEQUENCE_HEADING)
    assert body[body.index(header) + len(header) :].lstrip().startswith(SEQUENCE_HEADING)
    assert SEQUENCE_GOLDEN in body
    assert "| # | Interaction | Call site | Callee defined at |" in body

    report = (target / ".review-output.md").read_text(encoding="utf-8")
    assert "## Diagrams" in report
    assert SEQUENCE_HEADING in report
    # The section is inserted directly after ``# Review`` and leaves every
    # other section -- including the ``## Coverage`` block ``load-items``
    # appended before this step ran -- exactly where it was.
    assert report.index("# Review") < report.index("## Diagrams")
    assert report.index("## Diagrams") < report.index("## Coverage")
    assert report.index("## Diagrams") < report.index("## Issues")
    deep_report = (target / ".daydream" / "deep" / "review-output.md").read_text(
        encoding="utf-8"
    )
    assert SEQUENCE_HEADING in deep_report
    # diagram.md carries the same rendered blocks.
    assert SEQUENCE_HEADING in (
        target / ".daydream" / "deep" / "diagram.md"
    ).read_text(encoding="utf-8")


# --- Spec test 2: flowchart auto trigger ------------------------------------


async def test_flowchart_auto_trigger_renders_grounded_diagram(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """A branch-heavy single-module diff renders the flowchart and skips sequence."""
    target = dr.build_branch_heavy_repo(tmp_path)

    exit_code, stub = await review_run(
        target, specs={"flowchart": [dr.flowchart_spec()]}
    )

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["eligibility"]["flowchart"]["rule"] == "branch-points"
    assert artifact["eligibility"]["candidate_roots"] == [
        {
            "file": "app/pipeline.py",
            "name": "run",
            "line": 1,
            "end_line": 9,
            "branch_points": 4,
        }
    ]
    flowchart = artifact["results"]["flowchart"]
    assert flowchart["status"] == "rendered"
    assert artifact["results"]["sequence"]["status"] == "skipped"
    assert flowchart["grounding"]["root_range"] == [1, 9]
    assert flowchart["mermaid"] == FLOWCHART_GOLDEN
    _assert_grammar(flowchart["mermaid"], "flowchart")
    assert len(_diagram_calls(stub, "flowchart")) == 1
    assert _diagram_calls(stub, "sequence") == []

    body = captured_post.body()
    assert FLOWCHART_HEADING in body
    assert SEQUENCE_HEADING not in body
    assert FLOWCHART_GOLDEN in body
    assert "Control flow of `run` (`app/pipeline.py:1-9`)" in body
    assert "| Node | Statement | Location |" in body


# --- Spec test 3: both kinds -------------------------------------------------


async def test_both_signals_render_sequence_first(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """A diff carrying both signals renders both blocks, sequence first."""
    target = dr.build_both_signals_repo(tmp_path)

    exit_code, stub = await review_run(
        target,
        specs={
            "sequence": [dr.sequence_spec()],
            "flowchart": [dr.flowchart_spec(root_file="pkg_b/client.py", offset=10)],
        },
    )

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["results"]["sequence"]["status"] == "rendered"
    assert artifact["results"]["flowchart"]["status"] == "rendered"
    # One author session per kind, run as siblings.
    assert len(_diagram_calls(stub, "sequence")) == 1
    assert len(_diagram_calls(stub, "flowchart")) == 1

    body = captured_post.body()
    assert body.index(SEQUENCE_HEADING) < body.index(FLOWCHART_HEADING)
    assert "5 interactions across 3 components, each grounded to a cited call site." in body
    assert "Control flow of `run` (`pkg_b/client.py:11-19`): 7 nodes" in body


# --- Spec test 4: repair then prune -----------------------------------------

_FABRICATED = [
    {
        "from": "Client",
        "to": "Core",
        "label": "Ghost call",
        "kind": "call",
        "changed": True,
        # Nonexistent file -> FILE_MISSING.
        "evidence": {"file": "pkg_b/ghost.py", "line": 3, "symbol": "handle"},
    },
    {
        "from": "Client",
        "to": "Core",
        "label": "Unsnappable symbol",
        "kind": "call",
        "changed": True,
        # Real file and line, symbol nowhere within the +/-3 snap window.
        "evidence": {"file": "pkg_b/client.py", "line": 5, "symbol": "missing_fn"},
    },
    {
        "from": "Client",
        "to": "Util",
        "label": "Undefined callee",
        "kind": "call",
        "changed": True,
        # ``handle`` IS on line 7 but is not defined in the Util participant.
        "evidence": {"file": "pkg_b/client.py", "line": 7, "symbol": "handle"},
    },
]


def _fabricated_sequence_turns() -> list[dict[str, Any]]:
    """Turn 1 with three fabricated messages; the repair fixes exactly one."""
    turn_one = dr.sequence_spec()
    turn_one["messages"].extend(copy.deepcopy(_FABRICATED))
    repair = dr.sequence_spec()
    repair["messages"].extend(copy.deepcopy(_FABRICATED))
    repair["messages"][5]["evidence"] = {
        "file": "pkg_b/client.py",
        "line": 7,
        "symbol": "handle",
    }
    return [turn_one, repair]


async def test_fabricated_sequence_evidence_is_repaired_then_pruned(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Three ungrounded messages: one repaired, two pruned out of the diagram."""
    target = dr.build_cross_module_repo(tmp_path)

    exit_code, stub = await review_run(
        target,
        specs={"sequence": _fabricated_sequence_turns()},
        session_id="diagram-session-1",
    )

    assert exit_code == 0
    calls = _diagram_calls(stub, "sequence")
    assert len(calls) == 2, "exactly one author turn plus one repair turn"
    assert calls[1]["continuation"] is not None
    assert calls[1]["continuation"].data == {"session_id": "diagram-session-1"}
    assert "SYMBOL_NOT_ON_LINE" in calls[1]["prompt"]
    assert "CALLEE_NOT_DEFINED_IN_TARGET" in calls[1]["prompt"]

    sequence = _artifact(target)["results"]["sequence"]
    assert sequence["status"] == "rendered"
    assert sequence["grounding"]["summary"] == {
        "proposed": 11,
        "grounded_first_pass": 8,
        "repaired": 1,
        "pruned": 2,
    }
    assert _reasons(sequence) == {"SYMBOL_NOT_ON_LINE", "CALLEE_NOT_DEFINED_IN_TARGET"}

    mermaid = sequence["mermaid"]
    assert "Ghost call" in mermaid, "the repaired message must be drawn"
    assert "Unsnappable symbol" not in mermaid
    assert "Undefined callee" not in mermaid
    _assert_grammar(mermaid, "sequence")

    body = captured_post.body()
    assert "2 proposed interactions were dropped as ungrounded." in body
    # The evidence table lists only the rendered (grounded) rows.
    assert body.count("| 6 | Client → Core: Ghost call |") == 1
    assert "Unsnappable symbol" not in body


# --- Spec test 5: flowchart grounding ---------------------------------------


def _flowchart_grounding_turns() -> list[dict[str, Any]]:
    """Turn 1 roots outside the candidate list; the repair re-picks and offends."""
    wrong_root = dr.flowchart_spec()
    wrong_root["root"] = {"file": "app/pipeline.py", "name": "fast_path", "line": 12}

    repair = dr.flowchart_spec()
    repair["nodes"].extend(
        [
            {
                "id": "bad_out",
                "kind": "process",
                "label": "Outside root",
                # Line 13 is inside ``fast_path``, not inside ``run``.
                "evidence": {"file": "app/pipeline.py", "line": 13, "symbol": None},
            },
            {
                "id": "bad_dec",
                "kind": "decision",
                "label": "Not a branch",
                # Line 3 is a return statement.
                "evidence": {"file": "app/pipeline.py", "line": 3, "symbol": None},
            },
            {
                "id": "bad_sub_call",
                "kind": "subroutine",
                "label": "ghost_call",
                "evidence": {
                    "file": "app/pipeline.py",
                    "line": 8,
                    "symbol": "ghost_call",
                },
            },
            {
                "id": "bad_sub_def",
                "kind": "subroutine",
                "label": "item",
                # ``item`` IS the token on line 8, but nothing defines it.
                "evidence": {"file": "app/pipeline.py", "line": 8, "symbol": "item"},
            },
            {
                "id": "d3",
                "kind": "decision",
                "label": "item truthy?",
                "evidence": {"file": "app/pipeline.py", "line": 7, "symbol": None},
            },
            {
                "id": "orphan",
                "kind": "process",
                "label": "Unreachable",
                "evidence": {"file": "app/pipeline.py", "line": 9, "symbol": None},
            },
        ]
    )
    repair["edges"].extend(
        [
            {"from": "p1", "to": "d3", "label": None},
            # d3's only labeled outgoing edge -> demoted to a plain process.
            {"from": "d3", "to": "e2", "label": "yes"},
            {"from": "p1", "to": "bad_out", "label": None},
            {"from": "p1", "to": "bad_dec", "label": None},
            {"from": "p1", "to": "bad_sub_call", "label": None},
            {"from": "p1", "to": "bad_sub_def", "label": None},
        ]
    )
    return [wrong_root, repair]


async def test_flowchart_grounding_prunes_repairs_and_demotes(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Every flowchart reason code fires, offenders prune, a thin decision demotes."""
    target = dr.build_branch_heavy_repo(tmp_path)

    exit_code, stub = await review_run(
        target,
        specs={"flowchart": _flowchart_grounding_turns()},
        session_id="diagram-session-2",
    )

    assert exit_code == 0
    calls = _diagram_calls(stub, "flowchart")
    assert len(calls) == 2
    # The repair turn is asked to re-pick a root from the candidate list.
    assert "ROOT_NOT_CANDIDATE" in calls[1]["prompt"]
    assert "app/pipeline.py" in calls[1]["prompt"]

    flowchart = _artifact(target)["results"]["flowchart"]
    assert flowchart["status"] == "rendered"
    assert _reasons(flowchart) == {
        "NODE_OUTSIDE_ROOT",
        "NOT_A_BRANCH_STATEMENT",
        "SUBROUTINE_NOT_CALLED_HERE",
        "SUBROUTINE_NOT_DEFINED",
        "EDGE_ENDPOINT_UNGROUNDED",
    }
    final_kinds = {node["id"]: node["kind"] for node in flowchart["spec_final"]["nodes"]}
    assert "bad_out" not in final_kinds
    assert "bad_dec" not in final_kinds
    assert "bad_sub_call" not in final_kinds
    assert "bad_sub_def" not in final_kinds
    assert "orphan" not in final_kinds, "a node unreachable from start is dropped"
    assert final_kinds["d3"] == "process", "one labeled edge is not a decision"
    assert final_kinds["d1"] == "decision"

    mermaid = flowchart["mermaid"]
    assert "N8[item truthy?]" in mermaid, "the demoted decision renders as a process box"
    assert "Outside root" not in mermaid
    assert "Unreachable" not in mermaid
    _assert_grammar(mermaid, "flowchart")
    assert FLOWCHART_HEADING in captured_post.body()


# --- Spec test 6: read receipts ---------------------------------------------


async def test_unread_evidence_file_omits_sequence_when_pairs_break(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Unread calls also invalidate their dependent replies, below the render floor."""
    target = dr.build_cross_module_repo(tmp_path)

    exit_code, _ = await review_run(
        target,
        specs={"sequence": [dr.sequence_spec()]},
        unread=frozenset({"pkg_b/client.py"}),
    )

    assert exit_code == 0
    sequence = _artifact(target)["results"]["sequence"]
    assert sequence["status"] == "omitted"
    assert _reasons(sequence) == {
        "FILE_NOT_READ_BY_MODEL",
        "REPLY_NOT_PRECEDED_BY_CALL",
    }
    assert sequence["grounding"]["summary"]["pruned"] == 4
    assert sequence["omit_reasons"] == ["TOO_FEW_MESSAGES", "TOO_FEW_PARTICIPANTS"]
    assert sequence["mermaid"] is None
    assert SEQUENCE_HEADING not in captured_post.body()


async def test_unread_root_file_omits_the_flowchart(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """With no read receipts at all the flowchart is omitted, never rendered."""
    target = dr.build_branch_heavy_repo(tmp_path)

    exit_code, _ = await review_run(
        target,
        specs={"flowchart": [dr.flowchart_spec()]},
        emit_reads=False,
    )

    assert exit_code == 0
    flowchart = _artifact(target)["results"]["flowchart"]
    assert flowchart["status"] == "omitted"
    assert "FILE_NOT_READ_BY_MODEL" in _reasons(flowchart)
    assert set(flowchart["omit_reasons"]) == {"TOO_FEW_NODES", "NO_END", "NO_DECISION"}
    assert flowchart["mermaid"] is None
    assert FLOWCHART_HEADING not in captured_post.body()


# --- Spec test 7: omission floors -------------------------------------------


async def test_thin_sequence_is_omitted_and_flowchart_unaffected(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Two surviving messages is below the floor: no block, the other kind is fine."""
    target = dr.build_both_signals_repo(tmp_path)
    thin = dr.sequence_spec()
    thin["messages"] = thin["messages"][:2]

    exit_code, _ = await review_run(
        target,
        specs={
            "sequence": [thin],
            "flowchart": [dr.flowchart_spec(root_file="pkg_b/client.py", offset=10)],
        },
    )

    assert exit_code == 0
    results = _artifact(target)["results"]
    assert results["sequence"]["status"] == "omitted"
    assert results["sequence"]["omit_reasons"] == ["TOO_FEW_MESSAGES"]
    assert results["flowchart"]["status"] == "rendered"

    body = captured_post.body()
    assert SEQUENCE_HEADING not in body
    assert FLOWCHART_HEADING in body
    report = (target / ".review-output.md").read_text(encoding="utf-8")
    assert SEQUENCE_HEADING not in report
    assert FLOWCHART_HEADING in report


async def test_flowchart_without_a_decision_is_omitted(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """A flowchart whose only decisions fail grounding falls below its floor."""
    target = dr.build_branch_heavy_repo(tmp_path)
    spec = dr.flowchart_spec()
    for node in spec["nodes"]:
        if node["kind"] == "decision":
            # A return statement is not a branch statement.
            node["evidence"]["line"] = 3

    exit_code, _ = await review_run(target, specs={"flowchart": [spec]})

    assert exit_code == 0
    flowchart = _artifact(target)["results"]["flowchart"]
    assert flowchart["status"] == "omitted"
    assert "NO_DECISION" in flowchart["omit_reasons"]
    assert "NOT_A_BRANCH_STATEMENT" in _reasons(flowchart)
    assert FLOWCHART_HEADING not in captured_post.body()


# --- Spec test 8: below threshold -------------------------------------------


async def test_below_threshold_records_signals_without_any_agent_call(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Nothing eligible: both kinds skipped, zero diagram turns, no blocks."""
    target = dr.build_flat_repo(tmp_path)

    exit_code, stub = await review_run(target)

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["results"]["sequence"]["status"] == "skipped"
    assert artifact["results"]["flowchart"]["status"] == "skipped"
    # The signals behind the decision are recorded for audit.
    eligibility = artifact["eligibility"]
    assert eligibility["code_files"] == ["app/one.py", "app/two.py"]
    assert eligibility["modules"] == {"app/one.py": "app", "app/two.py": "app"}
    assert eligibility["cross_module_edges"] == 0
    assert eligibility["candidate_roots"] == []
    assert eligibility["thresholds"] == {
        "min_code_files": 3,
        "min_modules": 2,
        "min_branch_points": 3,
    }
    assert _diagram_calls(stub, "sequence") == []
    assert _diagram_calls(stub, "flowchart") == []

    body = captured_post.body()
    assert SEQUENCE_HEADING not in body
    assert FLOWCHART_HEADING not in body
    assert "## Diagrams" not in (target / ".review-output.md").read_text(encoding="utf-8")


# --- Spec test 9: cross-service trigger -------------------------------------


async def test_cross_service_trigger_fires_without_an_import_edge(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Two manifest-bearing services, no import edge: the cross-service rule fires."""
    target = dr.build_cross_service_repo(tmp_path)

    exit_code, _ = await review_run(
        target, specs={"sequence": [dr.cross_service_sequence_spec()]}
    )

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["eligibility"]["sequence"]["rule"] == "cross-service"
    assert artifact["eligibility"]["cross_module_edges"] == 0
    assert artifact["eligibility"]["services"] == {
        "services/alpha/api.py": "alpha",
        "services/beta/api.py": "beta",
    }
    assert artifact["results"]["sequence"]["status"] == "rendered"
    assert SEQUENCE_HEADING in captured_post.body()


# --- Spec test 10: force flags and config -----------------------------------


async def test_diagram_sequence_forces_the_kind_on_a_flat_diff(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    """``--diagram sequence`` forces sequence eligible and leaves flowchart skipped."""
    target = dr.build_flat_repo(tmp_path)

    exit_code, stub = await review_run(target, diagram="sequence")

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["eligibility"]["sequence"] == {
        "eligible": True,
        "rule": "forced",
        "reason": "Forced eligible: diagram mode 'sequence' names this kind.",
    }
    assert artifact["results"]["flowchart"]["status"] == "skipped"
    # Forcing changes eligibility, never verification: the empty spec the stub
    # returns for an unscripted kind still has to clear the floor, and does not.
    assert artifact["results"]["sequence"]["status"] == "omitted"
    assert len(_diagram_calls(stub, "sequence")) == 1


async def test_diagram_flowchart_forced_offers_every_changed_function(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    """With no function meeting the threshold, every changed function is a candidate."""
    target = dr.build_cross_module_repo(tmp_path)

    exit_code, stub = await review_run(target, diagram="flowchart")

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["eligibility"]["flowchart"]["rule"] == "forced"
    names = {root["name"] for root in artifact["eligibility"]["candidate_roots"]}
    assert names == {"handle", "normalize_payload", "normalize", "call_handle"}
    assert artifact["results"]["sequence"]["status"] == "skipped"
    prompt = _diagram_calls(stub, "flowchart")[0]["prompt"]
    assert "call_handle" in prompt


async def test_diagram_both_forces_both_kinds(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    """``--diagram both`` runs an author turn for each kind on a flat diff."""
    target = dr.build_flat_repo(tmp_path)

    exit_code, stub = await review_run(target, diagram="both")

    assert exit_code == 0
    eligibility = _artifact(target)["eligibility"]
    assert eligibility["sequence"]["rule"] == "forced"
    assert eligibility["flowchart"]["rule"] == "forced"
    assert len(_diagram_calls(stub, "sequence")) == 1
    assert len(_diagram_calls(stub, "flowchart")) == 1


async def test_diagram_off_suppresses_a_complex_diff(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """``--diagram off`` writes no artifact at all and makes no diagram call."""
    target = dr.build_cross_module_repo(tmp_path)

    exit_code, stub = await review_run(
        target, specs={"sequence": [dr.sequence_spec()]}, diagram="off"
    )

    assert exit_code == 0
    assert not (target / ".daydream" / "deep" / "diagram.json").exists()
    assert _diagram_calls(stub, "sequence") == []
    assert SEQUENCE_HEADING not in captured_post.body()


async def test_file_config_mode_off_suppresses_diagrams(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    """``[tool.daydream.diagram] mode = "off"`` suppresses without a CLI flag."""
    from daydream.config_file import DaydreamFileConfig

    target = dr.build_cross_module_repo(tmp_path)

    exit_code, stub = await review_run(
        target,
        specs={"sequence": [dr.sequence_spec()]},
        file_config=DaydreamFileConfig(diagram_mode="off"),
    )

    assert exit_code == 0
    assert not (target / ".daydream" / "deep" / "diagram.json").exists()
    assert _diagram_calls(stub, "sequence") == []


async def test_cli_diagram_both_overrides_file_config_off(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    """The CLI flag outranks the repository file's off switch."""
    from daydream.config_file import DaydreamFileConfig

    target = dr.build_cross_module_repo(tmp_path)

    exit_code, _ = await review_run(
        target,
        specs={"sequence": [dr.sequence_spec()]},
        diagram="both",
        file_config=DaydreamFileConfig(diagram_mode="off"),
    )

    assert exit_code == 0
    eligibility = _artifact(target)["eligibility"]
    assert eligibility["force"] == "both"
    assert eligibility["sequence"]["eligible"] is True
    assert eligibility["flowchart"]["eligible"] is True


async def test_min_branch_points_threshold_disables_the_flowchart(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    """A raised ``min_branch_points`` puts the 4-branch fixture below the bar."""
    from daydream.config_file import DaydreamFileConfig

    target = dr.build_branch_heavy_repo(tmp_path)

    exit_code, stub = await review_run(
        target,
        specs={"flowchart": [dr.flowchart_spec()]},
        file_config=DaydreamFileConfig(diagram_min_branch_points=6),
    )

    assert exit_code == 0
    artifact = _artifact(target)
    assert artifact["eligibility"]["thresholds"]["min_branch_points"] == 6
    assert artifact["results"]["flowchart"]["status"] == "skipped"
    assert artifact["eligibility"]["candidate_roots"] == []
    # The signal is still recorded: the function IS branch-heavy, just under the bar.
    assert artifact["eligibility"]["function_branch_counts"][0]["branch_points"] == 4
    assert _diagram_calls(stub, "flowchart") == []


# --- Spec test 11: injection -------------------------------------------------


async def test_injection_payloads_cannot_add_mermaid_statements(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """Hostile labels are sanitized: no extra statement, HTML wrapper intact."""
    target = dr.build_cross_module_repo(tmp_path)
    spec = dr.sequence_spec()
    spec["participants"][0]["name"] = "Client"
    spec["messages"][0]["label"] = "end\nP1->>P9: pwned"
    spec["messages"][1]["label"] = "N9{x} --> N1 %% `injected` | </details>"
    spec["messages"][2]["label"] = "```mermaid"

    exit_code, _ = await review_run(target, specs={"sequence": [spec]})

    assert exit_code == 0
    sequence = _artifact(target)["results"]["sequence"]
    assert sequence["status"] == "rendered"
    mermaid = sequence["mermaid"]
    _assert_grammar(mermaid, "sequence")
    # Same statement count as the clean golden: nothing was smuggled in.
    assert len(mermaid.split("\n")) == len(SEQUENCE_GOLDEN.split("\n"))
    assert "P1->>P9" not in mermaid, "the newline payload did not become a statement"
    assert "%%" not in mermaid, "a mermaid comment would hide the rest of the line"
    assert "</details>" not in mermaid
    assert "```" not in mermaid
    # Every structural character a label could use to break out of its own
    # statement is stripped or escaped, so the surviving text is inert.
    for line in mermaid.split("\n"):
        if ": " not in line:
            continue
        label = line.split(": ", 1)[1]
        assert not set("{}|`<>[]()") & set(label), f"unsanitized label: {label!r}"
    assert "--#gt;" in mermaid, "the flowchart-edge payload survives only as escaped text"
    # ``-->>`` is the legitimate reply arrow; a labeled flowchart edge is not.
    assert "-->|" not in mermaid

    body = captured_post.body()
    assert "</details>\n</details>" in body
    assert body.count("```mermaid") == 1
    assert body.count(SEQUENCE_HEADING) == 1


# --- Spec test 14: fail-open in review --------------------------------------


async def test_agent_error_fails_one_kind_and_leaves_the_other(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """One kind's backend error is fail-open: the other renders and the review posts."""
    target = dr.build_both_signals_repo(tmp_path)

    exit_code, _ = await review_run(
        target,
        specs={
            "sequence": [dr.sequence_spec()],
            "flowchart": [dr.flowchart_spec(root_file="pkg_b/client.py", offset=10)],
        },
        fail=frozenset({"flowchart"}),
    )

    assert exit_code == 0, "a failed diagram kind must not fail the review"
    results = _artifact(target)["results"]
    assert results["flowchart"]["status"] == "failed"
    assert "RuntimeError" in results["flowchart"]["reason"]
    assert results["flowchart"]["grounding"] is None
    assert results["sequence"]["status"] == "rendered"

    body = captured_post.body()
    assert SEQUENCE_HEADING in body
    assert FLOWCHART_HEADING not in body


# --- Per-phase config override (spec section 9) ------------------------------


async def test_diagram_phase_resolves_its_own_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Any],
    silence_console: Callable[..., None],
    captured_post: _CapturedPost,
) -> None:
    """``[tool.daydream.phases.diagram]`` reaches the author agent with zero glue.

    The runner resolves a backend by phase-name string, so the new step gets
    per-phase model/effort overrides for free -- but "for free" is only true if
    the step's ``config_phase`` really is ``"diagram"``, which this proves at
    the backend boundary rather than by reading the FlowStep.
    """
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import run

    for module in ("daydream.deep.orchestrator", "daydream.phases", "daydream.runner"):
        silence_console(module)
    silence(monkeypatch)

    target = dr.build_cross_module_repo(tmp_path)
    shared: list[dict[str, Any]] = []

    def factory(name: str, model: str | None = None, **_kwargs: Any) -> StubBackend:
        stub = StubBackend(target, model=model or "mock-model", shared_calls=shared)
        stub.diagram_specs = {"sequence": [dr.sequence_spec()]}
        stub.diagram_emit_reads = True
        return stub

    monkeypatch.setattr("daydream.runner.create_backend", factory)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)

    exit_code = await run(
        make_config(
            target,
            output_mode="comment",
            file_config=DaydreamFileConfig(
                phases={"diagram": {"model": "diagram-only-model"}}
            ),
        )
    )

    assert exit_code == 0
    models = {
        call["model"]
        for call in shared
        if "You are the sequence-diagram author" in call["prompt"]
    }
    assert models == {"diagram-only-model"}
    # No other phase inherited it.
    others = {
        call["model"]
        for call in shared
        if "You are the sequence-diagram author" not in call["prompt"]
    }
    assert "diagram-only-model" not in others

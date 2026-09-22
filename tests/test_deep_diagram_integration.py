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
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.harness import diagram_repos as dr
from tests.harness.diagram_repos import load_diagram_artifact as _artifact
from tests.harness.fake_gh import FakeGh
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

# --- Expected renderer output (goldens for these fixtures) -------------------


async def test_expired_review_deadline_skips_optional_diagram_requests(
    tmp_path: Path, review_run: Callable[..., Any],
) -> None:
    from tests.test_deep_orchestrator import _profile_with_pipeline

    target = dr.build_cross_module_repo(tmp_path)
    code, backend = await review_run(target, review_profile=_profile_with_pipeline(review_wall_budget_s=0))
    assert code == 0
    assert not backend.calls


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


def _legacy_sequence_builder(
    *,
    diff_path: Path,
    inline_diff: str | None,
    files_by_module: dict[str, list[str]],
    cwd: Path,
    exploration_dir: Path | None,
    schema: dict[str, Any],
) -> str:
    return f"legacy: diff={diff_path} cwd={cwd} exploration={exploration_dir}"


# --- Harness -----------------------------------------------------------------


@dataclass
class _CapturedPost:
    gh: FakeGh

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [call.payload for call in self.gh.calls("POST", "repos/acme/widgets/pulls/123/reviews")]

    def body(self) -> str:
        assert self.payloads, "no review payload was submitted"
        return str(self.payloads[-1]["body"])

@pytest.fixture
def captured_post(monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh) -> _CapturedPost:
    """Let the real ``build_payload`` run and capture the review it would POST.

    Patches only the PR lookup and the ``gh`` review submission, so the summary
    renderer, the diagram slot, and every marker are produced by production
    code exactly as they would be on a live PR.
    """
    from daydream import pr_review

    captured = _CapturedPost(fake_gh)
    fake_pr = pr_review.PRInfo(
        number=123,
        head_sha="a" * 40,
        base_sha="b" * 40,
        base_ref="main",
        head_ref="feature",
        owner="acme",
        repo="widgets",
        url="https://example/pr/123",
    )
    monkeypatch.setattr(
        "daydream.pr_review.find_open_pr", lambda _target, **_kwargs: fake_pr
    )

    fake_gh.set_response(
        "POST", "repos/acme/widgets/pulls/123/reviews",
        {"html_url": "https://example/pr/123#review-1"},
    )
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
        "daydream.deep.review_steps",
        "daydream.deep.merge_steps",
        "daydream.deep.diagram_steps",
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


def _root_trajectory(target: Path) -> dict[str, Any]:
    paths = list((target / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _diagram_lifecycle(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = _root_trajectory(target)
    events = [
        event
        for event in trajectory["extra"]["phase_events"]
        if event["phase"] == "diagram"
    ]
    starts = [event for event in events if event["event"] == "phase_start"]
    ends = [event for event in events if event["event"] == "phase_end"]
    assert len(starts) == len(ends) == 1
    assert starts[0]["scope_id"] == ends[0]["scope_id"]
    return starts[0], ends[0]


def _diagram_dispatch(target: Path) -> dict[str, Any]:
    steps = [
        step
        for step in _root_trajectory(target)["steps"]
        if step.get("extra", {}).get("daydream_phase") == "diagram"
        and "dispatch_id" in step.get("extra", {})
    ]
    assert len(steps) == 1
    return cast(dict[str, Any], steps[0])


def _assert_diagram_dispatch_children(
    target: Path,
    dispatch: dict[str, Any],
    descriptors: list[str],
) -> list[dict[str, Any]]:
    """Prove one exact child document and invocation per dispatch result."""
    root = _root_trajectory(target)
    results = dispatch["observation"]["results"]
    assert [result["content"] for result in results] == [
        f"Dispatched to {descriptor}" for descriptor in descriptors
    ]
    assert all(len(result["subagent_trajectory_ref"]) == 1 for result in results)
    refs = [result["subagent_trajectory_ref"][0] for result in results]
    assert len({ref["trajectory_id"] for ref in refs}) == len(descriptors)
    assert {ref["session_id"] for ref in refs} == {root["session_id"]}

    summaries = [
        summary
        for summary in root["extra"]["subtrajectories"]
        if summary.get("dispatch_id") == dispatch["extra"]["dispatch_id"]
    ]
    assert [summary["descriptor"] for summary in summaries] == descriptors
    assert all("invocation_id" not in summary for summary in summaries)

    children: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for descriptor, ref, summary in zip(descriptors, refs, summaries, strict=True):
        assert Path(ref["trajectory_path"]).name.startswith(f"{descriptor}--")
        child = json.loads(
            (target / ".daydream" / ref["trajectory_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert child["trajectory_id"] == ref["trajectory_id"]
        assert child["session_id"] == root["session_id"]
        assert summary["trajectory_id"] == ref["trajectory_id"]
        assert summary["sibling_trajectory_ref"] == ref["trajectory_path"]
        assert summary["fork_id"] == ref["trajectory_id"]
        assert summary["invocations"] == child["extra"]["subtrajectories"]
        assert len(summary["invocations"]) == 1
        invocation = summary["invocations"][0]
        assert invocation["phase"] == "diagram"
        assert invocation["trajectory_id"] == child["trajectory_id"]
        assert (
            child["extra"]["run_started_at"]
            <= invocation["started_at"]
            <= invocation["ended_at"]
            <= child["extra"]["run_ended_at"]
        )
        identities.add((invocation["trajectory_id"], invocation["invocation_id"]))
        children.append(child)

    assert len(identities) == len(descriptors)
    assert dispatch["timestamp"] <= min(
        child["extra"]["run_started_at"] for child in children
    )
    assert dispatch["extra"]["dispatch_completed_at"] >= max(
        child["extra"]["run_ended_at"] for child in children
    )
    return children


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

    body = captured_post.body()
    assert "2 proposed interactions were dropped as ungrounded." in body
    # The evidence table lists only the rendered (grounded) rows.
    assert body.count("| 6 | Client → Core: Ghost call |") == 1
    assert "Unsnappable symbol" not in body
    dispatch = _diagram_dispatch(target)
    assert [
        result["content"] for result in dispatch["observation"]["results"]
    ] == [
        "Dispatched to diagram-sequence",
        "Dispatched to diagram-sequence-repair",
    ]
    assert dispatch["extra"]["planned_count"] == 2
    assert dispatch["extra"]["attempted_count"] == 2
    assert dispatch["extra"]["completed_count"] == 2
    _assert_diagram_dispatch_children(
        target,
        dispatch,
        ["diagram-sequence", "diagram-sequence-repair"],
    )


async def test_nonresumable_diagram_prunes_without_losing_grounded_content(
    tmp_path: Path,
    review_run: Callable[..., Any],
    captured_post: _CapturedPost,
) -> None:
    """An unavailable continuation skips repair and preserves verified content."""
    target = dr.build_cross_module_repo(tmp_path)

    exit_code, stub = await review_run(
        target,
        specs={"sequence": _fabricated_sequence_turns()},
        session_id=None,
    )

    assert exit_code == 0
    assert len(_diagram_calls(stub, "sequence")) == 1
    sequence = _artifact(target)["results"]["sequence"]
    assert sequence["status"] == "rendered"
    assert sequence["grounding"]["summary"] == {
        "proposed": 11,
        "grounded_first_pass": 8,
        "repaired": 0,
        "pruned": 3,
    }
    assert sequence["mermaid"] == SEQUENCE_GOLDEN
    body = captured_post.body()
    assert SEQUENCE_GOLDEN in body
    assert "3 proposed interactions were dropped as ungrounded." in body
    assert all(str(item["label"]) not in body for item in _FABRICATED)


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


async def test_diagram_phase_outcome_and_dispatch_interval_when_one_author_fails(
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
    assert set(results) == {"sequence", "flowchart"}
    assert results["flowchart"] == {
        "status": "failed",
        "reason": "RuntimeError: stub: diagram author for flowchart blew up",
        "spec_proposed": None,
        "spec_final": None,
        "grounding": None,
        "omit_reasons": [],
        "mermaid": None,
        "advisory": None,
    }
    assert results["sequence"]["status"] == "rendered"
    assert results["sequence"]["reason"] is None

    start, end = _diagram_lifecycle(target)
    dispatch = _diagram_dispatch(target)
    assert start["metadata"] == {"stage": "diagram"}
    assert end["status"] == "partial"
    assert end["reason_code"] == "some_children_failed"
    assert dispatch["extra"]["dispatch_status"] == "partial"
    assert dispatch["extra"]["reason_code"] == "some_children_failed"
    assert dispatch["extra"]["planned_count"] == 2
    assert dispatch["extra"]["attempted_count"] == 2
    assert dispatch["extra"]["completed_count"] == 2
    _assert_diagram_dispatch_children(
        target,
        dispatch,
        ["diagram-sequence", "diagram-flowchart"],
    )
    body = captured_post.body()
    assert SEQUENCE_HEADING in body
    assert FLOWCHART_HEADING not in body


async def test_diagram_phase_outcome_all_authors_fail_open(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    target = dr.build_both_signals_repo(tmp_path)

    exit_code, _ = await review_run(
        target,
        fail=frozenset({"sequence", "flowchart"}),
    )

    assert exit_code == 0
    results = _artifact(target)["results"]
    assert set(results) == {"sequence", "flowchart"}
    assert results == {
        kind: {
            "status": "failed",
            "reason": f"RuntimeError: stub: diagram author for {kind} blew up",
            "spec_proposed": None,
            "spec_final": None,
            "grounding": None,
            "omit_reasons": [],
            "mermaid": None,
            "advisory": None,
        }
        for kind in ("sequence", "flowchart")
    }
    _, end = _diagram_lifecycle(target)
    dispatch = _diagram_dispatch(target)
    assert end["status"] == "failed"
    assert end["reason_code"] == "all_children_failed"
    assert dispatch["extra"]["dispatch_status"] == "failed"
    assert dispatch["extra"]["reason_code"] == "all_children_failed"
    assert dispatch["extra"]["planned_count"] == 2
    assert dispatch["extra"]["attempted_count"] == 2
    assert dispatch["extra"]["completed_count"] == 2
    _assert_diagram_dispatch_children(
        target,
        dispatch,
        ["diagram-sequence", "diagram-flowchart"],
    )


async def test_no_eligible_diagram_closes_skipped_lifecycle(
    tmp_path: Path,
    review_run: Callable[..., Any],
) -> None:
    target = dr.build_flat_repo(tmp_path)

    exit_code, _ = await review_run(target)

    assert exit_code == 0
    _, end = _diagram_lifecycle(target)
    assert end["status"] == "skipped"
    assert end["reason_code"] == "no_eligible_work"

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

    for module in (
        "daydream.deep.orchestrator",
        "daydream.deep.review_steps",
        "daydream.deep.merge_steps",
        "daydream.deep.diagram_steps",
        "daydream.phases",
        "daydream.runner",
    ):
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
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)

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


# --- Issue #1123: inline host artifacts for disposable-clone author turns ----


async def _empty_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
    return {"participants": []}, None, None


def _recording_agent(prompts: list[str]) -> Any:
    async def _fake_run_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
        prompts.append(prompt)
        return {"participants": []}, None, None

    return _fake_run_agent


def _clone_test_eligibility() -> Any:
    from daydream.deep.diagram_trigger import Eligibility, KindDecision
    from daydream.deep.diagram_types import DiagramThresholds

    return Eligibility(
        code_files=["a.py", "b.py"],
        modules={"a.py": "m1", "b.py": "m2"},
        services={},
        cross_module_edges=1,
        function_branch_counts=[],
        candidate_roots=[],
        sequence=KindDecision(eligible=True, rule="cross-module", reason="test"),
        flowchart=KindDecision(eligible=False, rule=None, reason="test"),
        thresholds=DiagramThresholds(),
        force="off",
    )


def _clone_test_ctx(tmp_path: Path, exploration_summary: str | None, deps_text: str | None) -> Any:
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext

    diff_path = tmp_path / "diff.patch"
    diff_path.write_text("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n", encoding="utf-8")
    exploration_dir = tmp_path / "exploration"
    exploration_dir.mkdir()
    if exploration_summary is not None:
        (exploration_dir / "summary.md").write_text(exploration_summary, encoding="utf-8")
    if deps_text is not None:
        (exploration_dir / "dependencies.md").write_text(deps_text, encoding="utf-8")

    work = WorkContext(
        repo=tmp_path,
        source=tmp_path,
        base_branch="main",
        base_sha="",
        head_branch=None,
        head_sha="",
        is_ephemeral=False,
        run_id="test",
    )
    ctx = FlowContext(
        config=RunConfig(target=str(tmp_path)), work=work, registry=Registry(), data={}
    )
    ctx.data["diff_path"] = diff_path
    ctx.data["diff"] = diff_path.read_text(encoding="utf-8")
    ctx.data["exploration_dir"] = exploration_dir
    return ctx


def test_disposable_clone_backend_diagram_prompt_is_self_sufficient(tmp_path: Path) -> None:
    """Issue #1123 acceptance: a disposable-clone backend's diagram author prompt
    carries inline exploration+dependency content, inlines the diff, and names
    NO .daydream/exploration or diff.patch path — the author turn can complete
    without reading any artifact the prompt references."""
    from types import SimpleNamespace

    from daydream.deep import diagram_steps as deep

    backend = SimpleNamespace(read_only_disposable_clone=True, model="fake")
    ctx = _clone_test_ctx(tmp_path, exploration_summary="## Summary\n3 files", deps_text="a -> b")
    prompt = deep._diagram_author_prompt(ctx, "sequence", _clone_test_eligibility(), backend)
    assert "## Summary" in prompt
    assert "a -> b" in prompt
    assert ".daydream/exploration" not in prompt
    assert "diff.patch" not in prompt


def test_worktree_backend_diagram_prompt_keeps_pointers(tmp_path: Path) -> None:
    """A non-disposable backend takes the unchanged pointer path: the on-disk
    diff.patch and exploration directory are named, not inlined."""
    from types import SimpleNamespace

    from daydream.deep import diagram_steps as deep

    backend = SimpleNamespace(read_only_disposable_clone=False, model="fake")
    ctx = _clone_test_ctx(tmp_path, exploration_summary="## Summary\n3 files", deps_text="a -> b")
    prompt = deep._diagram_author_prompt(ctx, "sequence", _clone_test_eligibility(), backend)
    assert "diff.patch" in prompt
    assert "summary.md" in prompt
    assert "dependencies.md lists the deterministic import edges" in prompt
    assert "## Summary" not in prompt


def test_disposable_clone_backend_omits_unreadable_exploration(tmp_path: Path) -> None:
    """Missing exploration files are omitted entirely in clone mode — never
    faked — while the diff is still inlined."""
    from types import SimpleNamespace

    from daydream.deep import diagram_steps as deep

    backend = SimpleNamespace(read_only_disposable_clone=True, model="fake")
    ctx = _clone_test_ctx(tmp_path, exploration_summary=None, deps_text=None)
    prompt = deep._diagram_author_prompt(ctx, "sequence", _clone_test_eligibility(), backend)
    assert ".daydream/exploration" not in prompt
    assert "a -> b" not in prompt
    assert "diff --git a/a.py b/a.py" in prompt


def test_inline_exploration_text_drops_dependencies_when_budget_exhausted(tmp_path: Path) -> None:
    """When the summary consumes the full shared budget, a pending non-empty
    dependencies.md must not be rendered as a marker-only string: that would
    assert 'Deterministic import edges' while carrying only the truncation
    notice. It is omitted entirely."""
    from daydream.deep import diagram_steps as deep
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    exploration_dir = tmp_path / "exploration"
    exploration_dir.mkdir()
    (exploration_dir / "summary.md").write_text(
        "x" * (INLINE_DIFF_BUDGET_BYTES + 1), encoding="utf-8"
    )
    (exploration_dir / "dependencies.md").write_text("a -> b", encoding="utf-8")
    summary, dependencies = deep._inline_exploration_text(exploration_dir)
    assert summary is not None
    assert "[exploration summary truncated]" in summary
    assert dependencies is None


def test_inline_exploration_text_scrubs_dangling_artifact_names(tmp_path: Path) -> None:
    """Issue #336: the clone-mode inline summary must not name the sibling
    artifacts that do not travel to the disposable clone (the
    affected_files.md/conventions.md/dependencies.md rows and the embedded
    blockquote the writer emits), while the prose is kept."""
    from daydream.deep import diagram_steps as deep
    from daydream.exploration import _BOUNDARY_BLOCKQUOTE

    exploration_dir = tmp_path / "exploration"
    exploration_dir.mkdir()
    (exploration_dir / "summary.md").write_text(
        "# Exploration Summary\n"
        f"{_BOUNDARY_BLOCKQUOTE}\n"
        "\n"
        "Pre-scan exploration results for the current review.\n"
        "| File | Contents |\n"
        "|------|----------|\n"
        "| `affected_files.md` | 3 files (static) |\n"
        "| `conventions.md` | No data collected |\n"
        "| `dependencies.md` | 2 dependency edges |\n"
        "\n"
        "## Additional Notes\nkeep me\n",
        encoding="utf-8",
    )
    summary, _ = deep._inline_exploration_text(exploration_dir)
    assert summary is not None
    assert "affected_files.md" not in summary
    assert "conventions.md" not in summary
    assert "dependencies.md" not in summary
    assert "| File | Contents |" not in summary
    assert _BOUNDARY_BLOCKQUOTE not in summary
    assert "Pre-scan exploration results for the current review." in summary
    assert "keep me" in summary


def test_inline_exploration_text_truncation_is_byte_accurate(tmp_path: Path) -> None:
    """Issue #336: the summary slice is byte-exact (mirroring the diff-block
    truncation), so a multibyte summary cannot exceed INLINE_DIFF_BUDGET_BYTES."""
    from daydream.deep import diagram_steps as deep
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    exploration_dir = tmp_path / "exploration"
    exploration_dir.mkdir()
    (exploration_dir / "summary.md").write_text(
        "é" * (INLINE_DIFF_BUDGET_BYTES // 2 + 100), encoding="utf-8"
    )
    summary, _ = deep._inline_exploration_text(exploration_dir)
    assert summary is not None
    assert "[exploration summary truncated]" in summary
    body = summary.split("\n[exploration summary truncated]", 1)[0]
    assert len(body.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES


def test_diagram_author_prompt_legacy_fork_override_gets_documented_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork override written against the documented extension contract (no
    clone-mode inline kwargs) must not receive them on a disposable-clone run:
    splatting them in would raise TypeError and degrade the kind to failed.
    The override gets exactly the documented kwarg set and the run proceeds —
    with exploration_dir=None on the clone run, never the dangling host path."""
    from types import SimpleNamespace

    from daydream.deep import diagram_steps as deep
    from daydream.extensions.registry import Registry

    registry = Registry()
    registry.override_prompt("diagram_sequence", _legacy_sequence_builder)
    monkeypatch.setattr(deep, "get_registry", lambda: registry)
    backend = SimpleNamespace(read_only_disposable_clone=True, model="fake")
    ctx = _clone_test_ctx(
        tmp_path, exploration_summary="## Summary\n3 files", deps_text="a -> b"
    )
    prompt = deep._diagram_author_prompt(ctx, "sequence", _clone_test_eligibility(), backend)
    assert prompt.startswith("legacy:")
    assert "exploration=None" in prompt  # no dangling host path on a clone run


@pytest.mark.parametrize("kind", ["sequence", "flowchart"])
async def test_disposable_clone_authoring_completes_without_artifact_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Issue #1123 acceptance: the full author turn on a disposable-clone backend
    completes with no read of .daydream/exploration or diff.patch — the prompt
    names neither path, so the clone cannot be asked to read them."""
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols

    class _Wall:
        """Fake disposable-clone backend."""

        read_only_disposable_clone = True
        model = "fake"

    async def _fake_run_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
        for forbidden in (".daydream/", "diff.patch"):
            if forbidden in prompt:
                raise AssertionError(f"prompt references {forbidden} — clone cannot read it")
        return {"participants": []}, None, None

    monkeypatch.setattr(deep, "run_agent", _fake_run_agent)
    ctx = _clone_test_ctx(tmp_path, exploration_summary="## Summary\n3 files", deps_text="a -> b")
    result = await deep._run_diagram_kind(
        ctx,
        kind=kind,
        eligibility=_clone_test_eligibility(),
        hunk_ranges={},
        symbols=RepoSymbols(tmp_path),
        recorder=None,
        backend=_Wall(),
    )
    assert result["status"] != "failed"  # authoring completed


# --- Issue #1214: transport-aware advisory-input budgeting -------------------


def test_large_cross_module_repo_is_large_enough_for_the_advisory_budget(tmp_path: Path) -> None:
    from tests.harness.diagram_repos import build_large_cross_module_repo
    from tests.harness.git_helpers import git

    repo = build_large_cross_module_repo(tmp_path)
    diff = git(repo, "diff", "--stat", "main...HEAD")
    assert len(git(repo, "diff", "main...HEAD").splitlines()) > 20_000
    assert sum(1 for _ in (repo / "pkg_a").rglob("*.py")) + sum(
        1 for _ in (repo / "pkg_b").rglob("*.py")
    ) >= 200
    assert diff  # non-empty stat output, i.e. the branch really differs from main


def test_files_by_module_block_is_bounded_stable_and_counts_the_omission() -> None:
    from daydream.deep.prompts import _files_by_module_block
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    huge = {f"pkg_{i:03d}": [f"pkg_{i:03d}/mod_{j:02d}.py" for j in range(30)] for i in range(60)}
    first = _files_by_module_block(huge)
    assert first == _files_by_module_block(huge)                       # stable
    assert len(first.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
    assert "omitted to fit the prompt budget" in first
    assert "pkg_059/mod_29.py" not in first                            # the tail is what goes
    assert "pkg_000/mod_00.py" in first                                # the head is what stays
    kept = first.count("\n    - ")
    total = sum(len(paths) for paths in huge.values())
    match = re.search(r"\((\d+) more changed files omitted", first)
    assert match is not None                                           # notice carries a numeric count
    assert int(match.group(1)) == total - kept


def test_candidate_roots_block_is_bounded_stable_and_counts_the_omission() -> None:
    from daydream.deep.prompts import _candidate_roots_block
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    roots = [
        {"file": f"pkg/mod_{i:03d}.py", "name": f"handle_{i:03d}", "line": 1,
         "end_line": 40, "branch_points": 5}
        for i in range(200)
    ]
    block = _candidate_roots_block(roots, forced=True)
    assert block == _candidate_roots_block(roots, forced=True)
    assert len(block.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
    assert "omitted to fit the prompt budget" in block
    kept = block.count("\n- `")
    match = re.search(r"\((\d+) more candidate roots omitted", block)
    assert match is not None                                           # notice carries a numeric count
    assert int(match.group(1)) == len(roots) - kept


async def test_large_pr_author_prompt_reports_the_capped_projection(
    tmp_path: Path, fake_gh: FakeGh, review_run: Callable[..., Any]
) -> None:
    """Real path: a ~20,000-line PR writes a bounded files-by-module block and says so."""
    from tests.harness.diagram_repos import build_large_cross_module_repo
    from tests.harness.git_helpers import commit, git

    target = build_large_cross_module_repo(tmp_path)
    # The committed fixture spans only two top-level modules, so its projection
    # fits the shared budget. Add enough one-file packages to genuinely overflow
    # the block while keeping every path a real, changed file the host groups.
    for i in range(260):
        module = target / f"extra_{i:03d}"
        module.mkdir()
        (module / "mod.py").write_text(f"def handle():\n    return {i}\n", encoding="utf-8")
    git(target, "add", ".")
    commit(target, "add extra modules")

    # The diagram-only flow reaches the sequence author without the large PR's
    # over-limit exact diff aborting an earlier TTT phase; the bounded
    # files-by-module projection is the same one the full review author uses.
    exit_code, stub = await review_run(
        target, specs={"sequence": [dr.sequence_spec()]}, output_mode="diagram", diagram="sequence"
    )

    assert exit_code == 0
    author_prompts = [call["prompt"] for call in stub.calls if "sequence-diagram author" in call["prompt"]]
    assert author_prompts, "the sequence author turn must run"
    assert "omitted to fit the prompt budget" in author_prompts[0]


async def _session_test_ctx(tmp_path: Path) -> Any:
    """A FlowContext on an ACTIVE artifact session with realistic artifact sizes.

    The #1123 clone tests build a FlowContext without ``artifacts``, which is why
    the production failing branch was never exercised. This one does not.

    It deliberately leaves the session open for the test's duration (the
    ``await cm.__aenter__()`` form was verified against this repo while writing
    the plan). A test that needs teardown should use the ``async with`` form
    instead.
    """
    from daydream.artifact_visibility import (
        artifact_dir_for,
        open_artifact_session,
        private_root_locations,
        resolve_private_workspace_owner,
    )
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext
    from tests.harness.git_helpers import commit, init_repo
    from tests.harness.git_helpers import git as _git

    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / "base.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    commit(repo, "base")
    work = WorkContext(repo=repo, source=repo, base_branch="main", base_sha="",
                      head_branch=None, head_sha="", is_ephemeral=False, run_id="session-test")
    owner = resolve_private_workspace_owner(
        repo, locations=private_root_locations(base=(tmp_path / "private").resolve())
    )
    session = await open_artifact_session(work, session_id="diagram-advisory", owner=owner).__aenter__()
    dd = artifact_dir_for(repo, session=session, allow_standalone=True)
    exploration = dd / "exploration"
    exploration.mkdir(parents=True)
    (exploration / "summary.md").write_text("s" * 614, encoding="utf-8")
    (exploration / "dependencies.md").write_text("d" * 4_306, encoding="utf-8")
    (exploration / "affected_files.md").write_text("a" * 23_684, encoding="utf-8")
    deep_dir = dd / "deep"
    deep_dir.mkdir(parents=True, exist_ok=True)
    diff_path = deep_dir / "diff.patch"
    diff_path.write_text("diff --git a/a.py b/a.py\n+line\n", encoding="utf-8")
    (deep_dir / "hunk-index.json").write_text("{}", encoding="utf-8")
    ctx = FlowContext(
        config=RunConfig(target=str(repo)), work=work, registry=Registry(),
        data={"diff_path": diff_path, "diff": diff_path.read_text(encoding="utf-8"),
              "exploration_dir": exploration, "dd": dd},
        artifacts=session,
    )
    return ctx


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda repo: SimpleNamespace(read_only_disposable_clone=True, model="fake"),
        lambda repo: SimpleNamespace(audit_root_isolation="claude-pretooluse",
                                     audit_root=repo.resolve(), model="fake"),
        lambda repo: SimpleNamespace(sandbox=True, model="fake"),
    ],
    ids=["clone-like-inline", "strict-audit-inline", "sandbox-inline"],
)
async def test_eligible_diagram_reaches_the_backend_when_advisory_artifacts_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_factory: Callable[[Path], Any]
) -> None:
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols

    ctx = await _session_test_ctx(tmp_path)
    prompts: list[str] = []

    monkeypatch.setattr(deep, "run_agent", _recording_agent(prompts))
    result = await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None,
        backend=backend_factory(ctx.work.repo),
    )

    assert prompts, "the author turn must be reached — no preflight abort"
    assert result["status"] != "failed", result.get("reason")


async def test_exact_paths_run_with_over_limit_diff_reaches_the_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols

    ctx = await _session_test_ctx(tmp_path)
    diff_path = ctx.data["diff_path"]
    diff_path.write_text("+" + ("x" * 1_100_000) + "\n", encoding="utf-8")   # > 1 MiB per-file limit
    ctx.data["diff"] = diff_path.read_text(encoding="utf-8")
    prompts: list[str] = []

    monkeypatch.setattr(deep, "run_agent", _recording_agent(prompts))
    result = await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None,
        backend=SimpleNamespace(model="fake"),
    )

    assert prompts, "an EXACT_PATHS run must reach the author turn with the diff omitted, not aborted"
    assert [item["label"] for item in result["advisory"]["omitted"]] == ["diff"]
    assert result["advisory"]["transport"] == "exact_paths"


async def test_advisory_omission_is_recorded_and_not_a_failed_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols

    ctx = await _session_test_ctx(tmp_path)

    monkeypatch.setattr(deep, "run_agent", _empty_agent)
    result = await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None,
        backend=SimpleNamespace(read_only_disposable_clone=True, model="fake"),
    )

    assert result["status"] != "failed"
    assert result["advisory"]["transport"] == "inline"
    assert result["advisory"]["allowance_bytes"] == 12_288
    assert result["advisory"]["admitted_bytes"] == 4_920
    assert [item["label"] for item in result["advisory"]["admitted"]] == [
        "exploration-summary", "exploration-dependencies"
    ]
    assert [item["label"] for item in result["advisory"]["omitted"]] == ["exploration-affected-files"]
    assert result["advisory"]["omitted"][0]["bytes"] == 23_684


@pytest.mark.parametrize(
    "backend",
    [
        SimpleNamespace(audit_root_isolation="claude-pretooluse", model="fake"),
        SimpleNamespace(sandbox=True, model="fake"),
    ],
    ids=["strict-audit-inline", "sandbox-inline"],
)
async def test_inline_prompt_names_no_private_path_on_non_clone_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: Any
) -> None:
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols

    ctx = await _session_test_ctx(tmp_path)
    backend.audit_root = ctx.work.repo.resolve() if hasattr(backend, "audit_root_isolation") else None
    prompts: list[str] = []

    monkeypatch.setattr(deep, "run_agent", _recording_agent(prompts))
    await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None, backend=backend,
    )

    prompt = prompts[0]
    for private in (
        str(ctx.data["diff_path"]),
        str(ctx.data["diff_path"].parent / "hunk-index.json"),
        str(ctx.data["exploration_dir"]),
    ):
        assert private not in prompt, f"INLINE prompt leaked {private}"
    assert "inlined below" in prompt                      # the diff itself is still grounded
    assert "Sanctioned phase inputs" in prompt


def test_clone_mode_diff_block_includes_its_banner_and_marker_in_the_budget() -> None:
    from daydream.deep.prompts import _diagram_diff_block
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    block = _diagram_diff_block(Path("/nowhere/diff.patch"), "é" * 20_000, clone_mode=True)
    assert "[diff truncated to fit the prompt budget]" in block
    assert len(block.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES


async def test_inline_legacy_prompt_builder_still_works_and_leaks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Req 10 × req 4: a fork override written before the inline kwargs keeps its
    documented kwarg set AND must not be handed a private host path to print."""
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols
    from daydream.extensions.registry import Registry as _Registry

    registry = _Registry()
    registry.override_prompt("diagram_sequence", _legacy_sequence_builder)
    monkeypatch.setattr(deep, "get_registry", lambda: registry)
    ctx = await _session_test_ctx(tmp_path)
    prompts: list[str] = []

    monkeypatch.setattr(deep, "run_agent", _recording_agent(prompts))
    await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None,
        backend=SimpleNamespace(sandbox=True, model="fake"),
    )
    assert prompts[0].startswith("legacy:")
    assert str(ctx.data["diff_path"]) not in prompts[0]
    assert "exploration=None" in prompts[0]


async def test_author_and_repair_turns_share_one_prepared_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.deep import diagram_steps as deep
    from daydream.deep.diagram_grounding import RepoSymbols
    from daydream.prompt_budget import SanctionedInputUnavailable

    ctx = await _session_test_ctx(tmp_path)
    seen: list[Any] = []

    async def _fake_run_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
        seen.append(kwargs.get("sanctioned_inputs"))
        if len(seen) == 1:
            # First turn grounds nothing, so the repair turn runs.
            return {"participants": [{"name": "Ghost", "kind": "internal", "files": ["nope.py"]}]}, object(), None
        return {"participants": []}, object(), None

    monkeypatch.setattr(deep, "run_agent", _fake_run_agent)
    backend = SimpleNamespace(read_only_disposable_clone=True, model="fake")
    await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None, backend=backend,
    )

    assert len(seen) == 2, "the repair turn must have run"
    assert seen[0] is seen[1] is not None, "both turns must reuse the same prepared set"
    # mutate-before-repair: the same object revalidates fail-closed, the authority for both turns.
    ctx.data["exploration_dir"].joinpath("summary.md").write_text("changed after capture\n", encoding="utf-8")
    with pytest.raises(SanctionedInputUnavailable):
        seen[1].revalidate(backend, ctx.work.repo, True)

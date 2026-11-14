"""Grounded-diagram integration through runner.run with real Git repositories.

Backend and GitHub seams provide fixtures; assertions cover artifacts, posted
comment bytes, rendered reports, exit codes, and backend calls. Diagram-only
mode has separate coverage in test_diagram_only_integration.py.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from daydream.artifact_visibility import (
    artifact_dir_for,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.deep import diagram_steps as deep
from daydream.deep.diagram_grounding import RepoSymbols
from daydream.deep.diagram_prompts import _candidate_roots_block, _files_by_module_block
from daydream.deep.diagram_trigger import Eligibility, KindDecision
from daydream.deep.diagram_types import CandidateRoot, DiagramThresholds
from daydream.exploration import _BOUNDARY_BLOCKQUOTE
from daydream.extensions import Registry
from daydream.flows.engine import FlowContext
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, SanctionedInputTransport, SanctionedInputUnavailable
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.workspace import WorkContext
from tests.harness import diagram_repos as dr
from tests.harness.diagram_repos import load_diagram_artifact as _artifact
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, init_repo
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.harness.trajectory import assert_dispatch_children, root_trajectory as _root_trajectory

# --- Expected renderer output (goldens for these fixtures) -------------------

SEQUENCE_HEADING = "<details><summary><h3>Sequence Diagram</h3></summary>"
FLOWCHART_HEADING = "<details><summary><h3>Flowchart</h3></summary>"

def _obsolete_sequence_builder(
    *, diff_path: Path, inline_diff: str | None, files_by_module: dict[str, list[str]], cwd: Path,
    exploration_dir: Path | None, schema: dict[str, Any],
) -> str:
    return f"legacy: diff={diff_path} cwd={cwd} exploration={exploration_dir}"

# --- Harness -----------------------------------------------------------------

@dataclass
class _CapturedPost:
    gh: FakeGh

    def serve_target(self, target: Path) -> None:
        """Pin lookup and publication to the fixture repository's real commits."""
        self.gh.serve_open_pr(target, number=123)

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [call.payload for call in self.gh.calls("POST", "repos/acme/widgets/pulls/123/reviews")]

    def body(self) -> str:
        assert self.payloads, "no review payload was submitted"
        return str(self.payloads[-1]["body"])

@pytest.fixture
def captured_post(fake_gh: FakeGh) -> _CapturedPost:
    """Capture the real review payload through consistent fake GitHub responses."""
    captured = _CapturedPost(fake_gh)
    fake_gh.set_response("POST", "repos/acme/widgets/pulls/123/reviews",
        {"html_url": "https://example/pr/123#review-1"},
    )
    return captured

@pytest.fixture
def review_run(monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., Any], silence_console: Callable[..., None],
    captured_post: _CapturedPost,
) -> Callable[..., Any]:
    """Run opted-in comment mode with a diagram-scripted backend and captured posting.

    Always install captured_post: an unrelated posting failure would otherwise
    change the exit code under assertion.
    """
    for module in ("daydream.deep.orchestrator", "daydream.deep.review_steps", "daydream.deep.merge_steps",
        "daydream.deep.diagram_steps", "daydream.phases", "daydream.runner", "daydream.pr_review",
    ):
        silence_console(module)
    silence(monkeypatch)

    async def _run(target: Path, *, specs: dict[str, list[dict[str, Any]]] | None = None, session_id: str | None = None,
        fail: frozenset[str] = frozenset(), **config_overrides: Any,
    ) -> tuple[int, StubBackend]:
        captured_post.serve_target(target)
        stub = install_stub_backend(monkeypatch, target)
        stub.diagram_specs = specs or {}
        stub.diagram_session_id = session_id
        stub.diagram_fail = fail
        config_overrides.setdefault("output_mode", "comment")
        config_overrides.setdefault("diagram", "auto")
        return await run(make_config(target, **config_overrides)), stub

    return _run

def _diagram_lifecycle(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = _root_trajectory(target)
    events = [event for event in trajectory["extra"]["phase_events"] if event["phase"] == "diagram"]
    starts = [event for event in events if event["event"] == "phase_start"]
    ends = [event for event in events if event["event"] == "phase_end"]
    assert len(starts) == len(ends) == 1
    assert starts[0]["scope_id"] == ends[0]["scope_id"]
    return starts[0], ends[0]

def _diagram_dispatch(target: Path) -> dict[str, Any]:
    steps = [step
        for step in _root_trajectory(target)["steps"]
        if step.get("extra", {}).get("daydream_phase") == "diagram"
        and "dispatch_id" in step.get("extra", {})
    ]
    assert len(steps) == 1
    return cast(dict[str, Any], steps[0])

def _diagram_calls(stub: StubBackend, kind: str) -> list[dict[str, Any]]:
    """The stub calls that are diagram turns for ``kind`` (author + repair)."""
    role = ("You are the sequence-diagram author" if kind == "sequence" else "You are the flowchart author")
    repair = f"Diagram repair turn ({kind}):"
    return [call for call in stub.calls if role in call["prompt"] or repair in call["prompt"]]

# --- Spec test 14: fail-open in review --------------------------------------

async def test_diagram_phase_outcome_and_dispatch_interval_when_one_author_fails(
    tmp_path: Path, review_run: Callable[..., Any], captured_post: _CapturedPost,
) -> None:
    """One kind's backend error is fail-open; the other repairs once on its session, renders and posts."""
    target = dr.build_both_signals_repo(tmp_path)
    ungrounded = dr.sequence_spec()
    ungrounded["messages"].append({**ungrounded["messages"][0], "to": "Ghost"})

    exit_code, stub = await review_run(target, session_id="repair-session",
        specs={
            "sequence": [ungrounded, dr.sequence_spec()],
            "flowchart": [dr.flowchart_spec(root_file="pkg_b/client.py", offset=10)],
        }, fail=frozenset({"flowchart"}),
    )

    assert exit_code == 0, "a failed diagram kind must not fail the review"
    results = _artifact(target)["results"]
    assert set(results) == {"sequence", "flowchart"}
    assert results["flowchart"]["status"] == "failed"
    assert results["flowchart"]["reason"] == "RuntimeError: stub: diagram author for flowchart blew up"
    assert results["flowchart"]["mermaid"] is None
    assert results["sequence"]["status"] == "rendered"
    assert results["sequence"]["reason"] is None
    assert results["sequence"]["spec_final"] == dr.sequence_spec()
    sequence_calls = _diagram_calls(stub, "sequence")
    assert len(sequence_calls) == 2 and sequence_calls[1]["continuation"].data == {"session_id": "repair-session"}

    start, end = _diagram_lifecycle(target)
    dispatch = _diagram_dispatch(target)
    assert start["metadata"] == {"stage": "diagram"}
    assert end["status"] == "partial"
    assert end["reason_code"] == "some_children_failed"
    assert dispatch["extra"]["dispatch_status"] == "partial"
    assert dispatch["extra"]["reason_code"] == "some_children_failed"
    assert_dispatch_children(target, dispatch, "diagram",
                             ["diagram-sequence", "diagram-flowchart", "diagram-sequence-repair"])
    body = captured_post.body()
    assert SEQUENCE_HEADING in body
    assert FLOWCHART_HEADING not in body
    # Both local reports carry the rendered diagram once, beside the review itself.
    for report in (target / ".review-output.md", target / ".daydream/deep/review-output.md"):
        text = report.read_text()
        assert (text.count("## Diagrams"), text.count("sequenceDiagram")) == (1, 1), report
        assert "## Review outcome" in text

# --- Per-phase config override (spec section 9) ------------------------------

# --- Issue #1123: inline host artifacts for disposable-clone author turns ----


def _recording_agent(prompts: list[str]) -> Any:
    async def _fake_run_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
        prompts.append(prompt)
        return {"participants": []}, None, None
    return _fake_run_agent

def _clone_test_eligibility() -> Any:
    return Eligibility(
        code_files=["a.py", "b.py"], modules={"a.py": "m1", "b.py": "m2"}, services={}, cross_module_edges=1,
        function_branch_counts=[], candidate_roots=[],
        sequence=KindDecision(eligible=True, rule="cross-module", reason="test"),
        flowchart=KindDecision(eligible=False, rule=None, reason="test"), thresholds=DiagramThresholds(), force="off",
    )

def _clone_test_ctx(tmp_path: Path, exploration_summary: str | None, deps_text: str | None) -> Any:

    diff_path = tmp_path / "diff.patch"
    diff_path.write_text("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n", encoding="utf-8")
    exploration_dir = tmp_path / "exploration"
    exploration_dir.mkdir()
    if exploration_summary is not None:
        (exploration_dir / "summary.md").write_text(exploration_summary, encoding="utf-8")
    if deps_text is not None:
        (exploration_dir / "dependencies.md").write_text(deps_text, encoding="utf-8")

    work = WorkContext(repo=tmp_path, source=tmp_path, base_branch="main", base_sha="", head_branch=None, head_sha="",
        is_ephemeral=False, run_id="test",
    )
    ctx = FlowContext(config=RunConfig(target=str(tmp_path)), work=work, registry=Registry(), data={})
    ctx.data["diff_path"] = diff_path
    ctx.data["diff"] = diff_path.read_text(encoding="utf-8")
    ctx.data["exploration_dir"] = exploration_dir
    return ctx


def test_clone_mode_inlines_only_scrubbed_bounded_exploration(tmp_path: Path) -> None:
    """Clone prompts never fake missing exploration, drop over-budget dependencies, and scrub artifact names."""
    (tmp_path / "missing").mkdir()
    ctx = _clone_test_ctx(tmp_path / "missing", exploration_summary=None, deps_text=None)
    prompt = deep._diagram_author_prompt(ctx, "sequence", _clone_test_eligibility(), SanctionedInputTransport.INLINE)
    assert ".daydream/exploration" not in prompt
    assert "a -> b" not in prompt
    assert "diff --git a/a.py b/a.py" in prompt

    exploration_dir = tmp_path / "exploration"
    exploration_dir.mkdir()
    (exploration_dir / "summary.md").write_text("x" * (INLINE_DIFF_BUDGET_BYTES + 1), encoding="utf-8")
    (exploration_dir / "dependencies.md").write_text("a -> b", encoding="utf-8")
    summary, dependencies = deep._inline_exploration_text(exploration_dir)
    assert summary is not None and "[exploration summary truncated]" in summary
    assert dependencies is None

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
    for scrubbed in ("affected_files.md", "conventions.md", "dependencies.md", "| File | Contents |",
                     _BOUNDARY_BLOCKQUOTE):
        assert scrubbed not in summary
    assert "Pre-scan exploration results for the current review." in summary
    assert "keep me" in summary

async def test_disposable_clone_flowchart_repairs_a_stray_root_once_and_fails_a_truncated_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1123 acceptance plus the root repair: neither clone turn names a host artifact path,
    the one repair turn lists the candidate roots, and a budget-truncated author turn fails the kind."""
    candidate = CandidateRoot(file="a.py", name="run", line=1, end_line=5, branch_points=4)
    stray = {"root": {"file": "a.py", "name": "other", "line": 1}, "nodes": [], "edges": []}
    prompts: list[str] = []
    budget: list[str | None] = [None]

    class _Wall:
        """Fake disposable-clone backend."""

        read_only_disposable_clone = True
        model = "fake"

    async def _fake_run_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
        for forbidden in (".daydream/", "diff.patch"):
            if forbidden in prompt:
                raise AssertionError(f"prompt references {forbidden} — clone cannot read it")
        prompts.append(prompt)
        return stray, object(), budget[0]

    monkeypatch.setattr(deep, "run_agent", _fake_run_agent)
    ctx = _clone_test_ctx(tmp_path, exploration_summary="## Summary\n3 files", deps_text="a -> b")
    eligibility = replace(_clone_test_eligibility(), candidate_roots=[candidate])

    async def author() -> dict[str, Any]:
        return await deep._run_diagram_kind(
            ctx, kind="flowchart", eligibility=eligibility, hunk_ranges={"a.py": [(1, 5)]},
            symbols=RepoSymbols(tmp_path), recorder=None, backend=_Wall())

    result = await author()
    assert len(prompts) == 2 and "## Summary" in prompts[0] and "a -> b" in prompts[0]
    assert "A `ROOT_NOT_CANDIDATE` verdict" in prompts[1] and "- `run` in a.py, lines 1-5" in prompts[1]
    assert (result["status"], result["reason"]) == ("omitted", "ROOT_NOT_CANDIDATE")
    assert result["spec_final"]["root"] == stray["root"]
    budget[0] = "wall_budget"
    truncated = await author()
    assert (truncated["status"], truncated["reason"]) == ("failed", "budget exhausted: wall_budget")
    assert len(prompts) == 3, "a truncated author turn gets no repair turn"

# --- Issue #1214: transport-aware advisory-input budgeting -------------------


def test_advisory_blocks_are_bounded_stable_and_count_the_omission() -> None:
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

    roots = [{"file": f"pkg/mod_{i:03d}.py", "name": f"handle_{i:03d}", "line": 1, "end_line": 40, "branch_points": 5}
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

@pytest.fixture
async def diagram_ctx(tmp_path: Path) -> AsyncIterator[FlowContext]:
    """Keep the artifact session owned and open for the entire author/repair test."""
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    (repo / "base.py").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", ".")
    commit(repo, "base")
    work = WorkContext(repo=repo, source=repo, base_branch="main", base_sha="",
                      head_branch=None, head_sha="", is_ephemeral=False, run_id="session-test")
    owner = resolve_private_workspace_owner(
        repo, locations=private_root_locations(base=(tmp_path / "private").resolve())
    )
    async with open_artifact_session(work, session_id="diagram-advisory", owner=owner) as session:
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
        ctx = FlowContext(config=RunConfig(target=str(repo)), work=work, registry=Registry(),
            data={"diff_path": diff_path, "diff": diff_path.read_text(encoding="utf-8"),
                  "exploration_dir": exploration, "dd": dd},
            artifacts=session,
        )
        yield ctx

async def _author_sequence(ctx: FlowContext, backend: Any) -> dict[str, Any]:
    return await deep._run_diagram_kind(
        ctx, kind="sequence", eligibility=_clone_test_eligibility(), hunk_ranges={},
        symbols=RepoSymbols(ctx.work.repo), recorder=None, backend=backend,
    )


async def test_author_inputs_stay_private_shared_and_fail_closed(
    diagram_ctx: FlowContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INLINE prompts name no private path; obsolete builders fail the kind; author and repair share one
    prepared set that revalidates fail-closed; an over-limit EXACT_PATHS diff is omitted, not aborted."""
    prompts: list[str] = []
    monkeypatch.setattr(deep, "run_agent", _recording_agent(prompts))
    result = await _author_sequence(diagram_ctx, SimpleNamespace(sandbox=True, model="fake"))
    assert result["status"] != "failed", result.get("reason")
    deep_dir = diagram_ctx.data["diff_path"].parent
    for private in (str(deep_dir / "diff.patch"), str(deep_dir / "hunk-index.json"),
                    str(diagram_ctx.data["exploration_dir"])):
        assert private not in prompts[0], f"INLINE prompt leaked {private}"
    assert "inlined below" in prompts[0] and "Sanctioned phase inputs" in prompts[0]

    registry = Registry()
    registry.override_prompt("diagram_sequence", _obsolete_sequence_builder)
    prompts.clear()
    with monkeypatch.context() as obsolete:
        obsolete.setattr(deep, "get_registry", lambda: registry)
        result = await _author_sequence(diagram_ctx, SimpleNamespace(sandbox=True, model="fake"))
    assert not prompts and result["status"] == "failed" and "TypeError" in result["reason"]
    assert "clone_mode" in result["reason"]

    seen: list[Any] = []

    async def _fake_run_agent(backend: Any, cwd: Any, prompt: str, **kwargs: Any) -> Any:
        seen.append(kwargs.get("sanctioned_inputs"))
        if len(seen) == 1:
            # First turn grounds nothing, so the repair turn runs.
            return {"participants": [{"name": "Ghost", "kind": "internal", "files": ["nope.py"]}]}, object(), None
        return {"participants": []}, object(), None

    monkeypatch.setattr(deep, "run_agent", _fake_run_agent)
    clone = SimpleNamespace(read_only_disposable_clone=True, model="fake")
    await _author_sequence(diagram_ctx, clone)
    assert len(seen) == 2, "the repair turn must have run"
    assert seen[0] is seen[1] is not None, "both turns must reuse the same prepared set"

    diff_path = diagram_ctx.data["diff_path"]
    diff_path.write_text("+" + ("x" * 1_100_000) + "\n", encoding="utf-8")   # > 1 MiB per-file limit
    diagram_ctx.data["diff"] = diff_path.read_text(encoding="utf-8")
    prompts.clear()
    monkeypatch.setattr(deep, "run_agent", _recording_agent(prompts))
    result = await _author_sequence(diagram_ctx, SimpleNamespace(model="fake"))
    assert prompts, "an EXACT_PATHS run must reach the author turn with the diff omitted, not aborted"
    assert [item["label"] for item in result["advisory"]["omitted"]] == ["diff"]
    assert result["advisory"]["transport"] == "exact_paths"

    # mutate-before-repair: the same object revalidates fail-closed, the authority for both turns.
    diagram_ctx.data["exploration_dir"].joinpath("summary.md").write_text("changed after capture\n", encoding="utf-8")
    with pytest.raises(SanctionedInputUnavailable):
        seen[1].revalidate(clone, diagram_ctx.work.repo, True)

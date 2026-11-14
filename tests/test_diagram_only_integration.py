"""Run diagram-only review and Phase B posting against real temporary Git data.

Mock only the backend and gh subprocess boundaries. Cover preserved deep
artifacts, honest manifest labels, empty/base-branch runs, and issue-comment
publication. CLI --start-at rejection is covered in test_cli.py."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.config import DIAGRAM_MAX_NODES
from daydream.findings import write_findings_artifact
from daydream.pr_review import diagram_marker, parse_diagram_markers, validate_diagram_payload
from daydream.runner import run
from tests.harness import diagram_repos as dr
from tests.harness.console import collapse_panel_text
from tests.harness.diagram_repos import load_diagram_artifact as _artifact
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.harness.scripts import cli_main as _cli_main
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

SEQUENCE_HEADING = "<details><summary><h3>Sequence Diagram</h3></summary>"
FLOWCHART_HEADING = "<details><summary><h3>Flowchart</h3></summary>"


# --- Harness -----------------------------------------------------------------


@pytest.fixture
def diagram_run(monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., Any], silence_console: Callable[..., None],
) -> Callable[..., Any]:
    """Run a ``--diagram-only`` flow with a diagram-scripted stub backend."""
    for module in (
        "daydream.deep.review_steps", "daydream.deep.diagram_steps", "daydream.phases",
        "daydream.runner", "daydream.pr_review",
    ):
        silence_console(module)
    # The orchestrator's error panels stay observable: a run that refuses to
    # export is only actionable if the operator can read why.
    silence_console("daydream.deep.orchestrator", keep=("console", "print_error"))
    silence(monkeypatch)

    async def _run(target: Path, *, diagram: str = "auto", specs: dict[str, list[dict[str, Any]]] | None = None,
        session_id: str | None = None, fail: frozenset[str] = frozenset(), exploration: bool = False,
        **config_overrides: Any,
    ) -> tuple[int, StubBackend]:

        stub = install_stub_backend(monkeypatch, target, enable_exploration=exploration)
        stub.diagram_specs = specs or {}
        stub.diagram_session_id = session_id
        stub.diagram_fail = fail
        config = make_config(target, output_mode="diagram", diagram=diagram, **config_overrides)
        return await run(config), stub

    return _run


def _issue_comments(fake_gh: FakeGh) -> list[dict[str, Any]]:
    """The bodies POSTed to the issue-comments endpoint, in order."""
    return [call.payload for call in fake_gh.calls("POST", "/repos/acme/widgets/issues/7/comments")]


def _post_findings_cli(artifact_path: Path, head_sha: str, *, pr: int = 7, repo: str = "acme/widgets") -> int:
    """Run the production ``post-findings`` Phase-B CLI for *artifact_path*."""
    return _cli_main(["post-findings", str(artifact_path), "--pr", str(pr), "--head-sha", head_sha, "--repo", repo])


def _diagram_target(tmp_path: Path, fake_gh: FakeGh) -> Path:
    """Build the cross-module repo and serve an open PR for it."""
    target = dr.build_cross_module_repo(tmp_path)
    fake_gh.serve_open_pr(target)
    return target


@pytest.fixture
def target(tmp_path: Path, fake_gh: FakeGh) -> Path:
    """A cross-module repo with an open PR served by the fake gh."""
    return _diagram_target(tmp_path, fake_gh)


def _diagram_phase_end(target: Path) -> dict[str, Any]:
    paths = list((target / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(paths) == 1
    trajectory = json.loads(paths[0].read_text(encoding="utf-8"))
    ends = [event
        for event in trajectory["extra"]["phase_events"]
        if event["event"] == "phase_end" and event["phase"] == "diagram"
    ]
    assert len(ends) == 1
    end = ends[0]
    assert isinstance(end, dict)
    return end


async def _render_flowchart_artifact(diagram_run: Callable[..., Any], tmp_path: Path, fake_gh: FakeGh
) -> tuple[Path, Path, dict[str, Any]]:
    """Run Phase A once; return *target*, the findings path, and its artifact."""
    target = dr.build_branch_heavy_repo(tmp_path)
    fake_gh.serve_open_pr(target)
    artifact_path = tmp_path / "findings.json"
    exit_code, _ = await diagram_run(
        target, diagram="flowchart", specs={"flowchart": [dr.flowchart_spec()]}, findings_out=str(artifact_path),
        pr_number=7,
    )
    assert exit_code == 0
    return target, artifact_path, json.loads(artifact_path.read_text(encoding="utf-8"))


# --- Spec test 12: end to end, per kind -------------------------------------


async def test_second_diagram_run_minimizes_only_its_own_kind(
    tmp_path: Path, fake_gh: FakeGh, diagram_run: Callable[..., Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat run folds its own prior comment and leaves the other kind alone."""
    target = dr.build_both_signals_repo(tmp_path)
    fake_gh.serve_open_pr(target)
    monkeypatch.setenv("DAYDREAM_BOT_HANDLE", "daydream")

    head = git_ops.head_sha(target)

    fake_gh.serve_prior_issue_comments([{"id": 1, "node_id": "IC_prior_sequence",
                "body": f"{diagram_marker('sequence', head)}\n\nold sequence",
                "user": {"login": "daydream[bot]"},
            }, {"id": 2, "node_id": "IC_prior_flowchart",
                "body": f"{diagram_marker('flowchart', head)}\n\nold flowchart",
                "user": {"login": "daydream[bot]"},
            }, {"id": 3, "node_id": "IC_human",
                "body": f"{diagram_marker('sequence', head)}\n\nimpersonation",
                "user": {"login": "someone-else"},
            },
        ]
    )

    exit_code, _ = await diagram_run(
        target, diagram="sequence", specs={"sequence": [dr.sequence_spec()]}, pr_repo="acme/widgets",
    )

    assert exit_code == 0
    minimized = [call.payload["variables"]["subjectId"]
        for call in fake_gh.calls("POST", "graphql")
        if "minimizeComment" in call.payload.get("query", "")
    ]
    assert minimized == ["IC_prior_sequence"], ("only the bot's own prior comment for the requested kind is folded")
    assert len(_issue_comments(fake_gh)) == 1


# --- Spec test 7 (diagram-only half): omission notice -----------------------

async def test_omitted_kind_posts_an_omission_notice_and_a_failed_kind_exits_one(
    tmp_path: Path, fake_gh: FakeGh, diagram_run: Callable[..., Any], target: Path,
) -> None:
    """An explicit request that grounds to nothing explains its omission; a failed kind exits 1."""
    thin = dr.sequence_spec()
    thin["messages"] = thin["messages"][:2]

    exit_code, _ = await diagram_run(target, diagram="sequence", specs={"sequence": [thin]}, exploration=True)

    assert exit_code == 0, "an omission is a successful run, not a failure"
    exploration = target / ".daydream" / "exploration"
    assert "| `pkg_b/client.py` | modified | static |" in (exploration / "affected_files.md").read_text()
    scanned = {path.name: path.stat().st_mtime_ns for path in exploration.iterdir()}
    assert "cache-key" in scanned
    assert _artifact(target)["results"]["sequence"]["status"] == "omitted"
    body = _issue_comments(fake_gh)[0]["body"]
    assert SEQUENCE_HEADING not in body
    assert "No sequence diagram was rendered for this pull request." in body
    assert "Grounding floor not met: TOO_FEW_MESSAGES." in body
    end = _diagram_phase_end(target)
    assert end["status"] == "succeeded"
    assert "reason_code" not in end
    # In diagram-only mode the diagram is the deliverable: a failed kind fails the run.
    exit_code, _ = await diagram_run(target, diagram="sequence", fail=frozenset({"sequence"}), exploration=True)
    assert exit_code == 1
    # The repeat run on the same head reuses the pre-scan: nothing under it is rewritten.
    assert {path.name: path.stat().st_mtime_ns for path in exploration.iterdir()} == scanned
    assert _artifact(target)["results"]["sequence"]["status"] == "failed"

async def test_nothing_eligible_posts_an_explanatory_comment(
    tmp_path: Path, fake_gh: FakeGh, diagram_run: Callable[..., Any],
) -> None:
    """``--diagram-only auto`` on a flat diff explains that nothing was eligible."""
    target = dr.build_flat_repo(tmp_path)
    fake_gh.serve_open_pr(target)

    exit_code, stub = await diagram_run(target, diagram="auto")

    assert exit_code == 0
    body = _issue_comments(fake_gh)[0]["body"]
    assert "No grounded diagram was eligible for this pull request" in body

    assert parse_diagram_markers(body) == []
    assert all("You are the sequence-diagram author" not in call["prompt"] for call in stub.calls)
    fake_gh.set_response("POST", "/repos/acme/widgets/issues/7/comments", value={"__error__": "HTTP 502"})
    exit_code, _ = await diagram_run(target, diagram="auto")
    assert exit_code == 1, "a comment that was not confirmed is not a delivered diagram"


# --- Spec test 13: Phase A -> Phase B ---------------------------------------

async def test_findings_out_rejects_dirty_moved_or_unresolved_heads_before_diagram_export(
    tmp_path: Path, fake_gh: FakeGh, diagram_run: Callable[..., Any], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit-bound diagram export refuses a tree it cannot bind to the PR head, before any agent turn.

    The diagram path reads its diff and grounding from working-tree content, so the
    clean-checkout gate is mode-independent. A moved PR head ends the run, and a
    failed PR lookup is reported as a lookup failure, never as a diff failure.
    """
    target = dr.build_branch_heavy_repo(tmp_path)
    fake_gh.serve_open_pr(target)
    artifact_path = tmp_path / "findings.json"

    async def export() -> tuple[int, str]:
        exit_code, stub = await diagram_run(
            target, diagram="flowchart", specs={"flowchart": [dr.flowchart_spec()]},
            findings_out=str(artifact_path), pr_number=7,
        )
        assert not artifact_path.exists(), "no artifact may name a commit it did not read"
        assert stub.calls == [], "the run must reject before spending an agent turn"
        return exit_code, collapse_panel_text(capsys)

    source = target / "app/pipeline.py"
    committed = source.read_bytes()
    source.write_text("DIRTY = True\n", encoding="utf-8")
    exit_code, panel = await export()
    assert exit_code == 1 and "clean analyzed checkout" in panel
    source.write_bytes(committed)

    base = git(target, "rev-parse", "main")
    fake_gh.serve_pr_view({"number": 7, "state": "OPEN", "headRefName": "feature", "baseRefName": "main",
        "headRefOid": base, "headRepository": {"name": "widgets", "nameWithOwner": "acme/widgets"},
        "headRepositoryOwner": {"login": "acme"}, "url": "https://github.com/acme/widgets/pull/7", "body": ""})
    exit_code, panel = await export()
    assert exit_code == 1
    assert base[:12] in panel and git_ops.head_sha(target)[:12] in panel
    assert "re-run against the new head" in panel
    exit_code, _ = await diagram_run(target, diagram="flowchart", specs={"flowchart": [dr.flowchart_spec()]})
    assert exit_code == 1 and _issue_comments(fake_gh) == []

    def explode(*_a: Any, **_k: Any) -> Any:
        raise git_ops.GitError("gh pr view: connection reset")

    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", explode)
    monkeypatch.setattr("daydream.pr_review.find_open_pr", explode)
    exit_code, panel = await export()
    assert exit_code == 1
    assert "Unable to resolve pull request for findings export" in panel
    assert "connection reset" in panel
    assert "Unable to determine base branch for diff" not in panel


async def test_phase_b_reposts_a_diagram_artifact_without_any_checkout(
    tmp_path: Path, fake_gh: FakeGh, diagram_run: Callable[..., Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a checkout, Phase B reads all citation evidence from the head-SHA API."""
    target, artifact_path, artifact = await _render_flowchart_artifact(diagram_run, tmp_path, fake_gh)

    expected = _artifact(target)["results"]["flowchart"]["mermaid"]
    head_sha = artifact["head_sha"]
    fake_gh.set_response("GET", "repos/acme/widgets/contents/app/pipeline.py",
        value={"type": "file", "path": "app/pipeline.py", "sha": "d" * 40, "encoding": "base64",
            "content": base64.b64encode(git_ops.show(target, head_sha, "app/pipeline.py")).decode(),
        },
    )

    no_checkout = tmp_path / "runner-workspace"
    no_checkout.mkdir()
    monkeypatch.chdir(no_checkout)
    rc = _post_findings_cli(artifact_path, head_sha)

    assert rc == 0
    posted = _issue_comments(fake_gh)
    assert len(posted) == 1
    assert expected in posted[0]["body"]
    assert [call.endpoint for call in fake_gh.calls("GET")].count(
        f"repos/acme/widgets/contents/app/pipeline.py?ref={head_sha}"
    ) == 1

_INVALID_REFERENCES: list[tuple[str, Callable[[dict[str, Any]], Any], str]] = [
    ("sequence", lambda s: s["participants"].append(s["participants"][0]), "duplicate participant"),
    ("sequence", lambda s: s["messages"][0].update({"to": "missing"}), "unknown participant"),
    ("sequence", lambda s: s.update({"blocks": [{"kind": "opt", "branches": [
        {"condition": "ready", "evidence": {"file": "a.py", "line": 1}, "messages": [999]}]}]}), "unknown message"),
    ("flowchart", lambda s: s["nodes"].append(s["nodes"][0]), "duplicate node"),
    ("flowchart", lambda s: s["edges"][0].update({"to": "missing"}), "unknown endpoint"),
    ("flowchart", lambda s: s["nodes"][0]["evidence"].update({"file": "other.py"}), "outside its root"),
]


def test_phase_b_rejects_invalid_diagram_payloads_before_any_network_call(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema failures, render-cap overruns and dangling references are rejected before rendering or posting."""
    artifact_path = tmp_path / "bad.json"
    write_findings_artifact(artifact_path,
        {"schema_version": 2, "review_warnings": [], "repo": "acme/widgets", "pr_number": 7, "head_sha": "h" * 40,
            "run_info": None, "kind": "diagram", "diagrams": {"eligibility": {"flowchart": {"eligible": True}},
                "results": {"flowchart": {"status": "rendered",
                        # ``root`` must be an object with file/name/line.
                        "spec_final": {"root": None, "nodes": [], "edges": []},
                    }
                },
            }, "findings": [],
        },
    )
    monkeypatch.chdir(tmp_path)
    assert _post_findings_cli(artifact_path, "h" * 40) == 1
    assert fake_gh.calls("POST") == []

    nodes = [{"id": f"node-{index}", "kind": "process", "label": "Invented source",
              "evidence": {"file": "a.py", "line": 1, "symbol": None}} for index in range(DIAGRAM_MAX_NODES + 1)]
    over_cap = validate_diagram_payload({"results": {"flowchart": {"status": "rendered",
        "spec_final": {"root": {"file": "a.py", "name": "run", "line": 1}, "nodes": nodes, "edges": []}}}})
    assert over_cap is not None and "nodes render cap" in over_cap
    for kind, mutate, problem in _INVALID_REFERENCES:
        spec = dr.sequence_spec() if kind == "sequence" else dr.flowchart_spec()
        mutate(spec)
        result = validate_diagram_payload({"results": {kind: {"status": "rendered", "spec_final": spec}}})
        assert result is not None and problem in result, problem


# --- Spec test 14 (diagram-only half): agent error exits 1 ------------------


async def test_pr_lookup_failure_in_diagram_only_mode_exits_one_with_diagnostic(
    tmp_path: Path, diagram_run: Callable[..., Any], fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operational lookup failure is reported, not misclassified as absence."""
    target = dr.build_cross_module_repo(tmp_path)
    fake_gh.set_response("pr-view", value={"__error__": "authentication required"})
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "daydream.deep.diagram_steps.print_error", lambda _console, title, message: errors.append((title, message)),
    )

    exit_code, _ = await diagram_run(target, diagram="sequence", specs={"sequence": [dr.sequence_spec()]})

    assert exit_code == 1
    assert len(errors) == 1
    assert errors[0][0] == "Diagram PR Lookup Failed"
    assert "authentication required" in errors[0][1]
    assert _issue_comments(fake_gh) == []


# --- Regression (a): prior deep artifacts survive ---------------------------


# --- Regression (b): recorder + manifest label the run honestly -------------

# --- Regressions (d) and (e) ------------------------------------------------

async def test_diagram_only_on_the_base_branch_is_not_a_wrong_branch_error(
    tmp_path: Path, fake_gh: FakeGh, diagram_run: Callable[..., Any], target: Path,
) -> None:
    """Diagram mode neither fixes nor commits, so the base branch is allowed."""
    git(target, "checkout", "main")

    exit_code, _ = await diagram_run(target, diagram="sequence")

    # An empty base..HEAD diff on ``main`` exits 0 with a warning -- crucially
    # NOT the WrongBranchError the loop/shallow paths would raise.
    assert exit_code == 0


# --- A review artifact may carry diagrams too --------------------------------

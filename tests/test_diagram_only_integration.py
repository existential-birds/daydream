"""Real-path tests for the ``--diagram-only`` flow (issue #1113).

Every test enters through ``daydream.runner.run`` (or, for Phase B,
``daydream.cli.main``) against a real temporary git repository. The only mocks
are the stub backend at the ``create_backend`` seam and the in-process ``gh``
fake at the ``subprocess.run`` boundary, so ``git_ops``, the ``gh api``
tempfile path, the marker round trip and the GraphQL minimize mutation all run
for real.

Spec test coverage: 7 (the diagram-only half), 12, 13, 14 (the diagram-only
half), plus the plan's regression tests (a) prior deep artifacts survive,
(b) the recorder/manifest label a diagram run honestly, (d) an empty diff exits
0, (e) a base-branch invocation is not a wrong-branch error, and (f) a
diagram-only run posts an issue comment and never a review. Regression (c)
(``--start-at`` rejection) is a CLI-level check and lives in
``tests/test_cli.py``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import cli, git_ops
from daydream.config import DIAGRAM_MAX_NODES
from tests.harness import diagram_repos as dr
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

SEQUENCE_HEADING = "<details><summary><h3>Sequence Diagram</h3></summary>"
FLOWCHART_HEADING = "<details><summary><h3>Flowchart</h3></summary>"


# --- Harness -----------------------------------------------------------------


def _serve_pr(fake_gh: FakeGh, target: Path) -> None:
    """Configure an open PR whose SHAs match the real fixture repository."""
    fake_gh.serve_pr_view(
        {
            "number": 7,
            "state": "OPEN",
            "headRefName": "feature",
            "baseRefName": "main",
            "headRefOid": git_ops.head_sha(target),
            "baseRefOid": git_ops.merge_base(target, "main"),
            "url": "https://github.com/acme/widgets/pull/7",
            "body": "",
        }
    )


@pytest.fixture
def diagram_run(
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Any],
    silence_console: Callable[..., None],
) -> Callable[..., Any]:
    """Run a ``--diagram-only`` flow with a diagram-scripted stub backend."""
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
        diagram: str = "auto",
        specs: dict[str, list[dict[str, Any]]] | None = None,
        emit_reads: bool = True,
        session_id: str | None = None,
        fail: frozenset[str] = frozenset(),
        **config_overrides: Any,
    ) -> tuple[int, StubBackend]:
        from daydream.runner import run

        stub = install_stub_backend(monkeypatch, target)
        stub.diagram_specs = specs or {}
        stub.diagram_emit_reads = emit_reads
        stub.diagram_session_id = session_id
        stub.diagram_fail = fail
        config = make_config(
            target, output_mode="diagram", diagram=diagram, **config_overrides
        )
        return await run(config), stub

    return _run


def _issue_comments(fake_gh: FakeGh) -> list[dict[str, Any]]:
    """The bodies POSTed to the issue-comments endpoint, in order."""
    return [
        call.payload
        for call in fake_gh.calls("POST", "/repos/acme/widgets/issues/7/comments")
    ]


def _artifact(target: Path) -> dict[str, Any]:
    path = target / ".daydream" / "deep" / "diagram.json"
    assert path.is_file(), f"diagram artifact missing at {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _cli_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    raise AssertionError("cli.main() must exit via sys.exit")


# --- Spec test 12: end to end, per kind -------------------------------------


@pytest.mark.parametrize(
    ("kind", "repo_builder", "spec_builder", "heading"),
    [
        (
            "sequence",
            dr.build_cross_module_repo,
            dr.sequence_spec,
            SEQUENCE_HEADING,
        ),
        (
            "flowchart",
            dr.build_branch_heavy_repo,
            dr.flowchart_spec,
            FLOWCHART_HEADING,
        ),
    ],
    ids=["sequence", "flowchart"],
)
async def test_diagram_only_posts_a_marked_issue_comment(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
    kind: str,
    repo_builder: Callable[[Path], Path],
    spec_builder: Callable[[], dict[str, Any]],
    heading: str,
) -> None:
    """Only exploration + diagram run, and the deliverable is an issue comment."""
    target = repo_builder(tmp_path)
    _serve_pr(fake_gh, target)

    exit_code, stub = await diagram_run(
        target, diagram=kind, specs={kind: [spec_builder()]}
    )

    assert exit_code == 0
    assert _artifact(target)["results"][kind]["status"] == "rendered"

    comments = _issue_comments(fake_gh)
    assert len(comments) == 1
    body = comments[0]["body"]
    from daydream.pr_review import parse_diagram_markers

    assert parse_diagram_markers(body) == [(kind, git_ops.head_sha(target))]
    assert heading in body
    assert "🧙 Posted by [daydream" in body

    # Regression (f): a diagram-only run never posts a review.
    assert fake_gh.calls("POST", "/repos/acme/widgets/pulls/7/reviews") == []
    # Only the diagram phase ran an agent: no per-stack review, no merge.
    prompts = [call["prompt"] for call in stub.calls]
    assert all("cross-stack merge agent" not in prompt for prompt in prompts)
    assert all("you are reviewing the" not in prompt.lower() for prompt in prompts)
    assert not (target / ".daydream" / "deep" / "merged-items.json").exists()


async def test_second_diagram_run_minimizes_only_its_own_kind(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat run folds its own prior comment and leaves the other kind alone."""
    target = dr.build_both_signals_repo(tmp_path)
    _serve_pr(fake_gh, target)
    monkeypatch.setenv("DAYDREAM_BOT_HANDLE", "daydream")

    head = git_ops.head_sha(target)
    from daydream.pr_review import diagram_marker

    fake_gh.serve_prior_issue_comments(
        [
            {
                "id": 1,
                "node_id": "IC_prior_sequence",
                "body": f"{diagram_marker('sequence', head)}\n\nold sequence",
                "user": {"login": "daydream[bot]"},
            },
            {
                "id": 2,
                "node_id": "IC_prior_flowchart",
                "body": f"{diagram_marker('flowchart', head)}\n\nold flowchart",
                "user": {"login": "daydream[bot]"},
            },
            {
                "id": 3,
                "node_id": "IC_human",
                "body": f"{diagram_marker('sequence', head)}\n\nimpersonation",
                "user": {"login": "someone-else"},
            },
        ]
    )

    exit_code, _ = await diagram_run(
        target,
        diagram="sequence",
        specs={"sequence": [dr.sequence_spec()]},
        pr_repo="acme/widgets",
    )

    assert exit_code == 0
    minimized = [
        call.payload["variables"]["subjectId"]
        for call in fake_gh.calls("POST", "graphql")
        if "minimizeComment" in call.payload.get("query", "")
    ]
    assert minimized == ["IC_prior_sequence"], (
        "only the bot's own prior comment for the requested kind is folded"
    )
    assert len(_issue_comments(fake_gh)) == 1


# --- Spec test 7 (diagram-only half): omission notice -----------------------


async def test_omitted_kind_posts_an_omission_notice(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
) -> None:
    """An explicit request that grounds to nothing says so, with counts and codes."""
    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)
    thin = dr.sequence_spec()
    thin["messages"] = thin["messages"][:2]

    exit_code, _ = await diagram_run(
        target, diagram="sequence", specs={"sequence": [thin]}
    )

    assert exit_code == 0, "an omission is a successful run, not a failure"
    assert _artifact(target)["results"]["sequence"]["status"] == "omitted"
    body = _issue_comments(fake_gh)[0]["body"]
    assert SEQUENCE_HEADING not in body
    assert "No sequence diagram was rendered for this pull request." in body
    assert "Grounding floor not met: TOO_FEW_MESSAGES." in body
    assert "5 elements proposed, 5 grounded on the first pass" in body


async def test_nothing_eligible_posts_an_explanatory_comment(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
) -> None:
    """``--diagram-only auto`` on a flat diff explains that nothing was eligible."""
    target = dr.build_flat_repo(tmp_path)
    _serve_pr(fake_gh, target)

    exit_code, stub = await diagram_run(target, diagram="auto")

    assert exit_code == 0
    body = _issue_comments(fake_gh)[0]["body"]
    assert "No grounded diagram was eligible for this pull request" in body
    from daydream.pr_review import parse_diagram_markers

    assert parse_diagram_markers(body) == []
    assert all(
        "You are the sequence-diagram author" not in call["prompt"] for call in stub.calls
    )


# --- Spec test 13: Phase A -> Phase B ---------------------------------------


async def test_findings_out_writes_a_diagram_artifact_phase_b_reposts_it(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase A writes ``kind == "diagram"``; Phase B re-renders identical mermaid."""
    target = dr.build_branch_heavy_repo(tmp_path)
    _serve_pr(fake_gh, target)
    artifact_path = tmp_path / "findings.json"

    exit_code, _ = await diagram_run(
        target,
        diagram="flowchart",
        specs={"flowchart": [dr.flowchart_spec()]},
        findings_out=str(artifact_path),
        pr_number=7,
    )

    assert exit_code == 0
    # Phase A stops before any GitHub write.
    assert _issue_comments(fake_gh) == []
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["kind"] == "diagram"
    assert artifact["findings"] == []
    results = artifact["diagrams"]["results"]
    assert results["flowchart"]["status"] == "rendered"
    assert "mermaid" not in results["flowchart"], "the poster re-renders from the spec"

    expected = _artifact(target)["results"]["flowchart"]["mermaid"]

    # Phase B: the privileged poster, entered from the production CLI.
    monkeypatch.chdir(target)
    rc = _cli_main(
        [
            "post-findings",
            str(artifact_path),
            "--pr",
            "7",
            "--head-sha",
            artifact["head_sha"],
            "--repo",
            "acme/widgets",
        ]
    )
    assert rc == 0
    posted = _issue_comments(fake_gh)
    assert len(posted) == 1
    assert expected in posted[0]["body"]
    assert fake_gh.calls("POST", "/repos/acme/widgets/pulls/7/reviews") == []


async def test_phase_b_rejects_diagram_evidence_missing_from_the_immutable_head(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase B must not trust a structurally valid, artifact-supplied citation."""
    target = dr.build_branch_heavy_repo(tmp_path)
    _serve_pr(fake_gh, target)
    artifact_path = tmp_path / "findings.json"

    exit_code, _ = await diagram_run(
        target,
        diagram="flowchart",
        specs={"flowchart": [dr.flowchart_spec()]},
        findings_out=str(artifact_path),
        pr_number=7,
    )

    assert exit_code == 0
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    spec_final = artifact["diagrams"]["results"]["flowchart"]["spec_final"]
    spec_final["root"]["file"] = "untrusted.py"
    for node in spec_final["nodes"]:
        node["evidence"]["file"] = "untrusted.py"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.chdir(target)
    rc = _cli_main(
        [
            "post-findings",
            str(artifact_path),
            "--pr",
            "7",
            "--head-sha",
            artifact["head_sha"],
            "--repo",
            "acme/widgets",
        ]
    )

    assert rc == 1
    assert fake_gh.calls("POST") == []


def test_phase_b_rejects_an_invalid_diagrams_payload(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rendered spec that fails its schema is rejected before any network call."""
    from daydream.findings import FINDINGS_SCHEMA_VERSION, write_findings_artifact

    artifact_path = tmp_path / "bad.json"
    write_findings_artifact(
        artifact_path,
        {
            "schema_version": FINDINGS_SCHEMA_VERSION,
            "repo": "acme/widgets",
            "pr_number": 7,
            "head_sha": "h" * 40,
            "run_info": None,
            "kind": "diagram",
            "diagrams": {
                "eligibility": {"flowchart": {"eligible": True}},
                "results": {
                    "flowchart": {
                        "status": "rendered",
                        # ``root`` must be an object with file/name/line.
                        "spec_final": {"root": None, "nodes": [], "edges": []},
                        "grounding": {"elements": [], "summary": {}, "capped": {}},
                    }
                },
            },
            "findings": [],
        },
    )
    monkeypatch.chdir(tmp_path)

    rc = _cli_main(
        [
            "post-findings",
            str(artifact_path),
            "--pr",
            "7",
            "--head-sha",
            "h" * 40,
            "--repo",
            "acme/widgets",
        ]
    )

    assert rc == 1
    assert fake_gh.calls("POST") == []


def test_phase_b_rejects_rendered_source_claims_without_grounding_attestations() -> None:
    from daydream.pr_review import validate_diagram_payload

    payload = {
        "results": {
            "flowchart": {
                "status": "rendered",
                "spec_final": {
                    "root": {"file": "does/not/exist.py", "name": "invented", "line": 999999},
                    "nodes": [
                        {
                            "id": "start",
                            "kind": "start",
                            "label": "Invented source",
                            "evidence": {
                                "file": "does/not/exist.py",
                                "line": 999999,
                                "symbol": "invented",
                            },
                        }
                    ],
                    "edges": [],
                },
                "grounding": {
                    "elements": [],
                    "summary": {
                        "proposed": 0,
                        "grounded_first_pass": 0,
                        "repaired": 0,
                        "pruned": 0,
                    },
                    "capped": {},
                    "root_range": [999999, 999999],
                },
            }
        }
    }

    problem = validate_diagram_payload(payload)

    assert problem == "flowchart grounding attestation does not cover root at final index 0"


def test_phase_b_rejects_over_cap_specs_before_rendering() -> None:
    from daydream.pr_review import validate_diagram_payload

    nodes = [
        {
            "id": f"node-{index}",
            "kind": "process",
            "label": "Invented source",
            "evidence": {"file": "a.py", "line": 1, "symbol": None},
        }
        for index in range(DIAGRAM_MAX_NODES + 1)
    ]
    payload = {
        "results": {
            "flowchart": {
                "status": "rendered",
                "spec_final": {
                    "root": {"file": "a.py", "name": "run", "line": 1},
                    "nodes": nodes,
                    "edges": [],
                },
                "grounding": {
                    "elements": [],
                    "summary": {
                        "proposed": 0,
                        "grounded_first_pass": 0,
                        "repaired": 0,
                        "pruned": 0,
                    },
                    "capped": {},
                    "root_range": None,
                },
            }
        }
    }

    problem = validate_diagram_payload(payload)

    assert problem is not None
    assert "nodes render cap" in problem


# --- Spec test 14 (diagram-only half): agent error exits 1 ------------------


async def test_agent_error_in_diagram_only_mode_exits_one(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
) -> None:
    """The diagram IS the deliverable here, so a failed kind fails the run."""
    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)

    exit_code, _ = await diagram_run(
        target,
        diagram="sequence",
        specs={"sequence": [dr.sequence_spec()]},
        fail=frozenset({"sequence"}),
    )

    assert exit_code == 1
    # The artifact is written BEFORE the run stops, so the evidence survives.
    failed = _artifact(target)["results"]["sequence"]
    assert failed["status"] == "failed"
    assert "RuntimeError" in failed["reason"]
    assert _issue_comments(fake_gh) == []


async def test_returned_failure_in_diagram_only_mode_exits_one(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned failed result follows the same diagram-only exit path."""
    from daydream.deep import orchestrator

    async def _return_failure(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": "stub: diagram author returned failure",
            "spec_proposed": None,
            "spec_final": None,
            "grounding": None,
            "omit_reasons": [],
            "mermaid": None,
        }

    monkeypatch.setattr(orchestrator, "_run_diagram_kind", _return_failure)
    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)

    exit_code, _ = await diagram_run(
        target,
        diagram="sequence",
    )

    assert exit_code == 1
    failed = _artifact(target)["results"]["sequence"]
    assert failed["status"] == "failed"
    assert failed["reason"] == "stub: diagram author returned failure"
    assert _issue_comments(fake_gh) == []


async def test_no_resolvable_pr_in_diagram_only_mode_exits_one(
    tmp_path: Path,
    diagram_run: Callable[..., Any],
    fake_gh: FakeGh,
) -> None:
    """Mirrors ``--comment``: no PR means the run's deliverable is unreachable."""
    target = dr.build_cross_module_repo(tmp_path)
    # No serve_pr_view: ``gh pr view`` finds nothing.

    exit_code, _ = await diagram_run(
        target, diagram="sequence", specs={"sequence": [dr.sequence_spec()]}
    )

    assert exit_code == 1
    assert _artifact(target)["results"]["sequence"]["status"] == "rendered"
    assert _issue_comments(fake_gh) == []


# --- Regression (a): prior deep artifacts survive ---------------------------


async def test_diagram_only_run_preserves_prior_deep_artifacts(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
) -> None:
    """A diagram-only run must not clear ``.daydream/deep/``.

    ``start_at`` defaults to ``"review"``, which is the spine's fresh-run
    branch, so without the mode guard the run would ``rmtree`` the previous
    deep review's resumable artifacts.
    """
    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True)
    (deep / "merged-items.json").write_text('{"items": []}', encoding="utf-8")
    (deep / "intent.md").write_text("prior intent\n", encoding="utf-8")
    (deep / "diff-key").write_text("prior-key\n", encoding="utf-8")

    exit_code, _ = await diagram_run(
        target, diagram="sequence", specs={"sequence": [dr.sequence_spec()]}
    )

    assert exit_code == 0
    assert (deep / "merged-items.json").read_text(encoding="utf-8") == '{"items": []}'
    assert (deep / "intent.md").read_text(encoding="utf-8") == "prior intent\n"
    # The diff key is NOT rewritten either: a diagram run attests none of the
    # artifacts that key stands for.
    assert (deep / "diff-key").read_text(encoding="utf-8") == "prior-key\n"
    assert (deep / "diagram.json").is_file()


# --- Regression (b): recorder + manifest label the run honestly -------------


async def test_diagram_run_flow_label_and_manifest_backends(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
    archive_dir: Path,
) -> None:
    """Every step is stamped ``daydream_run_flow: "diagram"``; no fix/test backend.

    Reusing the ``TTT`` label would have been worse than wrong: the archive's
    ``_flow_runs_merge`` returns True for TTT, so a diagram run would inherit a
    previous deep review's ``merged-items.json`` as its own pipeline state --
    and regression (a) guarantees that file is still on disk.
    """
    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)

    exit_code, _ = await diagram_run(
        target,
        diagram="sequence",
        specs={"sequence": [dr.sequence_spec()]},
        archive=True,
    )
    assert exit_code == 0

    runs = sorted((target / ".daydream" / "runs").iterdir())
    assert len(runs) == 1
    main = json.loads((runs[0] / "trajectory.json").read_text(encoding="utf-8"))
    assert {step["extra"]["daydream_run_flow"] for step in main["steps"]} == {"diagram"}
    assert "diagram" in {step["extra"]["daydream_phase"] for step in main["steps"]}
    # The fork holding the author turn is named for its kind.
    fork = runs[0] / "trajectories" / "diagram-sequence.json"
    assert fork.is_file()
    fork_data = json.loads(fork.read_text(encoding="utf-8"))
    assert {step["extra"]["daydream_phase"] for step in fork_data["steps"]} == {"diagram"}

    manifests = sorted(archive_dir.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["run"]["flow"] == "diagram"
    assert "fix_backend" not in manifest["run"]
    assert "test_backend" not in manifest["run"]


# --- Regressions (d) and (e) ------------------------------------------------


async def test_empty_diff_in_diagram_mode_exits_zero(
    tmp_path: Path,
    diagram_run: Callable[..., Any],
) -> None:
    """No diff is nothing to diagram, which is a success, not an error."""
    target = dr.build_cross_module_repo(tmp_path)
    # Fold the feature branch's content back so base..HEAD is empty.
    git(target, "checkout", "main")
    git(target, "checkout", "-b", "empty-feature")
    (target / "notes.txt").write_text("scratch\n", encoding="utf-8")
    git(target, "add", "notes.txt")
    commit(target, "add scratch")
    git(target, "rm", "-q", "notes.txt")
    commit(target, "remove scratch")
    git(target, "diff", "--quiet", "main")

    exit_code, stub = await diagram_run(target, diagram="sequence")

    assert exit_code == 0
    assert stub.calls == []
    assert not (target / ".daydream" / "deep" / "diagram.json").exists()


async def test_diagram_only_on_the_base_branch_is_not_a_wrong_branch_error(
    tmp_path: Path,
    fake_gh: FakeGh,
    diagram_run: Callable[..., Any],
) -> None:
    """Diagram mode neither fixes nor commits, so the base branch is allowed."""
    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)
    git(target, "checkout", "main")

    exit_code, _ = await diagram_run(target, diagram="sequence")

    # An empty base..HEAD diff on ``main`` exits 0 with a warning -- crucially
    # NOT the WrongBranchError the loop/shallow paths would raise.
    assert exit_code == 0


# --- A review artifact may carry diagrams too --------------------------------


async def test_review_findings_artifact_carries_diagrams_and_phase_b_renders_them(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Any],
    silence_console: Callable[..., None],
) -> None:
    """A ``--findings-out`` deep review ships its diagrams; Phase B posts them.

    Without this the blocks would exist in ``review-output.md`` but silently
    vanish from the PR whenever the two-phase CI path is used.
    """
    from daydream.runner import run

    for module in (
        "daydream.deep.orchestrator",
        "daydream.phases",
        "daydream.runner",
        "daydream.pr_review",
    ):
        silence_console(module)
    silence(monkeypatch)

    target = dr.build_cross_module_repo(tmp_path)
    _serve_pr(fake_gh, target)
    stub = install_stub_backend(monkeypatch, target)
    stub.diagram_specs = {"sequence": [dr.sequence_spec()]}
    stub.diagram_emit_reads = True
    artifact_path = tmp_path / "review-findings.json"

    exit_code = await run(
        make_config(target, findings_out=str(artifact_path), pr_number=7)
    )
    assert exit_code == 0

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["kind"] == "review"
    assert artifact["findings"], "the review half of the artifact is still populated"
    assert artifact["diagrams"]["results"]["sequence"]["status"] == "rendered"
    expected = _artifact(target)["results"]["sequence"]["mermaid"]

    monkeypatch.chdir(target)
    rc = _cli_main(
        [
            "post-findings",
            str(artifact_path),
            "--pr",
            "7",
            "--head-sha",
            artifact["head_sha"],
            "--repo",
            "acme/widgets",
        ]
    )
    assert rc == 0
    reviews = fake_gh.calls("POST", "/repos/acme/widgets/pulls/7/reviews")
    assert len(reviews) == 1
    body = reviews[0].payload["body"]
    assert SEQUENCE_HEADING in body
    assert expected in body
    # A diagram in a review artifact posts as part of the review, not as a
    # separate issue comment.
    assert _issue_comments(fake_gh) == []

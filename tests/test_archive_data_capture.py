"""Archive-time data-capture tests (issue #124).

Covers the two capture gaps this feature closes:

1. **Eval on by default.** ``analyze_session`` is file-based and cheap, so it
   runs on every archive unless ``--no-eval`` opts out. AC1/AC1b assert the
   manifest's eval metrics are populated on a default run and null with
   ``--no-eval``.
2. **Recommended-change patch.** A separate ``recommended.patch`` (daydream's
   proposed diff, captured post-fix) is archived distinct from ``diff.patch``
   (the PR-under-review diff), and the applied-signal cascades read it. AC3/AC4.

The deep AC1/AC3 test drives the production entrypoint (``runner.run`` →
``run_deep``) through a real temp git worktree, reusing the deep-orchestrator
stub harness. The shallow AC3 test drives the shallow single-pass path. Only the
backend seam is mocked.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream import git_ops
from daydream.backends import (
    AgentEvent,
    DiagnosticEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.runner import RunConfig, run
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git
from tests.harness.remote_ci import NoCIRemote

# The prompt-dispatching stub backend and its install helpers are the canonical
# shared stub (tests/harness/stub_backend.py); re-rolling the dispatch
# heuristics would be fragile. tests/ is a namespace package, so the harness
# imports cleanly.
from tests.harness.stub_backend import (
    StubBackend,
    force_interactive,
    install_stub_backend,
    silence,
)
from tests.harness.trajectory import diff_adding
from tests.test_deep_orchestrator import _merge_item, _noop_commit, _ok


class _ArchiveCaptureBackend(StubBackend):
    async def execute(
        self,
        cwd: Any,
        prompt: str,
        output_schema: Any=None,
        continuation: Any=None,
        agents: Any=None,
        max_turns: Any=None,
        read_only: Any=False,
    ) -> AsyncIterator[AgentEvent]:
        if prompt.startswith("The daydream changes are already staged"):
            run_id = prompt.split("Daydream-Run: ", 1)[1].splitlines()[0]
            version = prompt.split("Daydream-Version: ", 1)[1].splitlines()[0]
            # The index is pre-staged by _do_commit (deterministic staging,
            # issue #543) — commit it as-is, never `git add --all`.
            git(
                cwd,
                "commit",
                "-m",
                (f"fix: apply daydream recommendation\n\nDaydream-Run: {run_id}\nDaydream-Version: {version}"),
            )
            git(cwd, "push", "-u", "archive", git(cwd, "branch", "--show-current"))
            yield TextEvent(text="Committed and pushed the recommendation.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


def _only_archived_run(archive_dir: Path) -> Path:
    """Return the single archived run directory, asserting there is exactly one."""
    run_dirs = list((archive_dir / "runs").iterdir())
    assert len(run_dirs) == 1, f"expected exactly one archived run, got {run_dirs}"
    return run_dirs[0]


def _deep_python_trajectory(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "trajectories").glob("deep-python*.json"))
    assert len(candidates) == 1
    assert re.fullmatch(r"deep-python(?:--[0-9a-f]{64})?\.json", candidates[0].name)
    return candidates[0]


def _install_deep_capture_backend(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    real_internal_phases: bool = False,
) -> Any:
    """Install the shared deep-run backend and optional focused phase seams."""
    silence(monkeypatch)
    force_interactive(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    if real_internal_phases:
        stub = _ArchiveCaptureBackend(multi_stack_target)
        monkeypatch.setattr(
            "daydream.runner.create_backend",
            lambda name, model=None, **kwargs: stub,
        )
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    if not real_internal_phases:
        monkeypatch.setattr(
            "daydream.deep.orchestrator.phase_test_and_heal",
            lambda *a, **k: _ok(**k),
        )
        monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)
    return stub


async def _ok_with_heal_edit(target: Path, **kwargs: Any) -> Any:
    from daydream.phases import TestAndHealResult, TestAttemptEvidence

    before = kwargs["capture_tree_key"]()
    (target / "heal_edit.py").write_text("def healed():\n    pass\n")
    after = kwargs["capture_tree_key"]()
    return TestAndHealResult(
        passed=True,
        retries=0,
        proceed=True,
        ignored=False,
        attempts=(TestAttemptEvidence(
            session_id=kwargs["session_id"], kind="agent", command=None,
            passed=True, input_tree_key=before, output_tree_key=after,
        ),),
    )


# --- AC1 + AC3: default deep run populates eval metrics AND captures recommended.patch ---


async def test_default_deep_run_populates_eval_captures_patch_and_current_merge_phase_state(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    no_ci_remote: NoCIRemote,
) -> None:
    """AC1 + AC3: a default deep run (no --no-eval) populates the manifest's eval
    metrics AND writes a recommended.patch distinct from diff.patch.

    The fix stage edits a TRACKED file (api.py), so the pre-fix → post-fix diff
    is non-empty and the real test/heal and commit phases run before archiving.
    """
    remote = bare_remote(archive_dir.parent / "origin.git")
    no_ci_remote.connect(multi_stack_target, remote)
    stub = _install_deep_capture_backend(
        multi_stack_target,
        monkeypatch,
        real_internal_phases=True,
    )
    stub.fix_edit_line = "# daydream recommended change\n"
    head_before = git_ops.head_sha(multi_stack_target)

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0
    head_after = git_ops.head_sha(multi_stack_target)
    assert head_after != head_before
    assert git(remote, "rev-parse", "refs/heads/feature") == head_after
    commit_message = git_ops.head_commit_message(multi_stack_target)
    assert "Daydream-Run:" in commit_message
    assert "Daydream-Version:" in commit_message

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    test_steps = [step for step in trajectory["steps"] if step.get("extra", {}).get("daydream_phase") == "test"]
    assert any("Run the project's test suite" in step["message"] for step in test_steps)
    assert any("2 passed, 0 failed" in step["message"] for step in test_steps)

    # AC1: eval ran by default -> all four metrics non-null.
    metrics = manifest["metrics"]
    assert metrics["grounding_rate"] is not None
    assert metrics["total_findings"] is not None
    assert metrics["coverage_ratio"] is not None
    assert metrics["cost_per_finding_usd"] is not None
    assert (run_dir / "evaluation.json").is_file()
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "succeeded"}
    assert manifest["pipeline_status"] == "succeeded"
    merge_events = [
        event
        for event in trajectory["extra"]["phase_events"]
        if event["phase"] == "merge"
    ]
    assert len(merge_events) == 2
    assert {event["session_id"] for event in merge_events} == {manifest["session_id"]}

    # AC3: recommended.patch archived and distinct from diff.patch.
    recommended = run_dir / "recommended.patch"
    diff = run_dir / "diff.patch"
    assert recommended.is_file()
    assert diff.is_file()
    recommended_text = recommended.read_text()
    diff_text = diff.read_text()
    assert recommended_text != diff_text
    # The recommended patch carries daydream's fix line; the review diff does not.
    assert "# daydream recommended change" in recommended_text
    assert "# daydream recommended change" not in diff_text


async def test_mixed_case_pr_identity_reaches_remote_ci_and_archives_success(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    no_ci_remote: NoCIRemote,
    fake_gh: FakeGh,
) -> None:
    """Operator/P04 casing normalizes, while the exact PR URL stays bound."""
    response_base = "Base-User/Project"
    response_head = "Fork-User/Project"
    response_url = f"https://github.com/{response_base}/pull/{no_ci_remote.pr_number}"
    original_serve_no_ci = no_ci_remote._serve_no_ci  # noqa: SLF001

    def serve_pr(*, branch: str, head_sha: str) -> None:
        fake_gh.set_response("repo-view", value=response_base)
        fake_gh.serve_pr_view(
            {
                "number": no_ci_remote.pr_number,
                "title": "Fixture PR",
                "body": "",
                "state": "OPEN",
                "headRefName": branch,
                "baseRefName": "main",
                "headRefOid": head_sha,
                "url": response_url,
                "headRepository": {"nameWithOwner": response_head},
                "headRepositoryOwner": {"login": "Fork-User"},
            }
        )

    def serve_no_ci(*, branch: str, head_sha: str) -> None:
        # Preserve the harness's normalized lowercase endpoint catalog, then
        # replace only the REST response identity returned at that endpoint.
        original_serve_no_ci(branch=branch, head_sha=head_sha)
        fake_gh.set_response(
            "GET",
            f"repos/{no_ci_remote.base_repository}/pulls/{no_ci_remote.pr_number}",
            {
                "number": no_ci_remote.pr_number,
                "html_url": response_url,
                "state": "open",
                "base": {"ref": "main", "repo": {"full_name": response_base}},
                "head": {
                    "ref": branch,
                    "sha": head_sha,
                    "repo": {"full_name": response_head},
                },
                "merge_commit_sha": None,
            },
        )

    monkeypatch.setattr(no_ci_remote, "_serve_pr", serve_pr)
    monkeypatch.setattr(no_ci_remote, "_serve_no_ci", serve_no_ci)
    remote = bare_remote(archive_dir.parent / "mixed-case-origin.git")
    no_ci_remote.connect(multi_stack_target, remote)
    stub = _install_deep_capture_backend(
        multi_stack_target,
        monkeypatch,
        real_internal_phases=True,
    )
    stub.fix_edit_line = "# daydream mixed-case identity\n"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            pr_number=no_ci_remote.pr_number,
            pr_repo="bAsE-uSeR/pRoJeCt",
        )
    )

    assert exit_code == 0
    lower_base = no_ci_remote.base_repository
    pull_endpoint = f"repos/{lower_base}/pulls/{no_ci_remote.pr_number}"
    assert fake_gh.calls("GET", pull_endpoint)
    verdict = json.loads(
        (multi_stack_target / ".daydream/deep/remote-ci-verdict.json").read_text()
    )
    push = json.loads(
        (multi_stack_target / ".daydream/deep/push-verdict.json").read_text()
    )
    assert verdict["status"] == "no_ci"
    assert push["pushed_repository"] == no_ci_remote.head_repository
    assert verdict["target"]["base_repository"] == lower_base
    assert verdict["target"]["head_repository"] == no_ci_remote.head_repository
    assert verdict["binding"]["base_repository"] == lower_base
    assert verdict["binding"]["head_repository"] == no_ci_remote.head_repository
    assert verdict["target"]["pr_url"] == response_url
    assert verdict["binding"]["pr_url"] == response_url

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["phase_states"]["push"]["status"] == "succeeded"
    assert manifest["phase_states"]["remote_ci"]["status"] == "succeeded"
    assert manifest["pipeline_status"] == "succeeded"
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    remote_ends = [
        event
        for event in trajectory["extra"]["phase_events"]
        if event["phase"] == "remote-ci" and event["event"] == "phase_end"
    ]
    assert len(remote_ends) == 1
    assert remote_ends[0]["status"] == "succeeded"
    assert "reason_code" not in remote_ends[0]
    assert remote_ends[0]["metadata"]["stop_reason"] == "no_ci"


async def test_deep_archive_recommended_patch_excludes_preexisting_untracked_files(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    no_ci_remote: NoCIRemote,
) -> None:
    """A pre-existing untracked file (present before the run) is absent from the
    archived recommended.patch while a fix-created untracked file is present."""
    remote = bare_remote(archive_dir.parent / "origin.git")
    no_ci_remote.connect(multi_stack_target, remote)
    stub = _install_deep_capture_backend(
        multi_stack_target, monkeypatch, real_internal_phases=True
    )
    stub.fix_edit_line = "# daydream recommended change\n"
    stub.fix_new_generated = "migrations/0002_add_x.sql"  # fix-created, untracked
    (multi_stack_target / "notes.txt").write_text("pre-existing\n")  # pre-fix, untracked

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    recommended = (run_dir / "recommended.patch").read_text()
    assert "migrations/0002_add_x.sql" in recommended  # fix-created file present
    assert "notes.txt" not in recommended              # pre-existing file excluded


async def test_deep_heal_edit_lands_in_archived_recommended_patch(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    remote = bare_remote(archive_dir.parent / "origin.git")
    git(multi_stack_target, "remote", "add", "origin", str(remote))
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)  # real_internal_phases=False
    stub.fix_edit_line = "# daydream recommended change\n"
    monkeypatch.setattr(
        "daydream.deep.orchestrator.phase_test_and_heal",
        lambda *a, **k: _ok_with_heal_edit(multi_stack_target, **k),
    )

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    assert "heal_edit.py" not in (run_dir / "recommended.patch").read_text()
    assert not (multi_stack_target / "heal_edit.py").exists()
    # Session-bound capture-point sidecar, mirrored from fix-quality-gate.json.
    sidecar = json.loads(
        (multi_stack_target / ".daydream" / "deep" / "recommended-capture.json").read_text()
    )
    assert sidecar["session_id"] == run_dir.name
    assert sidecar["capture_point"] == "post_test"
    assert sidecar["tree_key"] == sidecar["evidence_key"]["tree_key"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["recommended_patch_capture"] == "post_test"


async def test_deep_archive_commit_excludes_preexisting_untracked_files(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    no_ci_remote: NoCIRemote,
) -> None:
    """A pre-existing untracked file (before the run) is absent from the daydream
    commit's tree; a fix-created untracked file is present (issue #543)."""
    remote = bare_remote(archive_dir.parent / "origin.git")
    no_ci_remote.connect(multi_stack_target, remote)
    stub = _install_deep_capture_backend(
        multi_stack_target, monkeypatch, real_internal_phases=True
    )
    stub.fix_edit_line = "# daydream recommended change\n"
    stub.fix_new_generated = "migrations/0002_add_x.sql"  # fix-created, untracked
    (multi_stack_target / "notes.txt").write_text("pre-existing\n")  # pre-fix, untracked

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0

    # The pushed branch's HEAD tree excludes notes.txt but includes the fix-created
    # file (the stub pushes the current branch, so query that ref on the remote).
    branch = git(multi_stack_target, "branch", "--show-current")
    committed = git(remote, "ls-tree", "-r", "--name-only", branch).splitlines()
    assert "migrations/0002_add_x.sql" in committed
    assert "notes.txt" not in committed
    # notes.txt still untracked in the working tree.
    assert "notes.txt" in git(multi_stack_target, "status", "--porcelain")


async def test_deep_run_with_unbalanced_quote_shell_command_still_archives_evaluation(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """A shell command ``shlex`` cannot tokenize must not lose the archive's
    evaluation.json (issue #327).

    The offending call contributes no read paths while sibling calls in the
    same trajectory are still analyzed; the eval completes (archive never
    blocks) and the run keeps non-null manifest eval metrics.
    """
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)
    stub.fix_edit_line = "# daydream recommended change\n"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    session_id = run_dir.name

    # Inject an unbalanced-quote shell command into the SOURCE trajectory (the
    # tree analyze_session reads) and re-run the production archive eval seam
    # against it. Drop the stale evaluation.json from the clean run first so
    # the assertions below observe the eval of the INJECTED trajectory.
    source_traj = multi_stack_target / ".daydream" / "runs" / session_id / "trajectory.json"
    traj = json.loads(source_traj.read_text())
    traj["steps"].append(
        {
            "step_id": len(traj["steps"]) + 1,
            "extra": {"daydream_phase": "deep"},
            "tool_calls": [
                {
                    "function_name": "shell",
                    "arguments": {"command": "rg -l '\"unclosed"},
                },
                {"function_name": "shell", "arguments": {"command": "cat api.py"}},
            ],
        }
    )
    source_traj.write_text(json.dumps(traj))

    eval_path = run_dir / "evaluation.json"
    eval_path.unlink(missing_ok=True)

    from daydream.eval.analyzer import analyze_session
    from daydream.trajectory import (
        RunWriteSnapshot,
        TrajectoryDocumentSnapshot,
        snapshot_trajectories,
    )

    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at=str((traj.get("extra") or {}).get("run_ended_at", "")),
        root_trajectory_id=str(traj["trajectory_id"]),
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id=str(traj["trajectory_id"]),
                path=source_traj,
                json_bytes=json.dumps(traj).encode(),
            ),
        ),
    )
    # The same evaluation seam the strict archive finalizer drives, over the
    # injected trajectory bytes.
    result = analyze_session(
        multi_stack_target / ".daydream",
        session_id=session_id,
        frozen_trajectories=snapshot_trajectories(snapshot),
    )
    assert "error" not in result
    eval_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Coverage degrades gracefully for the offending call only: the clean
    # sibling call still contributes its read, so api.py stays covered.
    evaluation = json.loads(eval_path.read_text())
    assert evaluation["coverage"]["files_read_by_reviewers"] >= 1
    assert "api.py" not in evaluation["coverage"]["uncovered_files"]

    # The archive as a whole keeps non-null eval metrics.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = manifest["metrics"]
    assert metrics["grounding_rate"] is not None
    assert metrics["total_findings"] is not None
    assert metrics["coverage_ratio"] is not None
    assert metrics["cost_per_finding_usd"] is not None


async def test_dump_artifacts_copies_full_bundle_to_target_dir(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    tmp_path: Path,
) -> None:
    """``--dump-artifacts DIR`` copies the fully-assembled run bundle into DIR so CI
    can upload it — trajectory, deep artifacts, diffs, manifest, and evaluation all
    land in the user-specified directory, mirroring the archived run."""
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)
    stub.fix_edit_line = "# daydream recommended change\n"

    dump_dir = tmp_path / "uploaded-artifacts"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            dump_artifacts=str(dump_dir),
        )
    )
    assert exit_code == 0

    # The dump directory mirrors the archived run bundle.
    run_dir = _only_archived_run(archive_dir)
    assert (dump_dir / "manifest.json").is_file()
    assert (dump_dir / "trajectory.json").is_file()
    assert (dump_dir / "diff.patch").is_file()
    assert (dump_dir / "evaluation.json").is_file()
    assert (dump_dir / "manifest.json").read_text() == (run_dir / "manifest.json").read_text()


async def test_no_dump_artifacts_leaves_no_extra_copy(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    tmp_path: Path,
) -> None:
    """Without ``--dump-artifacts`` no bundle copy is made outside the archive."""
    _install_deep_capture_backend(multi_stack_target, monkeypatch)

    dump_dir = tmp_path / "uploaded-artifacts"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    )
    assert exit_code == 0
    assert not dump_dir.exists()


# --- #1170: the dump secret-scan gate is tiered (advisory reports, blocking refuses) ---


def _commit_scanned_file(target: Path, name: str, body: str) -> None:
    """Commit *body* on the feature branch so it lands verbatim in ``diff.patch``.

    ``_copy_run_artifacts`` carries ``diff.patch`` into the bundle with no
    redaction, so a committed file is the real route by which arbitrary source
    text reaches the egress scanner (#1170).
    """
    (target / name).write_text(body, encoding="utf-8")
    git(target, "add", name)
    git(target, "commit", "-m", f"add {name}")


async def test_dump_artifacts_publishes_bundle_with_advisory_scan_findings(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """#1170: name-shape scan hits report and publish; they never refuse the run.

    A settings module holding a dict-key constant and an f-string DSN template
    carries no credential, but trips ``_ENV_VAR_PATTERN`` and the userinfo rules.
    Before the tiering that refused the whole run after every LLM call had been
    paid for. Now the operator gets a value-free advisory and the complete
    bundle: review published, archive installed, dump copied, exit 0 (#981's
    "preserving the local run").
    """
    from daydream.archive.index import query_runs

    _install_deep_capture_backend(multi_stack_target, monkeypatch)
    _commit_scanned_file(
        multi_stack_target,
        "settings.py",
        'FEATURE_FLAG_OVERRIDE_KEY = "override_flag"\n'
        'DSN = f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"\n',
    )

    dump_dir = tmp_path / "uploaded-artifacts"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            dump_artifacts=str(dump_dir),
        )
    )
    assert exit_code == 0

    # Everything publishes: the dump, the archive index row, and the report.
    run_dir = _only_archived_run(archive_dir)
    assert (dump_dir / "manifest.json").is_file()
    assert query_runs(archive_dir)
    assert (multi_stack_target / ".review-output.md").is_file()

    # The advisory is reported and value-free (M11): path and category only.
    out = "".join(capfd.readouterr())
    assert "diff.patch" in out
    assert "env_var" in out
    assert "override_flag" not in out
    assert "DB_PASSWORD" not in out

    # An advisory bundle is still dumped byte-for-byte — never redact-then-dump.
    dumped = (dump_dir / "diff.patch").read_bytes()
    assert dumped == (run_dir / "diff.patch").read_bytes()
    assert b"FEATURE_FLAG_OVERRIDE_KEY" in dumped


async def _assert_target_is_reusable(target: Path) -> None:
    """A blocking dump refusal must not wedge the checkout for the next run.

    Without a rollback that can restore an absent dump destination, the refusal
    left a ``DETACHED`` transaction behind and every later run at this path —
    with or without ``--dump-artifacts`` — exited 1 during session open before
    any review work started (#1172). ``exit_code == 1`` and a missing dump
    directory hold either way, so this is what makes the refusal tests real
    guards rather than assertions that pass through the wedge.
    """
    exit_code = await run(
        RunConfig(target=str(target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0


async def test_dump_artifacts_refuses_token_canary_in_diff(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """#1170: a real token prefix in the diff still refuses every egress path.

    The blocking tier keeps PR #1161's disposition exactly: no dump, no archive
    row, exit 1 — and the console now names the file and rule that refused,
    without echoing the credential.
    """
    from daydream.archive.index import query_runs

    canary = "ghp_canaryfake123"
    _install_deep_capture_backend(multi_stack_target, monkeypatch)
    _commit_scanned_file(multi_stack_target, "creds.py", f'GITHUB_TOKEN = "{canary}"\n')

    dump_dir = tmp_path / "uploaded-artifacts"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            dump_artifacts=str(dump_dir),
        )
    )
    assert exit_code == 1
    assert not dump_dir.exists()
    assert query_runs(archive_dir) == []

    out = "".join(capfd.readouterr())
    assert canary not in out
    assert "diff.patch" in out
    assert "api_key" in out
    # #1171: the scan refusal is the message, not the rollback's own failure.
    assert "dump destination projection is malformed" not in out

    await _assert_target_is_reusable(multi_stack_target)


async def test_dump_artifacts_refuses_multiline_pem_in_diff(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """#1170 D4: multi-line key armor in ``diff.patch`` blocks the dump.

    ``diff.patch`` is scanned line by line, and ``_PEM_KEY_PATTERN`` spans
    BEGIN..END, so real key material in a patch was invisible to the gate while
    the same key inside a JSON string leaf was caught. Separate from the token
    canary on purpose: a bundle carrying both blocks on the canary alone, so a
    combined test passes with the multi-line pass unimplemented.
    """
    from daydream.archive.index import query_runs

    canary = "MIIFAKEKEYMATERIALFORTESTSONLY"
    _install_deep_capture_backend(multi_stack_target, monkeypatch)
    _commit_scanned_file(
        multi_stack_target,
        "deploy_key.pem",
        "-----BEGIN PRIVATE KEY-----\n" f"{canary}\n{canary}\n" "-----END PRIVATE KEY-----\n",
    )

    dump_dir = tmp_path / "uploaded-artifacts"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            dump_artifacts=str(dump_dir),
        )
    )
    assert exit_code == 1
    assert not dump_dir.exists()
    assert query_runs(archive_dir) == []

    out = "".join(capfd.readouterr())
    assert canary not in out
    assert "diff.patch" in out
    assert "pem_key" in out
    assert "dump destination projection is malformed" not in out

    await _assert_target_is_reusable(multi_stack_target)


async def test_dump_refusal_reports_the_archive_error_when_the_rollback_also_fails(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """#1171: a failing rollback must not be the only thing the operator sees.

    The pending ``ArchiveFinalizationError`` reaches the re-raised storage error
    only through ``add_note``, which carries its type name and which nothing
    renders — so the gate that refused and the file that tripped it were
    invisible whenever the rollback itself failed. With an absent dump
    destination now restorable this branch is no longer on the ordinary refusal
    path, so the rollback is forced to fail to keep the report proven.
    """
    from daydream import artifact_visibility

    canary = "ghp_canaryfake123"
    _install_deep_capture_backend(multi_stack_target, monkeypatch)
    _commit_scanned_file(multi_stack_target, "creds.py", f'GITHUB_TOKEN = "{canary}"\n')

    def refuse_restore(_session: object) -> None:
        raise artifact_visibility.ArtifactVisibilityError("synthetic rollback failure")

    monkeypatch.setattr(artifact_visibility.ArtifactSession, "_restore_prior", refuse_restore)

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            dump_artifacts=str(tmp_path / "uploaded-artifacts"),
        )
    )
    assert exit_code == 1

    out = "".join(capfd.readouterr())
    assert "Artifact Finalization" in out
    assert "api_key" in out
    assert "diff.patch" in out
    assert "synthetic rollback failure" in out
    assert canary not in out


async def test_no_eval_leaves_manifest_eval_fields_null(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """AC1b: --no-eval (run_eval=False) skips the eval pass, leaving its metrics null."""
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)
    stub.fix_edit_line = "# daydream recommended change\n"

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
            run_eval=False,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = manifest["metrics"]
    assert metrics["grounding_rate"] is None
    assert metrics["total_findings"] is None
    assert metrics["coverage_ratio"] is None
    assert metrics["cost_per_finding_usd"] is None
    assert not (run_dir / "evaluation.json").exists()


# --- AC3 (shallow path): recommended.patch captured through the shallow runner ---


class _FixEditingBackend:
    """Shallow-dispatch backend whose fix stage edits a tracked file.

    Mirrors ``PhaseDispatchBackend`` dispatch but writes a real change to
    ``main.py`` on the fix turn so the shallow runner's recommended-patch capture
    has a non-empty diff to record.
    """

    model = "mock-model"

    def __init__(self, repo: Path) -> None:
        self._repo = repo

    async def execute(
        self,
        cwd: Any,
        prompt: str,
        output_schema: Any=None,
        continuation: Any=None,
        agents: Any=None,
        max_turns: Any=None,
        read_only: Any=False,
    ) -> AsyncIterator[AgentEvent]:
        from daydream.backends import ResultEvent, TextEvent

        pl = prompt.lower()
        # Native review prompts are skill-free (#886): dispatch on distinctive
        # judgment-prose markers instead of a ``beagle-*`` invocation token.
        if "review" in pl and (
            "inclusion obligation" in pl
            or "full change spans" in pl
            or "language-agnostic review practices" in pl
            or "assigned to this stack" in pl
            or "repository-wide interactions" in pl
        ):
            yield TextEvent(text="Review complete.")
            # Issue #745: the per-stack reviewer emits PER_STACK_RECORD_SCHEMA
            # structured output directly (no separate parse step).
            yield ResultEvent(
                structured_output={
                    "issues": [
                        {
                            "id": 1,
                            "description": "Add a guard",
                            "file": "main.py",
                            "line": 1,
                            "severity": "medium",
                            "confidence": "HIGH",
                            "rationale": "guard missing",
                            "evidence": "main.py:1",
                        }
                    ],
                    "verdicts": [],
                },
                continuation=None,
            )
        elif "fix this issue" in pl or pl.startswith("fix these"):
            main_py = self._repo / "main.py"
            main_py.write_text(main_py.read_text() + "# daydream recommended change\n")
            yield TextEvent(text="Fixed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "post-fix fix-verifier agent" in pl:
            ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "verdicts": [
                        {
                            "issue_id": issue_id,
                            "verdict": "resolved",
                            "reason": "complete",
                        }
                        for issue_id in ids
                    ]
                },
                continuation=None,
            )
        elif "test suite" in pl or "run the project" in pl:
            yield TextEvent(text="All 1 tests passed. 0 failed.")
            yield ResultEvent(structured_output=None, continuation=None)
        else:
            yield TextEvent(text="OK")
            yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass


async def test_shallow_run_captures_recommended_patch(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    no_ci_remote: NoCIRemote,
) -> None:
    """AC3 (shallow): the shallow single-pass fix path archives a recommended.patch
    carrying daydream's edit.

    Shallow mode is the deep flow with a forced single stack (#330), so it
    persists ``diff.patch`` like any deep run; recommended.patch must still be
    the distinct artifact carrying daydream's own edit (the deep test asserts
    the same distinctness where both artifacts exist).
    """
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "n")
    # Host-native commit/push (issue #726): the shallow --yes run commits and
    # pushes to 'origin' for real, so give the repo a bare remote.
    remote = bare_remote(archive_dir.parent / "origin.git")
    no_ci_remote.connect(feature_branch_repo, remote)
    backend = _FixEditingBackend(feature_branch_repo)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    exit_code = await run(
        RunConfig(
            target=str(feature_branch_repo),
            stack="python",
            quiet=True,
            cleanup=False,
            shallow=True,
            assume="yes",
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    recommended = run_dir / "recommended.patch"
    diff = run_dir / "diff.patch"
    assert recommended.is_file()
    assert diff.is_file()
    recommended_text = recommended.read_text()
    diff_text = diff.read_text()
    # recommended.patch is daydream's proposed diff, distinct from the
    # PR-under-review diff.patch — both carry the reviewed change, but only the
    # recommended patch carries the fix daydream applied.
    assert recommended_text != diff_text
    assert "+# daydream recommended change" in recommended_text
    assert "+# daydream recommended change" not in diff_text


# --- git_ops.capture_recommended_patch (the shared helper) ---


def _init_repo_with_commit(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_ops._run_git(repo, ["init", "-b", "main"], timeout=10)
    git_ops._run_git(repo, ["config", "user.email", "t@t.com"], timeout=10)
    git_ops._run_git(repo, ["config", "user.name", "T"], timeout=10)
    (repo / "a.py").write_text("x = 1\n")
    git_ops._run_git(repo, ["add", "."], timeout=10)
    git_ops._run_git(repo, ["commit", "-m", "init"], timeout=10)
    return repo


def test_capture_recommended_patch_clean_tree_uses_head_base(tmp_path: Path) -> None:
    """On a clean tree (stash_create is None) the pre-fix HEAD is the base: the
    post-fix worktree diff against it is captured."""
    repo = _init_repo_with_commit(tmp_path)
    base = git_ops.head_sha(repo)  # captured before the "fix"
    (repo / "a.py").write_text("x = 1\ny = 2\n")  # the fix

    out = repo / ".daydream" / "recommended.patch"
    wrote = git_ops.capture_recommended_patch(repo, base, out)

    assert wrote is True
    assert out.is_file()
    assert "+y = 2" in out.read_text()


def test_capture_recommended_patch_excludes_only_preexisting_untracked_files(
    tmp_path: Path,
) -> None:
    """R1-R4: a pre-existing untracked file (in preexisting_untracked) contributes
    no creation hunk; a fix-created untracked file still does; tracked edits
    serialize as today; omitting the snapshot keeps pre-existing files."""
    repo = _init_repo_with_commit(tmp_path)
    base = git_ops.head_sha(repo)                       # captured before the "fix"
    (repo / "a.py").write_text("x = 1\ny = 2\n")        # tracked fix edit
    (repo / "notes.txt").write_text("pre-existing\n")   # in the snapshot (pre-fix)
    (repo / "new.py").write_text("fix = 1\n")           # created during the fix

    out = repo / ".daydream" / "recommended.patch"
    wrote = git_ops.capture_recommended_patch(
        repo, base, out, preexisting_untracked={"notes.txt"}
    )
    assert wrote is True
    text = out.read_text()
    assert "new.py" in text            # fix-created untracked file captured
    assert "notes.txt" not in text     # pre-existing untracked file excluded
    assert "+y = 2" in text            # tracked diff unaffected

    # R4 backward-compat: no snapshot -> the pre-existing file IS captured.
    out2 = repo / ".daydream" / "recommended2.patch"
    git_ops.capture_recommended_patch(repo, base, out2)
    assert "notes.txt" in out2.read_text()


def test_capture_recommended_patch_none_base_writes_nothing(tmp_path: Path) -> None:
    """A None base (no pre-fix snapshot could be taken) is a no-op."""
    repo = _init_repo_with_commit(tmp_path)
    out = repo / ".daydream" / "recommended.patch"
    assert git_ops.capture_recommended_patch(repo, None, out) is False
    assert not out.exists()


def test_capture_recommended_patch_no_change_writes_empty_marker(
    tmp_path: Path,
) -> None:
    """When nothing changed (no fix landed) an EMPTY recommended.patch marker is
    written so the run is distinguishable from a legacy archive (which has no
    recommended.patch at all). This prevents _read_recommended_patch from
    falling back to diff.patch (the PR-under-review diff) and mislabeling a
    no-recommendation run as 'applied'. Returns False (no non-empty patch)."""
    repo = _init_repo_with_commit(tmp_path)
    base = git_ops.head_sha(repo)
    out = repo / ".daydream" / "recommended.patch"
    assert git_ops.capture_recommended_patch(repo, base, out) is False
    assert out.is_file()
    assert out.read_text() == ""


# --- AC4: applied-signal cascades read recommended.patch (fallback to diff.patch) ---


def test_fix_applied_signal_prefers_recommended_patch(tmp_path: Path) -> None:
    """AC4: with both patches present, the signal parses recommended.patch hunks,
    not diff.patch hunks — a run whose RECOMMENDATION landed labels 'applied' even
    though the reviewed line is absent post-window."""
    from daydream.training.labeler_signals import fix_applied_signal

    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(diff_adding("recommended = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    # Post-window state carries the RECOMMENDED line but NOT the reviewed line.
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "existing\nrecommended = 1\n",
    )
    assert sig.verdict == "applied"
    assert sig.hunks_applied == 1
    assert sig.hunks_total == 1


def test_fix_applied_signal_falls_back_to_diff_patch(tmp_path: Path) -> None:
    """AC4 backward compat: an old archive with only diff.patch still labels via
    the diff.patch hunks."""
    from daydream.training.labeler_signals import fix_applied_signal

    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "existing\nreviewed = 2\n",
    )
    assert sig.verdict == "applied"
    assert sig.hunks_total == 1


def test_fix_applied_signal_new_archive_no_recommendation_skips_fallback(
    tmp_path: Path,
) -> None:
    """A new-format archive (manifest ``recommended_patch_supported=True``) with
    no ``recommended.patch`` made NO recommendation (review-only / all-declined /
    wash). The cascade must score zero hunks and NOT fall back to ``diff.patch``
    (the PR-under-review diff), even when diff.patch's line is present
    post-window — otherwise such runs are mislabeled 'applied'."""
    from daydream.training.labeler_signals import fix_applied_signal

    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "recommended_patch_supported": True})
    )
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    # Post-window carries the REVIEWED line; diff.patch would match if the
    # (forbidden) fallback fired.
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "existing\nreviewed = 2\n",
    )
    assert sig.hunks_total == 0
    assert sig.verdict == "not_applied"


@pytest.mark.parametrize(
    ("file_contents", "expected_verdict"),
    [
        pytest.param("existing\nrecommended = 1\n", "applied", id="recommended-line-present"),
        pytest.param("existing\nreviewed = 2\n", "rejected", id="recommended-line-absent"),
    ],
)
def test_local_commit_applied_signal_uses_recommended_patch(
    tmp_path: Path,
    file_contents: str,
    expected_verdict: str,
) -> None:
    from daydream.training.labeler_signals import local_commit_applied_signal

    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(diff_adding("recommended = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "branch": "feature",
        "archive_path": str(tmp_path),
    }
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since: ["c1"],
        file_at_fetcher=lambda repo, path, sha: file_contents,
    )
    assert sig.verdict == expected_verdict


# --- location + shipped-duplication axes reach the archive (issue #1106) ---


async def test_deep_run_archives_location_and_shipped_duplication_axes(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """Real-path: the two new eval axes land in the archived ``evaluation.json``.

    Drives the production entrypoint over a real temp git worktree with only the
    backend seam mocked. The run ships two near-duplicate findings on ``api.py``
    (whose only hunk is ``(1, 2)``): one correctly anchored on line 1 and one
    anchored on line 88, which the live location validator demotes and stamps
    with ``location_cited_line``. The harness's structural meta-stack item is
    the third shipped item (in-hunk, differently worded). The archived
    evaluation must therefore show a ``beyond_tolerance`` item scored on the
    CITED line and a shipped near-duplicate pair -- the exact run shape that
    previously scored perfect.
    """
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)
    stub.fix_edit_line = "# daydream recommended change\n"
    anchored = _merge_item(
        1, "api.py", "high", desc="The loader does not validate its config path"
    )
    mis_anchored = {
        **_merge_item(
            2, "api.py", "medium", desc="The loader fails to validate the config path"
        ),
        "line": 88,
        "evidence": "api.py:88",
    }
    stub.merge_items = [anchored, mis_anchored]

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    evaluation = json.loads((run_dir / "evaluation.json").read_text())

    # The persisted hunk index (written next to diff.patch) supplied the ranges.
    location = evaluation["location"]
    assert location["hunk_source"] == "hunk-index.json"
    assert location["shipped_items"] == 3
    assert location["scored_items"] == 3
    assert location["tiers"] == {
        "in_hunk": 2,           # the anchored finding + the structural item
        "within_tolerance": 0,
        "beyond_tolerance": 1,  # the mis-anchored twin
        "file_absent": 0,
    }
    assert location["in_hunk_rate"] == 0.6667
    # The live validator demoted the mis-anchored item and recorded its citation.
    assert location["distrusted_items"] == 1
    assert location["relocated_items"] == 0   # demoted, not relocated (no snap)
    beyond = [row for row in location["items"] if row["tier"] == "beyond_tolerance"]
    assert len(beyond) == 1
    assert beyond[0]["cited_line"] == 88
    assert beyond[0]["location_distrust"] is True

    # The shipped duplication escaped merge and is counted as such.
    duplication = evaluation["findings"]["shipped_duplication"]
    assert duplication["shipped_items"] == 3
    assert duplication["comparable_pairs"] == 3
    assert duplication["near_duplicate_pairs"] == 1   # the escape
    assert duplication["same_file_pairs"] == 3        # every item cites api.py
    assert duplication["same_file_near_duplicate_pairs"] == 1
    assert duplication["max_similarity"] >= 0.5
    escape = duplication["pairs"][0]
    assert (escape["a_id"], escape["b_id"]) == ("1", "2")
    assert escape["same_file"] is True

    # Grounding records which artifact it checked against.
    assert evaluation["grounding"]["hunk_source"] == "hunk-index.json"

    # Pre-existing manifest eval metrics are unaffected.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["metrics"]["grounding_rate"] is not None
    assert manifest["metrics"]["total_findings"] == 3


class _CodexEvidenceBackend(StubBackend):
    """Emit one isolated standard-event evidence stream on the Python child."""

    def __init__(self, target: Path, *, evidence: bool) -> None:
        super().__init__(target)
        self.evidence = evidence

    async def execute(
        self,
        cwd: Any,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        if "you are reviewing the python stack" in prompt.lower():
            if self.evidence:
                for index in range(311):
                    call_id = f"shell-{index}"
                    yield ToolStartEvent(
                        id=call_id,
                        name="shell",
                        input={"command": f"printf 'shell {index}\\n'"},
                    )
                    if index == 1:
                        continue
                    yield ToolResultEvent(
                        id=call_id,
                        output="failed" if index == 0 else "ok",
                        is_error=index == 0,
                    )
                for index in range(15):
                    call_id = f"patch-{index}"
                    yield ToolStartEvent(
                        id=call_id,
                        name="patch",
                        input={"patch": f"*** patch {index} ***"},
                    )
                    yield ToolResultEvent(id=call_id, output="applied", is_error=False)
                yield ToolResultEvent(id="unmatched-result", output="orphan", is_error=True)
                yield DiagnosticEvent(
                    code="codex_transport_coverage",
                    message="current public stream has incomplete tool coverage",
                    metadata={"occurrences": 1},
                )
                yield DiagnosticEvent(
                    code="codex_parser_coverage",
                    message="bounded parser gap evidence",
                    metadata={"unknown_items": 1},
                )
            else:
                yield ToolStartEvent(
                    id="clean-read",
                    name="read",
                    input={"path": "api.py"},
                )
                yield ToolResultEvent(
                    id="clean-read",
                    output="file content",
                    is_error=False,
                )
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


def _install_codex_evidence_backend(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evidence: bool,
) -> _CodexEvidenceBackend:
    """Install the evidence backend while keeping unrelated phases tool-free."""
    _install_deep_capture_backend(target, monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.console", Console(file=StringIO(), force_terminal=False)
    )
    backend = _CodexEvidenceBackend(target, evidence=evidence)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )
    return backend


async def test_codex_evidence_integrity_archives_semantic_counts_and_review_flags(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """runner.run -> archive -> evaluation preserves all unsafe evidence."""
    _install_codex_evidence_backend(
        multi_stack_target,
        monkeypatch,
        evidence=True,
    )

    assert await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    ) == 0

    run_dir = _only_archived_run(archive_dir)
    child = json.loads(_deep_python_trajectory(run_dir).read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    child_calls = [
        call
        for step in child["steps"]
        for call in step.get("tool_calls") or []
    ]
    assert len(child_calls) == 326
    assert child_calls[0]["arguments"]["command"] == "printf 'shell 0\\n'"
    assert sum(evaluation["tools"]["by_agent"]["deep-python"].values()) == 326
    assert evaluation["tools"]["total_calls"] == 326
    assert evaluation["tools"]["by_type"] == {"shell": 311, "patch": 15}
    assert evaluation["tools"]["write_ratio"] == 0.046

    agent_steps = [step for step in child["steps"] if step["source"] == "agent"]
    result_extras = [
        result.get("extra", {})
        for step in agent_steps
        for result in (step.get("observation") or {}).get("results", [])
    ]
    assert any(extra.get("is_error") is True for extra in result_extras)
    assert any(extra.get("status") == "interrupted" for extra in result_extras)
    assert any(
        "unmatched-result" in step.get("extra", {}).get("unmatched_tool_results", [])
        for step in agent_steps
    )
    diagnostics = [
        diagnostic
        for step in agent_steps
        for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
    ]
    assert [diagnostic["code"] for diagnostic in diagnostics] == [
        "codex_transport_coverage",
        "codex_parser_coverage",
    ]
    training = next(
        row
        for row in evaluation["training_signals"]["trajectories"]
        if row["trajectory"] == "deep-python"
    )
    assert training["training_quality"] == "review"
    assert training["noise_flags"][:5] == [
        "failed_tool_result",
        "incomplete_tool_call",
        "unmatched_tool_result",
        "incomplete_telemetry",
        "parser_coverage_gap",
    ]


async def test_codex_evidence_integrity_clean_archive_stays_clean(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """An independent paired-success run acquires no telemetry review flag."""
    _install_codex_evidence_backend(
        multi_stack_target,
        monkeypatch,
        evidence=False,
    )

    assert await run(
        RunConfig(
            target=str(multi_stack_target),
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    ) == 0

    run_dir = _only_archived_run(archive_dir)
    child = json.loads(_deep_python_trajectory(run_dir).read_text(encoding="utf-8"))
    assert all(not step.get("extra", {}).get("backend_diagnostics") for step in child["steps"])
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    training = next(
        row
        for row in evaluation["training_signals"]["trajectories"]
        if row["trajectory"] == "deep-python"
    )
    assert evaluation["tools"]["total_calls"] == 1
    assert evaluation["tools"]["by_type"] == {"read": 1}
    assert training["noise_flags"] == []
    assert training["training_quality"] == "clean"


async def test_malformed_codex_tool_name_survives_real_log_mode_runner_archive(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """Replay CLI drift through the real parser, log-mode runner, and archive."""
    from unittest.mock import patch

    from daydream.backends.codex import CodexBackend
    from tests.harness.codex_replay import make_mock_process

    class MalformedToolBackend(StubBackend):
        async def execute(
            self,
            cwd: Any,
            prompt: str,
            output_schema: Any = None,
            continuation: Any = None,
            agents: Any = None,
            max_turns: Any = None,
            read_only: bool = False,
        ) -> AsyncIterator[AgentEvent]:
            if "you are reviewing the python stack" in prompt.lower():
                item = {
                    "id": "malformed-mcp", "type": "mcp_tool_call",
                    "tool": {"unexpected": "name-shape"}, "arguments": {"path": "api.py"},
                }
                process = make_mock_process(
                    [
                        json.dumps({"type": "item.started", "item": item}),
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {**item, "result": {"content": []}},
                            }
                        ),
                        json.dumps({"type": "turn.completed", "usage": {}}),
                    ]
                )
                with patch(
                    "daydream.backends._transport.asyncio.create_subprocess_exec",
                    return_value=process,
                ):
                    async for event in CodexBackend(model="fixture-model").execute(cwd, prompt):
                        if isinstance(event, (ToolStartEvent, ToolResultEvent, DiagnosticEvent)):
                            yield event
            async for event in super().execute(
                cwd, prompt, output_schema=output_schema, continuation=continuation,
                agents=agents, max_turns=max_turns, read_only=read_only,
            ):
                yield event

    _install_deep_capture_backend(multi_stack_target, monkeypatch)
    backend = MalformedToolBackend(multi_stack_target)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    assert await run(RunConfig(
        target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False,
        log_mode=True,
    )) == 0

    child = json.loads(_deep_python_trajectory(_only_archived_run(archive_dir)).read_text())
    calls = [call for step in child["steps"] for call in (step.get("tool_calls") or [])]
    assert [(call["tool_call_id"], call["function_name"]) for call in calls] == [("malformed-mcp", "unknown")]
    diagnostics = [
        diagnostic for step in child["steps"]
        for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
    ]
    assert diagnostics[0]["metadata"]["warnings"]["reasons"] == {"tool_not_string": 1}


class _JoinedArtifactEvidenceBackend(StubBackend):
    """Exercise sanctioned artifact reads at the external backend boundary."""

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.private_intent_paths: list[Path] = []
        self.private_intent_payloads: list[bytes] = []
        self.python_sanctioned_entries: list[tuple[tuple[str, Path], ...]] = []
        self.python_prompts: list[str] = []
        self.python_entry_visibility: list[tuple[bool, bool]] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        del persist_session  # StubBackend has no resumable external session.
        lowered = prompt.lower()
        python_request = "you are reviewing the python stack" in lowered
        generic_request = "you are reviewing the generic-fallback stack" in lowered
        artifact_reference: str | None = None

        if python_request:
            header = "Sanctioned phase inputs (read only these exact files):"
            prompt_lines = prompt.splitlines()
            assert prompt_lines.count(header) == 1
            start = prompt_lines.index(header) + 1
            rendered_entries: list[tuple[str, Path]] = []
            for line in prompt_lines[start:]:
                assert line.startswith("- ")
                label, raw_path = line.removeprefix("- ").split(": ", 1)
                rendered_entries.append((label, Path(raw_path)))

            intent_entries = [
                path for label, path in rendered_entries if label == "intent"
            ]
            assert len(intent_entries) == 1
            intent_path = intent_entries[0]
            assert str(intent_path).endswith("/deep/intent.md")
            assert intent_path.is_absolute()
            assert intent_path.is_file()
            assert not intent_path.is_symlink()
            assert not intent_path.resolve().is_relative_to(cwd.resolve())

            cwd_visibility = (
                (cwd / ".daydream").exists(),
                (cwd / ".review-output.md").exists(),
            )
            assert cwd_visibility == (False, False)
            self.python_entry_visibility.append(cwd_visibility)
            self.python_prompts.append(prompt)
            self.python_sanctioned_entries.append(tuple(rendered_entries))
            self.private_intent_paths.append(intent_path)
            self.private_intent_payloads.append(intent_path.read_bytes())
            artifact_reference = str(intent_path)
        elif generic_request:
            artifact_reference = ".daydream/deep/intent.md"

        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            if isinstance(event, ResultEvent) and artifact_reference is not None:
                call_id = (
                    "joined-private-intent"
                    if python_request
                    else "joined-relative-intent"
                )
                yield ToolStartEvent(
                    id=call_id,
                    name="Read",
                    input={"file_path": artifact_reference},
                )
                yield ToolResultEvent(
                    id=call_id,
                    output="artifact evidence",
                    is_error=False,
                )

                payload = event.structured_output
                assert isinstance(payload, dict)
                issues = payload.get("issues")
                assert isinstance(issues, list)
                assert len(issues) == 1
                assert isinstance(issues[0], dict)
                issue = {
                    **issues[0],
                    "rationale": f"Evidence: {artifact_reference}",
                }
                yield replace(
                    event,
                    structured_output={**payload, "issues": [issue]},
                )
                continue
            yield event


async def test_real_deep_archive_rejects_sanctioned_artifact_reads_but_credits_source(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    artifact_runtime_root: Path,
) -> None:
    """Real deep run keeps source credit while rejecting artifact evidence."""
    silence(monkeypatch)
    force_interactive(monkeypatch)
    backend = _JoinedArtifactEvidenceBackend(multi_stack_target)
    backend.per_stack_emit_reads = True
    backend.parse_by_stack = {
        "python": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "api.py",
            "line": 1,
            "description": "Python private-artifact rationale control",
        },
        "react": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "App.tsx",
            "line": 1,
            "description": "React source-only rationale control",
        },
        "generic": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "README.md",
            "line": 1,
            "description": "Generic relative-artifact rationale control",
        },
    }
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )

    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            output_mode="review",
            assume="no",
            non_interactive=True,
            cleanup=False,
            archive=True,
            run_eval=True,
            shallow_fanout_threshold=0,
        )
    )
    assert exit_code == 0

    assert len(backend.private_intent_paths) == 1
    private_intent = backend.private_intent_paths[0]
    assert backend.private_intent_payloads == [private_intent.read_bytes()]
    assert backend.private_intent_payloads[0]
    assert backend.python_entry_visibility == [(False, False)]
    assert len(backend.python_prompts) == 1
    assert len(backend.python_sanctioned_entries) == 1

    sanctioned_entries = backend.python_sanctioned_entries[0]
    assert len({label for label, _path in sanctioned_entries}) == len(
        sanctioned_entries
    )
    assert all(path.is_absolute() and path.is_file() for _label, path in sanctioned_entries)
    sanctioned_paths = {path for _label, path in sanctioned_entries}
    assert private_intent in sanctioned_paths
    private_relative = private_intent.relative_to(artifact_runtime_root)
    assert private_relative.parts[1] == "runs"
    assert private_relative.parts[3:] == (
        "live",
        ".daydream",
        "deep",
        "intent.md",
    )
    assert not private_intent.resolve().is_relative_to(multi_stack_target.resolve())
    live_dir = private_intent.parents[2]
    parent_artifact_dir = private_intent.parents[1]
    assert artifact_runtime_root not in sanctioned_paths
    assert live_dir not in sanctioned_paths
    assert parent_artifact_dir not in sanctioned_paths
    assert private_intent.parent not in sanctioned_paths

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (run_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert manifest["session_id"] == private_relative.parts[2]
    assert evaluation["daydream_dir"] == str(multi_stack_target / ".daydream")

    python_trajectory = json.loads(
        _deep_python_trajectory(run_dir).read_text(encoding="utf-8")
    )
    python_completed_ids = {
        result["source_call_id"]
        for step in python_trajectory["steps"]
        for result in (step.get("observation") or {}).get("results", [])
        if result.get("extra", {}).get("is_error") is False
    }
    python_reads = {
        call["arguments"].get("file_path") or call["arguments"].get("path")
        for step in python_trajectory["steps"]
        for call in step.get("tool_calls") or []
        if call["tool_call_id"] in python_completed_ids
        and call["function_name"].casefold() == "read"
    }
    assert "api.py" in python_reads
    assert str(private_intent) in python_reads

    generic_candidates = sorted(
        (run_dir / "trajectories").glob("deep-generic*.json")
    )
    assert len(generic_candidates) == 1
    assert re.fullmatch(
        r"deep-generic(?:--[0-9a-f]{64})?\.json",
        generic_candidates[0].name,
    )
    generic_trajectory = json.loads(
        generic_candidates[0].read_text(encoding="utf-8")
    )
    generic_completed_ids = {
        result["source_call_id"]
        for step in generic_trajectory["steps"]
        for result in (step.get("observation") or {}).get("results", [])
        if result.get("extra", {}).get("is_error") is False
    }
    generic_reads = {
        call["arguments"].get("file_path") or call["arguments"].get("path")
        for step in generic_trajectory["steps"]
        for call in step.get("tool_calls") or []
        if call["tool_call_id"] in generic_completed_ids
        and call["function_name"].casefold() == "read"
    }
    assert "README.md" in generic_reads
    assert ".daydream/deep/intent.md" in generic_reads

    controlled_records: list[dict[str, Any]] = []
    expected_records = {
        "python": (
            "api.py",
            "Python private-artifact rationale control",
            str(private_intent),
        ),
        "react": (
            "App.tsx",
            "React source-only rationale control",
            "stub",
        ),
        "generic": (
            "README.md",
            "Generic relative-artifact rationale control",
            ".daydream/deep/intent.md",
        ),
    }
    for stack, (file, description, rationale) in expected_records.items():
        records_payload = json.loads(
            (run_dir / "deep" / f"stack-{stack}-records.json").read_text(
                encoding="utf-8"
            )
        )
        records = (
            records_payload["issues"]
            if isinstance(records_payload, dict)
            else records_payload
        )
        matches = [
            record
            for record in records
            if record.get("file") == file
            and record.get("description") == description
        ]
        assert len(matches) == 1
        assert matches[0]["line"] == 1
        assert matches[0]["evidence"] == f"{file}:1"
        assert rationale in matches[0]["rationale"]
        controlled_records.append(matches[0])
    assert len({record["description"] for record in controlled_records}) == 3

    coverage = evaluation["coverage"]
    assert coverage["coverage_ratio"] == 1.0
    assert coverage["files_read_by_reviewers"] == coverage["files_in_diff"]
    assert coverage["artifact_reads_rejected"] == 2

    grounding = evaluation["grounding"]
    assert grounding["artifact_evidence_rejections"] == 2
    python_rows = [
        row
        for row in grounding["ungrounded"]
        if row["stack"] == "python" and row["file"] == "api.py"
    ]
    generic_rows = [
        row
        for row in grounding["ungrounded"]
        if row["stack"] == "generic" and row["file"] == "README.md"
    ]
    react_rows = [
        row
        for row in grounding["grounded"]
        if row["stack"] == "react" and row["file"] == "App.tsx"
    ]
    assert len(python_rows) == len(generic_rows) == len(react_rows) == 1
    python_row = python_rows[0]
    generic_row = generic_rows[0]
    react_row = react_rows[0]

    assert python_row["file_was_read"] is True
    assert python_row["line_grounded"] is True
    assert python_row["artifact_file_ref"] is None
    assert python_row["artifact_rationale_refs"] == [str(private_intent)]
    assert python_row["unread_rationale_refs"] == []
    assert python_row["grounded"] is False

    assert generic_row["file_was_read"] is True
    assert generic_row["line_grounded"] is True
    assert generic_row["artifact_file_ref"] is None
    assert generic_row["artifact_rationale_refs"] == [
        ".daydream/deep/intent.md"
    ]
    assert generic_row["unread_rationale_refs"] == []
    assert generic_row["grounded"] is False

    assert react_row["file_was_read"] is True
    assert react_row["line_grounded"] is True
    assert react_row["artifact_rationale_refs"] == []
    assert react_row["unread_rationale_refs"] == []
    assert react_row["grounded"] is True

    assert manifest["metrics"]["grounding_rate"] is not None
    assert (
        manifest["metrics"]["grounding_rate"]
        == evaluation["grounding"]["grounding_rate"]
    )

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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.runner import RunConfig, run
from tests.harness.git_helpers import bare_remote, git

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
            lambda *a, **k: _ok(),
        )
        monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)
    return stub


async def _ok_with_heal_edit(target: Path) -> Any:
    (target / "heal_edit.py").write_text("def healed():\n    pass\n")
    return await _ok()


# --- AC1 + AC3: default deep run populates eval metrics AND captures recommended.patch ---


async def test_default_deep_run_populates_eval_and_captures_recommended_patch(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """AC1 + AC3: a default deep run (no --no-eval) populates the manifest's eval
    metrics AND writes a recommended.patch distinct from diff.patch.

    The fix stage edits a TRACKED file (api.py), so the pre-fix → post-fix diff
    is non-empty and the real test/heal and commit phases run before archiving.
    """
    remote = bare_remote(archive_dir.parent / "origin.git")
    git(multi_stack_target, "remote", "add", "archive", str(remote))
    stub = _install_deep_capture_backend(
        multi_stack_target,
        monkeypatch,
        real_internal_phases=True,
    )
    stub.fix_edit_line = "# daydream recommended change\n"
    head_before = git_ops.head_sha(multi_stack_target)

    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
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


async def test_deep_archive_recommended_patch_excludes_preexisting_untracked_files(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """A pre-existing untracked file (present before the run) is absent from the
    archived recommended.patch while a fix-created untracked file is present."""
    remote = bare_remote(archive_dir.parent / "origin.git")
    git(multi_stack_target, "remote", "add", "archive", str(remote))
    stub = _install_deep_capture_backend(
        multi_stack_target, monkeypatch, real_internal_phases=True
    )
    stub.fix_edit_line = "# daydream recommended change\n"
    stub.fix_new_generated = "migrations/0002_add_x.sql"  # fix-created, untracked
    (multi_stack_target / "notes.txt").write_text("pre-existing\n")  # pre-fix, untracked

    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
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
    git(multi_stack_target, "remote", "add", "archive", str(remote))
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)  # real_internal_phases=False
    stub.fix_edit_line = "# daydream recommended change\n"
    monkeypatch.setattr(
        "daydream.deep.orchestrator.phase_test_and_heal",
        lambda *a, **k: _ok_with_heal_edit(multi_stack_target),
    )

    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    # The heal edit was written after the pre-test capture; only a post-test
    # re-capture can put it in the archived patch.
    assert "heal_edit.py" in (run_dir / "recommended.patch").read_text()
    # Session-bound capture-point sidecar, mirrored from fix-quality-gate.json.
    sidecar = json.loads(
        (multi_stack_target / ".daydream" / "deep" / "recommended-capture.json").read_text()
    )
    assert sidecar == {"session_id": run_dir.name, "capture_point": "post_test"}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["recommended_patch_capture"] == "post_test"


async def test_deep_archive_commit_excludes_preexisting_untracked_files(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
) -> None:
    """A pre-existing untracked file (before the run) is absent from the daydream
    commit's tree; a fix-created untracked file is present (issue #543)."""
    remote = bare_remote(archive_dir.parent / "origin.git")
    git(multi_stack_target, "remote", "add", "archive", str(remote))
    stub = _install_deep_capture_backend(
        multi_stack_target, monkeypatch, real_internal_phases=True
    )
    stub.fix_edit_line = "# daydream recommended change\n"
    stub.fix_new_generated = "migrations/0002_add_x.sql"  # fix-created, untracked
    (multi_stack_target / "notes.txt").write_text("pre-existing\n")  # pre-fix, untracked

    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
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
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
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
                {"function_name": "shell", "arguments": {"command": "rg -l '\"unclosed"}},
                {"function_name": "shell", "arguments": {"command": "cat api.py"}},
            ],
        }
    )
    source_traj.write_text(json.dumps(traj))

    eval_path = run_dir / "evaluation.json"
    eval_path.unlink(missing_ok=True)

    from daydream.archive import _run_eval

    result = _run_eval(multi_stack_target, session_id, run_dir)
    assert result is not None
    assert eval_path.is_file()

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
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0
    assert not dump_dir.exists()


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
) -> None:
    """AC3 (shallow): the shallow single-pass fix path archives a recommended.patch
    carrying daydream's edit.

    Shallow mode is the deep flow with a forced single stack (#330), so it
    persists ``diff.patch`` like any deep run; recommended.patch must still be
    the distinct artifact carrying daydream's own edit (the deep test asserts
    the same distinctness where both artifacts exist).
    """
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")
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


def test_capture_recommended_patch_excludes_only_preexisting_untracked_files(tmp_path: Path) -> None:
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


def test_capture_recommended_patch_no_change_writes_empty_marker(tmp_path: Path) -> None:
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


def test_fix_applied_signal_new_archive_no_recommendation_skips_fallback(tmp_path: Path) -> None:
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
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
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

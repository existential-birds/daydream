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
from pathlib import Path

import pytest

from daydream import git_ops
from daydream.runner import RunConfig, run

# Reuse the deep-orchestrator stub harness (the hard-won prompt-dispatch heuristics
# live there; re-rolling them would be fragile). tests/ is a namespace package, so
# a sibling test module imports cleanly.
from tests.test_deep_orchestrator import (
    _force_interactive,
    _install_stub_backend,
    _merge_item,
    _noop_commit,
    _ok,
    _silence,
)


def _only_archived_run(archive_dir: Path) -> Path:
    """Return the single archived run directory, asserting there is exactly one."""
    run_dirs = list((archive_dir / "runs").iterdir())
    assert len(run_dirs) == 1, f"expected exactly one archived run, got {run_dirs}"
    return run_dirs[0]


# --- AC1 + AC3: default deep run populates eval metrics AND captures recommended.patch ---


async def test_default_deep_run_populates_eval_and_captures_recommended_patch(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path
) -> None:
    """AC1 + AC3: a default deep run (no --no-eval) populates the manifest's eval
    metrics AND writes a recommended.patch distinct from diff.patch.

    The fix stage edits a TRACKED file (api.py), so the pre-fix → post-fix diff
    is non-empty; commit is a no-op so the edit stays in the worktree for capture.
    """
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_edit_line = "# daydream recommended change\n"
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())

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


async def test_no_eval_leaves_manifest_eval_fields_null(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path
) -> None:
    """AC1b: --no-eval (run_eval=False) skips the eval pass, leaving its metrics null."""
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_edit_line = "# daydream recommended change\n"
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

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
        self, cwd, prompt, output_schema=None, continuation=None,
        agents=None, max_turns=None, read_only=False,
    ):
        from daydream.backends import ResultEvent, TextEvent

        pl = prompt.lower()
        if "beagle-" in pl and "review" in pl:
            yield TextEvent(text="Review complete.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "extract" in pl and "json" in pl:
            yield TextEvent(text="Parsed.")
            yield ResultEvent(
                structured_output={
                    "issues": [{"id": 1, "description": "Add a guard", "file": "main.py", "line": 1}]
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

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_shallow_run_captures_recommended_patch(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path
) -> None:
    """AC3 (shallow): the shallow single-pass fix path archives a recommended.patch
    carrying daydream's edit.

    The shallow review-fix-test path never persists a ``diff.patch`` (the review
    agent runs ``git diff`` itself), so recommended.patch is the sole patch here;
    the "distinct from diff.patch" comparison is covered by the deep test where
    both artifacts exist.
    """
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")
    backend = _FixEditingBackend(feature_branch_repo)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    exit_code = await run(
        RunConfig(
            target=str(feature_branch_repo),
            skill="python",
            quiet=True,
            cleanup=False,
            loop=False,
            shallow=True,
            assume="yes",
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    recommended = run_dir / "recommended.patch"
    assert recommended.is_file()
    recommended_text = recommended.read_text()
    # The captured patch is daydream's fix edit as an ADDED line, diffed against
    # the pre-fix HEAD. It is NOT the reviewed world->universe change: 'world'
    # exists only on the base branch, so it never appears in this post-HEAD diff.
    assert "+# daydream recommended change" in recommended_text
    assert "world" not in recommended_text


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


def test_capture_recommended_patch_none_base_writes_nothing(tmp_path: Path) -> None:
    """A None base (no pre-fix snapshot could be taken) is a no-op."""
    repo = _init_repo_with_commit(tmp_path)
    out = repo / ".daydream" / "recommended.patch"
    assert git_ops.capture_recommended_patch(repo, None, out) is False
    assert not out.exists()


def test_capture_recommended_patch_no_change_writes_nothing(tmp_path: Path) -> None:
    """When nothing changed (base == worktree) no patch is written."""
    repo = _init_repo_with_commit(tmp_path)
    base = git_ops.head_sha(repo)
    out = repo / ".daydream" / "recommended.patch"
    assert git_ops.capture_recommended_patch(repo, base, out) is False
    assert not out.exists()


# --- AC4: applied-signal cascades read recommended.patch (fallback to diff.patch) ---


def _diff_adding(line: str, *, file: str = "app.py") -> str:
    """One-hunk unified diff that adds ``line`` to ``file``."""
    return (
        f"diff --git a/{file} b/{file}\n"
        "index 1111111..2222222 100644\n"
        f"--- a/{file}\n"
        f"+++ b/{file}\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        f"+{line}\n"
    )


def test_fix_applied_signal_prefers_recommended_patch(tmp_path: Path) -> None:
    """AC4: with both patches present, the signal parses recommended.patch hunks,
    not diff.patch hunks — a run whose RECOMMENDATION landed labels 'applied' even
    though the reviewed line is absent post-window."""
    from daydream.training.labeler_signals import fix_applied_signal

    (tmp_path / "diff.patch").write_text(_diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(_diff_adding("recommended = 1"))
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

    (tmp_path / "diff.patch").write_text(_diff_adding("reviewed = 2"))
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


def test_local_commit_applied_signal_prefers_recommended_patch(tmp_path: Path) -> None:
    """AC4: local-commit signal parses recommended.patch when present."""
    from daydream.training.labeler_signals import local_commit_applied_signal

    (tmp_path / "diff.patch").write_text(_diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(_diff_adding("recommended = 1"))
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
        # The later commit carries the recommended line, not the reviewed line.
        file_at_fetcher=lambda repo, path, sha: "existing\nrecommended = 1\n",
    )
    assert sig.verdict == "applied"


def test_local_commit_applied_signal_recommended_line_absent_is_rejected(tmp_path: Path) -> None:
    """AC4: when recommended.patch is present but its line never lands, the reviewed
    line (in diff.patch) must NOT rescue it — proving diff.patch is not consulted."""
    from daydream.training.labeler_signals import local_commit_applied_signal

    (tmp_path / "diff.patch").write_text(_diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(_diff_adding("recommended = 1"))
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
        # Commit carries only the REVIEWED line; recommended line absent.
        file_at_fetcher=lambda repo, path, sha: "existing\nreviewed = 2\n",
    )
    assert sig.verdict == "rejected"

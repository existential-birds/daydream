"""Real-path test for ``make hooks`` pre-push hook installation (issue #388).

Runs the real Makefile's ``hooks:`` target from a real git worktree — both a
primary worktree (``.git`` is a directory) and a linked worktree (``.git`` is a
gitdir-pointer file) — and asserts the hook is installed as a symlink at Git's
resolved ``hooks/pre-push`` path pointing at the invoking worktree's
``scripts/hooks/pre-push``. Installation only; never executes the installed hook.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.harness.git_helpers import git as _git


@pytest.mark.parametrize("worktree_name", ["main", "linked"])
def test_hooks_installs_pre_push_from_worktree(
    linked_worktree: tuple[Path, Path], worktree_name: str
) -> None:
    main_repo, linked = linked_worktree
    worktree = main_repo if worktree_name == "main" else linked

    # The fixture repo doesn't ship the daydream tooling — drop the real
    # Makefile + hook script into the invoking worktree so `make hooks`
    # exercises the real recipe against a real worktree topology.
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copy(repo_root / "Makefile", worktree / "Makefile")
    script_dir = worktree / "scripts" / "hooks"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_root / "scripts" / "hooks" / "pre-push", script_dir / "pre-push")

    proc = subprocess.run(
        ["make", "hooks"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Pre-push hook installed" in proc.stdout

    # Installed at Git's worktree-aware resolved path, as a symlink.
    dest = worktree / _git(worktree, "rev-parse", "--git-path", "hooks/pre-push")
    assert dest.is_symlink()
    # The symlink resolves to THIS worktree's source file (never a sibling's).
    assert dest.resolve() == (worktree / "scripts" / "hooks" / "pre-push").resolve()

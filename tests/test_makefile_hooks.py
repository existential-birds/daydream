"""Real-path behavior tests for the git hooks (issues #388, #717).

Both hook scripts are executed for real, from a real git worktree. What is
asserted is behavior with a silent failure mode:

- the pre-commit hook lints the git INDEX content, only the staged files, only
  the in-scope ones, hands raw (never C-quoted) paths to ruff, exits 0 when
  nothing Python is staged, and propagates a ruff failure;
- the pre-push hook scrubs Git's inherited ``GIT_DIR``/``GIT_WORK_TREE``/... env
  before running the gate, so a gate subprocess cannot write into the index of
  the worktree being pushed.

Which targets a Makefile recipe happens to depend on, and which command a hook
shells out to, are deliberately NOT asserted anywhere here: that only restates
the build back to itself and breaks on every legitimate edit to it.

The tooling itself never runs: PATH-shim recorders named ``uv``/``make``/``docker``
capture ``{command, cwd, args, stdin}`` as JSONL, so the real hooks are
observable without a Docker daemon, a network pull, or a resolver touch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.harness.git_helpers import git as _git


def _install_recording_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    exit_code: dict[str, int] | None = None,
) -> Path:
    """Prepend a fakebin of PATH-shim recorders for ``names`` to PATH.

    Each shim appends ``{"command": <basename>, "cwd": os.getcwd(),
    "args": sys.argv[1:]}`` to the JSONL log at ``$DAYDREAM_COMMAND_LOG``. The
    shebang is pinned to ``sys.executable`` (never the ambient python).
    Returns the log path.

    Exit behavior: every shim exits 0 by default, so the real Makefile graph
    can be observed without any real tooling — except ``git``, which always
    delegates to the real binary (passing stdout/stderr and the exit code
    through) because hook scripts need real git plumbing output (worktree
    topology, staged-file listings). ``exit_code`` overrides the exit status
    per shim name, e.g. ``{"uv": 1}`` to simulate ruff finding lint errors.
    """
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    log = tmp_path / "command-log.jsonl"
    monkeypatch.setenv("DAYDREAM_COMMAND_LOG", str(log))
    shim_body = (
        "import json, os, sys\n"
        "log = os.environ['DAYDREAM_COMMAND_LOG']\n"
        "# Capture piped stdin (e.g. `git show :path | uv ...`) so tests can assert\n"
        "# ruff was fed the INDEX content rather than the working tree. Skipped for\n"
        "# git (which must pass its real stdin through to the delegated binary) and\n"
        "# when stdin is a tty (nothing piped).\n"
        "stdin = (sys.stdin.read() if (os.path.basename(sys.argv[0]) != 'git'\n"
        "        and not sys.stdin.isatty()) else None)\n"
        "with open(log, 'a', encoding='utf-8') as f:\n"
        "    json.dump({'command': os.path.basename(sys.argv[0]),\n"
        "               'cwd': os.getcwd(), 'args': sys.argv[1:],\n"
        "               'stdin': stdin,\n"
        "               'git_local_env': {k: os.environ.get(k) for k in\n"
        "                       ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE',\n"
        "                        'GIT_COMMON_DIR', 'GIT_PREFIX')}}, f)\n"
        "    f.write('\\n')\n"
    )
    real_git = shutil.which("git")
    for name in names:
        shim = fakebin / name
        if name == "git":
            tail = (
                "import subprocess\n"
                f"sys.exit(subprocess.run([{real_git!r}, *sys.argv[1:]])"
                ".returncode)\n"
            )
        else:
            status = (exit_code or {}).get(name, 0)
            tail = f"sys.exit({status})\n"
        shim.write_text(f"#!{sys.executable}\n{shim_body}{tail}", encoding="utf-8")
        shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    return log


def _read_command_records(log: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line.encode("utf-8")))
    return records


def _stage_file(worktree: Path, relpath: str, content: str) -> None:
    p = worktree / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    _git(worktree, "add", relpath)


def _copy_pre_commit_hook(worktree: Path) -> Path:
    """Drop the real pre-commit script into a throwaway worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = worktree / "scripts" / "hooks"
    script_dir.mkdir(parents=True, exist_ok=True)
    hook = script_dir / "pre-commit"
    shutil.copy(repo_root / "scripts" / "hooks" / "pre-commit", hook)
    hook.chmod(0o755)
    return hook


def test_pre_commit_runs_ruff_only_on_staged_python_files(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    _copy_pre_commit_hook(worktree)
    # Stage scratch files BEFORE the shims shadow `git` on PATH (the shim
    # delegates anyway, but this keeps the staging setup unrecorded).
    _stage_file(worktree, "daydream/spike_a.py", "x = 1\n")
    _stage_file(worktree, "tests/spike_b.py", "y = 2\n")
    _stage_file(worktree, "notes.md", "not python\n")
    # An unstaged python edit in a different file must NOT reach ruff: only
    # the index is linted.
    (worktree / "daydream" / "unstaged.py").write_text("z = 3\n", encoding="utf-8")
    # A non-ASCII filename, the core.quotepath trap: git would C-quote it in
    # newline output, but the hook's NUL-delimited listing must hand the raw
    # name to ruff.
    _stage_file(worktree, "daydream/\u00e9t\u00e9.py", "x = 4\n")

    log = _install_recording_commands(tmp_path, monkeypatch, ("uv", "git"))
    proc = subprocess.run(
        [str(worktree / "scripts" / "hooks" / "pre-commit")],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recs = _read_command_records(log)
    uv_calls = [r for r in recs if r["command"] == "uv"]
    # One ruff invocation per staged .py file, each fed that file's INDEX
    # content via stdin (--stdin-filename; nothing is read from the working
    # tree). Unstaged and non-Python files are excluded.
    assert len(uv_calls) == 3
    by_path = {r["args"][4]: r for r in uv_calls}
    assert set(by_path) == {"daydream/spike_a.py", "tests/spike_b.py", "daydream/\u00e9t\u00e9.py"}
    for path, r in by_path.items():
        assert r["args"] == ["run", "ruff", "check", "--stdin-filename", path, "-"]
        assert r["cwd"] == str(worktree)
    # Each ruff feed carries exactly the staged bytes, never the working tree.
    assert by_path["daydream/spike_a.py"]["stdin"] == "x = 1\n"
    assert by_path["tests/spike_b.py"]["stdin"] == "y = 2\n"
    assert by_path["daydream/\u00e9t\u00e9.py"]["stdin"] == "x = 4\n"
    # The non-ASCII path reaches ruff unquoted (no C-quote/escape wrapper).
    assert by_path["daydream/\u00e9t\u00e9.py"]["args"][4] == "daydream/\u00e9t\u00e9.py"
    # No other commands may appear.
    assert {r["command"] for r in recs} <= {"uv", "git"}


def test_pre_commit_lints_index_content_not_working_tree(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    _copy_pre_commit_hook(worktree)
    # Stage a clean file, then leave a DIFFERENT (lint-breaking) state in the
    # working tree un-staged. The gate must lint the STAGED bytes, so the
    # un-staged experiment cannot block (or wrongly clear) the commit.
    scope_py = worktree / "daydream" / "scope.py"
    _stage_file(worktree, "daydream/scope.py", "STAGED_VALUE = 1\n")
    scope_py.write_text("STAGED_VALUE = import os  # unstaged experiment\n", encoding="utf-8")

    log = _install_recording_commands(tmp_path, monkeypatch, ("uv", "git"))
    proc = subprocess.run(
        [str(worktree / "scripts" / "hooks" / "pre-commit")],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recs = _read_command_records(log)
    uv_calls = [r for r in recs if r["command"] == "uv"]
    assert len(uv_calls) == 1
    call = uv_calls[0]
    assert call["args"][4] == "daydream/scope.py"
    # The index blob, not the file's working-tree bytes: a live unstaged edit
    # under the SAME path is excluded from the lint feed.
    assert call["stdin"] == "STAGED_VALUE = 1\n"


def test_pre_commit_exits_zero_without_staged_python(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    _copy_pre_commit_hook(worktree)
    log = _install_recording_commands(tmp_path, monkeypatch, ("uv", "git"))
    proc = subprocess.run(
        [str(worktree / "scripts" / "hooks" / "pre-commit")],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # No ruff invocation when nothing relevant is staged — the speed contract.
    assert not [r for r in _read_command_records(log) if r["command"] == "uv"]


def test_pre_commit_skips_python_files_outside_lint_scope(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    _copy_pre_commit_hook(worktree)
    # The commit gate is scoped to the canonical lint surface (`make lint` /
    # `make check`: root `daydream tests` plus the standalone rl project). A
    # staged .py under scripts/ or mypy_stubs/ is tracked by no lint scope, so
    # it must not be hard-blocked at commit time by a rule CI never enforces.
    _stage_file(worktree, "daydream/in_scope.py", "x = 1\n")
    _stage_file(worktree, "scripts/out_of_scope.py", "y = 2\n")
    _stage_file(worktree, "mypy_stubs/out_of_scope.py", "z = 3\n")

    log = _install_recording_commands(tmp_path, monkeypatch, ("uv", "git"))
    proc = subprocess.run(
        [str(worktree / "scripts" / "hooks" / "pre-commit")],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    uv_calls = [r for r in _read_command_records(log) if r["command"] == "uv"]
    # Only the in-scope staged file is linted; the others are skipped.
    assert len(uv_calls) == 1
    call = uv_calls[0]
    assert call["args"] == ["run", "ruff", "check", "--stdin-filename", "daydream/in_scope.py", "-"]
    assert call["stdin"] == "x = 1\n"


def test_pre_commit_propagates_ruff_failure(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    _copy_pre_commit_hook(worktree)
    _stage_file(worktree, "daydream/broken.py", "x = 1\n")
    # The uv shim exits 1 like ruff does on a lint error; the hook must
    # propagate that status, never swallow it.
    _install_recording_commands(tmp_path, monkeypatch, ("uv", "git"), exit_code={"uv": 1})
    proc = subprocess.run(
        [str(worktree / "scripts" / "hooks" / "pre-commit")],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    # Failure output names the gate and how to fix it:
    assert "ruff" in proc.stdout.lower() or "ruff" in proc.stderr.lower()


def test_pre_push_scrubs_inherited_git_env_from_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate subprocess must not inherit the pushing worktree's git env.

    Git exports ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE``/
    ``GIT_COMMON_DIR``/``GIT_PREFIX`` to hooks, so without the
    ``git rev-parse --local-env-vars`` scrub every gate subprocess that builds
    its own throwaway repository would write into the index of the worktree
    being pushed instead of honoring ``git -C <repo>``. Silent and destructive.

    Which target the hook runs is not asserted: that is the Makefile's business.
    """
    repo_root = Path(__file__).resolve().parents[1]
    _install_recording_commands(tmp_path, monkeypatch, ("make", "uv", "docker"))
    log = tmp_path / "command-log.jsonl"

    clean_env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    inherited_git_env = {
        "GIT_DIR": "/sentinel/git-dir",
        "GIT_WORK_TREE": "/sentinel/work-tree",
        "GIT_INDEX_FILE": "/sentinel/index",
        "GIT_COMMON_DIR": "/sentinel/common-dir",
        "GIT_PREFIX": "sentinel-prefix/",
    }
    clean_env.update(inherited_git_env)
    proc = subprocess.run(
        [str(repo_root / "scripts" / "hooks" / "pre-push")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        input="",
        env=clean_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "✓ All checks passed" in proc.stdout

    gate = next(r for r in _read_command_records(log) if r["command"] == "make")
    assert all(gate["git_local_env"][name] is None for name in inherited_git_env)

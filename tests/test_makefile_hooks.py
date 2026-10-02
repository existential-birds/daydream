"""Execute real hooks in real Git worktrees with recording PATH shims.

Pre-commit must lint staged Python bytes and raw paths, skip an empty scope,
and propagate lint failure. Pre-push must scrub inherited Git environment
so its gate cannot mutate the pushing worktree. Assert behavior rather than
Makefile recipe wiring; shims avoid external tools, Docker, and network."""

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...], exit_code: dict[str, int] | None = None,
) -> Path:
    """Record command, cwd, args and stdin to JSONL through PATH shims.

    Pin the Python interpreter; delegate Git to the real binary for worktree and
    index plumbing. Other commands succeed unless exit_code overrides them.
    Return the log path."""
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
            tail = ("import subprocess\n" f"sys.exit(subprocess.run([{real_git!r}, *sys.argv[1:]])" ".returncode)\n")
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


def _run_pre_commit(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exit_code: dict[str, int] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    """Install and execute the real pre-commit hook under recording shims.

    Stage first to keep setup out of the command log. Return process and log."""
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = worktree / "scripts" / "hooks"
    script_dir.mkdir(parents=True, exist_ok=True)
    hook = script_dir / "pre-commit"
    shutil.copy(repo_root / "scripts" / "hooks" / "pre-commit", hook)
    hook.chmod(0o755)
    log = _install_recording_commands(tmp_path, monkeypatch, ("uv", "git"), exit_code=exit_code)
    proc = subprocess.run([str(hook)], cwd=worktree, capture_output=True, text=True,)
    return proc, _read_command_records(log)

def test_pre_commit_runs_ruff_only_on_staged_python_files(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    # Stage before recording shims so setup stays out of the command log.
    _stage_file(worktree, "daydream/spike_a.py", "x = 1\n")
    _stage_file(worktree, "tests/spike_b.py", "y = 2\n")
    _stage_file(worktree, "notes.md", "not python\n")
    # Only staged bytes may reach Ruff.
    (worktree / "daydream" / "unstaged.py").write_text("z = 3\n", encoding="utf-8")
    # NUL-delimited paths must bypass Git's C-quoting for non-ASCII names.
    _stage_file(worktree, "daydream/\u00e9t\u00e9.py", "x = 4\n")

    proc, recs = _run_pre_commit(worktree, tmp_path, monkeypatch)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    uv_calls = [r for r in recs if r["command"] == "uv"]
    # Feed each in-scope staged Python blob through stdin; exclude unstaged and non-Python files.
    assert len(uv_calls) == 3
    by_path = {r["args"][4]: r for r in uv_calls}
    assert set(by_path) == {"daydream/spike_a.py", "tests/spike_b.py", "daydream/\u00e9t\u00e9.py"}
    for path, r in by_path.items():
        assert r["args"] == ["run", "ruff", "check", "--stdin-filename", path, "-"]
        assert r["cwd"] == str(worktree)
    assert by_path["daydream/spike_a.py"]["stdin"] == "x = 1\n"
    assert by_path["tests/spike_b.py"]["stdin"] == "y = 2\n"
    assert by_path["daydream/\u00e9t\u00e9.py"]["stdin"] == "x = 4\n"
    assert by_path["daydream/\u00e9t\u00e9.py"]["args"][4] == "daydream/\u00e9t\u00e9.py"
    assert {r["command"] for r in recs} <= {"uv", "git"}

def test_pre_commit_lints_index_content_not_working_tree(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    # Leave lint-breaking worktree bytes under the same path to prove Ruff reads the clean index blob.
    scope_py = worktree / "daydream" / "scope.py"
    _stage_file(worktree, "daydream/scope.py", "STAGED_VALUE = 1\n")
    scope_py.write_text("STAGED_VALUE = import os  # unstaged experiment\n", encoding="utf-8")

    proc, recs = _run_pre_commit(worktree, tmp_path, monkeypatch)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    uv_calls = [r for r in recs if r["command"] == "uv"]
    assert len(uv_calls) == 1
    call = uv_calls[0]
    assert call["args"][4] == "daydream/scope.py"
    assert call["stdin"] == "STAGED_VALUE = 1\n"

def test_pre_commit_exits_zero_without_staged_python(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    proc, recs = _run_pre_commit(worktree, tmp_path, monkeypatch)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not [r for r in recs if r["command"] == "uv"]

def test_pre_commit_skips_python_files_outside_lint_scope(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    # Match the CI lint scope: daydream, tests and RL. Scripts and stubs must not acquire
    # commit-only requirements that CI never enforces.
    _stage_file(worktree, "daydream/in_scope.py", "x = 1\n")
    _stage_file(worktree, "scripts/out_of_scope.py", "y = 2\n")
    _stage_file(worktree, "mypy_stubs/out_of_scope.py", "z = 3\n")

    proc, recs = _run_pre_commit(worktree, tmp_path, monkeypatch)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    uv_calls = [r for r in recs if r["command"] == "uv"]
    assert len(uv_calls) == 1
    call = uv_calls[0]
    assert call["args"] == ["run", "ruff", "check", "--stdin-filename", "daydream/in_scope.py", "-"]
    assert call["stdin"] == "x = 1\n"

def test_pre_commit_propagates_ruff_failure(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _main_repo, worktree = linked_worktree
    _stage_file(worktree, "daydream/broken.py", "x = 1\n")
    # A simulated Ruff failure must propagate through the hook.
    proc, _ = _run_pre_commit(worktree, tmp_path, monkeypatch, exit_code={"uv": 1})
    assert proc.returncode != 0
    assert "ruff" in proc.stdout.lower() or "ruff" in proc.stderr.lower()

def test_pre_push_scrubs_inherited_git_env_from_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inherited GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, GIT_COMMON_DIR and
    GIT_PREFIX can redirect gate fixtures into the pushing worktree's index.
    The hook must scrub them before the gate builds throwaway repositories."""
    repo_root = Path(__file__).resolve().parents[1]
    _install_recording_commands(tmp_path, monkeypatch, ("make", "uv", "docker"))
    log = tmp_path / "command-log.jsonl"

    clean_env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    inherited_git_env = {
        "GIT_DIR": "/sentinel/git-dir", "GIT_WORK_TREE": "/sentinel/work-tree", "GIT_INDEX_FILE": "/sentinel/index",
        "GIT_COMMON_DIR": "/sentinel/common-dir", "GIT_PREFIX": "sentinel-prefix/",
    }
    clean_env.update(inherited_git_env)
    proc = subprocess.run(
        [str(repo_root / "scripts" / "hooks" / "pre-push")], cwd=repo_root, capture_output=True, text=True, input="",
        env=clean_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "✓ All checks passed" in proc.stdout

    gate = next(r for r in _read_command_records(log) if r["command"] == "make")
    assert all(gate["git_local_env"][name] is None for name in inherited_git_env)

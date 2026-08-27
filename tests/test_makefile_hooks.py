"""Real-path contract tests for the local and pre-push make gates (issues #388, #717).

Three contracts are exercised here against the real Makefile and hook script
from a real git worktree:

- ``make hooks`` installs the pre-push hook as a worktree-aware symlink,
  pointing at the invoking worktree's own source file (both primary and linked
  worktrees).
- ``make check`` composes every gate CI enforces: the root lock/lint/types/tests
  suite, the Docker-backed actionlint pass over all workflow templates, and the
  standalone RL project's four gates — each run from the right working
  directory with the exact command line CI would issue.
- the pre-push hook delegates to ``make check`` after its signature loop.

The composition/delegation tests never execute the real tooling: PATH-shim
recorders named ``uv``/``docker`` capture ``{command, cwd, args}`` as JSONL so the
real Makefile dependency graph and the real hook are observable without a Docker
daemon, a network pull, or a resolver touch.
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
from tests.test_workflow_templates import (
    _ACTIONLINT_REF_RE,
    REPO_WORKFLOWS_DIR,
    TEMPLATES_DIR,
    job_steps,
    load_workflow,
)


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


def _install_recording_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> Path:
    """Prepend a fakebin of PATH-shim recorders for ``names`` to PATH.

    Each shim appends ``{"command": <basename>, "cwd": os.getcwd(),
    "args": sys.argv[1:]}`` to the JSONL log at ``$DAYDREAM_COMMAND_LOG`` and
    exits 0, so the real Makefile graph can be observed without any real
    tooling. The shebang is pinned to ``sys.executable`` (never the ambient
    python). Returns the log path.
    """
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    log = tmp_path / "command-log.jsonl"
    monkeypatch.setenv("DAYDREAM_COMMAND_LOG", str(log))
    shim_body = (
        "import json, os, sys\n"
        "log = os.environ['DAYDREAM_COMMAND_LOG']\n"
        "with open(log, 'a', encoding='utf-8') as f:\n"
        "    json.dump({'command': os.path.basename(sys.argv[0]),\n"
        "               'cwd': os.getcwd(), 'args': sys.argv[1:]}, f)\n"
        "    f.write('\\n')\n"
    )
    for name in names:
        shim = fakebin / name
        shim.write_text(f"#!{sys.executable}\n{shim_body}", encoding="utf-8")
        shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}:{os.environ.get('PATH', '')}")
    return log


def _read_command_records(log: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line.encode("utf-8")))
    return records


def test_check_runs_root_workflow_and_standalone_rl_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    log = _install_recording_commands(tmp_path, monkeypatch, ("uv", "docker"))

    # Strip make's jobserver/MAKELEVEL chatter so the shim children don't get
    # tangled in an inherited parallel-make env.
    clean_env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    proc = subprocess.run(
        ["make", "check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    recs = _read_command_records(log)
    rl_root = repo_root / "rl" / "daydream_review_v1"
    cmds = [(r["command"], r["cwd"]) for r in recs]
    assert cmds == (
        [("uv", str(repo_root))] * 4
        + [("docker", str(repo_root))]
        + [("uv", str(rl_root))] * 5
    )

    argvs = [r["args"] for r in recs]
    # Root suite: uv lock --check, ruff, mypy, pytest (-n auto parallel).
    assert argvs[0] == ["lock", "--check"]
    assert argvs[1] == ["run", "ruff", "check", "daydream", "tests"]
    assert argvs[2] == ["run", "mypy", "daydream", "tests"]
    assert argvs[3] == ["run", "pytest", "-n", "auto"]

    # actionlint: docker invocation shaped exactly like ci.yml's, with the
    # image digest read LIVE from the CI workflow (never hardcoded here).
    wf = load_workflow(REPO_WORKFLOWS_DIR / "ci.yml")
    steps = job_steps(wf, "check")
    actionlint = next(s for s in steps if s.get("name") == "Lint workflows with actionlint")
    (image_ref,) = _ACTIONLINT_REF_RE.findall(actionlint["run"])

    docker = argvs[4]
    assert docker[0:2] == ["run", "--rm"]
    mount_at = docker.index("-v")
    assert docker[mount_at + 1] == f"{repo_root}:/repo"
    assert [a for a in docker if a.endswith(":/repo")] == [f"{repo_root}:/repo"]
    w_at = docker.index("-w")
    assert docker[w_at + 1] == "/repo"
    assert image_ref in docker
    image_idx = docker.index(image_ref)
    assert "-color" in docker
    # The selectors expand (shell glob) to the full shipped-workflow set, each
    # exactly once, positionally grouped by the three GLB globs.
    file_args = docker[image_idx + 2 :]  # drop the image and -color
    expected_files = {
        *(p.relative_to(repo_root).as_posix() for p in REPO_WORKFLOWS_DIR.glob("*.yml")),
        *(p.relative_to(repo_root).as_posix() for p in TEMPLATES_DIR.rglob("*.yml")),
    }
    assert set(file_args) == expected_files
    assert len(file_args) == len(expected_files)

    # Standalone RL project: its own lock/sync/lint/types/tests, run from its
    # dir (mirroring ci.yml's rl-check job, which runs an explicit `uv sync`).
    assert argvs[5] == ["lock", "--check"]
    assert argvs[6] == ["sync"]
    assert argvs[7] == ["run", "ruff", "check", "."]
    assert argvs[8] == ["run", "mypy", "daydream_review_v1", "tests"]
    assert argvs[9] == ["run", "pytest"]


def test_pre_push_delegates_quality_gate_to_make_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _install_recording_commands(tmp_path, monkeypatch, ("make", "uv", "docker"))
    log = tmp_path / "command-log.jsonl"

    clean_env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
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

    recs = _read_command_records(log)
    assert len(recs) == 1
    assert recs[0]["command"] == "make"
    assert recs[0]["cwd"] == str(repo_root)
    assert recs[0]["args"] == ["check"]

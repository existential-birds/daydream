"""Tests for :mod:`daydream.git_ops`. These tests use real ``git`` (and optionally ``gh``) against ``tmp_path``
fixtures. Transport fault injection covers timeout, privacy, and response-shape boundaries;
filesystem and rollback regressions use actual repositories."""

from __future__ import annotations

import base64
import errno
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.git_ops import (
    BranchNotFoundError,
    GitError,
    NotAWorktreeError,
    WrongBranchError,
    github as git_github,
    mutations as git_mutations,
    process as git_process,
    snapshot as git_snapshot,
)
from tests.conftest import _make_repo_with_main
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import (
    bare_remote as _bare_remote,
    commit as _commit,
    configure_identity as _configure_identity,
    git as _git,
    init_repo as _init_repo,
    install_diff_base_git_shim,
    write_and_stage,
)


@pytest.fixture
def repo(git_repo: Path) -> Path:
    """Return the git_repo fixture under the local name repo."""
    return git_repo


def _patch_subprocess_run(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int = 0, stdout: str = "", stderr: str = "",
) -> None:
    """Install a fixed ``gh`` subprocess result at the ``git_ops`` seam."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr),
    )

def _repo_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "remote", "add", "origin", str(bare))
    return repo, bare

def _topic_repo(tmp_path: Path) -> Path:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    return repo

def test_resolve_diff_merge_base_prefers_present_origin_ref(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    write_and_stage(repo, "upstream.py", "UPSTREAM = 1\n")
    remote_tip = _commit(repo, "remote advancement")
    _git(repo, "branch", "feature", remote_tip)
    _git(repo, "reset", "--hard", base)
    _git(repo, "update-ref", "refs/remotes/origin/main", remote_tip)
    _git(repo, "checkout", "feature")
    write_and_stage(repo, "feature.py", "FEATURE = 1\n")
    head = _commit(repo, "feature")
    assert git_ops.merge_base(repo, "main", head) == base
    assert git_ops.resolve_diff_merge_base(repo, "main", head) == remote_tip

def test_resolve_diff_merge_base_carries_only_ancestor_of_dangling_base(repo: Path) -> None:
    common = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    write_and_stage(repo, "feature.py", "FEATURE = 1\n")
    head = _commit(repo, "feature")
    _git(repo, "checkout", "--detach", common)
    write_and_stage(repo, "side.py", "SIDE = 1\n")
    dangling_tip = _commit(repo, "dangling selected base")
    _git(repo, "checkout", "feature")
    assert git_ops.resolve_diff_merge_base(repo, dangling_tip, head) == common

@pytest.mark.parametrize(
    ("mode", "expected_message", "private_fragment"),
    [
        ("malformed-head", "branch-focus recorded head commit cannot be resolved", "PRIVATE_STDOUT_SENTINEL"),
        ("invalid-utf8-head", "branch-focus recorded head commit probe failed", "\\xff\\xfe"),
        ("uppercase-head", "branch-focus recorded head commit cannot be resolved",
         "A PRIVATE VALUE THAT CANNOT APPEAR"),
        (
            "abbreviated-head", "branch-focus recorded head commit cannot be resolved",
            "A PRIVATE VALUE THAT CANNOT APPEAR",
        ),
        ("sha256-head", "branch-focus recorded head commit cannot be resolved", "A PRIVATE VALUE THAT CANNOT APPEAR"),
        ("malformed-merge", "branch-focus diff merge-base cannot be resolved", "PRIVATE_MERGE_SENTINEL"),
        ("remove-after-preference", "branch-focus recorded head commit probe failed", "diff base git shim"),
    ],
)
def test_resolve_diff_merge_base_rejects_and_redacts_external_git_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, expected_message: str, private_fragment: str,
                                                                           repo: Path,
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    with install_diff_base_git_shim(tmp_path, monkeypatch, mode=mode, head=head):
        with pytest.raises(GitError) as raised:
            git_ops.resolve_diff_merge_base(repo, "main", head)
    message = str(raised.value)
    assert message == expected_message
    assert private_fragment not in message
    assert str(repo) not in message
    assert head not in message
    assert len(message) < 200

def test_resolve_diff_merge_base_redacts_external_git_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path,
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    with install_diff_base_git_shim(tmp_path, monkeypatch, mode="timeout-preference", head=head):
        with pytest.raises(GitError) as raised:
            git_ops.resolve_diff_merge_base(repo, "main", head)
    assert str(raised.value) == "branch-focus preferred base probe failed"
    assert str(repo) not in str(raised.value)
    assert head not in str(raised.value)

@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_preserves_exact_filename_bytes(
    tmp_path: Path, include_untracked: bool, repo: Path,
) -> None:
    tracked = [" leading and trailing ", "line\nbreak"]
    untracked = [" scratch ", "scratch\nline"]
    for name in tracked:
        (repo / name).write_bytes(b"tracked\x00\xff")
    _git(repo, "add", "-A")
    _commit(repo, "unusual paths")
    for name in untracked:
        (repo / name).write_bytes(b"scratch\x00\xfe")
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "independent", include_untracked=include_untracked)
    assert set(git_ops.ls_tree_files(snapshot.repo, "HEAD", strict=True)) >= set(tracked)
    for name in tracked:
        assert (snapshot.repo / name).read_bytes() == b"tracked\x00\xff"
    for name in untracked:
        assert (snapshot.repo / name).exists() is include_untracked
        if include_untracked:
            assert (snapshot.repo / name).read_bytes() == b"scratch\x00\xfe"

def _repository_identity(repo: Path) -> tuple[str, str, bytes, str]:
    """Observe HEAD, worktree/index status, exact staged bytes, and all refs."""
    return (
        _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--short"), git_ops.staged_patch(repo),
        _git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
    )


@pytest.mark.parametrize("include_untracked", [False, True])
@pytest.mark.parametrize(
    ("staged", "children", "replacement"),
    [
        (True, ("a.txt", "b.txt"), b"replacement file\x00\xff"),
        (False, ("child.txt",), b"untracked replacement\x00\xff"),
    ], ids=["staged", "unstaged"],
)
def test_independent_snapshot_preserves_directory_to_file_change(
    tmp_path: Path, include_untracked: bool, staged: bool, children: tuple[str, ...], replacement: bytes,
                                                                 repo: Path,
) -> None:
    nested = repo / "x"
    nested.mkdir()
    for child in children:
        (nested / child).write_text(f"old {child}\n", encoding="utf-8")
    _git(repo, "add", "x")
    _commit(repo, "add tracked directory")
    shutil.rmtree(nested)
    nested.write_bytes(replacement)
    if staged:
        _git(repo, "add", "-A")
    before = _repository_identity(repo)
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=include_untracked)
    for child in children:
        assert not (snapshot.repo / "x" / child).exists()
    if staged or include_untracked:
        assert (snapshot.repo / "x").is_file()
        assert (snapshot.repo / "x").read_bytes() == replacement
        assert _git(snapshot.repo, "status", "--short") == before[1]
    else:
        assert not (snapshot.repo / "x").is_file()
        assert _git(snapshot.repo, "status", "--short") == "D x/child.txt"
    assert git_ops.staged_patch(snapshot.repo) == before[2]
    assert git_ops.ls_files(snapshot.repo, strict=True) == git_ops.ls_files(repo, strict=True)
    assert _repository_identity(repo) == before
    assert (repo / "x").read_bytes() == replacement


def test_independent_snapshot_preserves_staged_file_to_directory_change(tmp_path: Path, repo: Path) -> None:
    replaced = repo / "x"
    replaced.write_text("old file\n", encoding="utf-8")
    _git(repo, "add", "x")
    _commit(repo, "add tracked file")
    replaced.unlink()
    replaced.mkdir()
    (replaced / "child.txt").write_bytes(b"replacement child\x00\xff")
    _git(repo, "add", "-A")
    before_status = _git(repo, "status", "--short")
    before_patch = git_ops.staged_patch(repo)
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    assert (snapshot.repo / "x").is_dir()
    assert (snapshot.repo / "x" / "child.txt").read_bytes() == (
        b"replacement child\x00\xff"
    )
    assert _git(snapshot.repo, "status", "--short") == before_status
    assert git_ops.staged_patch(snapshot.repo) == before_patch
    assert _git(repo, "status", "--short") == before_status

@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_preserves_non_utf8_paths(tmp_path: Path, include_untracked: bool, repo: Path) -> None:
    tracked = os.fsdecode(b"tracked-\xff")
    scratch = os.fsdecode(b"scratch-\xfe")
    try:
        (repo / tracked).write_bytes(b"tracked bytes")
    except OSError as exc:
        if exc.errno == errno.EILSEQ:
            pytest.skip("host filesystem rejects non-UTF-8 filenames; covered on Linux CI")
        raise
    _git(repo, "add", "-A")
    _commit(repo, "non-UTF-8 name")
    (repo / scratch).write_bytes(b"scratch bytes")
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=include_untracked)
    assert (snapshot.repo / tracked).read_bytes() == b"tracked bytes"
    assert (snapshot.repo / scratch).exists() is include_untracked
    if include_untracked:
        assert (snapshot.repo / scratch).read_bytes() == b"scratch bytes"

def test_independent_snapshot_rejects_destination_parent_symlink(tmp_path: Path, repo: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "child.txt"
    sentinel.write_text("untouched")
    (repo / "parent").symlink_to(outside, target_is_directory=True)
    _git(repo, "add", "parent")
    _commit(repo, "committed outward link")
    (repo / "parent").unlink()
    (repo / "parent").mkdir()
    (repo / "parent" / "child.txt").write_text("new tracked child")
    _git(repo, "add", "-A")
    before = git_ops.staged_patch(repo)
    with pytest.raises(git_ops.SnapshotPreparationError, match="symlinked parent"):
        git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    assert sentinel.read_text() == "untouched"
    assert (repo / "parent" / "child.txt").read_text() == "new tracked child"
    assert git_ops.staged_patch(repo) == before

def test_independent_snapshot_rejects_shared_alternates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path,
) -> None:
    real_clone = git_ops.clone
    def clone_with_alternates(
        remote_url: str, target: Path, *, blobless: bool = False, no_local: bool = False, timeout: int = 300,
    ) -> None:
        assert no_local
        real_clone(remote_url, target, blobless=blobless, no_local=no_local, timeout=timeout)
        (target / ".git" / "objects" / "info" / "alternates").write_text(str(repo / ".git" / "objects") + "\n")
    monkeypatch.setattr(git_mutations, "clone", clone_with_alternates)
    with pytest.raises(git_ops.SnapshotPreparationError, match="alternates or remotes"):
        git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)

@pytest.mark.parametrize("unborn", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("variable", [
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_OBJECT_DIRECTORY", "GIT_DIR",
    "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES", "GIT_PREFIX", "GIT_SHALLOW_FILE", "GIT_GRAFT_FILE",
    "GIT_REPLACE_REF_BASE", "GIT_REFERENCE_BACKEND", "GIT_TEMPLATE_DIR",
    "GIT_CONFIG", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
    "GIT_CONFIG_PARAMETERS", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS", "GIT_TRACE", "GIT_EXEC_PATH",
])
def test_independent_snapshot_rejects_inherited_git_overrides_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, empty: bool, unborn: bool,
) -> None:
    repo = tmp_path / "source"
    _init_repo(repo)
    write_and_stage(repo, "app.py", b"VALUE = 1\n")
    if not unborn:
        _commit(repo, "initial")
    (repo / "app.py").write_bytes(b"VALUE = 2\n")
    external = tmp_path / "external"
    shutil.copytree(repo / ".git", external)
    destination = tmp_path / "snapshot"
    def file_bytes(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()
        }
    before_source = file_bytes(repo)
    before_external = file_bytes(external)
    value = "" if empty else str(
        repo / ".git" / "objects"
        if variable == "GIT_ALTERNATE_OBJECT_DIRECTORIES"
        else external / "objects" if variable == "GIT_OBJECT_DIRECTORY" else external
    )
    with monkeypatch.context() as poison:
        poison.setenv(variable, value)
        with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
            git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
        assert os.environ[variable] == value
    assert variable in str(raised.value)
    if value:
        assert value not in str(raised.value)
    assert not destination.exists()
    assert file_bytes(repo) == before_source
    assert file_bytes(external) == before_external

def test_independent_snapshot_redacts_malformed_git_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      repo: Path,
) -> None:
    destination = tmp_path / "snapshot"
    newline_name = "GIT_TRACE_PRIVATE\nSECRET_NAME"
    oversized_name = "GIT_CONFIG_" + "S" * 500
    private_value = "PRIVATE_ENV_VALUE_SENTINEL"
    monkeypatch.setenv(newline_name, private_value)
    monkeypatch.setenv(oversized_name, private_value)
    with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
        git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
    message = str(raised.value)
    assert "2 invalid Git variable names" in message
    assert newline_name not in message
    assert oversized_name not in message
    assert private_value not in message
    assert "SECRET_NAME" not in message
    assert len(message) < 500
    assert not destination.exists()
    assert os.environ[newline_name] == private_value
    assert os.environ[oversized_name] == private_value

def test_independent_snapshot_redacts_malformed_names_at_process_boundary(tmp_path: Path, repo: Path) -> None:
    destination = tmp_path / "snapshot"
    newline_name = "GIT_TRACE_PRIVATE\nSECRET_NAME"
    oversized_name = "GIT_CONFIG_" + "S" * 500
    private_value = "PRIVATE_ENV_VALUE_SENTINEL"
    child_env = {
        name: value
        for name, value in os.environ.items()
        if not (
            name in git_snapshot._SNAPSHOT_GIT_REDIRECTS  # noqa: SLF001 - boundary fixture
            or name.startswith("GIT_TRACE")
            or name.startswith("GIT_CONFIG")
            or name == "GIT_EXEC_PATH"
        )
    }
    child_env[newline_name] = private_value
    child_env[oversized_name] = private_value
    script = """
import sys
from pathlib import Path
from daydream import git_ops
try:
    git_ops.prepare_independent_snapshot(Path(sys.argv[1]), Path(sys.argv[2]), include_untracked=False)
except git_ops.SnapshotPreparationError as exc:
    print(exc)
    raise SystemExit(7)
raise SystemExit(9)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(repo), str(destination)], check=False, capture_output=True, text=True,
        env=child_env,
    )
    assert proc.returncode == 7
    assert "2 invalid Git variable names" in proc.stdout
    assert newline_name not in proc.stdout
    assert oversized_name not in proc.stdout
    assert private_value not in proc.stdout
    assert "SECRET_NAME" not in proc.stdout
    assert len(proc.stdout) < 500
    assert proc.stderr == ""
    assert not destination.exists()

def test_independent_snapshot_bounds_git_environment_name_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      repo: Path,
) -> None:
    for index in range(30):
        monkeypatch.setenv(f"GIT_TRACE_TEST_{index:02d}", "PRIVATE")
    with pytest.raises(git_ops.SnapshotPreparationError) as raised:
        git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    message = str(raised.value)
    assert "GIT_TRACE_TEST_00" in message
    assert "GIT_TRACE_TEST_15" in message
    assert "GIT_TRACE_TEST_16" not in message
    assert "and 14 more" in message
    assert "PRIVATE" not in message
    assert len(message) < 500

def test_independent_snapshot_known_git_violation_skips_exec_path_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                        repo: Path,
) -> None:
    trace_file = tmp_path / "git-trace-side-effect.log"
    monkeypatch.setenv("GIT_TRACE", str(trace_file))
    monkeypatch.setenv("GIT_EXEC_PATH", str(tmp_path / "invalid git exec path"))
    with pytest.raises(git_ops.SnapshotPreparationError) as raised:
        git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    message = str(raised.value)
    assert "GIT_TRACE" in message
    assert str(trace_file) not in message
    assert not trace_file.exists()

def test_independent_snapshot_redacts_invalid_utf8_exec_path_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   repo: Path,
) -> None:
    private_exec_path = str(tmp_path / "PRIVATE_EXEC_PATH_SENTINEL")
    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "exec path git shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "if sys.argv[1:] == ['--exec-path']:\n"
        "    os.write(1, b'\\xff\\xfe\\n')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(subprocess.run([os.environ['DAYDREAM_TEST_REAL_GIT'], *sys.argv[1:]]).returncode)\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("DAYDREAM_TEST_REAL_GIT", real_git)
    monkeypatch.setenv("PATH", str(shim_dir))
    monkeypatch.setenv("GIT_EXEC_PATH", private_exec_path)
    with pytest.raises(git_ops.SnapshotPreparationError) as raised:
        git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    message = str(raised.value)
    assert "GIT_EXEC_PATH" in message
    assert private_exec_path not in message
    assert "\\xff" not in message
    assert len(message) < 500

def test_independent_snapshot_strict_queries_reject_broken_repository(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        git_ops.list_remotes(tmp_path, strict=True)
    with pytest.raises(GitError):
        git_ops.object_alternates(tmp_path, strict=True)
    with pytest.raises(GitError):
        git_ops.git_common_dir(tmp_path)
    with pytest.raises(GitError):
        git_ops.symbolic_head(tmp_path, strict=True)
    with pytest.raises(GitError):
        git_ops.ls_tree_files(tmp_path, "HEAD", strict=True)
    assert git_ops.list_remotes(tmp_path) == []
    assert git_ops.object_alternates(tmp_path) == ()
    assert git_ops.ls_tree_files(tmp_path, "HEAD") == []

@pytest.mark.parametrize("key", [
    "core.worktree", "core.hooksPath", "include.path", "filter.private.clean",
    "remote.origin.uploadpack", "init.templateDir", "extensions.refStorage",
])
def test_independent_snapshot_rejects_indexed_config_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str,
                                                               repo: Path,
) -> None:
    destination = tmp_path / "snapshot"
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{count}", key)
    monkeypatch.setenv(f"GIT_CONFIG_VALUE_{count}", str(tmp_path / "external"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(count + 1))
    before = dict(os.environ)
    with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
        git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
    assert "GIT_CONFIG_COUNT" in str(raised.value)
    assert f"GIT_CONFIG_KEY_{count}" in str(raised.value)
    assert f"GIT_CONFIG_VALUE_{count}" in str(raised.value)
    assert str(tmp_path / "external") not in str(raised.value)
    assert not destination.exists()
    assert dict(os.environ) == before

@pytest.mark.parametrize("signing", ["true", "false"])
def test_independent_snapshot_preserves_safe_config_and_independent_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signing: str,
                                                                            repo: Path,
) -> None:
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{count}", "commit.gpgsign")
    monkeypatch.setenv(f"GIT_CONFIG_VALUE_{count}", signing)
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(count + 1))
    before = dict(os.environ)
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    assert dict(os.environ) == before
    assert _git(snapshot.repo, "config", "--get", "commit.gpgsign") == signing
    assert _git(snapshot.repo, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    _git(snapshot.repo, "fsck", "--full", "--no-dangling")
    assert any((snapshot.repo / ".git" / "objects").rglob("*.pack"))
    assert git_ops.object_alternates(snapshot.repo, strict=True) == ()

def test_independent_snapshot_accepts_git_exported_default_exec_path_in_pre_push(tmp_path: Path) -> None:
    repo, remote = _repo_with_origin(tmp_path)
    destination = tmp_path / "snapshot"
    observed = tmp_path / "hook-exec-path"
    probe = tmp_path / "prepare-in-hook.py"
    project_root = Path(__file__).resolve().parents[1]
    probe.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
        "from pathlib import Path\n"
        "from daydream import git_ops\n"
        "before = dict(os.environ)\n"
        f"source = Path({str(repo)!r})\n"
        f"destination = Path({str(destination)!r})\n"
        "git_ops.prepare_independent_snapshot(\n"
        "    source, destination, include_untracked=False,\n"
        ")\n"
        "if dict(os.environ) != before:\n"
        "    raise RuntimeError('snapshot preparation changed the hook environment')\n"
        f"Path({str(observed)!r}).write_text(os.environ['GIT_EXEC_PATH'], encoding='utf-8')\n",
        encoding="utf-8",
    )
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "while IFS= read -r git_env_var; do\n"
        "    unset \"$git_env_var\"\n"
        "done < <(git rev-parse --local-env-vars)\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(probe))}\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    before = dict(os.environ)
    push_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_CONFIG")
    }
    clean_exec_env = dict(push_env)
    clean_exec_env.pop("GIT_EXEC_PATH", None)
    expected_exec_path = subprocess.run(
        ["git", "--exec-path"], cwd=repo, capture_output=True, text=True, check=True, env=clean_exec_env,
    ).stdout.removesuffix("\n")
    pushed = subprocess.run(
        ["git", "push", "origin", "main"], cwd=repo, capture_output=True, text=True, check=False, env=push_env,
    )
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    assert dict(os.environ) == before
    assert observed.read_text(encoding="utf-8") == expected_exec_path
    assert _git(destination, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    _git(destination, "fsck", "--full", "--no-dangling")
    assert git_ops.object_alternates(destination, strict=True) == ()

def test_independent_snapshot_rejects_exec_path_when_default_query_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         repo: Path,
) -> None:
    destination = tmp_path / "snapshot"
    monkeypatch.setenv("GIT_EXEC_PATH", "/trusted-looking/git-core")
    monkeypatch.setenv("PATH", str(tmp_path / "missing-bin"))
    before = dict(os.environ)
    with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git"):
        git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
    assert not destination.exists()
    assert dict(os.environ) == before

def test_independent_snapshot_unborn_detection_rejects_corrupt_head(repo: Path) -> None:
    (repo / ".git" / "HEAD").write_text("invalid head contents\n")
    with pytest.raises(GitError):
        git_ops.is_unborn_head(repo)

def test_independent_snapshot_symbolic_head_is_not_ambiguous_with_tag(tmp_path: Path, repo: Path) -> None:
    _git(repo, "tag", "main")
    assert git_ops.symbolic_head(repo, strict=True) == "main"
    assert not git_ops.is_unborn_head(repo)
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    assert _git(snapshot.repo, "for-each-ref", "--format=%(refname)", "refs/heads") == _git(
        repo, "for-each-ref", "--format=%(refname)", "refs/heads",
    )

def test_clone_no_local_requests_independent_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    def record_clone(remote_url: str, cmd: list[str], timeout: int) -> None:
        calls.append(cmd)
    monkeypatch.setattr(git_process, "_run_clone", record_clone)
    git_ops.clone("source", tmp_path / "snapshot", no_local=True)
    assert calls == [["git", "clone", "--no-local", "source", str(tmp_path / "snapshot")]]

@pytest.mark.parametrize("name", [" leading and trailing ", "line\nbreak", "café.py"])
def test_diff_name_only_strict_preserves_exact_git_paths(repo: Path, name: str) -> None:
    write_and_stage(repo, name, b"content\n")
    _commit(repo, "unusual path")
    assert git_ops.diff_name_only_strict(repo, "HEAD^", "HEAD") == [name]
    assert git_ops.diff_name_only_strict(repo, "HEAD", "HEAD") == []

def test_assert_is_worktree_passes_for_real_repo(repo: Path) -> None:
    git_ops.assert_is_worktree(repo)
    assert git_ops.is_inside_worktree(repo) is True

def test_assert_is_worktree_rejects_non_repo(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path)
    assert git_ops.is_inside_worktree(tmp_path) is False

def test_assert_is_worktree_rejects_org_dir(tmp_path: Path) -> None:
    org = tmp_path / "org"
    org.mkdir()
    _make_repo_with_main(org, name="child-repo")
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(org)
    assert git_ops.is_inside_worktree(org) is False

def test_assert_is_worktree_rejects_subdir_of_repo(repo: Path) -> None:
    sub = repo / "src"
    sub.mkdir()
    (sub / "x.txt").write_text("x\n")
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(sub)
    assert git_ops.is_inside_worktree(sub) is False

def test_assert_is_worktree_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path / "does-not-exist")

def test_head_sha_returns_full_sha(repo: Path) -> None:
    expected = _git(repo, "rev-parse", "HEAD")
    assert git_ops.head_sha(repo) == expected
    assert len(expected) == 40

def test_head_sha_raises_on_empty_repo(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    with pytest.raises(GitError):
        git_ops.head_sha(repo)

def test_current_branch_on_named_branch(repo: Path) -> None:
    assert git_ops.current_branch(repo) == "main"

def test_current_branch_returns_none_when_detached(repo: Path) -> None:
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", sha)
    assert git_ops.current_branch(repo) is None

def test_default_branch_uses_origin_head(tmp_path: Path) -> None:
    repo, bare = _repo_with_origin(tmp_path)
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "remote", "set-head", "origin", "main")
    assert git_ops.default_branch(repo) == "main"

def test_default_branch_falls_back_to_main(repo: Path) -> None:
    assert git_ops.default_branch(repo) == "main"

def test_list_local_branches_maps_names_to_oids(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path, name="list_branches")
    _git(repo, "checkout", "-b", "feat/slash-name")
    write_and_stage(repo, "feature.txt", "feature\n")
    _commit(repo, "on feature")
    _git(repo, "checkout", "main")
    branches = git_ops.list_local_branches(repo)
    assert set(branches) == {"main", "feat/slash-name"}
    assert branches["main"] == git_ops.head_sha(repo)  # main is checked out here
    assert branches["feat/slash-name"] == _git(repo, "rev-parse", "feat/slash-name")

def test_list_local_branches_raises_on_failure(tmp_path: Path) -> None:
    with pytest.raises(git_ops.GitError):
        git_ops.list_local_branches(tmp_path / "not-a-repo")

@pytest.mark.parametrize(
    ("init_branch", "expected"),
    [
        ("master", "master"),  # falls back to local master when no main/origin
        ("trunk", BranchNotFoundError),  # neither main nor master present -> raises
    ], ids=["falls_back_to_master", "raises_when_none_present"],
)
def test_default_branch_fallback(tmp_path: Path, init_branch: str, expected: object) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", init_branch)
    _configure_identity(repo)
    write_and_stage(repo, "f.txt", "hi\n")
    _commit(repo, "first")
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            git_ops.default_branch(repo)
    else:
        assert git_ops.default_branch(repo) == expected

def test_remote_url_returns_url_when_remote_configured(tmp_path: Path) -> None:
    repo, bare = _repo_with_origin(tmp_path)
    assert git_ops.remote_url(repo) == str(bare)

def test_remote_url_returns_none_when_remote_missing(repo: Path) -> None:
    assert git_ops.remote_url(repo) is None

def test_remote_url_returns_none_for_unknown_remote_name(tmp_path: Path) -> None:
    repo, bare = _repo_with_origin(tmp_path)
    assert git_ops.remote_url(repo, "upstream") is None

def test_branch_exists_local(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feat-local")
    assert git_ops.branch_exists(repo, "feat-local") is True

def test_branch_exists_missing(repo: Path) -> None:
    assert git_ops.branch_exists(repo, "nonexistent") is False

def _ref_raw_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")

def _ref_abbreviated_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "--short", "HEAD")

def _ref_tag(repo: Path) -> str:
    _git(repo, "tag", "v1.0")
    return "v1.0"

def _ref_relative_commit_ish(repo: Path) -> str:
    write_and_stage(repo, "two.txt", "two\n")
    _commit(repo, "second")
    return "HEAD~1"

def _ref_named_branch(repo: Path) -> str:
    _git(repo, "checkout", "-b", "feat-local")
    return "feat-local"

@pytest.mark.parametrize("probe", [git_ops.ref_exists, git_ops.commit_exists], ids=["ref_exists", "commit_exists"])
@pytest.mark.parametrize(
    "build_ref", [_ref_raw_sha, _ref_abbreviated_sha, _ref_tag, _ref_relative_commit_ish, _ref_named_branch],
    ids=["raw_sha", "abbreviated_sha", "tag", "relative_commit_ish", "named_branch"],
)
def test_ref_and_commit_exist_true(repo: Path, build_ref: Any, probe: Any) -> None:
    assert probe(repo, build_ref(repo)) is True

def test_ref_exists_missing(repo: Path) -> None:
    assert git_ops.ref_exists(repo, "nonexistent") is False
    # A tree-ish that is not a commit must be rejected, not accepted.
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert git_ops.ref_exists(repo, tree) is False

def test_ref_exists_rejects_leading_dash(repo: Path) -> None:
    assert git_ops.ref_exists(repo, "-not-a-ref") is False
    assert git_ops.ref_exists(repo, "--exec=evil") is False

def test_commit_exists_rejects_origin_only(tmp_path: Path) -> None:
    """Remote-only names satisfy branch lookup but cannot resolve as local commit-ish names."""
    repo, bare = _repo_with_origin(tmp_path)
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "checkout", "-b", "remote-only")
    write_and_stage(repo, "r.txt", "r\n")
    _commit(repo, "remote-only commit")
    _git(repo, "push", "-u", "origin", "remote-only")
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "remote-only")
    assert git_ops.branch_exists(repo, "remote-only") is True
    assert git_ops.commit_exists(repo, "remote-only") is False

def test_commit_exists_missing(repo: Path) -> None:
    assert git_ops.commit_exists(repo, "nonexistent") is False
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert git_ops.commit_exists(repo, tree) is False

def test_commit_exists_rejects_leading_dash(repo: Path) -> None:
    assert git_ops.commit_exists(repo, "-not-a-ref") is False

@pytest.mark.parametrize(
    ("ancestor", "descendant", "expected"),
    [
        pytest.param("HEAD~1", "HEAD", True, id="ancestor_of_head"),
        pytest.param("HEAD", "HEAD~1", False, id="reversed_is_not_ancestor"),
        pytest.param("missing", "HEAD", False, id="missing_ref"),
        pytest.param("-not-a-ref", "HEAD", False, id="leading_dash_ref"),
    ],
)
def test_is_ancestor_reports_relationship(repo: Path, ancestor: str, descendant: str, expected: bool) -> None:
    write_and_stage(repo, "second.txt", "second\n")
    _commit(repo, "second")
    assert git_ops.is_ancestor(repo, ancestor, descendant) is expected

def test_merge_base_returns_shared_commit(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feat")
    write_and_stage(repo, "feat.txt", "feat\n")
    _commit(repo, "feat commit")
    _git(repo, "checkout", "main")
    write_and_stage(repo, "main2.txt", "more main\n")
    _commit(repo, "main commit")
    _git(repo, "checkout", "feat")
    expected = _git(repo, "merge-base", "HEAD", "main")
    assert git_ops.merge_base(repo, "main") == expected

def test_merge_base_prefers_upstream_when_remote_ahead(tmp_path: Path) -> None:
    repo, bare = _repo_with_origin(tmp_path)
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "checkout", "-b", "feature")
    write_and_stage(repo, "feature.txt", "feature\n")
    _commit(repo, "feature commit")
    # Rewrite local main as an unrelated history; track origin/main so the
    # upstream is "ahead" of the rewritten local main.
    _git(repo, "checkout", "--orphan", "rewrite")
    _git(repo, "rm", "-rf", ".")
    write_and_stage(repo, "new-main.txt", "rewritten\n")
    _commit(repo, "rewrite main")
    _git(repo, "branch", "-M", "rewrite", "main")
    _git(repo, "branch", "--set-upstream-to=origin/main", "main")
    _git(repo, "checkout", "feature")
    _git(repo, "fetch", "origin")
    expected = _git(repo, "merge-base", "HEAD", "origin/main")
    assert git_ops.merge_base(repo, "main") == expected

def test_merge_base_returns_none_for_missing_branch(repo: Path) -> None:
    assert git_ops.merge_base(repo, "missing-branch") is None

def test_merge_base_returns_none_when_head_missing(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    assert git_ops.merge_base(repo, "main") is None

@pytest.mark.parametrize(
    "refs", [pytest.param(("-no-such-ref",), id="base"), pytest.param(("main", "-no-such-ref"), id="head")],
)
def test_merge_base_returns_none_for_leading_dash_ref(repo: Path, refs: tuple[str, ...]) -> None:
    assert git_ops.merge_base(repo, *refs) is None

def test_remote_urls_is_strict_and_allows_no_remotes(repo: Path) -> None:
    assert git_ops.remote_urls(repo) == {}
    _git(repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    _git(repo, "remote", "add", "upstream", "git@github.com:other/widgets.git")
    assert git_ops.remote_urls(repo) == {
        "origin": "https://github.com/acme/widgets.git", "upstream": "git@github.com:other/widgets.git",
    }
    _git(repo, "config", "remote.broken.fetch", "+refs/heads/*:refs/remotes/broken/*")
    with pytest.raises(GitError, match="remote 'broken'.*fetch URL"):
        git_ops.remote_urls(repo)

def test_resolve_pr_merge_base_uses_local_branch_without_remotes(repo: Path) -> None:
    base = git_ops.head_sha(repo)
    _git(repo, "checkout", "-b", "feature")
    write_and_stage(repo, "feature.txt", "feature\n")
    head = _commit(repo, "feature")
    assert git_ops.resolve_pr_merge_base(repo, [], "refs/heads/main", head) == base

def test_resolve_pr_merge_base_prefers_present_remote_over_stale_local(tmp_path: Path) -> None:
    repo, remote = _repo_with_origin(tmp_path)
    stale_local = git_ops.head_sha(repo)
    _git(repo, "push", "-u", "origin", "main")
    write_and_stage(repo, "base-update.txt", "new base\n")
    remote_base = _commit(repo, "base update")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "-b", "feature")
    write_and_stage(repo, "feature.txt", "feature\n")
    head = _commit(repo, "feature")
    _git(repo, "branch", "-f", "main", stale_local)
    assert git_ops.resolve_pr_merge_base(repo, ["refs/remotes/origin/main"], "refs/heads/main", head) == remote_base

def test_resolve_pr_merge_base_rejects_divergent_matching_remotes(repo: Path) -> None:
    oldest = git_ops.head_sha(repo)
    write_and_stage(repo, "base-update.txt", "new base\n")
    newer = _commit(repo, "base update")
    _git(repo, "checkout", "-b", "feature")
    write_and_stage(repo, "feature.txt", "feature\n")
    head = _commit(repo, "feature")
    _git(repo, "update-ref", "refs/remotes/one/main", newer)
    _git(repo, "update-ref", "refs/remotes/two/main", oldest)
    with pytest.raises(GitError, match="fetch/align the base remote"):
        git_ops.resolve_pr_merge_base(repo, ["refs/remotes/one/main", "refs/remotes/two/main"], "refs/heads/main", head)

@pytest.mark.parametrize("head", ["HEAD", "deadbeef", "f" * 40])
def test_resolve_pr_merge_base_requires_exact_present_head(repo: Path, head: str) -> None:
    with pytest.raises(GitError, match="exact PR head"):
        git_ops.resolve_pr_merge_base(repo, [], "refs/heads/main", head)

@pytest.mark.parametrize("local_ref", ["main", "refs/heads/-bad", "refs/heads/main~1"])
def test_resolve_pr_merge_base_rejects_invalid_base_ref(repo: Path, local_ref: str) -> None:
    with pytest.raises(GitError, match="base ref"):
        git_ops.resolve_pr_merge_base(repo, [], local_ref, git_ops.head_sha(repo))

def test_diff_returns_changes(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    write_and_stage(repo, "added.txt", "hello\n")
    _commit(repo, "topic commit")
    out = git_ops.diff(repo, "main")
    assert "added.txt" in out
    assert "+hello" in out

def test_diff_includes_staged_and_unstaged_worktree_changes(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    write_and_stage(repo, "staged.txt", "staged\n")
    (repo / "base.txt").write_text("unstaged\n")
    out = git_ops.diff(repo, "main")
    assert "staged.txt" in out
    assert "-base" in out
    assert "+unstaged" in out

def test_diff_excludes_paths(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    _commit(repo, "topic")
    out = git_ops.diff(repo, "main", exclude=["drop.txt"])
    assert "keep.txt" in out
    assert "drop.txt" not in out

def test_diff_prefers_origin_when_on_default_branch(tmp_path: Path, repo: Path) -> None:
    remote_dir = tmp_path / "remote.git"
    remote_dir.mkdir()
    _git(remote_dir, "init", "--bare", "-b", "main")
    _git(repo, "remote", "add", "origin", str(remote_dir))
    _git(repo, "push", "-u", "origin", "main")
    # Local commit on main — not pushed.
    write_and_stage(repo, "local.txt", "local change\n")
    _commit(repo, "local only")
    out = git_ops.diff(repo, "main")
    assert "local.txt" in out, "diff should show unpushed changes vs origin/main"

def test_diff_name_only_returns_changed_files(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    write_and_stage(repo, "added.txt", "hello\n")
    _commit(repo, "add file")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert result == ["added.txt"]

def test_diff_name_only_returns_multiple_files_in_order(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    (repo / "alpha.txt").write_text("a\n")
    (repo / "beta.txt").write_text("b\n")
    _git(repo, "add", "alpha.txt", "beta.txt")
    _commit(repo, "add two files")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert result == ["alpha.txt", "beta.txt"]

def test_diff_name_only_returns_empty_list_on_bad_ref(repo: Path) -> None:
    result = git_ops.diff_name_only(repo, "nonexistent-ref", "HEAD")
    assert result == []

def test_changed_files_against_compares_tracked_changes_to_snapshot(repo: Path) -> None:
    """A pre-fix snapshot, rather than HEAD, is the guard's tracked baseline."""
    tracked = repo / "tracked.txt"
    tracked.write_text("committed\n")
    _git(repo, "add", "tracked.txt")
    _commit(repo, "add tracked file")
    tracked.write_text("pre-fix edit\n")
    snapshot = git_ops.stash_create(repo)
    assert snapshot is not None
    tracked.write_text("post-fix edit\n")
    assert git_ops.changed_files_against(repo, snapshot) == ["tracked.txt"]

def test_changed_files_against_raises_when_git_query_fails(tmp_path: Path) -> None:
    """Safety guards must not mistake a failed enumeration for a clean tree."""
    with pytest.raises(GitError):
        git_ops.changed_files_against(tmp_path, "HEAD")

def test_log_returns_oneline_commits(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    write_and_stage(repo, "a.txt", "a\n")
    _commit(repo, "topic-msg")
    out = git_ops.log(repo, "main")
    assert "topic-msg" in out

def test_show_returns_file_bytes_at_ref(repo: Path) -> None:
    out = git_ops.show(repo, "HEAD", "base.txt")
    assert out == b"base\n"

def test_show_raises_on_missing_path(repo: Path) -> None:
    with pytest.raises(git_ops.PathAbsentError):
        git_ops.show(repo, "HEAD", "nope.txt")

def test_show_raises_a_plain_error_when_the_object_store_is_damaged(repo: Path) -> None:
    """Absence recognition is positive-only: an unreadable blob is not a missing path."""
    blob = _git(repo, "rev-parse", "HEAD:base.txt").strip()
    (repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    with pytest.raises(GitError) as excinfo:
        git_ops.show(repo, "HEAD", "base.txt")
    assert not isinstance(excinfo.value, git_ops.PathAbsentError)

def test_grep_fixed_matches_returns_path_pattern_pairs(repo: Path) -> None:
    (repo / "widget_user.py").write_text("import widget\n")
    (repo / "gadget_user.ts").write_text("gadget\n")
    (repo / "partial.py").write_text("widget_factory\n")
    (repo / "notes.md").write_text("widget gadget\n")
    _git(repo, "add", "widget_user.py", "gadget_user.ts", "partial.py", "notes.md")
    _commit(repo, "add files")
    matches = git_ops.grep_fixed_matches(repo, ("widget", "gadget"), word=True, pathspecs=("*.py", "*.ts"))
    assert len(matches) == 2
    assert set(matches) == {("widget_user.py", "widget"), ("gadget_user.ts", "gadget")}

def test_grep_fixed_matches_raises_on_nonzero_exit(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(args=["git"], returncode=2, stdout=b"", stderr=b"fatal: unknown option")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GitError, match="git grep -F -o -z -f failed"):
        git_ops.grep_fixed_matches(repo, ("widget",))

def test_grep_fixed_matches_raises_on_malformed_record(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A line without a NUL separator produces parts with len != 2.
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=b"file.py\n", stderr=b"")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GitError, match="malformed record"):
        git_ops.grep_fixed_matches(repo, ("widget",))

def test_grep_fixed_matches_empty_when_no_matches(repo: Path) -> None:
    """Exit code 1 ("no matches") is treated as success: an empty list."""
    write_and_stage(repo, "widget.py", "widget\n")
    _commit(repo, "add widget")
    assert git_ops.grep_fixed_matches(repo, ("absent_pattern",)) == []

def test_grep_fixed_matches_dedups_and_skips_nul_cr_lf_patterns(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_and_stage(repo, "widget.py", "widget\n")
    _commit(repo, "add widget")
    patterns_files: list[bytes] = []
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        argv = args[0] if args else []
        for index, arg in enumerate(argv):
            if arg == "-f" and index + 1 < len(argv):
                patterns_files.append(Path(argv[index + 1]).read_bytes())
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(subprocess, "run", fake_run)
    matches = git_ops.grep_fixed_matches(
        repo,
        ("widget", "widget", "", "gadget", "bad\x00pattern", "bad\rpattern", "bad\npattern"),
        word=True,
    )
    assert matches == []
    assert patterns_files == [b"widget\ngadget"]

def test_grep_fixed_matches_empty_when_all_patterns_unsuitable(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_and_stage(repo, "widget.py", "widget\n")
    _commit(repo, "add widget")
    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("git must not run when every pattern is unsuitable")
    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_ops.grep_fixed_matches(repo, ("", "a\x00b", "a\rb", "a\nb")) == []

def test_grep_fixed_matches_default_word_false_matches_substrings(repo: Path) -> None:
    write_and_stage(repo, "app.py", "application\n")
    _commit(repo, "add app")
    assert git_ops.grep_fixed_matches(repo, ("app",)) == [("app.py", "app")]

def test_status_porcelain_clean_and_dirty(repo: Path) -> None:
    assert git_ops.status_porcelain(repo) == ""
    (repo / "untracked.txt").write_text("u\n")
    out = git_ops.status_porcelain(repo)
    assert "untracked.txt" in out

def test_upstream_ahead_count_no_upstream(repo: Path) -> None:
    assert git_ops.upstream_ahead_count(repo, "main") == 0

def test_upstream_ahead_count_when_remote_ahead(tmp_path: Path) -> None:
    repo, bare = _repo_with_origin(tmp_path)
    _git(repo, "push", "-u", "origin", "main")
    # Push two extra commits to origin/main via a sidecar clone, then fetch.
    sidecar = tmp_path / "sidecar"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    write_and_stage(sidecar, "x.txt", "x\n")
    _commit(sidecar, "x")
    write_and_stage(sidecar, "y.txt", "y\n")
    _commit(sidecar, "y")
    _git(sidecar, "push", "origin", "main")
    _git(repo, "fetch", "origin")
    assert git_ops.upstream_ahead_count(repo, "main") == 2

def test_fetch_pulls_new_commits(tmp_path: Path) -> None:
    repo, bare = _repo_with_origin(tmp_path)
    _git(repo, "push", "-u", "origin", "main")
    sidecar = tmp_path / "sidecar"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    write_and_stage(sidecar, "z.txt", "z\n")
    new_sha = _commit(sidecar, "z")
    _git(sidecar, "push", "origin", "main")
    git_ops.fetch(repo)
    assert _git(repo, "rev-parse", "origin/main") == new_sha

def test_worktree_add_and_remove_round_trip(tmp_path: Path, repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    wt = tmp_path / "wt"
    git_ops.worktree_add(repo, wt, head)
    assert wt.exists()
    assert (wt / "base.txt").read_text() == "base\n"
    git_ops.assert_is_worktree(wt)
    git_ops.worktree_remove(repo, wt)
    assert not wt.exists()

def test_worktree_move_preserves_registered_worktree_with_spaces(tmp_path: Path, repo: Path) -> None:
    source = tmp_path / "legacy worktree"
    destination = tmp_path / "private workspaces" / "moved worktree"
    destination.parent.mkdir()
    git_ops.worktree_add(repo, source, "main", detach=True)
    git_ops.worktree_move(repo, source, destination)
    assert not source.exists()
    assert destination.is_dir()
    assert git_ops.git_common_dir(destination) == git_ops.git_common_dir(repo)
    porcelain = _git(repo, "worktree", "list", "--porcelain")
    assert str(destination) in porcelain
    assert str(source) not in porcelain

def test_worktree_move_propagates_git_failure(tmp_path: Path, repo: Path) -> None:
    missing = tmp_path / "missing worktree"
    destination = tmp_path / "destination"
    with pytest.raises(GitError, match="git worktree move"):
        git_ops.worktree_move(repo, missing, destination)
    assert not destination.exists()

def test_commit_paths_on_new_branch_pushes_to_origin(repo_with_origin: Path) -> None:
    (repo_with_origin / ".github/workflows").mkdir(parents=True)
    (repo_with_origin / ".github/workflows/daydream-review.yml").write_text("name: x\n")
    git_ops.create_branch(repo_with_origin, "daydream/setup")
    git_ops.commit_paths(repo_with_origin, [Path(".github/workflows/daydream-review.yml")], "add bot workflows")
    git_ops.push_branch(repo_with_origin, "daydream/setup")
    assert git_ops.ref_exists(repo_with_origin, "origin/daydream/setup")

def test_create_branch_raises_when_branch_exists(repo_with_origin: Path) -> None:
    git_ops.create_branch(repo_with_origin, "daydream/dup")
    _git(repo_with_origin, "checkout", "main")
    with pytest.raises(GitError):
        git_ops.create_branch(repo_with_origin, "daydream/dup")

def test_commit_paths_commits_only_named_paths(repo_with_origin: Path) -> None:
    git_ops.create_branch(repo_with_origin, "daydream/selective")
    (repo_with_origin / "tracked.txt").write_text("staged\n")
    (repo_with_origin / "untouched.txt").write_text("left behind\n")
    git_ops.commit_paths(repo_with_origin, [Path("tracked.txt")], "add tracked only")
    committed = _git(repo_with_origin, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["tracked.txt"]
    assert "untouched.txt" in _git(repo_with_origin, "status", "--porcelain")

def test_stage_paths_stages_only_named_paths(repo_with_origin: Path) -> None:
    git_ops.create_branch(repo_with_origin, "daydream/stage")
    (repo_with_origin / "a.txt").write_text("a\n")
    (repo_with_origin / "b.txt").write_text("b\n")
    git_ops.stage_paths(repo_with_origin, [Path("a.txt")])
    staged = _git(repo_with_origin, "diff", "--cached", "--name-only").split()
    assert staged == ["a.txt"]
    assert "b.txt" in _git(repo_with_origin, "status", "--porcelain")

def test_push_branch_failure_raises_git_error(git_repo: Path) -> None:
    git_ops.create_branch(git_repo, "daydream/no-remote")
    (git_repo / "f.txt").write_text("x\n")
    git_ops.commit_paths(git_repo, [Path("f.txt")], "add f")
    with pytest.raises(GitError):
        git_ops.push_branch(git_repo, "daydream/no-remote")

def test_error_hierarchy_is_consistent() -> None:
    assert issubclass(NotAWorktreeError, GitError)
    assert issubclass(BranchNotFoundError, GitError)
    assert issubclass(WrongBranchError, GitError)
    assert issubclass(git_ops.GitTimeoutError, GitError)

def _timeout_run(*, cmd: list[str], timeout: float, calls: dict[str, int] | None = None) -> Any:
    """Return a ``subprocess.run`` double that raises ``TimeoutExpired`` on every call."""
    def run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
    return run

def _flaky_timeout_run(
    calls: dict[str, int], ok: subprocess.CompletedProcess[Any], *, cmd: list[str], timeout: float,
) -> Any:
    """Return a double that times out on the first call and returns *ok* after."""
    def run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        return ok
    return run

def test_run_git_timeout_retry_behavior(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    ok = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="true\n", stderr="")
    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _flaky_timeout_run(calls, ok, cmd=["git"], timeout=5))
    assert git_process._run_git(repo, ["rev-parse", "HEAD"]).returncode == 0
    assert calls["n"] == 2  # timed out once, then succeeded
    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["git"], timeout=5, calls=calls))
    with pytest.raises(git_ops.GitTimeoutError):
        git_process._run_git(repo, ["rev-parse", "HEAD"], retries=2)
    assert calls["n"] == 3  # 1 initial + 2 retries
    # Case 3: non-timeout failure raises plain GitError immediately (no retry).
    calls["n"] = 0
    def os_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise OSError("git binary missing")
    monkeypatch.setattr(subprocess, "run", os_error)
    with pytest.raises(GitError) as exc:
        git_process._run_git(repo, ["rev-parse", "HEAD"], retries=2)
    assert not isinstance(exc.value, git_ops.GitTimeoutError)
    assert calls["n"] == 1  # no retries for non-timeout failures

def test_run_git_encodes_input_text_when_capturing_bytes(repo: Path) -> None:
    """Binary capture requires encoded input; a raw TypeError would mask Git failures."""
    bytes_proc = git_process._run_git(repo, ["hash-object", "--stdin"], capture_bytes=True, input_text="abc\n")
    text_proc = git_process._run_git(repo, ["hash-object", "--stdin"], input_text="abc\n")
    assert bytes_proc.returncode == 0
    assert bytes_proc.stdout == text_proc.stdout.encode()

@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda repo: git_ops.fetch(repo), id="git-fetch"),
        pytest.param(
            lambda repo: git_ops.gh_pr_create(repo, head="feature", base="main", title="t", body="b"),
            id="gh-pr-create",
        ),
        pytest.param(
            lambda repo: git_ops.gh_issue_create(repo, title="t", body="b", repo_slug="octocat/hello"),
            id="gh-issue-create",
        ),
    ],
)
def test_mutating_wrapper_does_not_retry_on_timeout(
    monkeypatch: pytest.MonkeyPatch, repo: Path, operation: Any,
) -> None:
    """A timed-out mutation may have completed, so retry could duplicate changes or PRs."""
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["command"], timeout=60, calls=calls))
    with pytest.raises(git_ops.GitTimeoutError):
        operation(repo)
    assert calls["n"] == 1  # no retries for mutating operations

@pytest.mark.parametrize(
    ("env_value", "expected_timeout", "expected_warning"),
    [
        ("6O", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='6O' is not a valid integer; using default 60"),
        ("-5", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='-5' must be positive; using default 60"),
        ("0", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='0' must be positive; using default 60"), ("7", 7, None),
    ], ids=["malformed", "negative", "zero", "valid"],
)
def test_run_gh_timeout_environment_validation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, repo: Path, env_value: str,
    expected_timeout: int, expected_warning: str | None,
) -> None:
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_SECONDS", env_value)
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["gh"], timeout=expected_timeout))
    with pytest.raises(git_ops.GitTimeoutError) as exc:
        git_process._run_gh(repo, ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    assert str(exc.value) == (
        "gh repo view --json nameWithOwner -q .nameWithOwner "
        f"timed out after {expected_timeout}s"
    )
    if expected_warning is None:
        assert "DAYDREAM_GH_TIMEOUT_SECONDS" not in caplog.text
    else:
        assert expected_warning in caplog.text

@pytest.mark.parametrize(
    ("env_value", "expected_attempts", "expected_warning"),
    [
        (None, 3, None), ("", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='' is not a valid integer; using default 2"),
        ("abc", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='abc' is not a valid integer; using default 2"),
        ("-1", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='-1' is negative; using default 2"), ("0", 1, None), ("1", 2, None),
    ], ids=["default", "empty-warns", "malformed-warns", "negative-warns", "zero-valid", "one-valid"],
)
def test_read_only_gh_retry_environment_validation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, repo: Path, env_value: str | None,
    expected_attempts: int, expected_warning: str | None,
) -> None:
    """Read-only ``gh`` retry budget is validated at call time; retry 0 stays a valid 1 attempt."""
    if env_value is not None:
        monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", env_value)
    else:
        monkeypatch.delenv("DAYDREAM_GH_TIMEOUT_RETRIES", raising=False)
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["gh"], timeout=60, calls=calls))
    with pytest.raises(git_ops.GitTimeoutError) as exc:
        git_ops.gh_repo_view(repo)
    assert calls["n"] == expected_attempts
    suffix = f" ({expected_attempts} attempts)" if expected_attempts > 1 else ""
    assert str(exc.value) == (
        "gh repo view --json nameWithOwner -q .nameWithOwner timed out after 60s" + suffix
    )
    if expected_warning is None:
        assert "DAYDREAM_GH_TIMEOUT_RETRIES" not in caplog.text
    else:
        assert expected_warning in caplog.text

def test_run_gh_read_wrapper_retries_then_succeeds_and_exhausts(
    monkeypatch: pytest.MonkeyPatch, repo: Path,
) -> None:
    ok = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="octocat/hello\n", stderr="")
    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _flaky_timeout_run(calls, ok, cmd=["gh"], timeout=60))
    assert git_ops.gh_repo_view(repo) == ("octocat", "hello")
    assert calls["n"] == 2  # timed out once, then succeeded
    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["gh"], timeout=60, calls=calls))
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_pr_diff(repo, 7)
    assert calls["n"] == git_process._gh_retries() + 1

def test_gh_api_retries_only_when_idempotent(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Caller-declared reads may retry; GraphQL POST mutations must execute once despite
    sharing the HTTP method.
    """
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["gh"], timeout=60, calls=calls))
    # Read: idempotent=True -> retried to exhaustion.
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(repo, "/user", idempotent=True)
    assert calls["n"] == git_process._gh_retries() + 1
    # Mutation: a GraphQL-shaped POST with a body, default flag -> no retry.
    calls["n"] = 0
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(repo, "graphql", method="POST", input_data={"query": "mutation { x }"})
    assert calls["n"] == 1

@pytest.mark.parametrize("labels", [None, ["daydream", "tech-debt"]], ids=["no-labels", "two-labels"])
def test_gh_issue_create_constructs_argv_with_body_file_and_labels(
    monkeypatch: pytest.MonkeyPatch, repo: Path, labels: list[str] | None,
) -> None:
    """Issue bodies travel through an unlinked file, never argv; labels remain optional."""
    captured: dict[str, Any] = {}
    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Snapshot argv + cwd + body-file contents at call time (the tempfile
        # is unlinked after `_run_gh` returns, matching real gh's read-then-exit).
        captured["argv"] = list(args)
        captured["cwd"] = kwargs.get("cwd")
        i_body = list(args).index("--body-file")
        captured["body"] = Path(args[i_body + 1]).read_text()
        return subprocess.CompletedProcess(
            args=list(args), returncode=0,
            stdout="https://github.com/octocat/hello/issues/42\n", stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)
    url = git_ops.gh_issue_create(
        repo, title="out-of-scope: refactor handler.py error path",
        body="evidence and rationale\nthat the fix loop overreached\n",
        repo_slug="octocat/hello", labels=labels,
    )
    assert url == "https://github.com/octocat/hello/issues/42"
    argv = captured["argv"]
    assert argv[:5] == ["gh", "issue", "create", "--repo", "octocat/hello"]
    i_title = argv.index("--title")
    assert argv[i_title + 1] == "out-of-scope: refactor handler.py error path"
    i_body = argv.index("--body-file")
    body_path = argv[i_body + 1]
    assert captured["body"] == "evidence and rationale\nthat the fix loop overreached\n"
    assert not Path(body_path).exists()
    assert "that the fix loop overreached" not in argv
    label_positions = [i for i, tok in enumerate(argv) if tok == "--label"]
    assert [argv[i + 1] for i in label_positions] == (labels or [])
    assert captured["cwd"] == repo

def test_gh_issue_create_raises_on_non_zero_exit(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    body_path: list[str] = []
    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Capture the body tempfile path while it still exists (the unlink-in-
        # finally runs after _run_gh returns).
        i_body = list(args).index("--body-file")
        body_path.append(args[i_body + 1])
        return subprocess.CompletedProcess(args=list(args), returncode=1, stdout="", stderr="gh: not authenticated\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GitError):
        git_ops.gh_issue_create(repo, title="t", body="b", repo_slug="octocat/hello")
    assert body_path and not Path(body_path[0]).exists()

def test_gh_issue_list_returns_parsed_rows(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Issue rows retain bodies so scope-finding deduplication can read their fingerprint markers."""
    rows = [
        {
            "number": 42, "title": "[daydream] out-of-scope finding: notes.txt",
            "body": "desc\n<!-- daydream-scope-finding: abc123 -->",
            "url": "https://github.com/octocat/hello/issues/42",
        }, {"number": 7, "title": "unrelated", "body": "", "url": ""},
    ]
    captured: dict[str, Any] = {}
    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        captured["argv"] = list(args)
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout=json.dumps(rows), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = git_ops.gh_issue_list(repo, search="out-of-scope", repo_slug="octocat/hello")
    assert result == rows
    argv = captured["argv"]
    assert argv[:3] == ["gh", "issue", "list"]
    assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
    assert "--json" in argv and "number,title,body,url" in argv
    assert "--search" in argv and argv[argv.index("--search") + 1] == "out-of-scope"
    assert "--repo" in argv and "octocat/hello" in argv

def test_gh_issue_list_returns_empty_on_failure(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Failed dedup lookup must allow filing the finding instead of dropping or blocking it."""
    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(args=list(args), returncode=1, stdout="", stderr="gh: not authenticated\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_ops.gh_issue_list(repo) == []

_gh_available = shutil.which("gh") is not None
gh_required = pytest.mark.skipif(not _gh_available, reason="gh CLI not installed")

@pytest.fixture
def local_only_gh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reach local remote discovery without depending on host authentication. These repositories have no remotes, so
    gh fails before making a request. The synthetic token only suppresses its earlier login/configuration gate."""
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "isolated-gh-config"))
    monkeypatch.setenv("GH_HOST", "github.com")
    monkeypatch.setenv("GH_TOKEN", "test-local-only-no-network")
    monkeypatch.delenv("GH_REPO", raising=False)

@gh_required
def test_gh_repo_view_returns_none_outside_github_repo(repo: Path) -> None:
    assert git_ops.gh_repo_view(repo) is None

@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda repo: git_ops.gh_pr_view(repo, 999999), id="gh-pr-view"),
        pytest.param(lambda repo: git_ops.gh_pr_list_for_branch(repo, "main"), id="gh-pr-list"),
    ],
)
@gh_required
@pytest.mark.usefixtures("local_only_gh")
def test_gh_pr_raises_without_remote(repo: Path, call: Any) -> None:
    with pytest.raises(GitError, match="no git remotes found"):
        call(repo)

@gh_required
def test_gh_pr_diff_raises_without_remote(repo: Path) -> None:
    with pytest.raises(GitError):
        git_ops.gh_pr_diff(repo, 1)

@gh_required
@pytest.mark.parametrize(
    ("mode", "guidance"),
    [
        ("local", "gh auth login"),
        ("automation", "GitHub CLI in automation"),
        ("actions", "GitHub CLI in a GitHub Actions workflow"),
    ],
    ids=["local", "automation", "actions"],
)
def test_gh_api_raises_without_auth(
    repo: Path, monkeypatch: pytest.MonkeyPatch, mode: str, guidance: str,
) -> None:
    """The installed CLI rejects credentials absent at the actual authentication guard."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if mode != "local":
        monkeypatch.setenv("CI", "true")
    if mode == "actions":
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(GitError, match=guidance) as error:
        git_ops.gh_api(repo, "/user")
    if mode != "local":
        assert "set the GH_TOKEN environment variable" in str(error.value)

def _make_divergent_history(tmp_path: Path) -> tuple[Path, str, str]:
    """Modify shared.txt on both main and feat so direct and merge-base diffs differ."""
    repo = _make_repo_with_main(tmp_path)
    write_and_stage(repo, "shared.txt", "line one\nline two\nline three\n")
    _commit(repo, "shared baseline")
    _git(repo, "checkout", "-b", "feat")
    write_and_stage(repo, "shared.txt", "line one\nline two FEAT\nline three\n")
    _commit(repo, "feat edit")
    _git(repo, "checkout", "main")
    write_and_stage(repo, "shared.txt", "line one\nline two MAIN\nline three\n")
    _commit(repo, "main edit after branch")
    _git(repo, "checkout", "feat")
    return repo, "main", "feat"

def test_diff_paths_direct_vs_merge_base_differ_on_divergent_history(tmp_path: Path) -> None:
    """Direct diff includes main's later commit; merge-base diff does not. Pin the diff."""
    repo, base, head = _make_divergent_history(tmp_path)
    direct = git_ops.diff_paths(repo, base, head, ["shared.txt"], merge_base_diff=False)
    since_merge_base = git_ops.diff_paths(repo, base, head, ["shared.txt"], merge_base_diff=True)
    assert direct != since_merge_base
    assert "MAIN" in direct
    assert "MAIN" not in since_merge_base

def test_diff_paths_restricts_to_paths(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feat")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    _commit(repo, "two files")
    out = git_ops.diff_paths(repo, "main", "feat", ["keep.txt"])
    assert "keep.txt" in out
    assert "drop.txt" not in out

def test_diff_paths_unified_context_lines(repo: Path) -> None:
    write_and_stage(repo, "ctx.txt", "\n".join(f"line {i}" for i in range(1, 31)) + "\n")
    _commit(repo, "ctx baseline")
    _git(repo, "checkout", "-b", "feat")
    lines = [f"line {i}" for i in range(1, 31)]
    lines[14] = "line 15 CHANGED"
    write_and_stage(repo, "ctx.txt", "\n".join(lines) + "\n")
    _commit(repo, "ctx edit")
    small = git_ops.diff_paths(repo, "main", "feat", ["ctx.txt"], unified=1)
    big = git_ops.diff_paths(repo, "main", "feat", ["ctx.txt"], unified=10)
    assert len(big) > len(small)

def test_diff_paths_raises_on_invalid_ref(repo: Path) -> None:
    with pytest.raises(GitError):
        git_ops.diff_paths(repo, "definitely-not-a-ref", "HEAD", ["base.txt"])


def test_gh_pr_queries_request_only_legacy_fields(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        stdout = "{}" if cmd[1:3] == ["pr", "view"] else "[]"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_ops.gh_pr_view(repo) == {}
    assert git_ops.gh_pr_list_for_branch(repo, "feature") == []
    assert calls[0][calls[0].index("--json") + 1].split(",") == list(git_ops.GH_PR_VIEW_FIELDS)
    assert calls[1][calls[1].index("--json") + 1].split(",") == list(git_ops.GH_PR_LIST_FIELDS)
    assert "baseRefOid" not in git_ops.GH_PR_VIEW_FIELDS
    assert "baseRefOid" not in git_ops.GH_PR_LIST_FIELDS
    for fields in (git_ops.GH_PR_VIEW_FIELDS, git_ops.GH_PR_LIST_FIELDS):
        assert "headRepository" in fields
        assert "headRepositoryOwner" in fields

@pytest.mark.parametrize(
    ("pr", "stderr"),
    [
        (None, 'no pull requests found for branch "feature"\n'),
        (42, "GraphQL: Could not resolve to a PullRequest with the number of 42. (repository.pullRequest)\n"),
    ],
)
def test_gh_pr_view_returns_none_only_for_anchored_absence(
    repo: Path, monkeypatch: pytest.MonkeyPatch, pr: int | None, stderr: str,
) -> None:
    _patch_subprocess_run(monkeypatch, returncode=1, stderr=stderr)
    assert git_ops.gh_pr_view(repo, pr) is None

@pytest.mark.parametrize(
    "stderr",
    [
        "no git remotes found", "HTTP 401: authentication required", "Unknown JSON field: futureField",
        "GraphQL: Could not resolve to a PullRequest with the number of 41. (repository.pullRequest)",
        'prefix: no pull requests found for branch "feature"', "HTTP 404: resource not found",
    ],
)
def test_gh_pr_view_unknown_failures_raise(repo: Path, monkeypatch: pytest.MonkeyPatch, stderr: str) -> None:
    _patch_subprocess_run(monkeypatch, returncode=1, stderr=stderr)
    with pytest.raises(GitError, match=re.escape(stderr)):
        git_ops.gh_pr_view(repo, 42)

@pytest.mark.parametrize("stdout", ["not-json", "[]", "null", "42"])
def test_gh_pr_view_rejects_invalid_json_shape(repo: Path, monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    _patch_subprocess_run(monkeypatch, stdout=stdout)
    with pytest.raises(GitError, match="invalid JSON|JSON object"):
        git_ops.gh_pr_view(repo, 42)

@pytest.mark.parametrize("stdout", ["not-json", "{}", "[42]", '[{"number": 1}, null]'])
def test_gh_pr_list_rejects_failure_and_invalid_shape(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    _patch_subprocess_run(monkeypatch, stdout=stdout)
    with pytest.raises(GitError, match="invalid JSON|JSON list|row"):
        git_ops.gh_pr_list_for_branch(repo, "feature")

def test_gh_pr_list_nonzero_uses_gh_error_classifier(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_subprocess_run(monkeypatch, returncode=1, stderr="rate limit exceeded; retry-after: 7")
    with pytest.raises(git_ops.RateLimitError) as excinfo:
        git_ops.gh_pr_list_for_branch(repo, "feature")
    assert excinfo.value.retry_after == 7

@pytest.mark.parametrize("slug", ["owner", "owner/repo/extra", "owner/ ", " owner/repo", "/repo"])
def test_gh_repo_view_required_rejects_invalid_slug(repo: Path, monkeypatch: pytest.MonkeyPatch, slug: str) -> None:
    _patch_subprocess_run(monkeypatch, stdout=slug + "\n")
    with pytest.raises(GitError, match="invalid repository slug"):
        git_ops.gh_repo_view_required(repo)

def test_gh_repo_view_required_returns_exact_slug(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_subprocess_run(monkeypatch, stdout="Owner/Repo\n")
    assert git_ops.gh_repo_view_required(repo) == ("Owner", "Repo")

def test_diagnostic_url_redaction_is_bounded_on_long_untrusted_text() -> None:
    """Non-URL diagnostics must not trigger quadratic scheme-prefix searches."""
    checked = subprocess.run(
        [
            sys.executable, "-c",
            "from daydream.git_ops.process import _redact_sensitive_text\n"
            "for text in ('A' * 100_000, 'A.' * 50_000):\n"
            "    assert _redact_sensitive_text(text) == text\n"
            "for scheme in ('https', 'ssh', 'git+custom.transport'):\n"
            "    raw = f'failed {scheme}://user:password@example.invalid/o/r'\n"
            "    assert _redact_sensitive_text(raw) == "
            "f'failed {scheme}://***@example.invalid/o/r'\n",
        ], cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=5, check=False,
    )
    assert checked.returncode == 0, checked.stderr

def test_gh_repo_view_required_preserves_safe_failure_diagnostic(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_subprocess_run(
        monkeypatch, returncode=1,
        stderr="HTTP 401: authentication required for token ghp_abcdefghijklmnopqrstuvwxyz1234567890",
    )
    with pytest.raises(GitError) as excinfo:
        git_ops.gh_repo_view_required(repo)
    assert "authentication required" in str(excinfo.value)
    assert "ghp_abcdefghijklmnopqrstuvwxyz1234567890" not in str(excinfo.value)

@gh_required
@pytest.mark.parametrize("fields", ["view", "list"])
def test_installed_gh_accepts_production_pr_fields_without_network(repo: Path, fields: str) -> None:
    selected = git_ops.GH_PR_VIEW_FIELDS if fields == "view" else git_ops.GH_PR_LIST_FIELDS
    proc = subprocess.run(
        ["gh", "pr", fields, "--json", ",".join(selected)], cwd=repo, capture_output=True, text=True, check=False,
    )
    assert "Unknown JSON field" not in proc.stderr

def test_gh_api_input_data_passes_tempfile_and_cleans_up(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        idx = cmd.index("--input")
        captured["input_path"] = cmd[idx + 1]
        captured["payload"] = Path(cmd[idx + 1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='{"ok": true}', stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = git_ops.gh_api(
        repo, "repos/owner/repo/pulls/1/reviews", method="POST", input_data={"event": "COMMENT", "body": "hi"},
    )
    assert result == {"ok": True}
    cmd = captured["cmd"]
    assert cmd[:2] == ["gh", "api"]
    assert "--input" in cmd
    assert "--method" in cmd
    method_idx = cmd.index("--method")
    assert cmd[method_idx + 1] == "POST"
    assert json.loads(captured["payload"]) == {"event": "COMMENT", "body": "hi"}
    assert not Path(captured["input_path"]).exists()

@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        ("http", GitError), ("invalid-json", GitError), ("timeout", git_ops.GitTimeoutError),
        ("rate-limit", git_ops.RateLimitError),
    ],
)
def test_gh_api_input_data_preserves_tempfile_on_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch, failure: str, expected_type: type[GitError],
) -> None:
    """Preserve failed API payloads and disclose their recovery path in the error."""
    captured: dict[str, Any] = {}
    attempts: list[list[str]] = []
    class RequestAuth:
        calls = 0
        def environment_for_request(self) -> dict[str, str]:
            self.calls += 1
            return {"OWNING_RUN": "payload-metadata-test"}
    auth = RequestAuth()
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        attempts.append(cmd)
        idx = cmd.index("--input")
        captured["input_path"] = cmd[idx + 1]
        assert kwargs["env"] == {"OWNING_RUN": "payload-metadata-test"}
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
        if failure == "invalid-json":
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="not-json", stderr="")
        if failure == "rate-limit":
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="HTTP 429: rate limit; retry-after: 7"
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="HTTP 422: Validation failed")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", "4")
    with pytest.raises(expected_type) as excinfo:
        git_ops.gh_api(
            repo, "repos/owner/repo/pulls/1/reviews", method="POST", input_data={"bad": "payload"}, auth=auth,
        )
    preserved = Path(captured["input_path"])
    try:
        assert type(excinfo.value) is expected_type
        assert len(attempts) == 1
        assert auth.calls == 1
        msg = str(excinfo.value)
        if failure == "timeout":
            assert "timed out" in msg
        else:
            assert "payload preserved at" in msg
            assert str(preserved) in msg
        assert preserved.exists()
        assert json.loads(preserved.read_text(encoding="utf-8")) == {"bad": "payload"}
        assert excinfo.value.preserved_payload_path == preserved
        if isinstance(excinfo.value, git_ops.RateLimitError):
            assert excinfo.value.retry_after == 7
    finally:
        preserved.unlink(missing_ok=True)

@pytest.mark.parametrize(
    ("pr", "number"), [(None, 7), (42, 42)], ids=["omits_pr_arg_when_none", "includes_pr_arg_when_given"],
)
def test_gh_pr_view_pr_arg(repo: Path, monkeypatch: pytest.MonkeyPatch, pr: int | None, number: int) -> None:
    captured: dict[str, list[str]] = {}
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=f'{{"number": {number}}}', stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = git_ops.gh_pr_view(repo) if pr is None else git_ops.gh_pr_view(repo, pr)
    assert result == {"number": number}
    cmd = captured["cmd"]
    if pr is None:
        assert cmd[:3] == ["gh", "pr", "view"]
        assert all(not part.isdigit() for part in cmd)
    else:
        assert cmd[:4] == ["gh", "pr", "view", str(pr)]

def test_daydream_commits_returns_tagged_commits(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feat/x")
    write_and_stage(repo, "a.py", "a\n")
    _git(repo, "commit", "-m", "fix: something\n\nDaydream-Run: test-123\nDaydream-Version: 0.14.0")
    result = git_ops.daydream_commits(repo, "main")
    assert result is not None
    assert "fix: something" in result

def test_daydream_commits_excludes_untagged(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feat/x")
    write_and_stage(repo, "b.py", "b\n")
    _commit(repo, "chore: unrelated change")
    result = git_ops.daydream_commits(repo, "main")
    assert result is None

def test_daydream_commits_none_when_no_commits(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feat/x")
    result = git_ops.daydream_commits(repo, "main")
    assert result is None

def _make_bare_remote(tmp_path: Path) -> Path:
    """Create a bare remote repo with one committed file."""
    repo = _make_repo_with_main(tmp_path / "src")
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, capture_output=True)  # noqa: S603, S607 - arguments are not user-controlled
    return bare

def test_clone_creates_working_tree(tmp_path: Path) -> None:
    bare = _make_bare_remote(tmp_path)
    target = tmp_path / "cloned"
    git_ops.clone(str(bare), target)
    assert (target / ".git").is_dir()
    assert (target / "base.txt").read_text() == "base\n"

def test_clone_of_linked_worktree_materializes_head_and_staged_patch(
    tmp_path: Path, linked_worktree: tuple[Path, Path],
) -> None:
    _main, linked = linked_worktree
    # Stage a modification to a feature-only file (absent from main).
    parser = linked / "services" / "taste" / "parser.go"
    parser.write_text("package taste\n\n// staged spike\nfunc Spiked() {}\n")
    _git(linked, "add", "services/taste/parser.go")
    source_patch = git_ops.staged_patch(linked)
    assert source_patch, "expected a nonempty staged patch"
    clone = tmp_path / "spike-clone"
    git_ops.clone(str(linked), clone)
    assert git_ops.head_sha(clone) == git_ops.head_sha(linked)
    # Copy the working file so the clone worktree matches the staged content.
    shutil.copy2(parser, clone / "services" / "taste" / "parser.go")
    git_ops.apply_staged_patch(clone, source_patch)
    assert git_ops.staged_patch(clone) == source_patch
    assert _git(clone, "diff", "--", "services/taste/parser.go") == ""

def test_remove_remote_deletes_configured_remote(tmp_path: Path) -> None:
    source = _make_repo_with_main(tmp_path / "src")
    clone = tmp_path / "clone"
    git_ops.clone(str(source), clone)
    assert git_ops.remote_url(clone) == str(source)
    before = git_ops.head_sha(clone)
    git_ops.remove_remote(clone)
    assert git_ops.remote_url(clone) is None
    assert git_ops.head_sha(clone) == before

def test_update_refs_snapshots_a_batch_and_aborts_atomically(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path, name="update_refs")
    oid = _git(repo, "rev-parse", "HEAD")
    good = {f"refs/heads/{name}": oid for name in ("main", "unlock", "release/9.9", "topic.LOCK")}
    git_ops.update_refs(repo, good)
    for ref, expected in good.items():
        assert _git(repo, "rev-parse", ref) == expected
    with pytest.raises(git_ops.GitError, match="git update-ref --stdin failed"):
        # ".." passes the Python guards but git rejects it, so the whole
        # batch still aborts atomically on git's side: nothing is written.
        git_ops.update_refs(repo, {"refs/heads/ok": oid, "refs/heads/foo..bar": oid})
    assert "ok" not in git_ops.list_local_branches(repo)  # whole batch rolled back

def test_update_refs_rejects_injectable_ref_names_before_shell_out(tmp_path: Path) -> None:
    """Reject invalid refs and OIDs before they can inject extra update-ref stdin commands."""
    repo = _make_repo_with_main(tmp_path, name="update_refs_inject")
    oid = git_ops.head_sha(repo)
    for bad in ("refs/heads/bad\nname", "refs/heads/two words", "-dash-start"):
        with pytest.raises(git_ops.GitError, match="invalid ref name"):
            git_ops.update_refs(repo, {bad: oid})
    with pytest.raises(git_ops.GitError, match="invalid OID"):
        git_ops.update_refs(repo, {"refs/heads/ok": "not-a-sha"})
    assert "bad" not in git_ops.list_local_branches(repo)
    assert "two" not in git_ops.list_local_branches(repo)

def test_staged_patch_round_trips_index_state(tmp_path: Path) -> None:
    """Binary content exercises the patch's base85 encoding through a real clone round-trip."""
    source = _make_repo_with_main(tmp_path / "src")
    clone = tmp_path / "clone"
    git_ops.clone(str(source), clone)
    payload = bytes(range(256))  # every byte value, incl. NUL/newline — not text
    write_and_stage(source, "blob.bin", payload)
    shutil.copy2(source / "blob.bin", clone / "blob.bin")
    patch = git_ops.staged_patch(source)
    assert patch  # nonempty bytes
    git_ops.apply_staged_patch(clone, patch)
    assert git_ops.staged_patch(clone) == git_ops.staged_patch(source)
    assert _git(clone, "diff", "--", "blob.bin") == ""
    assert (clone / "blob.bin").read_bytes() == payload

def test_strict_enumeration_raises_where_soft_fails(tmp_path: Path) -> None:
    """Strict enumeration prevents snapshot preparation from silently omitting files after Git failure."""
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    assert git_ops.ls_files(not_a_repo) == []
    assert git_ops.list_untracked(not_a_repo) == []
    # Strict: the same failure raises GitError that callers wrap in CodexError.
    with pytest.raises(git_ops.GitError):
        git_ops.ls_files(not_a_repo, strict=True)
    with pytest.raises(git_ops.GitError):
        git_ops.list_untracked(not_a_repo, strict=True)
    repo = _make_repo_with_main(tmp_path / "src")
    (repo / "tracked.txt").write_text("x")
    (repo / "untracked.txt").write_text("y")
    _git(repo, "add", "tracked.txt")
    assert git_ops.ls_files(repo, strict=True) == git_ops.ls_files(repo)
    assert git_ops.list_untracked(repo, strict=True) == git_ops.list_untracked(repo)
    assert "tracked.txt" in git_ops.ls_files(repo, strict=True)
    assert "untracked.txt" in git_ops.list_untracked(repo, strict=True)

def test_clone_raises_on_invalid_remote(tmp_path: Path) -> None:
    target = tmp_path / "nope"
    with pytest.raises(git_ops.GitError, match="git clone .* failed"):
        git_ops.clone("file:///nonexistent/repo.git", target)

@pytest.mark.parametrize(
    ("blobless", "filter_present"), [(True, True), (False, False)],
    ids=["blobless_passes_filter_flag", "default_no_filter_flag"],
)
def test_clone_filter_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blobless: bool, filter_present: bool,
) -> None:
    captured: dict[str, list[str]] = {}
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    git_ops.clone("https://example.com/repo.git", tmp_path / "out", blobless=blobless)
    assert ("--filter=blob:none" in captured["cmd"]) is filter_present

@pytest.mark.parametrize(
    ("stderr", "expected_type"),
    [
        pytest.param("gh: API rate limit exceeded for user (HTTP 403)", git_ops.RateLimitError, id="rate-limit"),
        pytest.param("gh: Not Found (HTTP 404)", git_ops.GitError, id="plain-git-error"),
    ],
)
def test_gh_api_classifies_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stderr: str, expected_type: type[GitError],
) -> None:
    proc = subprocess.CompletedProcess(args=["gh"], returncode=1, stdout="", stderr=stderr)
    monkeypatch.setattr(git_process, "_run_gh", lambda *a, **k: proc)
    with pytest.raises(expected_type) as exc:
        git_ops.gh_api(tmp_path, "repos/o/r/pulls/1")
    assert type(exc.value) is expected_type
    assert exc.value.preserved_payload_path is None

@pytest.mark.parametrize(
    ("stdout", "endpoint", "paginate", "jq", "expected", "expected_args"),
    [
        pytest.param(
            '{"id": 1}\n{"id": 2}\n',
            "/app/installations", True, ".[]", [{"id": 1}, {"id": 2}],
            ["api", "--paginate", "--jq", "(.[]) | @json", "/app/installations"], id="ndjson-list",
        ),
        pytest.param(
            '"anderskev"\n',
            "user", False, ".login", ["anderskev"], ["api", "--jq", "(.login) | @json", "user"], id="raw-scalar",
        ),
    ],
)
def test_gh_api_jq_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str, endpoint: str, paginate: bool, jq: str,
    expected: list[Any], expected_args: list[str],
) -> None:
    captured: dict[str, list[str]] = {}
    def fake_run_gh(
        repo: Path, args: list[str], **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Capture gh arguments and return the parameterized response payload."""
        captured["args"] = args
        return subprocess.CompletedProcess(args=["gh"], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(git_process, "_run_gh", fake_run_gh)
    result = git_ops.gh_api(tmp_path, endpoint, paginate=paginate, jq=jq)
    assert result == expected
    assert captured["args"] == expected_args

def test_gh_api_headers_pass_dash_h_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []
    captured_auth: list[Any] = []
    class _Proc:
        returncode = 0
        stdout = "{}"
        stderr = ""
    def fake_run_gh(repo: Path, args: list[str], **kwargs: Any) -> _Proc:
        captured.append(args)
        captured_auth.append(kwargs["auth"])
        return _Proc()
    monkeypatch.setattr(git_process, "_run_gh", fake_run_gh)
    headers = {"Authorization": "Bearer jwt-abc"}
    auth = git_ops.StaticGitHubAuth({"PATH": "/tools", "GH_TOKEN": "jwt-abc"})
    git_ops.gh_api(tmp_path, "/app/installations", auth=auth, headers=headers)
    git_ops.gh_api(
        tmp_path, "/app/installations/1/access_tokens", auth=auth, method="POST", input_data={"repositories": ["r"]},
        headers=headers,
    )
    assert captured[0][:3] == ["api", "-H", "Authorization: Bearer jwt-abc"]
    assert captured[1][:3] == ["api", "-H", "Authorization: Bearer jwt-abc"]
    assert "--input" in captured[1]
    assert captured_auth == [auth, auth]

def test_gh_api_error_message_redacts_authorization_token(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake only subprocess.run; the real gh_api path must redact its own Bearer header."""
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("gh executable failed to spawn")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GitError) as excinfo:
        git_ops.gh_api(repo, "/app/installations", headers={"Authorization": "Bearer jwt-super-secret-xyz"})
    msg = str(excinfo.value)
    assert "jwt-super-secret-xyz" not in msg
    assert "Authorization: ***" in msg

def test_gh_api_timeout_redacts_authorization_token(
    repo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Force one retry so the warning-log branch fires.
    monkeypatch.setattr(git_process, "_gh_retries", lambda: 1)
    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        with pytest.raises(git_ops.GitTimeoutError) as excinfo:
            git_ops.gh_api(
                repo, "/app/installations", headers={"Authorization": "Bearer jwt-super-secret-xyz"}, idempotent=True,
            )
    msg = str(excinfo.value)
    assert "jwt-super-secret-xyz" not in msg
    assert "Authorization: ***" in msg
    # The retry warning fires, but must not carry any argv-derived text at all:
    # the shared retry helper logs only the program and counters, keeping the
    # credential-bearing header out of the log sink entirely.
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings, "expected a retry warning to be logged"
    assert all("jwt-super-secret-xyz" not in w for w in warnings)
    assert all("Authorization" not in w for w in warnings)
    assert any(w.startswith("gh timed out after") for w in warnings)

@pytest.mark.parametrize(
    ("failure_mode", "expected_fragment"),
    [
        pytest.param("spawn-error", "synthetic subprocess failure", id="spawn-error"),
        pytest.param("timeout", "timed out after", id="timeout"),
        pytest.param("api-error", "synthetic API failure", id="api-error"),
        pytest.param("invalid-json", "returned invalid JSON", id="invalid-json"),
        pytest.param("rate-limit", "rate limit", id="rate-limit"),
        pytest.param("timeout-warning", "timed out after", id="timeout-warning"),
    ],
)
def test_gh_api_manifest_conversion_failures_redact_code(
    repo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure_mode: str,
    expected_fragment: str,
) -> None:
    """Fake subprocess failures through real gh_api; redact conversion codes while retaining diagnostics."""
    sentinel = "manifest-code-sentinel"
    endpoint = f"/app-manifests/{sentinel}/conversions"
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure_mode == "spawn-error":
            raise OSError(f"synthetic subprocess failure: {endpoint}")
        if failure_mode in ("timeout", "timeout-warning"):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
        if failure_mode == "rate-limit":
            return subprocess.CompletedProcess(
                args=["gh"], returncode=1, stdout="", stderr=f"secondary rate limit: {endpoint}"
            )
        if failure_mode == "api-error":
            return subprocess.CompletedProcess(
                args=["gh"], returncode=1, stdout="", stderr=f"synthetic API failure: {endpoint}"
            )
        return subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="not-json", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Only the warning case enables an idempotent retry; other failure paths execute once.
    kwargs: dict[str, Any] = {"method": "POST"}
    if failure_mode == "timeout-warning":
        monkeypatch.setattr(git_process, "_gh_retries", lambda: 1)
        kwargs["idempotent"] = True
    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        with pytest.raises(GitError) as excinfo:
            git_ops.gh_api(repo, endpoint, **kwargs)
    if failure_mode == "rate-limit":
        assert isinstance(excinfo.value, git_ops.RateLimitError)
    msg = str(excinfo.value)
    assert sentinel not in msg
    assert "/app-manifests/***/conversions" in msg
    assert expected_fragment in msg
    if failure_mode == "timeout-warning":
        warnings = [r.getMessage() for r in caplog.records]
        assert warnings, "expected a retry warning to be logged"
        assert all(sentinel not in w for w in warnings)
        assert all("/app-manifests" not in w for w in warnings)
        assert any(w.startswith("gh timed out after") for w in warnings)

def test_gh_api_jq_invalid_line_raises_git_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Proc:
        returncode = 0
        stdout = '{"id": 1}\nnot-json\n'
        stderr = ""
    monkeypatch.setattr(git_process, "_run_gh", lambda *a, **k: _Proc())
    with pytest.raises(git_ops.GitError, match="invalid JSON"):
        git_ops.gh_api(tmp_path, "/app/installations", jq=".[]")

def test_github_request_budget_preserves_subsecond_deadline() -> None:
    readings = iter((10.25, 10.75, 11.0))
    budget = git_ops.GitHubRequestBudget(deadline=11.0, per_request_seconds=0.6, monotonic=lambda: next(readings))
    assert budget.next_timeout() == pytest.approx(0.6)
    assert budget.next_timeout() == pytest.approx(0.25)
    with pytest.raises(git_ops.DeadlineExpired, match="deadline"):
        budget.next_timeout()

def test_pr_list_fields_include_gh_245_head_ref_name() -> None:
    assert "headRefName" in git_ops.GH_PR_LIST_FIELDS
    assert "baseRefOid" not in git_ops.GH_PR_LIST_FIELDS

@pytest.mark.parametrize("extra_field", ["baseRefOid", "futureCompatibilityFloor"])
@pytest.mark.parametrize("query_kind", ["view", "list"])
def test_fake_gh_rejects_every_field_outside_legacy_allowlist(
    fake_gh: FakeGh, git_repo: Path, monkeypatch: pytest.MonkeyPatch, query_kind: str, extra_field: str,
) -> None:
    fake_gh.serve_pr_view({"number": 7})
    constant = "GH_PR_VIEW_FIELDS" if query_kind == "view" else "GH_PR_LIST_FIELDS"
    monkeypatch.setattr(git_github, constant, (*getattr(git_github, constant), extra_field))
    with pytest.raises(GitError, match=rf'Unknown JSON field: "{extra_field}"'):
        if query_kind == "view":
            git_ops.gh_pr_view(git_repo, 7)
        else:
            git_ops.gh_pr_list_for_branch(git_repo, "feature")

def test_gh_secret_set_requires_exactly_one_scope(fake_gh: FakeGh, git_repo: Path) -> None:
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v")
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v", org="acme", repo_slug="o/r")

def test_gh_secret_list_returns_names(fake_gh: FakeGh, git_repo: Path) -> None:
    fake_gh.serve_secret_list(["DAYDREAM_APP_ID", "ANTHROPIC_API_KEY"])
    assert git_ops.gh_secret_list(git_repo, repo_slug="o/r") == ["DAYDREAM_APP_ID", "ANTHROPIC_API_KEY"]

def test_gh_variable_list_returns_names(fake_gh: FakeGh, git_repo: Path) -> None:
    fake_gh.serve_variable_list(["DAYDREAM_BOT_HANDLE"])
    assert git_ops.gh_variable_list(git_repo, org="acme") == ["DAYDREAM_BOT_HANDLE"]

def test_gh_pr_create_returns_url(fake_gh: FakeGh, git_repo: Path) -> None:
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/9")
    url = git_ops.gh_pr_create(git_repo, head="b", base="main", title="t", body="b")
    assert url == "https://github.com/o/r/pull/9"

def test_gh_pr_create_failure_raises_git_error(fake_gh: FakeGh, git_repo: Path) -> None:
    # No "pr-create" response configured → the shim exits non-zero.
    with pytest.raises(GitError):
        git_ops.gh_pr_create(git_repo, head="b", base="main", title="t", body="b")

_FILE_AT_REF_SHA = "0123456789abcdef0123456789abcdef01234567"

def test_gh_file_at_ref_reads_a_nested_path_outside_any_checkout(fake_gh: FakeGh, tmp_path: Path) -> None:
    fake_gh.set_response(
        "GET", "repos/o/r/contents/src/deep/a%20b.py",
        value={
            "type": "file", "path": "src/deep/a b.py", "sha": "c" * 40, "encoding": "base64",
            "content": base64.b64encode(b"VALUE = 1\n").decode(),
        },
    )
    content = git_ops.gh_file_at_ref(tmp_path, "o/r", _FILE_AT_REF_SHA, "src/deep/a b.py")
    assert content == b"VALUE = 1\n"
    endpoints = [call.endpoint for call in fake_gh.calls("GET")]
    assert endpoints == [f"repos/o/r/contents/src/deep/a%20b.py?ref={_FILE_AT_REF_SHA}"]

def test_gh_file_at_ref_forwards_auth_to_contents_and_blob_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    auth = git_ops.StaticGitHubAuth({"PATH": "/tools", "GH_TOKEN": "ghs_file_reader_token_1234567890"})
    seen_auth: list[Any] = []
    def api(_repo: Path, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        seen_auth.append(kwargs["auth"])
        if "/contents/" in endpoint:
            return {"type": "file", "sha": "c" * 40, "encoding": "none"}
        return {"encoding": "base64", "content": base64.b64encode(b"large file\n").decode()}
    monkeypatch.setattr(git_github, "gh_api", api)
    assert git_ops.gh_file_at_ref(tmp_path, "o/r", _FILE_AT_REF_SHA, "large.bin", auth=auth) == b"large file\n"
    assert seen_auth == [auth, auth]

@pytest.mark.parametrize(
    ("slug", "ref", "path"),
    [
        ("o", _FILE_AT_REF_SHA, "a.py"), ("../o/r", _FILE_AT_REF_SHA, "a.py"), ("o/r", "main", "a.py"),
        ("o/r", _FILE_AT_REF_SHA, "../../etc/passwd"), ("o/r", _FILE_AT_REF_SHA, "/etc/passwd"),
        ("o/r", _FILE_AT_REF_SHA, "a.py\n"),
    ],
)
def test_gh_file_at_ref_rejects_unsafe_arguments_before_any_request(
    fake_gh: FakeGh, tmp_path: Path, slug: str, ref: str, path: str
) -> None:
    """Slug, ref, and the model-derived path all become endpoint text."""
    with pytest.raises(GitError):
        git_ops.gh_file_at_ref(tmp_path, slug, ref, path)
    assert fake_gh.process_calls() == []

def test_gh_file_at_ref_rejects_a_directory_response(fake_gh: FakeGh, tmp_path: Path) -> None:
    fake_gh.set_response("GET", "repos/o/r/contents/src", value=[{"type": "file", "path": "src/a.py"}])
    with pytest.raises(GitError, match="does not name a file"):
        git_ops.gh_file_at_ref(tmp_path, "o/r", _FILE_AT_REF_SHA, "src")

def test_log_shas_returns_none_when_ref_is_gone(repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A deleted ref makes the query unavailable; an empty list would falsely imply no follow-up commits."""
    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        result = git_ops.log_shas(repo, "deleted-branch", since="main")
    assert result is None
    assert any("log_shas" in record.message for record in caplog.records)

def test_log_shas_returns_empty_list_when_range_is_genuinely_empty(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    assert git_ops.log_shas(repo, "topic", since="main") == []

def test_log_shas_returns_commits_ahead_of_since(tmp_path: Path) -> None:
    """The success path still returns SHAs, newest first."""
    repo = _topic_repo(tmp_path)
    write_and_stage(repo, "a.txt", "a\n")
    _commit(repo, "topic-1")
    shas = git_ops.log_shas(repo, "topic", since="main")
    assert shas == [_git(repo, "rev-parse", "topic").strip()]

def test_log_shas_since_returns_commits_in_range(tmp_path: Path) -> None:
    repo = _topic_repo(tmp_path)
    write_and_stage(repo, "a.txt", "a\n")
    _commit(repo, "topic-1")
    write_and_stage(repo, "b.txt", "b\n")
    _commit(repo, "topic-2")
    shas = git_ops.log_shas_since(repo, "main", "topic")
    assert len(shas) == 2
    tip = _git(repo, "rev-parse", "topic").strip()
    parent = _git(repo, "rev-parse", "topic^").strip()
    assert shas == [tip, parent]  # newest first, exact, both entries
    assert all(len(s) == 40 for s in shas)

def test_log_shas_since_warns_on_git_error(repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        result = git_ops.log_shas_since(repo, "main", "nonexistent-ref")
    assert result == []
    assert any("log_shas_since" in record.message for record in caplog.records)

def test_worktree_lock_mtime_fails_closed_when_exact_worktree_disappears(tmp_path: Path, repo: Path) -> None:
    wt = tmp_path / "linked" / "same"
    git_ops.worktree_add(repo, wt, "main", detach=True)
    retained = tmp_path / "moved-outside-git"
    wt.rename(retained)
    with pytest.raises(GitError, match="Git directory"):
        git_ops.worktree_lock_mtime(wt)
    assert retained.is_dir()
    assert (retained / ".git").is_file()

def test_worktree_lock_mtime_rejects_nonregular_lock_metadata(tmp_path: Path, repo: Path) -> None:
    wt = tmp_path / "linked" / "same"
    git_ops.worktree_add(repo, wt, "main", detach=True)
    locked = git_ops.git_dir(wt) / "locked"
    locked.mkdir()
    with pytest.raises(GitError, match="lock metadata is unsafe"):
        git_ops.worktree_lock_mtime(wt)
    assert wt.is_dir()
    assert locked.is_dir()

@pytest.mark.parametrize("lock_reason", ["run-A", None], ids=["locked", "unlocked"])
def test_worktree_remove_unlocked_unlocks_before_removing(repo: Path, lock_reason: str | None) -> None:
    worktree = repo / "worktree"
    git_ops.worktree_add(repo, worktree, "main", detach=True, lock_reason=lock_reason)
    assert (git_ops.worktree_lock_mtime(worktree) is not None) is (lock_reason is not None)
    git_ops.worktree_remove_unlocked(repo, worktree)
    assert not worktree.exists()


def test_worktree_add_with_lock_reason_arms_lock_atomically(repo: Path) -> None:
    """Arm the lock during creation so concurrent pruning has no unlocked removal window."""
    wt = repo / "wt-locked"
    git_ops.worktree_add(repo, wt, "main", detach=True, lock_reason="run-A")
    assert git_ops.worktree_lock_mtime(wt) is not None
    git_dir = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    locked = git_dir / "worktrees" / wt.name / "locked"
    assert locked.is_file()
    assert "run-A" in locked.read_text(encoding="utf-8")

def test_clone_error_message_never_contains_remote_url(tmp_path: Path) -> None:
    with pytest.raises(GitError) as excinfo:
        git_ops.clone("https://user:ghp_canaryfake123@unreachable.invalid/o/r.git", tmp_path / "t", timeout=5)
    message = str(excinfo.value)
    assert "ghp_canaryfake123" not in message
    assert "user:" not in message
    assert "unreachable.invalid" in message  # host is safe to keep

def test_clone_error_message_redacts_stderr_url_echo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run
    def fake_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 128,
            stderr=(
                "fatal: unable to access 'https://user:ghp_canaryfake123@unreachable.invalid/o/r.git/': "
                "Could not resolve host"
            ),
        )
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GitError) as excinfo:
        git_ops.clone("https://user:ghp_canaryfake123@unreachable.invalid/o/r.git", tmp_path / "t", timeout=5)
    assert "ghp_canaryfake123" not in str(excinfo.value)
    assert real_run is not None

def test_has_executable_pre_push_hook_default_hooks_dir(tmp_path: Path) -> None:
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)
    pp = hooks / "pre-push"
    pp.write_text("#!/bin/sh\n")
    pp.chmod(0o755)
    assert git_ops.has_executable_pre_push_hook(tmp_path) is True
    pp.chmod(0o644)  # non-executable => no hook
    assert git_ops.has_executable_pre_push_hook(tmp_path) is False

def test_has_executable_pre_push_hook_honors_core_hooks_path(tmp_path: Path) -> None:
    custom = tmp_path / "myhooks"
    custom.mkdir()
    pp = custom / "pre-push"
    pp.write_text("#!/bin/sh\n")
    pp.chmod(0o755)
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.hooksPath", str(custom)], check=True)
    assert git_ops.has_executable_pre_push_hook(tmp_path) is True

def test_remote_contains_commit_true_after_push_false_before(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    clone = tmp_path / "clone"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(clone)], check=True, capture_output=True)
    for k, v in {"user.email": "t@t", "user.name": "t"}.items():
        subprocess.run(["git", "-C", str(clone), "config", k, v], check=True)
    (clone / "a").write_text("x")
    subprocess.run(["git", "-C", str(clone), "add", "a"], check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", "c1"], check=True, capture_output=True)
    sha = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True ).stdout.strip()
    subprocess.run(["git", "-C", str(clone), "remote", "add", "origin", str(remote)], check=True)
    assert git_ops.remote_contains_commit(clone, "main", sha) is False
    subprocess.run(["git", "-C", str(clone), "push", "-u", "origin", "main"], check=True, capture_output=True)
    assert git_ops.remote_contains_commit(clone, "main", sha) is True

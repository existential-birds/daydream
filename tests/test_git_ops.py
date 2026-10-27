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
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.git_ops import (
    BranchNotFoundError,
    GitError,
    NotAWorktreeError,
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


def test_resolve_diff_merge_base_prefers_present_origin_ref(repo: Path, tmp_path: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    write_and_stage(repo, "upstream.py", "UPSTREAM = 1\n")
    remote_tip = _commit(repo, "remote advancement")
    _git(repo, "branch", "feature", remote_tip)
    _git(repo, "reset", "--hard", base)
    _git(repo, "remote", "add", "origin", str(_bare_remote(tmp_path / "remote.git")))
    _git(repo, "update-ref", "refs/remotes/origin/main", remote_tip)
    _git(repo, "checkout", "feature")
    write_and_stage(repo, "feature.py", "FEATURE = 1\n")
    head = _commit(repo, "feature")
    assert git_ops.merge_base(repo, "main", head) == base
    assert git_ops.resolve_diff_merge_base(repo, "main", head) == remote_tip
    _git(repo, "branch", "--set-upstream-to=origin/main", "main")
    assert git_ops.merge_base(repo, "main") == remote_tip


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
    original_run_git = git_process._run_git

    def run_with_short_probe_deadline(target: Path, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Keep the real subprocess and retry loop; only the deliberately hung
        # external preference probe needs a shorter deadline than production.
        if args == ["rev-parse", "--verify", "refs/remotes/origin/main"]:
            kwargs["timeout"] = 0.5
        return original_run_git(target, args, **kwargs)

    monkeypatch.setattr(git_process, "_run_git", run_with_short_probe_deadline)
    with install_diff_base_git_shim(tmp_path, monkeypatch, mode="timeout-preference", head=head) as shim_dir:
        with pytest.raises(GitError) as raised:
            git_ops.resolve_diff_merge_base(repo, "main", head)
    assert (shim_dir / "timeout-probes").read_text().splitlines() == ["started"] * 3
    assert str(raised.value) == "branch-focus preferred base probe failed"
    assert str(repo) not in str(raised.value)
    assert head not in str(raised.value)
    assert str(shim_dir) not in str(raised.value)


def _repository_identity(repo: Path) -> tuple[str, str, bytes, str]:
    """Observe HEAD, worktree/index status, exact staged bytes, and all refs."""
    return (
        _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--short"), git_ops.staged_patch(repo),
        _git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
    )


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


@pytest.mark.parametrize(("settings", "diagnostic"), [
    pytest.param({
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "SOURCE_OBJECTS", "GIT_OBJECT_DIRECTORY": "EXTERNAL_OBJECTS",
        "GIT_DIR": "EXTERNAL", "GIT_COMMON_DIR": "EXTERNAL", "GIT_INDEX_FILE": "EXTERNAL",
        "GIT_WORK_TREE": "EXTERNAL", "GIT_NAMESPACE": "EXTERNAL", "GIT_CEILING_DIRECTORIES": "EXTERNAL",
        "GIT_PREFIX": "EXTERNAL", "GIT_SHALLOW_FILE": "EXTERNAL", "GIT_GRAFT_FILE": "EXTERNAL",
        "GIT_REPLACE_REF_BASE": "EXTERNAL", "GIT_REFERENCE_BACKEND": "EXTERNAL", "GIT_TEMPLATE_DIR": "EXTERNAL",
        "GIT_EXTERNAL_DIFF": "EXTERNAL", "GIT_DIFF_OPTS": "EXTERNAL", "GIT_TRACE": "EXTERNAL",
    }, (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES, GIT_CEILING_DIRECTORIES, GIT_COMMON_DIR, GIT_DIFF_OPTS, GIT_DIR, "
        "GIT_EXTERNAL_DIFF, GIT_GRAFT_FILE, GIT_INDEX_FILE, GIT_NAMESPACE, GIT_OBJECT_DIRECTORY, GIT_PREFIX, "
        "GIT_REFERENCE_BACKEND, GIT_REPLACE_REF_BASE, GIT_SHALLOW_FILE, GIT_TEMPLATE_DIR, GIT_TRACE, GIT_WORK_TREE"
    ), id="repository-redirects-and-trace"),
    pytest.param({
        "GIT_CONFIG": "EXTERNAL", "GIT_CONFIG_GLOBAL": "EXTERNAL", "GIT_CONFIG_SYSTEM": "EXTERNAL",
        "GIT_CONFIG_NOSYSTEM": "EXTERNAL", "GIT_CONFIG_COUNT": "EXTERNAL", "GIT_CONFIG_KEY_0": "EXTERNAL",
        "GIT_CONFIG_VALUE_0": "EXTERNAL", "GIT_CONFIG_PARAMETERS": "EXTERNAL",
    }, (
        "GIT_CONFIG, GIT_CONFIG_COUNT, GIT_CONFIG_GLOBAL, GIT_CONFIG_KEY_0, GIT_CONFIG_NOSYSTEM, "
        "GIT_CONFIG_PARAMETERS, GIT_CONFIG_SYSTEM, GIT_CONFIG_VALUE_0"
    ), id="configuration-namespace-and-count"),
    pytest.param({
        "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "commit.gpgsign", "GIT_CONFIG_VALUE_0": "INVALID_SIGNING_VALUE",
        "GIT_CONFIG_KEY_1": "core.excludesfile", "GIT_CONFIG_VALUE_1": "/dev/null",
    }, (
        "GIT_CONFIG_COUNT, GIT_CONFIG_KEY_0, GIT_CONFIG_KEY_1, GIT_CONFIG_VALUE_0, GIT_CONFIG_VALUE_1"
    ), id="indexed-signing-value"),
    pytest.param({"GIT_EXEC_PATH": "EXTERNAL"}, "GIT_EXEC_PATH", id="executable-path"),
])
def test_independent_snapshot_rejects_inherited_git_overrides_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, str], diagnostic: str,
) -> None:
    """Every unsafe key must appear in the complete rejection of one real inherited environment."""
    repo = tmp_path / "source"
    _init_repo(repo)
    write_and_stage(repo, "app.py", b"VALUE = 1\n")
    _commit(repo, "initial")
    (repo / "app.py").write_bytes(b"VALUE = 2\n")
    external = tmp_path / "external"
    shutil.copytree(repo / ".git", external)
    destination = tmp_path / "snapshot"

    def file_bytes(root: Path) -> dict[str, bytes]:
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    paths = {"SOURCE_OBJECTS": repo / ".git/objects", "EXTERNAL_OBJECTS": external / "objects", "EXTERNAL": external}
    environment = {name: str(paths.get(value, value)) for name, value in settings.items()}
    before_source, before_external = file_bytes(repo), file_bytes(external)
    with monkeypatch.context() as poison:
        for name in tuple(os.environ):
            if name.startswith("GIT_CONFIG"):
                poison.delenv(name)
        for name, value in environment.items():
            poison.setenv(name, value)
        with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
            git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
        assert {name: os.environ[name] for name in environment} == environment
    assert str(raised.value) == (
        "snapshot refuses inherited Git repository, configuration, diff, or trace overrides; offending variables: "
        + diagnostic
    )
    assert str(external) not in str(raised.value) and "INVALID_SIGNING_VALUE" not in str(raised.value)
    assert not destination.exists()
    assert file_bytes(repo) == before_source and file_bytes(external) == before_external


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_EXEC_PATH"])
def test_empty_git_override_refuses_before_repository_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    """Empty overrides remain unsafe even before the source repository can be resolved."""
    monkeypatch.setenv(variable, "")
    destination = tmp_path / "snapshot"
    with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
        git_ops.prepare_independent_snapshot(tmp_path / "absent", destination, include_untracked=False)
    assert variable in str(raised.value)
    assert not destination.exists()
    assert os.environ[variable] == ""


def test_independent_snapshot_redacts_malformed_git_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      repo: Path,
) -> None:
    destination = tmp_path / "snapshot"
    newline_name = "GIT_TRACE_PRIVATE\nSECRET_NAME"
    oversized_name = "GIT_CONFIG_" + "S" * 500
    private_value = "PRIVATE_ENV_VALUE_SENTINEL"
    for key in tuple(os.environ):
        if key.startswith("GIT_CONFIG"):
            monkeypatch.delenv(key)
    for index in range(30):
        monkeypatch.setenv(f"GIT_TRACE_TEST_{index:02d}", "PRIVATE")
    monkeypatch.setenv(newline_name, private_value)
    monkeypatch.setenv(oversized_name, private_value)
    with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
        git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
    message = str(raised.value)
    # The 320-byte diagnostic budget admits these 16 safe names before the 20-name cap.
    assert message == (
        "snapshot refuses inherited Git repository, configuration, diff, or trace overrides; offending variables: "
        "GIT_TRACE_TEST_00, GIT_TRACE_TEST_01, GIT_TRACE_TEST_02, GIT_TRACE_TEST_03, "
        "GIT_TRACE_TEST_04, GIT_TRACE_TEST_05, GIT_TRACE_TEST_06, GIT_TRACE_TEST_07, "
        "GIT_TRACE_TEST_08, GIT_TRACE_TEST_09, GIT_TRACE_TEST_10, GIT_TRACE_TEST_11, "
        "GIT_TRACE_TEST_12, GIT_TRACE_TEST_13, GIT_TRACE_TEST_14, GIT_TRACE_TEST_15; "
        "and 14 more; 2 invalid Git variable names"
    )
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


def test_independent_snapshot_rejects_indexed_config_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path,
) -> None:
    destination = tmp_path / "snapshot"
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{count}", "core.hooksPath")
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


def test_assert_is_worktree_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path / "does-not-exist")


def test_git_workspace_detection_distinguishes_nested_linked_and_standalone(repo: Path, tmp_path: Path) -> None:
    # Owner: Git discovery for bounded inventories. Root-only preflight rejects
    # nested directories; a .git-directory check loses linked worktrees.
    nested = repo / "src"
    nested.mkdir()
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
    try:
        for directory in (repo, nested, linked):
            assert git_ops.is_git_workspace(directory)
        standalone = tmp_path / "standalone"
        standalone.mkdir()
        assert not git_ops.is_git_workspace(standalone)
        assert git_ops.is_inside_worktree(repo) and git_ops.is_inside_worktree(linked)
        with pytest.raises(NotAWorktreeError, match="top-level"):
            git_ops.assert_is_worktree(nested)
        with pytest.raises(NotAWorktreeError, match="not inside"):
            git_ops.assert_is_worktree(standalone)
        assert not git_ops.is_inside_worktree(nested) and not git_ops.is_inside_worktree(standalone)
    finally:
        _git(repo, "worktree", "remove", str(linked))


@pytest.mark.parametrize("kind", ["broken-link", "invalid-gitfile", "bare"])
def test_git_workspace_detection_refuses_unusable_git_storage(tmp_path: Path, kind: str) -> None:
    # Owner: fail-closed discovery. A failed rev-parse must not authorize an
    # unrestricted source walk in damaged Git storage or a bare repository.
    directory = tmp_path / "workspace"
    directory.mkdir()
    if kind == "bare":
        _git(directory, "init", "--bare")
    elif kind == "broken-link":
        (directory / ".git").symlink_to(directory / "missing")
    else:
        (directory / ".git").write_text("gitdir: missing\n")
    nested = directory / "src"
    nested.mkdir()
    with pytest.raises(GitError):
        git_ops.is_git_workspace(nested)


def test_git_workspace_detection_allows_standalone_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    # Owner: standalone discovery on a separate filesystem. Ordinary tmp_path
    # storage does not exercise Git's filesystem-boundary diagnostic.
    directory = Path("/dev")
    if not directory.is_mount():
        pytest.skip("/dev is not a separate filesystem")
    monkeypatch.delenv("GIT_DISCOVERY_ACROSS_FILESYSTEM", raising=False)
    monkeypatch.delenv("GIT_CEILING_DIRECTORIES", raising=False)
    probe = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, check=False, timeout=5, env={**os.environ, "LC_ALL": "C"},
    )
    if probe.returncode != 128 or "Stopping at filesystem boundary" not in probe.stderr:
        pytest.skip("Git does not report filesystem-boundary discovery for /dev")
    assert not git_ops.is_git_workspace(directory)


@pytest.mark.parametrize("marker", [None, "broken-link", "invalid-gitfile"])
def test_git_workspace_detection_checks_markers_after_filesystem_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str | None,
) -> None:
    # Owner: fail-closed discovery. The new diagnostic must still inspect real
    # ancestor markers; existing damaged-storage tests use a different response.
    if marker == "broken-link":
        (tmp_path / ".git").symlink_to(tmp_path / "missing")
    elif marker == "invalid-gitfile":
        (tmp_path / ".git").write_text("gitdir: missing\n")
    nested = tmp_path / "src"
    nested.mkdir()
    _patch_subprocess_run(
        monkeypatch, returncode=128,
        stderr=(
            "fatal: not a git repository (or any parent up to mount point /mounted volume)\n"
            "Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).\n"
        ),
    )
    if marker is None:
        assert not git_ops.is_git_workspace(nested)
    else:
        with pytest.raises(GitError):
            git_ops.is_git_workspace(nested)


@pytest.mark.parametrize("diagnostic", [
    "fatal: not a git repository: unexpected failure\n",
    "fatal: not a git repository (or any parent up to mount point /)\n",
    "fatal: not a git repository (or any of the parent directories) but discovery failed\n",
])
def test_git_workspace_detection_rejects_unrecognized_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic: str,
) -> None:
    # Owner: inventory confinement. Accepting every no-repository prefix would
    # turn unexpected Git failures into unrestricted recursive source scans.
    _patch_subprocess_run(monkeypatch, returncode=128, stderr=diagnostic)
    with pytest.raises(GitError):
        git_ops.is_git_workspace(tmp_path)


def test_git_workspace_detection_propagates_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Owner: transport ambiguity. No Git answer means no permission to recurse;
    # real-repository cases cannot deterministically force a hung subprocess.
    calls = 0

    def timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.is_git_workspace(tmp_path)
    assert calls == 1


def test_head_sha_raises_on_empty_repo(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    with pytest.raises(GitError):
        git_ops.head_sha(repo)


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


def test_ref_exists_rejects_leading_dash(repo: Path) -> None:
    assert git_ops.ref_exists(repo, "-not-a-ref") is False
    assert git_ops.ref_exists(repo, "--exec=evil") is False


@pytest.mark.parametrize(
    ("ancestor", "descendant", "expected"),
    [
    pytest.param("missing", "HEAD", False, id="missing_ref"),
    pytest.param("-not-a-ref", "HEAD", False, id="leading_dash_ref"),
],
)
def test_is_ancestor_reports_relationship(repo: Path, ancestor: str, descendant: str, expected: bool) -> None:
    assert git_ops.is_ancestor(repo, ancestor, descendant) is expected


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
    assert git_ops.remote_url(repo) is None
    _git(repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    _git(repo, "remote", "add", "upstream", "git@github.com:other/widgets.git")
    assert git_ops.remote_urls(repo) == {
        "origin": "https://github.com/acme/widgets.git", "upstream": "git@github.com:other/widgets.git",
    }
    assert git_ops.remote_url(repo) == "https://github.com/acme/widgets.git"
    assert git_ops.remote_url(repo, "absent") is None
    _git(repo, "config", "remote.broken.fetch", "+refs/heads/*:refs/remotes/broken/*")
    with pytest.raises(GitError, match="remote 'broken'.*fetch URL"):
        git_ops.remote_urls(repo)


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


@pytest.mark.parametrize("head", ["HEAD", "f" * 40])
def test_resolve_pr_merge_base_requires_exact_present_head(repo: Path, head: str) -> None:
    with pytest.raises(GitError, match="exact PR head"):
        git_ops.resolve_pr_merge_base(repo, [], "refs/heads/main", head)


@pytest.mark.parametrize("local_ref", ["main", "refs/heads/-bad", "refs/heads/main~1"])
def test_resolve_pr_merge_base_rejects_invalid_base_ref(repo: Path, local_ref: str) -> None:
    expected = "PR base refs/heads/-bad is unavailable" if local_ref == "refs/heads/-bad" else "invalid PR base ref"
    with pytest.raises(GitError, match=f"^{expected}"):
        git_ops.resolve_pr_merge_base(repo, [], local_ref, git_ops.head_sha(repo))


def test_changed_files_against_raises_when_git_query_fails(tmp_path: Path) -> None:
    """Safety guards must not mistake a failed enumeration for a clean tree."""
    with pytest.raises(GitError):
        git_ops.changed_files_against(tmp_path, "HEAD")


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


def test_grep_fixed_matches_returns_path_pattern_pairs(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / "widget_user.py").write_text("import widget\n")
    (repo / "gadget_user.ts").write_text("gadget\n")
    (repo / "partial.py").write_text("widget_factory\n")
    (repo / "notes.md").write_text("widget gadget\n")
    _git(repo, "add", "widget_user.py", "gadget_user.ts", "partial.py", "notes.md")
    _commit(repo, "add files")
    patterns_files: list[bytes] = []
    real_run = subprocess.run
    def recording_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        argv = args[0]
        if "-f" in argv:
            patterns_files.append(Path(argv[argv.index("-f") + 1]).read_bytes())
        return real_run(*args, **kwargs)
    monkeypatch.setattr(subprocess, "run", recording_run)
    matches = git_ops.grep_fixed_matches(
        repo, ("widget", "widget", "", "gadget", "bad\x00pattern", "bad\rpattern", "bad\npattern"),
        word=True, pathspecs=("*.py", "*.ts"),
    )
    assert patterns_files == [b"widget\ngadget"]
    assert len(matches) == 2
    assert set(matches) == {("widget_user.py", "widget"), ("gadget_user.ts", "gadget")}
    assert git_ops.grep_fixed_matches(repo, ("absent_pattern",)) == []
    assert git_ops.grep_fixed_matches(repo, ("widget",), pathspecs=("partial.py",)) == [("partial.py", "widget")]


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


def test_grep_fixed_matches_empty_when_all_patterns_unsuitable(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_and_stage(repo, "widget.py", "widget\n")
    _commit(repo, "add widget")
    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("git must not run when every pattern is unsuitable")
    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_ops.grep_fixed_matches(repo, ("", "a\x00b", "a\rb", "a\nb")) == []


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


def test_create_branch_raises_when_branch_exists(repo_with_origin: Path) -> None:
    git_ops.create_branch(repo_with_origin, "daydream/dup")
    _git(repo_with_origin, "checkout", "main")
    with pytest.raises(GitError):
        git_ops.create_branch(repo_with_origin, "daydream/dup")


def test_push_branch_failure_raises_git_error(git_repo: Path) -> None:
    git_ops.create_branch(git_repo, "daydream/no-remote")
    (git_repo / "f.txt").write_text("x\n")
    git_ops.commit_paths(git_repo, [Path("f.txt")], "add f")
    with pytest.raises(GitError):
        git_ops.push_branch(git_repo, "daydream/no-remote")


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


def test_run_git_bounds_real_blob_capture_and_preserves_text_decoding(repo: Path) -> None:
    """Immutable patches need a pipe limit; post-capture checks cannot constrain allocation.

    Ordinary retry tests replace subprocess.run and cannot prove bounded draining
    of real stdout, or preserve its text newline conversion alongside raw bytes.
    """
    body = b"first\r\nsecond\rthird\n" + b"x" * (512 * 1024)
    write_and_stage(repo, "large.txt", body)
    oid = _git(repo, "rev-parse", ":large.txt")
    args = ["cat-file", "blob", oid]
    with pytest.raises(GitError, match="output exceeds the supported size"):
        git_process._run_git(repo, args, max_output_bytes=16 * 1024)
    raw = git_process._run_git(repo, args, max_output_bytes=len(body), capture_bytes=True)
    text = git_process._run_git(repo, args, max_output_bytes=len(body))
    assert raw.stdout == body
    assert text.stdout == body.decode().replace("\r\n", "\n").replace("\r", "\n")


def test_immutable_diff_refuses_to_replay_completed_setup_after_timeout(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed init can lose its response; preparation must fail without replaying it.

    Generic mutation wrappers never enter the immutable owner's temporary Git
    setup, and ordinary diff coverage cannot expose timeout recovery behavior.
    """
    from daydream.git_ops import immutable

    head = _git(repo, "rev-parse", "HEAD")
    before = _repository_identity(repo)
    original_run = subprocess.run
    initialized: list[Path] = []

    def completed_init_timeout(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        result = original_run(argv, **kwargs)
        if argv[:2] == ["git", "init"]:
            prepared = Path(kwargs["cwd"])
            assert (prepared / ".git/config").is_file()
            initialized.append(prepared)
            if len(initialized) == 1:
                raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])
        return result

    monkeypatch.setattr(subprocess, "run", completed_init_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        immutable.immutable_review_diff(repo, head, head)
    assert len(initialized) == 1 and not initialized[0].exists()
    assert _repository_identity(repo) == before


def test_immutable_diff_refuses_expansion_without_buffering_the_whole_patch(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tree fits while their replacement patch exceeds the independent capture limit.

    Oversized-source rejection cannot exercise this path: it stops at inventory.
    The prior helper rejected only after buffering the complete textual patch.
    """
    from daydream.git_ops import immutable

    write_and_stage(repo, "replace.txt", "a" * (3 * 1024 * 1024) + "\n")
    base = _commit(repo, "old text within source bounds")
    write_and_stage(repo, "replace.txt", "b" * (3 * 1024 * 1024) + "\n")
    head = _commit(repo, "replacement text within source bounds")
    monkeypatch.setattr(immutable, "_MAX_SOURCE_BYTES", 4 * 1024 * 1024)
    tracemalloc.start()
    try:
        with pytest.raises(GitError, match="supported size"):
            immutable.immutable_review_diff(repo, base, head)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024


@pytest.mark.parametrize("mode", ["stdout-overflow", "stderr-overflow", "timeout", "nonzero"])
def test_bounded_git_capture_reaps_failures_and_retries_only_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """Reap every real launch, including timeouts before the child reaches its own code."""
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "mode = sys.argv[1]\n"
        "if mode == 'nonzero':\n"
        "    os.write(2, b'bounded failure diagnostic')\n"
        "    sys.exit(7)\n"
        "if mode.endswith('-overflow'):\n"
        "    os.write(1 if mode.startswith('stdout') else 2, b'x' * 65536)\n"
        "while True: time.sleep(1)\n"
    )
    shim.chmod(0o755)
    processes: list[subprocess.Popen[Any]] = []
    real_popen = subprocess.Popen

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setenv("PATH", str(shim_dir) + os.pathsep + os.defpath)
    expected = ("timed out" if mode == "timeout" else "bounded failure diagnostic" if mode == "nonzero"
                else "output exceeds the supported size")
    with pytest.raises(git_ops.GitTimeoutError if mode == "timeout" else GitError, match=expected):
        git_process._run_git(tmp_path, [mode], timeout=0.3, retries=1, max_output_bytes=1024,
                             error_context="bounded command failed")
    assert len(processes) == (2 if mode == "timeout" else 1)
    for process in processes:
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)


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
        ("0", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='0' must be positive; using default 60"),
    ], ids=["malformed", "zero"],
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
        ("", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='' is not a valid integer; using default 2"),
        ("-1", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='-1' is negative; using default 2"),
        ("0", 1, None),
    ], ids=["empty-warns", "negative-warns", "zero-valid"],
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
    monkeypatch: pytest.MonkeyPatch, repo: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", "1")
    ok = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="octocat/hello\n", stderr="")
    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _flaky_timeout_run(calls, ok, cmd=["gh"], timeout=7))
    assert git_ops.gh_repo_view(repo) == ("octocat", "hello")
    assert calls["n"] == 2  # timed out once, then succeeded
    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["gh"], timeout=7, calls=calls))
    with pytest.raises(git_ops.GitTimeoutError) as exc:
        git_ops.gh_pr_diff(repo, 7)
    assert calls["n"] == 2
    assert str(exc.value) == "gh pr diff 7 timed out after 7s (2 attempts)"
    assert "DAYDREAM_GH_TIMEOUT" not in caplog.text


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


def test_diff_paths_raises_on_invalid_ref(repo: Path) -> None:
    with pytest.raises(GitError):
        git_ops.diff_paths(repo, "definitely-not-a-ref", "HEAD", ["base.txt"])


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
    ("pr", "stderr"),
    [
        (42, "no git remotes found"), (42, "HTTP 401: authentication required"),
        (42, "Unknown JSON field: futureField"),
        (42, "GraphQL: Could not resolve to a PullRequest with the number of 41. (repository.pullRequest)"),
        (None, 'prefix: no pull requests found for branch "feature"'), (42, "HTTP 404: resource not found"),
    ],
)
def test_gh_pr_view_unknown_failures_raise(
    repo: Path, monkeypatch: pytest.MonkeyPatch, pr: int | None, stderr: str,
) -> None:
    _patch_subprocess_run(monkeypatch, returncode=1, stderr=stderr)
    with pytest.raises(GitError, match=re.escape(stderr)):
        git_ops.gh_pr_view(repo, pr)


@pytest.mark.parametrize("stdout", [
        "not-json",
        "[]",
    ])
def test_gh_pr_view_rejects_invalid_json_shape(repo: Path, monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    _patch_subprocess_run(monkeypatch, stdout=stdout)
    with pytest.raises(GitError, match="invalid JSON|JSON object"):
        git_ops.gh_pr_view(repo, 42)


@pytest.mark.parametrize("stdout", [
        "not-json",
        "{}",
        '[{"number": 1}, null]',
    ])
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


@pytest.mark.parametrize("slug", [
        "owner/repo/extra",
        "owner/ ",
    ])
def test_gh_repo_view_required_rejects_invalid_slug(repo: Path, monkeypatch: pytest.MonkeyPatch, slug: str) -> None:
    _patch_subprocess_run(monkeypatch, stdout=slug + "\n")
    with pytest.raises(GitError, match="invalid repository slug"):
        git_ops.gh_repo_view_required(repo)


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


def test_clone_of_linked_worktree_materializes_head_and_staged_patch(
    tmp_path: Path, linked_worktree: tuple[Path, Path],
) -> None:
    _main, linked = linked_worktree
    parser = linked / "services/taste/parser.go"
    parser.write_text("package taste\n\n// staged spike\nfunc Spiked() {}\n")
    binary = bytes(range(256))
    write_and_stage(linked, "blob.bin", binary)
    _git(linked, "add", "services/taste/parser.go")
    source_patch = git_ops.staged_patch(linked)
    assert source_patch and b"GIT binary patch" in source_patch
    clone = tmp_path / "spike-clone"
    git_ops.clone(str(linked), clone)
    head = git_ops.head_sha(linked)
    assert (clone / ".git").is_dir() and git_ops.head_sha(clone) == head
    assert git_ops.remote_url(clone) == str(linked)
    shutil.copy2(parser, clone / "services/taste/parser.go")
    shutil.copy2(linked / "blob.bin", clone / "blob.bin")
    git_ops.apply_staged_patch(clone, source_patch)
    assert git_ops.staged_patch(clone) == source_patch
    assert _git(clone, "diff", "--", "services/taste/parser.go", "blob.bin") == ""
    assert (clone / "blob.bin").read_bytes() == binary
    git_ops.remove_remote(clone)
    assert git_ops.remote_url(clone) is None and git_ops.head_sha(clone) == head


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
    ("blobless", "expected_flags"), [(True, ["--filter=blob:none", "--no-local"]), (False, [])],
    ids=["blobless_passes_filter_flag", "default_no_filter_flag"],
)
def test_clone_filter_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blobless: bool, expected_flags: list[str],
) -> None:
    captured: dict[str, list[str]] = {}
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    git_ops.clone("https://example.com/repo.git", tmp_path / "out", blobless=blobless, no_local=blobless)
    assert captured["cmd"] == ["git", "clone", *expected_flags, "https://example.com/repo.git", str(tmp_path / "out")]


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


def test_gh_api_jq_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}
    def fake_run_gh(repo: Path, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(["gh"], 0, stdout='"anderskev"\n', stderr="")
    monkeypatch.setattr(git_process, "_run_gh", fake_run_gh)
    assert git_ops.gh_api(tmp_path, "user", jq=".login") == ["anderskev"]
    assert captured["args"] == ["api", "--jq", "(.login) | @json", "user"]


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


def test_gh_secret_set_requires_exactly_one_scope(fake_gh: FakeGh, git_repo: Path) -> None:
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v")
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v", org="acme", repo_slug="o/r")


def test_gh_pr_create_failure_raises_git_error(fake_gh: FakeGh, git_repo: Path) -> None:
    # No "pr-create" response configured → the shim exits non-zero.
    with pytest.raises(GitError):
        git_ops.gh_pr_create(git_repo, head="b", base="main", title="t", body="b")


_FILE_AT_REF_SHA = "0123456789abcdef0123456789abcdef01234567"


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
        ("../o/r", _FILE_AT_REF_SHA, "a.py"), ("o/r", "main", "a.py"),
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


def test_clone_error_message_never_contains_remote_url(tmp_path: Path) -> None:
    with pytest.raises(GitError, match="git clone .* failed") as excinfo:
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


def test_feature_history_diff_and_publication_follow_the_same_commit_window(tmp_path: Path) -> None:
    """A real branch advances through empty, untagged, tagged, divergent and published states."""
    repo, _bare = _repo_with_origin(tmp_path)
    original = [f"line {n}" for n in range(1, 31)]
    write_and_stage(repo, "shared.txt", "\n".join(original) + "\n")
    common = _commit(repo, "shared baseline")
    _git(repo, "push", "origin", "main")
    git_ops.create_branch(repo, "feat/slash-name")
    assert git_ops.upstream_ahead_count(repo, "feat/slash-name") == 0
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert not git_ops.branch_exists(repo, "absent")
    assert not git_ops.ref_exists(repo, "absent") and not git_ops.ref_exists(repo, tree)
    assert not git_ops.commit_exists(repo, "absent") and not git_ops.commit_exists(repo, tree)
    assert git_ops.merge_base(repo, "absent") is None
    assert git_ops.diff_name_only(repo, "absent", "HEAD") == []
    assert git_ops.log_shas(repo, "HEAD", since="main") == []
    assert git_ops.daydream_commits(repo, "main") is None
    feature = original.copy()
    feature[14] = "line 15 FEAT"
    write_and_stage(repo, "shared.txt", "\n".join(feature) + "\n")
    write_and_stage(repo, "keep.txt", "keep\n")
    write_and_stage(repo, "drop.txt", "drop\n")
    untagged = _commit(repo, "chore: unrelated change")
    assert git_ops.daydream_commits(repo, "main") is None
    assert git_ops.log_shas(repo, "HEAD", since="main") == [untagged]
    write_and_stage(repo, "tagged.txt", "validated\n")
    tip = _commit(repo, "fix: validated\n\nDaydream-Run: test-123\nDaydream-Version: 0.14.0")
    assert git_ops.log_shas(repo, "HEAD", since="main") == [tip, untagged]
    assert git_ops.log_shas_since(repo, "main", "HEAD") == [tip, untagged]
    tagged = git_ops.daydream_commits(repo, "main")
    assert tagged is not None and "fix: validated" in tagged and "unrelated change" not in tagged
    _git(repo, "checkout", "main")
    main = original.copy()
    main[14] = "line 15 MAIN"
    write_and_stage(repo, "shared.txt", "\n".join(main) + "\n")
    main_tip = _commit(repo, "advance main")
    on_main = git_ops.diff(repo, "main")
    assert "-line 15\n+line 15 MAIN" in on_main
    _git(repo, "checkout", "feat/slash-name")
    assert git_ops.list_local_branches(repo) == {"main": main_tip, "feat/slash-name": tip}
    assert git_ops.merge_base(repo, "main") == common
    assert git_ops.is_ancestor(repo, common, tip) and not git_ops.is_ancestor(repo, tip, common)
    direct = git_ops.diff_paths(repo, "main", "HEAD", ["shared.txt"])
    focused = git_ops.diff_paths(repo, "main", "HEAD", ["shared.txt"], unified=1, merge_base_diff=True)
    broad = git_ops.diff_paths(repo, "main", "HEAD", ["shared.txt"], unified=10, merge_base_diff=True)
    assert "-line 15 MAIN" in direct and "+line 15 FEAT" in direct
    assert "MAIN" not in focused and "-line 15\n+line 15 FEAT" in focused
    assert "keep.txt" not in broad and "drop.txt" not in broad
    assert " line 5\n" not in focused and " line 5\n" in broad
    full = git_ops.diff(repo, "main")
    excluded = git_ops.diff(repo, "main", exclude=["drop.txt"])
    assert "keep.txt" in full and "+keep" in full and "drop.txt" in full
    assert "keep.txt" in excluded and "drop.txt" not in excluded
    assert not git_ops.remote_contains_commit(repo, "feat/slash-name", tip)
    git_ops.push_branch(repo, "feat/slash-name")
    assert git_ops.remote_contains_commit(repo, "feat/slash-name", tip)
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "feat/slash-name")
    assert git_ops.branch_exists(repo, "feat/slash-name") and git_ops.ref_exists(repo, "feat/slash-name")
    assert not git_ops.commit_exists(repo, "feat/slash-name")


@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_materializes_mixed_index_and_worktree_changes(
    tmp_path: Path,
    repo: Path,
    include_untracked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One snapshot must preserve simultaneous path-type transitions and exact filename bytes."""
    monkeypatch.delenv("GIT_EXEC_PATH", raising=False)
    for name in ("staged/a.txt", "staged/b.txt", "working/child.txt", "to-directory"):
        write_and_stage(repo, name, "old\n")
    for name in (" leading and trailing ", "line\nbreak"):
        write_and_stage(repo, name, b"tracked\x00\xff")
    _commit(repo, "mixed source tree")
    shutil.rmtree(repo / "staged")
    (repo / "staged").write_bytes(b"staged replacement\x00\xff")
    (repo / "to-directory").unlink()
    (repo / "to-directory").mkdir()
    (repo / "to-directory/child.txt").write_bytes(b"child\x00\xff")
    _git(repo, "add", "-A")
    shutil.rmtree(repo / "working")
    (repo / "working").write_bytes(b"working replacement\x00\xfe")
    for name in (" scratch ", "scratch\nline"):
        (repo / name).write_bytes(b"untracked\x00\xfe")
    before = _repository_identity(repo)
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=include_untracked)
    assert (snapshot.repo / "staged").read_bytes() == b"staged replacement\x00\xff"
    assert (snapshot.repo / "to-directory/child.txt").read_bytes() == b"child\x00\xff"
    assert git_ops.staged_patch(snapshot.repo) == before[2]
    assert git_ops.ls_files(snapshot.repo, strict=True) == git_ops.ls_files(repo, strict=True)
    assert _git(snapshot.repo, "status", "--short", "--untracked-files=no") == _git(
        repo,
        "status",
        "--short",
        "--untracked-files=no",
    )
    for name in (" leading and trailing ", "line\nbreak"):
        assert (snapshot.repo / name).read_bytes() == b"tracked\x00\xff"
    if include_untracked:
        assert (snapshot.repo / "working").read_bytes() == b"working replacement\x00\xfe"
        assert (snapshot.repo / " scratch ").read_bytes() == b"untracked\x00\xfe"
        assert (snapshot.repo / "scratch\nline").read_bytes() == b"untracked\x00\xfe"
    else:
        assert not (snapshot.repo / "working").is_file()
        assert not (snapshot.repo / " scratch ").exists()
        assert not (snapshot.repo / "scratch\nline").exists()
    assert _repository_identity(repo) == before

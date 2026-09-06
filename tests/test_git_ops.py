"""Tests for :mod:`daydream.git_ops`.

These tests use real ``git`` (and optionally ``gh``) against ``tmp_path``
fixtures.  No subprocess mocking — every code path is exercised against an
actual repository.  ``gh``-dependent tests are skipped when the binary is
unavailable.
"""

from __future__ import annotations

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
)
from tests.conftest import _make_repo_with_main
from tests.harness.git_helpers import bare_remote as _bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import configure_identity as _configure_identity
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo

# --- assert_is_worktree / is_inside_worktree --------------------------------


def test_resolve_diff_merge_base_prefers_present_origin_ref(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "upstream.py").write_text("UPSTREAM = 1\n")
    _git(repo, "add", "upstream.py")
    remote_tip = _commit(repo, "remote advancement")
    _git(repo, "branch", "feature", remote_tip)
    _git(repo, "reset", "--hard", base)
    _git(repo, "update-ref", "refs/remotes/origin/main", remote_tip)
    _git(repo, "checkout", "feature")
    (repo / "feature.py").write_text("FEATURE = 1\n")
    _git(repo, "add", "feature.py")
    head = _commit(repo, "feature")

    assert git_ops.merge_base(repo, "main", head) == base
    assert git_ops.resolve_diff_merge_base(repo, "main", head) == remote_tip


def test_resolve_diff_merge_base_carries_only_ancestor_of_dangling_base(
    tmp_path: Path,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    common = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.py").write_text("FEATURE = 1\n")
    _git(repo, "add", "feature.py")
    head = _commit(repo, "feature")
    _git(repo, "checkout", "--detach", common)
    (repo / "side.py").write_text("SIDE = 1\n")
    _git(repo, "add", "side.py")
    dangling_tip = _commit(repo, "dangling selected base")
    _git(repo, "checkout", "feature")

    assert git_ops.resolve_diff_merge_base(repo, dangling_tip, head) == common


def _install_diff_base_git_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    head: str,
) -> Path:
    """Install an external Git shim for resolver boundary failures."""
    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "diff base git shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys, time\n"
        "args = sys.argv[1:]\n"
        "mode = os.environ['DAYDREAM_TEST_GIT_SHIM_MODE']\n"
        "real = os.environ['DAYDREAM_TEST_REAL_GIT']\n"
        "head = os.environ['DAYDREAM_TEST_HEAD']\n"
        "is_preference = args[:2] == ['rev-parse', '--verify'] and "
        "len(args) == 3 and args[2].startswith('refs/remotes/origin/')\n"
        "is_head = args[:2] == ['rev-parse', '--verify'] and "
        "len(args) == 3 and args[2] == head + '^{commit}'\n"
        "if mode == 'malformed-head' and is_head:\n"
        "    print('PRIVATE_STDOUT_SENTINEL')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'uppercase-head' and is_head:\n"
        "    print(head.upper())\n"
        "    raise SystemExit(0)\n"
        "if mode == 'abbreviated-head' and is_head:\n"
        "    print(head[:12])\n"
        "    raise SystemExit(0)\n"
        "if mode == 'sha256-head' and is_head:\n"
        "    print(head + head[:24])\n"
        "    raise SystemExit(0)\n"
        "if mode == 'invalid-utf8-head' and is_head:\n"
        "    os.write(1, b'\\xff\\xfe\\n')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'malformed-merge' and args[:1] == ['merge-base']:\n"
        "    print('PRIVATE_MERGE_SENTINEL')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'timeout-preference' and is_preference:\n"
        "    time.sleep(30)\n"
        "if mode == 'remove-after-preference' and is_preference:\n"
        "    result = subprocess.run([real, *args])\n"
        "    os.unlink(sys.argv[0])\n"
        "    raise SystemExit(result.returncode)\n"
        "raise SystemExit(subprocess.run([real, *args]).returncode)\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("DAYDREAM_TEST_GIT_SHIM_MODE", mode)
    monkeypatch.setenv("DAYDREAM_TEST_REAL_GIT", real_git)
    monkeypatch.setenv("DAYDREAM_TEST_HEAD", head)
    monkeypatch.setenv("PATH", str(shim_dir))
    return shim


@pytest.mark.parametrize(
    ("mode", "expected_message", "private_fragment"),
    [
        (
            "malformed-head",
            "branch-focus recorded head commit cannot be resolved",
            "PRIVATE_STDOUT_SENTINEL",
        ),
        (
            "invalid-utf8-head",
            "branch-focus recorded head commit probe failed",
            "\\xff\\xfe",
        ),
        (
            "uppercase-head",
            "branch-focus recorded head commit cannot be resolved",
            "A PRIVATE VALUE THAT CANNOT APPEAR",
        ),
        (
            "abbreviated-head",
            "branch-focus recorded head commit cannot be resolved",
            "A PRIVATE VALUE THAT CANNOT APPEAR",
        ),
        (
            "sha256-head",
            "branch-focus recorded head commit cannot be resolved",
            "A PRIVATE VALUE THAT CANNOT APPEAR",
        ),
        (
            "malformed-merge",
            "branch-focus diff merge-base cannot be resolved",
            "PRIVATE_MERGE_SENTINEL",
        ),
        (
            "remove-after-preference",
            "branch-focus recorded head commit probe failed",
            "diff base git shim",
        ),
    ],
)
def test_resolve_diff_merge_base_rejects_and_redacts_external_git_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_message: str,
    private_fragment: str,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    _install_diff_base_git_shim(tmp_path, monkeypatch, mode=mode, head=head)

    with pytest.raises(GitError) as raised:
        git_ops.resolve_diff_merge_base(repo, "main", head)

    message = str(raised.value)
    assert message == expected_message
    assert private_fragment not in message
    assert str(repo) not in message
    assert head not in message
    assert len(message) < 200


def test_resolve_diff_merge_base_redacts_external_git_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    _install_diff_base_git_shim(
        tmp_path, monkeypatch, mode="timeout-preference", head=head,
    )

    with pytest.raises(GitError) as raised:
        git_ops.resolve_diff_merge_base(repo, "main", head)

    assert str(raised.value) == "branch-focus preferred base probe failed"
    assert str(repo) not in str(raised.value)
    assert head not in str(raised.value)


@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_preserves_exact_filename_bytes(
    tmp_path: Path, include_untracked: bool,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    tracked = [" leading and trailing ", "line\nbreak"]
    untracked = [" scratch ", "scratch\nline"]
    for name in tracked:
        (repo / name).write_bytes(b"tracked\x00\xff")
    _git(repo, "add", "-A")
    _commit(repo, "unusual paths")
    for name in untracked:
        (repo / name).write_bytes(b"scratch\x00\xfe")
    snapshot = git_ops.prepare_independent_snapshot(
        repo, tmp_path / "independent", include_untracked=include_untracked,
    )
    assert set(git_ops.ls_tree_files(snapshot.repo, "HEAD", strict=True)) >= set(tracked)
    for name in tracked:
        assert (snapshot.repo / name).read_bytes() == b"tracked\x00\xff"
    for name in untracked:
        assert (snapshot.repo / name).exists() is include_untracked
        if include_untracked:
            assert (snapshot.repo / name).read_bytes() == b"scratch\x00\xfe"


@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_preserves_staged_directory_to_file_change(
    tmp_path: Path, include_untracked: bool,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    nested = repo / "x"
    nested.mkdir()
    (nested / "a.txt").write_text("old a\n", encoding="utf-8")
    (nested / "b.txt").write_text("old b\n", encoding="utf-8")
    _git(repo, "add", "x")
    _commit(repo, "add tracked directory")
    shutil.rmtree(nested)
    nested.write_bytes(b"replacement file\x00\xff")
    _git(repo, "add", "-A")
    before_head = _git(repo, "rev-parse", "HEAD")
    before_status = _git(repo, "status", "--short")
    before_patch = git_ops.staged_patch(repo)
    before_refs = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")

    snapshot = git_ops.prepare_independent_snapshot(
        repo, tmp_path / "snapshot", include_untracked=include_untracked,
    )

    assert (snapshot.repo / "x").is_file()
    assert (snapshot.repo / "x").read_bytes() == b"replacement file\x00\xff"
    assert not (snapshot.repo / "x" / "a.txt").exists()
    assert not (snapshot.repo / "x" / "b.txt").exists()
    assert _git(snapshot.repo, "status", "--short") == before_status
    assert git_ops.staged_patch(snapshot.repo) == before_patch
    assert git_ops.ls_files(snapshot.repo, strict=True) == git_ops.ls_files(
        repo, strict=True,
    )
    assert _git(repo, "rev-parse", "HEAD") == before_head
    assert _git(repo, "status", "--short") == before_status
    assert git_ops.staged_patch(repo) == before_patch
    assert _git(repo, "for-each-ref", "--format=%(refname) %(objectname)") == before_refs
    assert (repo / "x").read_bytes() == b"replacement file\x00\xff"


@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_preserves_unstaged_directory_to_file_change(
    tmp_path: Path, include_untracked: bool,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    nested = repo / "x"
    nested.mkdir()
    (nested / "child.txt").write_text("old child\n", encoding="utf-8")
    _git(repo, "add", "x/child.txt")
    _commit(repo, "add tracked directory")
    shutil.rmtree(nested)
    nested.write_bytes(b"untracked replacement\x00\xff")
    before_head = _git(repo, "rev-parse", "HEAD")
    before_status = _git(repo, "status", "--short")
    before_patch = git_ops.staged_patch(repo)
    before_refs = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")

    snapshot = git_ops.prepare_independent_snapshot(
        repo, tmp_path / "snapshot", include_untracked=include_untracked,
    )

    assert not (snapshot.repo / "x" / "child.txt").exists()
    if include_untracked:
        assert (snapshot.repo / "x").is_file()
        assert (snapshot.repo / "x").read_bytes() == b"untracked replacement\x00\xff"
        assert _git(snapshot.repo, "status", "--short") == before_status
    else:
        assert not (snapshot.repo / "x").is_file()
        assert _git(snapshot.repo, "status", "--short") == "D x/child.txt"
    assert git_ops.staged_patch(snapshot.repo) == before_patch
    assert _git(repo, "rev-parse", "HEAD") == before_head
    assert _git(repo, "status", "--short") == before_status
    assert git_ops.staged_patch(repo) == before_patch
    assert _git(repo, "for-each-ref", "--format=%(refname) %(objectname)") == before_refs
    assert (repo / "x").read_bytes() == b"untracked replacement\x00\xff"


def test_independent_snapshot_preserves_staged_file_to_directory_change(
    tmp_path: Path,
) -> None:
    repo = _make_repo_with_main(tmp_path)
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

    snapshot = git_ops.prepare_independent_snapshot(
        repo, tmp_path / "snapshot", include_untracked=False,
    )

    assert (snapshot.repo / "x").is_dir()
    assert (snapshot.repo / "x" / "child.txt").read_bytes() == (
        b"replacement child\x00\xff"
    )
    assert _git(snapshot.repo, "status", "--short") == before_status
    assert git_ops.staged_patch(snapshot.repo) == before_patch
    assert _git(repo, "status", "--short") == before_status


@pytest.mark.parametrize("include_untracked", [False, True])
def test_independent_snapshot_preserves_non_utf8_paths(
    tmp_path: Path, include_untracked: bool,
) -> None:
    import errno

    repo = _make_repo_with_main(tmp_path)
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
    snapshot = git_ops.prepare_independent_snapshot(
        repo, tmp_path / "snapshot", include_untracked=include_untracked,
    )
    assert (snapshot.repo / tracked).read_bytes() == b"tracked bytes"
    assert (snapshot.repo / scratch).exists() is include_untracked
    if include_untracked:
        assert (snapshot.repo / scratch).read_bytes() == b"scratch bytes"


def test_independent_snapshot_rejects_destination_parent_symlink(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    real_clone = git_ops.clone

    def clone_with_alternates(
        remote_url: str, target: Path, *, blobless: bool = False,
        no_local: bool = False, timeout: int = 300,
    ) -> None:
        assert no_local
        real_clone(remote_url, target, blobless=blobless, no_local=no_local, timeout=timeout)
        (target / ".git" / "objects" / "info" / "alternates").write_text(
            str(repo / ".git" / "objects") + "\n",
        )

    monkeypatch.setattr(git_ops, "clone", clone_with_alternates)
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
    "GIT_CONFIG_PARAMETERS", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS", "GIT_TRACE",
    "GIT_EXEC_PATH",
])
def test_independent_snapshot_rejects_inherited_git_overrides_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    variable: str, empty: bool, unborn: bool,
) -> None:
    repo = tmp_path / "source"
    _init_repo(repo)
    (repo / "app.py").write_bytes(b"VALUE = 1\n")
    _git(repo, "add", "app.py")
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
) -> None:
    repo = _make_repo_with_main(tmp_path)
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


def test_independent_snapshot_redacts_malformed_names_at_process_boundary(
    tmp_path: Path,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    destination = tmp_path / "snapshot"
    newline_name = "GIT_TRACE_PRIVATE\nSECRET_NAME"
    oversized_name = "GIT_CONFIG_" + "S" * 500
    private_value = "PRIVATE_ENV_VALUE_SENTINEL"
    child_env = {
        name: value
        for name, value in os.environ.items()
        if not (
            name in git_ops._SNAPSHOT_GIT_REDIRECTS  # noqa: SLF001 - boundary fixture
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
        [sys.executable, "-c", script, str(repo), str(destination)],
        check=False,
        capture_output=True,
        text=True,
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
) -> None:
    repo = _make_repo_with_main(tmp_path)
    for index in range(30):
        monkeypatch.setenv(f"GIT_TRACE_TEST_{index:02d}", "PRIVATE")

    with pytest.raises(git_ops.SnapshotPreparationError) as raised:
        git_ops.prepare_independent_snapshot(
            repo, tmp_path / "snapshot", include_untracked=False,
        )

    message = str(raised.value)
    assert "GIT_TRACE_TEST_00" in message
    assert "GIT_TRACE_TEST_15" in message
    assert "GIT_TRACE_TEST_16" not in message
    assert "and 14 more" in message
    assert "PRIVATE" not in message
    assert len(message) < 500


def test_independent_snapshot_known_git_violation_skips_exec_path_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    trace_file = tmp_path / "git-trace-side-effect.log"
    monkeypatch.setenv("GIT_TRACE", str(trace_file))
    monkeypatch.setenv("GIT_EXEC_PATH", str(tmp_path / "invalid git exec path"))

    with pytest.raises(git_ops.SnapshotPreparationError) as raised:
        git_ops.prepare_independent_snapshot(
            repo, tmp_path / "snapshot", include_untracked=False,
        )

    message = str(raised.value)
    assert "GIT_TRACE" in message
    assert str(trace_file) not in message
    assert not trace_file.exists()


def test_independent_snapshot_redacts_invalid_utf8_exec_path_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo_with_main(tmp_path)
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
        git_ops.prepare_independent_snapshot(
            repo, tmp_path / "snapshot", include_untracked=False,
        )

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
) -> None:
    repo = _make_repo_with_main(tmp_path)
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
) -> None:
    repo = _make_repo_with_main(tmp_path)
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{count}", "commit.gpgsign")
    monkeypatch.setenv(f"GIT_CONFIG_VALUE_{count}", signing)
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(count + 1))
    before = dict(os.environ)
    snapshot = git_ops.prepare_independent_snapshot(
        repo, tmp_path / "snapshot", include_untracked=False,
    )
    assert dict(os.environ) == before
    assert _git(snapshot.repo, "config", "--get", "commit.gpgsign") == signing
    assert _git(snapshot.repo, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    _git(snapshot.repo, "fsck", "--full", "--no-dangling")
    assert any((snapshot.repo / ".git" / "objects").rglob("*.pack"))
    assert git_ops.object_alternates(snapshot.repo, strict=True) == ()


def test_independent_snapshot_accepts_git_exported_default_exec_path_in_pre_push(
    tmp_path: Path,
) -> None:
    """A real pre-push hook may pass Git's own helper path to the snapshot."""
    repo = _make_repo_with_main(tmp_path)
    remote = _bare_remote(tmp_path / "remote.git")
    _git(repo, "remote", "add", "origin", str(remote))
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
        ["git", "--exec-path"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env=clean_exec_env,
    ).stdout.removesuffix("\n")

    pushed = subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=push_env,
    )

    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    assert dict(os.environ) == before
    assert observed.read_text(encoding="utf-8") == expected_exec_path
    assert _git(destination, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    _git(destination, "fsck", "--full", "--no-dangling")
    assert git_ops.object_alternates(destination, strict=True) == ()


def test_independent_snapshot_rejects_exec_path_when_default_query_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    destination = tmp_path / "snapshot"
    monkeypatch.setenv("GIT_EXEC_PATH", "/trusted-looking/git-core")
    monkeypatch.setenv("PATH", str(tmp_path / "missing-bin"))
    before = dict(os.environ)

    with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git"):
        git_ops.prepare_independent_snapshot(
            repo, destination, include_untracked=False,
        )

    assert not destination.exists()
    assert dict(os.environ) == before


def test_independent_snapshot_unborn_detection_rejects_corrupt_head(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / ".git" / "HEAD").write_text("invalid head contents\n")
    with pytest.raises(GitError):
        git_ops.is_unborn_head(repo)


def test_independent_snapshot_symbolic_head_is_not_ambiguous_with_tag(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "tag", "main")
    assert git_ops.symbolic_head(repo, strict=True) == "main"
    assert not git_ops.is_unborn_head(repo)
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
    assert _git(snapshot.repo, "for-each-ref", "--format=%(refname)", "refs/heads") == _git(
        repo, "for-each-ref", "--format=%(refname)", "refs/heads",
    )


def test_clone_no_local_requests_independent_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def record_clone(remote_url: str, cmd: list[str], timeout: int) -> None:
        calls.append(cmd)

    monkeypatch.setattr(git_ops, "_run_clone", record_clone)
    git_ops.clone("source", tmp_path / "snapshot", no_local=True)
    assert calls == [["git", "clone", "--no-local", "source", str(tmp_path / "snapshot")]]


@pytest.mark.parametrize("name", [" leading and trailing ", "line\nbreak", "café.py"])
def test_diff_name_only_strict_preserves_exact_git_paths(tmp_path: Path, name: str) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / name).write_bytes(b"content\n")
    _git(repo, "add", "--", name)
    _commit(repo, "unusual path")
    assert git_ops.diff_name_only_strict(repo, "HEAD^", "HEAD") == [name]
    assert git_ops.diff_name_only_strict(repo, "HEAD", "HEAD") == []


def test_assert_is_worktree_passes_for_real_repo(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    git_ops.assert_is_worktree(repo)
    assert git_ops.is_inside_worktree(repo) is True


def test_assert_is_worktree_rejects_non_repo(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path)
    assert git_ops.is_inside_worktree(tmp_path) is False


def test_assert_is_worktree_rejects_org_dir(tmp_path: Path) -> None:
    """An "org" dir that contains repos is NOT itself a worktree."""
    org = tmp_path / "org"
    org.mkdir()
    _make_repo_with_main(org, name="child-repo")

    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(org)
    assert git_ops.is_inside_worktree(org) is False


def test_assert_is_worktree_rejects_subdir_of_repo(tmp_path: Path) -> None:
    """A subdir inside a repo is "inside a worktree" but not its top-level."""
    repo = _make_repo_with_main(tmp_path)
    sub = repo / "src"
    sub.mkdir()
    (sub / "x.txt").write_text("x\n")

    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(sub)
    assert git_ops.is_inside_worktree(sub) is False


def test_assert_is_worktree_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path / "does-not-exist")


# --- Read-only queries ------------------------------------------------------


def test_head_sha_returns_full_sha(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    assert git_ops.head_sha(repo) == expected
    assert len(expected) == 40


def test_head_sha_raises_on_empty_repo(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    with pytest.raises(GitError):
        git_ops.head_sha(repo)


def test_current_branch_on_named_branch(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.current_branch(repo) == "main"


def test_current_branch_returns_none_when_detached(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", sha)
    assert git_ops.current_branch(repo) is None


def test_default_branch_uses_origin_head(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "remote", "set-head", "origin", "main")
    assert git_ops.default_branch(repo) == "main"


def test_default_branch_falls_back_to_main(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    # No origin/HEAD — must fall back to local main.
    assert git_ops.default_branch(repo) == "main"


def test_list_local_branches_maps_names_to_oids(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path, name="list_branches")
    _git(repo, "checkout", "-b", "feat/slash-name")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "feature.txt")
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
    ],
    ids=["falls_back_to_master", "raises_when_none_present"],
)
def test_default_branch_fallback(tmp_path: Path, init_branch: str, expected: object) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", init_branch)
    _configure_identity(repo)
    (repo / "f.txt").write_text("hi\n")
    _git(repo, "add", "f.txt")
    _commit(repo, "first")
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            git_ops.default_branch(repo)
    else:
        assert git_ops.default_branch(repo) == expected


def test_remote_url_returns_url_when_remote_configured(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    assert git_ops.remote_url(repo) == str(bare)


def test_remote_url_returns_none_when_remote_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.remote_url(repo) is None


def test_remote_url_returns_none_for_unknown_remote_name(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    assert git_ops.remote_url(repo, "upstream") is None


def test_branch_exists_local(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat-local")
    assert git_ops.branch_exists(repo, "feat-local") is True


def test_branch_exists_origin_only(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "checkout", "-b", "remote-only")
    (repo / "r.txt").write_text("r\n")
    _git(repo, "add", "r.txt")
    _commit(repo, "remote-only commit")
    _git(repo, "push", "-u", "origin", "remote-only")
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "remote-only")
    assert git_ops.branch_exists(repo, "remote-only") is True


def test_branch_exists_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.branch_exists(repo, "nonexistent") is False


# --- ref_exists -------------------------------------------------------------


def _ref_raw_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _ref_abbreviated_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "--short", "HEAD")


def _ref_tag(repo: Path) -> str:
    _git(repo, "tag", "v1.0")
    return "v1.0"


def _ref_relative_commit_ish(repo: Path) -> str:
    (repo / "two.txt").write_text("two\n")
    _git(repo, "add", "two.txt")
    _commit(repo, "second")
    return "HEAD~1"


def _ref_named_branch(repo: Path) -> str:
    _git(repo, "checkout", "-b", "feat-local")
    return "feat-local"


@pytest.mark.parametrize(
    "build_ref",
    [_ref_raw_sha, _ref_abbreviated_sha, _ref_tag, _ref_relative_commit_ish, _ref_named_branch],
    ids=["raw_sha", "abbreviated_sha", "tag", "relative_commit_ish", "named_branch"],
)
def test_ref_exists_true(tmp_path: Path, build_ref: Any) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.ref_exists(repo, build_ref(repo)) is True


def test_ref_exists_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.ref_exists(repo, "nonexistent") is False
    # A tree-ish that is not a commit must be rejected, not accepted.
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert git_ops.ref_exists(repo, tree) is False


def test_ref_exists_rejects_leading_dash(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.ref_exists(repo, "-not-a-ref") is False
    assert git_ops.ref_exists(repo, "--exec=evil") is False


# --- commit_exists -----------------------------------------------------------


@pytest.mark.parametrize(
    "build_ref",
    [_ref_raw_sha, _ref_abbreviated_sha, _ref_tag, _ref_relative_commit_ish, _ref_named_branch],
    ids=["raw_sha", "abbreviated_sha", "tag", "relative_commit_ish", "named_branch"],
)
def test_commit_exists_true(tmp_path: Path, build_ref: Any) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.commit_exists(repo, build_ref(repo)) is True


def test_commit_exists_rejects_origin_only(tmp_path: Path) -> None:
    """A bare name present only as origin/<name> does NOT resolve (old behavior).

    branch_exists/ref_exists accept it via refs/remotes/origin/<ref>, but a
    plain name has no local commit-ish, so commit_exists (the rev-parse --verify probe)
    must report it as not existing.
    """
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "checkout", "-b", "remote-only")
    (repo / "r.txt").write_text("r\n")
    _git(repo, "add", "r.txt")
    _commit(repo, "remote-only commit")
    _git(repo, "push", "-u", "origin", "remote-only")
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "remote-only")
    assert git_ops.branch_exists(repo, "remote-only") is True
    assert git_ops.commit_exists(repo, "remote-only") is False


def test_commit_exists_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.commit_exists(repo, "nonexistent") is False
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert git_ops.commit_exists(repo, tree) is False


def test_commit_exists_rejects_leading_dash(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
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
def test_is_ancestor_reports_relationship(
    tmp_path: Path, ancestor: str, descendant: str, expected: bool
) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "second.txt").write_text("second\n")
    _git(repo, "add", "second.txt")
    _commit(repo, "second")
    assert git_ops.is_ancestor(repo, ancestor, descendant) is expected


# --- merge_base -------------------------------------------------------------


def test_merge_base_returns_shared_commit(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)

    _git(repo, "checkout", "-b", "feat")
    (repo / "feat.txt").write_text("feat\n")
    _git(repo, "add", "feat.txt")
    _commit(repo, "feat commit")

    _git(repo, "checkout", "main")
    (repo / "main2.txt").write_text("more main\n")
    _git(repo, "add", "main2.txt")
    _commit(repo, "main commit")

    _git(repo, "checkout", "feat")
    expected = _git(repo, "merge-base", "HEAD", "main")
    assert git_ops.merge_base(repo, "main") == expected


def test_merge_base_prefers_upstream_when_remote_ahead(tmp_path: Path) -> None:
    """Port of codex's merge_base_prefers_upstream_when_remote_ahead test."""
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")

    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "feature.txt")
    _commit(repo, "feature commit")

    # Rewrite local main as an unrelated history; track origin/main so the
    # upstream is "ahead" of the rewritten local main.
    _git(repo, "checkout", "--orphan", "rewrite")
    _git(repo, "rm", "-rf", ".")
    (repo / "new-main.txt").write_text("rewritten\n")
    _git(repo, "add", "new-main.txt")
    _commit(repo, "rewrite main")
    _git(repo, "branch", "-M", "rewrite", "main")
    _git(repo, "branch", "--set-upstream-to=origin/main", "main")

    _git(repo, "checkout", "feature")
    _git(repo, "fetch", "origin")

    expected = _git(repo, "merge-base", "HEAD", "origin/main")
    assert git_ops.merge_base(repo, "main") == expected


def test_merge_base_returns_none_for_missing_branch(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.merge_base(repo, "missing-branch") is None


def test_merge_base_returns_none_when_head_missing(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    # No commits → no HEAD.
    assert git_ops.merge_base(repo, "main") is None


@pytest.mark.parametrize(
    "refs",
    [
        pytest.param(("-no-such-ref",), id="base"),
        pytest.param(("main", "-no-such-ref"), id="head"),
    ],
)
def test_merge_base_returns_none_for_leading_dash_ref(
    tmp_path: Path,
    refs: tuple[str, ...],
) -> None:
    """Treat leading-dash revisions as invalid refs rather than Git options."""
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.merge_base(repo, *refs) is None


def test_remote_urls_is_strict_and_allows_no_remotes(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.remote_urls(repo) == {}

    _git(repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    _git(repo, "remote", "add", "upstream", "git@github.com:other/widgets.git")
    assert git_ops.remote_urls(repo) == {
        "origin": "https://github.com/acme/widgets.git",
        "upstream": "git@github.com:other/widgets.git",
    }

    _git(repo, "config", "remote.broken.fetch", "+refs/heads/*:refs/remotes/broken/*")
    with pytest.raises(GitError, match="remote 'broken'.*fetch URL"):
        git_ops.remote_urls(repo)


def test_resolve_pr_merge_base_uses_local_branch_without_remotes(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    base = git_ops.head_sha(repo)
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "feature.txt")
    head = _commit(repo, "feature")

    assert git_ops.resolve_pr_merge_base(repo, [], "refs/heads/main", head) == base


def test_resolve_pr_merge_base_prefers_present_remote_over_stale_local(tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    stale_local = git_ops.head_sha(repo)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    (repo / "base-update.txt").write_text("new base\n")
    _git(repo, "add", "base-update.txt")
    remote_base = _commit(repo, "base update")
    _git(repo, "push", "origin", "main")
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "feature.txt")
    head = _commit(repo, "feature")
    _git(repo, "branch", "-f", "main", stale_local)

    assert git_ops.resolve_pr_merge_base(
        repo, ["refs/remotes/origin/main"], "refs/heads/main", head
    ) == remote_base


def test_resolve_pr_merge_base_rejects_divergent_matching_remotes(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    oldest = git_ops.head_sha(repo)
    (repo / "base-update.txt").write_text("new base\n")
    _git(repo, "add", "base-update.txt")
    newer = _commit(repo, "base update")
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "feature.txt")
    head = _commit(repo, "feature")
    _git(repo, "update-ref", "refs/remotes/one/main", newer)
    _git(repo, "update-ref", "refs/remotes/two/main", oldest)

    with pytest.raises(GitError, match="fetch/align the base remote"):
        git_ops.resolve_pr_merge_base(
            repo,
            ["refs/remotes/one/main", "refs/remotes/two/main"],
            "refs/heads/main",
            head,
        )


@pytest.mark.parametrize("head", ["HEAD", "deadbeef", "f" * 40])
def test_resolve_pr_merge_base_requires_exact_present_head(tmp_path: Path, head: str) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError, match="exact PR head"):
        git_ops.resolve_pr_merge_base(repo, [], "refs/heads/main", head)


@pytest.mark.parametrize("local_ref", ["main", "refs/heads/-bad", "refs/heads/main~1"])
def test_resolve_pr_merge_base_rejects_invalid_base_ref(tmp_path: Path, local_ref: str) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError, match="base ref"):
        git_ops.resolve_pr_merge_base(repo, [], local_ref, git_ops.head_sha(repo))


# --- diff / log / show / grep / status / upstream_ahead_count ---------------


def test_diff_returns_changes(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "added.txt").write_text("hello\n")
    _git(repo, "add", "added.txt")
    _commit(repo, "topic commit")
    out = git_ops.diff(repo, "main")
    assert "added.txt" in out
    assert "+hello" in out


def test_diff_includes_staged_and_unstaged_worktree_changes(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "staged.txt").write_text("staged\n")
    _git(repo, "add", "staged.txt")
    (repo / "base.txt").write_text("unstaged\n")

    out = git_ops.diff(repo, "main")

    assert "staged.txt" in out
    assert "-base" in out
    assert "+unstaged" in out


def test_diff_excludes_paths(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    _commit(repo, "topic")
    out = git_ops.diff(repo, "main", exclude=["drop.txt"])
    assert "keep.txt" in out
    assert "drop.txt" not in out


def test_diff_prefers_origin_when_on_default_branch(tmp_path: Path) -> None:
    """When HEAD is on main, diff against origin/main shows unpushed commits."""
    remote_dir = tmp_path / "remote.git"
    remote_dir.mkdir()
    _git(remote_dir, "init", "--bare", "-b", "main")

    repo = _make_repo_with_main(tmp_path)
    _git(repo, "remote", "add", "origin", str(remote_dir))
    _git(repo, "push", "-u", "origin", "main")

    # Local commit on main — not pushed.
    (repo / "local.txt").write_text("local change\n")
    _git(repo, "add", "local.txt")
    _commit(repo, "local only")

    out = git_ops.diff(repo, "main")
    assert "local.txt" in out, "diff should show unpushed changes vs origin/main"


# --- diff_name_only ---------------------------------------------------------


def test_diff_name_only_returns_changed_files(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "added.txt").write_text("hello\n")
    _git(repo, "add", "added.txt")
    _commit(repo, "add file")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert result == ["added.txt"]


def test_diff_name_only_returns_multiple_files_in_order(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "alpha.txt").write_text("a\n")
    (repo / "beta.txt").write_text("b\n")
    _git(repo, "add", "alpha.txt", "beta.txt")
    _commit(repo, "add two files")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert result == ["alpha.txt", "beta.txt"]


def test_diff_name_only_returns_empty_list_on_bad_ref(tmp_path: Path) -> None:
    """Soft-failure: unresolvable ref yields [] rather than raising."""
    repo = _make_repo_with_main(tmp_path)
    result = git_ops.diff_name_only(repo, "nonexistent-ref", "HEAD")
    assert result == []


def test_changed_files_against_compares_tracked_changes_to_snapshot(tmp_path: Path) -> None:
    """A pre-fix snapshot, rather than HEAD, is the guard's tracked baseline."""
    repo = _make_repo_with_main(tmp_path)
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
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _commit(repo, "topic-msg")
    out = git_ops.log(repo, "main")
    assert "topic-msg" in out


def test_show_returns_file_bytes_at_ref(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    out = git_ops.show(repo, "HEAD", "base.txt")
    assert out == b"base\n"


def test_show_raises_on_missing_path(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.show(repo, "HEAD", "nope.txt")


def test_grep_returns_matching_paths(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "needle.txt").write_text("findme\n")
    (repo / "miss.txt").write_text("nope\n")
    _git(repo, "add", "needle.txt", "miss.txt")
    _commit(repo, "add files")
    matches = git_ops.grep(repo, "findme")
    assert "needle.txt" in matches
    assert "miss.txt" not in matches


def test_grep_returns_empty_when_no_matches(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.grep(repo, "doesnotexistanywhere") == []


def test_grep_word_matches_whole_words_only(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "whole.txt").write_text("the app runs\n")
    (repo / "part.txt").write_text("the application runs\n")
    _git(repo, "add", "whole.txt", "part.txt")
    _commit(repo, "add files")
    matches = git_ops.grep(repo, "app", word=True)
    assert "whole.txt" in matches
    assert "part.txt" not in matches


def test_grep_pathspecs_restrict_search(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "code.py").write_text("widget = 1\n")
    (repo / "notes.md").write_text("widget docs\n")
    _git(repo, "add", "code.py", "notes.md")
    _commit(repo, "add files")
    matches = git_ops.grep(repo, "widget", pathspecs=("*.py",))
    assert "code.py" in matches
    assert "notes.md" not in matches


def test_grep_fixed_matches_returns_path_pattern_pairs(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "widget_user.py").write_text("import widget\n")
    (repo / "gadget_user.ts").write_text("gadget\n")
    (repo / "partial.py").write_text("widget_factory\n")
    (repo / "notes.md").write_text("widget gadget\n")
    _git(repo, "add", "widget_user.py", "gadget_user.ts", "partial.py", "notes.md")
    _commit(repo, "add files")
    matches = git_ops.grep_fixed_matches(repo, ("widget", "gadget"), word=True, pathspecs=("*.py", "*.ts"))
    assert len(matches) == 2
    assert set(matches) == {("widget_user.py", "widget"), ("gadget_user.ts", "gadget")}


def test_grep_fixed_matches_raises_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep_fixed_matches raises GitError when git exits with a non-zero,
    non-1 exit code."""
    repo = _make_repo_with_main(tmp_path)
    # Craft a subprocess.run return with exit code 2 (generic git error).
    # _run_git passes the CompletedProcess through directly.
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(
            args=["git"], returncode=2,
            stdout=b"", stderr=b"fatal: unknown option",
        )
    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    with pytest.raises(GitError, match="git grep -F -o -z -f failed"):
        git_ops.grep_fixed_matches(repo, ("widget",))


def test_grep_fixed_matches_raises_on_malformed_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep_fixed_matches raises GitError when a NUL-separated record is
    malformed (no NUL separator, empty path, or empty pattern)."""
    repo = _make_repo_with_main(tmp_path)
    # A line without a NUL separator produces parts with len != 2.
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(
            args=["git"], returncode=0,
            stdout=b"file.py\n", stderr=b"",
        )
    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    with pytest.raises(GitError, match="malformed record"):
        git_ops.grep_fixed_matches(repo, ("widget",))


def test_grep_fixed_matches_empty_when_no_matches(tmp_path: Path) -> None:
    """Exit code 1 ("no matches") is treated as success: an empty list."""
    repo = _make_repo_with_main(tmp_path)
    (repo / "widget.py").write_text("widget\n")
    _git(repo, "add", "widget.py")
    _commit(repo, "add widget")
    assert git_ops.grep_fixed_matches(repo, ("absent_pattern",)) == []


def test_grep_fixed_matches_dedups_and_skips_nul_cr_lf_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate patterns are written once; empty/NUL/CR/LF patterns never
    reach the patterns file handed to git."""
    repo = _make_repo_with_main(tmp_path)
    (repo / "widget.py").write_text("widget\n")
    _git(repo, "add", "widget.py")
    _commit(repo, "add widget")

    patterns_files: list[bytes] = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        argv = args[0] if args else []
        for index, arg in enumerate(argv):
            if arg == "-f" and index + 1 < len(argv):
                patterns_files.append(Path(argv[index + 1]).read_bytes())
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    matches = git_ops.grep_fixed_matches(
        repo,
        ("widget", "widget", "", "gadget", "bad\x00pattern", "bad\rpattern", "bad\npattern"),
        word=True,
    )
    assert matches == []
    assert patterns_files == [b"widget\ngadget"]


def test_grep_fixed_matches_empty_when_all_patterns_unsuitable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-unsuitable pattern set short-circuits with no git invocation."""
    repo = _make_repo_with_main(tmp_path)
    (repo / "widget.py").write_text("widget\n")
    _git(repo, "add", "widget.py")
    _commit(repo, "add widget")

    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("git must not run when every pattern is unsuitable")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", no_git)
    assert git_ops.grep_fixed_matches(repo, ("", "a\x00b", "a\rb", "a\nb")) == []


def test_grep_fixed_matches_default_word_false_matches_substrings(tmp_path: Path) -> None:
    """The default word=False mode matches fixed substrings, not whole words."""
    repo = _make_repo_with_main(tmp_path)
    (repo / "app.py").write_text("application\n")
    _git(repo, "add", "app.py")
    _commit(repo, "add app")
    assert git_ops.grep_fixed_matches(repo, ("app",)) == [("app.py", "app")]


def test_status_porcelain_clean_and_dirty(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.status_porcelain(repo) == ""
    (repo / "untracked.txt").write_text("u\n")
    out = git_ops.status_porcelain(repo)
    assert "untracked.txt" in out


def test_upstream_ahead_count_no_upstream(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.upstream_ahead_count(repo, "main") == 0


def test_upstream_ahead_count_when_remote_ahead(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")

    # Push two extra commits to origin/main via a sidecar clone, then fetch.
    sidecar = tmp_path / "sidecar"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    (sidecar / "x.txt").write_text("x\n")
    _git(sidecar, "add", "x.txt")
    _commit(sidecar, "x")
    (sidecar / "y.txt").write_text("y\n")
    _git(sidecar, "add", "y.txt")
    _commit(sidecar, "y")
    _git(sidecar, "push", "origin", "main")

    _git(repo, "fetch", "origin")
    assert git_ops.upstream_ahead_count(repo, "main") == 2


# --- Mutating ---------------------------------------------------------------


def test_fetch_pulls_new_commits(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")

    sidecar = tmp_path / "sidecar"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    (sidecar / "z.txt").write_text("z\n")
    _git(sidecar, "add", "z.txt")
    new_sha = _commit(sidecar, "z")
    _git(sidecar, "push", "origin", "main")

    git_ops.fetch(repo)
    assert _git(repo, "rev-parse", "origin/main") == new_sha


def test_checkout_paths_restores_working_tree(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "base.txt").write_text("MUTATED\n")
    assert (repo / "base.txt").read_text() == "MUTATED\n"
    git_ops.checkout_paths(repo, [Path(".")])
    assert (repo / "base.txt").read_text() == "base\n"


def test_clean_untracked_removes_files(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "junk.txt").write_text("trash\n")
    (repo / "junkdir").mkdir()
    (repo / "junkdir" / "x.txt").write_text("x\n")
    git_ops.clean_untracked(repo)
    assert not (repo / "junk.txt").exists()
    assert not (repo / "junkdir").exists()


def test_worktree_add_and_remove_round_trip(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    wt = tmp_path / "wt"

    git_ops.worktree_add(repo, wt, head)
    assert wt.exists()
    assert (wt / "base.txt").read_text() == "base\n"
    # The worktree itself should pass the assertion.
    git_ops.assert_is_worktree(wt)

    git_ops.worktree_remove(repo, wt)
    assert not wt.exists()


# --- branch / commit / push primitives (Task 3) -----------------------------


def test_commit_paths_on_new_branch_pushes_to_origin(repo_with_origin: Path) -> None:
    (repo_with_origin / ".github/workflows").mkdir(parents=True)
    (repo_with_origin / ".github/workflows/daydream-review.yml").write_text("name: x\n")
    git_ops.create_branch(repo_with_origin, "daydream/setup")
    git_ops.commit_paths(
        repo_with_origin,
        [Path(".github/workflows/daydream-review.yml")],
        "add bot workflows",
    )
    git_ops.push_branch(repo_with_origin, "daydream/setup")
    assert git_ops.ref_exists(repo_with_origin, "origin/daydream/setup")


def test_create_branch_raises_when_branch_exists(repo_with_origin: Path) -> None:
    git_ops.create_branch(repo_with_origin, "daydream/dup")
    _git(repo_with_origin, "checkout", "main")
    with pytest.raises(GitError):
        git_ops.create_branch(repo_with_origin, "daydream/dup")


def test_commit_paths_commits_only_named_paths(repo_with_origin: Path) -> None:
    """commit_paths stages only the named files — never a blanket ``-A``."""
    git_ops.create_branch(repo_with_origin, "daydream/selective")
    (repo_with_origin / "tracked.txt").write_text("staged\n")
    (repo_with_origin / "untouched.txt").write_text("left behind\n")
    git_ops.commit_paths(repo_with_origin, [Path("tracked.txt")], "add tracked only")
    # The committed tree contains tracked.txt but not untouched.txt.
    committed = _git(repo_with_origin, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["tracked.txt"]
    # untouched.txt is still an uncommitted, untracked file.
    assert "untouched.txt" in _git(repo_with_origin, "status", "--porcelain")


def test_stage_paths_stages_only_named_paths(repo_with_origin: Path) -> None:
    """stage_paths stages exactly the named files into the index — never -A."""
    git_ops.create_branch(repo_with_origin, "daydream/stage")
    (repo_with_origin / "a.txt").write_text("a\n")
    (repo_with_origin / "b.txt").write_text("b\n")
    git_ops.stage_paths(repo_with_origin, [Path("a.txt")])
    staged = _git(repo_with_origin, "diff", "--cached", "--name-only").split()
    assert staged == ["a.txt"]
    # b.txt remains unstaged + untracked.
    assert "b.txt" in _git(repo_with_origin, "status", "--porcelain")


def test_push_branch_failure_raises_git_error(git_repo: Path) -> None:
    """push_branch propagates a push failure (no ``origin`` remote) as GitError."""
    git_ops.create_branch(git_repo, "daydream/no-remote")
    (git_repo / "f.txt").write_text("x\n")
    git_ops.commit_paths(git_repo, [Path("f.txt")], "add f")
    with pytest.raises(GitError):
        git_ops.push_branch(git_repo, "daydream/no-remote")


# --- Error type identity ----------------------------------------------------


def test_error_hierarchy_is_consistent() -> None:
    assert issubclass(NotAWorktreeError, GitError)
    assert issubclass(BranchNotFoundError, GitError)
    assert issubclass(WrongBranchError, GitError)
    assert issubclass(git_ops.GitTimeoutError, GitError)


# --- _run_git timeout retry (issue #120) ------------------------------------


def test_run_git_timeout_retry_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`_run_git` retries transient timeouts but not other failures (#120).

    One test, three cases:
      * a timeout on the first attempt is retried and the later success returns;
      * timeouts that exhaust every attempt raise the distinct GitTimeoutError;
      * a non-timeout subprocess error raises a plain GitError immediately, with
        no wasted retries.
    """
    repo = _make_repo_with_main(tmp_path)
    ok = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="true\n", stderr="")

    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}

    def flaky_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["git"], timeout=5)
        return ok

    monkeypatch.setattr("daydream.git_ops.subprocess.run", flaky_run)
    assert git_ops._run_git(repo, ["rev-parse", "HEAD"]).returncode == 0
    assert calls["n"] == 2  # timed out once, then succeeded

    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=5)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops._run_git(repo, ["rev-parse", "HEAD"], retries=2)
    assert calls["n"] == 3  # 1 initial + 2 retries

    # Case 3: non-timeout failure raises plain GitError immediately (no retry).
    calls["n"] = 0

    def os_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise OSError("git binary missing")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", os_error)
    with pytest.raises(GitError) as exc:
        git_ops._run_git(repo, ["rev-parse", "HEAD"], retries=2)
    assert not isinstance(exc.value, git_ops.GitTimeoutError)
    assert calls["n"] == 1  # no retries for non-timeout failures


def test_run_git_encodes_input_text_when_capturing_bytes(tmp_path: Path) -> None:
    """`_run_git(capture_bytes=True, input_text=...)` encodes the payload to
    bytes: subprocess.run raises a raw TypeError for str input with text=False,
    which in a retry loop would mask a genuine git failure."""
    repo = _make_repo_with_main(tmp_path)
    bytes_proc = git_ops._run_git(
        repo, ["hash-object", "--stdin"], capture_bytes=True, input_text="abc\n"
    )
    text_proc = git_ops._run_git(repo, ["hash-object", "--stdin"], input_text="abc\n")
    assert bytes_proc.returncode == 0
    assert bytes_proc.stdout == text_proc.stdout.encode()


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda repo: git_ops.fetch(repo), id="git-fetch"),
        pytest.param(
            lambda repo: git_ops.gh_pr_create(
                repo,
                head="feature",
                base="main",
                title="t",
                body="b",
            ),
            id="gh-pr-create",
        ),
        pytest.param(
            lambda repo: git_ops.gh_issue_create(
                repo,
                title="t",
                body="b",
                repo_slug="octocat/hello",
            ),
            id="gh-issue-create",
        ),
    ],
)
def test_mutating_wrapper_does_not_retry_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: Any,
) -> None:
    """Mutating git/gh wrappers never retry after an ambiguous timeout (#120).

    Re-running a non-idempotent command after a timeout could land on top of
    partial repo changes or open a duplicate PR.
    """
    repo = _make_repo_with_main(tmp_path)
    calls = {"n": 0}

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["command"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        operation(repo)
    assert calls["n"] == 1  # no retries for mutating operations


# --- _run_gh timeout retry (fake-gh flake under load) -----------------------


@pytest.mark.parametrize(
    ("env_value", "expected_timeout", "expected_warning"),
    [
        ("6O", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='6O' is not a valid integer; using default 60"),
        ("-5", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='-5' must be positive; using default 60"),
        ("0", 60, "DAYDREAM_GH_TIMEOUT_SECONDS='0' must be positive; using default 60"),
        ("7", 7, None),
    ],
    ids=["malformed", "negative", "zero", "valid"],
)
def test_run_gh_timeout_environment_validation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    env_value: str,
    expected_timeout: int,
    expected_warning: str | None,
) -> None:
    """`_run_gh` resolves the env timeout at call time, warning+falling back on bad values."""
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_SECONDS", env_value)

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=expected_timeout)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError) as exc:
        git_ops._run_gh(repo, ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
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
        (None, 3, None),
        ("", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='' is not a valid integer; using default 2"),
        ("abc", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='abc' is not a valid integer; using default 2"),
        ("-1", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='-1' is negative; using default 2"),
        ("0", 1, None),
        ("1", 2, None),
    ],
    ids=["default", "empty-warns", "malformed-warns", "negative-warns", "zero-valid", "one-valid"],
)
def test_read_only_gh_retry_environment_validation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    env_value: str | None,
    expected_attempts: int,
    expected_warning: str | None,
) -> None:
    """Read-only ``gh`` retry budget is validated at call time; retry 0 stays a valid 1 attempt."""
    repo = _make_repo_with_main(tmp_path)
    if env_value is not None:
        monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", env_value)
    else:
        monkeypatch.delenv("DAYDREAM_GH_TIMEOUT_RETRIES", raising=False)
    calls = {"n": 0}

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read-only ``gh`` wrappers ride out transient host-load timeouts.

    A `gh` subprocess that times out under CPU starvation is not a hung command,
    so read-only wrappers retry. Two cases through the production path:
      * a timeout on the first attempt is retried and the later success returns;
      * timeouts that exhaust every attempt raise GitTimeoutError after exactly
        ``_gh_retries() + 1`` tries.
    """
    repo = _make_repo_with_main(tmp_path)
    ok = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="octocat/hello\n", stderr="")

    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}

    def flaky_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)
        return ok

    monkeypatch.setattr("daydream.git_ops.subprocess.run", flaky_run)
    assert git_ops.gh_repo_view(repo) == ("octocat", "hello")
    assert calls["n"] == 2  # timed out once, then succeeded

    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_pr_diff(repo, 7)
    assert calls["n"] == git_ops._gh_retries() + 1


def test_gh_api_retries_only_when_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``gh_api`` retries reads but never mutations — incl. GraphQL.

    HTTP method cannot tell a GraphQL query from a mutation (both POST), so the
    retry decision is the caller's ``idempotent`` flag. A read (``idempotent=True``)
    rides out timeouts; a mutation (the ``input_data`` POST path, default flag)
    raises after a single attempt so a comment is never double-posted.
    """
    repo = _make_repo_with_main(tmp_path)
    calls = {"n": 0}

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)

    # Read: idempotent=True -> retried to exhaustion.
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(repo, "/user", idempotent=True)
    assert calls["n"] == git_ops._gh_retries() + 1

    # Mutation: a GraphQL-shaped POST with a body, default flag -> no retry.
    calls["n"] = 0
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(repo, "graphql", method="POST", input_data={"query": "mutation { x }"})
    assert calls["n"] == 1


# --- gh issue create ---------------------------------------------------------


def test_gh_issue_create_constructs_argv_with_body_file_and_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`gh_issue_create` shells out via `_run_gh` with title inline, body in a
    tempfile (`--body-file`, never on argv), optional `--label` flags and a
    `--repo owner/name` target. Returns the parsed issue URL.

    Issue-filing is the routing target for out-of-scope findings (issue #336);
    bodies can be large, so they never appear in argv (process-list hygiene,
    same rule as `gh secret set`'s stdin path).
    """
    repo = _make_repo_with_main(tmp_path)
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

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    url = git_ops.gh_issue_create(
        repo,
        title="out-of-scope: refactor handler.py error path",
        body="evidence and rationale\nthat the fix loop overreached\n",
        repo_slug="octocat/hello",
        labels=["daydream", "tech-debt"],
    )

    # URL is parsed from stdout.
    assert url == "https://github.com/octocat/hello/issues/42"

    argv = captured["argv"]
    # Subcommand shape and --repo target.
    assert argv[:5] == ["gh", "issue", "create", "--repo", "octocat/hello"]
    # Title inline.
    i_title = argv.index("--title")
    assert argv[i_title + 1] == "out-of-scope: refactor handler.py error path"
    # Body via --body-file (never inline): the tempfile held the body at call time.
    i_body = argv.index("--body-file")
    body_path = argv[i_body + 1]
    assert captured["body"] == "evidence and rationale\nthat the fix loop overreached\n"
    # The tempfile is unlinked post-call (no leak).
    assert not Path(body_path).exists()
    # The body text itself must not appear anywhere in argv.
    assert "that the fix loop overreached" not in argv
    # Both labels, in order, after --label.
    label_positions = [i for i, tok in enumerate(argv) if tok == "--label"]
    assert [argv[i + 1] for i in label_positions] == ["daydream", "tech-debt"]
    # Ran in the target repo's cwd.
    assert captured["cwd"] == repo


def test_gh_issue_create_omits_label_flags_when_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No `labels` argument → no `--label` flag on argv at all."""
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        captured["argv"] = list(args)
        return subprocess.CompletedProcess(
            args=list(args), returncode=0,
            stdout="https://github.com/octocat/hello/issues/7\n", stderr="",
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    url = git_ops.gh_issue_create(repo, title="t", body="b", repo_slug="octocat/hello")
    assert url == "https://github.com/octocat/hello/issues/7"
    assert "--label" not in captured["argv"]


def test_gh_issue_create_raises_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-zero `gh` exit raises GitError, matching sibling gh_* wrappers."""
    repo = _make_repo_with_main(tmp_path)

    body_path: list[str] = []

    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Capture the body tempfile path while it still exists (the unlink-in-
        # finally runs after _run_gh returns).
        i_body = list(args).index("--body-file")
        body_path.append(args[i_body + 1])
        return subprocess.CompletedProcess(
            args=list(args), returncode=1, stdout="", stderr="gh: not authenticated\n",
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    with pytest.raises(GitError):
        git_ops.gh_issue_create(repo, title="t", body="b", repo_slug="octocat/hello")
    # The body tempfile is unlinked on the failure path too (unlink-in-finally).
    assert body_path and not Path(body_path[0]).exists()


def test_gh_issue_list_returns_parsed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`gh_issue_list` parses ``number/title/body/url`` rows; best-effort dedup.

    Issue #336 — the fix loop dedups out-of-scope findings against already-filed
    open issues (GitHub is the store), so the wrapper returns the parsed rows
    it needs to scan bodies for the finding's fingerprint marker.
    """
    repo = _make_repo_with_main(tmp_path)
    rows = [
        {
            "number": 42,
            "title": "[daydream] out-of-scope finding: notes.txt",
            "body": "desc\n<!-- daydream-scope-finding: abc123 -->",
            "url": "https://github.com/octocat/hello/issues/42",
        },
        {"number": 7, "title": "unrelated", "body": "", "url": ""},
    ]
    captured: dict[str, Any] = {}

    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        captured["argv"] = list(args)
        return subprocess.CompletedProcess(
            args=list(args), returncode=0, stdout=json.dumps(rows), stderr="",
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    result = git_ops.gh_issue_list(repo, search="out-of-scope", repo_slug="octocat/hello")
    assert result == rows

    argv = captured["argv"]
    assert argv[:3] == ["gh", "issue", "list"]
    assert "--state" in argv and argv[argv.index("--state") + 1] == "open"
    assert "--json" in argv and "number,title,body,url" in argv
    assert "--search" in argv and argv[argv.index("--search") + 1] == "out-of-scope"
    assert "--repo" in argv and "octocat/hello" in argv


def test_gh_issue_list_returns_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Best-effort: a non-zero ``gh`` exit returns [] (never raises).

    Dedup must fail open: a failed lookup degrades to filing (the prior
    behavior) rather than blocking the scope decision or dropping the finding.
    """
    repo = _make_repo_with_main(tmp_path)

    def fake_run(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(
            args=list(args), returncode=1, stdout="", stderr="gh: not authenticated\n",
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    assert git_ops.gh_issue_list(repo) == []


# --- gh wrappers (skipped when gh missing) ----------------------------------


_gh_available = shutil.which("gh") is not None
gh_required = pytest.mark.skipif(not _gh_available, reason="gh CLI not installed")


@pytest.fixture
def local_only_gh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reach local remote discovery without depending on host authentication.

    These repositories have no remotes, so gh fails before making a request.
    The synthetic token only suppresses its earlier login/configuration gate.
    """
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "isolated-gh-config"))
    monkeypatch.setenv("GH_HOST", "github.com")
    monkeypatch.setenv("GH_TOKEN", "test-local-only-no-network")
    monkeypatch.delenv("GH_REPO", raising=False)


@gh_required
def test_gh_repo_view_returns_none_outside_github_repo(tmp_path: Path) -> None:
    """A local-only repo with no GitHub remote yields ``None``."""
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.gh_repo_view(repo) is None


@gh_required
@pytest.mark.usefixtures("local_only_gh")
def test_gh_pr_view_raises_without_remote(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError, match="no git remotes found"):
        git_ops.gh_pr_view(repo, 999999)


@gh_required
@pytest.mark.usefixtures("local_only_gh")
def test_gh_pr_list_for_branch_raises_without_remote(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError, match="no git remotes found"):
        git_ops.gh_pr_list_for_branch(repo, "main")


@gh_required
def test_gh_pr_diff_raises_without_remote(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.gh_pr_diff(repo, 1)


@gh_required
def test_gh_api_raises_without_auth(tmp_path: Path) -> None:
    """``gh api`` against a relative endpoint with no GitHub remote fails."""
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.gh_api(repo, "repos/{owner}/{repo}")


# --- diff_paths -------------------------------------------------------------


def _make_divergent_history(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a repo where main has advanced after `feat` branched off.

    Returns the repo path plus the names of the two branches (`main`, `feat`).
    The same file (`shared.txt`) is modified on both branches so two-dot vs
    three-dot diffs differ in content.
    """
    repo = _make_repo_with_main(tmp_path)
    (repo / "shared.txt").write_text("line one\nline two\nline three\n")
    _git(repo, "add", "shared.txt")
    _commit(repo, "shared baseline")

    _git(repo, "checkout", "-b", "feat")
    (repo / "shared.txt").write_text("line one\nline two FEAT\nline three\n")
    _git(repo, "add", "shared.txt")
    _commit(repo, "feat edit")

    _git(repo, "checkout", "main")
    (repo / "shared.txt").write_text("line one\nline two MAIN\nline three\n")
    _git(repo, "add", "shared.txt")
    _commit(repo, "main edit after branch")

    _git(repo, "checkout", "feat")
    return repo, "main", "feat"


def test_diff_paths_direct_vs_merge_base_differ_on_divergent_history(
    tmp_path: Path,
) -> None:
    """Direct diff includes main's later commit; merge-base diff does not. Pin the diff."""
    repo, base, head = _make_divergent_history(tmp_path)
    direct = git_ops.diff_paths(repo, base, head, ["shared.txt"], merge_base_diff=False)
    since_merge_base = git_ops.diff_paths(repo, base, head, ["shared.txt"], merge_base_diff=True)
    assert direct != since_merge_base
    # Direct diff shows main's "MAIN" line as the - side; merge-base diff doesn't.
    assert "MAIN" in direct
    assert "MAIN" not in since_merge_base


def test_diff_paths_restricts_to_paths(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    _commit(repo, "two files")
    out = git_ops.diff_paths(repo, "main", "feat", ["keep.txt"])
    assert "keep.txt" in out
    assert "drop.txt" not in out


def test_diff_paths_unified_context_lines(tmp_path: Path) -> None:
    """Larger --unified yields a longer diff for the same change."""
    repo = _make_repo_with_main(tmp_path)
    (repo / "ctx.txt").write_text("\n".join(f"line {i}" for i in range(1, 31)) + "\n")
    _git(repo, "add", "ctx.txt")
    _commit(repo, "ctx baseline")

    _git(repo, "checkout", "-b", "feat")
    lines = [f"line {i}" for i in range(1, 31)]
    lines[14] = "line 15 CHANGED"
    (repo / "ctx.txt").write_text("\n".join(lines) + "\n")
    _git(repo, "add", "ctx.txt")
    _commit(repo, "ctx edit")

    small = git_ops.diff_paths(repo, "main", "feat", ["ctx.txt"], unified=1)
    big = git_ops.diff_paths(repo, "main", "feat", ["ctx.txt"], unified=10)
    assert len(big) > len(small)


def test_diff_paths_raises_on_invalid_ref(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.diff_paths(repo, "definitely-not-a-ref", "HEAD", ["base.txt"])


# --- gh_api(input_data=...) and gh_pr_view(pr=None) -------------------------
# These tests exercise wrapper logic, not gh itself: subprocess is monkeypatched
# to capture argv and drive success/failure paths deterministically.


def test_gh_pr_queries_request_only_legacy_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        stdout = "{}" if cmd[1:3] == ["pr", "view"] else "[]"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pr: int | None,
    stderr: str,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr),
    )
    assert git_ops.gh_pr_view(repo, pr) is None


@pytest.mark.parametrize(
    "stderr",
    [
        "no git remotes found",
        "HTTP 401: authentication required",
        "Unknown JSON field: futureField",
        "GraphQL: Could not resolve to a PullRequest with the number of 41. (repository.pullRequest)",
        'prefix: no pull requests found for branch "feature"',
        "HTTP 404: resource not found",
    ],
)
def test_gh_pr_view_unknown_failures_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr),
    )
    with pytest.raises(GitError, match=re.escape(stderr)):
        git_ops.gh_pr_view(repo, 42)


@pytest.mark.parametrize("stdout", ["not-json", "[]", "null", "42"])
def test_gh_pr_view_rejects_invalid_json_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=""),
    )
    with pytest.raises(GitError, match="invalid JSON|JSON object"):
        git_ops.gh_pr_view(repo, 42)


@pytest.mark.parametrize("stdout", ["not-json", "{}", "[42]", '[{"number": 1}, null]'])
def test_gh_pr_list_rejects_failure_and_invalid_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=""),
    )
    with pytest.raises(GitError, match="invalid JSON|JSON list|row"):
        git_ops.gh_pr_list_for_branch(repo, "feature")


def test_gh_pr_list_nonzero_uses_gh_error_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="rate limit exceeded; retry-after: 7"
        ),
    )
    with pytest.raises(git_ops.RateLimitError) as excinfo:
        git_ops.gh_pr_list_for_branch(repo, "feature")
    assert excinfo.value.retry_after == 7


@pytest.mark.parametrize("slug", ["owner", "owner/repo/extra", "owner/ ", " owner/repo", "/repo"])
def test_gh_repo_view_required_rejects_invalid_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slug: str
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=slug + "\n", stderr=""),
    )
    with pytest.raises(GitError, match="invalid repository slug"):
        git_ops.gh_repo_view_required(repo)


def test_gh_repo_view_required_returns_exact_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="Owner/Repo\n", stderr=""),
    )
    assert git_ops.gh_repo_view_required(repo) == ("Owner", "Repo")


def test_diagnostic_url_redaction_is_bounded_on_long_untrusted_text() -> None:
    """Non-URL diagnostics must not trigger quadratic scheme-prefix searches."""
    checked = subprocess.run(
        [
            sys.executable,
            "-c",
            "from daydream.git_ops import _redact_sensitive_text\n"
            "for text in ('A' * 100_000, 'A.' * 50_000):\n"
            "    assert _redact_sensitive_text(text) == text\n"
            "for scheme in ('https', 'ssh', 'git+custom.transport'):\n"
            "    raw = f'failed {scheme}://user:password@example.invalid/o/r'\n"
            "    assert _redact_sensitive_text(raw) == "
            "f'failed {scheme}://***@example.invalid/o/r'\n",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr


def test_gh_repo_view_required_preserves_safe_failure_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    monkeypatch.setattr(
        "daydream.git_ops.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="HTTP 401: authentication required for token ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        ),
    )

    with pytest.raises(GitError) as excinfo:
        git_ops.gh_repo_view_required(repo)
    assert "authentication required" in str(excinfo.value)
    assert "ghp_abcdefghijklmnopqrstuvwxyz1234567890" not in str(excinfo.value)


@gh_required
@pytest.mark.parametrize("fields", ["view", "list"])
def test_installed_gh_accepts_production_pr_fields_without_network(
    tmp_path: Path, fields: str
) -> None:
    repo = _make_repo_with_main(tmp_path)
    selected = git_ops.GH_PR_VIEW_FIELDS if fields == "view" else git_ops.GH_PR_LIST_FIELDS
    proc = subprocess.run(
        ["gh", "pr", fields, "--json", ",".join(selected)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Unknown JSON field" not in proc.stderr


def test_gh_api_input_data_passes_tempfile_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serialize API input through a temporary file and remove it after success."""
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        idx = cmd.index("--input")
        captured["input_path"] = cmd[idx + 1]
        # Confirm the tempfile exists at call time and holds our payload.
        captured["payload"] = Path(cmd[idx + 1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    result = git_ops.gh_api(
        repo,
        "repos/owner/repo/pulls/1/reviews",
        method="POST",
        input_data={"event": "COMMENT", "body": "hi"},
    )

    assert result == {"ok": True}
    cmd = captured["cmd"]
    assert cmd[:2] == ["gh", "api"]
    assert "--input" in cmd
    assert "--method" in cmd
    method_idx = cmd.index("--method")
    assert cmd[method_idx + 1] == "POST"
    assert json.loads(captured["payload"]) == {"event": "COMMENT", "body": "hi"}
    # Success path: tempfile must have been deleted.
    assert not Path(captured["input_path"]).exists()


def test_gh_api_input_data_preserves_tempfile_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve failed API payloads and disclose their recovery path in the error."""
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        idx = cmd.index("--input")
        captured["input_path"] = cmd[idx + 1]
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="HTTP 422: Validation failed")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    with pytest.raises(GitError) as excinfo:
        git_ops.gh_api(
            repo,
            "repos/owner/repo/pulls/1/reviews",
            method="POST",
            input_data={"bad": "payload"},
        )

    msg = str(excinfo.value)
    assert "payload preserved at" in msg
    # The tempfile path mentioned in the error must still exist on disk.
    preserved = Path(captured["input_path"])
    assert str(preserved) in msg
    assert preserved.exists()
    # Cleanup so the test doesn't leave debris behind.
    preserved.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("pr", "number"),
    [(None, 7), (42, 42)],
    ids=["omits_pr_arg_when_none", "includes_pr_arg_when_given"],
)
def test_gh_pr_view_pr_arg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pr: int | None, number: int) -> None:
    """Include an explicit PR number only when the caller supplies one."""
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=f'{{"number": {number}}}', stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    result = git_ops.gh_pr_view(repo) if pr is None else git_ops.gh_pr_view(repo, pr)
    assert result == {"number": number}
    cmd = captured["cmd"]
    if pr is None:
        assert cmd[:3] == ["gh", "pr", "view"]
        # No PR number anywhere in the argv.
        assert all(not part.isdigit() for part in cmd)
    else:
        assert cmd[:4] == ["gh", "pr", "view", str(pr)]


# --- daydream_commits ---------------------------------------------------------


def test_daydream_commits_returns_tagged_commits(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat/x")
    (repo / "a.py").write_text("a\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-m", "fix: something\n\nDaydream-Run: test-123\nDaydream-Version: 0.14.0")
    result = git_ops.daydream_commits(repo, "main")
    assert result is not None
    assert "fix: something" in result


def test_daydream_commits_excludes_untagged(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat/x")
    (repo / "b.py").write_text("b\n")
    _git(repo, "add", "b.py")
    _commit(repo, "chore: unrelated change")
    result = git_ops.daydream_commits(repo, "main")
    assert result is None


def test_daydream_commits_none_when_no_commits(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat/x")
    result = git_ops.daydream_commits(repo, "main")
    assert result is None


# --- clone -------------------------------------------------------------------


def _make_bare_remote(tmp_path: Path) -> Path:
    """Create a bare remote repo with one committed file."""
    repo = _make_repo_with_main(tmp_path / "src")
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, capture_output=True)  # noqa: S603, S607 - arguments are not user-controlled
    return bare


def test_clone_creates_working_tree(tmp_path: Path) -> None:
    """clone() creates a functional git working tree from a bare remote."""
    bare = _make_bare_remote(tmp_path)
    target = tmp_path / "cloned"
    git_ops.clone(str(bare), target)
    assert (target / ".git").is_dir()
    assert (target / "base.txt").read_text() == "base\n"


def test_clone_of_linked_worktree_materializes_head_and_staged_patch(
    tmp_path: Path,
    linked_worktree: tuple[Path, Path],
) -> None:
    """A linked-worktree clone materializes that worktree's HEAD and a staged
    binary patch round-trips into it (issue #221 / false-assumption STOP)."""
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
    """remove_remote drops the clone's origin without touching its HEAD."""
    source = _make_repo_with_main(tmp_path / "src")
    clone = tmp_path / "clone"
    git_ops.clone(str(source), clone)
    assert git_ops.remote_url(clone) == str(source)
    before = git_ops.head_sha(clone)

    git_ops.remove_remote(clone)
    assert git_ops.remote_url(clone) is None
    assert git_ops.head_sha(clone) == before


def test_update_ref_sets_explicit_oid(tmp_path: Path) -> None:
    """update_ref repoints an existing ref to a *different* OID: the ref is
    seeded to HEAD and repointed at a newer commit, so a no-op write path
    cannot satisfy the assertion."""
    repo = _make_repo_with_main(tmp_path, name="update_ref")
    start = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/heads/scratch", start)  # seed a ref to repoint
    (repo / "second.txt").write_text("second\n")
    _git(repo, "add", "second.txt")
    _commit(repo, "second")  # HEAD moves, so update_ref must write a new OID
    target = _git(repo, "rev-parse", "HEAD")
    assert target != start

    git_ops.update_ref(repo, "refs/heads/scratch", target)

    assert _git(repo, "rev-parse", "refs/heads/scratch") == target


def test_update_ref_accepts_names_merely_ending_in_lock(tmp_path: Path) -> None:
    """git forbids only the literal lowercase ``.lock`` component suffix, so
    branches like unlock/block/deadlock/xLock as well as case-variants git
    accepts (topic.LOCK, release.LOCK, x.lOck) are valid refs and snapshot
    cleanly."""
    repo = _make_repo_with_main(tmp_path, name="update_ref_lockish")
    oid = _git(repo, "rev-parse", "HEAD")
    for name in ("unlock", "block", "deadlock", "xLock", "topic.LOCK", "release.LOCK", "x.lOck"):
        ref = f"refs/heads/{name}"
        git_ops.update_ref(repo, ref, oid)
        assert _git(repo, "rev-parse", ref) == oid


def test_update_refs_snapshots_a_batch_and_aborts_atomically(tmp_path: Path) -> None:
    """update_refs writes many refs in one transactional git call, and a bad
    ref aborts the whole batch: nothing is written."""
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


def test_update_ref_rejects_invalid_ref_name(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path, name="update_ref_bad")
    oid = git_ops.head_sha(repo)
    for bad in (
        "refs/heads/..",
        "refs/heads/foo bar",
        "refs/heads/bad\nname",
        "-dash-start",
        "refs/heads/topic.lock",
    ):
        with pytest.raises(git_ops.GitError, match="invalid ref name"):
            git_ops.update_ref(repo, bad, oid)


def test_update_ref_rejects_malformed_oid(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path, name="update_ref_badoid")
    with pytest.raises(git_ops.GitError, match="invalid OID"):
        git_ops.update_ref(repo, "refs/heads/newbranch", "not-a-sha")


def test_update_refs_rejects_injectable_ref_names_before_shell_out(tmp_path: Path) -> None:
    """update_refs rejects refs carrying whitespace/control characters up
    front, naming the ref, so a newline can never smuggle extra ``update``
    lines into the stdin payload; a malformed OID is rejected the same way."""
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
    """A staged index patch from source reproduces source's staged index in a clone.

    Binary payload exercises the ``--binary``/base85 machinery, so the
    byte-captured round-trip claim in :func:`git_ops.staged_patch` is real.
    """
    source = _make_repo_with_main(tmp_path / "src")
    clone = tmp_path / "clone"
    git_ops.clone(str(source), clone)
    payload = bytes(range(256))  # every byte value, incl. NUL/newline — not text
    (source / "blob.bin").write_bytes(payload)
    _git(source, "add", "blob.bin")
    shutil.copy2(source / "blob.bin", clone / "blob.bin")

    patch = git_ops.staged_patch(source)
    assert patch  # nonempty bytes

    git_ops.apply_staged_patch(clone, patch)
    assert git_ops.staged_patch(clone) == git_ops.staged_patch(source)
    assert _git(clone, "diff", "--", "blob.bin") == ""
    assert (clone / "blob.bin").read_bytes() == payload


def test_strict_enumeration_raises_where_soft_fails(tmp_path: Path) -> None:
    """strict=True propagates a non-zero git exit; the soft default returns [].

    The disposable read-only-checkout prep relies on strict enumeration so a
    mid-prep git failure surfaces (its documented error-propagation contract)
    instead of silently producing a clone missing tracked/untracked files.
    """
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    # Soft default: no crash, empty result.
    assert git_ops.ls_files(not_a_repo) == []
    assert git_ops.list_untracked(not_a_repo) == []

    # Strict: the same failure raises GitError that callers wrap in CodexError.
    with pytest.raises(git_ops.GitError):
        git_ops.ls_files(not_a_repo, strict=True)
    with pytest.raises(git_ops.GitError):
        git_ops.list_untracked(not_a_repo, strict=True)

    # On a real repo, strict returns the same paths as the soft call.
    repo = _make_repo_with_main(tmp_path / "src")
    (repo / "tracked.txt").write_text("x")
    (repo / "untracked.txt").write_text("y")
    _git(repo, "add", "tracked.txt")
    assert git_ops.ls_files(repo, strict=True) == git_ops.ls_files(repo)
    assert git_ops.list_untracked(repo, strict=True) == git_ops.list_untracked(repo)
    assert "tracked.txt" in git_ops.ls_files(repo, strict=True)
    assert "untracked.txt" in git_ops.list_untracked(repo, strict=True)


def test_clone_raises_on_invalid_remote(tmp_path: Path) -> None:
    """clone() raises GitError when the remote URL is invalid."""
    target = tmp_path / "nope"
    with pytest.raises(git_ops.GitError, match="git clone .* failed"):
        git_ops.clone("file:///nonexistent/repo.git", target)


@pytest.mark.parametrize(
    ("blobless", "filter_present"),
    [(True, True), (False, False)],
    ids=["blobless_passes_filter_flag", "default_no_filter_flag"],
)
def test_clone_filter_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blobless: bool,
    filter_present: bool,
) -> None:
    """clone(blobless=True) includes --filter=blob:none; the default omits it."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    git_ops.clone("https://example.com/repo.git", tmp_path / "out", blobless=blobless)
    assert ("--filter=blob:none" in captured["cmd"]) is filter_present


@pytest.mark.parametrize(
    ("stderr", "expected_type"),
    [
        pytest.param(
            "gh: API rate limit exceeded for user (HTTP 403)",
            git_ops.RateLimitError,
            id="rate-limit",
        ),
        pytest.param(
            "gh: Not Found (HTTP 404)",
            git_ops.GitError,
            id="plain-git-error",
        ),
    ],
)
def test_gh_api_classifies_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stderr: str,
    expected_type: type[GitError],
) -> None:
    """Map authentication and rate-limit responses to specialized Git errors."""
    proc = subprocess.CompletedProcess(
        args=["gh"],
        returncode=1,
        stdout="",
        stderr=stderr,
    )
    monkeypatch.setattr(git_ops, "_run_gh", lambda *a, **k: proc)
    with pytest.raises(expected_type) as exc:
        git_ops.gh_api(tmp_path, "repos/o/r/pulls/1")
    assert type(exc.value) is expected_type


@pytest.mark.parametrize(
    ("stdout", "endpoint", "paginate", "jq", "expected", "expected_args"),
    [
        pytest.param(
            '{"id": 1}\n{"id": 2}\n',
            "/app/installations",
            True,
            ".[]",
            [{"id": 1}, {"id": 2}],
            ["api", "--paginate", "--jq", "(.[]) | @json", "/app/installations"],
            id="ndjson-list",
        ),
        pytest.param(
            '"anderskev"\n',
            "user",
            False,
            ".login",
            ["anderskev"],
            ["api", "--jq", "(.login) | @json", "user"],
            id="raw-scalar",
        ),
    ],
)
def test_gh_api_jq_parsing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stdout: str,
    endpoint: str,
    paginate: bool,
    jq: str,
    expected: list[Any],
    expected_args: list[str],
) -> None:
    """Pass pagination and jq arguments through while decoding varied output."""
    captured: dict[str, list[str]] = {}

    def fake_run_gh(
        repo: Path,
        args: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Capture gh arguments and return the parameterized response payload."""
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=["gh"],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(git_ops, "_run_gh", fake_run_gh)
    result = git_ops.gh_api(tmp_path, endpoint, paginate=paginate, jq=jq)
    assert result == expected
    assert captured["args"] == expected_args


def test_gh_api_headers_pass_dash_h_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Explicit headers reach gh as -H args, in both the plain and --input branches."""
    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run_gh(repo: Path, args: list[str], **kwargs: Any) -> _Proc:
        captured.append(args)
        return _Proc()

    monkeypatch.setattr(git_ops, "_run_gh", fake_run_gh)
    headers = {"Authorization": "Bearer jwt-abc"}
    git_ops.gh_api(tmp_path, "/app/installations", headers=headers)
    git_ops.gh_api(
        tmp_path,
        "/app/installations/1/access_tokens",
        method="POST",
        input_data={"repositories": ["r"]},
        headers=headers,
    )

    assert captured[0][:3] == ["api", "-H", "Authorization: Bearer jwt-abc"]
    assert captured[1][:3] == ["api", "-H", "Authorization: Bearer jwt-abc"]
    assert "--input" in captured[1]


def test_gh_api_error_message_redacts_authorization_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess failure must not leak the Bearer token into the GitError.

    Real-path: ``gh_api`` builds ``-H "Authorization: Bearer <jwt>"`` and calls the
    real ``_run_gh``; only ``subprocess.run`` is faked (the external boundary).
    """
    repo = _make_repo_with_main(tmp_path)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("gh executable failed to spawn")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    with pytest.raises(GitError) as excinfo:
        git_ops.gh_api(repo, "/app/installations", headers={"Authorization": "Bearer jwt-super-secret-xyz"})

    msg = str(excinfo.value)
    assert "jwt-super-secret-xyz" not in msg
    assert "Authorization: ***" in msg


def test_gh_api_timeout_redacts_authorization_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Timeout must redact the Bearer token in both the retry warning and the error."""
    repo = _make_repo_with_main(tmp_path)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    # Force one retry so the warning-log branch fires.
    monkeypatch.setattr(git_ops, "_gh_retries", lambda: 1)

    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        with pytest.raises(git_ops.GitTimeoutError) as excinfo:
            git_ops.gh_api(
                repo,
                "/app/installations",
                headers={"Authorization": "Bearer jwt-super-secret-xyz"},
                idempotent=True,
            )

    msg = str(excinfo.value)
    assert "jwt-super-secret-xyz" not in msg
    assert "Authorization: ***" in msg
    # The retry warning fired and must also be redacted.
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings, "expected a retry warning to be logged"
    assert all("jwt-super-secret-xyz" not in w for w in warnings)
    assert any("Authorization: ***" in w for w in warnings)


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_mode: str,
    expected_fragment: str,
) -> None:
    """Manifest-conversion codes must never appear in caller-visible gh failures.

    Real-path: ``gh_api`` builds the credential-bearing endpoint and calls the
    real ``_run_gh``; only ``subprocess.run`` (the external boundary) is faked.
    Each row exercises one failure producer and asserts the synthetic code is
    replaced by ``***`` while the route and diagnostic fragment survive.
    """
    repo = _make_repo_with_main(tmp_path)
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

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    # The timeout retry-warning renderer (git_ops.py:403) only fires when the
    # call is idempotent and _gh_retries()>0; the other rows run with the
    # default retries=0 to exercise the non-retry paths.
    kwargs: dict[str, Any] = {"method": "POST"}
    if failure_mode == "timeout-warning":
        monkeypatch.setattr(git_ops, "_gh_retries", lambda: 1)
        kwargs["idempotent"] = True

    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        with pytest.raises(GitError) as excinfo:
            git_ops.gh_api(repo, endpoint, **kwargs)

    # The rate-limit stderr must classify into a RateLimitError, not a plain GitError.
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
        assert any("/app-manifests/***/conversions" in w for w in warnings)


def test_gh_api_jq_invalid_line_raises_git_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Proc:
        returncode = 0
        stdout = '{"id": 1}\nnot-json\n'
        stderr = ""

    monkeypatch.setattr(git_ops, "_run_gh", lambda *a, **k: _Proc())
    with pytest.raises(git_ops.GitError, match="invalid JSON"):
        git_ops.gh_api(tmp_path, "/app/installations", jq=".[]")


# --- gh secret/variable/PR primitives (Task 2) ------------------------------

from tests.harness.fake_gh import FakeGh  # noqa: E402


@pytest.mark.parametrize("extra_field", ["baseRefOid", "futureCompatibilityFloor"])
@pytest.mark.parametrize("query_kind", ["view", "list"])
def test_fake_gh_rejects_every_field_outside_legacy_allowlist(
    fake_gh: FakeGh,
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_kind: str,
    extra_field: str,
) -> None:
    fake_gh.serve_pr_view({"number": 7})
    constant = "GH_PR_VIEW_FIELDS" if query_kind == "view" else "GH_PR_LIST_FIELDS"
    monkeypatch.setattr(git_ops, constant, (*getattr(git_ops, constant), extra_field))

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
    assert git_ops.gh_secret_list(git_repo, repo_slug="o/r") == [
        "DAYDREAM_APP_ID",
        "ANTHROPIC_API_KEY",
    ]


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


def test_log_shas_returns_none_when_ref_is_gone(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A deleted branch ref yields None ("could not look"), never [].

    Squash-merge deletes the branch, so the recorded ref no longer resolves and
    git exits 128. Returning [] here made callers read the unanswerable query as
    "no follow-up commits" and label the run rejected.
    """
    repo = _make_repo_with_main(tmp_path)

    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        result = git_ops.log_shas(repo, "deleted-branch", since="main")

    assert result is None
    assert any("log_shas" in record.message for record in caplog.records)


def test_log_shas_returns_empty_list_when_range_is_genuinely_empty(tmp_path: Path) -> None:
    """A resolvable ref with no commits ahead yields [] — distinct from None."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")

    assert git_ops.log_shas(repo, "topic", since="main") == []


def test_log_shas_returns_commits_ahead_of_since(tmp_path: Path) -> None:
    """The success path still returns SHAs, newest first."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _commit(repo, "topic-1")

    shas = git_ops.log_shas(repo, "topic", since="main")

    assert shas == [_git(repo, "rev-parse", "topic").strip()]


def test_log_shas_since_returns_commits_in_range(tmp_path: Path) -> None:
    """log_shas_since returns SHAs for commits in head..base range."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _commit(repo, "topic-1")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "b.txt")
    _commit(repo, "topic-2")
    shas = git_ops.log_shas_since(repo, "main", "topic")
    assert len(shas) == 2
    # newest first (git log order), exact match
    tip = _git(repo, "rev-parse", "topic").strip()
    parent = _git(repo, "rev-parse", "topic^").strip()
    assert shas == [tip, parent]  # newest first, exact, both entries
    assert all(len(s) == 40 for s in shas)


def test_log_shas_since_warns_on_git_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """log_shas_since returns [] and logs a warning on git error."""
    repo = _make_repo_with_main(tmp_path)

    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        result = git_ops.log_shas_since(repo, "main", "nonexistent-ref")
    assert result == []
    assert any("log_shas_since" in record.message for record in caplog.records)


def test_worktree_lock_and_lock_mtime_roundtrip(tmp_path: Path) -> None:
    """The lock primitives arm a real git worktree lock, expose its mtime,
    make a single-force remove refuse it, and release it on unlock."""
    repo = _make_repo_with_main(tmp_path)
    wt = repo / "wt1"
    git_ops.worktree_add(repo, wt, "main", detach=True)

    assert git_ops.worktree_lock_mtime(repo, wt) is None  # unlocked

    git_ops.worktree_lock(repo, wt, reason="run-A")
    locked_at = git_ops.worktree_lock_mtime(repo, wt)
    assert locked_at is not None

    # the liveness guard: a single --force remove is refused while locked
    with pytest.raises(GitError):
        git_ops.worktree_remove(repo, wt, force=True)

    git_ops.worktree_unlock(repo, wt)
    assert git_ops.worktree_lock_mtime(repo, wt) is None  # released


def test_worktree_remove_unlocked_unlocks_before_removing(tmp_path: Path) -> None:
    """worktree_remove_unlocked centralizes the unlock-before-remove ordering:
    it removes a locked worktree (unlocking it first) and an already-unlocked
    one (the unlock attempt is a no-op), so callers never inline the rule."""
    repo = _make_repo_with_main(tmp_path)

    # Locked worktree: removal must unlock first, then remove.
    locked_wt = repo / "wt-locked"
    git_ops.worktree_add(repo, locked_wt, "main", detach=True)
    git_ops.worktree_lock(repo, locked_wt, reason="run-A")
    assert git_ops.worktree_lock_mtime(repo, locked_wt) is not None
    git_ops.worktree_remove_unlocked(repo, locked_wt)
    assert not locked_wt.exists()

    # Unlocked worktree: the unlock attempt fails harmlessly, removal proceeds.
    unlocked_wt = repo / "wt-unlocked"
    git_ops.worktree_add(repo, unlocked_wt, "main", detach=True)
    assert git_ops.worktree_lock_mtime(repo, unlocked_wt) is None
    git_ops.worktree_remove_unlocked(repo, unlocked_wt)
    assert not unlocked_wt.exists()


def test_worktree_add_with_lock_reason_arms_lock_atomically(
    tmp_path: Path,
) -> None:
    """Fix #3: worktree_add(lock_reason=...) must create the worktree already
    locked in a single git invocation, so there is no unlocked window in which
    a concurrent prune could force-remove the fresh worktree."""
    repo = _make_repo_with_main(tmp_path)
    wt = repo / "wt-locked"

    git_ops.worktree_add(repo, wt, "main", detach=True, lock_reason="run-A")

    # locked marker present with the reason; no separate worktree_lock call needed
    assert git_ops.worktree_lock_mtime(repo, wt) is not None
    git_dir = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    locked = git_dir / "worktrees" / wt.name / "locked"
    assert locked.is_file()
    assert "run-A" in locked.read_text(encoding="utf-8")


def test_clone_error_message_never_contains_remote_url(tmp_path: Path) -> None:
    """GitError from clone() must not echo a credential-bearing remote URL (issue #981)."""
    with pytest.raises(GitError) as excinfo:
        git_ops.clone("https://user:ghp_canaryfake123@unreachable.invalid/o/r.git", tmp_path / "t", timeout=5)
    message = str(excinfo.value)
    assert "ghp_canaryfake123" not in message
    assert "user:" not in message
    assert "unreachable.invalid" in message  # host is safe to keep


def test_clone_error_message_redacts_stderr_url_echo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when git's stderr echoes the URL, the raised message is redacted."""
    real_run = subprocess.run

    def fake_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args,
            128,
            stderr=(
                "fatal: unable to access 'https://user:ghp_canaryfake123@unreachable.invalid/o/r.git/': "
                "Could not resolve host"
            ),
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    with pytest.raises(GitError) as excinfo:
        git_ops.clone("https://user:ghp_canaryfake123@unreachable.invalid/o/r.git", tmp_path / "t", timeout=5)
    assert "ghp_canaryfake123" not in str(excinfo.value)
    assert real_run is not None


# --- pre-push hook detection (issue #726 task 6) -----------------------------


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
    sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(clone), "remote", "add", "origin", str(remote)], check=True)
    assert git_ops.remote_contains_commit(clone, "main", sha) is False
    subprocess.run(["git", "-C", str(clone), "push", "-u", "origin", "main"], check=True, capture_output=True)
    assert git_ops.remote_contains_commit(clone, "main", sha) is True

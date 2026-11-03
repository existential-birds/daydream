"""Tests for :mod:`daydream.git_ops`. These tests use real ``git`` (and optionally ``gh``) against ``tmp_path``
fixtures. Transport fault injection covers timeout, privacy, and response-shape boundaries;
filesystem and rollback regressions use actual repositories."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.git_ops import (
    GitError,
    NotAWorktreeError,
    github as git_github,
    mutations as git_mutations,
    process as git_process,
    queries as git_queries,
)
from tests.conftest import _make_repo_with_main
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import (
    bare_remote as _bare_remote,
    commit as _commit,
    git as _git,
    init_repo as _init_repo,
    install_diff_base_git_shim,
    write_and_stage,
)


def _case_dir(tmp_path: Path, index: int) -> Path:
    case = tmp_path / f"case-{index}"
    case.mkdir()
    return case


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


def test_resolve_diff_merge_base_rejects_and_redacts_external_git_failures(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only one canonical lowercase SHA-1 is admitted; external Git failures never echo private output."""
    head = _git(repo, "rev-parse", "HEAD")
    hidden = "A PRIVATE VALUE THAT CANNOT APPEAR"
    cases = [
        ("malformed-head", "branch-focus recorded head commit cannot be resolved", "PRIVATE_STDOUT_SENTINEL"),
        ("invalid-utf8-head", "branch-focus recorded head commit probe failed", "\\xff\\xfe"),
        ("uppercase-head", "branch-focus recorded head commit cannot be resolved", hidden),
        ("abbreviated-head", "branch-focus recorded head commit cannot be resolved", hidden),
        ("sha256-head", "branch-focus recorded head commit cannot be resolved", hidden),
        ("malformed-merge", "branch-focus diff merge-base cannot be resolved", "PRIVATE_MERGE_SENTINEL"),
        ("remove-after-preference", "branch-focus recorded head commit probe failed", "diff base git shim"),
    ]
    for index, (mode, expected_message, private_fragment) in enumerate(cases):
        with pytest.MonkeyPatch.context() as shimmed:
            with install_diff_base_git_shim(_case_dir(tmp_path, index), shimmed, mode=mode, head=head):
                with pytest.raises(GitError) as raised:
                    git_ops.resolve_diff_merge_base(repo, "main", head)
        message = str(raised.value)
        assert message == expected_message, mode
        assert private_fragment not in message and str(repo) not in message and head not in message, mode
        assert len(message) < 200, mode

    def failed_probe(*_args: Any, **_kwargs: Any) -> str:
        raise GitError("PRIVATE_PROBE_SENTINEL")

    monkeypatch.setattr(git_queries, "_prefer_remote_base", failed_probe)
    with pytest.raises(GitError) as raised:
        git_ops.resolve_diff_merge_base(repo, "main", _git(repo, "rev-parse", "HEAD"))
    assert str(raised.value) == "branch-focus preferred base probe failed"


def _repository_identity(repo: Path) -> tuple[str, str, bytes, str]:
    """Observe HEAD, worktree/index status, exact staged bytes, and all refs."""
    return (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "status", "--short"),
        git_ops.staged_patch(repo),
        _git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
    )


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
) -> None:
    real_clone = git_ops.clone

    def clone_with_alternates(
        remote_url: str,
        target: Path,
        *,
        blobless: bool = False,
        no_local: bool = False,
        timeout: int = 300,
    ) -> None:
        assert no_local
        real_clone(remote_url, target, blobless=blobless, no_local=no_local, timeout=timeout)
        (target / ".git" / "objects" / "info" / "alternates").write_text(str(repo / ".git" / "objects") + "\n")

    monkeypatch.setattr(git_mutations, "clone", clone_with_alternates)
    with pytest.raises(git_ops.SnapshotPreparationError, match="alternates or remotes"):
        git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)


def test_independent_snapshot_rejects_inherited_git_overrides_before_mutation(tmp_path: Path) -> None:
    """Every unsafe key must appear in the complete rejection of one real inherited environment."""
    repo = tmp_path / "source"
    _init_repo(repo)
    # This test snapshots every metadata byte. Background maintenance can
    # remove its transient lock while copytree or the byte checks read it.
    _git(repo, "config", "maintenance.auto", "false")
    write_and_stage(repo, "app.py", b"VALUE = 1\n")
    _commit(repo, "initial")
    (repo / "app.py").write_bytes(b"VALUE = 2\n")
    external = tmp_path / "external"
    shutil.copytree(repo / ".git", external)
    destination = tmp_path / "snapshot"

    def file_bytes(root: Path) -> dict[str, bytes]:
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    paths = {"SOURCE_OBJECTS": repo / ".git/objects", "EXTERNAL_OBJECTS": external / "objects", "EXTERNAL": external}
    before_source, before_external = file_bytes(repo), file_bytes(external)
    redirects = ("GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES", "GIT_COMMON_DIR", "GIT_DIFF_OPTS",
                 "GIT_DIR", "GIT_EXTERNAL_DIFF", "GIT_GRAFT_FILE", "GIT_INDEX_FILE", "GIT_NAMESPACE",
                 "GIT_OBJECT_DIRECTORY", "GIT_PREFIX", "GIT_REFERENCE_BACKEND", "GIT_REPLACE_REF_BASE",
                 "GIT_SHALLOW_FILE", "GIT_TEMPLATE_DIR", "GIT_TRACE", "GIT_WORK_TREE")
    namespace = ("GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_KEY_0", "GIT_CONFIG_NOSYSTEM",
                 "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_VALUE_0")
    cases: list[tuple[dict[str, str], str]] = [
        ({name: {"GIT_ALTERNATE_OBJECT_DIRECTORIES": "SOURCE_OBJECTS",
                 "GIT_OBJECT_DIRECTORY": "EXTERNAL_OBJECTS"}.get(name, "EXTERNAL") for name in redirects},
         ", ".join(redirects)),
        (dict.fromkeys(namespace, "EXTERNAL"), ", ".join(namespace)),
        ({"GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "commit.gpgsign", "GIT_CONFIG_VALUE_0": "INVALID_SIGNING_VALUE",
          "GIT_CONFIG_KEY_1": "core.excludesfile", "GIT_CONFIG_VALUE_1": "/dev/null"},
         "GIT_CONFIG_COUNT, GIT_CONFIG_KEY_0, GIT_CONFIG_KEY_1, GIT_CONFIG_VALUE_0, GIT_CONFIG_VALUE_1"),
        ({"GIT_EXEC_PATH": "EXTERNAL"}, "GIT_EXEC_PATH"),
    ]
    for settings, diagnostic in cases:
        environment = {name: str(paths.get(value, value)) for name, value in settings.items()}
        with pytest.MonkeyPatch.context() as poison:
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
        assert not destination.exists(), diagnostic
        assert file_bytes(repo) == before_source and file_bytes(external) == before_external, diagnostic


def test_independent_snapshot_refuses_ambient_git_overrides_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
) -> None:
    """Empty, indexed, unresolvable and malformed Git overrides refuse with a bounded, redacted diagnostic."""
    destination = tmp_path / "snapshot"
    for key in tuple(os.environ):
        if key.startswith(("GIT_CONFIG", "GIT_EXEC_PATH")):
            monkeypatch.delenv(key)

    with monkeypatch.context() as empty:
        # Empty overrides remain unsafe even before the source repository can be resolved.
        empty.setenv("GIT_EXEC_PATH", "")
        with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
            git_ops.prepare_independent_snapshot(tmp_path / "absent", destination, include_untracked=False)
        assert "GIT_EXEC_PATH" in str(raised.value)
        assert os.environ["GIT_EXEC_PATH"] == ""

    with monkeypatch.context() as indexed:
        indexed.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
        indexed.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "external"))
        indexed.setenv("GIT_CONFIG_COUNT", "1")
        before = dict(os.environ)
        with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git") as raised:
            git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
        for name in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
            assert name in str(raised.value)
        assert str(tmp_path / "external") not in str(raised.value)
        assert dict(os.environ) == before

    with monkeypatch.context() as exec_path:
        exec_path.setenv("GIT_EXEC_PATH", "/trusted-looking/git-core")
        exec_path.setenv("PATH", str(tmp_path / "missing-bin"))
        before = dict(os.environ)
        with pytest.raises(git_ops.SnapshotPreparationError, match="inherited Git"):
            git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
        assert dict(os.environ) == before

    with monkeypatch.context() as traced:
        # A known violation refuses before probing GIT_EXEC_PATH, so the trace sink is never written.
        trace_file = tmp_path / "git-trace-side-effect.log"
        traced.setenv("GIT_TRACE", str(trace_file))
        traced.setenv("GIT_EXEC_PATH", str(tmp_path / "invalid git exec path"))
        with pytest.raises(git_ops.SnapshotPreparationError) as raised:
            git_ops.prepare_independent_snapshot(repo, destination, include_untracked=False)
        assert "GIT_TRACE" in str(raised.value) and str(trace_file) not in str(raised.value)
        assert not trace_file.exists()

    with monkeypatch.context() as hook_environment:
        # Git exports its own default exec path to hooks; that value is not an override.
        hook_environment.setenv("GIT_EXEC_PATH", _git(repo, "--exec-path"))
        before = dict(os.environ)
        accepted = git_ops.prepare_independent_snapshot(repo, tmp_path / "accepted", include_untracked=False)
        assert accepted.repo.is_dir() and dict(os.environ) == before

    newline_name = "GIT_TRACE_PRIVATE\nSECRET_NAME"
    oversized_name = "GIT_CONFIG_" + "S" * 500
    private_value = "PRIVATE_ENV_VALUE_SENTINEL"
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
    for private in (newline_name, oversized_name, private_value, "SECRET_NAME"):
        assert private not in message
    assert len(message) < 500
    assert not destination.exists()
    assert os.environ[newline_name] == private_value
    assert os.environ[oversized_name] == private_value


def test_strict_queries_reject_failed_enumeration_where_soft_queries_degrade(tmp_path: Path) -> None:
    """Safety guards must not mistake a failed Git enumeration for a clean or empty tree."""
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path / "does-not-exist")
    with pytest.raises(GitError):
        git_ops.list_local_branches(tmp_path / "not-a-repo")
    with pytest.raises(GitError):
        git_ops.changed_files_against(tmp_path, "HEAD")
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
    with pytest.raises(GitError):
        git_ops.ls_files(tmp_path, strict=True)
    with pytest.raises(GitError):
        git_ops.list_untracked(tmp_path, strict=True)
    assert git_ops.list_remotes(tmp_path) == []
    assert git_ops.object_alternates(tmp_path) == ()
    assert git_ops.ls_tree_files(tmp_path, "HEAD") == []
    assert git_ops.ls_files(tmp_path) == []
    assert git_ops.list_untracked(tmp_path) == []
    repo = _make_repo_with_main(tmp_path / "src")
    (repo / "nested").mkdir()
    with pytest.raises(NotAWorktreeError, match="pass the worktree root"):
        git_ops.assert_is_worktree(repo / "nested")
    (repo / "tracked.txt").write_text("x")
    (repo / "untracked.txt").write_text("y")
    _git(repo, "add", "tracked.txt")
    assert "tracked.txt" in git_ops.ls_files(repo, strict=True) == git_ops.ls_files(repo)
    assert "untracked.txt" in git_ops.list_untracked(repo, strict=True) == git_ops.list_untracked(repo)
    empty = tmp_path / "empty"
    _init_repo(empty)
    assert git_ops.merge_base(empty, "main") is None


def test_git_workspace_detection_fails_closed_on_unrecognized_or_boundary_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting every no-repository prefix would turn Git failures into unrestricted recursive scans.

    A filesystem-boundary diagnostic must still inspect real ancestor markers.
    """
    # Real Git: nested and linked directories are workspaces, standalone ones are not, bare storage refuses.
    repo = _make_repo_with_main(tmp_path / "real")
    (repo / "src").mkdir()
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
    for directory in (repo, repo / "src", linked):
        assert git_ops.is_git_workspace(directory), directory
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    assert not git_ops.is_git_workspace(standalone)
    bare = tmp_path / "bare"
    bare.mkdir()
    _git(bare, "init", "--bare")
    (bare / "src").mkdir()
    with pytest.raises(GitError):
        git_ops.is_git_workspace(bare / "src")
    for diagnostic in (
        "fatal: not a git repository: unexpected failure\n",
        "fatal: not a git repository (or any parent up to mount point /)\n",
        "fatal: not a git repository (or any of the parent directories) but discovery failed\n",
    ):
        _patch_subprocess_run(monkeypatch, returncode=128, stderr=diagnostic)
        with pytest.raises(GitError):
            git_ops.is_git_workspace(tmp_path)
    nested = tmp_path / "src"
    nested.mkdir()
    _patch_subprocess_run(
        monkeypatch,
        returncode=128,
        stderr=(
            "fatal: not a git repository (or any parent up to mount point /mounted volume)\n"
            "Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).\n"
        ),
    )
    assert not git_ops.is_git_workspace(nested)
    (tmp_path / ".git").write_text("gitdir: missing\n")
    with pytest.raises(GitError):
        git_ops.is_git_workspace(nested)
    (tmp_path / ".git").unlink()
    (tmp_path / ".git").symlink_to(tmp_path / "missing")
    with pytest.raises(GitError):
        git_ops.is_git_workspace(nested)


def test_ref_queries_reject_option_injection_and_inexact_pr_refs(repo: Path) -> None:
    """Dash-leading or missing refs never reach Git as options; PR bases and heads must be exact."""
    assert git_ops.ref_exists(repo, "-not-a-ref") is False
    assert git_ops.ref_exists(repo, "--exec=evil") is False
    assert git_ops.is_ancestor(repo, "missing", "HEAD") is False
    assert git_ops.is_ancestor(repo, "-not-a-ref", "HEAD") is False
    assert git_ops.merge_base(repo, "-no-such-ref") is None
    assert git_ops.merge_base(repo, "main", "-no-such-ref") is None
    for head in ("HEAD", "f" * 40):
        with pytest.raises(GitError, match="exact PR head"):
            git_ops.resolve_pr_merge_base(repo, [], "refs/heads/main", head)
    for local_ref in ("main", "refs/heads/-bad", "refs/heads/main~1"):
        expected = "PR base refs/heads/-bad is unavailable" if local_ref == "refs/heads/-bad" else "invalid PR base ref"
        with pytest.raises(GitError, match=f"^{expected}"):
            git_ops.resolve_pr_merge_base(repo, [], local_ref, git_ops.head_sha(repo))

    oldest = git_ops.head_sha(repo)
    write_and_stage(repo, "base-update.txt", "new base\n")
    newer = _commit(repo, "base update")
    _git(repo, "checkout", "-b", "feature")
    write_and_stage(repo, "feature.txt", "feature\n")
    head = _commit(repo, "feature")
    _git(repo, "update-ref", "refs/remotes/one/main", newer)
    _git(repo, "update-ref", "refs/remotes/two/main", oldest)
    with pytest.raises(GitError, match="fetch/align the base remote"):
        git_ops.resolve_pr_merge_base(
            repo, ["refs/remotes/one/main", "refs/remotes/absent/main", "refs/remotes/two/main"], "refs/heads/main",
            head,
        )

    assert git_ops.remote_urls(repo) == {}
    assert git_ops.remote_url(repo) is None
    _git(repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    _git(repo, "remote", "add", "upstream", "git@github.com:other/widgets.git")
    assert git_ops.remote_urls(repo) == {
        "origin": "https://github.com/acme/widgets.git",
        "upstream": "git@github.com:other/widgets.git",
    }
    assert git_ops.remote_url(repo) == "https://github.com/acme/widgets.git"
    assert git_ops.remote_url(repo, "absent") is None
    # A base whose upstream is ahead measures the feature against that upstream, not the stale local branch.
    upstream_tip = _git(repo, "commit-tree", _git(repo, "rev-parse", "HEAD^{tree}"), "-p", head, "-m", "upstream")
    _git(repo, "update-ref", "refs/remotes/origin/main", upstream_tip)
    _git(repo, "branch", "--set-upstream-to=origin/main", "main")
    assert git_ops.merge_base(repo, "main") == head
    _git(repo, "config", "remote.broken.fetch", "+refs/heads/*:refs/remotes/broken/*")
    with pytest.raises(GitError, match="remote 'broken'.*fetch URL"):
        git_ops.remote_urls(repo)

    (repo / ".git" / "HEAD").write_text("invalid head contents\n")
    with pytest.raises(GitError):
        git_ops.is_unborn_head(repo)


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


def test_worktree_move_propagates_git_failure(tmp_path: Path, repo: Path) -> None:
    missing = tmp_path / "missing worktree"
    destination = tmp_path / "destination"
    with pytest.raises(GitError, match="git worktree move"):
        git_ops.worktree_move(repo, missing, destination)
    assert not destination.exists()


def _timeout_run(*, cmd: list[str], timeout: float, calls: dict[str, int] | None = None) -> Any:
    """Return a ``subprocess.run`` double that raises ``TimeoutExpired`` on every call."""

    def run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    return run


def test_immutable_diff_refuses_to_replay_completed_setup_after_timeout(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed init can lose its response; preparation must fail without replaying it.

    Generic mutation wrappers never enter the immutable owner's temporary Git
    setup, and ordinary diff coverage cannot expose timeout recovery behavior.
    """
    from daydream.git_ops import immutable

    head = _git(repo, "rev-parse", "HEAD")
    before = _repository_identity(repo)
    original_run = git_process._bounded_subprocess
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

    monkeypatch.setattr(git_process, "_bounded_subprocess", completed_init_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        immutable.immutable_review_diff(repo, head, head)
    assert len(initialized) == 1 and not initialized[0].exists()
    assert _repository_identity(repo) == before


def test_immutable_diff_bounds_physical_metadata_before_borrowing_objects(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternates expose metadata skipped by object copying, including maintained sidecars."""
    from daydream.git_ops import immutable

    head = _git(repo, "rev-parse", "HEAD")
    metadata = repo / ".git/objects/pack/multi-pack-index.d"
    metadata.mkdir()
    for name in ("first", "second"):
        (metadata / name).write_bytes(b"x" * (1024 * 1024 + 1))
    monkeypatch.setattr(immutable, "_MAX_OBJECT_BYTES", 2 * 1024 * 1024)
    with pytest.raises(GitError, match="object population exceeds the supported size"):
        immutable.immutable_review_diff(repo, head, head)


def test_bounded_git_capture_reaps_failures_and_retries_only_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that overflows the output bound or outlives the deadline is killed and reaped; only timeouts retry."""
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text('#!/bin/sh\nif [ "$1" = overflow ]; then head -c 65536 /dev/zero; fi\n'
                    'if [ "$1" = stderr-overflow ]; then head -c 65536 /dev/zero >&2; fi\n'
                    'if [ "$1" = nonzero ]; then echo "bounded failure diagnostic" >&2; exit 7; fi\nexec sleep 30\n')
    shim.chmod(0o755)
    processes: list[subprocess.Popen[Any]] = []
    real_popen = subprocess.Popen

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setenv("PATH", str(shim_dir) + os.pathsep + os.defpath)
    oversized = "output exceeds the supported size"
    for mode, error, expected, timeout, launches in [
        ("overflow", GitError, oversized, 5.0, 1), ("stderr-overflow", GitError, oversized, 5.0, 1),
        ("nonzero", GitError, "bounded failure diagnostic", 5.0, 1),
        ("timeout", git_ops.GitTimeoutError, "timed out", 0.2, 2),
    ]:
        processes.clear()
        with pytest.raises(error, match=expected):
            git_process._run_git(tmp_path, [mode], timeout=timeout, retries=1, max_output_bytes=1024,
                                 error_context="bounded command failed")
        assert len(processes) == launches, mode
        for process in processes:
            assert process.returncode is not None
            with pytest.raises(ProcessLookupError):
                os.kill(process.pid, 0)


def test_gh_timeouts_retry_only_idempotent_reads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A timed-out mutation may have completed, so only reads retry; the retry budget is validated per call."""
    mutations: list[Any] = [
        lambda: git_ops.fetch(tmp_path),
        lambda: git_ops.gh_pr_create(tmp_path, head="feature", base="main", title="t", body="b"),
        lambda: git_ops.gh_issue_create(tmp_path, title="t", body="b", repo_slug="octocat/hello"),
        lambda: git_ops.gh_api(tmp_path, "graphql", method="POST", input_data={"query": "mutation { x }"}),
    ]
    calls = {"n": 0}
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["command"], timeout=60, calls=calls))
    for operation in mutations:
        calls["n"] = 0
        with pytest.raises(git_ops.GitTimeoutError):
            operation()
        assert calls["n"] == 1  # no retries for mutating operations
    calls["n"] = 0
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(tmp_path, "/user", idempotent=True)
    assert calls["n"] == git_process._gh_retries() + 1

    for env_value, expected_attempts, expected_warning in [
        ("", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='' is not a valid integer; using default 2"),
        ("-1", 3, "DAYDREAM_GH_TIMEOUT_RETRIES='-1' is negative; using default 2"),
        ("0", 1, None),
    ]:
        caplog.clear()
        monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", env_value)
        calls["n"] = 0
        with pytest.raises(git_ops.GitTimeoutError) as exc:
            git_ops.gh_repo_view(tmp_path)
        assert calls["n"] == expected_attempts
        suffix = f" ({expected_attempts} attempts)" if expected_attempts > 1 else ""
        assert str(exc.value) == "gh repo view --json nameWithOwner -q .nameWithOwner timed out after 60s" + suffix
        if expected_warning is None:
            assert "DAYDREAM_GH_TIMEOUT_RETRIES" not in caplog.text
        else:
            assert expected_warning in caplog.text

    # A zero or malformed timeout keeps the 60s default and warns.
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", "0")
    for raw, problem in (("0", "must be positive"), ("6O", "is not a valid integer")):
        caplog.clear()
        monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_SECONDS", raw)
        with pytest.raises(git_ops.GitTimeoutError, match="timed out after 60s"):
            git_ops.gh_repo_view(tmp_path)
        assert f"DAYDREAM_GH_TIMEOUT_SECONDS='{raw}' {problem}; using default 60" in caplog.text

    caplog.clear()
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", "1")
    ok = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="octocat/hello\n", stderr="")
    calls["n"] = 0
    def flaky(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["gh"], timeout=7)
        return ok

    monkeypatch.setattr(subprocess, "run", flaky)
    assert git_ops.gh_repo_view(tmp_path) == ("octocat", "hello")
    assert calls["n"] == 2  # timed out once, then succeeded
    calls["n"] = 0
    monkeypatch.setattr(subprocess, "run", _timeout_run(cmd=["gh"], timeout=7, calls=calls))
    with pytest.raises(git_ops.GitTimeoutError) as exhausted:
        git_ops.gh_pr_diff(tmp_path, 7)
    assert calls["n"] == 2
    assert str(exhausted.value) == "gh pr diff 7 timed out after 7s (2 attempts)"
    assert "DAYDREAM_GH_TIMEOUT" not in caplog.text
    # A credential-bearing header never reaches the timeout error or the retry warning.
    caplog.clear()
    with pytest.raises(git_ops.GitTimeoutError) as redacted:
        git_ops.gh_api(tmp_path, "/app/installations", headers={"Authorization": "Bearer jwt-super-secret-xyz"},
                       idempotent=True)
    assert "jwt-super-secret-xyz" not in str(redacted.value) and "Authorization: ***" in str(redacted.value)
    warnings = [record.getMessage() for record in caplog.records]
    assert any(warning.startswith("gh timed out after") for warning in warnings)
    assert all("jwt-super-secret-xyz" not in warning and "Authorization" not in warning for warning in warnings)
    caplog.clear()
    with pytest.raises(git_ops.GitTimeoutError) as masked:
        git_ops.gh_api(tmp_path, "/app-manifests/manifest-code-sentinel/conversions", method="POST", idempotent=True)
    assert "manifest-code-sentinel" not in str(masked.value) and "/app-manifests/***/" in str(masked.value)
    assert caplog.records and all("/app-manifests" not in record.getMessage() for record in caplog.records)


def test_gh_issue_wrappers_keep_bodies_out_of_argv_and_return_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue bodies travel through an unlinked file, never argv; listed rows retain fingerprint bodies."""
    captured: dict[str, Any] = {}

    def fake_create(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Snapshot argv + cwd + body-file contents at call time (the tempfile
        # is unlinked after `_run_gh` returns, matching real gh's read-then-exit).
        captured["argv"] = list(args)
        captured["cwd"] = kwargs.get("cwd")
        i_body = list(args).index("--body-file")
        captured["body"] = Path(args[i_body + 1]).read_text()
        return subprocess.CompletedProcess(
            args=list(args),
            returncode=0,
            stdout="https://github.com/octocat/hello/issues/42\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_create)
    labels = ["daydream", "tech-debt"]
    url = git_ops.gh_issue_create(
        tmp_path,
        title="out-of-scope: refactor handler.py error path",
        body="evidence and rationale\nthat the fix loop overreached\n",
        repo_slug="octocat/hello",
        labels=labels,
    )
    assert url == "https://github.com/octocat/hello/issues/42"
    argv = captured["argv"]
    assert argv[:5] == ["gh", "issue", "create", "--repo", "octocat/hello"]
    assert argv[argv.index("--title") + 1] == "out-of-scope: refactor handler.py error path"
    body_path = argv[argv.index("--body-file") + 1]
    assert captured["body"] == "evidence and rationale\nthat the fix loop overreached\n"
    assert not Path(body_path).exists()
    assert "that the fix loop overreached" not in argv
    assert [argv[i + 1] for i, tok in enumerate(argv) if tok == "--label"] == labels
    assert captured["cwd"] == tmp_path

    rows = [
        {
            "number": 42,
            "title": "[daydream] out-of-scope finding: notes.txt",
            "body": "desc\n<!-- daydream-scope-finding: abc123 -->",
            "url": "https://github.com/octocat/hello/issues/42",
        },
        {"number": 7, "title": "unrelated", "body": "", "url": ""},
    ]

    def fake_list(args: Any, *pargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        captured["argv"] = list(args)
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout=json.dumps(rows), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_list)
    assert git_ops.gh_issue_list(tmp_path, search="out-of-scope", repo_slug="octocat/hello") == rows
    argv = captured["argv"]
    assert argv[:3] == ["gh", "issue", "list"]
    assert argv[argv.index("--state") + 1] == "open"
    assert "--json" in argv and "number,title,body,url" in argv
    assert argv[argv.index("--search") + 1] == "out-of-scope"
    assert "--repo" in argv and "octocat/hello" in argv


def test_gh_read_wrappers_reject_invalid_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only anchored absence reads as no PR; malformed gh output never becomes an empty result."""
    for pr, stderr in [
        (None, 'no pull requests found for branch "feature"\n'),
        (42, "GraphQL: Could not resolve to a PullRequest with the number of 42. (repository.pullRequest)\n"),
    ]:
        _patch_subprocess_run(monkeypatch, returncode=1, stderr=stderr)
        assert git_ops.gh_pr_view(tmp_path, pr) is None, stderr
    invalid: list[tuple[str, str, Any]] = [
        *[(stdout, "invalid JSON|JSON object", lambda: git_ops.gh_pr_view(tmp_path, 42))
          for stdout in ("not-json", "[]")],
        *[(stdout, "invalid JSON|JSON list|row", lambda: git_ops.gh_pr_list_for_branch(tmp_path, "feature"))
          for stdout in ("not-json", "{}", '[{"number": 1}, null]')],
        *[(slug + "\n", "invalid repository slug", lambda: git_ops.gh_repo_view_required(tmp_path))
          for slug in ("owner/repo/extra", "owner/ ")],
    ]
    for stdout, match, read in invalid:
        _patch_subprocess_run(monkeypatch, stdout=stdout)
        try:
            with pytest.raises(GitError, match=match):
                read()
        except BaseException as exc:
            exc.add_note(f"case: {stdout!r}")
            raise
    # Unanchored or mismatched failures are errors, never absence.
    for pr, stderr in [
        (42, "GraphQL: Could not resolve to a PullRequest with the number of 41. (repository.pullRequest)"),
        (None, 'prefix: no pull requests found for branch "feature"'), (42, "HTTP 404: resource not found"),
    ]:
        _patch_subprocess_run(monkeypatch, returncode=1, stderr=stderr)
        try:
            with pytest.raises(GitError, match=re.escape(stderr)):
                git_ops.gh_pr_view(tmp_path, pr)
        except BaseException as exc:
            exc.add_note(f"case: {stderr!r}")
            raise
    # A bare GitHub token is redacted while the safe diagnostic survives.
    _patch_subprocess_run(monkeypatch, returncode=1,
                          stderr="HTTP 401: authentication required for token ghp_abcdefghijklmnopqrstuvwxyz1234567890")
    with pytest.raises(GitError) as unauthenticated:
        git_ops.gh_repo_view_required(tmp_path)
    assert "ghp_abcdefghij" not in str(unauthenticated.value)
    assert "authentication required" in str(unauthenticated.value)
    # Failed dedup lookup must allow filing the finding instead of dropping or blocking it.
    _patch_subprocess_run(monkeypatch, returncode=1, stderr="gh: not authenticated\n")
    assert git_ops.gh_issue_list(tmp_path) == []
    _patch_subprocess_run(monkeypatch, stdout='{"id": 1}\nnot-json\n')
    with pytest.raises(git_ops.GitError, match="invalid JSON"):
        git_ops.gh_api(tmp_path, "/app/installations", jq=".[]")

    def spawn_failure(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("gh executable failed to spawn")

    monkeypatch.setattr(subprocess, "run", spawn_failure)
    with pytest.raises(GitError) as failed:
        git_ops.gh_api(tmp_path, "/app/installations", headers={"Authorization": "Bearer jwt-super-secret-xyz"})
    assert "jwt-super-secret-xyz" not in str(failed.value) and "Authorization: ***" in str(failed.value)
    # App manifest conversion codes are masked on spawn, API, rate-limit and invalid-JSON failures.
    conversion = "/app-manifests/manifest-code-sentinel/conversions"
    with pytest.raises(GitError) as failed:
        git_ops.gh_api(tmp_path, conversion, method="POST")
    for stderr, returncode, stdout in ((f"synthetic API failure: {conversion}", 1, ""),
                                       (f"secondary rate limit: {conversion}", 1, ""), ("", 0, "not-json")):
        _patch_subprocess_run(monkeypatch, returncode=returncode, stdout=stdout, stderr=stderr)
        with pytest.raises(GitError) as masked:
            git_ops.gh_api(tmp_path, conversion, method="POST")
        assert "manifest-code-sentinel" not in str(masked.value), stderr or stdout
        assert "/app-manifests/***/conversions" in str(masked.value), stderr or stdout
    assert "manifest-code-sentinel" not in str(failed.value) and "/app-manifests/***/" in str(failed.value)


def test_gh_read_wrappers_fall_back_to_rest_when_graphql_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an HTTP refusal of the GraphQL endpoint degrades porcelain reads to gh-selected REST."""
    def pull(number: int, ref: str, *, merged: bool = False) -> dict[str, Any]:
        return {"number": number, "title": "t", "body": None, "state": "closed" if merged else "open",
                "merged_at": "2026-11-01T00:00:00Z" if merged else None, "html_url": f"https://github.com/o/r/pull/{number}",
                "head": {"ref": ref, "sha": "a" * 40, "user": {"login": "fork"},
                         "repo": {"name": "r", "full_name": "fork/r"}},
                "base": {"ref": "main", "sha": "b" * 40}}

    rest = {"repos/{owner}/{repo}": {"full_name": "Canonical/r"},
            "repos/{owner}/{repo}/pulls/7": pull(7, "feature", merged=True),
            "repos/{owner}/{repo}/pulls?state=open&per_page=100": [pull(8, "other"), pull(9, "feature")]}

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[1] != "api":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=(
                "HTTP 403: GraphQL is not available; use the REST API (https://api.github.com/graphql)\n"))
        if cmd[-1] not in rest:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="gh: Not Found (HTTP 404)\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(rest[cmd[-1]]), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_ops.gh_repo_view_required(tmp_path) == ("Canonical", "r")
    assert git_ops.gh_pr_view(tmp_path, 7) == {
        "number": 7, "title": "t", "body": "", "state": "MERGED", "headRefName": "feature", "baseRefName": "main",
        "headRefOid": "a" * 40, "url": "https://github.com/o/r/pull/7",
        "headRepository": {"name": "r", "nameWithOwner": "fork/r"}, "headRepositoryOwner": {"login": "fork"},
    }
    assert git_ops.gh_pr_view(tmp_path, 404) is None
    assert [row["number"] for row in git_ops.gh_pr_list_for_branch(tmp_path, "feature")] == [9]
    for branch, expected in (("feature", 9), ("unopened", None)):
        monkeypatch.setattr(git_ops.queries, "current_branch", lambda _repo, branch=branch: branch)
        assert (git_ops.gh_pr_view(tmp_path) or {}).get("number") == expected
    monkeypatch.setattr(git_ops.queries, "current_branch", lambda _repo: None)
    with pytest.raises(GitError, match="current branch"):
        git_ops.gh_pr_view(tmp_path)
    # gh api names no URL, so only the graphql endpoint's 403/404 counts; rate limits stay rate limits.
    for endpoint, stderr, error_type in [
        ("graphql", "gh: GraphQL is not available (HTTP 403)", git_ops.GraphQLUnavailableError),
        ("graphql", "gh: API rate limit exceeded (HTTP 403)", git_ops.RateLimitError),
        ("repos/o/r/pulls/1/comments", "gh: Resource not accessible by integration (HTTP 403)", GitError),
    ]:
        _patch_subprocess_run(monkeypatch, returncode=1, stderr=stderr)
        with pytest.raises(GitError) as excinfo:
            git_ops.gh_api(tmp_path, endpoint)
        assert type(excinfo.value) is error_type, stderr


def test_gh_api_input_data_preserves_tempfile_only_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed API payloads are preserved and disclosed for recovery; a successful payload file is deleted."""
    class RequestAuth:
        calls = 0

        def environment_for_request(self) -> dict[str, str]:
            self.calls += 1
            return {"OWNING_RUN": "payload-metadata-test"}

    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", "4")
    responses: dict[str, tuple[int, str, str]] = {
        "http": (1, "", "HTTP 422: Validation failed"), "invalid-json": (0, "not-json", ""),
        "rate-limit": (1, "", "HTTP 429: rate limit; retry-after: 7"), "success": (0, '{"id": 1}', ""),
    }
    for failure, expected_type in [("http", GitError), ("invalid-json", GitError),
                                   ("timeout", git_ops.GitTimeoutError), ("rate-limit", git_ops.RateLimitError),
                                   ("success", None)]:
        attempts: list[list[str]] = []
        auth = RequestAuth()

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            attempts.append(cmd)
            assert kwargs["env"] == {"OWNING_RUN": "payload-metadata-test"}
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
            returncode, stdout, stderr = responses[failure]
            return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        try:
            request = {"bad": "payload"}
            if expected_type is None:
                assert git_ops.gh_api(tmp_path, "repos/o/r/pulls/1/reviews", method="POST", input_data=request,
                                      auth=auth) == {"id": 1}
                assert not Path(attempts[0][attempts[0].index("--input") + 1]).exists()
                continue
            with pytest.raises(expected_type) as excinfo:
                git_ops.gh_api(tmp_path, "repos/o/r/pulls/1/reviews", method="POST", input_data=request, auth=auth)
            preserved = Path(attempts[0][attempts[0].index("--input") + 1])
            try:
                assert type(excinfo.value) is expected_type and len(attempts) == 1 and auth.calls == 1
                message = str(excinfo.value)
                if failure == "timeout":
                    assert "timed out" in message
                else:
                    assert "payload preserved at" in message and str(preserved) in message
                assert json.loads(preserved.read_text(encoding="utf-8")) == request
                assert excinfo.value.preserved_payload_path == preserved
                if isinstance(excinfo.value, git_ops.RateLimitError):
                    assert excinfo.value.retry_after == 7
            finally:
                preserved.unlink(missing_ok=True)
        except BaseException as exc:
            exc.add_note(f"case: {failure}")
            raise


def test_update_refs_rejects_injectable_ref_names_before_shell_out(tmp_path: Path) -> None:
    """Reject invalid refs and OIDs before they can inject extra update-ref stdin commands."""
    repo = _make_repo_with_main(tmp_path, name="update_refs_inject")
    oid = git_ops.head_sha(repo)
    git_ops.update_refs(repo, {})  # empty input is a no-op
    good = {f"refs/heads/{name}": oid for name in ("unlock", "release/9.9", "topic.LOCK")}
    git_ops.update_refs(repo, good)
    for ref, expected in good.items():
        assert _git(repo, "rev-parse", ref) == expected
    with pytest.raises(git_ops.GitError, match="git update-ref --stdin failed"):
        # ".." passes the Python guards but git rejects it, so the whole batch aborts: nothing is written.
        git_ops.update_refs(repo, {"refs/heads/ok": oid, "refs/heads/foo..bar": oid})
    assert "ok" not in git_ops.list_local_branches(repo)
    for bad in ("refs/heads/bad\nname", "refs/heads/two words", "-dash-start"):
        with pytest.raises(git_ops.GitError, match="invalid ref name"):
            git_ops.update_refs(repo, {bad: oid})
    with pytest.raises(git_ops.GitError, match="invalid OID"):
        git_ops.update_refs(repo, {"refs/heads/ok": "not-a-sha"})
    assert "bad" not in git_ops.list_local_branches(repo)
    assert "two" not in git_ops.list_local_branches(repo)


def test_clone_flags_failures_and_credential_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(git_ops.GitError, match="git clone .* failed"):
        git_ops.clone("file:///nonexistent/repo.git", tmp_path / "nope")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    git_ops.clone("https://example.com/repo.git", tmp_path / "out", blobless=True, no_local=True)
    assert captured["cmd"] == [
        "git",
        "clone",
        "--filter=blob:none",
        "--no-local",
        "https://example.com/repo.git",
        str(tmp_path / "out"),
    ]
    _patch_subprocess_run(
        monkeypatch,
        returncode=128,
        stderr=(
            "fatal: unable to access 'https://user:ghp_canaryfake123@unreachable.invalid/o/r.git/': "
            "Could not resolve host"
        ),
    )
    with pytest.raises(GitError) as excinfo:
        git_ops.clone("https://user:ghp_canaryfake123@unreachable.invalid/o/r.git", tmp_path / "t", timeout=5)
    assert "ghp_canaryfake123" not in str(excinfo.value)
    # Non-URL diagnostics must not trigger quadratic scheme-prefix searches.
    for text in ("A" * 100_000, "A." * 50_000):
        assert git_process._redact_sensitive_text(text) == text
    for scheme in ("https", "ssh", "git+custom.transport"):
        raw = f"failed {scheme}://user:password@example.invalid/o/r"
        assert git_process._redact_sensitive_text(raw) == f"failed {scheme}://***@example.invalid/o/r"


_FILE_AT_REF_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_gh_file_and_scope_wrappers_validate_before_requests(
    fake_gh: FakeGh,
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slug, ref and the model-derived path all become endpoint text; auth reaches both reads."""
    for slug, ref, path in [
        ("../o/r", _FILE_AT_REF_SHA, "a.py"),
        ("o/r", "main", "a.py"),
        ("o/r", _FILE_AT_REF_SHA, "../../etc/passwd"),
    ]:
        with pytest.raises(GitError):
            git_ops.gh_file_at_ref(tmp_path, slug, ref, path)
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v")
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v", org="acme", repo_slug="o/r")
    assert fake_gh.process_calls() == []
    # No "pr-create" response configured, so the shim exits non-zero.
    with pytest.raises(GitError):
        git_ops.gh_pr_create(git_repo, head="b", base="main", title="t", body="b")

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


def test_worktree_lock_mtime_rejects_nonregular_lock_metadata(tmp_path: Path, repo: Path) -> None:
    wt = tmp_path / "linked" / "same"
    git_ops.worktree_add(repo, wt, "main", detach=True)
    locked = git_ops.git_dir(wt) / "locked"
    locked.mkdir()
    with pytest.raises(GitError, match="lock metadata is unsafe"):
        git_ops.worktree_lock_mtime(wt)
    assert wt.is_dir()
    assert locked.is_dir()


def test_has_executable_pre_push_hook_honors_mode_and_core_hooks_path(tmp_path: Path) -> None:
    default = tmp_path / "default"
    hooks = default / ".git" / "hooks"
    hooks.mkdir(parents=True)
    pp = hooks / "pre-push"
    pp.write_text("#!/bin/sh\n")
    pp.chmod(0o755)
    assert git_ops.has_executable_pre_push_hook(default) is True
    pp.chmod(0o644)  # non-executable => no hook
    assert git_ops.has_executable_pre_push_hook(default) is False
    configured = tmp_path / "configured"
    custom = tmp_path / "myhooks"
    custom.mkdir()
    pp = custom / "pre-push"
    pp.write_text("#!/bin/sh\n")
    pp.chmod(0o755)
    _init_repo(configured)
    _git(configured, "config", "core.hooksPath", str(custom))
    assert git_ops.has_executable_pre_push_hook(configured) is True


def test_feature_history_diff_and_publication_follow_the_same_commit_window(tmp_path: Path) -> None:
    """A real branch advances through empty, untagged, tagged, divergent and published states."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "remote", "add", "origin", str(_bare_remote(tmp_path / "remote.git")))
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
    assert git_ops.log_shas(repo, "absent", since="main") is None  # a vanished ref is not an empty history
    assert git_ops.daydream_commits(repo, "main") is None
    feature = original.copy()
    feature[14] = "line 15 FEAT"
    write_and_stage(repo, "shared.txt", "\n".join(feature) + "\n")
    write_and_stage(repo, "keep.txt", "keep\n")
    write_and_stage(repo, "drop.txt", "drop\n")
    # A tracked path can share a revision's spelling. Every diff layer must
    # separate revisions from paths, including the tracked-working layer.
    write_and_stage(repo, "HEAD/module.py", "VALUE = 1\n")
    assert "HEAD/module.py" in git_ops.snapshot_index(repo).paths
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
    assert "HEAD/module.py" in full and "+VALUE = 1" in full
    assert "keep.txt" in excluded and "drop.txt" not in excluded
    assert not git_ops.remote_contains_commit(repo, "feat/slash-name", tip)
    git_ops.push_branch(repo, "feat/slash-name")
    assert git_ops.remote_contains_commit(repo, "feat/slash-name", tip)
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "feat/slash-name")
    assert git_ops.branch_exists(repo, "feat/slash-name") and git_ops.ref_exists(repo, "feat/slash-name")
    assert not git_ops.commit_exists(repo, "feat/slash-name")


def test_independent_snapshot_materializes_mixed_index_and_worktree_changes(
    tmp_path: Path,
    repo: Path,
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
    snapshot = git_ops.prepare_independent_snapshot(repo, tmp_path / "snapshot", include_untracked=False)
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
    assert not (snapshot.repo / "working").is_file()
    assert not (snapshot.repo / " scratch ").exists()
    assert not (snapshot.repo / "scratch\nline").exists()
    assert _repository_identity(repo) == before

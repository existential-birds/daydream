"""Focused contracts for the authorized fix footprint (issue #1135)."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import stat
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, Backend, ResultEvent
from daydream.deep.scope_issues import ScopeEnforcementResult, enforce_authorized_fix_footprint
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.git_ops import GitError, GitPathState, WorktreeRollbackSnapshot
from daydream.phases import (
    phase_fix_parallel,
)
from daydream.phases.fix_fanout import (
    _isolated_fix_fanout,
)
from daydream.repository_paths import (
    InvalidRepositoryFilePath,
    canonicalize_repository_file_path,
    git_observed_path_is_confined,
)
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import (
    commit as _commit,
    git as _git,
    init_repo,
    work_context,
    write_and_stage,
)


def _seed(repo: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(repo, "add", ".")
    _commit(repo, "seed footprint files")


def _enforce_fix(
    git_repo: Path, footprint: AuthorizedFixFootprint, *, phase: str, preexisting_untracked: dict[str, GitPathState],
    file_scope_issues: bool = False, preexisting_gitlinks: tuple[GitPathState, ...] = (),
) -> ScopeEnforcementResult:
    """Enforce ``footprint`` against ``git_repo`` at HEAD with the test-run context."""
    return enforce_authorized_fix_footprint(
        work_context(git_repo, run_id="test-run"), "HEAD", footprint, preexisting_untracked=preexisting_untracked,
        preexisting_gitlinks=preexisting_gitlinks, phase=phase, round_number=1, file_scope_issues=file_scope_issues,
    )


def _install_scope_boundary_shims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_diff_path: str | None = None,
    watched_paths: tuple[str, ...] = (), fail_issue_create: bool = False,
) -> Path:
    """Install executable Git/gh fakes while leaving production subprocess calls intact."""
    real_git = shutil.which("git")
    assert real_git is not None
    bin_dir = tmp_path / "scope-bin"
    bin_dir.mkdir()
    git_script = bin_dir / "git"
    git_script.write_text(
        "#!/bin/sh\n"
        + (
            f'if [ "$1" = "diff" ] && [ "$2" = "HEAD" ] && [ "$3" = "--" ] '
            f'&& [ "$4" = "{fail_diff_path}" ]; then\n'
            '  echo "SECRET_TOKEN optional evidence failure" >&2\n'
            "  exit 2\n"
            "fi\n"
            if fail_diff_path is not None
            else ""
        )
        + f'exec "{real_git}" "$@"\n',
        encoding="utf-8",
    )
    git_script.chmod(0o755)

    record_path = tmp_path / "gh-calls.jsonl"
    gh_script = bin_dir / "gh"
    gh_script.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

argv = sys.argv[1:]
if argv[:2] == ["issue", "list"]:
    record = Path(os.environ["P01_GH_RECORD"])
    lines = record.read_text().splitlines() if record.is_file() else []
    print(json.dumps([{"body": json.loads(line)["body"]} for line in lines]))
    raise SystemExit(0)
if argv[:2] != ["issue", "create"]:
    print("unexpected gh invocation", file=sys.stderr)
    raise SystemExit(2)
body_path = Path(argv[argv.index("--body-file") + 1])
watched = {
    name: (Path.cwd() / name).read_bytes().hex()
    for name in json.loads(os.environ["P01_GH_WATCHED"])
}
record = {"argv": argv, "body": body_path.read_text(), "watched": watched}
with Path(os.environ["P01_GH_RECORD"]).open("a") as stream:
    stream.write(json.dumps(record) + "\\n")
if os.environ.get("P01_GH_FAIL") == "1":
    print("offline", file=sys.stderr)
    raise SystemExit(1)
print("https://example.invalid/issues/1")
""",
        encoding="utf-8",
    )
    gh_script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("P01_GH_RECORD", str(record_path))
    monkeypatch.setenv("P01_GH_WATCHED", json.dumps(watched_paths))
    if fail_issue_create:
        monkeypatch.setenv("P01_GH_FAIL", "1")
    return record_path


def _items() -> list[dict[str, object]]:
    return [
        {"id": 1, "item_uid": "item:1", "file": "src/a.py", "related_files": ["./src/shared.py", "tests/test_a.py"]},
        {"id": 2, "item_uid": "item:2", "file": "src/b.py", "related_files": ["src/shared.py"]},
        {"id": 3, "item_uid": "item:3", "file": "src/c.py", "related_files": None},
    ]

def test_model_and_reviewed_paths_normalize_or_fail_closed(tmp_path: Path) -> None:
    """Model paths drop "./" and reject symlink escapes; reviewed and related paths must be confined."""
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()

    assert canonicalize_repository_file_path(repo, "./src/a.py") == "src/a.py"
    with pytest.raises(InvalidRepositoryFilePath, match="invalid reviewed repository path"):
        AuthorizedFixFootprint.build(repo, {"/outside"}, [])
    item = {"id": 1, "item_uid": "item:1", "file": "src/a.py", "related_files": "src/b.py"}
    with pytest.raises(InvalidRepositoryFilePath, match="invalid repository file path"):
        AuthorizedFixFootprint.build(repo, set(), [item])

    (repo / "src").symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidRepositoryFilePath, match="invalid repository file path"):
        canonicalize_repository_file_path(repo, "src/a.py")

@pytest.mark.parametrize("reviewed_path", ["native-\udcff.py"])
def test_footprint_treats_reviewed_git_names_separately_from_model_paths(git_repo: Path, reviewed_path: str,) -> None:

    if reviewed_path == "odd\n$name[1].txt":
        assert git_observed_path_is_confined(git_repo, reviewed_path)
    try:
        _seed(git_repo, {reviewed_path: b"before\n", "normal.py": b"before\n"})
    except OSError as exc:
        if exc.errno == errno.EILSEQ:
            pytest.skip("host filesystem rejects non-UTF-8 filenames; exercised on Linux CI")
        raise
    (git_repo / reviewed_path).write_bytes(b"reviewed\n")
    reviewed = set(git_ops.changed_paths_z(git_repo, "HEAD"))
    assert reviewed == {reviewed_path}
    item = {"item_uid": "item:normal", "file": "normal.py"}
    footprint = AuthorizedFixFootprint.build(git_repo, reviewed, [item])

    assert footprint.run_allowed_paths == frozenset({reviewed_path, "normal.py"})
    assert footprint.group_paths([item]) == frozenset({"normal.py"})
    reviewed_event = next(event for event in footprint.events if event.origin == "reviewed")
    assert reviewed_event.path == reviewed_path
    assert reviewed_event.path_kind == "git"
    assert footprint.accept_retarget(git_repo, "item:normal", reviewed_path, phase="fix", round_number=1,) is None
    assert json.loads(json.dumps(footprint.audit_payload("run")))["run_allowed_paths"]

def test_preexisting_untracked_state_is_restored_exactly_and_new_residual_is_removed(git_repo: Path) -> None:
    _seed(git_repo, {"allowed.txt": b"base\n", "outside.txt": b"outside base\n"})
    scratch = git_repo / "scratch.bin"
    scratch.write_bytes(b"\x00user-bytes\xff")
    scratch.chmod(0o640)
    link = git_repo / "scratch-link"
    link.symlink_to("original-target")
    deleted = git_repo / "scratch-deleted"
    deleted.write_bytes(b"restore deleted bytes")
    deleted.chmod(0o600)
    baseline = git_ops.snapshot_untracked_paths(git_repo)
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.txt"}, [])

    (git_repo / "allowed.txt").write_bytes(b"authorized\n")
    scratch.unlink()
    scratch.symlink_to("wrong-target")
    link.unlink()
    link.write_bytes(b"wrong type")
    deleted.unlink()
    (git_repo / "outside.txt").write_bytes(b"unrelated\n")
    residual = git_repo / "odd\n$(ignored).txt"
    residual.write_bytes(b"remove me")

    result = _enforce_fix(git_repo, footprint, phase="post-fix", preexisting_untracked=baseline)

    assert result.retained_paths == frozenset({"allowed.txt"})
    assert result.mutated
    assert scratch.read_bytes() == b"\x00user-bytes\xff"
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o640
    assert link.is_symlink() and os.readlink(link) == "original-target"
    assert deleted.read_bytes() == b"restore deleted bytes"
    assert stat.S_IMODE(deleted.stat().st_mode) == 0o600
    assert (git_repo / "outside.txt").read_bytes() == b"outside base\n"
    assert not residual.exists()
    recorded = {(event.action, event.path) for event in footprint.events}
    assert ("restore", "scratch.bin") in recorded
    assert ("restore", "scratch-link") in recorded
    assert ("restore", "scratch-deleted") in recorded
    assert ("restore", "outside.txt") in recorded
    assert ("remove", "odd\n$(ignored).txt") in recorded
    payload = footprint.audit_payload("session-123", tree_key="abc")
    assert payload["session_id"] == "session-123"
    assert payload["tree_key"] == "abc"
    sequences = [event["sequence"] for event in payload["events"]]
    assert sequences == list(range(1, len(sequences) + 1))
    assert "odd\\n$(ignored).txt" in json.dumps(payload)

def test_scope_issue_diff_failure_skips_only_that_filing_after_restoring_all_residuals(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _seed(git_repo, {"allowed.py": b"allowed\n", "outside-a.py": b"owner a\n", "outside-b.py": b"owner b\n"},)
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.py"}, [])
    index_before = git_ops.snapshot_index(git_repo)
    (git_repo / "outside-a.py").write_bytes(b"unauthorized a\n")
    (git_repo / "outside-b.py").write_bytes(b"unauthorized b\n")
    gh_record = _install_scope_boundary_shims(
        tmp_path, monkeypatch, fail_diff_path="outside-a.py", watched_paths=("outside-a.py", "outside-b.py"),
    )
    result = _enforce_fix(git_repo, footprint, phase="fix", preexisting_untracked={}, file_scope_issues=True)

    assert result.mutated is True
    assert (git_repo / "outside-a.py").read_bytes() == b"owner a\n"
    assert (git_repo / "outside-b.py").read_bytes() == b"owner b\n"
    assert git_ops.snapshot_index(git_repo) == index_before
    assert [(event.action, event.path) for event in footprint.events if event.origin == "guard"] == [
        ("restore", "outside-a.py"), ("restore", "outside-b.py"),
    ]
    assert gh_record.is_file()
    calls = [json.loads(line) for line in gh_record.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["watched"] == {"outside-a.py": b"owner a\n".hex(), "outside-b.py": b"owner b\n".hex()}
    assert "outside-b.py" in calls[0]["body"]
    assert "+unauthorized b" in calls[0]["body"]
    assert "outside-a.py" not in calls[0]["body"]
    warning_output = capsys.readouterr().out
    assert "continuing to" in warning_output
    assert "restoration without filing" in warning_output
    assert "SECRET_TOKEN" not in warning_output
    assert "outside-a.py" not in warning_output

def test_scope_issue_filing_failure_occurs_after_verified_restore_and_is_best_effort(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(git_repo, {"allowed.py": b"allowed\n", "outside.py": b"owner\n"})
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.py"}, [])
    (git_repo / "outside.py").write_bytes(b"unauthorized\n")
    gh_record = _install_scope_boundary_shims(
        tmp_path, monkeypatch, watched_paths=("outside.py",), fail_issue_create=True,
    )
    result = _enforce_fix(git_repo, footprint, phase="fix", preexisting_untracked={}, file_scope_issues=True)

    assert result.mutated is True
    assert (git_repo / "outside.py").read_bytes() == b"owner\n"
    calls = [json.loads(line) for line in gh_record.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["watched"] == {"outside.py": b"owner\n".hex()}
    assert "+unauthorized" in calls[0]["body"]
    # An identical edit already carried by an open issue is restored without a duplicate filing.
    (git_repo / "outside.py").write_bytes(b"unauthorized\n")
    assert _enforce_fix(git_repo, footprint, phase="fix", preexisting_untracked={}, file_scope_issues=True).mutated
    assert (git_repo / "outside.py").read_bytes() == b"owner\n"
    assert len(gh_record.read_text().splitlines()) == 1

@pytest.mark.asyncio
async def test_parallel_fix_cancellation_closes_backend_before_restoring_round_index(git_repo: Path,) -> None:


    _seed(git_repo, {"a.py": b"A = 1\n"})
    round_index = git_ops.snapshot_index(git_repo)
    snapshot = WorktreeRollbackSnapshot(
        ref="HEAD", index=round_index, path_states=git_ops.snapshot_worktree_paths(git_repo, ["a.py"]), untracked={},
    )
    item = {"id": 1, "item_uid": "item:a", "file": "a.py", "description": "a"}
    footprint = AuthorizedFixFootprint.build(git_repo, {"a.py"}, [item])
    staged = anyio.Event()
    stream_closed = anyio.Event()
    never = anyio.Event()

    def responder(cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> Any:
        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            try:
                write_and_stage(cwd, "a.py", b"A = staged by live fixer\n")
                staged.set()
                await never.wait()
                yield ResultEvent(structured_output=None, continuation=None)
            finally:
                stream_closed.set()

        return _gen()

    backend = ScriptedBackend(responder=responder, model="cancel-backend", fanout_concurrency=1, retry_attempts=0,)

    async def _run_phase() -> None:
        await phase_fix_parallel(
            cast(Backend, backend), work_context(git_repo, run_id="test-run"), [item], footprint=footprint,
            round_snapshot=snapshot, limiter_size=1,
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_run_phase)
        await staged.wait()
        task_group.cancel_scope.cancel()

    assert stream_closed.is_set()
    assert git_ops.snapshot_index(git_repo) == round_index
    assert (git_repo / "a.py").read_bytes() == b"A = 1\n"

@pytest.mark.asyncio
async def test_fanout_cancellation_and_index_restore_failure_are_both_reported(git_repo: Path,) -> None:


    _seed(git_repo, {"a.py": b"A = 1\n"})
    footprint = AuthorizedFixFootprint.build(git_repo, {"a.py"}, [])
    lock = git_repo / ".git" / "index.lock"
    with pytest.raises(BaseExceptionGroup) as raised:
        async with _isolated_fix_fanout(
            work_context(git_repo, run_id="test-run"), footprint,
            file_scope_issues=False, auth=git_ops.INHERIT_GITHUB_AUTH,
        ):
            lock.write_bytes(b"held")
            raise asyncio.CancelledError("primary cancellation")
    lock.unlink(missing_ok=True)
    assert len(raised.value.exceptions) == 2
    assert isinstance(raised.value.exceptions[0], asyncio.CancelledError)
    assert isinstance(raised.value.exceptions[1], GitError)

def test_authorized_symlink_substitution_fails_before_read_restore_or_stage(git_repo: Path, tmp_path: Path) -> None:
    """A parent or leaf replaced by a symlink after authorization is never read, restored, or staged."""
    _seed(git_repo, {"allowed.txt": b"base\n"})
    footprint = AuthorizedFixFootprint.build(git_repo, {"nested/allowed.txt", "allowed.txt"}, [])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "allowed.txt").write_bytes(b"outside")
    (git_repo / "nested").symlink_to(outside, target_is_directory=True)
    (git_repo / "allowed.txt").unlink()
    (git_repo / "allowed.txt").symlink_to("base.txt")

    assert "nested/allowed.txt" in footprint.run_allowed_paths
    for paths in ({"nested/allowed.txt"}, {"allowed.txt"}):
        with pytest.raises(GitError, match="not confined"):
            git_ops.snapshot_worktree_paths(git_repo, frozenset(paths))
        with pytest.raises(GitError, match="not confined"):
            git_ops.restore_worktree_paths_from_ref(git_repo, "HEAD", frozenset(paths))
        with pytest.raises(GitError, match="not confined"):
            git_ops.stage_paths(git_repo, [Path(next(iter(paths)))])
    assert (outside / "allowed.txt").read_bytes() == b"outside"

def _add_gitlink(git_repo: Path) -> Path:
    """Seed a committed ``dependency`` gitlink and return the nested repo."""
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    return nested


def _gitlink_rollback_snapshot(repo: Path, path: str = "dependency") -> WorktreeRollbackSnapshot:
    return WorktreeRollbackSnapshot(
        ref="HEAD", index=git_ops.snapshot_index(repo), path_states=git_ops.snapshot_worktree_paths(repo, [path]),
        untracked=git_ops.snapshot_untracked_paths(repo),
    )

def test_dirty_gitlink_cannot_claim_evidence_or_be_rolled_back(git_repo: Path,) -> None:
    """A dirty nested tree neither snapshots as commit-only evidence nor is mutated by group rollback."""
    nested = _add_gitlink(git_repo)
    snapshot = _gitlink_rollback_snapshot(git_repo)
    original_head = _git(nested, "rev-parse", "HEAD")
    (nested / "source.py").write_bytes(b"owner dirty bytes\n")
    with pytest.raises(GitError, match="dirty gitlink"):
        git_ops.snapshot_worktree_paths(git_repo, ["dependency"])
    write_and_stage(git_repo, "agent-staged.txt", b"agent index mutation\n")
    with pytest.raises(GitError, match="dirty gitlink"):
        git_ops.restore_group_from_snapshot(git_repo, snapshot, ["dependency"])

    assert _git(nested, "rev-parse", "HEAD") == original_head
    assert (nested / "source.py").read_bytes() == b"owner dirty bytes\n"
    assert git_ops.snapshot_index(git_repo) == snapshot.index
    assert (git_repo / "agent-staged.txt").read_bytes() == b"agent index mutation\n"

def test_scope_guard_restores_pre_run_non_index_gitlink_checkout(git_repo: Path) -> None:
    nested = _add_gitlink(git_repo)
    indexed = _git(nested, "rev-parse", "HEAD")
    _seed(nested, {"source.py": b"value = 2\n"})
    protected = _git(nested, "rev-parse", "HEAD")
    assert protected != indexed
    gitlinks = git_ops.snapshot_worktree_gitlinks(git_repo)
    footprint = AuthorizedFixFootprint.build(git_repo, set(), [])
    _git(nested, "checkout", "--detach", indexed)
    result = _enforce_fix(
        git_repo, footprint, phase="terminal", preexisting_untracked={}, preexisting_gitlinks=gitlinks,
    )

    assert result.mutated
    assert _git(nested, "rev-parse", "HEAD") == protected
    assert any(
        event.action == "restore"
        and event.path == "dependency"
        and "gitlink" in event.reason
        for event in footprint.events
    )

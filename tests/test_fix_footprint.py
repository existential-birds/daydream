"""Focused contracts for the authorized fix footprint (issue #1135)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from daydream import git_ops
from daydream.deep.scope_issues import enforce_authorized_fix_footprint
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.git_ops import GitError, WorktreeRollbackSnapshot
from daydream.repository_paths import (
    InvalidRepositoryFilePath,
    canonicalize_repository_file_path,
    git_observed_path_is_confined,
)
from daydream.workspace import WorkContext
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo


def _work(repo: Path) -> WorkContext:
    head = _git(repo, "rev-parse", "HEAD")
    return WorkContext(
        repo=repo,
        source=repo,
        base_branch="main",
        base_sha=head,
        head_branch="main",
        head_sha=head,
        is_ephemeral=False,
        run_id="test-run",
    )


def _seed(repo: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(repo, "add", ".")
    _commit(repo, "seed footprint files")


def _install_scope_boundary_shims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_diff_path: str | None = None,
    watched_paths: tuple[str, ...] = (),
    fail_issue_create: bool = False,
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
    print("[]")
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
        {
            "id": 1,
            "item_uid": "item:1",
            "file": "src/a.py",
            "related_files": ["./src/shared.py", "tests/test_a.py"],
        },
        {
            "id": 2,
            "item_uid": "item:2",
            "file": "src/b.py",
            "related_files": ["src/shared.py"],
        },
        {
            "id": 3,
            "item_uid": "item:3",
            "file": "src/c.py",
            "related_files": None,
        },
    ]


@pytest.mark.parametrize(
    "value",
    [
        "",
        7,
        None,
        "/tmp/outside",
        "../outside",
        "src/../outside",
        "a\n.py",
        "a`touch nope`",
        "bad-\udcff.py",
    ],
)
def test_model_path_normalization_rejects_malformed_and_escaping_values(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(InvalidRepositoryFilePath, match="invalid repository file path"):
        canonicalize_repository_file_path(tmp_path, value)


def test_model_path_normalization_removes_dot_prefix_and_rejects_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()

    assert canonicalize_repository_file_path(repo, "./src/a.py") == "src/a.py"

    (repo / "src").symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidRepositoryFilePath, match="invalid repository file path"):
        canonicalize_repository_file_path(repo, "src/a.py")


def test_git_observed_confinement_accepts_non_model_names_but_rejects_escapes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    assert git_observed_path_is_confined(repo, "odd\n$name[1].txt")
    assert not git_observed_path_is_confined(repo, "../outside")
    assert not git_observed_path_is_confined(repo, "/outside")


def test_footprint_uses_item_uids_and_exact_item_group_and_run_unions(tmp_path: Path) -> None:
    footprint = AuthorizedFixFootprint.build(tmp_path, {"README.md", "./src/a.py"}, _items())

    assert footprint.item_paths("item:1") == frozenset(
        {"src/a.py", "src/shared.py", "tests/test_a.py"}
    )
    assert footprint.item_paths("item:2") == frozenset({"src/b.py", "src/shared.py"})
    assert footprint.item_paths("item:3") == frozenset({"src/c.py"})
    assert footprint.group_paths([_items()[0], _items()[1]]) == frozenset(
        {"src/a.py", "src/b.py", "src/shared.py", "tests/test_a.py"}
    )
    assert footprint.run_allowed_paths == frozenset(
        {"README.md", "src/a.py", "src/b.py", "src/c.py", "src/shared.py", "tests/test_a.py"}
    )

    actions = [(event.origin, event.path, event.item_uid) for event in footprint.events]
    assert ("reviewed", "README.md", None) in actions
    assert ("primary", "src/a.py", "item:1") in actions
    assert ("related", "tests/test_a.py", "item:1") in actions


def test_accepted_retarget_is_audited_without_widening_policy(tmp_path: Path) -> None:
    footprint = AuthorizedFixFootprint.build(tmp_path, {"src/a.py"}, _items())
    revision = footprint.policy_revision
    run_scope = footprint.run_allowed_paths
    item_scope = footprint.item_paths("item:1")
    event_count = len(footprint.events)

    accepted = footprint.accept_retarget(
        tmp_path,
        "item:1",
        "tests/test_a.py",
        phase="fix",
        round_number=2,
    )

    assert accepted == "tests/test_a.py"
    assert footprint.policy_revision == revision
    assert footprint.run_allowed_paths == run_scope
    assert footprint.item_paths("item:1") == item_scope
    assert len(footprint.events) == event_count + 1
    event = footprint.events[-1]
    assert event.action == "authorize"
    assert event.origin == "retarget"
    assert event.path_kind == "model"
    assert event.path == "tests/test_a.py"
    assert event.item_uid == "item:1"
    assert event.phase == "fix"
    assert event.round_number == 2


@pytest.mark.parametrize("reviewed_path", ["odd\nname.py", "literal-$(value)[1].py", "native-\udcff.py"])
def test_footprint_treats_reviewed_git_names_separately_from_model_paths(
    git_repo: Path, reviewed_path: str,
) -> None:
    import errno

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
    assert footprint.accept_retarget(
        git_repo, "item:normal", reviewed_path, phase="fix", round_number=1,
    ) is None
    assert json.loads(json.dumps(footprint.audit_payload("run")))["run_allowed_paths"]


@pytest.mark.parametrize("reviewed_path", ["", ".", "/outside", "../outside", "src/../outside", "src//a.py"])
def test_footprint_rejects_unconfined_reviewed_git_paths(tmp_path: Path, reviewed_path: str) -> None:
    with pytest.raises(InvalidRepositoryFilePath, match="invalid reviewed repository path"):
        AuthorizedFixFootprint.build(tmp_path, {reviewed_path}, [])


@pytest.mark.parametrize(
    "related",
    ["src/b.py", ["src/b.py", 7], ["../escape.py"], ["src/../escape.py"]],
)
def test_footprint_rejects_malformed_related_paths(tmp_path: Path, related: object) -> None:
    item = {"id": 1, "item_uid": "item:1", "file": "src/a.py", "related_files": related}
    with pytest.raises(InvalidRepositoryFilePath, match="invalid repository file path"):
        AuthorizedFixFootprint.build(tmp_path, set(), [item])


def test_generated_authorization_is_idempotent_and_retarget_is_item_bounded(tmp_path: Path) -> None:
    footprint = AuthorizedFixFootprint.build(tmp_path, set(), _items())
    initial_revision = footprint.policy_revision

    footprint.authorize_new_generated(
        tmp_path,
        "generated/schema.py",
        phase="generated-guard",
        round_number=1,
        reason="approved migration output",
    )
    footprint.authorize_new_generated(
        tmp_path,
        "./generated/schema.py",
        phase="generated-guard",
        round_number=2,
        reason="same generated output observed again",
    )
    assert footprint.policy_revision == initial_revision + 1
    assert sum(event.action == "approve_generated" for event in footprint.events) == 1

    assert footprint.accept_retarget(
        tmp_path, "item:1", "./tests/test_a.py", phase="verify", round_number=2
    ) == "tests/test_a.py"
    assert footprint.accept_retarget(
        tmp_path, "item:1", "src/b.py", phase="verify", round_number=2
    ) is None
    assert footprint.item_paths("item:1") == frozenset(
        {"src/a.py", "src/shared.py", "tests/test_a.py"}
    )
    rejection = footprint.events[-1]
    assert (rejection.action, rejection.path, rejection.item_uid) == (
        "rejected_retarget",
        "src/b.py",
        "item:1",
    )


def test_audit_payload_is_session_bound_monotonic_and_json_escapes_git_paths(tmp_path: Path) -> None:
    footprint = AuthorizedFixFootprint.build(tmp_path, {"src/a.py"}, _items()[:1])
    footprint.record_git_event(
        action="remove",
        path="odd\n$name.txt",
        origin="guard",
        phase="scope",
        round_number=1,
        reason="new untracked file",
    )

    payload = footprint.audit_payload("session-123", tree_key="abc")
    assert payload["session_id"] == "session-123"
    assert payload["tree_key"] == "abc"
    sequences = [event["sequence"] for event in payload["events"]]
    assert sequences == list(range(1, len(sequences) + 1))
    assert "odd\\n$name.txt" in json.dumps(payload)


def test_changed_paths_z_preserves_newline_shell_metacharacters_and_surrogate_bytes(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weird = "odd\n$(not-a-command).txt"
    (git_repo / weird).write_bytes(b"odd")
    paths = git_ops.changed_paths_z(git_repo, "HEAD")
    assert weird in paths

    raw_name = b"invalid-\xff.txt"
    monkeypatch.setattr(
        git_ops,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=raw_name + b"\0", stderr=b""
        ),
    )
    paths = git_ops.changed_paths_z(git_repo, "HEAD", include_untracked=False)
    assert os.fsdecode(raw_name) in paths


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

    result = enforce_authorized_fix_footprint(
        _work(git_repo),
        "HEAD",
        footprint,
        preexisting_untracked=baseline,
        phase="post-fix",
        round_number=1,
    )

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


def test_runtime_artifacts_do_not_enter_fix_scope_or_invalidate_content_evidence(git_repo: Path) -> None:
    _seed(git_repo, {"allowed.txt": b"base\n"})
    artifacts = git_repo / ".daydream" / "deep"
    artifacts.mkdir(parents=True)
    audit = artifacts / "fix-footprint.json"
    audit.write_text('{"version": 1}')
    report = git_repo / ".review-output.md"
    report.write_text("old report")
    scratch = git_repo / "scratch.txt"
    scratch.write_bytes(b"user scratch")
    protected = git_ops.snapshot_untracked_paths(git_repo, include_runtime_artifacts=False)
    assert set(protected) == {"scratch.txt"}
    assert ".daydream/deep/fix-footprint.json" in git_ops.changed_paths_z(git_repo, "HEAD")

    before = git_ops.tree_key(git_ops.snapshot_worktree_delta(
        git_repo, "HEAD", preexisting_untracked=protected,
    ))
    audit.write_text('{"version": 2}')
    report.write_text("new report")
    assert git_ops.tree_key(git_ops.snapshot_worktree_delta(
        git_repo, "HEAD", preexisting_untracked=protected,
    )) == before
    scratch.write_bytes(b"changed scratch")
    assert git_ops.tree_key(git_ops.snapshot_worktree_delta(
        git_repo, "HEAD", preexisting_untracked=protected,
    )) != before
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.txt"}, [])
    result = enforce_authorized_fix_footprint(
        _work(git_repo), "HEAD", footprint, preexisting_untracked=protected,
        phase="post-test", round_number=1,
    )
    assert result.mutated
    assert not result.retained_paths
    assert scratch.read_bytes() == b"user scratch"
    assert audit.read_text() == '{"version": 2}'
    assert report.read_text() == "new report"
    assert {event.path for event in footprint.events if event.action == "restore"} == {"scratch.txt"}


def test_runtime_exclusion_keeps_tracked_artifacts_and_similar_user_paths_visible(git_repo: Path) -> None:
    _seed(git_repo, {".daydream/tracked.txt": b"tracked\n"})
    (git_repo / ".daydream/tracked.txt").write_bytes(b"changed\n")
    for name in (".daydream.toml", ".review-output.md.user", "src/.daydream/user.txt"):
        path = git_repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("user file")
    expected = {".daydream/tracked.txt", ".daydream.toml", ".review-output.md.user", "src/.daydream/user.txt"}
    assert set(git_ops.changed_paths_z(git_repo, "HEAD", include_runtime_artifacts=False)) == expected
    protected = git_ops.snapshot_untracked_paths(git_repo, include_runtime_artifacts=False)
    assert set(protected) == expected - {".daydream/tracked.txt"}


def test_worktree_restore_leaves_the_index_tree_unchanged(git_repo: Path) -> None:
    _seed(git_repo, {"tracked.txt": b"base\n"})
    path = git_repo / "tracked.txt"
    path.write_bytes(b"staged\n")
    _git(git_repo, "add", "tracked.txt")
    before = git_ops.snapshot_index(git_repo)
    path.write_bytes(b"worktree-only\n")

    git_ops.restore_worktree_paths_from_ref(git_repo, "HEAD", ["tracked.txt"])

    assert path.read_bytes() == b"base\n"
    assert git_ops.snapshot_index(git_repo) == before


def test_group_rollback_restores_all_paths_untracked_and_supplied_index(git_repo: Path) -> None:
    _seed(git_repo, {"a.txt": b"base a\n", "b.txt": b"base b\n"})
    (git_repo / "a.txt").write_bytes(b"round baseline a\n")
    (git_repo / "b.txt").write_bytes(b"round baseline b\n")
    (git_repo / "round-scratch").write_bytes(b"scratch baseline")
    (git_repo / "round-scratch").chmod(0o600)
    round_ref = git_ops.stash_create(git_repo) or "HEAD"
    snapshot = WorktreeRollbackSnapshot(
        ref=round_ref,
        index=git_ops.snapshot_index(git_repo),
        path_states=git_ops.snapshot_worktree_paths(git_repo, ["a.txt", "b.txt"]),
        untracked=git_ops.snapshot_untracked_paths(git_repo),
    )

    (git_repo / "a.txt").write_bytes(b"failed a\n")
    (git_repo / "b.txt").unlink()
    (git_repo / "new.txt").write_bytes(b"failed new")
    (git_repo / "round-scratch").write_bytes(b"failed scratch")
    _git(git_repo, "add", "a.txt", "b.txt")

    git_ops.restore_group_from_snapshot(
        git_repo, snapshot, ["a.txt", "b.txt", "new.txt", "round-scratch"]
    )

    assert (git_repo / "a.txt").read_bytes() == b"round baseline a\n"
    assert (git_repo / "b.txt").read_bytes() == b"round baseline b\n"
    assert not (git_repo / "new.txt").exists()
    assert (git_repo / "round-scratch").read_bytes() == b"scratch baseline"
    assert stat.S_IMODE((git_repo / "round-scratch").stat().st_mode) == 0o600
    assert git_ops.snapshot_index(git_repo) == snapshot.index


def test_group_worktree_rollback_preserves_sibling_index_entry(git_repo: Path) -> None:
    _seed(git_repo, {"a.py": b"A = 1\n", "b.py": b"B = 1\n"})
    snapshot = WorktreeRollbackSnapshot(
        ref="HEAD",
        index=git_ops.snapshot_index(git_repo),
        path_states=git_ops.snapshot_worktree_paths(git_repo, ["a.py"]),
        untracked={},
    )
    (git_repo / "a.py").write_bytes(b"A = partial\n")
    (git_repo / "b.py").write_bytes(b"B = sibling\n")
    _git(git_repo, "add", "b.py")
    sibling_index = git_ops.snapshot_index(git_repo)

    git_ops.restore_group_worktree_from_snapshot(git_repo, snapshot, ["a.py"])

    assert (git_repo / "a.py").read_bytes() == b"A = 1\n"
    assert (git_repo / "b.py").read_bytes() == b"B = sibling\n"
    assert git_ops.snapshot_index(git_repo) == sibling_index


def test_scope_issue_diff_failure_still_restores_and_audits_without_sensitive_warning(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed(git_repo, {"allowed.py": b"allowed\n", "outside.py": b"owner\n"})
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.py"}, [])
    index_before = git_ops.snapshot_index(git_repo)
    (git_repo / "outside.py").write_bytes(b"unauthorized\n")
    gh_record = _install_scope_boundary_shims(
        tmp_path,
        monkeypatch,
        fail_diff_path="outside.py",
        watched_paths=("outside.py",),
    )

    result = enforce_authorized_fix_footprint(
        _work(git_repo),
        "HEAD",
        footprint,
        preexisting_untracked={},
        phase="fix",
        round_number=1,
        file_scope_issues=True,
    )

    assert result.mutated is True
    assert (git_repo / "outside.py").read_bytes() == b"owner\n"
    assert git_ops.snapshot_index(git_repo) == index_before
    assert not gh_record.exists()
    assert [(event.action, event.path) for event in footprint.events][-1] == (
        "restore",
        "outside.py",
    )
    warning_output = capsys.readouterr().out
    assert "continuing to" in warning_output
    assert "restoration without filing" in warning_output
    assert "SECRET_TOKEN" not in warning_output
    assert "outside.py" not in warning_output


def test_scope_issue_diff_failure_skips_only_that_filing_after_restoring_all_residuals(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(
        git_repo,
        {"allowed.py": b"allowed\n", "outside-a.py": b"owner a\n", "outside-b.py": b"owner b\n"},
    )
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.py"}, [])
    (git_repo / "outside-a.py").write_bytes(b"unauthorized a\n")
    (git_repo / "outside-b.py").write_bytes(b"unauthorized b\n")
    gh_record = _install_scope_boundary_shims(
        tmp_path,
        monkeypatch,
        fail_diff_path="outside-a.py",
        watched_paths=("outside-a.py", "outside-b.py"),
    )

    result = enforce_authorized_fix_footprint(
        _work(git_repo),
        "HEAD",
        footprint,
        preexisting_untracked={},
        phase="fix",
        round_number=1,
        file_scope_issues=True,
    )

    assert result.mutated is True
    assert (git_repo / "outside-a.py").read_bytes() == b"owner a\n"
    assert (git_repo / "outside-b.py").read_bytes() == b"owner b\n"
    calls = [json.loads(line) for line in gh_record.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["watched"] == {
        "outside-a.py": b"owner a\n".hex(),
        "outside-b.py": b"owner b\n".hex(),
    }
    assert "outside-b.py" in calls[0]["body"]
    assert "+unauthorized b" in calls[0]["body"]
    assert "outside-a.py" not in calls[0]["body"]


def test_scope_issue_filing_failure_occurs_after_verified_restore_and_is_best_effort(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(git_repo, {"allowed.py": b"allowed\n", "outside.py": b"owner\n"})
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.py"}, [])
    (git_repo / "outside.py").write_bytes(b"unauthorized\n")
    gh_record = _install_scope_boundary_shims(
        tmp_path,
        monkeypatch,
        watched_paths=("outside.py",),
        fail_issue_create=True,
    )

    result = enforce_authorized_fix_footprint(
        _work(git_repo),
        "HEAD",
        footprint,
        preexisting_untracked={},
        phase="fix",
        round_number=1,
        file_scope_issues=True,
    )

    assert result.mutated is True
    assert (git_repo / "outside.py").read_bytes() == b"owner\n"
    calls = [json.loads(line) for line in gh_record.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["watched"] == {"outside.py": b"owner\n".hex()}
    assert "+unauthorized" in calls[0]["body"]


@pytest.mark.asyncio
async def test_parallel_group_fallback_never_restores_index_while_sibling_is_live(
    git_repo: Path,
) -> None:
    """Real fix dispatch uses a join barrier before the one complete index restore."""
    import anyio

    from daydream.backends import AgentEvent, ResultEvent
    from daydream.fix_footprint import AuthorizedFixFootprint
    from daydream.phases import phase_fix_parallel

    _seed(git_repo, {"a.py": b"A = 1\n", "b.py": b"B = 1\n"})
    round_index = git_ops.snapshot_index(git_repo)
    snapshot = WorktreeRollbackSnapshot(
        ref="HEAD",
        index=round_index,
        path_states=git_ops.snapshot_worktree_paths(git_repo, ["a.py", "b.py"]),
        untracked={},
    )
    items = [
        {"id": 1, "item_uid": "item:a1", "file": "a.py", "description": "a one"},
        {"id": 2, "item_uid": "item:a2", "file": "a.py", "description": "a two"},
        {"id": 3, "item_uid": "item:b1", "file": "b.py", "description": "b one"},
        {"id": 4, "item_uid": "item:b2", "file": "b.py", "description": "b two"},
    ]
    footprint = AuthorizedFixFootprint.build(git_repo, {"a.py", "b.py"}, items)
    b_staged = anyio.Event()
    a_fallback_started = anyio.Event()
    b_observed_sibling_index = anyio.Event()

    class BarrierBackend:
        model = "barrier-backend"
        fanout_concurrency = 2
        retry_attempts = 0

        def __init__(self) -> None:
            self.a_fallback_calls = 0
            self.prompts: list[str] = []
            self.b_cached_while_live = ""

        async def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: object = None,
            continuation: object = None,
            agents: object = None,
            max_turns: int | None = None,
            read_only: bool = False,
            persist_session: bool = True,
        ) -> AsyncGenerator[AgentEvent, None]:
            del output_schema, continuation, agents, max_turns, read_only, persist_session
            self.prompts.append(prompt)
            if prompt.startswith("Fix these 2 issues") and "b one" in prompt:
                (cwd / "b.py").write_bytes(b"B = sibling\n")
                _git(cwd, "add", "b.py")
                b_staged.set()
                await a_fallback_started.wait()
                self.b_cached_while_live = _git(cwd, "diff", "--cached", "--name-only")
                b_observed_sibling_index.set()
            elif prompt.startswith("Fix these 2 issues") and "a one" in prompt:
                await b_staged.wait()
                (cwd / "a.py").write_bytes(b"A = partial\n")
                raise RuntimeError("force batch fallback")
            elif prompt.startswith("Fix this issue:") and ("a one" in prompt or "a two" in prompt):
                self.a_fallback_calls += 1
                (cwd / "a.py").write_bytes(b"A = fixed\n")
                a_fallback_started.set()
                await b_observed_sibling_index.wait()
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self) -> None:
            return None

    backend = BarrierBackend()
    failures = await phase_fix_parallel(
        backend,
        _work(git_repo),
        items,
        footprint=footprint,
        round_snapshot=snapshot,
        limiter_size=2,
    )

    assert failures == {}
    assert backend.a_fallback_calls == 2, backend.prompts
    assert b_observed_sibling_index.is_set()
    assert backend.b_cached_while_live == "b.py"
    assert (git_repo / "a.py").read_bytes() == b"A = fixed\n"
    assert (git_repo / "b.py").read_bytes() == b"B = sibling\n"
    assert git_ops.snapshot_index(git_repo) == round_index


@pytest.mark.asyncio
async def test_parallel_fix_cancellation_closes_backend_before_restoring_round_index(
    git_repo: Path,
) -> None:
    import anyio

    from daydream.backends import AgentEvent, ResultEvent
    from daydream.fix_footprint import AuthorizedFixFootprint
    from daydream.phases import phase_fix_parallel

    _seed(git_repo, {"a.py": b"A = 1\n"})
    round_index = git_ops.snapshot_index(git_repo)
    snapshot = WorktreeRollbackSnapshot(
        ref="HEAD",
        index=round_index,
        path_states=git_ops.snapshot_worktree_paths(git_repo, ["a.py"]),
        untracked={},
    )
    item = {"id": 1, "item_uid": "item:a", "file": "a.py", "description": "a"}
    footprint = AuthorizedFixFootprint.build(git_repo, {"a.py"}, [item])
    staged = anyio.Event()
    stream_closed = anyio.Event()
    never = anyio.Event()

    class CancelBackend:
        model = "cancel-backend"
        fanout_concurrency = 1
        retry_attempts = 0

        async def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: object = None,
            continuation: object = None,
            agents: object = None,
            max_turns: int | None = None,
            read_only: bool = False,
            persist_session: bool = True,
        ) -> AsyncGenerator[AgentEvent, None]:
            del prompt, output_schema, continuation, agents, max_turns, read_only, persist_session
            try:
                (cwd / "a.py").write_bytes(b"A = staged by live fixer\n")
                _git(cwd, "add", "a.py")
                staged.set()
                await never.wait()
                yield ResultEvent(structured_output=None, continuation=None)
            finally:
                stream_closed.set()

        async def cancel(self) -> None:
            return None

    async def _run_phase() -> None:
        await phase_fix_parallel(
            CancelBackend(),
            _work(git_repo),
            [item],
            footprint=footprint,
            round_snapshot=snapshot,
            limiter_size=1,
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_run_phase)
        await staged.wait()
        task_group.cancel_scope.cancel()

    assert stream_closed.is_set()
    assert git_ops.snapshot_index(git_repo) == round_index


@pytest.mark.asyncio
async def test_fanout_cancellation_and_index_restore_failure_are_both_reported(
    git_repo: Path,
) -> None:
    import asyncio

    from daydream.phases import _restore_round_index_after_fanout

    _seed(git_repo, {"a.py": b"A = 1\n"})
    index = git_ops.snapshot_index(git_repo)
    lock = git_repo / ".git" / "index.lock"

    with pytest.raises(BaseExceptionGroup) as raised:
        async with _restore_round_index_after_fanout(git_repo, index):
            lock.write_bytes(b"held")
            raise asyncio.CancelledError("primary cancellation")

    lock.unlink(missing_ok=True)
    assert len(raised.value.exceptions) == 2
    assert isinstance(raised.value.exceptions[0], asyncio.CancelledError)
    assert isinstance(raised.value.exceptions[1], GitError)


def test_authorized_parent_symlink_substitution_fails_before_read_restore_or_stage(
    git_repo: Path, tmp_path: Path
) -> None:
    footprint = AuthorizedFixFootprint.build(git_repo, {"nested/allowed.txt"}, [])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "allowed.txt").write_bytes(b"outside")
    (git_repo / "nested").symlink_to(outside, target_is_directory=True)

    assert "nested/allowed.txt" in footprint.run_allowed_paths
    with pytest.raises(GitError, match="not confined"):
        git_ops.snapshot_worktree_paths(git_repo, footprint.run_allowed_paths)
    with pytest.raises(GitError, match="not confined"):
        git_ops.restore_worktree_paths_from_ref(git_repo, "HEAD", footprint.run_allowed_paths)
    with pytest.raises(GitError, match="not confined"):
        git_ops.stage_paths(git_repo, [Path("nested/allowed.txt")])
    assert (outside / "allowed.txt").read_bytes() == b"outside"


def test_authorized_leaf_symlink_substitution_fails_before_read_restore_or_stage(git_repo: Path) -> None:
    _seed(git_repo, {"allowed.txt": b"base\n"})
    footprint = AuthorizedFixFootprint.build(git_repo, {"allowed.txt"}, [])
    (git_repo / "allowed.txt").unlink()
    (git_repo / "allowed.txt").symlink_to("base.txt")

    with pytest.raises(GitError, match="not confined"):
        git_ops.snapshot_worktree_paths(git_repo, footprint.run_allowed_paths)
    with pytest.raises(GitError, match="not confined"):
        git_ops.restore_worktree_paths_from_ref(git_repo, "HEAD", footprint.run_allowed_paths)
    with pytest.raises(GitError, match="not confined"):
        git_ops.stage_paths(git_repo, [Path("allowed.txt")])


def test_tree_key_is_binary_safe_mode_type_delete_new_and_order_deterministic(git_repo: Path) -> None:
    _seed(git_repo, {"binary.bin": b"\x00before\xff", "delete.txt": b"delete me", "mode.sh": b"#!/bin/sh\n"})
    baseline = git_ops.snapshot_worktree_paths(git_repo, ["binary.bin", "delete.txt", "mode.sh", "new.bin"])

    (git_repo / "binary.bin").write_bytes(b"\x00after\xff")
    (git_repo / "delete.txt").unlink()
    (git_repo / "mode.sh").chmod(0o755)
    (git_repo / "new.bin").write_bytes(b"\x00new\xfe")
    changed = git_ops.snapshot_worktree_paths(git_repo, ["new.bin", "mode.sh", "delete.txt", "binary.bin"])

    assert git_ops.tree_key(baseline) != git_ops.tree_key(changed)
    assert git_ops.tree_key(changed) == git_ops.tree_key(reversed(changed))
    states = {state.path: state for state in changed}
    assert states["delete.txt"].state == "missing"
    assert states["new.bin"].state == "regular"
    assert states["mode.sh"].mode == 0o100755

    footprint = AuthorizedFixFootprint.build(git_repo, {"binary.bin"}, [])
    key = git_ops.tree_key(changed)
    footprint.authorize_new_generated(
        git_repo,
        "generated.py",
        phase="guard",
        round_number=1,
        reason="approved",
    )
    assert git_ops.tree_key(changed) == key


def test_gitlink_evidence_uses_checked_out_head_not_staged_commit(git_repo: Path) -> None:
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    original_head = _git(nested, "rev-parse", "HEAD")
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    baseline = git_ops.snapshot_worktree_paths(git_repo, ["dependency"])

    _seed(nested, {"source.py": b"value = 2\n"})
    checked_out_head = _git(nested, "rev-parse", "HEAD")
    current = git_ops.snapshot_worktree_paths(git_repo, ["dependency"])

    assert baseline[0].digest == original_head
    assert git_ops.snapshot_index_paths(git_repo, ["dependency"])[0].digest == original_head
    assert current[0].digest == checked_out_head
    assert git_ops.tree_key(current) != git_ops.tree_key(baseline)


@pytest.mark.parametrize("dirty_path", ["source.py", "untracked.py"])
def test_dirty_gitlink_cannot_claim_commit_only_test_evidence(
    git_repo: Path, dirty_path: str,
) -> None:
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    (nested / dirty_path).write_bytes(b"value = 2\n")

    with pytest.raises(GitError, match="dirty gitlink"):
        git_ops.snapshot_worktree_paths(git_repo, ["dependency"])


def test_uninitialized_gitlink_cannot_capture_parent_repository_head(git_repo: Path) -> None:
    nested = git_repo / "dependency"
    nested.mkdir()
    head = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "update-index", "--add", "--cacheinfo", f"160000,{head},dependency")
    _commit(git_repo, "record uninitialized dependency")

    with pytest.raises(GitError, match="gitlink working tree is unavailable"):
        git_ops.snapshot_worktree_paths(git_repo, ["dependency"])


def _gitlink_rollback_snapshot(
    repo: Path, path: str = "dependency"
) -> WorktreeRollbackSnapshot:
    return WorktreeRollbackSnapshot(
        ref="HEAD",
        index=git_ops.snapshot_index(repo),
        path_states=git_ops.snapshot_worktree_paths(repo, [path]),
        untracked=git_ops.snapshot_untracked_paths(repo),
    )


def test_gitlink_group_rollback_restores_captured_nested_oid(git_repo: Path) -> None:
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    captured = _git(nested, "rev-parse", "HEAD")
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    snapshot = _gitlink_rollback_snapshot(git_repo)
    _seed(nested, {"source.py": b"value = 2\n"})
    assert _git(nested, "rev-parse", "HEAD") != captured

    git_ops.restore_group_from_snapshot(git_repo, snapshot, ["dependency"])

    assert _git(nested, "rev-parse", "HEAD") == captured
    assert _git(nested, "status", "--porcelain=v1", "--untracked-files=all") == ""


def test_gitlink_group_rollback_preserves_initial_non_index_checkout(git_repo: Path) -> None:
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    indexed = _git(nested, "rev-parse", "HEAD")
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    _seed(nested, {"source.py": b"value = 2\n"})
    captured = _git(nested, "rev-parse", "HEAD")
    assert captured != indexed
    snapshot = _gitlink_rollback_snapshot(git_repo)
    _git(nested, "checkout", "--detach", indexed)

    git_ops.restore_group_from_snapshot(git_repo, snapshot, ["dependency"])

    assert _git(nested, "rev-parse", "HEAD") == captured
    assert _git(git_repo, "status", "--porcelain=v1", "--untracked-files=all") == "M dependency"


def test_gitlink_group_rollback_refuses_dirty_nested_tree_without_mutating_it(
    git_repo: Path,
) -> None:
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    snapshot = _gitlink_rollback_snapshot(git_repo)
    original_head = _git(nested, "rev-parse", "HEAD")
    (nested / "source.py").write_bytes(b"owner dirty bytes\n")
    (git_repo / "agent-staged.txt").write_bytes(b"agent index mutation\n")
    _git(git_repo, "add", "agent-staged.txt")

    with pytest.raises(GitError, match="dirty gitlink"):
        git_ops.restore_group_from_snapshot(git_repo, snapshot, ["dependency"])

    assert _git(nested, "rev-parse", "HEAD") == original_head
    assert (nested / "source.py").read_bytes() == b"owner dirty bytes\n"
    assert git_ops.snapshot_index(git_repo) == snapshot.index
    assert (git_repo / "agent-staged.txt").read_bytes() == b"agent index mutation\n"


def test_scope_guard_restores_pre_run_non_index_gitlink_checkout(git_repo: Path) -> None:
    nested = git_repo / "dependency"
    init_repo(nested)
    _seed(nested, {"source.py": b"value = 1\n"})
    indexed = _git(nested, "rev-parse", "HEAD")
    _git(git_repo, "add", "dependency")
    _commit(git_repo, "record dependency")
    _seed(nested, {"source.py": b"value = 2\n"})
    protected = _git(nested, "rev-parse", "HEAD")
    assert protected != indexed
    gitlinks = git_ops.snapshot_worktree_gitlinks(git_repo)
    footprint = AuthorizedFixFootprint.build(git_repo, set(), [])
    _git(nested, "checkout", "--detach", indexed)

    result = enforce_authorized_fix_footprint(
        _work(git_repo),
        "HEAD",
        footprint,
        preexisting_untracked={},
        preexisting_gitlinks=gitlinks,
        phase="terminal",
        round_number=1,
    )

    assert result.mutated
    assert _git(nested, "rev-parse", "HEAD") == protected
    assert any(
        event.action == "restore"
        and event.path == "dependency"
        and "gitlink" in event.reason
        for event in footprint.events
    )


def test_strict_recommended_patch_contains_binary_change_and_commit_staged_does_not_restage(
    git_repo: Path
) -> None:
    _seed(git_repo, {"binary.bin": b"\x00before\xff", "keep.txt": b"base\n"})
    (git_repo / "binary.bin").write_bytes(b"\x00after\xfe")
    (git_repo / "new.bin").write_bytes(b"\x00new\xfd")
    patch = git_ops.build_recommended_patch_strict(
        git_repo, "HEAD", ["new.bin", "binary.bin"]
    )
    assert b"binary.bin" in patch
    assert b"new.bin" in patch
    assert b"GIT binary patch" in patch

    git_ops.stage_paths(git_repo, [Path("binary.bin")])
    (git_repo / "binary.bin").write_bytes(b"changed after stage")
    git_ops.commit_staged(git_repo, "commit validated index")
    assert git_ops.show(git_repo, "HEAD", "binary.bin") == b"\x00after\xfe"
    assert (git_repo / "binary.bin").read_bytes() == b"changed after stage"

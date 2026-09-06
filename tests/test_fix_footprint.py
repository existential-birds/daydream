"""Focused contracts for the authorized fix footprint (issue #1135)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
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

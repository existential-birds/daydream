"""Real-run shell writes must obey each fixer's exact assigned footprint."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from daydream import git_ops
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.fix_isolation import FixIsolationRound
from daydream.workspace import WorkContext
from tests.harness.git_helpers import git
from tests.test_deep_orchestrator import _merge_item


@pytest.mark.parametrize("index_kind", ["staged", ])
def test_round_recovers_parent_types_modes_ignored_owner_files_and_exact_index(
    multi_stack_target: Path, tmp_path: Path, index_kind: str,
) -> None:
    """Supplement the runner cases with exact index flags and path types."""
    target = multi_stack_target
    (target / "pkg").mkdir()
    (target / "pkg/owner.py").write_text("owner source\n")
    (target / ".gitignore").write_text("owner.env\nhidden.py\n")
    git(target, "add", "pkg/owner.py", ".gitignore")
    git(target, "commit", "-m", "test: owner sources")
    (target / "owner.env").write_bytes(b"private ignored owner bytes\x00")
    outside = tmp_path / "external.txt"
    outside.write_text("external owner content\n")
    (target / "owner-link").symlink_to(outside)
    original_readme = (target / "README.md").read_bytes()
    original_api = (target / "api.py").read_bytes()
    if index_kind == "staged":
        (target / "README.md").write_text("owner staged edit\n")
        git(target, "add", "README.md")
        (target / "README.md").write_bytes(original_readme)
    elif index_kind == "intent-to-add":
        (target / "intent.py").write_text("owner intent to add\n")
        git(target, "add", "-N", "intent.py")
    else:
        git(target, "update-index", "--assume-unchanged", "api.py")
    index_before = git_ops.snapshot_raw_index(target)
    item = {**_merge_item(1, "App.tsx", "high"), "item_uid": "item:1"}
    footprint = AuthorizedFixFootprint.build(target, {"api.py", "App.tsx", "README.md"}, [item])
    work = WorkContext(target, target, "main", git(target, "rev-parse", "main"), "feature",
        git(target, "rev-parse", "HEAD"), False, "scope-regression",
    )
    isolation = FixIsolationRound(work, footprint)
    try:
        sibling, baseline = isolation.create_group()
        (sibling.repo / "App.tsx").write_text("// retained assigned fix\n")
        isolation.retain_group(sibling.repo, frozenset({"App.tsx"}), baseline)
        (target / "README.md").unlink()
        (target / "README.md").symlink_to(outside)
        (target / "pkg/owner.py").unlink()
        (target / "pkg").rmdir()
        (target / "pkg").symlink_to(tmp_path)
        (target / "api.py").write_text("hidden unauthorized write\n")
        (target / "api.py").chmod(0o600)
        (target / "owner.env").write_text("overwritten ignored owner file\n")
        (target / "hidden.py").write_text("new ignored source\n")
        git(target, "add", "App.tsx")
        isolation.restore_parent()
        isolation.publish()
        assert git_ops.snapshot_raw_index(target) == index_before
        assert (target / "README.md").read_bytes() == original_readme
        assert (target / "api.py").read_bytes() == original_api
        assert stat.S_IMODE((target / "api.py").stat().st_mode) == 0o644
        assert (target / "pkg/owner.py").read_text() == "owner source\n"
        assert (target / "owner.env").read_bytes() == b"private ignored owner bytes\x00"
        assert not (target / "hidden.py").exists()
        assert (target / "owner-link").is_symlink()
        assert outside.read_text() == "external owner content\n"
        assert (target / "App.tsx").read_text() == "// retained assigned fix\n"
    finally:
        isolation.close()

def test_round_accepts_owner_deletion_of_a_tracked_directory(multi_stack_target: Path) -> None:
    target = multi_stack_target
    (target / "pkg").mkdir()
    (target / "pkg/removed.py").write_text("owner deleted source\n")
    git(target, "add", "pkg/removed.py")
    git(target, "commit", "-m", "test: deleted directory baseline")
    (target / "pkg/removed.py").unlink()
    (target / "pkg").rmdir()
    item = {**_merge_item(1, "App.tsx", "high"), "item_uid": "item:1"}
    footprint = AuthorizedFixFootprint.build(target, {"App.tsx"}, [item])
    work = WorkContext(target, target, "main", git(target, "rev-parse", "main"), "feature",
        git(target, "rev-parse", "HEAD"), False, "deleted-directory",
    )
    isolation = FixIsolationRound(work, footprint)
    try:
        isolation.restore_parent()
        assert not (target / "pkg").exists()
    finally:
        isolation.close()

"""Fix Guards And Scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.runner import run
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import git as _git
from tests.harness.remote_ci import NoCIRemote
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _add_bare_remote,
    _build_scope_creep_target,
    _ExtraEditBackend,
    _force_interactive,
    _install_stub_backend,
    _merge_item,
    _migration_project,
    _silence,
)


async def test_test_healing_guard_reverts_generated_migration_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:

    project, migration = _migration_project(tmp_path, "heal_migration_repo")
    pre_migration = migration.read_bytes()
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    mute_side_effects(heal=False)
    stub = _install_stub_backend(monkeypatch, project, pin_skill_availability=False)
    stub.merge_items = [_merge_item(1, "migrations/0001_init.sql", "high", desc="schema fix")]
    stub.heal_fix_generated = "migrations/0001_init.sql"
    stub.heal_fix_new_generated = "migrations/0002_add_x.sql"
    suite_dir = tmp_path / "suite"
    exit_code = await run(make_config(project, assume="yes", output_mode="loop", non_interactive=False, archive=False,
                                     test_command=host_test_command(suite_dir, mode="first"))
    )

    assert exit_code == 0
    assert int((suite_dir / "suite-count.txt").read_text()) >= 2
    assert stub.test_suite_calls == 0
    assert migration.read_bytes() == pre_migration
    new_migration = project / "migrations" / "0002_add_x.sql"
    assert new_migration.is_file()
    assert new_migration.read_text() == "-- new healing migration\n"
    assert not (project / ".daydream-heal-fix-applied").exists()
    violations = project / ".daydream" / "deep" / "generated-file-violations.json"
    recorded = json.loads(violations.read_text())
    # The host's full confinement runs after the repair turn and is this
    # artifact's last writer, so it records its own phase instead of the
    # per-turn guard's pre-repair ref. The offending path is recorded either way.
    assert recorded["violations"] == ["migrations/0001_init.sql"]
    assert recorded["phase"] == "test_heal"
    assert recorded["session_id"]

@pytest.mark.parametrize("scope_issue_filing", [True])
async def test_fix_reverts_post_fix_edit_outside_reviewed_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    no_ci_remote: NoCIRemote, scope_issue_filing: bool,
) -> None:
    """Revert an out-of-diff edit before commit; file an issue only when opted in."""

    target = _build_scope_creep_target(tmp_path, "scope_creep_residual")
    bare = _add_bare_remote(target)
    no_ci_remote.connect(target, bare)
    pre_fix_unrelated = (target / "unrelated.py").read_text()

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    # Commit runs for real (the scope-creep backend answers the commit prompt
    # with a real commit of the pre-staged index); heal is stubbed to pass.
    mute_side_effects(commit=False)
    stub = _ExtraEditBackend(target, target / "unrelated.py", "\n# scope creep\n")
    stub.fix_edit_line = "\n# daydream fix\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)

    issues: list[tuple[Any, ...]] = []

    def _record_issue(repo: Any, *, title: str, body: str, **kwargs: Any) -> str:
        issues.append((repo, title, body))
        return "https://github.com/owner/repo/issues/9"

    monkeypatch.setattr("daydream.git_ops.gh_issue_create", _record_issue)

    exit_code = await run(make_config(target, assume="yes", output_mode="loop", non_interactive=False, archive=False,
            scope_issue_filing=scope_issue_filing, pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0

    # 1. unrelated.py reverted to pre-fix content (git shows no change there).
    assert (target / "unrelated.py").read_text() == pre_fix_unrelated
    unrelated_diff = _git(target, "diff", "HEAD", "--", "unrelated.py")
    assert unrelated_diff == "", f"unrelated.py still differs from HEAD:\n{unrelated_diff}"
    assert len(issues) == int(scope_issue_filing), issues
    if scope_issue_filing:
        assert "unrelated.py" in issues[0][2]

    # 3/4. The commit's tree contains the fix but zero out-of-scope files: the
    # creep edit to unrelated.py must never survive. The commit tree may carry
    # test-stub sentinels and .daydream/ artifacts (untracked files created
    # mid-run, pre-staged by _do_commit deterministically), so the invariant is
    # inclusion + absence, not exact equality.
    committed_paths = _git(target, "show", "--name-only", "--format=", "HEAD").split()
    assert "api.py" in committed_paths
    assert "unrelated.py" not in committed_paths, f"commit tree leaks out-of-scope files: {committed_paths}"
    # The fix itself landed (api.py carries the daydream edit).
    assert "# daydream fix" in (target / "api.py").read_text()

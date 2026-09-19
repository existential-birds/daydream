"""Fix Guards And Scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import daydream
from tests.deep_orchestrator.support import (
    _scan_phase_events,
    _scan_trajectory_extra,
)
from tests.harness.git_helpers import bare_remote as _bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.remote_ci import NoCIRemote
from tests.test_deep_orchestrator import (
    INTENT_SENTINEL,
    MakeConfig,
    Mute,
    _add_bare_remote,
    _add_to_reviewed_diff,
    _build_gate_target,
    _build_scope_creep_target,
    _ExtraEditBackend,
    _fix_prompts,
    _force_interactive,
    _go_quote_project,
    _install_stub_backend,
    _merge_item,
    _migration_project,
    _prompt_ref,
    _PromptHookStub,
    _read_quality_gate,
    _run_quality_gate_fixture,
    _silence,
    _StubBackend,
)


async def test_fix_quality_gate_clamps_invalid_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path (#329/Finding 7): invalid thresholds resolve to the default, never the bad value."""
    from daydream.config_file import DaydreamFileConfig

    target = _build_gate_target(tmp_path, "gate_clamped_thresholds")
    exit_code = await _run_quality_gate_fixture(
        target,
        monkeypatch,
        make_config,
        mute_side_effects,
        fix_edit_line=None,
        file_config=DaydreamFileConfig(
            quality_gate_erosion_delta=-0.1,
            quality_gate_verbosity_delta=float("nan"),
        ),
    )
    assert exit_code == 0

    gate = _read_quality_gate(target)
    assert gate["enabled"] is True
    assert gate["erosion_delta_threshold"] == 0.05
    assert gate["verbosity_delta_threshold"] == 0.05
    entry = gate["rounds"][0]["per_file"]["api.py"]
    assert entry["erosion_delta"] == 0.0
    assert entry["flagged"] is False


async def test_fix_guard_reverts_generated_migration_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    no_ci_remote: NoCIRemote,
) -> None:
    from daydream.runner import run

    project, migration = _migration_project(tmp_path, "migration_repo")
    bare = _bare_remote(tmp_path / "remote.git")
    no_ci_remote.connect(project, bare)

    # Issue #336: the fix loop only auto-fixes findings whose file is in the
    # reviewed diff. The draft migration finding must therefore be part of the
    # diff (committed) or the gate would route it to a GitHub issue instead of
    # exercising the generated-file guard. Pin core.autocrlf so the CRLF draft
    # round-trips byte-identically through git.
    _git(project, "config", "core.autocrlf", "false")
    preexisting_tracked = project / "migrations" / "0000_local_draft.sql"
    preexisting_tracked.write_bytes(b"-- local draft\r\n")
    _git(project, "add", "migrations/0000_local_draft.sql")
    _commit(project, "test: commit local draft to the reviewed diff")

    pre_migration = migration.read_bytes()
    head_before = _git(project, "rev-parse", "HEAD")
    untouched_untracked = project / "migrations" / "0000_untouched.sql"
    untouched_untracked.write_bytes(b"-- untouched draft\r\n")
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    mute_side_effects(heal=False, commit=False)
    stub = _StubBackend(project)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    stub.merge_items = [
        _merge_item(1, "migrations/0001_init.sql", "high", desc="schema fix"),
        _merge_item(2, "api.py", "high", desc="source fix"),
        _merge_item(3, "migrations/0000_local_draft.sql", "high", desc="local schema fix"),
    ]
    stub.fix_edit_line = "\n-- FORBIDDEN EDIT\n"
    stub.fix_new_generated = "migrations/0002_add_x.sql"

    exit_code = await run(
        make_config(
            project,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )

    assert exit_code == 0
    assert migration.read_bytes() == pre_migration
    assert b"FORBIDDEN EDIT" not in migration.read_bytes()
    assert preexisting_tracked.read_bytes() == b"-- local draft\r\n"
    assert untouched_untracked.read_bytes() == b"-- untouched draft\r\n"
    assert (project / "migrations" / "0002_add_x.sql").read_text() == "-- new migration\n"
    assert "FORBIDDEN EDIT" in (project / "api.py").read_text()
    violations = project / ".daydream" / "deep" / "generated-file-violations.json"
    assert violations.exists()
    violations_payload = json.loads(violations.read_text())
    assert violations_payload["session_id"]
    assert violations_payload["phase"] == "fix"
    assert violations_payload["round_number"] == 1
    assert set(violations_payload["violations"]) == {
        "migrations/0001_init.sql",
        "migrations/0000_local_draft.sql",
    }
    head_after = _git(project, "rev-parse", "HEAD")
    assert head_after != head_before
    committed_paths = _git(project, "show", "--name-only", "--format=", "HEAD").split()
    assert "api.py" in committed_paths
    assert "migrations/0002_add_x.sql" in committed_paths
    commit_message = _git(project, "log", "-1", "--format=%B")
    assert "Daydream-Run: " in commit_message
    assert f"Daydream-Version: {daydream.__version__}" in commit_message
    assert head_after in _git(project, "ls-remote", "--heads", "origin", "feature")


async def test_fix_scrub_normalizes_smart_quote_in_changed_go_comment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    no_ci_remote: NoCIRemote,
) -> None:
    """Real path: a fix writing U+201D into a changed .go comment is scrubbed pre-commit."""
    from daydream.runner import run

    project = _go_quote_project(tmp_path)
    bare = _bare_remote(tmp_path / "remote.git")
    no_ci_remote.connect(project, bare)
    notes_before = (project / "notes.md").read_bytes()
    head_before = _git(project, "rev-parse", "HEAD")
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    mute_side_effects(heal=False, commit=False)
    stub = _StubBackend(project)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    # A one-file diff collapses to single-stack mode (no cross-stack merge
    # agent), so the finding is driven through the per-stack parse: the go
    # review's record points at the sole reviewed file, main.go.
    stub.parse_by_stack = {
        "go": {
            "severity": "high",
            "confidence": "HIGH",
            "file": "main.go",
            "line": 1,
            "description": "comment doc fix",
        }
    }
    stub.fix_edit_line = "\n// not \u201d\n"  # fix agent writes U+201D into the changed .go comment
    exit_code = await run(
        make_config(
            project,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0
    go_src = (project / "main.go").read_text()
    assert "not \u201d" not in go_src  # U+201D never lands in the committed tree
    assert '// not "' in go_src  # normalized to ASCII straight quote
    assert (project / "notes.md").read_bytes() == notes_before  # non-changed file untouched
    assert _git(project, "rev-parse", "HEAD") != head_before


async def test_test_healing_guard_reverts_generated_migration_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """The runner snapshots and restores a forbidden edit made by a heal turn."""
    from daydream.runner import run

    project, migration = _migration_project(tmp_path, "heal_migration_repo")
    pre_migration = migration.read_bytes()
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    mute_side_effects(heal=False)
    stub = _StubBackend(project)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    stub.merge_items = [_merge_item(1, "migrations/0001_init.sql", "high", desc="schema fix")]
    stub.fail_first_test_run = True
    stub.heal_fix_generated = "migrations/0001_init.sql"
    stub.heal_fix_new_generated = "migrations/0002_add_x.sql"

    exit_code = await run(
        make_config(
            project,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
        )
    )

    assert exit_code == 0
    assert migration.read_bytes() == pre_migration
    new_migration = project / "migrations" / "0002_add_x.sql"
    assert new_migration.is_file()
    assert new_migration.read_text() == "-- new healing migration\n"
    assert not (project / ".daydream-heal-fix-applied").exists()
    violations = project / ".daydream" / "deep" / "generated-file-violations.json"
    assert json.loads(violations.read_text()) == {
        "violations": ["migrations/0001_init.sql"],
        "ref": "HEAD",
    }


async def test_fix_guard_restore_failure_aborts_before_commit(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A forbidden generated edit cannot reach commit when restoration fails."""
    from daydream.git_ops import GitError
    from daydream.runner import run

    migration = multi_stack_target / "migrations" / "0001_init.sql"
    migration.parent.mkdir()
    migration.write_text("SELECT 1;\n")
    _git(multi_stack_target, "add", "migrations/0001_init.sql")
    head_before = _git(multi_stack_target, "rev-parse", "HEAD")

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(heal=True, commit=False)
    stub = _StubBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    stub.merge_items = [_merge_item(1, "migrations/0001_init.sql", "high", desc="schema fix")]
    stub.fix_edit_line = "-- FORBIDDEN EDIT\n"
    monkeypatch.setattr(
        "daydream.git_ops.restore_paths_from_ref",
        lambda *args, **kwargs: (_ for _ in ()).throw(GitError("restore failed")),
    )

    exit_code = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
        )
    )

    assert exit_code == 1
    assert _git(multi_stack_target, "rev-parse", "HEAD") == head_before


async def test_parallel_fix_commit_runs_once_after_all(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC#6: commit stays serial and runs exactly once, after every parallel fix lands."""
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    files = ["f1.py", "f2.py", "f3.py"]
    _add_to_reviewed_diff(multi_stack_target, files)
    stub.merge_items = [_merge_item(i + 1, f, "high") for i, f in enumerate(files)]
    fix_marker = "\n# fixed before commit\n"
    stub.fix_edit_line = fix_marker
    seen_at_commit: list[bool] = []

    async def _spy_commit(backend: Any, work: Any, **kwargs: Any) -> None:
        seen_at_commit.append(all(fix_marker in (multi_stack_target / path).read_text() for path in files))

    monkeypatch.setattr("daydream.deep.fix_steps.phase_commit_push", _spy_commit)
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    assert seen_at_commit == [True]  # exactly one commit, and every fix already landed


@pytest.mark.parametrize("scope_issue_filing", [False, True])
async def test_fix_reverts_post_fix_edit_outside_reviewed_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    no_ci_remote: NoCIRemote,
    scope_issue_filing: bool,
) -> None:
    """Revert an out-of-diff edit before commit; file an issue only when opted in."""
    from daydream.runner import run

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

    exit_code = await run(
        make_config(
            target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
            scope_issue_filing=scope_issue_filing,
            pr_number=no_ci_remote.pr_number,
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


async def test_reverted_edit_dedups_across_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    no_ci_remote: NoCIRemote,
) -> None:
    """#1051 regression: an opted-in run does not re-file an issue for a
    reverted edit whose fingerprint marker already sits on an open issue."""
    from daydream.runner import run

    target = _build_scope_creep_target(tmp_path, "scope_creep_dedup")
    bare = _add_bare_remote(target)
    no_ci_remote.connect(target, bare)
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
    stub = _ExtraEditBackend(target, target / "unrelated.py", "\n# scope creep\n")
    stub.fix_edit_line = "\n# daydream fix\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    created: list[tuple[str, str]] = []

    def _record_and_list_create(repo: Any, *, title: str, body: str, **kw: Any) -> str:
        created.append((title, body))
        return "https://github.com/owner/repo/issues/9"

    monkeypatch.setattr("daydream.git_ops.gh_issue_create", _record_and_list_create)
    # The dedup lookup sees a prior issue carrying this revert's marker —
    # computed the same way the edit filer computes it (path + diff head).
    # Spy on the evidence diff the filer captures pre-revert: a prior run would
    # have filed a marker over this exact patch (same path + same edit → same
    # fingerprint), so the dedup lookup serves an issue body carrying it.
    from daydream import git_ops as _git_ops
    from daydream.deep.scope_issues import _scope_edit_fingerprint, _scope_edit_marker

    recorded: list[str] = []
    _real_diff = _git_ops.diff_worktree_against

    def _spy_diff(repo: Any, ref: str, paths: Any, **kw: Any) -> str:
        patch = _real_diff(repo, ref, paths, **kw)
        if list(paths) == ["unrelated.py"]:
            recorded.append(patch)
        return patch

    monkeypatch.setattr("daydream.git_ops.diff_worktree_against", _spy_diff)

    def _list_with_prior_marker(repo: Any, **kw: Any) -> list[dict[str, Any]]:
        marker = _scope_edit_marker(_scope_edit_fingerprint("unrelated.py", recorded[-1]))
        return [
            {
                "number": 11,
                "title": "[daydream] out-of-scope edit reverted: unrelated.py",
                "body": f"prior run\n{marker}",
                "url": "u",
            }
        ]

    monkeypatch.setattr("daydream.git_ops.gh_issue_list", _list_with_prior_marker)
    exit_code = await run(
        make_config(
            target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
            scope_issue_filing=True,
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0
    # Revert still happened; the duplicate issue did not.
    committed_paths = _git(target, "show", "--name-only", "--format=", "HEAD").split()
    assert "unrelated.py" not in committed_paths
    assert created == [], f"stale revert must not re-file, got {created!r}"


async def test_fix_reverts_post_fix_edit_outside_reviewed_diff_restore_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#336 real-path: a failed residual revert aborts before commit."""
    from daydream.git_ops import GitError
    from daydream.runner import run

    target = _build_scope_creep_target(tmp_path, "scope_creep_residual_fail")
    head_before = _git(target, "rev-parse", "HEAD")

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
    stub = _ExtraEditBackend(target, target / "unrelated.py", "\n# scope creep\n")
    stub.fix_edit_line = "\n# daydream fix\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    monkeypatch.setattr(
        "daydream.git_ops.restore_group_from_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(GitError("restore failed")),
    )

    exit_code = await run(
        make_config(
            target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=False,
        )
    )

    assert exit_code == 1
    assert _git(target, "rev-parse", "HEAD") == head_before


async def test_fix_tool_veto_blocks_denied_write(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Built-in rules veto a deferred denied Write and record the abort/event."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="protected write")]
    stub.deferred_write_pairs = ["api.py"]
    (multi_stack_target / ".daydream.toml").write_text(
        'tool_supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )
    source_before = (multi_stack_target / "api.py").read_bytes()
    traj = tmp_path / "trajectory.json"

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
            trajectory_path=traj,
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "api.py").read_bytes() == source_before
    stop_reasons = _scan_trajectory_extra(multi_stack_target / ".daydream", traj, "stop_reason")
    assert "tool_vetoed:Write" in stop_reasons
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "tool_veto")
    assert any(event.get("metadata", {}).get("tool_name") == "Write" for event in events)


async def test_fix_tool_veto_allows_unmatched_write(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Built-in rules allow a Write whose path does not match the deny glob."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "App.tsx", "high", desc="allowed write")]
    stub.deferred_write_pairs = ["App.tsx"]
    (multi_stack_target / ".daydream.toml").write_text(
        'tool_supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "App.tsx").read_text() == "backend resumed"
    assert not _scan_trajectory_extra(multi_stack_target / ".daydream", Path("/missing"), "stop_reason")


async def test_fix_tool_veto_stops_subsequent_calls(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A vetoed first deferred Write prevents the generator's later Write."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="first"), _merge_item(2, "App.tsx", "low", desc="second")]
    stub.deferred_write_pairs = ["api.py", "App.tsx"]
    (multi_stack_target / ".daydream.toml").write_text(
        'tool_supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )
    api_before = (multi_stack_target / "api.py").read_bytes()
    app_before = (multi_stack_target / "App.tsx").read_bytes()

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "api.py").read_bytes() == api_before
    assert (multi_stack_target / "App.tsx").read_bytes() == app_before


async def test_fix_tool_supervisor_off_writes(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """With tool supervision off, the deferred Write resumes and writes."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="unprotected write")]
    stub.deferred_write_pairs = ["api.py"]
    (multi_stack_target / ".daydream.toml").write_text('tool_supervisor = "off"\n')

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "api.py").read_text() == "backend resumed"


async def test_confirmed_intent_reaches_fix_prompt(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """The confirmed author intent reaches every deep fix prompt so a fixer can't undo a deliberate decision."""
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None, **_kwargs: {"body": INTENT_SENTINEL},
    )

    observed_intent: list[str] = []

    class _IntentReadingStub(_PromptHookStub):
        def intercept(self, cwd: Path, prompt: str) -> None:
            if prompt.lower().startswith(("fix this issue", "fix these")):
                intent_ref = Path(_prompt_ref(prompt, "intent"))
                observed_intent.append(intent_ref.read_text(encoding="utf-8"))
            return None

    stub = _IntentReadingStub(multi_stack_target)
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: stub,
    )
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    stub.merge_items = [_merge_item(1, "api.py", "high")]

    rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
        )
    )
    assert rc == 0
    fix_prompts = _fix_prompts(stub)
    assert fix_prompts, "expected at least one fix prompt"
    joined = "\n".join(fix_prompts)
    assert INTENT_SENTINEL not in joined
    assert observed_intent
    assert all(INTENT_SENTINEL in body for body in observed_intent)
    assert "- intent:" in joined
    assert "/live/.daydream/deep/intent.md" in joined
    low = joined.lower()
    assert "deliberate" in low and ("do not" in low or "don't" in low)

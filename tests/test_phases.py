# tests/test_phases.py
"""Tests for phase functions with backend abstraction."""

import functools
import json
import shlex
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import anyio
import pytest

from daydream import artifact_visibility as av, git_ops, phases
from daydream.artifact_visibility import ArtifactVisibilityError, OutputLabel, artifact_dir_for
from daydream.backends import (
    AgentEvent,
    ResultEvent,
    TextEvent,
)
from daydream.config import TEST_WALL_BUDGET_S
from daydream.deep.artifacts import DeepArtifact
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.git_ops import GitError, WorktreeRollbackSnapshot
from daydream.phases import (
    PushAttemptError,
    TestAttemptEvidence,
    phase_commit_push,
    publish,
)
from daydream.phases.handoff import (
    _changed_files,
    _resolve_handoff_paths,
    _run_failure_summarizer,
)
from daydream.phases.publish import (
    _do_commit,
)
from daydream.phases.repair_checkpoint import read_repair_checkpoint
from daydream.phases.repair_outcome import (
    RepairOutcome,
    classify_repair_outcome,
)
from daydream.phases.testing import (
    _reject_test_healing_generated_file_edits,
)
from daydream.test_execution import TestExecutionIdentity, TestExecutionResult, resolve_test_recipe, run_test_command
from daydream.trajectory import (
    DaydreamRunFlow,
    run_directory,
    siblings_directory,
)
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import commit as git_commit, configure_identity, git, init_repo
from tests.harness.review_result import merge_result
from tests.harness.trajectory import make_recorder

_RESULT = ResultEvent(structured_output=None, continuation=None)
_FIX_TURN: tuple[AgentEvent, ...] = (TextEvent(text="Applied fix attempt"), _RESULT)


def _structured_turn(structured: object) -> tuple[AgentEvent, ...]:
    if isinstance(structured, dict) and isinstance(structured.get("items"), list):
        structured = merge_result(structured["items"])
    return (ResultEvent(structured_output=structured, continuation=None),)


def _verdict(verdict: str, suggested_command: str | None, reason: str) -> dict[str, str | None]:
    return {"verdict": verdict, "suggested_command": suggested_command, "reason": reason}


def _record_host_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Observe calls while retaining the real bounded subprocess and its exit status."""
    calls: list[dict[str, Any]] = []
    async def record_run(**kwargs: Any) -> TestExecutionResult:
        calls.append(kwargs)
        return await run_test_command(**kwargs)
    monkeypatch.setattr("daydream.phases.test_evidence.run_test_command", record_run)
    return calls


def _host_suite_config(repo: Path, *, mode: Literal["first", "all"] = "first") -> Any:
    """Keep suite instrumentation outside the tested tree; exit status owns its verdict."""
    return SimpleNamespace(
        file_config=None,
        test_command=host_test_command(
            repo.parent / f"{repo.name}-host-suite", mode=mode, output="1 failed, 0 passed",
        ),
    )


def _execution_identity(
    repo: Path, *, recipe: Any, argv: tuple[str, ...], output_tree_key: str = "t", outcome: str = "passed",
) -> TestExecutionIdentity:
    """Match the gate identity; unborn HEAD fixtures use empty revision facts."""
    try:
        head_sha = git_ops.head_sha(repo)
        branch = git_ops.current_branch(repo) or ""
    except GitError:
        head_sha = ""
        branch = ""
    return TestExecutionIdentity(
        session_id="s", argv=argv, cwd_relative=recipe.package.cwd_relative, runner=recipe.package.runner,
        interpreter=recipe.package.interpreter, config_digest=recipe.package.config_digest,
        absent_components=recipe.package.absent_components, input_tree_key="t", output_tree_key=output_tree_key,
        head_sha=head_sha, branch=branch, kind="host", outcome=cast(Any, outcome),
    )


def _handoff_turn(body: str) -> tuple[AgentEvent, ...]:
    return _structured_turn({"handoff_prompt": body})


def _private_session(tmp_path: Path, work: WorkContext, session_id: str) -> Any:
    """Open a real private session with ownership resolution, locking, and filesystem I/O."""
    locations = av.private_root_locations(base=(tmp_path / "private").resolve())
    owner = av.resolve_private_workspace_owner(work.source, locations=locations)
    return av.open_artifact_session(work, session_id=session_id, owner=owner)


def _inline_or_exact_backend(repo: Path, *, inline: bool, events: tuple[AgentEvent, ...] | None = None,
    script: list[list[AgentEvent]] | None = None,
) -> ScriptedBackend:
    """Strict Claude PreToolUse isolation forces inline inputs; ordinary backends use exact paths."""
    if inline:
        return ScriptedBackend(
            script=script, events=events, audit_root_isolation="claude-pretooluse", audit_root=repo.resolve(),
        )
    return ScriptedBackend(script=script, events=events)




def _unconfined_finding_file(tmp_path: Path, path_kind: str) -> str:
    """Build a unique traversal, absolute, or outward-symlink finding path for rejection."""
    if path_kind == "traversal":
        return "../outside.py"
    if path_kind == "absolute":
        outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
        outside.write_bytes(b"x")
        return str(outside.resolve())
    if path_kind == "symlink":
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_path.parent / f"{tmp_path.name}-target.py"
        target.write_bytes(b"x")
        (src_dir / "handler.py").symlink_to(target)
        return "src/handler.py"
    raise AssertionError(f"unknown path_kind: {path_kind!r}")


@pytest.fixture
def _quiet_phase_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence output while preserving UI exports captured by lazy imports."""
    monkeypatch.setattr("daydream.agent.console.print", lambda *args, **kwargs: None)


async def test_phase_entry_points_require_explicit_authority_contracts(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """A caller omission of evidence, edit scope, or round state fails before any dispatch."""
    backend = ScriptedBackend()
    init_repo(tmp_path)
    (tmp_path / "app.py").write_text("original\n")
    git(tmp_path, "add", "app.py")
    git_commit(tmp_path, "round contract baseline")
    work = make_work(tmp_path)
    with pytest.raises(TypeError, match="requires session_id, capture_tree_key, and footprint"):
        await phases.phase_test_and_heal(backend, work, config=None, session_id="missing-contract",
                                         capture_tree_key=lambda: "unchanged")
    item = {"file": "app.py", "description": "repair"}
    with pytest.raises(TypeError, match="require explicit edit_scope and read_scope"):
        await phases.phase_fix_batched(backend, work, [item], [1], 1, read_scope=frozenset({"app.py"}))
    items = [{"item_uid": "item:app", "file": "app.py"}]
    with pytest.raises(TypeError, match="requires footprint and round_snapshot"):
        await phases.phase_fix_parallel(backend, work, items,
                                        round_snapshot=_round_snapshot(tmp_path, frozenset({"app.py"})))
    assert backend.prompts == []


def _round_snapshot(repo: Path, paths: frozenset[str]) -> WorktreeRollbackSnapshot:
    """Capture the caller's real round state without changing findings or authority."""
    return WorktreeRollbackSnapshot(
        ref=git_ops.head_sha(repo), index=git_ops.snapshot_index(repo),
        path_states=git_ops.snapshot_worktree_paths(repo, paths),
        untracked=git_ops.snapshot_untracked_paths(repo, include_runtime_artifacts=False),
    )


def _seed_healing_repo(tmp_path: Path, path: str,
    contents: str = "-- original\n",
    message: str = "initial migration",
) -> tuple[Path, str | None]:
    """Commit one seeded file and return it with the pre-fix stash snapshot."""
    init_repo(tmp_path)
    file_path = tmp_path / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(contents)
    git(tmp_path, "add", path)
    git_commit(tmp_path, message)
    return file_path, git_ops.stash_create(tmp_path)


def _reject_violations(tmp_path: Path, snapshot: str | None) -> list[str] | None:
    """Run the healing guard with the standard captured-snapshot arguments."""
    return _reject_test_healing_generated_file_edits(
        tmp_path, snapshot=snapshot, snapshot_captured=True, pre_untracked=set(), allow_standalone=True,
    )


def test_test_healing_guard_restores_generated_population_and_paired_manifests(
    tmp_path: Path, _quiet_phase_ui: None,
) -> None:
    """One repair cannot erase header evidence or retain mismatched lockfile manifests."""
    init_repo(tmp_path)
    originals = {
        "migrations/0001_init.sql": "-- original migration\n",
        "alembic/versions/abc123_add_col.py": "revision = 'abc123'\n",
        "migrations/a/b.sql": "-- nested migration\n",
        "migrations/a-b.sql": "-- flat migration\n",
        "internal/model_generated.go": "package generated\n",
        "src/client.py": "# @generated\nORIGINAL = True\n",
        "src/client.go": "// Code generated by protoc. DO NOT EDIT.\npackage client\n",
        "src/client.txt": "Code generated by tool. DO NOT EDIT.\n",
        "Cargo.lock": "# original lock\n", "Cargo.toml": "[package]\nname = 'original'\n",
        "uv.lock": "version = 1\n",
        "frontend/package-lock.json": '{"version":1}\n', "frontend/package.json": '{"version":1}\n',
        "src/app.py": "def app():\n    pass\n# @generated\n", "src/runtime.py": "# generated at runtime\n",
    }
    for name, body in originals.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    git(tmp_path, "add", ".")
    git_commit(tmp_path, "generated population baseline")
    protected = {
        "migrations/0000_local_draft.sql": b"-- local draft\r\n",
        "migrations/0000_untouched.sql": b"-- untouched draft\r\n",
    }
    for name, content in protected.items():
        (tmp_path / name).write_bytes(content)
    snapshot = git_ops.stash_create(tmp_path)
    for name in originals:
        (tmp_path / name).write_text("REPAIR = True\n")

    (tmp_path / "migrations/0000_local_draft.sql").write_bytes(b"-- forbidden healing edit\n")

    violations = _reject_test_healing_generated_file_edits(
        tmp_path, snapshot=snapshot, snapshot_captured=True, pre_untracked=set(protected),
        pre_untracked_contents=protected, allow_standalone=True,
    )
    assert (tmp_path / "alembic/versions/abc123_add_col.py").read_bytes() == b"revision = 'abc123'\n"
    assert (tmp_path / "uv.lock").read_bytes() == b"version = 1\n"
    assert set(violations or ()) == {
        "uv.lock",
        "alembic/versions/abc123_add_col.py",
        "migrations/0001_init.sql", "internal/model_generated.go", "src/client.py", "src/client.go",
        "src/client.txt", "Cargo.lock", "frontend/package-lock.json",
        "migrations/a/b.sql", "migrations/a-b.sql", "migrations/0000_local_draft.sql",
    }
    assert {name: (tmp_path / name).read_text() for name in originals} == {
        **originals, "src/app.py": "REPAIR = True\n", "src/runtime.py": "REPAIR = True\n",
    }
    assert violations is not None and violations.count("migrations/0000_local_draft.sql") == 1
    assert (tmp_path / "migrations/0000_local_draft.sql").read_bytes() == b"-- local draft\r\n"
    assert (tmp_path / "migrations/0000_untouched.sql").read_bytes() == b"-- untouched draft\r\n"
    recovery = tmp_path / ".daydream" / "partial-fixes"
    patches = [path.read_text() for path in recovery.glob("migrations-a-b.sql*.patch")]
    assert len(patches) == 2
    assert any("--- a/migrations/a/b.sql" in patch for patch in patches)
    assert any("--- a/migrations/a-b.sql" in patch for patch in patches)


def test_test_healing_guard_fails_safe_when_discovery_or_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _quiet_phase_ui: None,
) -> None:
    """Unknown changes never drive destructive recovery; an unrestorable forbidden edit stays unsafe."""
    migration, snapshot = _seed_healing_repo(tmp_path, "migrations/0001_init.sql")
    migration.write_text("-- healing edit\n")
    with monkeypatch.context() as patch:
        patch.setattr("daydream.git_ops.changed_files_against",
            lambda *args, **kwargs: (_ for _ in ()).throw(GitError("unavailable")),
        )
        assert _reject_violations(tmp_path, snapshot) == []
    assert migration.read_text() == "-- healing edit\n"
    monkeypatch.setattr("daydream.git_ops.restore_paths_from_ref",
        lambda *args, **kwargs: (_ for _ in ()).throw(GitError("restore failed")),
    )
    assert _reject_violations(tmp_path, snapshot) is None
    assert migration.read_text() == "-- healing edit\n"
    # Byte baselines are captured only for untracked files that are actually generated.
    from daydream.generated_files import _snapshot_untracked_generated_files
    (tmp_path / "Cargo.lock").write_bytes(b"lock\n")
    (tmp_path / "notes.txt").write_bytes(b"notes\n")
    assert _snapshot_untracked_generated_files(tmp_path, {"Cargo.lock", "notes.txt"}) == {"Cargo.lock": b"lock\n"}


def _retained_commit_tree(repo: Path, paths: set[str]) -> dict[str, Any]:
    """Supply real retained-tree evidence with the test's explicit authorized paths."""
    return {
        "retained_paths": frozenset(paths),
        "retained_states": git_ops.snapshot_worktree_paths(repo, paths),
        "initial_index": git_ops.snapshot_index(repo),
    }


@pytest.mark.asyncio
async def test_push_rejects_remote_url_changed_by_real_hook(tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """A hook cannot make verification attest a different configured remote."""

    remote = tmp_path / "remote-url-race.git"
    git(tmp_path, "init", "--bare", str(remote))
    repo = _init_committed_repo(tmp_path / "remote-url-race-checkout", "feature")
    original = "https://github.com/fork-user/widgets.git"
    replacement = "https://github.com/other-user/widgets.git"
    for url in (original, replacement):
        git(repo, "config", "--add", f"url.{remote.resolve().as_uri()}.insteadOf", url)
    git(repo, "remote", "add", "origin", original)
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(
        "#!/bin/sh\n"
        f"git config remote.origin.url {shlex.quote(replacement)}\n"
    )
    hook.chmod(0o755)
    (repo / "app.py").write_text("x = 1\n")

    with pytest.raises(PushAttemptError) as exc_info:
        await _do_commit(ScriptedBackend(), make_work(repo), push=True, interactive=False,
            **_retained_commit_tree(repo, {"app.py"}),
            config=_hook_run_config(),
        )

    assert exc_info.value.receipt.pushed_repository == "fork-user/widgets"
    assert git(repo, "config", "--get", "remote.origin.url") == replacement
    assert git(remote, "rev-parse", "refs/heads/feature") == exc_info.value.receipt.sha


@pytest.mark.asyncio
async def test_push_verification_failure_surfaces_even_when_push_succeeds(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful push still fails unless the remote reports the pushed SHA."""
    monkeypatch.setattr(git_ops, "remote_contains_commit", lambda *a, **k: False)
    work_repo = _pushable_repo(tmp_path)
    (work_repo / "fix.py").write_text("fixed\n")
    work = make_work(work_repo)
    with pytest.raises(git_ops.GitError):
        await _do_commit(
            ScriptedBackend(), work, push=True, interactive=False, items=[{"file": "fix.py", "description": "fix bug"}],
            **_retained_commit_tree(work.repo, {"fix.py"}),
         config=_hook_run_config(),)


@pytest.mark.asyncio
async def test_do_commit_requires_retained_tree_authority(
    git_repo: Path, make_work: Callable[..., WorkContext],
) -> None:
    """Missing retained authority fails before any index or worktree mutation."""
    (git_repo / "app.py").write_text("x = 0\n")
    git(git_repo, "add", "app.py")
    git_commit(git_repo, "baseline app.py")
    (git_repo / "app.py").write_text("x = 1\n")
    (git_repo / "notes.txt").write_text("user scratch\n")
    before = git(git_repo, "status", "--porcelain")
    head = git_ops.head_sha(git_repo)
    with pytest.raises(TypeError, match="retained_paths, retained_states, and initial_index are required"):
        await _do_commit(ScriptedBackend(), make_work(git_repo), push=False, config=_hook_run_config(),)
    assert git(git_repo, "status", "--porcelain") == before
    assert git_ops.head_sha(git_repo) == head
    assert git(git_repo, "diff", "--cached") == ""


def _init_committed_repo(path: Path, branch: str) -> Path:
    """A real repo on ``branch`` with one baseline commit of ``app.py``."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-b", branch)
    configure_identity(path)
    (path / "app.py").write_text("x = 0\n")
    git(path, "add", "app.py")
    git_commit(path, "baseline")
    return path


def _pushable_repo(tmp_path: Path) -> Path:
    """A real clone of a real bare remote with one baseline commit."""
    remote = tmp_path / "remote.git"
    tmp_path.mkdir(parents=True, exist_ok=True)
    git(tmp_path, "init", "--bare", remote.name)
    work_repo = _init_committed_repo(tmp_path / "clone", "main")
    git(work_repo, "remote", "add", "origin", str(remote))
    return work_repo


def _install_pre_push_hook(work_repo: Path) -> None:
    """Install a real, executable (passing) pre-push hook."""
    hooks = work_repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    hook = hooks / "pre-push"
    hook.write_text("#!/bin/sh\nexit 0\n")
    hook.chmod(0o755)


def _hook_run_config(test_command: str = "true") -> Any:
    """Minimal RunConfig stand-in resolving a canonical test command."""
    return SimpleNamespace(file_config=None, test_command=test_command)


@pytest.mark.asyncio
async def test_hook_aware_push_red_suite_blocks_push(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A red host suite blocks hook-bearing pushes despite an existing local commit."""

    repo = _pushable_repo(tmp_path)
    _install_pre_push_hook(repo)
    (repo / "fix.py").write_text("fixed\n")
    remote_head_before = git(repo, "ls-remote", "origin", "refs/heads/main")

    _record_host_runs(monkeypatch)

    with pytest.raises(RuntimeError, match="Pre-push validation"):
        await _do_commit(ScriptedBackend(), make_work(repo), push=True, interactive=False,
            items=[{"file": "fix.py", "description": "fix bug"}], **_retained_commit_tree(repo, {"fix.py"}),
            config=_hook_run_config("false"),
        )
    # Nothing was pushed: the remote still reports the baseline sha only.
    assert git(repo, "ls-remote", "origin", "refs/heads/main") == remote_head_before
    # The local commit exists but is unpushed.
    assert git(repo, "log", "-1", "--format=%B").startswith("fix:")


def _footprint(repo: Path) -> AuthorizedFixFootprint:
    """A no-edit-authority footprint: the repair record, not the prompt, carries scope."""
    return AuthorizedFixFootprint.build(repo, set(), [])


@pytest.mark.asyncio
async def test_confinement_runs_for_every_repair_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    """Requirement 12: timeout and exception are covered, not just completion."""
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)

    async def stall() -> AsyncIterator[TextEvent]:
        while True:
            yield TextEvent(text="partial")
            await anyio.sleep(0)

    def boom() -> Any:
        raise RuntimeError("boom")

    turns: dict[str, Callable[[], Any]] = {"normal": lambda: _FIX_TURN, "timeout": stall, "exception": boom}
    for outcome_case, repair_turn in turns.items():
        repo = tmp_path / outcome_case
        repo.mkdir()
        calls: list[str] = []

        def responder(_cwd: Path, prompt: str, *_rest: Any, repair_turn: Callable[[], Any] = repair_turn) -> Any:
            if prompt.lower().startswith("the tests failed"):
                return repair_turn()
            return _handoff_turn("Repair stopped")

        call = phases.phase_test_and_heal(
            ScriptedBackend(responder=responder), make_work(repo), config=_host_suite_config(repo), session_id="s1",
            capture_tree_key=lambda: "tree-1", footprint=_footprint(repo),
            confinement=functools.partial(calls.append, "ran"), allow_standalone=True,
        )
        if outcome_case == "exception":
            # The host error still reaches the caller; confinement must not swallow it.
            with pytest.raises(RuntimeError, match="boom"):
                await call
        else:
            result = await call
            assert result.repairs[0].outcome is (RepairOutcome.BUDGET_INTERRUPTED if outcome_case == "timeout"
                                                 else RepairOutcome.DIAGNOSIS_UNRESOLVED), outcome_case
        assert calls == ["ran"], f"confinement skipped for outcome={outcome_case}"


@pytest.mark.asyncio
async def test_failing_confinement_blocks_the_repair_and_records_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    """A confinement that cannot converge stops the loop and names the failure by type."""

    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")
    def responder(_cwd: Path, prompt: str, *_rest: Any) -> tuple[AgentEvent, ...]:
        if prompt.lower().startswith("the tests failed"):
            return _FIX_TURN
        return _handoff_turn("Confinement blocked repair")

    def confine() -> None:
        raise GitError("restore failed")

    backend = ScriptedBackend(responder=responder)
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), config=_host_suite_config(tmp_path), session_id="s1",
        capture_tree_key=lambda: "tree-1", footprint=_footprint(tmp_path),
        confinement=confine, allow_standalone=True,
    )

    # The suite is never rerun against the unconverged tree.
    assert (result.passed, result.retries, result.proceed) == (False, 1, False)
    # Only repair and failure handoff are agent turns; the failed suite ran host-side.
    assert backend.call_count == 2
    assert "failure-summarizer" in backend.prompts[-1].lower()
    assert [r.outcome for r in result.repairs] == [RepairOutcome.SCOPE_BLOCKED]
    assert any("GitError" in diagnostic and "restore failed" in diagnostic
        for r in result.repairs for diagnostic in r.diagnostics)


def test_host_interruption_outranks_model_claim_of_success() -> None:
    """A nonbudget stop maps to execution error even when the phase caller raises."""
    assert classify_repair_outcome("backend_failure", "") is RepairOutcome.EXECUTION_ERROR


async def test_grouped_fix_delivers_scopes_verdicts_evidence_and_test_map_then_retains_both_patches(
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    """The real grouped FIX owner carries all judgment fields while preserving disjoint fixes and trajectories."""
    repo = tmp_path / "repo"
    init_repo(repo)
    before = {
        "src/handler.py": "VALUE = 0\n",
        "src/util.py": "CONTEXT = 0\n",
        "tests/test_app.py": "VALUE = 0\n",
        "daydream/app.py": "SOURCE = 0\n",
        "boom.py": "BROKEN = 0\n",
    }
    for name, body in before.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    git(repo, "add", ".")
    git_commit(repo, "baseline fix inputs")
    items = [
        {
            "id": 1,
            "item_uid": "item:1",
            "file": "src/handler.py",
            "line": 42,
            "description": "Off-by-one in loop bound",
            "related_files": ["src/util.py"],
            "verifier_verdict": "contradicts",
            "evidence": "the spec says otherwise",
            "unverified_assumptions": ["assumes UTC timezone"],
        },
        {
            "id": 2,
            "item_uid": "item:2",
            "file": "src/handler.py",
            "line": 88,
            "description": "Unchecked None deref",
            "verifier_verdict": "uncertain",
            "evidence": "could not reproduce",
            "unverified_assumptions": ["assumes single-threaded"],
        },
        {
            "id": 3,
            "item_uid": "item:3",
            "file": "src/handler.py",
            "line": None,
            "description": "Missing await on coroutine",
            "evidence": "tests/test_handler.py:10",
        },
        {
            "id": 4,
            "item_uid": "item:4",
            "file": "tests/test_app.py",
            "line": 1,
            "description": "Incorrect test expectation",
            "evidence": "tests/test_app.py:10",
        },
        {
            "id": 5,
            "item_uid": "item:boom",
            "file": "boom.py",
            "line": 1,
            "description": "Failed singleton",
        },
    ]
    test_map = repo / "test-map.json"
    test_map.write_text(
        json.dumps({"test_mapping": [{"test_file": "tests/test_app.py", "source_file": "daydream/app.py"}]})
    )
    footprint = AuthorizedFixFootprint.build(repo, set(before), items)
    snapshot = _round_snapshot(repo, footprint.run_allowed_paths)

    def apply(cwd: Path, prompt: str, *_args: Any) -> tuple[AgentEvent, ...]:
        if "Failed singleton" in prompt:
            raise RuntimeError("kaboom")
        if "Off-by-one in loop bound" in prompt:
            (cwd / "src/handler.py").write_text("VALUE = 1\n")
        else:
            assert "Incorrect test expectation" in prompt
            (cwd / "tests/test_app.py").write_text("VALUE = 1\n")
        return _FIX_TURN

    backend = ScriptedBackend(responder=apply)
    recorder = make_recorder(repo, run_flow=DaydreamRunFlow.DEEP)
    async with recorder:
        failures = await phases.phase_fix_parallel(
            backend, make_work(repo), items, footprint=footprint, round_snapshot=snapshot, test_map_path=test_map
        )
    assert failures == {"boom.py": "RuntimeError: kaboom"}
    assert len(backend.prompts) == 3
    assert (repo / "boom.py").read_text() == "BROKEN = 0\n"
    group = next(prompt for prompt in backend.prompts if "Off-by-one in loop bound" in prompt)
    assert all(
        value in group
        for value in (
            "Fix these 3 issues",
            "address ALL of the above findings in one coherent patch",
            "Off-by-one in loop bound",
            "Unchecked None deref",
            "Missing await on coroutine",
            "   Line: 42\n",
            "   Line: 88\n",
            "Verifier verdict: contradicts",
            "the spec says otherwise",
            "assumes UTC timezone",
            "Verifier verdict: uncertain",
            "could not reproduce",
            "assumes single-threaded",
            "tests/test_handler.py:10",
            "Authorized edit scope",
            "src/util.py",
            "Anchor the change",
            "the contract wins",
        )
    )
    assert group.count("Line: ") == 2
    singleton = next(prompt for prompt in backend.prompts if "Incorrect test expectation" in prompt)
    assert "This test covers daydream/app.py — read it to understand the expected behavior." in singleton
    assert "tests/test_app.py:10" in singleton
    assert all(
        "Authorized edit scope" in prompt and "Forbid working-tree or index git mutation" in prompt
        for prompt in backend.prompts
    )
    assert all(
        token in group for token in ("`git add`", "`git stash`", "`git checkout`", "`git reset`", "`git commit`")
    )
    assert (repo / "src/handler.py").read_text() == (repo / "tests/test_app.py").read_text() == "VALUE = 1\n"
    assert (repo / "src/util.py").read_text() == before["src/util.py"]
    assert (repo / "daydream/app.py").read_text() == before["daydream/app.py"]
    traces = [
        json.loads(path.read_text())
        for path in siblings_directory(run_directory(repo / ".daydream", recorder.session_id)).glob("*.json")
    ]
    assert len(traces) == 3
    messages = [step.get("message", "") for trace in traces for step in trace["steps"]]
    assert any("Off-by-one in loop bound" in str(message) for message in messages)
    assert any("Incorrect test expectation" in str(message) for message in messages)
    assert any("Failed singleton" in str(message) for message in messages)


async def test_bound_phase_fix_transports_only_named_private_inputs(
    tmp_path: Path, make_work: Callable[..., WorkContext], _quiet_phase_ui: None,
) -> None:
    """A production fix gets the exploration index bytes inline without an artifact-dir grant."""

    repo = tmp_path / "repo"
    init_repo(repo)
    source_file = repo / "src" / "app.py"
    source_file.parent.mkdir()
    source_file.write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", ".")
    git_commit(repo, "base")
    work = make_work(repo)
    backend = _inline_or_exact_backend(repo, inline=True)

    async with _private_session(tmp_path, work, "phase-fix-inline"):
        deep = artifact_dir_for(repo, allow_standalone=True) / "deep"
        affected = deep / "exploration" / "affected_files.md"
        affected.parent.mkdir(parents=True)
        affected.write_text("src/app.py -> tests/test_app.py", encoding="utf-8")

        await phases.phase_fix(backend, work, {"id": 1, "description": "repair", "file": "src/app.py", "line": 1}, 1, 1,
            exploration_dir=affected.parent,
            edit_scope=frozenset({"src/app.py"}), read_scope=frozenset({"src/app.py"}),
        )

        prompt = backend.last_prompt
        pointer_free = prompt.replace(str(affected), "")
        assert str(affected.parent) not in pointer_free
        assert "src/app.py -> tests/test_app.py" in prompt
        assert str(affected) not in prompt

        assert "\nLine: 1" in prompt
        # An exploration dir without its index must not reach the fix prompt; a finding without a line names none.
        affected.unlink()
        lineless = {"id": 1, "description": "repair", "file": "src/app.py", "line": None}
        await phases.phase_fix(backend, work, lineless, 1, 1,
            exploration_dir=affected.parent,
            edit_scope=frozenset({"src/app.py"}), read_scope=frozenset({"src/app.py"}),
        )
        assert "affected_files.md" not in backend.last_prompt
        assert "Line:" not in backend.last_prompt and "\nFile: " in backend.last_prompt


@pytest.mark.asyncio
async def test_fix_entrypoints_reject_invalid_finding_file_refs(
    git_repo: Path, make_work: Callable[..., WorkContext], _quiet_phase_ui: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every fix entry point rejects unconfined or missing files; turn wall cuts follow the group budget.

    Batched cases put the bad reference second to exercise the entire preflight loop.
    """
    backend = ScriptedBackend()
    valid = [
        {"id": 1, "item_uid": "item:good", "description": "Confined", "file": "src/ok.py", "line": 1},
        {"id": 99, "item_uid": "item:bad", "file": "src/bad.py"},
    ]
    footprint = AuthorizedFixFootprint.build(git_repo, set(), valid)
    snapshot = _round_snapshot(git_repo, footprint.run_allowed_paths)
    work = make_work(git_repo)
    cases = [("phase_fix", "traversal"), ("phase_fix_batched", "traversal"),
             *[("phase_fix_parallel", kind) for kind in ("traversal", "absolute", "symlink", "missing")]]
    for entry_point, bad_ref in cases:
        # Prepare valid authority first; the invalid finding must reach phase preflight.
        bad: dict[str, Any] = {"id": 99, "item_uid": "item:bad", "description": "Escape", "line": 1}
        if bad_ref != "missing":
            bad["file"] = _unconfined_finding_file(git_repo, bad_ref)
        items = [valid[0], bad]
        call: Any
        if entry_point == "phase_fix":
            call = phases.phase_fix(backend, work, bad, 1, 1,
                edit_scope=footprint.run_allowed_paths, read_scope=footprint.run_allowed_paths,
            )
        elif entry_point == "phase_fix_batched":
            call = phases.phase_fix_batched(backend, work, items, [1, 2], 2,
                edit_scope=footprint.run_allowed_paths, read_scope=footprint.run_allowed_paths,
            )
        else:
            call = phases.phase_fix_parallel(backend, work, items, footprint=footprint, round_snapshot=snapshot)

        with pytest.raises(ValueError, match="Finding file must be a confined repository-relative path"):
            await call
    assert backend.prompts == []

    # A per-turn wall cut is absorbed while a longer group budget remains; at the
    # group's own ceiling it stops the group with a budget failure instead.
    from daydream.config import DEFAULT_WALL_BUDGET_S
    from daydream.phases import fix as fixing

    async def wall_cut(*_a: Any, **_k: Any) -> str:
        return "wall_budget_exceeded"

    monkeypatch.setattr(fixing, "phase_fix", wall_cut)
    confined = [valid[0]]
    for group_wall, expected in ((DEFAULT_WALL_BUDGET_S + 600, {}),
                                 (DEFAULT_WALL_BUDGET_S, {"src/ok.py": "file_group_budget_exceeded: "
                                                                       "group_wall_budget_exceeded"})):
        failures = await phases.phase_fix_parallel(
            backend, work, confined, footprint=footprint, round_snapshot=_round_snapshot(
                git_repo, footprint.run_allowed_paths), group_max_wall_s=group_wall)
        assert failures == expected


@pytest.mark.asyncio
async def test_fix_verify_keeps_one_cleaned_outcome_per_finding_and_carries_check_claims(
    git_repo: Path, make_work: Callable[..., WorkContext], _quiet_phase_ui: None,
) -> None:
    """Verifier output is normalized per dispatched finding; check fields survive to host substantiation.

    Repeat-round prompts then carry the prior verdict and reason back to the fixer.
    """
    from daydream.phases.fix import _build_verifier_suffix

    items = [{"id": 1, "item_uid": "item:1", "file": "src/a.py", "description": "a"},
             {"id": 2, "item_uid": "item:2", "file": "src/b.py", "description": "b"},
             {"id": 3, "item_uid": "item:3", "file": "src/c.py", "description": "c"}]
    backend = ScriptedBackend(script=[_structured_turn({"verdicts": [
        {"issue_id": 1, "verdict": "regressed", "reason": "returns the wrong value", "check_command": None,
         "check_only": False, "path": None},
        {"issue_id": 2, "verdict": "wrong_target", "reason": "edit belongs elsewhere", "path": " src/other.py "},
        {"issue_id": "x", "verdict": "resolved", "reason": "ignored"}, {"issue_id": 3, "verdict": "bogus"},
    ]})])
    verdicts = await phases.phase_fix_verify(backend, make_work(git_repo), items, "", round_number=2)
    assert verdicts == [
        {"issue_id": 1, "verdict": "regressed", "reason": "returns the wrong value", "check_command": None,
         "check_only": False},
        {"issue_id": 2, "verdict": "wrong_target", "reason": "edit belongs elsewhere", "path": "src/other.py"},
        {"issue_id": 3, "verdict": "unresolved", "reason": "no verifier verdict"},
    ]
    assert _build_verifier_suffix({"fix_verify_verdict": "regressed", "fix_verify_reason": "still red"}) == (
        "\nPrevious round fix-verify verdict: regressed. Verifier reason: still red. "
        "Re-address this finding accordingly.")
    assert _build_verifier_suffix({}) == ""










# phase_commit_push — declined gate still validates applied fixes (issue #726)


def _init_plain_repo(tmp_path: Path) -> Path:
    """Minimal real git repo for decline-path tests (no commit is made)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    return repo


@pytest.mark.asyncio
async def test_declined_commit_requires_green_configured_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    make_config: Callable[..., Any], _quiet_phase_ui: None,
) -> None:
    """A declined commit cannot fabricate a validation verdict from red evidence or a missing command."""

    monkeypatch.setattr("daydream.run_context.RunContext.confirm", lambda self, **k: False)
    _record_host_runs(monkeypatch)
    red_repo = _init_plain_repo(tmp_path / "red")
    red_config = make_config(red_repo, test_command="false")
    recipe = resolve_test_recipe(red_config, red_config, repo_root=red_repo)
    identity = _execution_identity(red_repo, recipe=recipe, argv=("false",), outcome="failed")
    with pytest.raises(RuntimeError, match="validation"):
        await phase_commit_push(ScriptedBackend(), make_work(red_repo), config=red_config, recipe=recipe,
            evidence=TestAttemptEvidence(session_id="s", kind="host", command=("false",), passed=False,
                input_tree_key=identity.output_tree_key, output_tree_key=identity.output_tree_key, identity=identity,
            ), retained_tree_key=identity.output_tree_key,
        )

    async def fake_run(*a: Any, **k: Any) -> None:
        raise AssertionError("run_test_command must not be called without a command")

    monkeypatch.setattr("daydream.phases.test_evidence.run_test_command", fake_run)

    repo = _init_plain_repo(tmp_path)
    work = make_work(repo)
    config = make_config(tmp_path, test_command=None)
    with pytest.raises(RuntimeError, match="--test-command"):
        await phase_commit_push(ScriptedBackend(), work, config=config)


# phase_test_and_heal — option 1 setup-investigator wiring


@pytest.mark.asyncio
async def test_approved_investigator_command_stays_host_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    """Approval runs quoted argv once host-side; agent prompts retain the test-output contract."""
    argv = [sys.executable, "-c", "print('ok')"]
    command = shlex.join(argv)
    reason = "The documented command runs a Python check"
    backend = ScriptedBackend(script=[_structured_turn(_verdict("replace", command, reason))])
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "1" if "Choice" in a[1] else "y")
    config = _host_suite_config(tmp_path, mode="all")

    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), config=config, feedback_items=None, allow_standalone=True,
        session_id="healing-menu", capture_tree_key=lambda: git_ops.tree_key(()),
        footprint=_footprint(tmp_path),
    )

    assert result.passed is True
    assert result.retries == 0
    assert result.proceed is True
    assert [attempt.command for attempt in result.attempts] == [tuple(shlex.split(config.test_command)), tuple(argv)]
    assert [attempt.passed for attempt in result.attempts] == [False, True]
    assert result.attempts[-1].identity is not None
    assert result.attempts[-1].identity.cwd_relative == "."
    assert len(backend.prompts) == 1
    assert all("print('ok')" not in prompt for prompt in backend.prompts)
    assert "Run this exact test command" not in "\n".join(backend.prompts)
    assert "read-only setup-investigator" in backend.prompts[0]


@pytest.mark.asyncio
async def test_approved_investigator_backtick_only_command_is_skipped_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    """A suggestion that sanitizes to empty argv is skipped with a warning."""


    backend = ScriptedBackend(script=[_structured_turn(_verdict("replace", "```", "verdict reason"))])

    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "1" if "Choice" in a[1] else "y")
    config = _host_suite_config(tmp_path)

    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), config=config, feedback_items=None, allow_standalone=True,
        session_id="healing-menu", capture_tree_key=lambda: git_ops.tree_key(()),
        footprint=_footprint(tmp_path),
    )

    assert result.passed is True
    assert result.retries == 1
    assert result.proceed is True
    # Only the original suite executes; the empty approved suggestion never gets argv.
    assert [attempt.command for attempt in result.attempts] == [tuple(shlex.split(config.test_command))] * 2
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_unreplaced_test_command_retries_original_host_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    """A correct, declined, or failed investigation preserves the original test command."""
    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr("daydream.ui.print_warning", lambda _console, message: warnings.append(message))
    monkeypatch.setattr("daydream.ui.print_info", lambda _console, message: infos.append(message))
    for investigation in ("correct", "declined", "failed"):
        repo = tmp_path / investigation
        repo.mkdir()
        verdict = _verdict("correct", None, "make test is the canonical target")
        if investigation == "declined":
            verdict = _verdict("replace", "make check", "Makefile defines check")
        turn = (RuntimeError("scripted investigator failure"),) if investigation == "failed" else _structured_turn(
            verdict)
        backend = ScriptedBackend(script=[turn])

        def prompt(*args: Any, investigation: str = investigation, **_kwargs: Any) -> str:
            if "Choice" in args[1]:
                return "1"
            assert investigation == "declined"
            assert any("Suggested command:" in message and "make check" in message for message in infos), infos
            return "n"
        monkeypatch.setattr("daydream.run_context._prompt_user", prompt)
        config = _host_suite_config(repo)
        result = await phases.phase_test_and_heal(
            backend, make_work(repo), config=config, allow_standalone=True,
            session_id="healing-menu", capture_tree_key=lambda: git_ops.tree_key(()),
            footprint=_footprint(repo),
        )
        assert result.passed is True
        assert result.retries == 1
        assert [attempt.command for attempt in result.attempts] == [tuple(shlex.split(config.test_command))] * 2
        assert [attempt.passed for attempt in result.attempts] == [False, True]
        assert len(backend.prompts) == 1
        assert "read-only setup-investigator" in backend.prompts[0]
        assert backend.read_only_calls == [True]
    assert any("Setup investigator failed" in message for message in warnings), warnings


# phase_test_and_heal — option 4 failure-summarizer + handoff


def _install_recorder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, on_write: Any=None) -> Any:
    """Install a recorder/path fixture and no-op forks, with on_write indicating archival."""
    class _FakeRecorder:
        target_dir = tmp_path
        session_id = "test-session-id"
        partial_writes = 0

        def __init__(self) -> None:
            self.on_write = on_write

        def write_partial(self) -> None:
            self.partial_writes += 1

    fake = _FakeRecorder()
    monkeypatch.setattr("daydream.phases.handoff.get_current_recorder", lambda: fake)

    @asynccontextmanager
    async def _noop_fork(recorder: Any, descriptor: Any) -> AsyncIterator[Any]:
        yield

    monkeypatch.setattr("daydream.phases.handoff.maybe_fork", _noop_fork)
    return fake


async def _run_option4_handoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_work: Callable[..., WorkContext],
    turn: Sequence[AgentEvent | BaseException], *, recorder: bool = True, clipboard: bool = False,
    prompt_fn: Callable[..., Any] | None = None, prepare: Callable[[ScriptedBackend, Any], None] | None = None,
) -> tuple[ScriptedBackend, bool, int]:
    """Run choice 4 after optional preparation; return backend, success, and retry count."""
    fake_recorder = _install_recorder(monkeypatch, tmp_path) if recorder else None
    if not recorder:
        monkeypatch.setattr("daydream.phases.handoff.get_current_recorder", lambda: None)
    monkeypatch.setattr("daydream.phases.handoff.clipboard_available", lambda: clipboard)
    monkeypatch.setattr(
        "daydream.run_context._prompt_user", prompt_fn if prompt_fn is not None else (lambda *a, **k: "4"),
    )
    backend = ScriptedBackend(script=[turn])
    if prepare is not None:
        prepare(backend, fake_recorder)
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), config=_host_suite_config(tmp_path, mode="all"), allow_standalone=True,
        session_id="healing-menu", capture_tree_key=lambda: git_ops.tree_key(()),
        footprint=_footprint(tmp_path),
    )
    return backend, result.passed, result.retries


@pytest.mark.asyncio
async def test_phase_test_and_heal_option4_clipboard_offer_and_summarizer_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:

    copied: list[str] = []
    def _copy_to_clipboard(text: Any) -> bool:
        copied.append(text)
        return True

    monkeypatch.setattr("daydream.phases.handoff.copy_to_clipboard", _copy_to_clipboard)

    # Abort at the menu, then approve copying the handoff.
    backend, success, _ = await _run_option4_handoff(
        monkeypatch, tmp_path, make_work, _handoff_turn("BODY"), clipboard=True,
        prompt_fn=lambda *a, **kw: "4" if "Choice" in a[1] else "y",
    )

    assert success is False
    assert copied == ["BODY"]
    assert "## Files readable in the current model workspace\n(none detected)" in backend.last_prompt
    warnings: list[str] = []
    monkeypatch.setattr("daydream.ui.print_warning", lambda _console, message: warnings.append(message))
    monkeypatch.setattr("daydream.phases.handoff.copy_to_clipboard", lambda _text: False)
    await _run_option4_handoff(monkeypatch, tmp_path, make_work, _handoff_turn("BODY"), clipboard=True,
                               prompt_fn=lambda *a, **kw: "4" if "Choice" in a[1] else "y")
    assert any("Clipboard copy failed; copy manually from path above" in message for message in warnings)

    # A failed summarizer still writes a minimal handoff quoting the ground truth.
    _, success, _ = await _run_option4_handoff(
        monkeypatch, tmp_path, make_work, (RuntimeError("scripted summarizer failure"),))
    assert success is False
    body = (tmp_path / ".daydream" / "runs" / "test-session-id" / "handoff.md").read_text(encoding="utf-8")
    for expected in ("# Daydream handoff", "Instructions for the next agent", "```", "## Verified facts",
                     "## Hypotheses (unverified)", "1 failed, 0 passed"):
        assert expected in body
    # The unknown-cause statement lives under Hypotheses, not Verified facts.
    facts_section, hypotheses = body.split("## Hypotheses (unverified)")
    assert "unknown" not in facts_section.lower() and "unknown" in hypotheses.lower()


# _resolve_handoff_paths — ephemeral worktree + archive routing


@pytest.mark.asyncio
async def test_standalone_ephemeral_handoff_survives_worktree_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standalone handoff belongs to the source even when archive storage is unavailable."""

    source = tmp_path / "source"
    init_repo(source)
    (source / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    git(source, "add", "tracked.py")
    head = git_commit(source, "base")
    worktree = tmp_path / "ephemeral-worktree"
    git(source, "worktree", "add", "--detach", str(worktree), head)
    work = WorkContext(repo=worktree, source=source, base_branch="main", base_sha=head, head_branch=None, head_sha=head,
        is_ephemeral=True, run_id="20260101000000-deadbeef",
    )
    unavailable = tmp_path / "unavailable-archive"
    unavailable.touch()
    monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(unavailable))
    recorder = make_recorder(worktree, on_write=lambda *_args: None)
    backend = ScriptedBackend(events=_handoff_turn(""))

    try:
        async with recorder:
            body, handoff_path, written = await _run_failure_summarizer(
                backend, work, "1 failed, 0 passed", allow_standalone=True,
            )
        git(source, "worktree", "remove", "--force", str(worktree))

        assert written is True
        expected = source / ".daydream"
        assert handoff_path.parent == expected / "runs" / recorder.session_id
        assert handoff_path.read_text(encoding="utf-8") == body
        assert "trajectory unavailable for this run" in body
        assert str(unavailable) not in body
        assert str(worktree) not in body
    finally:
        if worktree.exists():
            git(source, "worktree", "remove", "--force", str(worktree))


@pytest.mark.asyncio
async def test_recorderless_handoff_rejects_a_bound_session_without_explicit_ownership(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """Compatibility routing cannot borrow a private session for a durable handoff."""
    repo = tmp_path / "repo"
    init_repo(repo)
    work = make_work(repo)
    async with _private_session(tmp_path, work, "missing-handoff-session"):
        with pytest.raises(ArtifactVisibilityError, match="explicit artifact session"):
            _resolve_handoff_paths(None, work, allow_standalone=True)


# _write_handoff — must report write failure so the caller can fall back


@pytest.mark.asyncio
async def test_phase_test_and_heal_option4_inlines_body_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    _quiet_phase_ui: None,
) -> None:
    real_write = Path.write_text
    failed_paths: list[Path] = []

    def fail_handoff_write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == "handoff.md":
            failed_paths.append(path)
            raise OSError("disk full")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_handoff_write)

    printed: list[str] = []
    monkeypatch.setattr(
        "daydream.agent.console.print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
    )
    warnings: list[str] = []
    monkeypatch.setattr("daydream.ui.print_warning",
        lambda console_arg, message: warnings.append(message),  # noqa
    )

    _, success, _ = await _run_option4_handoff(monkeypatch, tmp_path, make_work,
        _handoff_turn("FULL_BODY_LINE_1\nFULL_BODY_LINE_2"),
    )

    assert success is False
    assert failed_paths and all(not path.exists() for path in failed_paths)
    # A warning explaining the failure was emitted.
    assert any("Failed to write handoff" in m for m in warnings), warnings
    # Write failure prints the complete body instead of a preview.
    assert any("FULL_BODY_LINE_1" in line for line in printed), printed
    assert any("FULL_BODY_LINE_2" in line for line in printed), printed



# _changed_files — untracked files must appear in the handoff change list


def test_changed_files_skips_unsafe_lexical_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Each unsafe Git name is warned about and skipped without escaping."""

    repo = tmp_path / "repo"
    names = [
        "safe.py", "", "/absolute.py", ".", "./nested.py", "nested/../escape.py", "../escape.py", "nested//empty.py",
    ]
    monkeypatch.setattr("daydream.git_ops.changed_files", lambda _repo: names)

    with caplog.at_level("WARNING", logger="daydream.phases"):
        paths = _changed_files(repo)

    assert paths == [repo / "safe.py"]
    assert sum("unsafe changed-file name" in record.message for record in caplog.records) == 7


# _run_failure_summarizer — writes a partial trajectory snapshot pre-exit


@pytest.mark.asyncio
async def test_failure_summarizer_falls_back_for_non_live_private_runtime_paths(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, make_work: Callable[..., WorkContext],
) -> None:
    """A model cannot echo private control or sibling-workspace identities."""

    repo = tmp_path / "repo"
    init_repo(repo)
    work = make_work(repo)
    async with _private_session(tmp_path, work, "handoff-private-roots") as session:
        session.register_destination(session.layout.public_daydream_dir, label=OutputLabel.PUBLIC_DAYDREAM,)
        control = session.layout.artifact_runtime_root / "control.json"
        leaked = session.layout.operational_workspaces_root / "sibling-run/log.txt"
        with caplog.at_level("WARNING", logger="daydream.phases"):
            control_body, _, _ = await _run_failure_summarizer(
                ScriptedBackend(events=_handoff_turn(f"# Handoff\n\n{control}\n")), work, "1 failed",
                artifact_session=session,
            )
            body, handoff_path, written = await _run_failure_summarizer(
                ScriptedBackend(events=_handoff_turn(f"# Handoff\n\n{leaked}\n")), work, "1 failed",
                artifact_session=session,
            )
        assert str(control) not in control_body and "structured handoff" in control_body
        # A failed summarizer falls back to the test output, unless it names a private path too.
        fallback, _, _ = await _run_failure_summarizer(
            ScriptedBackend(script=[(RuntimeError("summarizer down"),)]), work, f"1 failed at {leaked}",
            artifact_session=session,
        )
        assert str(leaked) not in fallback
        assert "Test output omitted because it contained a private runtime path." in fallback

    assert written is True
    assert handoff_path.parent == repo / ".daydream"
    assert handoff_path.name.startswith("handoff-")
    assert str(leaked) not in body
    assert "structured handoff" in body
    assert "private runtime path" not in body
    assert "output contained a private runtime path" in caplog.text


# The host renders review-output.md from validated merge items.






@pytest.mark.asyncio
async def test_phase_test_and_heal_records_each_host_attempt_and_heal_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
) -> None:

    feedback = [{"id": 1, "item_uid": "item:1", "file": "a.py"}]
    footprint = AuthorizedFixFootprint.build(tmp_path, {"readme.md"}, feedback)
    backend = ScriptedBackend(script=[_FIX_TURN])
    setattr(backend, "concise_fix_prompts", True)
    config = SimpleNamespace(file_config=None, test_command=host_test_command(
        tmp_path.parent / f"{tmp_path.name}-host-suite", mode="first",
        output="\n".join(f"line {i}" for i in range(500)),
    ))
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *args, **kwargs: "2")
    # Two keys per test attempt plus three per repair turn: the tree the repair
    # started from, the tree its checkpoint captured before any restoration, and
    # the tree it left behind after confinement.
    keys = iter(["in-1", "out-1", "fix-in-1", "fix-captured-1", "fix-out-1", "in-2", "out-2",])
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), feedback_items=feedback, config=config,
        session_id="session-2",
        capture_tree_key=lambda: next(keys), footprint=footprint, allow_standalone=True,
    )

    assert result.passed is True
    assert result.ignored is False
    assert [(a.input_tree_key, a.output_tree_key) for a in result.attempts] == [("in-1", "out-1"), ("in-2", "out-2",),]
    assert all(a.kind == "host" and a.command is not None for a in result.attempts)
    assert [a.passed for a in result.attempts] == [False, True]
    assert [(r.input_tree_key, r.output_tree_key) for r in result.repairs] == [("fix-in-1", "fix-out-1"),]
    # Without Git there is no candidate patch to capture: the capture is skipped
    # and named, exactly as the generated-file guard fails open in the same mode.
    read = read_repair_checkpoint(tmp_path / ".daydream" / "deep")
    assert (read.checkpoint, read.blocked) == (None, False)
    assert any("checkpoint_capture_unavailable" in diagnostic
        for diagnostic in result.repairs[0].diagnostics)
    heal_prompt = backend.prompts[0]
    assert "Authorized edit scope" in heal_prompt
    assert "a.py" in heal_prompt and "readme.md" in heal_prompt

    assert "tail of the test output (truncated to its last 100 lines" in heal_prompt
    assert "line 400\n" in heal_prompt and "line 499" in heal_prompt
    assert "line 399\n" not in heal_prompt and "line 0\n" not in heal_prompt
    assert "- a.py" in heal_prompt and str(tmp_path / "a.py") not in heal_prompt
    assert "ONLY these repository-relative paths" in heal_prompt and "another file" not in heal_prompt
    assert all(value in heal_prompt for value in ("CONCISE MODE", "Apply the fix directly",
                                                  "Output only the tool calls needed to apply the fix"))

    # An infrastructure failure cannot be healed by a fix turn: abort before the heal gate.
    environmental = SimpleNamespace(file_config=None, test_command=host_test_command(
        tmp_path.parent / f"{tmp_path.name}-env-suite", mode="all", output="psycopg: connection refused"))
    idle = ScriptedBackend()
    result = await phases.phase_test_and_heal(
        idle, make_work(tmp_path), config=environmental, session_id="session-env",
        capture_tree_key=lambda: "unchanged", footprint=footprint, allow_standalone=True,
    )
    assert (result.passed, result.retries, len(result.attempts)) == (False, 0, 1)
    assert idle.prompts == []
    # A configured wall budget kills a hung suite; the timeout is recorded, not read as a failure.
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *args, **kwargs: "3")
    from daydream.config_file import DaydreamFileConfig
    hung = SimpleNamespace(file_config=DaydreamFileConfig(test_command_wall_s=0.5),
                           test_command=shlex.join([sys.executable, "-c", "import time; time.sleep(30)"]))
    result = await phases.phase_test_and_heal(
        idle, make_work(tmp_path), config=hung, session_id="session-timeout",
        capture_tree_key=lambda: "unchanged", footprint=footprint, allow_standalone=True,
    )
    assert result.ignored is True and result.attempts[0].identity is not None
    assert result.attempts[0].identity.outcome == "timed-out"
    # Repair excerpts are bounded and silent output records none.
    from daydream.phases.testing import _repair_excerpt
    assert _repair_excerpt("   \n") == ()
    assert [len(excerpt) for excerpt in _repair_excerpt("x" * 5000)] == [2000]


@pytest.mark.asyncio
@pytest.mark.parametrize(("artifact_path", "tracked", "preexisting", "must_block"),
    [

        (".daydream/tracked.json", True, True, True),
    ],
)
async def test_strict_commit_real_hook_distinguishes_runtime_artifacts_from_user_files(
    tmp_path: Path, make_work: Callable[..., WorkContext], artifact_path: str, tracked: bool, preexisting: bool,
    must_block: bool,
) -> None:
    """A real hook may update runtime output, never tracked or lookalike user files."""

    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / "app.py").write_text("before\n")
    artifact = repo / artifact_path
    if preexisting:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("before runtime\n")
    git(repo, "add", "app.py")
    if tracked:
        git(repo, "add", artifact_path)
    git_commit(repo, "baseline")
    initial_index = phases.require_empty_staged_index(make_work(repo))
    (repo / "app.py").write_text("after\n")
    retained_states = git_ops.snapshot_worktree_paths(repo, ["app.py"])
    hook = repo / ".git" / "hooks" / "post-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "mkdir -p .daydream .daydreamish\n"
        f"printf '%s\\n' 'hook runtime' > '{artifact_path}'\n"
    )
    hook.chmod(0o755)

    async def commit_retained() -> phases.CommitPushResult:
        return await publish._do_commit(
            ScriptedBackend(), make_work(repo), retained_paths=frozenset({"app.py"}), retained_states=retained_states,
            initial_index=initial_index,
         config=_hook_run_config(),)

    if must_block:
        with pytest.raises(git_ops.GitError, match="push blocked"):
            await commit_retained()
    else:
        assert (await commit_retained()).committed is True
    assert artifact.read_text() == "hook runtime\n"
    assert git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD") == "app.py"
    assert git(repo, "diff", "--cached") == ""


# --- Issue #172 Fix B extended: inline small diffs into intent / wonder ------


def test_registered_fix_prompt_keeps_relative_targets_without_repository() -> None:
    """Extensions may use the built-in builder's default repo without granting an absolute target."""
    registry = Registry()
    register_builtins(registry)
    prompt = registry.prompt("fix")("failure", [{"file": "src/a.py", "evidence": "src/a.py:3"}], concise_mode=True)
    assert "Finding target files (not a diff of what changed):\n- src/a.py" in prompt
    assert "Evidence exemplars:\n- src/a.py:3" in prompt
    assert "CONCISE MODE" in prompt and "Apply the fix directly" in prompt


def _reuse_offer(identity: TestExecutionIdentity) -> TestAttemptEvidence:
    """A matching green host offer bound to *identity*'s output tree key."""
    return TestAttemptEvidence(session_id="s", kind="host", command=("true",), passed=True,
        input_tree_key=identity.output_tree_key, output_tree_key=identity.output_tree_key, identity=identity,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_mutation", ["index"])
async def test_matching_evidence_cannot_bypass_failed_commit_verification(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch, hook_mutation: str,
) -> None:
    """Content-only tree identity cannot replace explicit post-commit verification."""
    repo = _pushable_repo(tmp_path)
    _install_pre_push_hook(repo)
    (repo / "fix.py").write_text("fixed\n")
    runs = _record_host_runs(monkeypatch)
    config = _hook_run_config()
    recipe = resolve_test_recipe(config, config, repo_root=repo)
    identity = _execution_identity(repo, recipe=recipe, argv=("true",))

    # A post-commit mutation invalidates an otherwise matching evidence offer.
    hook = repo / ".git" / "hooks" / "post-commit"
    hook.write_text("#!/bin/sh\nprintf 'hook mutation\n' > fix.py\n" + (
        "git add fix.py\nprintf 'fixed\n' > fix.py\n" if hook_mutation == "index" else ""))
    hook.chmod(0o755)
    with pytest.raises(GitError, match="post-commit validation failed; push blocked"):
        await _do_commit(
            ScriptedBackend(), make_work(repo), push=True, interactive=False,
            items=[{"file": "fix.py", "description": "fix bug"}], config=config, recipe=recipe,
            evidence=_reuse_offer(identity), retained_tree_key=identity.output_tree_key,
            **_retained_commit_tree(repo, {"fix.py"}),
        )
    assert runs == [], "failed commit verification must stop before reusing or rerunning tests"
    assert git(repo, "ls-remote", "origin", "refs/heads/main") == ""
    assert git(repo, "show", "HEAD:fix.py") == "fixed"
    assert (repo / "fix.py").read_text() == ("fixed\n" if hook_mutation == "index" else "hook mutation\n")
    assert git(repo, "diff", "--cached", "--name-only") == ("fix.py" if hook_mutation == "index" else "")


def _capture_gate_report(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the human reuse line(s) the gate prints through the UI helpers."""
    reported: list[str] = []
    monkeypatch.setattr('daydream.ui.print_success', lambda _c, msg, *a, **k: reported.append(str(msg)))
    monkeypatch.setattr('daydream.ui.print_info', lambda _c, msg, *a, **k: reported.append(str(msg)))
    return reported


@pytest.mark.asyncio
async def test_reuse_decision_or_named_mismatch_is_reported_and_persisted(
    tmp_path: Path, make_work: Callable[..., WorkContext], make_config: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch, _quiet_phase_ui: None,
) -> None:
    monkeypatch.setattr("daydream.run_context.RunContext.confirm", lambda self, **k: False)
    runs = _record_host_runs(monkeypatch)
    reported = _capture_gate_report(monkeypatch)
    repo = _init_plain_repo(tmp_path)
    deep = repo / ".daydream" / "deep"
    deep.mkdir(parents=True)
    config = make_config(repo, test_command="true")
    recipe = resolve_test_recipe(config, config, repo_root=repo)
    identity = _execution_identity(repo, recipe=recipe, argv=("true",))

    result = await phase_commit_push(ScriptedBackend(), make_work(repo), config=config, recipe=recipe,
        evidence=_reuse_offer(identity), retained_tree_key=identity.output_tree_key,
    )

    assert result is None and runs == []
    assert git(repo, "diff", "--cached", "--name-only") == ""
    record = json.loads(DeepArtifact.EVIDENCE_REUSE.at(deep).read_text())
    gate = record["gates"]["declined-commit"]
    assert gate["gate"] == "declined-commit"
    assert gate["result"] == "reused"
    assert gate["reused"] is True
    assert gate["mismatched_components"] == []
    assert gate["before_head_sha"] == identity.head_sha
    assert any("reused" in line for line in reported), reported

    # A stale identity reruns the suite and names the mismatched component.
    reported.clear()
    await phase_commit_push(ScriptedBackend(), make_work(repo), config=config, recipe=recipe,
        evidence=TestAttemptEvidence(session_id="s", kind="host", command=("true",), passed=True,
            input_tree_key="stale", output_tree_key="stale",
            identity=_execution_identity(repo, recipe=recipe, argv=("true",), output_tree_key="stale"),
        ), retained_tree_key="stale",
    )

    assert len(runs) == 1 and runs[0]["cmd"] == ["true"]
    assert runs[0]["wall_budget_s"] == TEST_WALL_BUDGET_S
    record = json.loads(DeepArtifact.EVIDENCE_REUSE.at(deep).read_text())
    gate = record["gates"]["declined-commit"]
    assert gate["result"] == "identity-mismatch"
    assert gate["mismatched_components"] == ["tree_key"]
    assert any("tree_key" in line for line in reported), reported

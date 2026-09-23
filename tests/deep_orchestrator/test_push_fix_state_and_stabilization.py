"""Push Fix State And Stabilization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.archive.pipeline import derive_phase_states, derive_pipeline_status
from daydream.backends import ResultEvent
from daydream.deep.fix_steps import (
    EvidenceKey,
    RetainedTreeSnapshot,
    _authorize_final_red_override,
    _persist_push_verdict,
    _render_fix_outcome_summary,
    _resolve_remote_ci_target,
    _round_dispatch_items,
    _step_fix_gate,
    _step_remote_ci,
    _step_test,
    finalize_retained_tree_after_test,
    verify_retained_tree,
)
from daydream.deep.orchestrator import _remote_ci_enabled
from daydream.extensions.api import Stop
from daydream.git_ops import GitError
from daydream.phases import PushReceipt, TestAndHealResult, TestAttemptEvidence
from daydream.run_context import InteractionPolicy, RunContext
from tests.deep_orchestrator.support import (
    _base_repo,
    _direct_fix_context,
    _direct_fix_state,
    _finalization_fixture,
    _remote_identity_context,
)
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.test_deep_orchestrator import (
    _merge_item,
)


def test_push_verdict_is_current_session_and_exact_identity(tmp_path: Path) -> None:

    repo = _base_repo(tmp_path, "push-verdict")
    ctx = _direct_fix_context(repo, [], changed_files=set())
    _direct_fix_state(ctx, [], set())
    sha = git_ops.head_sha(repo)

    _persist_push_verdict(
        ctx,
        PushReceipt("origin", "feature", sha, "fork/project"),
        status="succeeded",
        started_at="2026-09-06T12:00:00Z",
    )

    payload = json.loads((ctx.data["dd"] / "push-verdict.json").read_text())
    assert payload == {
        "schema_version": 1,
        "session_id": "session-current",
        "status": "succeeded",
        "remote": "origin",
        "branch": "feature",
        "pushed_sha": sha,
        "pushed_repository": "fork/project",
        "started_at": "2026-09-06T12:00:00Z",
        "updated_at": payload["updated_at"],
    }
    assert payload["updated_at"].endswith("Z")

    _persist_push_verdict(
        ctx,
        PushReceipt("origin", "feature", sha, "fork/project"),
        status="failed",
        started_at="2026-09-06T12:01:00Z",
        diagnostic="token=top-secret push rejected",
    )
    failed = json.loads((ctx.data["dd"] / "push-verdict.json").read_text())
    assert failed["status"] == "failed"
    assert failed["pushed_sha"] == sha
    assert "top-secret" not in failed["diagnostic"]


@pytest.mark.anyio
async def test_successful_non_github_push_gets_unavailable_handoff(tmp_path: Path) -> None:

    repo = _base_repo(tmp_path, "non-github-push")
    ctx = _direct_fix_context(repo, [], changed_files=set())
    _direct_fix_state(ctx, [], set())
    ctx.data["push_receipt"] = PushReceipt("origin", "main", git_ops.head_sha(repo), None)

    assert _remote_ci_enabled(ctx) is True
    result = await _step_remote_ci(ctx)

    assert isinstance(result, Stop) and result.exit_code == 1
    verdict = json.loads((ctx.data["dd"] / "remote-ci-verdict.json").read_text())
    handoff = json.loads((ctx.data["dd"] / "remote-ci-handoff.json").read_text())
    assert verdict["session_id"] == "session-current"
    assert verdict["status"] == "unavailable"
    assert verdict["polling"]["poll_count"] == 0
    assert verdict["target"] is None
    assert handoff["status"] == "unavailable"
    assert handoff["target"] is None


@pytest.mark.parametrize(
    ("head_repository", "pushed_repository"),
    [
        ("base-user/project", "base-user/project"),
        ("fork-user/project", "fork-user/project"),
    ],
)
def test_remote_target_accepts_matching_same_repo_and_fork_identity(
    tmp_path: Path,
    fake_gh: Any,
    head_repository: str,
    pushed_repository: str,
) -> None:

    ctx, sha = _remote_identity_context(tmp_path, fake_gh, head_repository=head_repository)
    target = _resolve_remote_ci_target(ctx, PushReceipt("origin", "feature", sha, pushed_repository))

    assert target.base_repository == "base-user/project"
    assert target.head_repository == head_repository
    assert target.head_ref == "feature"
    assert target.pushed_sha == sha


@pytest.mark.parametrize(
    (
        "head_repository",
        "pushed_repository",
        "configured_repository",
        "base_ref",
        "configured_pr",
    ),
    [
        ("fork-user/project", "other-user/project", "base-user/project", "main", 7),
        (None, "fork-user/project", "base-user/project", "main", 7),
        ("fork-user/project", "fork-user/project", "wrong-user/project", "main", 7),
        ("fork-user/project", "fork-user/project", "base-user/project", "feature", 7),
        ("fork-user/project", "fork-user/project", "base-user/project", "main", 8),
    ],
)
def test_remote_target_rejects_untrusted_identity_combinations(
    tmp_path: Path,
    fake_gh: Any,
    head_repository: str | None,
    pushed_repository: str,
    configured_repository: str,
    base_ref: str,
    configured_pr: int,
) -> None:

    ctx, sha = _remote_identity_context(
        tmp_path,
        fake_gh,
        head_repository=head_repository,
        configured_repository=configured_repository,
        base_ref=base_ref,
        configured_pr=configured_pr,
    )
    with pytest.raises(GitError):
        _resolve_remote_ci_target(ctx, PushReceipt("origin", "feature", sha, pushed_repository))


async def test_fix_cycle_malformed_related_stops_before_backend_and_clears_stale_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start-at-fix preflight cannot inherit green evidence when policy parsing fails."""

    repo = _base_repo(tmp_path, "malformed-related")
    (repo / "a.py").write_text("A = 2\n")
    item = _merge_item(1, "a.py", "high")
    item["related_files"] = ["../outside.py"]
    ctx = _direct_fix_context(repo, [item], changed_files={"a.py"}, start_at="fix")
    ctx.run_context = RunContext(InteractionPolicy(assume="yes"))
    stale = ctx.data["dd"] / "test-verdict.json"
    stale.write_text(json.dumps({"session_id": "prior", "passed": True}))
    index_before = git_ops.snapshot_index(repo)
    bytes_before = (repo / "a.py").read_bytes()
    result = await _step_fix_gate(ctx)

    assert isinstance(result, Stop) and result.exit_code == 1
    assert not stale.exists()
    assert "fix_cycle_state" not in ctx.data
    assert ctx._backend_cache == {}
    assert git_ops.snapshot_index(repo) == index_before
    assert (repo / "a.py").read_bytes() == bytes_before


async def test_fix_cycle_nonempty_index_stops_before_backend_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    repo = _base_repo(tmp_path, "staged-preflight")
    (repo / "a.py").write_text("A = 2\n")
    _git(repo, "add", "a.py")
    item = _merge_item(1, "a.py", "high")
    ctx = _direct_fix_context(repo, [item], changed_files={"a.py"})
    ctx.run_context = RunContext(InteractionPolicy(assume="yes"))
    stale = ctx.data["dd"] / "test-verdict.json"
    stale_bytes = b'{"session_id":"prior","passed":true}\n'
    stale.write_bytes(stale_bytes)
    index_before = git_ops.snapshot_index(repo)
    bytes_before = (repo / "a.py").read_bytes()
    result = await _step_fix_gate(ctx)

    assert isinstance(result, Stop) and result.exit_code == 1
    assert "fix_cycle_state" not in ctx.data
    assert ctx._backend_cache == {}
    assert git_ops.snapshot_index(repo) == index_before
    assert (repo / "a.py").read_bytes() == bytes_before
    assert stale.read_bytes() == stale_bytes


def test_fix_cycle_round_two_rejects_cross_item_retarget(tmp_path: Path) -> None:

    repo = tmp_path / "cross-item-retarget"
    _init_repo(repo)
    for path in ("a.py", "b.py", "c.py"):
        (repo / path).write_text(f"# {path}\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    items = [
        {**_merge_item(1, "a.py", "high"), "item_uid": "item:a", "related_files": ["b.py"]},
        {**_merge_item(2, "c.py", "high"), "item_uid": "item:c", "related_files": []},
    ]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py", "c.py"})
    state = _direct_fix_state(ctx, items, {"a.py", "c.py"})
    ctx.data["iteration"] = 2
    ctx.data["fix_outcomes"] = {
        "item:a": {
            "issue_id": 1,
            "verdict": "wrong_target",
            "path": "c.py",
            "reason": "try the other item's file",
        }
    }

    dispatched = _round_dispatch_items(ctx, items)

    assert [item["item_uid"] for item in dispatched] == ["item:a"]
    assert dispatched[0]["file"] == "a.py"
    assert state.footprint.item_paths("item:a") == frozenset({"a.py", "b.py"})
    rejected = [event for event in state.footprint.events if event.action == "rejected_retarget"]
    assert len(rejected) == 1
    assert rejected[0].path == "c.py"
    assert rejected[0].round_number == 2


def test_fix_cycle_tracks_last_dispatched_target_after_accepted_then_rejected_retarget(
    tmp_path: Path,
) -> None:

    repo = tmp_path / "accepted-then-rejected-retarget"
    _init_repo(repo)
    for path in ("a.py", "b.py", "c.py"):
        (repo / path).write_text(f"# {path}\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    items = [
        {**_merge_item(1, "a.py", "high"), "item_uid": "item:a", "related_files": ["b.py"]},
        {**_merge_item(2, "c.py", "high"), "item_uid": "item:c", "related_files": []},
    ]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py", "c.py"})
    state = _direct_fix_state(ctx, items, {"a.py", "c.py"})

    ctx.data["iteration"] = 1
    assert [item["file"] for item in _round_dispatch_items(ctx, items)] == ["a.py", "c.py"]
    assert state.last_fix_target_by_uid == {"item:a": "a.py", "item:c": "c.py"}

    ctx.data["iteration"] = 2
    ctx.data["fix_outcomes"] = {"item:a": {"issue_id": 1, "verdict": "wrong_target", "path": "b.py", "reason": "moved"}}
    assert [item["file"] for item in _round_dispatch_items(ctx, items)] == ["b.py"]
    assert state.last_fix_target_by_uid["item:a"] == "b.py"

    ctx.data["iteration"] = 3
    ctx.data["fix_outcomes"] = {
        "item:a": {
            "issue_id": 1,
            "verdict": "wrong_target",
            "path": "c.py",
            "reason": "other item",
        }
    }
    assert [item["file"] for item in _round_dispatch_items(ctx, items)] == ["a.py"]
    assert state.last_fix_target_by_uid["item:a"] == "a.py"


async def test_resolved_verdict_uses_last_dispatched_target_not_raw_verifier_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    repo = tmp_path / "resolved-target-provenance"
    _init_repo(repo)
    for path in ("a.py", "b.py", "untrusted.py"):
        (repo / path).write_text(f"# {path}\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    item = {
        **_merge_item(1, "a.py", "high"),
        "item_uid": "item:a",
        "related_files": ["b.py"],
    }
    ctx = _direct_fix_context(repo, [item], changed_files={"a.py"})
    state = _direct_fix_state(ctx, [item], {"a.py"})
    state.last_fix_target_by_uid["item:a"] = "b.py"
    snapshot = RetainedTreeSnapshot(
        paths=frozenset(),
        states=(),
        tree_key="tree",
        verifier_patch="patch",
        recommended_patch=b"patch",
    )

    backend = ScriptedBackend(
        events=[
            ResultEvent(
                structured_output={
                    "verdicts": [
                        {
                            "issue_id": 1,
                            "verdict": "resolved",
                            "path": "untrusted.py",
                            "reason": "model candidate must not become provenance",
                        }
                    ]
                },
                continuation=None,
            )
        ]
    )
    monkeypatch.setattr("daydream.runner._resolve_backend", lambda *_a, **_k: backend)

    outcomes = await verify_retained_tree(ctx, snapshot, [item], pass_number=2)

    assert backend.call_count == 1
    assert backend.read_only_calls == [True]
    assert outcomes["item:a"]["path"] == "b.py"


@pytest.mark.parametrize("prior_verdict", [None, "resolved", "unresolved", "wrong_target"])
@pytest.mark.parametrize("final_verdict", ["unresolved", "wrong_target", "regressed"])
async def test_post_heal_actionable_verifier_stops_without_test_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    prior_verdict: str | None, final_verdict: str,
) -> None:

    ctx, state, snapshot = _finalization_fixture(tmp_path)
    ctx.data["fix_outcomes"] = (
        {"item:a": {"issue_id": 1, "verdict": prior_verdict}}
        if prior_verdict else {}
    )
    state.verifier_key = EvidenceKey("prior-tree", state.footprint.policy_revision)
    evidence = TestAttemptEvidence(
        session_id=state.session_id,
        kind="host",
        command=("pytest",),
        passed=True,
        input_tree_key=snapshot.tree_key,
        output_tree_key=snapshot.tree_key,
    )
    calls = {"verify": 0, "test": 0}

    monkeypatch.setattr("daydream.deep.fix_steps._strict_scope_and_scrub", lambda *_a, **_k: False)
    monkeypatch.setattr("daydream.deep.fix_steps.capture_retained_tree", lambda *_a, **_k: snapshot)

    async def _actionable(*_a: Any, **_k: Any) -> dict[str, dict[str, Any]]:
        calls["verify"] += 1
        return {"item:a": {"issue_id": 1, "verdict": final_verdict, "reason": "still broken"}}

    async def _no_test(*_a: Any, **_k: Any) -> Any:
        calls["test"] += 1
        raise AssertionError("actionable verifier must stop before a no-heal test")

    monkeypatch.setattr("daydream.deep.fix_steps.verify_retained_tree", _actionable)
    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_once", _no_test)

    result = await finalize_retained_tree_after_test(
        ctx,
        TestAndHealResult(True, 0, True, False, (evidence,)),
    )

    if prior_verdict in {"unresolved", "wrong_target"} and final_verdict != "regressed":
        assert result is None
        assert calls == {"verify": 1, "test": 0}
        assert state.latest_retained == snapshot
        assert ctx.data["fix_outcomes"]["item:a"]["verdict"] == final_verdict
        return
    assert isinstance(result, Stop) and result.exit_code == 1
    assert calls == {"verify": 1, "test": 0}
    assert state.latest_retained is None
    failure = json.loads((ctx.data["dd"] / "stabilization-failed.json").read_text())
    assert failure["session_id"] == state.session_id
    assert "actionable" in failure["reason"]

    phase_states = derive_phase_states(ctx.work.repo, phase_events=[], session_id=state.session_id)
    assert phase_states["fix"]["status"] == "failed"
    assert phase_states["test"]["status"] == "failed"
    assert derive_pipeline_status("complete", None, phase_states, runs_fix=True, runs_test=True) == "failed"


@pytest.mark.parametrize("terminal_mode", ["red", "exception"])
async def test_terminal_red_after_heal_restores_unrelated_and_protected_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_mode: str
) -> None:
    """A healer's red/exception exit is confined like a successful path."""

    repo = tmp_path / "red-heal-confinement"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    (repo / "unrelated.py").write_text("OWNER = 1\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    scratch = repo / "scratch.bin"
    scratch.write_bytes(b"\x00owner")
    items = [{**_merge_item(1, "a.py", "high"), "item_uid": "item:a"}]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py"})
    state = _direct_fix_state(ctx, items, {"a.py"})

    async def _red_after_mutation(*_a: Any, **_k: Any) -> TestAndHealResult:
        (repo / "a.py").write_text("A = 2\n")
        (repo / "unrelated.py").write_text("OWNER = 9\n")
        scratch.unlink()
        (repo / "orphan.txt").write_text("outside\n")
        if terminal_mode == "exception":
            raise RuntimeError("healer transport failed after writing")
        key = "red-tree"
        attempt = TestAttemptEvidence(
            session_id=state.session_id,
            kind="host",
            command=("false",),
            passed=False,
            input_tree_key=key,
            output_tree_key=key,
        )
        return TestAndHealResult(False, 1, False, False, (attempt,))

    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_and_heal", _red_after_mutation)
    result = await _step_test(ctx)

    assert isinstance(result, Stop) and result.exit_code == 1
    assert (repo / "a.py").read_text() == "A = 2\n"
    assert (repo / "unrelated.py").read_text() == "OWNER = 1\n"
    assert scratch.read_bytes() == b"\x00owner"
    assert not (repo / "orphan.txt").exists()
    audit = json.loads((ctx.data["dd"] / "fix-footprint.json").read_text())
    events = {(event["action"], event["path"]) for event in audit["events"]}
    assert {("restore", "unrelated.py"), ("restore", "scratch.bin"), ("remove", "orphan.txt")} <= events
    assert git_ops.snapshot_index(repo) == state.initial_index


@pytest.mark.parametrize("failure_mode", ["pass2_mutates", "unstable_test"])
async def test_stabilization_stops_after_two_passes_without_third_or_heal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:

    ctx, state, snapshot = _finalization_fixture(tmp_path)
    state.verifier_key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
    stale = TestAttemptEvidence(
        session_id=state.session_id,
        kind="host",
        command=("pytest",),
        passed=False,
        input_tree_key="before-heal",
        output_tree_key="after-heal",
    )
    guard_calls: list[int] = []
    test_calls = 0

    def _guard(*_a: Any, **kwargs: Any) -> bool:
        guard_calls.append(kwargs["round_number"])
        return failure_mode == "pass2_mutates"

    async def _one_test(*_a: Any, **_k: Any) -> Any:
        nonlocal test_calls
        test_calls += 1
        output_key = "unstable-output" if failure_mode == "unstable_test" else snapshot.tree_key
        return (
            TestAttemptEvidence(
                session_id=state.session_id,
                kind="host",
                command=("pytest",),
                passed=True,
                input_tree_key=snapshot.tree_key,
                output_tree_key=output_key,
            ),
            None,
            "test output",
        )

    monkeypatch.setattr("daydream.deep.fix_steps._strict_scope_and_scrub", _guard)
    monkeypatch.setattr("daydream.deep.fix_steps.capture_retained_tree", lambda *_a, **_k: snapshot)
    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_once", _one_test)
    monkeypatch.setattr(ctx, "backend_for", lambda _phase: object())

    result = await finalize_retained_tree_after_test(
        ctx,
        TestAndHealResult(False, 1, True, True, (stale,)),
    )

    assert isinstance(result, Stop) and result.exit_code == 1
    assert guard_calls == [1, 2]
    assert test_calls == 1
    assert state.latest_retained is None
    assert (ctx.data["dd"] / "stabilization-failed.json").is_file()

    phase_states = derive_phase_states(ctx.work.repo, phase_events=[], session_id=state.session_id)
    assert phase_states["fix"]["status"] == "failed"
    assert phase_states["test"]["status"] == "failed"
    assert derive_pipeline_status("complete", None, phase_states, runs_fix=True, runs_test=True) == "failed"


async def test_stabilization_audit_write_failure_stops_before_retest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    ctx, state, snapshot = _finalization_fixture(tmp_path)
    evidence = TestAttemptEvidence(
        session_id=state.session_id,
        kind="host",
        command=("pytest",),
        passed=True,
        input_tree_key=snapshot.tree_key,
        output_tree_key=snapshot.tree_key,
    )
    monkeypatch.setattr("daydream.deep.fix_steps._strict_scope_and_scrub", lambda *_a, **_k: False)
    monkeypatch.setattr("daydream.deep.fix_steps.capture_retained_tree", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        "daydream.deep.fix_steps._write_footprint_audit",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("audit disk full")),
    )
    monkeypatch.setattr(
        "daydream.deep.fix_steps.phase_test_once",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not retest")),
    )

    result = await finalize_retained_tree_after_test(
        ctx,
        TestAndHealResult(True, 0, True, False, (evidence,)),
    )

    assert isinstance(result, Stop) and result.exit_code == 1
    assert state.latest_retained is None
    failure = json.loads((ctx.data["dd"] / "stabilization-failed.json").read_text())
    assert "audit disk full" in failure["reason"]


def test_fix_outcome_summary_renders_uid_keyed_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:

    rendered: list[tuple[int, int, str | None]] = []
    monkeypatch.setattr(
        "daydream.deep.fix_steps.print_fix_complete",
        lambda _console, number, total, *, outcome=None: rendered.append((number, total, outcome)),
    )
    items = [{**_merge_item(7, "a.py", "high"), "item_uid": "item:a"}]
    outcomes = {"item:a": {"issue_id": 7, "verdict": "resolved", "reason": "fixed"}}

    _render_fix_outcome_summary(items, outcomes)

    assert rendered == [(1, 1, "resolved")]


@pytest.mark.parametrize(("new_override", "expect_stop"), [(False, True), (True, False)])
async def test_changed_tree_red_retest_requires_new_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    new_override: bool,
    expect_stop: bool,
) -> None:
    """A prior-tree red override cannot authorize a newly executed red result."""

    ctx, state, snapshot = _finalization_fixture(tmp_path)
    state.verifier_key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
    prior_override = TestAttemptEvidence(
        session_id=state.session_id,
        kind="host",
        command=("pytest",),
        passed=False,
        input_tree_key="prior-input",
        output_tree_key="prior-output",
    )

    async def _red_retest(*_a: Any, **_k: Any) -> Any:
        return (
            TestAttemptEvidence(
                session_id=state.session_id,
                kind="host",
                command=("pytest",),
                passed=False,
                input_tree_key=snapshot.tree_key,
                output_tree_key=snapshot.tree_key,
            ),
            None,
            "1 failed",
        )

    monkeypatch.setattr("daydream.deep.fix_steps._strict_scope_and_scrub", lambda *_a, **_k: False)
    monkeypatch.setattr("daydream.deep.fix_steps.capture_retained_tree", lambda *_a, **_k: snapshot)
    monkeypatch.setattr("daydream.deep.fix_steps.phase_test_once", _red_retest)
    monkeypatch.setattr(
        "daydream.deep.fix_steps._authorize_final_red_override",
        lambda _ctx: new_override,
    )
    monkeypatch.setattr(ctx, "backend_for", lambda _phase: object())

    result = await finalize_retained_tree_after_test(
        ctx,
        TestAndHealResult(False, 0, True, True, (prior_override,)),
    )

    assert isinstance(result, Stop) is expect_stop
    if isinstance(result, Stop):
        assert result.exit_code == 1
    verdict = json.loads((ctx.data["dd"] / "test-verdict.json").read_text())
    assert verdict["passed"] is False
    assert verdict["ignored"] is new_override


def test_final_red_override_requires_fresh_interactive_prompt(tmp_path: Path) -> None:

    prompts: list[dict[str, Any]] = []

    class RecordingContext(RunContext):
        def confirm(
            self,
            question: str,
            *,
            safe_default: bool,
            default: str = "n",
            console: Any = None,
        ) -> bool:
            prompts.append(
                {
                    "question": question,
                    "safe_default": safe_default,
                    "default": default,
                    "console": console,
                }
            )
            return True

    repo = _base_repo(tmp_path, "red-override")
    ctx = _direct_fix_context(repo, [], changed_files=set())
    ctx.run_context = RecordingContext(InteractionPolicy())

    assert _authorize_final_red_override(ctx) is True
    assert len(prompts) == 1
    assert prompts[0]["safe_default"] is False
    assert "still red" in prompts[0]["question"]

    ctx.run_context = RunContext(InteractionPolicy(assume="yes"))
    assert _authorize_final_red_override(ctx) is False
    assert len(prompts) == 1

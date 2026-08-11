"""Tests for the fail-closed service review worker (daydream/service/worker.py).

Real-path wherever possible: a real temp git repo detached at a real candidate
SHA, a real ``run_service_review`` / ``runner.run_service`` entry, real
``run_agent`` — only the external backend (or, for the Pi path, the pi
subprocess) is mocked. Mirrors the exemplar in tests/test_backend_pi.py.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.agent import classify_agent_abort
from daydream.backends import ResultEvent, TextEvent, ToolStartEvent
from daydream.backends.pi import PiError
from daydream.extensions import set_registry
from daydream.extensions.registry import Registry
from daydream.service.artifact import MAX_FINDINGS
from daydream.service.models import ReviewJobV1, ReviewTargetV1
from daydream.service.worker import (
    GitSnapshot,
    ServiceGitError,
    assert_unchanged,
    capture_git_state,
    changed_surfaces,
    run_service_review,
    terminal_exit_code,
)
from daydream.supervision import RuleBasedToolSupervisor
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import git as _git
from tests.harness.service_fakes import (
    BASE_SHA,
    CANDIDATE_SHA,
    CANDIDATE_TREE,
    CONFIG_DIGEST,
    DIFF_DIGEST,
)

SERVICE_FIXTURES = Path(__file__).parent / "fixtures" / "service"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _raw_git(repo: Path, *args: str) -> str:
    """Run git and return raw stdout (unstripped — digests are byte-exact)."""
    import subprocess

    return subprocess.run(  # noqa: S603 - test args are module-local
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _job_for(
    repo: Path,
    *,
    candidate_sha: str | None = None,
    candidate_tree_digest: str | None = None,
    base_sha: str | None = None,
    full_diff_digest: str | None = None,
    required_lenses: tuple[str, ...] = ("python",),
) -> ReviewJobV1:
    """Build a job whose digests match *repo* exactly, unless overridden."""
    head = git_ops.head_sha(repo)
    base = base_sha or _git(repo, "rev-parse", "main")
    tree = candidate_tree_digest or _git(repo, "rev-parse", "HEAD^{tree}")
    diff = _raw_git(repo, "diff", "--no-ext-diff", f"{base}..HEAD")
    return ReviewJobV1(
        job_id="job-1",
        idempotency_key="idem-1",
        target=ReviewTargetV1(
            target_kind="pr_head",
            repo="acme/demo",
            candidate_sha=candidate_sha or head,
            candidate_tree_digest=tree,
            base_sha=base,
            pr_numbers=(7,),
            full_diff_digest=full_diff_digest or _sha256(diff),
            invalidation_id="inv-1",
        ),
        effective_config_digest=CONFIG_DIGEST,
        reviewer_bundle_digest="f" * 64,
        required_lenses=required_lenses,
        round=1,
        attempt=1,
        deadline="2030-01-01T00:00:00Z",
        created_at="2030-01-01T00:00:00Z",
    )


def _detach(repo: Path) -> None:
    git_ops.checkout_detach(repo, git_ops.head_sha(repo))


def _clean_payload(lenses: tuple[str, ...] = ("python",)) -> dict[str, Any]:
    return {"completed_lenses": list(lenses), "issues": []}


def _issue(*, severity: str = "high", lens: str = "python") -> dict[str, Any]:
    return {
        "id": 1,
        "lens": lens,
        "file": "main.py",
        "line": 1,
        "severity": severity,
        "confidence": "HIGH",
        "title": "Title",
        "body": "Body",
    }


class MutatingBackend(ScriptedBackend):
    """A ScriptedBackend that mutates the working tree before yielding."""

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
        persist_session: bool = True,
    ):
        (Path(cwd) / "mutated.txt").write_text("mutated\n")
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
            persist_session=persist_session,
        ):
            yield event


# --- Git snapshot helpers ---------------------------------------------------


def test_capture_git_state_and_assert_unchanged(feature_branch_repo) -> None:
    _detach(feature_branch_repo)
    before = capture_git_state(feature_branch_repo)
    after = capture_git_state(feature_branch_repo)
    assert before == after
    assert assert_unchanged(before, after)
    assert changed_surfaces(before, after) == ()

    (feature_branch_repo / "new-untracked.txt").write_text("x\n")
    mutated = capture_git_state(feature_branch_repo)
    assert not assert_unchanged(before, mutated)
    assert changed_surfaces(before, mutated) == ("untracked",)

    (feature_branch_repo / "new-untracked.txt").unlink()
    (feature_branch_repo / "main.py").write_text("def hello():\n    return 'mutated'\n")
    mutated2 = capture_git_state(feature_branch_repo)
    assert changed_surfaces(before, mutated2) == ("tracked",)


def test_capture_git_state_fails_on_non_repo(tmp_path) -> None:
    with pytest.raises(ServiceGitError):
        capture_git_state(tmp_path)


# --- Worker outcome classification ------------------------------------------


async def test_clean_review_is_clean_and_read_only(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    payload = _clean_payload()
    backend = ScriptedBackend(
        events=(TextEvent(text=json.dumps(payload)), ResultEvent(structured_output=payload, continuation=None))
    )

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "clean"
    assert artifact.process_outcome == "exited_0"
    assert artifact.completed_lenses == ("python",)
    assert artifact.missing_lenses == ()
    assert artifact.findings == ()
    assert backend.read_only_calls == [True]
    assert backend.max_turns == [None]
    assert backend.calls[0]["persist_session"] is False


async def test_blocking_findings_keep_findings_terminal(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    payload = {"completed_lenses": ["python"], "issues": [_issue(severity="high")]}
    backend = ScriptedBackend(events=(ResultEvent(structured_output=payload, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "findings"
    assert artifact.process_outcome == "exited_0"
    assert len(artifact.findings) == 1


async def test_non_blocking_findings_with_exit_0_are_clean(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    payload = {"completed_lenses": ["python"], "issues": [_issue(severity="low")]}
    backend = ScriptedBackend(events=(ResultEvent(structured_output=payload, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "clean"
    assert artifact.findings


async def test_missing_lens_after_dispatch_is_infra_error(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo, required_lenses=("python", "react"))
    payload = _clean_payload(lenses=("python",))  # react never completes
    backend = ScriptedBackend(events=(ResultEvent(structured_output=payload, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python", "react"))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "incomplete_lenses"
    assert artifact.missing_lenses == ("react",)
    assert artifact.completed_lenses == ("python",)


async def test_missing_lens_before_dispatch_is_infra_error(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo, required_lenses=("python", "react"))
    backend = ScriptedBackend(events=(ResultEvent(structured_output=None, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "lens_unavailable"
    assert artifact.missing_lenses == ("react",)


async def test_mutation_is_detected(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    payload = _clean_payload()
    backend = MutatingBackend(events=(ResultEvent(structured_output=payload, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "mutation_detected"
    assert "before_state" in artifact.hashes and "after_state" in artifact.hashes


async def test_backend_error_is_process_loss(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    backend = ScriptedBackend(events=(TextEvent(text="x"), PiError("provider blew up", category="UNKNOWN")))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "process_loss"


async def test_backend_process_exit_is_exited_nonzero(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    backend = ScriptedBackend(
        events=(TextEvent(text="x"), PiError("Pi CLI exited with return code 1", category="PROCESS_EXIT"))
    )

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "exited_nonzero"


async def test_tool_veto_is_infra_error(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    registry = Registry()
    registry.register_tool_supervisor(RuleBasedToolSupervisor(deny_globs=["x.py"], bash_deny=[]))
    set_registry(registry)
    backend = ScriptedBackend(events=(ToolStartEvent(id="t1", name="Write", input={"path": "x.py"}),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "tool_vetoed"


async def test_cancellation_produces_cancelled_terminal(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    backend = ScriptedBackend(events=(asyncio.CancelledError(),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "cancelled"
    assert artifact.process_outcome == "cancelled"


async def test_parse_loss_is_infra_error(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    backend = ScriptedBackend(
        events=(TextEvent(text="I cannot produce JSON"), ResultEvent(structured_output=None, continuation=None))
    )

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "parse_loss"


async def test_findings_overflow_is_infra_error(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    issues = [_issue(severity="low", lens="python") for _ in range(MAX_FINDINGS + 1)]
    for i, issue in enumerate(issues, start=1):
        issue["id"] = i
    payload = {"completed_lenses": ["python"], "issues": issues}
    backend = ScriptedBackend(events=(ResultEvent(structured_output=payload, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "findings_overflow"


# --- Pre-flight verification -------------------------------------------------


async def test_preflight_rejects_wrong_candidate_sha(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo, candidate_sha="0" * 40)
    backend = ScriptedBackend(events=(ResultEvent(structured_output=None, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "git_preflight_failed"


async def test_preflight_rejects_attached_branch(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    # Deliberately NOT detached: still on the feature branch.
    job = _job_for(feature_branch_repo)
    backend = ScriptedBackend(events=(ResultEvent(structured_output=None, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "git_preflight_failed"


async def test_preflight_rejects_diff_digest_mismatch(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo, full_diff_digest="1" * 64)
    backend = ScriptedBackend(events=(ResultEvent(structured_output=None, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "git_preflight_failed"


async def test_preflight_rejects_dirty_tree(feature_branch_repo, monkeypatch, silence_console) -> None:
    _silence_modules(silence_console, monkeypatch)
    _detach(feature_branch_repo)
    (feature_branch_repo / "main.py").write_text("def hello():\n    return 'dirty'\n")
    job = _job_for(feature_branch_repo)
    backend = ScriptedBackend(events=(ResultEvent(structured_output=None, continuation=None),))

    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "infra_error"
    assert artifact.process_outcome == "git_preflight_failed"


# --- Exit code mapping + abort classification --------------------------------


def _bare_job() -> ReviewJobV1:
    return ReviewJobV1(
        job_id="job-1",
        idempotency_key="idem-1",
        target=ReviewTargetV1(
            target_kind="pr_head",
            repo="acme/demo",
            candidate_sha=CANDIDATE_SHA,
            candidate_tree_digest=CANDIDATE_TREE,
            base_sha=BASE_SHA,
            pr_numbers=(7,),
            full_diff_digest=DIFF_DIGEST,
            invalidation_id="inv-1",
        ),
        effective_config_digest=CONFIG_DIGEST,
        reviewer_bundle_digest="f" * 64,
        required_lenses=("python",),
        round=1,
        attempt=1,
        deadline="2030-01-01T00:00:00Z",
        created_at="2030-01-01T00:00:00Z",
    )


def test_terminal_exit_code_mapping() -> None:
    job = _bare_job()
    assert terminal_exit_code(_artifact(job, "clean")) == 0
    assert terminal_exit_code(_artifact(job, "findings")) == 0
    assert terminal_exit_code(_artifact(job, "infra_error")) == 1
    assert terminal_exit_code(_artifact(job, "cancelled")) == 2


def test_classify_agent_abort() -> None:
    assert classify_agent_abort(None) is None
    assert classify_agent_abort("tool_vetoed:Write") == "tool_vetoed"
    assert classify_agent_abort("wall_budget_exceeded") == "budget_exhausted"
    assert classify_agent_abort("tool_call_budget_exceeded") == "budget_exhausted"
    assert classify_agent_abort("anything_else") == "process_loss"


def test_git_snapshot_digest_is_stable() -> None:
    a = GitSnapshot(head_sha="h", tree_digest="t", index_digest="i", tracked_digest="k", untracked_digest="u")
    b = GitSnapshot(head_sha="h", tree_digest="t", index_digest="i", tracked_digest="k", untracked_digest="u")
    assert a.digest() == b.digest()
    assert a.digest() == _sha256("h\nt\ni\nk\nu")


# --- Real-path runner hook ---------------------------------------------------


async def test_run_service_hook_returns_clean_code(
    make_config, feature_branch_repo, monkeypatch, silence_console
) -> None:
    _silence_modules(silence_console, monkeypatch, extra=("daydream.runner",))
    from daydream.runner import run_service

    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    payload = _clean_payload()
    backend = ScriptedBackend(events=(ResultEvent(structured_output=payload, continuation=None),))
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    rc = await run_service(make_config(feature_branch_repo), job)

    assert rc == 0
    assert backend.read_only_calls == [True]


# --- Real-path Pi backend (subprocess mocked) --------------------------------


def _capture_pi_subprocess(monkeypatch, captured: list[list[str]], fixture: str) -> None:
    import asyncio as _asyncio

    real_exec: Any = _asyncio.create_subprocess_exec

    async def _fake_exec(*args: object, **kwargs: object):
        if args and args[0] == "pi":
            captured.append([str(a) for a in args])
            return _service_mock_process(fixture)
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr("daydream.backends.pi.asyncio.create_subprocess_exec", _fake_exec)


def _service_mock_process(name: str):
    from tests.harness.pi_replay import make_mock_process

    lines = (SERVICE_FIXTURES / name).read_text().strip().split("\n")
    return make_mock_process(lines)


async def test_pi_real_path_read_only_clean_review(
    feature_branch_repo, monkeypatch, silence_console
) -> None:
    _silence_modules(silence_console, monkeypatch)
    from daydream.backends import create_backend

    _detach(feature_branch_repo)
    job = _job_for(feature_branch_repo)
    captured: list[list[str]] = []
    _capture_pi_subprocess(monkeypatch, captured, "service_clean_review.jsonl")

    backend = create_backend("pi", model="glm-5.2")
    artifact = await run_service_review(feature_branch_repo, job, backend, lens_inventory=("python",))

    assert artifact.terminal == "clean"
    assert artifact.process_outcome == "exited_0"
    assert captured, "expected at least one pi subprocess spawn"
    argv = captured[0]
    assert argv[argv.index("--tools") + 1] == "read,find,ls,grep"
    assert "--no-session" in argv
    assert any("service-mode" in arg for arg in argv if arg.startswith("You are a read-only code reviewer"))


# --- Helpers -----------------------------------------------------------------


def _artifact(job: ReviewJobV1, terminal: str):
    from daydream.service.artifact import WorkerArtifactV1

    if terminal in ("clean", "findings"):
        return WorkerArtifactV1.complete(
            job,
            completed_lenses=("python",),
            findings=() if terminal == "clean" else (_issue(severity="high"),),
        )
    if terminal == "cancelled":
        return WorkerArtifactV1.cancelled(job, completed_lenses=(), missing_lenses=("python",))
    return WorkerArtifactV1.infra_error(job, process_outcome="process_loss")


def _silence_modules(silence_console, monkeypatch, *, extra: tuple[str, ...] = ()) -> None:
    """Silence print_* helpers but keep a REAL Rich console.

    ``silence_console``'s stub console (only ``.print``) breaks the worker's
    genuine rendering path (``run_agent`` → Rich ``Live`` panels when the
    backend returns issues), so the console is replaced with a real Rich
    Console writing to a StringIO instead — output is captured, semantics are
    untouched.
    """
    import importlib
    import io

    from rich.console import Console

    quiet = Console(file=io.StringIO())
    for module in ("daydream.agent", "daydream.phases", *extra):
        silence_console(module, keep=("console",))
        if hasattr(importlib.import_module(module), "console"):
            monkeypatch.setattr(f"{module}.console", quiet)

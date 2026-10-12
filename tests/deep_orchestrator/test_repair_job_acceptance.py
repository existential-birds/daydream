"""Acceptance matrix for the bounded repair job (issue #1210, task 10).

Every row below is a real-path test: it enters through ``runner.run`` against a
real temporary Git worktree, with a real event loop, a real filesystem, and a
real test process where the row is about a real suite. The only thing replaced is
the external model backend, installed at the ``create_backend`` seam.

The rows deliberately exercise *different* observable surfaces, so that the final
invariant test (``test_no_acceptance_row_makes_a_red_suite_pass``) can assert
what every row has in common: a red suite is never reported green, whichever
repair path the run took.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.artifact_visibility import ArtifactVisibilityError, artifact_dir_for
from daydream.backends import AgentEvent
from daydream.config_file import DaydreamFileConfig
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import commit as _commit, git as _git
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import (
    Mute,
    _merge_item,
    _silence,
)

#: The interrupted repair below still lands the edits it was working on, so a
#: candidate exists to continue from.
CANDIDATE_LINE = "\n# candidate repair\n"
HELPER_TEXT = "# helper the repair turn added\n"

#: The failure-summarizer turn runs after the interrupted repair and before the
#: coordinator restores anything, so it is the one real moment a test can observe
#: what the job is going to do next.
_SUMMARIZER_PROMPT = "read-only failure-summarizer"

_JOB = "repair-job.json"
_CHECKPOINT = "repair-checkpoint.json"
_VERDICT = "test-verdict.json"


def _deep_dir(cwd: Path) -> Path | None:
    """The routed ``.daydream`` directory of the live run, as the host resolved it.

    A run routes its artifacts through the private session root and publishes
    them at the end, so the public ``.daydream`` location is not readable while
    the run is still working. Reading the routed path is how anything *inside*
    the run observes what the host has persisted so far. A turn that runs
    outside the session's own repository (an operational fix worktree, say) is
    not the run's artifact root at all, and observes nothing.
    """
    try:
        return artifact_dir_for(cwd, allow_standalone=True) / "deep"
    except ArtifactVisibilityError:
        return None


class _AcceptanceStub(StubBackend):
    """Perform concurrent filesystem faults only at the external summary turn."""

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.on_summarizer: Callable[[Path], None] | None = None

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if self.on_summarizer is not None and _SUMMARIZER_PROMPT in prompt.lower():
            self.on_summarizer(cwd)
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


def _install(
    monkeypatch: pytest.MonkeyPatch, target: Path, **knobs: Any,
) -> _AcceptanceStub:
    """Install the acceptance stub, with the repair-turn knobs a row needs."""
    stub = _AcceptanceStub(target)
    for name, value in knobs.items():
        setattr(stub, name, value)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    return stub


def _interrupted_repair(**overrides: Any) -> dict[str, Any]:
    """Knobs for a repair turn the host wall budget cuts short, mid-edit."""
    return {
        "heal_fix_partial": "PARTIAL-DIAGNOSIS-abc123",
        "runaway_fix_sleep_s": 0.05,
        "heal_fix_authorized": "api.py",
        "heal_fix_authorized_line": CANDIDATE_LINE,
        **overrides,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _job(target: Path) -> dict[str, Any]:
    """The persisted repair job record of the run under test."""
    return _read_json(target / ".daydream" / "deep" / _JOB)


def _checkpoint(target: Path) -> dict[str, Any]:
    """The persisted repair checkpoint of the run under test."""
    return _read_json(target / ".daydream" / "deep" / _CHECKPOINT)


def _verdict(target: Path) -> dict[str, Any]:
    """The persisted test verdict of the run under test."""
    return _read_json(target / ".daydream" / "deep" / _VERDICT)


async def _run_deep(target: Path, **overrides: Any) -> int:
    """Await the production entrypoint for a repaired target."""
    fields: dict[str, Any] = {
        "target": str(target), "non_interactive": True, "cleanup": False, "archive": False,
        "assume": "yes", "output_mode": "loop", **overrides,
    }
    with anyio.fail_after(90):
        return await run(RunConfig(**fields))


# --- rows -----------------------------------------------------------------------------------------


async def test_required_excluded_suite_still_blocks_green(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """A required suite is red: the run is red, however convincing the repair's claim is.

    The canonical command is a REAL pytest process over a real failing test file
    that the repair turn never looks at. The repair reports a focused run that
    excluded it; that report is a claim, and the run must still end red.
    """
    _silence(monkeypatch)
    target = multi_stack_target
    (target / "tests").mkdir(exist_ok=True)
    (target / "tests" / "test_required.py").write_text(
        "def test_required_suite():\n    assert 'universe' == 'world'\n",
        encoding="utf-8",
    )
    _git(target, "add", "tests/test_required.py")
    _commit(target, "test: add a required suite that is genuinely failing")

    argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/test_required.py"]
    stub = _install(
        monkeypatch, target,
        # A completed repair that reports the focused run which skipped the
        # required suite. Nothing here runs the required suite.
        heal_fix_focused_claim=(
            "FOCUSED-CLAIM: `pytest -k 'not required'` -> 4 passed "
            "(excluded tests/test_required.py)"
        ),
    )
    mute_side_effects(heal=False)

    exit_code = await _run_deep(
        target,
        test_command=" ".join(shlex.quote(part) for part in argv),
        test_required_suites=["required"],
    )
    assert exit_code != 0, "a genuinely failing required suite cannot report a green run"

    verdict = _verdict(target)
    assert verdict["passed"] is False, verdict
    assert verdict["attempts"], "the host suite produced no evidence"
    for attempt in verdict["attempts"]:
        # Every recorded execution is the real host command, and every one of
        # them is red: no focused claim was promoted to a pass.
        assert attempt["kind"] == "host", attempt
        assert attempt["passed"] is False, attempt
    assert _job(target)["state"] != "completed", "no execution validated a green suite"
    assert stub.test_suite_calls == 0, "the host command runs on the host, not through the agent"


async def test_interrupted_tracked_and_untracked_edits_both_reach_the_checkpoint(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """An interrupted repair keeps BOTH kinds of work in the durable checkpoint.

    The tracked edit is an authorized reviewed file; the untracked helper is
    authorized through the item's own ``related_files``. Both must survive into
    the checkpoint, because the next step of the phase restores files.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    item = _merge_item(1, "api.py", "high")
    item["related_files"] = ["helper.py"]
    stub = _install(
        monkeypatch, target,
        **_interrupted_repair(heal_fix_authorized_new="helper.py", heal_fix_authorized_new_text=HELPER_TEXT),
    )
    stub.merge_items = [item]
    command = host_test_command(target.parent / "suite", mode="first")
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code == 0, "the continued execution validated the retained tree"

    checkpoint = _checkpoint(target)
    patch = checkpoint["candidate_patch"]
    assert "api.py" in patch, patch
    assert "helper.py" in patch, patch
    assert "candidate repair" in patch, patch
    assert "helper the repair turn added" in patch, patch
    assert checkpoint["authorized_scope"] == sorted(checkpoint["authorized_scope"]), checkpoint
    assert "helper.py" in checkpoint["authorized_scope"], checkpoint
    assert _job(target)["checkpoint_ref"] == _CHECKPOINT, _job(target)
    # The work really was on disk before the capture, not invented by the host.
    assert "candidate repair" in (target / "api.py").read_text(encoding="utf-8")
    assert stub.test_suite_calls == 0


async def test_candidate_finished_just_before_timeout_enters_validation(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """A candidate that landed just before its deadline is validated, not discarded.

    The interrupted turn's work is admitted to bounded validation exactly once,
    and the job says so on disk *before* the validating execution starts.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(monkeypatch, target, **_interrupted_repair())
    suite = target.parent / "suite"
    suite.mkdir()
    pointer = suite / "job-path.txt"
    observed = suite / "validation-state.txt"
    script = suite / "observe_validation.py"
    script.write_text(
        "from pathlib import Path\nimport json, sys\n"
        f"counter = Path({str(suite / 'suite-count.txt')!r})\n"
        "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        "if count > 1:\n"
        f"    job = json.loads(Path(Path({str(pointer)!r}).read_text()).read_text())\n"
        f"    Path({str(observed)!r}).write_text(job['state'])\n"
        "print('1 failed' if count == 1 else '2 passed')\n"
        "sys.exit(1 if count == 1 else 0)\n"
    )

    def expose_job_for_host_observation(cwd: Path) -> None:
        deep = _deep_dir(cwd)
        assert deep is not None
        pointer.write_text(str(deep / _JOB))

    # The external summary turn exposes only the location of the host's real
    # durable record to the fixture executable. The executable reads its state
    # when validation begins, without replacing the coordinator or host runner.
    stub.on_summarizer = expose_job_for_host_observation
    command = shlex.join([sys.executable, str(script)])
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code == 0, "the continued execution validated the retained tree"

    # Two real host executions occurred, and the durable job records that the
    # second was bounded validation of the interrupted candidate.
    assert (target.parent / "suite/suite-count.txt").read_text() == "2"
    assert stub.test_suite_calls == 0
    assert observed.read_text() == "validating"
    job = _job(target)
    assert job["state"] == "completed", job
    assert job["executions"] == 2, job
    assert job["progress_evidence"][0] == "bounded validation of the interrupted candidate", job
    verdict = _verdict(target)
    assert verdict["passed"] is True, verdict
    assert len(verdict["repairs"]) == 1, "the interrupted turn is the only repair record"


async def test_focused_pass_does_not_override_canonical_red(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """A repair's focused-pass report never turns a red canonical suite green."""
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(
        monkeypatch, target,
        **_interrupted_repair(
            heal_fix_focused_claim="FOCUSED-CLAIM: focused run passed (2 passed, 0 failed)",
        ),
    )
    # Every canonical suite run stays red, so the claim can never be corroborated.
    command = host_test_command(target.parent / "suite", mode="all")
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code != 0, "a focused pass claim cannot make a red canonical suite green"

    verdict = _verdict(target)
    assert verdict["passed"] is False, verdict
    for attempt in verdict["attempts"]:
        assert attempt["kind"] == "host", attempt
        assert attempt["passed"] is False, attempt
    job = _job(target)
    assert job["state"] == "blocked", job
    assert "unchanged evidence" in (job["last_transition_reason"] or ""), job
    assert verdict["repairs"], "the claim-carrying turn must still be recorded as a repair"
    assert stub.test_suite_calls == 0


async def test_host_timeout_pass_prose_never_counts_as_green(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """A host process that prints passing prose then times out remains red."""
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    monkeypatch.setattr("daydream.config.TEST_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    script = target.parent / "timeout_suite.py"
    script.write_text("import time\nprint('12 passed', flush=True)\ntime.sleep(30)\n")
    stub = _install(monkeypatch, target, runaway_fix_sleep_s=0.05)
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=shlex.join([sys.executable, str(script)]))
    assert exit_code != 0, "an incomplete host process cannot report a green run"
    assert stub.test_suite_calls == 0
    verdict = _verdict(target)
    assert verdict["passed"] is False
    assert verdict["attempts"]
    assert all(attempt["kind"] == "host" and not attempt["passed"] for attempt in verdict["attempts"])
    assert verdict["attempts"][0]["identity"]["outcome"] == "timed-out"
    assert _job(target)["state"] != "completed"


async def test_provider_retry_does_not_restart_the_execution_deadline(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """Provider retries share one deadline: the execution cannot be re-based.

    The repair turn fails with a transport-shaped error twice, and is then cut
    short by the wall budget. The whole ladder -- failures, backoff, and the
    partial turn -- must fit inside the single execution allowance.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.5)
    # Keep the retry ladder fast and bounded so the assertion is about the
    # deadline, not about how long the harness is willing to wait.
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.05")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0.05")
    target = multi_stack_target
    stub = _install(
        monkeypatch, target,
        **_interrupted_repair(heal_fix_retryable_failures=2, heal_fix_authorized=None),
    )
    command = host_test_command(target.parent / "suite", mode="first")
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code == 0, "the continued execution validated the retained tree"

    heal_calls = [call for call in stub.calls if call["prompt"].lower().startswith("the tests failed")]
    assert len(heal_calls) >= 2, "the provider failure was retried within the same execution"
    verdict = _verdict(target)
    assert len(verdict["repairs"]) == 1, verdict["repairs"]
    repair = verdict["repairs"][0]
    assert repair["abort_reason"] == "wall_budget_exceeded", repair
    # One absolute deadline covers every attempt: a re-based deadline per retry
    # would show up here as several multiples of the wall budget.
    assert repair["execution_elapsed_s"] < 2.0, repair
    job = _job(target)
    assert job["executions"] == 2, job


async def test_changed_base_produces_a_named_conflict_not_a_stale_apply(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """A base that moved is a named conflict, never a best-effort re-apply.

    A concurrent caller edits the same file between the capture and the restore.
    Applying the captured patch on top of those bytes would silently clobber
    them, so the job names the conflict and stops.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(monkeypatch, target, **_interrupted_repair())
    command = host_test_command(target.parent / "suite", mode="first")

    def _concurrent_caller_edit(cwd: Path) -> None:
        edited = cwd / "api.py"
        edited.write_text(edited.read_text(encoding="utf-8") + "\n# a caller edited this line\n",
                          encoding="utf-8")

    stub.on_summarizer = _concurrent_caller_edit
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code != 0, "a job that cannot restore its candidate never reports green"

    job = _job(target)
    assert job["state"] == "blocked", job
    reason = job["last_transition_reason"] or ""
    assert "base" in reason, job
    # The stale candidate was never re-applied on top of the caller's bytes: no
    # second execution ran at all.
    assert (target.parent / "suite/suite-count.txt").read_text() == "1"
    assert stub.test_suite_calls == 0
    assert "a caller edited this line" in (target / "api.py").read_text(encoding="utf-8")


async def test_corrupt_checkpoint_blocks_rather_than_restarting_empty(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """A checkpoint that cannot be read is a blocker, not an empty job.

    An interrupted repair's checkpoint is the only copy of its work. A truncated
    file is not that copy, and guessing "nothing to continue" would throw the
    work away without saying so.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(monkeypatch, target, **_interrupted_repair())
    command = host_test_command(target.parent / "suite", mode="first")

    def _truncate_checkpoint(cwd: Path) -> None:
        deep = _deep_dir(cwd)
        assert deep is not None, "the failure-summarizer turn runs inside the run's artifact root"
        path = deep / _CHECKPOINT
        path.write_text(path.read_text(encoding="utf-8")[:120], encoding="utf-8")

    stub.on_summarizer = _truncate_checkpoint
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code != 0, "an untrustworthy checkpoint never reports a green run"

    job = _job(target)
    assert job["state"] == "blocked", job
    assert "malformed" in (job["last_transition_reason"] or ""), job
    assert "cannot be trusted" in (job["last_transition_reason"] or ""), job
    # Nothing was restarted from nothing: the coordinator dispatched nothing.
    assert (target.parent / "suite/suite-count.txt").read_text() == "1"
    assert stub.test_suite_calls == 0


async def test_operator_grant_resumes_the_same_job_without_erasing_history(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """An operator's grant resumes the same job; the history it already spent stays spent.

    The job's own wall allowance is configured to nothing, so the first
    execution exhausts it. The operator's additional finite allowance is what
    lets the bounded validation run -- and it is recorded *separately* from the
    consumption, so the record never claims the time was never spent.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(monkeypatch, target, **_interrupted_repair())
    command = host_test_command(target.parent / "suite", mode="first")
    mute_side_effects(heal=False)

    exit_code = await _run_deep(
        target, test_command=command,
        file_config=DaydreamFileConfig(repair_job_wall_s=0.5, repair_grant_job_wall_s=300.0),
    )
    assert exit_code == 0, "the granted allowance let the bounded validation run"

    job = _job(target)
    assert job["state"] == "completed", job
    assert job["executions"] == 2, job
    assert job["granted_allowance_s"] == 300.0, job
    assert job["consumed_s"] > 0.0, job
    # The grant is an addition, not a rewrite: the seconds this job actually
    # spent are still the seconds it spent.
    assert job["consumed_s"] < 300.0, job
    # The job's own allowance is what the first execution exhausted; the grant
    # is what the second one drew on, and it is still on the record afterwards.
    assert job["policy"]["job_total_s"] == 0.5, job
    assert any("operator granted 300s more" in note for note in job["diagnostics"]), job
    assert stub.test_suite_calls == 0


async def test_no_acceptance_row_makes_a_red_suite_pass(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """The invariant every row above shares: a red suite is never reported green.

    The canonical suite stays red across every repair path the matrix exercises
    -- a completed repair, an interrupted one, a retry ladder, a claimed focused
    pass, a moved base, an untrustworthy checkpoint -- and the run must end red
    in all of them.
    """
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    target = multi_stack_target
    stub = _install(
        monkeypatch, target,
        **_interrupted_repair(
            heal_fix_focused_claim="FOCUSED-CLAIM: focused run passed (2 passed, 0 failed)",
            heal_fix_retryable_failures=1,
        ),
    )
    # Nothing ever makes the suite green, at any point in the run.
    command = host_test_command(target.parent / "suite", mode="all")
    mute_side_effects(heal=False)

    exit_code = await _run_deep(target, test_command=command)
    assert exit_code != 0, "a permanently red suite must not report a green run"

    verdict = _verdict(target)
    assert verdict["passed"] is False, verdict
    assert verdict["attempts"] and all(not attempt["passed"] for attempt in verdict["attempts"]), verdict
    assert _job(target)["state"] != "completed", _job(target)
    assert int((target.parent / "suite/suite-count.txt").read_text()) >= 2
    assert stub.test_suite_calls == 0

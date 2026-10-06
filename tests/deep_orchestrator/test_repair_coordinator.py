"""The repair coordinator: automatic continuation, identity-checked restore, one live owner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.deep.repair_job import read_repair_job_record
from daydream.runner import run
from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _merge_item,
    _silence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The interrupted heal turn below still lands its authorized edit, so a candidate
#: exists to continue from. The base stub makes no edit before it stalls, which is
#: the shape of an interrupted turn that captured nothing.
CANDIDATE_LINE = "\n# candidate repair\n"


def _install_interrupted_heal(monkeypatch: pytest.MonkeyPatch, target: Path) -> StubBackend:
    """Install one stub whose repair turn is cut short by the host wall budget."""
    stub = StubBackend(target)
    stub.heal_fix_authorized = "api.py"
    stub.heal_fix_authorized_line = CANDIDATE_LINE
    stub.heal_fix_partial = "PARTIAL-DIAGNOSIS-abc123"
    stub.runaway_fix_sleep_s = 0.05
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    return stub


def _repair_job(target: Path) -> dict[str, Any]:
    """The persisted job record of the run under test."""
    payload: dict[str, Any] = json.loads(
        (target / ".daydream" / "deep" / "repair-job.json").read_text(encoding="utf-8"),
    )
    return payload


async def test_repair_continues_automatically_within_one_run(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Requirement 29: the coordinator dispatches the next execution itself."""

    _silence(monkeypatch)
    # The repair turn is cut short by the host wall budget; only the coordinator
    # can give the job another execution.
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    stub = _install_interrupted_heal(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fail_first_test_run = True
    mute_side_effects(heal=False)

    with anyio.fail_after(60):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=tmp_path / "t.json",
                                          assume="yes", output_mode="loop"))
    assert exit_code == 0, "the continued execution validated the retained tree"

    job = _repair_job(multi_stack_target)
    assert job["state"] == "completed", job
    assert job["executions"] == 2, job
    # Two separate test executions ran: the interrupted one and the one the
    # coordinator dispatched. The second saw the suite green, so it never
    # launched a repair turn of its own.
    assert stub.test_suite_calls == 2, stub.test_suite_calls
    verdict = json.loads((multi_stack_target / ".daydream" / "deep" / "test-verdict.json").read_text())
    assert verdict["passed"] is True
    assert len(verdict["repairs"]) == 1, "only the interrupted turn is a repair record"


async def test_repair_completes_only_after_canonical_validation_passes(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Requirement 45: a partially-verified candidate does not complete the job."""

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    stub = _install_interrupted_heal(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    # Every suite run stays red, so no execution can ever be validated.
    stub.fail_all_test_runs = True
    mute_side_effects(heal=False)

    with anyio.fail_after(60):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=tmp_path / "t.json",
                                          assume="yes", output_mode="loop"))
    assert exit_code != 0, "a red suite cannot report a green run"

    job = _repair_job(multi_stack_target)
    assert job["state"] != "completed", job
    assert job["state"] == "blocked", job
    assert "unchanged evidence" in (job["last_transition_reason"] or ""), job
    # The candidate was captured and admitted to bounded validation exactly once;
    # the repeat of the same candidate is what blocks the job.
    assert job["executions"] == 2, job
    assert job["progress_evidence"] == ["bounded validation of the interrupted candidate"], job
    verdict = json.loads((multi_stack_target / ".daydream" / "deep" / "test-verdict.json").read_text())
    assert verdict["passed"] is False
    assert len(verdict["repairs"]) == 2


# The two recovery workers below are REAL processes. The first one repairs, is
# charged, and then dies mid-dispatch (exactly the crash the durable job record
# exists for); the second one is a different interpreter that reloads the
# artifacts from disk and continues. An in-process reload would share memory and
# prove nothing about cross-process recovery.
_RECOVERY_DRIVER = '''\
"""One repair worker: charge the job, restore the candidate, run the next execution."""
import io
import json
import os
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import anyio
import rich.console

import daydream.agent as agent
import daydream.config as phase_config
import daydream.run_context as run_context
import daydream.ui as ui
from daydream import git_ops
from daydream.deep.repair_coordinator import continue_repair_job, repair_job_id
from daydream.deep.repair_job import read_repair_job_record
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.phases import phase_test_and_heal
from daydream.workspace import WorkContext
from tests.harness.stub_backend import StubBackend

REPO = Path(sys.argv[1])
MODE = sys.argv[2]
OUT = Path(sys.argv[3])
SESSION = "repair-resume-probe"
AUTHORIZED = "api.py"
CANDIDATE_LINE = "\\n# candidate repair\\n"

_quiet = rich.console.Console(file=io.StringIO(), width=200)
agent.console = _quiet
ui.console = _quiet
# Interactive "fix and retry", the bounded automatic choice the host would make.
run_context._prompt_user = lambda *a, **k: "2"
phase_config.DEFAULT_TOOL_CALL_BUDGET = 2


def work_context():
    head = git_ops.head_sha(REPO)
    return WorkContext(repo=REPO, source=REPO, base_branch="main", base_sha=head,
                       head_sha=head, head_branch=git_ops.current_branch(REPO),
                       is_ephemeral=False, run_id=SESSION)


def tree_key():
    return git_ops.tree_key(
        git_ops.snapshot_worktree_delta(REPO, "HEAD", preexisting_untracked={},
                                        preexisting_gitlinks=()))


async def execute(stub, work, footprint):
    return await phase_test_and_heal(stub, work, session_id=SESSION, capture_tree_key=tree_key,
                                     footprint=footprint, allow_standalone=True)


async def die_mid_dispatch():
    """The process that charged the execution never returns from it."""
    os._exit(9)


async def main():
    work = work_context()
    footprint = AuthorizedFixFootprint(run_allowed_paths=frozenset({AUTHORIZED}), policy_revision=1)
    deep = REPO / ".daydream" / "deep"
    stub = StubBackend(REPO)
    stub.heal_fix_authorized = AUTHORIZED
    stub.heal_fix_authorized_line = CANDIDATE_LINE
    stub.heal_fix_partial = "PARTIAL-DIAGNOSIS-abc123"
    stub.runaway_fix_sleep_s = 0.0
    stub.fail_first_test_run = MODE == "first"

    first = None if MODE == "resume" else await execute(stub, work, footprint)
    outcome = await continue_repair_job(
        work=work,
        deep_dir_path=deep,
        job_id=repair_job_id(SESSION),
        footprint=footprint,
        capture_tree_key=tree_key,
        first=first,
        dispatch=lambda: die_mid_dispatch() if MODE == "first" else execute(stub, work, footprint),
    )
    job = read_repair_job_record(deep)
    OUT.write_text(json.dumps({
        "mode": MODE,
        "state": None if job is None else job.state.value,
        "executions": 0 if job is None else job.executions,
        "continued": outcome.continued,
        "passed": False if outcome.result is None else outcome.result.passed,
        "reason": outcome.reason,
        "candidate_restored": CANDIDATE_LINE in (REPO / AUTHORIZED).read_text(),
        "reloads": None if job is None else job.checkpoint_ref,
    }), encoding="utf-8")


anyio.run(main)
'''


def _run_worker(driver: Path, target: Path, mode: str, out: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run one recovery worker as a real, separate interpreter."""
    return subprocess.run(
        [sys.executable, str(driver), str(target), mode, str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


async def test_repair_resumes_in_a_genuinely_separate_process(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Requirements 30/48: a SECOND PROCESS reloads the job and continues.

    Spawning ``sys.executable <driver>`` as a real subprocess is deliberate: an
    in-process reload shares memory and proves nothing about cross-process
    recovery. There is no CLI verb for a stubbed backend, so the driver calls
    the same production entrypoints the CLI would reach.
    """
    mute_side_effects(heal=False)
    driver = tmp_path / "repair_recovery_worker.py"
    driver.write_text(_RECOVERY_DRIVER, encoding="utf-8")

    first_out = tmp_path / "first.json"
    first = _run_worker(driver, multi_stack_target, "first", first_out, tmp_path)
    assert first.returncode == 9, f"the first worker must die mid-dispatch:\\n{first.stdout}\\n{first.stderr}"

    deep = multi_stack_target / ".daydream" / "deep"
    # The charged execution is durable: the record survived the process that wrote it.
    charged = read_repair_job_record(deep)
    assert charged is not None, "the job record must outlive the process that charged it"
    assert charged.executions == 1, charged.payload()
    # The record names the execution the crashed process had already dispatched:
    # the validating state is persisted before that dispatch, so a process that
    # dies inside it says so instead of looking untouched.
    assert charged.state.value == "validating", charged.payload()
    assert charged.checkpoint_ref == "repair-checkpoint.json", charged.payload()
    assert not first_out.exists(), "the crashed worker never reported a result"

    # Put the worktree back on the captured base so the resuming process has to
    # restore the candidate from the checkpoint rather than find it in place.
    _git(multi_stack_target, "checkout", "--", "api.py")
    assert "candidate repair" not in (multi_stack_target / "api.py").read_text()

    second_out = tmp_path / "second.json"
    second = _run_worker(driver, multi_stack_target, "resume", second_out, tmp_path)
    assert second.returncode == 0, f"the resuming worker failed:\\n{second.stdout}\\n{second.stderr}"
    resumed = json.loads(second_out.read_text(encoding="utf-8"))
    assert resumed["candidate_restored"] is True, "the second process restored the captured candidate"
    assert resumed["continued"] is True
    assert resumed["passed"] is True, resumed
    assert resumed["state"] == "completed", resumed
    assert resumed["executions"] == 2, f"the second process started a fresh job: {resumed}"

"""Real-process tests for the bounded host-side test runner."""

import asyncio
import os
import time
from pathlib import Path

import pytest

from daydream.test_execution import (
    MissingTestCommandError,
    TestExecutionResult,
    canonical_test_command,
    run_test_command,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True



def test_runner_returns_exit_status_cwd_and_merged_redacted_output(tmp_path: Path) -> None:
    async def go() -> TestExecutionResult:
        return await run_test_command(
            [
                "python",
                "-c",
                "import os,sys; print(os.getcwd()); print('hello-stdout');"
                " print('sec+REAL_SECRET+', file=sys.stderr)",
            ],
            cwd=tmp_path,
            wall_budget_s=10.0,
            env={"SOME_ENV": "REAL_SECRET"},
        )

    res = asyncio.run(go())
    assert isinstance(res, TestExecutionResult)
    assert res.exit_status == 0
    assert res.timed_out is False
    assert res.passed is True
    assert str(tmp_path) in res.merged_output  # ran in cwd
    assert "hello-stdout" in res.merged_output  # stdout captured
    assert "sec+REDACTED+" not in res.merged_output  # env secret scrubbed
    assert "REAL_SECRET" not in res.merged_output  # redacted before storage


def test_runner_nonzero_exit_sets_passed_false(tmp_path: Path) -> None:
    async def go() -> TestExecutionResult:
        return await run_test_command(
            ["python", "-c", "import sys; print('boom'); sys.exit(3)"],
            cwd=tmp_path,
            wall_budget_s=10.0,
        )

    res = asyncio.run(go())
    assert res.exit_status == 3
    assert res.timed_out is False
    assert res.passed is False
    assert "boom" in res.merged_output


def test_canonical_command_from_cli_overrides_config() -> None:
    from types import SimpleNamespace

    cfg = SimpleNamespace(test_command="pytest -x")  # config value
    run = SimpleNamespace(test_command="/cli/cmd")  # CLI wins
    assert canonical_test_command(cfg, run) == ["/cli/cmd"]


def test_canonical_command_from_config_when_cli_unset() -> None:
    from types import SimpleNamespace

    cfg = SimpleNamespace(test_command="uv run pytest -n auto")
    run = SimpleNamespace(test_command=None)
    assert canonical_test_command(cfg, run) == ["uv", "run", "pytest", "-n", "auto"]


def test_canonical_command_missing_fails_safely_with_diagnostic() -> None:
    from types import SimpleNamespace

    cfg = SimpleNamespace(test_command=None)
    run = SimpleNamespace(test_command=None)
    with pytest.raises(MissingTestCommandError) as e:
        canonical_test_command(cfg, run)
    msg = str(e.value)
    assert "test_command" in msg
    assert "tool.daydream" in msg  # naming the precedence source
    assert "--test-command" in msg  # actionable: what to set


def test_runner_timeout_kills_process_group_and_reports_timed_out(tmp_path: Path) -> None:
    marker = tmp_path / "grandkid.pid"

    async def go() -> TestExecutionResult:
        return await run_test_command(
            [
                "python",
                "-c",
                "import subprocess,os,time;"
                f"subprocess.Popen(['python','-c',"
                f"'import os,time;open(r\"{marker}\",\"w\").write(str(os.getpid()));time.sleep(30)']);"
                "time.sleep(30)",
            ],
            cwd=tmp_path,
            # Headroom for two cold interpreter startups + Popen + marker
            # write: a tighter budget made the marker write lose the race
            # to the group kill under CI load.
            wall_budget_s=5.0,
        )

    res = asyncio.run(go())
    assert res.timed_out is True
    assert res.passed is False
    # The grandchild writes its pid before sleeping; poll until it lands
    # instead of assuming startup beat the wall budget.
    deadline = time.monotonic() + 5.0
    pid: int | None = None
    while pid is None and time.monotonic() < deadline:
        try:
            pid = int(marker.read_text().strip())
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    assert pid is not None
    # grandchild must be dead, not reparented — poll a generous window
    deadline = time.monotonic() + 5.0
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _pid_alive(pid) is False


async def test_runner_records_duration_and_phase(tmp_path: Path) -> None:
    """Issue #726 task 12: with a trajectory recorder active, the host runner
    emits phase events distinguishable as ``test-execution``, carrying a
    ``duration_ms`` and a ``stop_reason`` in {completed, timed_out}."""
    from daydream.trajectory import DaydreamPhase
    from tests.harness.trajectory import make_recorder

    rec = make_recorder(tmp_path)
    async with rec:
        await run_test_command(["python", "-c", "pass"], cwd=tmp_path, wall_budget_s=10.0)

    events = [
        e
        for e in rec.phase_event_dicts()
        if e["phase"] == DaydreamPhase.TEST_EXECUTION.value and e["event"] == "phase_end"
    ]
    assert len(events) == 1
    md = events[0]["metadata"]
    assert md["stop_reason"] in {"completed", "timed_out"}
    assert md["duration_ms"] >= 0


async def test_runner_records_timed_out_stop_reason(tmp_path: Path) -> None:
    from daydream.trajectory import DaydreamPhase
    from tests.harness.trajectory import make_recorder

    rec = make_recorder(tmp_path)
    async with rec:
        await run_test_command(
            ["python", "-c", "import time; time.sleep(30)"], cwd=tmp_path, wall_budget_s=0.3
        )

    events = [
        e
        for e in rec.phase_event_dicts()
        if e["phase"] == DaydreamPhase.TEST_EXECUTION.value and e["event"] == "phase_end"
    ]
    assert len(events) == 1
    assert events[0]["metadata"]["stop_reason"] == "timed_out"


def test_runner_fails_closed_when_env_value_survives_scrub(tmp_path: Path) -> None:
    """An env value the replacement marker itself carries (a substring of
    "[REDACTED_ENV_VAR]") can never be scrubbed clean by replace(); the
    fail-closed gate -- keyed off the pre-replacement buffer -- degrades the
    whole field rather than emit a buffer that still shows the secret."""
    async def go() -> TestExecutionResult:
        return await run_test_command(
            ["python", "-c", "print('REDACTED', flush=True)"],
            cwd=tmp_path,
            wall_budget_s=10.0,
            env={"STUCK": "REDACTED"},
        )

    res = asyncio.run(go())
    assert res.passed is True
    assert res.merged_output == "[REDACTION_FAILED]"


def test_runner_scrubs_inherited_env_when_env_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production call sites pass no env, so the subprocess inherits the
    parent environment; the scrub must cover exactly those inherited values
    (the only env values that can appear in the merged output)."""
    secret = "ENV-SECRET-8f3a"
    monkeypatch.setenv("DAYDREAM_TEST_SECRET", secret)

    async def go() -> TestExecutionResult:
        return await run_test_command(
            [
                "python",
                "-c",
                "import os; print('value=' + os.environ['DAYDREAM_TEST_SECRET'], flush=True)",
            ],
            cwd=tmp_path,
            wall_budget_s=10.0,
        )

    res = asyncio.run(go())
    assert res.passed is True
    assert "value=" in res.merged_output
    assert secret not in res.merged_output

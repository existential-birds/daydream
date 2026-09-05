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
            wall_budget_s=0.5,
        )

    res = asyncio.run(go())
    assert res.timed_out is True
    assert res.passed is False
    pid = int(marker.read_text().strip())
    # grandchild must be dead, not reparented — poll a short window
    for _ in range(10):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    assert _pid_alive(pid) is False

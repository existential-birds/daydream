"""Real-process tests for the bounded host-side test runner."""

import asyncio

from daydream.test_execution import TestExecutionResult, run_test_command


def test_runner_returns_exit_status_cwd_and_merged_redacted_output(tmp_path):
    async def go():
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

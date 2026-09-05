"""Bounded host-side test runner.

Executes shell test commands as real subprocesses (issue #726): streaming +
redacting merged output, wall-budget timeout that kills the whole process
group via the existing ``backends/_subprocess.terminate_process``, returning a
typed :class:`TestExecutionResult` whose ``passed`` is derived only from exit
status. "Green" means the subprocess exited 0 — nothing else.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from daydream.backends._subprocess import terminate_process
from daydream.trajectory import redact_structured_text

_REDACTED_ENV_VAR = "[REDACTED_ENV_VAR]"


@dataclass
class TestExecutionResult:
    """Outcome of one host-side test-command run."""

    # Not a pytest test class despite the name prefix.
    __test__ = False

    exit_status: int
    timed_out: bool
    merged_output: str

    @property
    def passed(self) -> bool:
        """Single source of truth: exit status (a timeout is never a pass)."""
        return self.exit_status == 0 and not self.timed_out


def _redact_merged(output: str, env: dict[str, str] | None) -> str:
    """Scrub secret-shaped content and the literal env values from the buffer.

    The env vars handed to the test command are treated as sensitive: their
    values are redacted wherever they appear in the merged output, in addition
    to the structured pattern-based scrub. If a secret value still survives,
    the whole field degrades to ``[REDACTION_FAILED]`` (fail closed).
    """
    redacted = redact_structured_text(output)
    for value in (env or {}).values():
        if value and value in redacted:
            redacted = redacted.replace(value, _REDACTED_ENV_VAR)
    if any(value and value in redacted for value in (env or {}).values()):
        return "[REDACTION_FAILED]"
    return redacted


async def run_test_command(
    cmd: list[str],
    *,
    cwd: Path,
    wall_budget_s: float,
    env: dict[str, str] | None = None,
) -> TestExecutionResult:
    """Run *cmd* as a real subprocess with a wall-budget group kill.

    Spawns with ``start_new_session=True`` so the budget timeout can terminate
    the whole process group (reusing the shielded ``terminate_process``).
    Both pipes are streamed concurrently into one merged buffer; the merged
    output is redacted before it lands in the result. Spawn errors propagate —
    never a bogus default result.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    chunks: list[str] = []

    async def _pump(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            chunks.append(line.decode(errors="replace"))

    pump_stdout = asyncio.ensure_future(_pump(proc.stdout))
    pump_stderr = asyncio.ensure_future(_pump(proc.stderr))

    timed_out = False
    try:
        exit_status = await asyncio.wait_for(proc.wait(), wall_budget_s)
    except TimeoutError:
        timed_out = True
        await terminate_process(proc)
        # After the group kill the pipes hit EOF; give the pumps a moment to
        # drain whatever the process produced before it died.
        try:
            await asyncio.wait_for(
                asyncio.gather(pump_stdout, pump_stderr), timeout=5.0
            )
        except (TimeoutError, asyncio.CancelledError):  # noqa: BLE001 - salvage
            pass
        exit_status = proc.returncode if proc.returncode is not None else -1
    else:
        await asyncio.gather(pump_stdout, pump_stderr)
    return TestExecutionResult(
        exit_status=exit_status,
        timed_out=timed_out,
        merged_output=_redact_merged("".join(chunks), env),
    )

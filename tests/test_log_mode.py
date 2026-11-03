"""Exercise --log through runner.run with real Git/filesystem and a stub backend.

Assert plain tool/cost output, non-interactive compatibility, retained trajectories,
and the default Rich path.
"""
from __future__ import annotations

import io
import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import redirect_stdout
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ResultEvent,
    TextEvent,
    ToolStartEvent,
)
from daydream.phases.handoff import _emit_failure_handoff
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext, bind_run_context
from daydream.trajectory import DaydreamPhase
from daydream.ui.agent_stream import AgentDisplay
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]


_SECRET = "ghp_" + "x" * 16


async def test_log_mode_rendering_and_failure_handoff_redact_credentials(
    capsys: pytest.CaptureFixture[str], git_repo: Path, make_work: Callable[..., WorkContext],
) -> None:
    """Log-mode tool summaries redact before the display cap; a handoff body is redacted too.

    A credential that crosses the 200-character command cap must not leak its
    visible prefix, and the plain handoff print path shares the redaction policy.
    """
    display = AgentDisplay(Console(file=io.StringIO()), InteractionPolicy(log_mode=True), None, structured=False)
    command = "cd /srv/app && echo " + "a" * 156 + " --key " + _SECRET + " tail"
    for event in (ToolStartEvent('t', 'Bash', {'command': 'the config: token=opaque-summary-canary'}),
                  ToolStartEvent('cap', 'Bash', {'command': command})):
        await display.observe(event)
    await display.flush()
    display.tools.finish_all()
    output = capsys.readouterr().out
    assert '[REDACTED' in output and 'opaque-summary-canary' not in output and _SECRET[:8] not in output
    summary = next(line for line in output.splitlines() if 'cd /srv/app' in line)
    assert '[REDACTED' in summary

    backend = ScriptedBackend(events=[TextEvent(''), ResultEvent(
        structured_output={'handoff_prompt': f'token {_SECRET}'}, continuation=None)])
    with bind_run_context(RunContext(InteractionPolicy(log_mode=True))):
        await _emit_failure_handoff(backend, make_work(git_repo), 'failing test output',
                                   offer_clipboard=False, allow_standalone=True)
    captured = capsys.readouterr()
    assert _SECRET not in captured.out + captured.err and 'REDACTED' in captured.out + captured.err


@pytest.mark.parametrize(('credential_type', 'closed_pipe'), [('public', True)])
async def test_cancelled_agent_drains_complete_log_text_without_losing_cancellation(
    tmp_path: Path, credential_type: str, closed_pipe: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    secret = 'opaque-ui-cancel-native-auth-canary'
    fields = {'key': secret}
    content = ('ordinary public narration' if credential_type == 'public' else
               json.dumps({'type': credential_type, **fields, 'public': {'key': 'ordinary-public-key'}}))
    split = len(content) // 2
    events = [TextEvent(content[:split]), TextEvent(content[split:])]
    ready, closed = anyio.Event(), anyio.Event()
    captured: list[BaseException] = []

    async def respond(*_args: object) -> AsyncIterator[AgentEvent]:
        try:
            for event in events:
                yield event
            ready.set()
            await anyio.sleep_forever()
        finally:
            closed.set()

    backend = ScriptedBackend(responder=respond)

    async def invoke() -> None:
        try:
            await run_agent(backend, tmp_path, 'go', phase=DaydreamPhase.REVIEW)
        except BaseException as exc:
            captured.append(exc)
            raise

    async def cancel_after_observation() -> None:
        with bind_run_context(RunContext(InteractionPolicy(log_mode=True))):
            async with anyio.create_task_group() as group:
                group.start_soon(invoke)
                with anyio.fail_after(5):
                    await ready.wait()
                group.cancel_scope.cancel()

    if closed_pipe:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        # Unbuffered actual OS writes avoid a deferred context-close failure.
        with io.TextIOWrapper(io.FileIO(write_fd, 'w'), write_through=True) as stream:
            with redirect_stdout(stream):
                await cancel_after_observation()
    else:
        await cancel_after_observation()
    assert len(captured) == 1 and isinstance(captured[0], anyio.get_cancelled_exc_class())
    assert closed.is_set()
    output = capsys.readouterr().out
    assert secret not in output
    if closed_pipe:
        assert any('BrokenPipeError' in note for note in getattr(captured[0], '__notes__', []))
    elif credential_type == 'public':
        assert 'ordinary public narration' in output
    else:
        assert 'ordinary-public-key' in output and '[REDACTED' in output
    assert ''.join(event.text for event in events) == content

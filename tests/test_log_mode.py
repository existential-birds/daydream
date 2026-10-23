"""Exercise --log through runner.run with real Git/filesystem and a stub backend.

Assert plain tool/cost output, non-interactive compatibility, retained trajectories,
and the default Rich path.
"""
from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from daydream.agent import run_agent
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.commands.review import _parse_args
from daydream.phases.handoff import (
    _emit_failure_handoff,
)
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext, bind_run_context
from daydream.runner import run
from daydream.trajectory import DaydreamPhase
from daydream.ui.agent_stream import AgentDisplay
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend

# Synthetic credential-shaped sentinel: the ``{6,}`` API-key rule requires at
# least 6 chars after the ``ghp_`` prefix, so a truncated ``ghp_yyy`` fragment
# (only 3) is unmatchable — the discriminating boundary tests rely on this.
REDACTION_SENTINEL = "ghp_" + "x" * 16

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]


def _capture_stdout_and_run(config: RunConfig, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run daydream with the given config and capture stdout output."""
    # Mock external dependencies
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda repo, *, auth=None: ("test", "repo"))
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, _branch, *, auth=None: None)

    # Capture stdout
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        # Run the actual runner.run function
        exit_code = anyio.run(run, config)
        assert exit_code == 0, "Expected successful run"
    finally:
        sys.stdout = old_stdout

    return captured_output.getvalue()

@pytest.mark.parametrize(("events", "config_overrides", "required", "forbidden"),
    [
        pytest.param([ToolStartEvent('t', 'TaskCreate', {'subject': 'Add rate limiting',
            'description': 'Investigate and implement rate limiting for the API gateway',
            'prompt': 'Add rate limiting'})], {'log_mode': True}, ('[tool:TaskCreate] Add rate limiting',),
            ('Investigate and implement',), id='task-subject'),
        pytest.param([ToolStartEvent('t', 'Bash', {'command': 'the config: token=opaque-summary-canary'})],
            {'log_mode': True}, ('[REDACTED',), ('opaque-summary-canary',), id='structured-command'),
        pytest.param([ToolResultEvent('t', 'Authorization: Basic dXNlcjpwYXNzd29yZA== more', False)],
            {'log_mode': True}, ('[REDACTED',), ('dXNlcjpwYXNzd29yZA==',), id='basic-header-output'),
        pytest.param([ToolResultEvent('t', 'x' * 190 + ' ' + 'ghp_' + 'y' * 18, False)],
            {'log_mode': True}, ('[REDACTED',), ('ghp_',), id='output-crossing-cap'),
        pytest.param([TextEvent("hello world"), CostEvent(cost_usd=0.0042, input_tokens=100, output_tokens=50)],
            {"log_mode": False, "quiet": False, "output_mode": "review"}, (), ("[cost] $0.0042",), id="default-off",
        )
    ],
)
async def test_log_mode_rendering(
    capsys: pytest.CaptureFixture[str],
    events: list[AgentEvent], config_overrides: dict[str, object], required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> None:
    """Public AgentDisplay owns event routing; the CLI smoke owns runner integration."""
    display = AgentDisplay(Console(file=io.StringIO()),
        InteractionPolicy(log_mode=bool(config_overrides['log_mode'])), None, structured=False)
    for event in events:
        await display.observe(event)
    await display.flush()
    display.tools.finish_all()
    output = capsys.readouterr().out

    for substring in required:
        assert substring in output
    for substring in forbidden:
        assert substring not in output


@pytest.mark.parametrize("name", ["Bash", "shell"])
def test_log_mode_tool_event_redacts_credential_crossing_command_cap(
    tiny_diff_target: Path, make_config: MakeConfig, install_backend: InstallBackend, monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """The runner's actual log emission applies the command display policy."""
    cd_prefix = "cd /srv/app && "
    before_token = "echo " + "a" * (185 - len(cd_prefix) - 12) + " --key " if name == "Bash" else (
        "echo " + "a" * (185 - 12) + " --key "
    )
    command = (cd_prefix if name == "Bash" else "") + before_token + REDACTION_SENTINEL + " tail"
    if name == "shell":
        command = cd_prefix + command
    event = ToolStartEvent(id="boundary-command", name=name, input={"command": command})
    install_backend(
        ScriptedBackend(events=[event, TextEvent("Update the changed code."),
                                ResultEvent(structured_output=None, continuation=None)], retryable=False)
    )
    config = make_config(tiny_diff_target, non_interactive=True, log_mode=True, quiet=True, output_mode="review")

    output = _capture_stdout_and_run(config, monkeypatch)
    tool_line = next(line for line in output.splitlines() if line.startswith(f"[tool:{name}] "))
    summary = tool_line.removeprefix(f"[tool:{name}] ")
    assert len(summary) == 200
    assert "[REDACTED" in summary
    assert REDACTION_SENTINEL[:8] not in output
    assert (cd_prefix in summary) is (name == "Bash")
    assert event.input == {"command": command}





@pytest.mark.parametrize('credential_type', ['api_key', 'oauth', 'public'])
@pytest.mark.parametrize('closed_pipe', [False, True])
async def test_cancelled_agent_drains_complete_log_text_without_losing_cancellation(
    tmp_path: Path, credential_type: str, closed_pipe: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    """Graceful cancellation must drain the same complete block as normal flush.

    Rendering tests call flush explicitly, while run_agent cancellation skips
    its normal flush. Stream real backend events into the event loop, rendezvous
    after observation, then cancel. A real closed pipe must preserve cancellation
    rather than replacing it with a secondary presentation failure.
    """
    secret = 'opaque-ui-cancel-native-auth-canary'
    fields = ({'key': secret} if credential_type == 'api_key' else
              {'access': secret + '-access', 'refresh': secret + '-refresh'})
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






@pytest.mark.parametrize('log_mode', [False, True])
async def test_log_mode_failure_handoff_redacts_credential_body(
    git_repo: Path, make_work: Callable[..., WorkContext], capsys: pytest.CaptureFixture[str], log_mode: bool,
) -> None:
    """The actual failure handoff owns the mode-dependent console boundary."""
    backend = ScriptedBackend(events=[TextEvent(''), ResultEvent(
        structured_output={'handoff_prompt': f'token {REDACTION_SENTINEL}'}, continuation=None)])
    with bind_run_context(RunContext(InteractionPolicy(log_mode=log_mode))):
        await _emit_failure_handoff(backend, make_work(git_repo), 'failing test output',
                                   offer_clipboard=False, allow_standalone=True)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert (REDACTION_SENTINEL in output) is not log_mode
    if log_mode:
        assert 'REDACTED' in output


def test_verbose_flag_invocation_reaches_log_mode_pipeline(
    tiny_diff_target: Path, install_backend: InstallBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """One real verbose invocation presents all event kinds and retains its trajectory."""
    rejected_secret = "ghp_" + "z" * 16
    install_backend(ScriptedBackend(events=[
        TextEvent(f"token=hello {REDACTION_SENTINEL} world"),
        ThinkingEvent(f"thinking about {REDACTION_SENTINEL}"),
        ToolStartEvent(id="read", name="bash", input={"command": f"echo {REDACTION_SENTINEL}"}),
        ToolResultEvent(id="read", output=f"token={REDACTION_SENTINEL}", is_error=False),
        ToolStartEvent(id="error", name="bash", input={"command": "false", "description": "failing command"}),
        ToolResultEvent(id="error", output="command failed with exit code 1", is_error=True),
        MetricsEvent(message_id="test-msg", prompt_tokens=100, completion_tokens=50, cached_tokens=None, cost_usd=None),
        CostEvent(cost_usd=0.0042, input_tokens=100, output_tokens=50),
        ResultEvent(structured_output={"status": "complete", "token": rejected_secret}, continuation=None),
        TextEvent("Update the changed code."),
    ], retryable=False))
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda repo, **_kwargs: ("test", "repo"))
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, _branch, **_kwargs: None)
    config = _parse_args(["--verbose", "--review", str(tiny_diff_target)])
    assert config.log_mode is True and config.output_mode == "review"
    trajectory = tmp_path / "trajectory.json"
    config.trajectory_path = trajectory
    output = _capture_stdout_and_run(config, monkeypatch)
    assert all(value in output for value in (
        "hello", "[thinking] thinking about", "[tool:bash]", "[tool:bash result]", "[tool:bash ERROR] command failed",
        "[metrics] prompt=100 completion=50", "[cost] $0.0042", "[REDACTED_API_KEY]",
    ))
    agent_lines = [line for line in output.splitlines() if line.startswith(("token=hello ", "[thinking] ", "[tool:"))]
    assert all(line in agent_lines for line in (
        "token=hello [REDACTED_API_KEY] world", "[thinking] thinking about [REDACTED_API_KEY]",
        "[tool:bash] echo [REDACTED_API_KEY]", "[tool:bash result] token=[REDACTED_API_KEY]",
    ))
    assert REDACTION_SENTINEL not in "\n".join(agent_lines)
    assert all(value not in output for value in (rejected_secret, "[result]", "\x1b[", "[bold]", "[dim]"))
    document = json.loads(trajectory.read_text())
    assert document["steps"] and any(step["source"] == "agent" for step in document["steps"])
    assert atif_validate(document, validate_images=False)
    assert REDACTION_SENTINEL not in trajectory.read_text()

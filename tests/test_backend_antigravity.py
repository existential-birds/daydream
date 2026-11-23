"""The Antigravity adapter boundary, replayed from transcripts recorded on agy 1.5.15."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from daydream import clock
from daydream.backends import (
    AgentEvent,
    BackendExecutionInput,
    ContinuationToken,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    request_config_controls,
)
from daydream.backends.antigravity import AntigravityBackend, AntigravityError
from daydream.backends.lifecycle import BackendLifecycleCollector
from daydream.host_tools import HostToolBinding, HostToolFatal, UnsubmittedTurn
from tests.harness.antigravity_replay import AgyReplay, fixture_lines, install_agy

_FORMAT = ("--input-format", "stream-json", "--output-format", "stream-json")
_SCHEMA = {
    "type": "object",
    "properties": {"bug_found": {"type": "boolean"}, "line": {"type": "integer"}, "summary": {"type": "string"}},
    "required": ["bug_found", "line", "summary"], "additionalProperties": False,
}
_CONVERSATION = "00000000-0000-4000-8000-000000000001"
_TRANSCRIPT = fixture_lines("read_only_host_tool.jsonl")
_INIT = _TRANSCRIPT[0]
# Recorded on exit 3 for an agent whose tool list names an unregistered tool.
_AGY_ERROR = ('AGY_ERROR: {"short_error":"failed to construct executor: failed to resolve components: unknown '
              'component: tool \\"call_mcp_tool\\" not found in registry","status":"UNKNOWN","error_code":2,'
              '"code_kind":"grpc","retryable":false,"error_id":"e-1"}')


def _result(**fields: Any) -> str:
    usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cache_read_tokens": 0, "total_tokens": 0}
    return json.dumps({"event": "result", "result": {
        "conversation_id": _CONVERSATION, "response": "", "duration_seconds": 6.1, "num_turns": 1, "usage": usage,
        **fields}})


def _step(index: Any, step_type: str, state: str, **fields: Any) -> str:
    return json.dumps({"event": "step_update", "step_update": {
        "conversation_id": _CONVERSATION, "step_index": index, "state": state, "step_type": step_type, **fields}})


async def _events(backend: AntigravityBackend, cwd: Path, prompt: str, **options: Any) -> list[AgentEvent]:
    return [event async for event in backend.execute(cwd, prompt, **options)]


async def test_recorded_read_only_turn_becomes_events_and_per_step_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The recorded transcript maps to native tool events, streamed text and one usage record per step.

    Regressions: charging the cumulative ``result.usage`` again, inventing a cost, sending the prompt
    through argv, running the default agent, or leaving the per-run directory behind.
    """
    replay = install_agy(monkeypatch, _TRANSCRIPT)
    events = await _events(AntigravityBackend(), tmp_path, "Review calc.py", read_only=True)

    version_check, invocation = replay.spawner.argvs
    assert version_check == ("agy", "--version")
    assert invocation == ("agy", *_FORMAT, "--add-dir", str(replay.run_dirs[0]), "--agent", "daydream-review")
    stdin = replay.spawner.procs[1].stdin
    assert stdin.closed and stdin.data.endswith(b"\n") and stdin.data.count(b"\n") == 1
    assert json.loads(stdin.data) == {"event": "user", "message": {"content": "Review calc.py"}}
    assert replay.spawner.kwargs[1]["cwd"] == str(tmp_path)
    assert set(replay.run_files[0]) == {
        ".agents/agents/daydream-review.md", ".agents/hooks.json", "guard.py", "guard-policy.json"}
    assert replay.run_dir_modes == [0o700] and not replay.run_dirs[0].exists()

    request = events[0]
    assert isinstance(request, RequestEvent) and request.model_name is None
    assert request_config_controls(request.config) == {
        "finalization": None, "temperature": None, "read_only": True, "persist_session": True,
        "continuation_mode": "fresh", "model_mode": "single", "custom_agent": True, "tools_count": 4,
        "guard_hook": True, "skip_permissions": False, "native_output_schema": False, "host_mcp": False,
        "dedicated_home": False}
    version = next(event for event in events if isinstance(event, DiagnosticEvent))
    assert (version.code, version.metadata) == (
        "antigravity_cli_version", {"observed": "1.5.15", "verified": "1.5.15", "newer_than_verified": False})

    descriptor = {"Path": "/agy-home/.gemini/antigravity-cli/mcp/daydream/submit_review.json"}
    submission = {"Arguments": {"verdict": "bug"}, "ServerName": "daydream", "ToolName": "submit_review"}
    assert [(event.id, event.name, event.input) for event in events if isinstance(event, ToolStartEvent)] == [
        ("step-2", "read_file", descriptor), ("step-4", "read_file", {"Path": "/run-root/run/calc.py"}),
        ("step-6", "read_file", {"Path": "/run-root/ws/calc.py"}), ("step-8", "call_mcp_tool", submission)]
    assert [(event.id, event.output, event.is_error, event.status, event.duration_ms)
            for event in events if isinstance(event, ToolResultEvent)] == [
        ("step-2", "1 lines, 183 bytes", False, "done", 369.872),
        ("step-4", "file does not exist", True, "error", 101.651),
        ("step-6", "2 lines, 32 bytes", False, "done", 97.244),
        ("step-8", "accepted", False, "done", 92.549)]
    # The final message streamed as deltas, so the terminal response is not repeated.
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["I have", "\n"]

    metrics = [event for event in events if isinstance(event, MetricsEvent)]
    assert [event.message_id for event in metrics] == ["step-1", "step-3", "step-5", "step-7", "step-9"]
    # Exactly the recorded cumulative result.usage: per-step usage is charged once and the total never again.
    assert (sum(event.prompt_tokens for event in metrics), sum(event.completion_tokens for event in metrics),
            sum(event.cached_tokens or 0 for event in metrics)) == (11103, 311, 3197)
    assert {(event.cost_usd, event.model_name, event.reasoning_tokens) for event in metrics} == {
        (None, "gemini-3.8-flash-low", 0)}
    assert not any(isinstance(event, CostEvent) for event in events)

    result = events[-1]
    assert isinstance(result, ResultEvent)
    assert (result.structured_output, result.session_id, result.model_name, result.duration_ms) == (
        None, _CONVERSATION, "gemini-3.8-flash-low", 14150.832)
    assert result.continuation == ContinuationToken("antigravity", {"conversation_id": _CONVERSATION})


async def test_mutating_turns_pass_native_controls_and_return_schema_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A mutating turn resumes, selects model/effort/home and returns agy's schema output.

    Regressions: a dedicated agy home read from process globals or not applied, the version check
    repeated per invocation, a mutating turn without the permission flag, or ``persist_session=False``
    resuming or handing back a continuation.
    """
    structured = fixture_lines("structured_output.jsonl")
    replay = install_agy(monkeypatch, structured, structured)
    home = tmp_path / "agy-home"
    monkeypatch.setenv("DAYDREAM_ANTIGRAVITY_HOME", "/ambient-home-must-not-be-read")
    environment = {"PATH": os.environ["PATH"], "HOME": "/operator-home", "DAYDREAM_ANTIGRAVITY_HOME": str(home)}
    backend = AntigravityBackend("gemini-3.8-pro", "low", execution_input=BackendExecutionInput.from_environment(
        environment, backend="antigravity"))
    assert backend.fanout_concurrency == 2
    token = ContinuationToken("antigravity", {"conversation_id": "previous"})

    events = await _events(backend, tmp_path, "Fix calc.py", output_schema=_SCHEMA, continuation=token)
    ephemeral = await _events(backend, tmp_path, "Again", continuation=token, persist_session=False)

    resumed, fresh = replay.invocations
    assert len(replay.spawner.argvs) == 3  # one version check for the backend instance
    assert resumed == (
        "agy", *_FORMAT, "--model", "gemini-3.8-pro", "--effort", "low", "--add-dir", str(replay.run_dirs[0]),
        "--agent", "daydream-edit", "--json-schema", json.dumps(_SCHEMA, separators=(",", ":")),
        "--conversation", "previous", "--dangerously-skip-permissions")
    assert fresh == ("agy", *_FORMAT, "--model", "gemini-3.8-pro", "--effort", "low", "--add-dir",
                     str(replay.run_dirs[1]), "--agent", "daydream-edit", "--dangerously-skip-permissions")
    assert [kwargs["env"]["HOME"] for kwargs in replay.spawner.kwargs] == [str(home)] * 3

    request, result = events[0], events[-1]
    assert isinstance(request, RequestEvent) and isinstance(result, ResultEvent)
    assert (request.session_id, request.config.continuation_mode, request.reasoning_effort) == (
        "previous", "resume", "low")
    assert result.structured_output == {"bug_found": True, "line": 2, "summary": "add subtracts"}
    assert result.continuation == ContinuationToken(
        "antigravity", {"conversation_id": "00000000-0000-4000-8000-000000000002"})
    # Nothing streamed, so the terminal response is the turn's only text.
    assert [event.text for event in events if isinstance(event, TextEvent)] == [
        '{"bug_found":true,"line":2,"summary":"add subtracts"}\n']
    last = ephemeral[-1]
    assert isinstance(last, ResultEvent) and last.continuation is None and last.structured_output is None


@pytest.mark.parametrize(("version", "refusal"), [
    ("1.5.14", "older than the verified floor"),
    ("agy 1.5.15 (stable)", "did not report a bare X.Y.Z"),
    (None, "could not be started"),
    ("1.6.0", None),
])
async def test_version_floor_is_checked_before_the_first_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str | None, refusal: str | None,
) -> None:
    """Builds below 1.5.15, or unidentifiable ones, are a non-retryable configuration refusal; newer ones proceed.

    Regression: agy's background updater changes the build between runs, so an unverified older
    contract must stop before any prompt is sent, and a newer one must be visible in evidence.
    """
    replay = install_agy(monkeypatch, _TRANSCRIPT, version=version or "")
    if version is None:
        async def missing(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("agy")

        monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", missing)
    backend = AntigravityBackend()
    if refusal is None:
        events = await _events(backend, tmp_path, "Review", read_only=True)
        (diagnostic,) = (event for event in events if isinstance(event, DiagnosticEvent))
        assert diagnostic.metadata == {"observed": "1.6.0", "verified": "1.5.15", "newer_than_verified": True}
        assert isinstance(events[-1], ResultEvent)
        return
    with pytest.raises(AntigravityError, match=refusal) as raised:
        await _events(backend, tmp_path, "Review", read_only=True)
    assert "requires agy 1.5.15 or newer" in str(raised.value)
    assert (raised.value.category, raised.value.retryable) == ("AUTH_CONFIG", False)
    assert replay.invocations == []


@pytest.mark.parametrize(("lines", "exit_code", "stderr", "category", "retryable", "message"), [
    pytest.param([_result(status="ERROR", error="interrupted")], 1, ["error: interrupted"],
                 "CANCELLED", False, "interrupted", id="interrupt"),
    pytest.param([_result(status="ERROR", error="authentication failed or timed out")], 1,
                 ["Sign in at https://accounts.example/o/oauth2/auth"],
                 "AUTH_CONFIG", False, "DAYDREAM_ANTIGRAVITY_HOME", id="no-credentials"),
    pytest.param([], 3, [_AGY_ERROR], "PROCESS_EXIT", False, "unknown component", id="startup-not-retryable"),
    pytest.param([], 3, [_AGY_ERROR.replace('"retryable":false', '"retryable":true')],
                 "PROCESS_EXIT", True, "return code 3", id="startup-retryable"),
    pytest.param(_TRANSCRIPT[:5], 0, None, "TRANSPORT", None, "without a terminal result", id="truncated"),
    pytest.param([_INIT, _result(status="ERROR", error="quota exhausted")], 0, None,
                 "TRANSPORT", False, "quota exhausted", id="error-status"),
    pytest.param([_TRANSCRIPT[-1]], 0, None, "TRANSPORT", False, "did not succeed", id="success-without-init"),
    pytest.param([_INIT.replace("daydream-review", "default"), *_TRANSCRIPT[1:]], 0, None,
                 "TRANSPORT", False, "custom agent", id="default-agent"),
    pytest.param(_TRANSCRIPT[1:], 0, None, "TRANSPORT", False, "before naming its agent", id="step-before-init"),
])
async def test_unsuccessful_turns_are_classified_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lines: list[str], exit_code: int, stderr: list[str] | None,
    category: str, retryable: bool | None, message: str,
) -> None:
    """Only SUCCESS from the Daydream agent with a zero exit completes; every other ending raises its class.

    Regressions: an interrupt or missing sign-in retried as a transient failure, agy's own ``retryable``
    verdict ignored, a truncated stream or the default agent reported as a completed turn.
    """
    replay = install_agy(monkeypatch, lines, exit_code=exit_code, stderr_lines=stderr)
    with pytest.raises(AntigravityError, match=message) as raised:
        await _events(AntigravityBackend(), tmp_path, "Review", read_only=True)
    assert (raised.value.category, getattr(raised.value, "retryable", None)) == (category, retryable)
    assert replay.spawner.procs[1].reaped and not replay.run_dirs[0].exists()


async def test_shapes_outside_the_recorded_contract_are_reported_not_interpreted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Unknown or malformed stream shapes become one parser-coverage diagnostic and no invented events.

    Regression: a newer agy build adding shapes must neither crash the turn nor be guessed into
    tool calls or usage; a completed tool step whose start was never seen still correlates.
    """
    tool = {"name": "list_dir", "parameters": {"DirectoryPath": "."}}
    replay = install_agy(monkeypatch, [
        _INIT, "", "not json", '["event"]', '{"event":"notice","notice":{"text":"update available"}}',
        _step(1, "planning", "DONE"),
        _step(2, "tool", "PAUSED", tool_name="list_dir", tool_info=tool),
        _step(3, "tool", "DONE", tool_info={"parameters": {}}),
        _step("4", "tool", "DONE", tool_name="list_dir", tool_info=tool),
        _step(5, "agent_response", "DONE", usage={"input_tokens": "many", "output_tokens": 1}),
        _step(6, "tool", "DONE", tool_name="list_dir", tool_info=tool, duration_seconds=True),
        _result(status="SUCCESS", denied_actions=[{"action": "shell", "display_name": "Shell"}]),
        _step(7, "finish", "DONE"),
    ])
    events = await _events(AntigravityBackend(), tmp_path, "Review", read_only=True)

    diagnostics = {event.code: event.metadata for event in events if isinstance(event, DiagnosticEvent)}
    assert diagnostics["antigravity_parser_coverage"] == {"reasons": {
        "non_json_line": 1, "malformed_event": 1, "unknown_event": 1, "unknown_step_type": 1,
        "unknown_tool_state": 1, "malformed_step": 2, "malformed_usage": 1, "event_after_result": 1}}
    # Without host tools an agy-side denial is evidence, not a failed turn.
    assert diagnostics["antigravity_denied_actions"] == {"actions": {"shell": 1}, "total": 1}
    assert [(type(event).__name__, event.id) for event in events
            if isinstance(event, (ToolStartEvent, ToolResultEvent))] == [
        ("ToolStartEvent", "step-6"), ("ToolResultEvent", "step-6")]
    assert not any(isinstance(event, MetricsEvent) for event in events)
    assert isinstance(events[-1], ResultEvent) and replay.spawner.procs[1].reaped


async def test_unsupported_requests_start_no_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Host tools need a fresh read-only schema turn, and subagents are refused, before agy is launched.

    Regression: a mutating or resumed host-tool turn, or instructions dropped silently without host
    tools, would run the model outside the committed-review boundary.
    """
    replay = install_agy(monkeypatch, _TRANSCRIPT)
    backend = AntigravityBackend()
    binding = HostToolBinding(deadline=asyncio.get_running_loop().time() + 30, accept=lambda value: value)
    token = ContinuationToken("antigravity", {"conversation_id": "previous"})
    unsupported: tuple[dict[str, Any], ...] = (
        {"host_tools": binding}, {"host_tools": binding, "read_only": True, "continuation": token},
        {"host_tools": binding, "read_only": True, "agents": {"explorer": {}}},
        {"host_tools": binding, "read_only": True, "output_schema": None}, {"review_instructions": "policy"})
    for arguments in unsupported:
        with pytest.raises(ValueError, match="fresh read-only invocation|only with host tools"):
            await _events(backend, tmp_path, "Review", **{"output_schema": _SCHEMA, **arguments})
    with pytest.raises(NotImplementedError, match="exploration subagents"):
        await _events(backend, tmp_path, "Explore", agents={"explorer": {}})
    assert replay.spawner.argvs == [] and binding.schema is None


async def _host_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binding: HostToolBinding, lines: list[str], *,
    list_tools: bool, submission: dict[str, Any] | None,
) -> tuple[AgyReplay, list[AgentEvent], asyncio.Task[None]]:
    """Run one host-tool invocation while a scripted MCP client plays agy against the per-run config."""
    replay = install_agy(monkeypatch, [], hang=True)
    events: list[AgentEvent] = []

    async def consume() -> None:
        async for event in AntigravityBackend().execute(
                tmp_path, "Review calc.py", output_schema=_SCHEMA, read_only=True,
                review_instructions="HOST POLICY", host_tools=binding):
            events.append(event)

    turn = asyncio.create_task(consume())
    await replay.spawner.wait_for_read(1)
    server = json.loads(replay.run_files[0][".agents/mcp_config.json"])["mcpServers"]["daydream"]
    if list_tools:
        async with (
            httpx2.AsyncClient(headers=server["headers"], timeout=30.0) as http_client,
            streamable_http_client(server["url"], http_client=http_client) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            assert [tool.name for tool in (await session.list_tools()).tools] == ["structured_output"]
            if submission is not None:
                assert not (await session.call_tool("structured_output", submission)).is_error
    replay.spawner.procs[1].finish(lines)
    return replay, events, turn


async def test_host_tool_turn_settles_on_the_accepted_submission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The host-accepted submission is the result, settled after SUCCESS, EOF, zero exit and cleanup.

    Regressions: the bearer reaching argv, stdin or the child environment; instructions outside the
    agent file; agy-owned tool starts left uncharged or the host's own call charged twice.
    """
    accepted: list[Any] = []

    def accept(value: Any) -> dict[str, Any]:
        accepted.append(value)
        return {"accepted": True}

    binding = HostToolBinding(deadline=asyncio.get_running_loop().time() + 30, accept=accept)
    lifecycle = BackendLifecycleCollector(attempt=1, started_at=clock.monotonic(), wall_allowance_s=None,
                                          deadline=None)
    bound = lifecycle.bind()
    try:
        submission = {"bug_found": True, "line": 2, "summary": "add subtracts"}
        replay, events, turn = await _host_turn(
            monkeypatch, tmp_path, binding, _TRANSCRIPT, list_tools=True, submission=submission)
        await turn
    finally:
        BackendLifecycleCollector.reset(bound)

    result = events[-1]
    assert isinstance(result, ResultEvent) and result.structured_output == {"accepted": True}
    assert accepted == [submission] and binding.native_settled and binding.cleanup_complete
    # The submission and agy's three own reads; the transcript's host call is not charged again.
    assert (binding.calls, binding.budget.charged) == (1, 4)
    snapshot = lifecycle.finalize()
    assert snapshot is not None
    assert (snapshot.backend, snapshot.host_native_settled, snapshot.process_returncode) == ("antigravity", True, 0)
    assert None not in (snapshot.terminal_received_s, snapshot.stdout_eof_s, snapshot.teardown_completed_s)

    (argv,) = replay.invocations
    assert argv == ("agy", *_FORMAT, "--add-dir", str(replay.run_dirs[0]), "--agent", "daydream-review",
                    "--settings", '{"permissions":{"allow":["mcp(daydream/*)"]}}')
    files, environment = replay.run_files[0], replay.spawner.kwargs[1]["env"]
    bearer = json.loads(files[".agents/mcp_config.json"])["mcpServers"]["daydream"]["headers"]["Authorization"]
    token = bearer.removeprefix("Bearer ")
    assert token and token != bearer
    exposed = (*argv, *environment.values(), replay.spawner.procs[1].stdin.data.decode(),
               *(text for name, text in files.items() if name != ".agents/mcp_config.json"))
    assert not any(token in text for text in exposed)
    assert "127.0.0.1" in environment["NO_PROXY"].split(",")
    agent = files[".agents/agents/daydream-review.md"]
    assert agent.endswith("# System Prompt\n\nHOST POLICY\n") and "  - call_mcp_tool\n" in agent
    request = events[0]
    assert isinstance(request, RequestEvent) and request.system_prompt == "HOST POLICY"
    controls = request_config_controls(request.config)
    assert (controls["host_mcp"], controls["native_output_schema"]) == (True, False)
    assert not replay.run_dirs[0].exists()


@pytest.mark.parametrize(("list_tools", "fixture", "error", "message"), [
    pytest.param(False, "read_only_host_tool.jsonl", HostToolFatal, "did not list the host tools", id="not-ready"),
    pytest.param(True, "denied_mcp_call.jsonl", AntigravityError, "denied a host tool call", id="denied-action"),
    pytest.param(True, "read_only_host_tool.jsonl", UnsubmittedTurn, "without an accepted", id="unsubmitted"),
])
async def test_host_tool_turn_without_acceptance_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, list_tools: bool, fixture: str, error: type[Exception],
    message: str,
) -> None:
    """A SUCCESS turn is not a review: unlisted tools, an agy-side MCP denial or no submission each fail.

    Regression: agy exits 0 with ``status: SUCCESS`` when it auto-denies the host call, so trusting the
    status alone would report an empty review as complete.
    """
    binding = HostToolBinding(deadline=asyncio.get_running_loop().time() + 30, accept=lambda value: value)
    replay, events, turn = await _host_turn(
        monkeypatch, tmp_path, binding, fixture_lines(fixture), list_tools=list_tools, submission=None)
    with pytest.raises(error, match=message) as raised:
        await turn
    assert not binding.native_settled and not binding.accepted
    assert getattr(raised.value, "code", None) == ("native_readiness" if error is HostToolFatal else None)
    # Only the unsubmitted turn reaches its terminal event; the others stop before any result.
    assert isinstance(events[-1], ResultEvent) is (error is UnsubmittedTurn)
    assert replay.spawner.procs[1].reaped and not replay.run_dirs[0].exists()


async def test_cancelling_a_turn_reaps_its_process_and_removes_its_run_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cancelling one stream terminates only that invocation's process and deletes its private directory.

    Regression: a cancelled attempt leaving agy running or its policy and bearer directory on disk.
    """
    replay = install_agy(monkeypatch, [_INIT], hang=True)
    turn = asyncio.create_task(_events(AntigravityBackend(), tmp_path, "Review", read_only=True))
    await replay.spawner.wait_for_read(1)
    assert replay.run_dirs[0].exists()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    process = replay.spawner.procs[1]
    assert process.terminate_calls == 1 and process.reaped and not replay.run_dirs[0].exists()

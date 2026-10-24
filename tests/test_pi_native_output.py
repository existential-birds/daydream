"""Exercise the packaged native output extension with the installed Pi CLI."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from mcp.types import Tool

from daydream.agent import run_agent
from daydream.backends import (
    DiagnosticEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends._subprocess import StreamStalledError
from daydream.backends._transport import CliTransport
from daydream.backends.pi import PiBackend, PiError
from daydream.host_tools import NATIVE_READ_FILE, READ_FILE, HostToolBinding, HostToolLimits
from daydream.host_tools_server import PiHostToolServer, _ReadServer
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_clock import FakeClock
from tests.harness.otlp import _loopback_http_server, _QuietHTTPHandler
from tests.harness.trajectory import make_recorder

_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}}, "required": ["verdict", "findings"]}


@dataclass
class _NativeProvider:
    config: Path
    requests: list[dict[str, Any]] = field(default_factory=list)
    turns: list[list[tuple[str, str, dict[str, Any]]]] = field(default_factory=list)
    choose: Callable[[dict[str, Any]], list[tuple[str, str, dict[str, Any]]]] | None = None


@pytest.fixture
def native_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_NativeProvider]:
    """Real installed Pi talks only to a counted, scripted loopback provider."""
    if shutil.which("pi") is None or shutil.which("node") is None:
        pytest.skip("The installed official Pi CLI and Node are required for native tool loading.")
    config = tmp_path / "isolated-host-pi"
    config.mkdir()
    (config / "settings.json").write_text(json.dumps({"retry": {"enabled": False, "maxRetries": 0}}))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(config))
    for name in ("PI_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    fixture = _NativeProvider(config)

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fixture.requests.append(request)
            index = len(fixture.requests) - 1
            batch = (fixture.choose(request) if fixture.choose is not None else
                     fixture.turns[index] if index < len(fixture.turns) else [])
            delta: dict[str, Any] = {"role": "assistant", "content": f"Native reasoning {index}."}
            if batch:
                delta["tool_calls"] = [{"index": i, "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}
                    for i, (call_id, name, arguments) in enumerate(batch)]
            chunks: list[dict[str, Any]] = [
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if batch else "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps({"id": f"host-output-{index}",
                    "object": "chat.completion.chunk", "created": 1, "model": "output-model", **chunk})
                    + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / "models.json").write_text(json.dumps({"providers": {"output-fixture": {
            "baseUrl": f"{base_url}/v1", "api": "openai-completions", "apiKey": "synthetic-loopback-only",
            "models": [{"id": "output-model", "reasoning": False, "input": ["text"],
                        "contextWindow": 32768, "maxTokens": 8192}]}}}))
        monkeypatch.setenv("PI_PROVIDER", "output-fixture")
        yield fixture


@pytest.mark.parametrize("read_only", [False, True], ids=["ordinary", "read-only"])
async def test_native_host_rejection_correction_records_exact_bytes_without_output_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, read_only: bool,
) -> None:
    """MCP authority rejects an existing forbidden file and permits correction in the same allowance.

    Structured-output and replay tests miss actual native MCP exposure, correlated rejection history,
    exact host bytes, prospective charging and recorder finalization without an output schema.
    A shared fake clock also catches real-time arithmetic rejecting a valid host deadline during native setup.
    """
    fake = FakeClock(monotonic_value=100).install(monkeypatch)
    root = tmp_path.resolve()
    allowed, forbidden = root / "permitted.txt", root / "forbidden.txt"
    expected = "Independently known bytes: café 界\nsecond line\n"
    allowed.write_bytes(expected.encode())
    forbidden.write_text("FORBIDDEN_EXISTING_FILE_MUST_NOT_BE_READ")
    native_provider.turns = [[("denied", NATIVE_READ_FILE, {"path": str(forbidden)})],
                             [("corrected", NATIVE_READ_FILE, {"path": str(allowed)})]]
    binding = HostToolBinding((allowed,), deadline=fake.monotonic() + 30,
                              limits=HostToolLimits(max_calls=2))
    recorder = make_recorder(root, agent_model_name="output-model", backend_name="pi")
    async with recorder:
        result, _, reason = await run_agent(PiBackend(model="output-model"), root, "Read the granted file.",
            phase=DaydreamPhase.REVIEW, read_only=read_only, persist_session=False, tool_call_budget=2,
            host_tools=binding, wall_budget_s=30, progress_callback=lambda _: None)

    assert reason is None and "Native reasoning 2." in result
    assert len(native_provider.requests) == 3
    tool = next(tool for tool in native_provider.requests[0]["tools"]
                if tool["function"]["name"] == NATIVE_READ_FILE)
    assert tool["function"]["parameters"] == READ_FILE.schema()
    tool_names = {tool["function"]["name"] for tool in native_provider.requests[0]["tools"]}
    if read_only:
        assert {"read", "grep", "find", "ls"} <= tool_names
        assert not {"bash", "edit", "write"} & tool_names
    else:
        assert {"read", "bash", "edit", "write"} <= tool_names
    for index, request in enumerate(native_provider.requests):
        assert len([message for message in request["messages"] if message["role"] == "user"]) == 1
        history = {message["tool_call_id"]: message["content"] for message in request["messages"]
                   if message["role"] == "tool"}
        assert set(history) == (set() if index == 0 else {"denied"} if index == 1 else {"denied", "corrected"})
        if index:
            assert "Path is not permitted by this invocation" in history["denied"]
        if index == 2:
            assert history["corrected"] == expected
    trajectory = json.loads(recorder.path.read_text())
    calls = [call for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    observations = {item["source_call_id"]: item["content"] for step in trajectory["steps"]
                    for item in (step.get("observation") or {}).get("results", [])}
    assert [call["tool_call_id"] for call in calls] == ["denied", "corrected"]
    assert "Path is not permitted by this invocation" in observations["denied"]
    recorded_result = json.loads(observations["corrected"])
    assert recorded_result["content"] == [{"type": "text", "text": expected}]
    assert recorded_result["structuredContent"]["isError"] is False
    assert recorded_result["structuredContent"]["content"] == [{"type": "text", "text": expected}]
    assert "FORBIDDEN_EXISTING_FILE_MUST_NOT_BE_READ" not in json.dumps(trajectory)
    assert binding.calls == 2 and binding.rejections == 1
    assert not binding.authority_open and binding.native_settled and binding.cleanup_complete
    lifecycle = trajectory["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["host_authority_closed"] is True
    assert lifecycle["host_native_settled"] is True
    assert lifecycle["host_cleanup_complete"] is True
    assert lifecycle["host_tool_call_count"] == 2 and lifecycle["host_tool_rejection_count"] == 1
    assert sum((step.get("metrics") or {}).get("prompt_tokens", 0) or 0 for step in trajectory["steps"]) == 30


@pytest.mark.parametrize("sdk", ["missing", "2.0.0"])
async def test_required_native_host_binding_rejects_incompatible_optional_sdk_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, sdk: str,
) -> None:
    """An explicit binding fails actionably before native/provider work when its SDK is unavailable.

    The ordinary fresh-process import test protects optionality, but cannot prove fail-closed required behavior.
    """
    def installed_version(distribution: str) -> str:
        if sdk == "missing":
            raise PackageNotFoundError(distribution)
        return sdk

    monkeypatch.setattr("daydream.backends.pi.version", installed_version)
    binding = HostToolBinding((tmp_path.resolve() / "source",), deadline=time.monotonic() + 10)
    backend = PiBackend(model="output-model")
    with pytest.raises(PiError, match=r"daydream\[host-tools\].*mcp==1\.30\.0") as failure:
        await run_agent(backend, tmp_path, "Require the optional host operation.", phase=DaydreamPhase.REVIEW,
                        host_tools=binding, persist_session=False, progress_callback=lambda _: None)
    assert failure.value.category == "HOST_TOOLS"
    assert native_provider.requests == []
    assert backend._transports == [] and binding.calls == 0
    assert not binding.authority_open and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["missing-tool", "wrong-schema", "slow-list", "disabled", "missing-init",
                                  "failed-init", "slow-init", "missing-extension", "broken-extension",
                                  "unsupported-capability", "unsupported-version", "unsupported-platform"])
async def test_native_host_readiness_failure_never_reaches_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, fault: str,
) -> None:
    """Actual native loadout/setup faults must stop before a counted provider request.

    A host self-probe or argv assertion misses disabled/absent tools and swallowed extension errors.
    """
    root = tmp_path.resolve()
    allowed = root / "permitted.txt"
    allowed.write_text("permitted")
    binding = HostToolBinding((allowed,), deadline=time.monotonic() + 15,
        limits=HostToolLimits(setup_timeout_s=0.8, callback_timeout_s=1, cleanup_timeout_s=0.8))
    reached: list[str] = []
    if fault in {"missing-tool", "wrong-schema", "slow-list"}:
        async def list_tools(self: Any) -> list[Tool]:
            reached.append(fault)
            if fault == "slow-list":
                await asyncio.Event().wait()
            return [] if fault == "missing-tool" else [Tool(name=READ_FILE.name,
                inputSchema={"type": "object", "properties": {"wrong": {"type": "string"}}})]
        monkeypatch.setattr(_ReadServer, "list_tools", list_tools)
    elif fault == "disabled":
        start = PiHostToolServer.start

        async def start_disabled(server: PiHostToolServer) -> None:
            await start(server)
            reached.append(fault)
            (native_provider.config / "mcp.json").write_text(json.dumps({"mcpServers": {"daydream_host": {
                "url": server.url + "/mcp", "enabled": False}}}))
        monkeypatch.setattr(PiHostToolServer, "start", start_disabled)
    elif fault in {"missing-init", "failed-init", "slow-init"}:
        handle = PiHostToolServer.__call__

        async def faulty_initialize(server: PiHostToolServer, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] != "http" or scope["path"] != "/mcp" or scope["method"] != "POST":
                await handle(server, scope, receive, send)
                return
            message = await receive()
            value = json.loads(message.get("body", b"{}"))
            if value.get("method") == "initialize":
                reached.append(fault)
                if fault == "slow-init":
                    await asyncio.Event().wait()
                body = json.dumps({"jsonrpc": "2.0", "id": value.get("id"),
                    "error": {"code": -32603, "message": "Initialization deliberately failed"}}).encode()
                await send({"type": "http.response.start", "status": 404 if fault == "missing-init" else 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
            first = True

            async def replay() -> Any:
                nonlocal first
                if first:
                    first = False
                    return message
                return await receive()
            await handle(server, scope, replay, send)
        monkeypatch.setattr(PiHostToolServer, "__call__", faulty_initialize)
    elif fault in {"missing-extension", "broken-extension", "unsupported-capability"}:
        assets = root / "faulty-assets"
        assets.mkdir()
        if fault == "broken-extension":
            (assets / "pi_host_tools.ts").write_text("this is an invalid TypeScript module {{")
        elif fault == "unsupported-capability":
            source = files("daydream.backends").joinpath("pi_host_tools.ts").read_text()
            source = source.replace("export default function (pi: ExtensionAPI) {",
                "export default function (pi: ExtensionAPI) {\n  (pi as any).registerMcpServer = undefined;")
            (assets / "pi_host_tools.ts").write_text(source)
        monkeypatch.setattr("daydream.backends.pi.files", lambda _: assets)
    elif fault == "unsupported-platform":
        monkeypatch.delattr("daydream.host_tools.os.O_NOFOLLOW")
    elif fault == "unsupported-version":
        native_pi = shutil.which("pi")
        assert native_pi is not None
        shim = root / "bin"
        shim.mkdir()
        executable = shim / "pi"
        executable.write_text(f"#!{sys.executable}\nimport os, sys\n"
            "if sys.argv[1:] == ['--version']:\n print('1.0.4')\n"
            f"else:\n os.execv({native_pi!r}, [{native_pi!r}, *sys.argv[1:]])\n")
        executable.chmod(0o700)
        monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    started = time.monotonic()
    with pytest.raises((PiError, StreamStalledError, NotImplementedError)) as failure:
        async with asyncio.timeout(8):
            await run_agent(PiBackend(model="output-model"), root, "Must fail before model work.",
                phase=DaydreamPhase.REVIEW, persist_session=False, host_tools=binding,
                progress_callback=lambda _: None)
    assert native_provider.requests == []
    if fault in {"missing-tool", "wrong-schema", "slow-list", "disabled", "missing-init", "failed-init", "slow-init"}:
        assert reached, "the intended native readiness fault must actually be exercised"
    if fault == "unsupported-capability":
        assert isinstance(failure.value, PiError) and "binding is unavailable" in str(failure.value)
    elif fault == "unsupported-version":
        assert isinstance(failure.value, PiError) and "verified Pi 1.4.0" in str(failure.value)
    elif fault == "unsupported-platform":
        assert isinstance(failure.value, NotImplementedError)
    assert time.monotonic() - started < 8
    assert binding.calls == 0 and not binding.authority_open and not binding.native_settled


async def test_native_same_schema_config_override_cannot_impersonate_intended_server(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    """Native initialize/list must belong to this invocation, even when an override's schema matches.

    Missing-tool/schema failures cannot detect accepting another endpoint with identical declarations.
    """
    root = tmp_path.resolve()
    allowed = root / "permitted.txt"
    allowed.write_text("never requested")
    limits = HostToolLimits(setup_timeout_s=0.8, cleanup_timeout_s=1)
    other = HostToolBinding((allowed,), deadline=time.monotonic() + 15, limits=limits)
    override = PiHostToolServer(other)
    await override.start()
    try:
        packet = override.packet()
        (native_provider.config / "mcp.json").write_text(json.dumps({"mcpServers": {"daydream_host": {
            "url": override.url + "/mcp", "headers": {"Authorization": packet["authorization"]},
            "exposure": "direct"}}}))
        binding = HostToolBinding((allowed,), deadline=time.monotonic() + 15, limits=limits)
        with pytest.raises(PiError, match="binding is unavailable"):
            await run_agent(PiBackend(model="output-model"), root, "Require intended endpoint.",
                phase=DaydreamPhase.REVIEW, host_tools=binding, persist_session=False,
                progress_callback=lambda _: None)
        assert override.initialized and override.listed
        assert native_provider.requests == [] and binding.calls == 0
        assert not binding.authority_open and binding.cleanup_complete
    finally:
        await override.close()


@pytest.mark.parametrize("mode", ["tools-disabled", "finalization", "continuation", "unsupported-backend"])
async def test_required_host_binding_rejects_unsupported_mode_before_provider(
    tmp_path: Path, native_provider: _NativeProvider, mode: str,
) -> None:
    """Explicit host requirements cannot silently degrade to ordinary or tool-free calls.

    Existing no-binding mode tests prove legacy behavior, but never request required MCP authority.
    """
    from daydream.backends import ContinuationToken
    from tests.harness.backend import ScriptedBackend

    binding = HostToolBinding((tmp_path.resolve() / "file.txt",), deadline=time.monotonic() + 10)
    if mode == "finalization":
        with pytest.raises(PiError, match="tools-enabled"):
            _ = [event async for event in PiBackend(model="output-model").execute(
                tmp_path, "Serialize", finalization=True, host_tools=binding)]
    else:
        backend = ScriptedBackend() if mode == "unsupported-backend" else PiBackend(model="output-model")
        options: dict[str, Any] = {"tools_disabled": True} if mode == "tools-disabled" else (
            {"continuation": ContinuationToken(backend="pi", data={"session_id": "prior"})}
            if mode == "continuation" else {})
        with pytest.raises((ValueError, NotImplementedError), match="host tools|host-tool|host tools|Host"):
            await run_agent(backend, tmp_path, "Must not dispatch", phase=DaydreamPhase.REVIEW,
                            host_tools=binding, **options)
        if isinstance(backend, ScriptedBackend):
            assert backend.calls == []
    assert native_provider.requests == [] and binding.calls == 0 and not binding.authority_open


@pytest.mark.parametrize("allowance", [1, 2], ids=["spent-by-schema-rejection", "correction-capacity"])
@pytest.mark.parametrize("invalid_arguments", [
    pytest.param({}, id="missing"),
    pytest.param({"path": "x" * 20_000}, id="oversized"),
    pytest.param({f"x{i}": "a" for i in range(70)}, id="items"),
    pytest.param(json.loads('{"path":' + "[" * 18 + "0" + "]" * 18 + "}"), id="depth"),
])
async def test_native_schema_rejection_spends_original_host_allowance(
    tmp_path: Path, native_provider: _NativeProvider, allowance: int, invalid_arguments: dict[str, Any],
) -> None:
    """Native validation failures consume capacity before any corrected filesystem effect.

    The Python authorization journey has valid native schemas and cannot catch charging after native validation
    or outer transport rejection of item/depth arguments before the native tool_call hook.
    Drive the adapter directly so run_agent's later observer cannot accidentally enforce the asserted admission.
    """
    allowed = tmp_path.resolve() / "permitted.txt"
    expected = "ONLY_ADMITTED_CORRECTION_MAY_READ_THIS\n"
    allowed.write_text(expected)
    native_provider.turns = [[("invalid-schema", NATIVE_READ_FILE, invalid_arguments)],
                             [("corrected", NATIVE_READ_FILE, {"path": str(allowed)})]]
    binding = HostToolBinding((allowed,), deadline=time.monotonic() + 15,
                              limits=HostToolLimits(max_calls=2))
    events = [event async for event in PiBackend(model="output-model").execute(
        tmp_path, "Correct the invalid request.", host_tools=binding, tool_call_budget=allowance,
        persist_session=False)]
    assert len(native_provider.requests) == 3
    history = {message["tool_call_id"]: message["content"] for message in native_provider.requests[-1]["messages"]
               if message["role"] == "tool"}
    assert "Validation failed" in history["invalid-schema"]
    if allowance == 1:
        assert "Original cumulative tool allowance exhausted" in history["corrected"]
        assert expected not in json.dumps(native_provider.requests)
    else:
        assert history["corrected"] == expected
    assert binding.calls == 2
    starts = [event.id for event in events if isinstance(event, ToolStartEvent)]
    assert starts == ["invalid-schema", "corrected"]
    assert [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error] == (
        ["invalid-schema", "corrected"] if allowance == 1 else ["invalid-schema"])


async def test_native_host_tools_compose_with_structured_submission_and_shared_accounting(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    """Host MCP and the final-output extension coexist in a real read-only staged invocation.

    Independent output-only and host-only journeys miss extension composition and double charging.
    """
    allowed = tmp_path.resolve() / "permitted.txt"
    allowed.write_text("composition bytes\n")
    output = {"verdict": "complete", "findings": []}
    native_provider.turns = [[("host-read", NATIVE_READ_FILE, {"path": str(allowed)})],
                             [("submit", "structured_output", output)]]
    binding = HostToolBinding((allowed,), deadline=time.monotonic() + 15,
                              limits=HostToolLimits(max_calls=2))
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(15, 2, 2))
    recorder = make_recorder(tmp_path, agent_model_name="output-model", backend_name="pi")
    async with recorder:
        result, _, reason = await run_agent(PiBackend(model="output-model"), tmp_path, "Read then submit.",
            phase=DaydreamPhase.DEEP, output_schema=_SCHEMA, host_tools=binding, read_only=True,
            investigation_budget=budget, persist_session=False, progress_callback=lambda _: None)
    assert result == output and reason is None
    assert budget.observed_tool_starts == binding.calls == 2
    assert budget.remaining_tool_calls == 0
    assert binding.native_settled and binding.cleanup_complete
    assert len(native_provider.requests) == 2
    tools = {tool["function"]["name"] for tool in native_provider.requests[0]["tools"]}
    assert {NATIVE_READ_FILE, "structured_output"} <= tools
    document = json.loads(recorder.path.read_text())
    assert [call["tool_call_id"] for step in document["steps"] for call in step.get("tool_calls") or []] == [
        "host-read", "submit"]


async def test_native_duplicate_call_identity_cannot_repeat_host_effect_or_charge(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    """A reused native ID is rejected before a second host effect and is emitted/charged once.

    Unique-ID correction journeys cannot detect replaying one callback admission or duplicate native accounting.
    """
    allowed = tmp_path.resolve() / "source"
    allowed.write_text("first read only\n")
    call = ("same-native-id", NATIVE_READ_FILE, {"path": str(allowed)})
    native_provider.turns = [[call], [call]]
    binding = HostToolBinding((allowed,), deadline=time.monotonic() + 15)
    events = []
    with pytest.raises(PiError, match="identity conflict"):
        async for event in PiBackend(model="output-model").execute(
                tmp_path, "Attempt a duplicate call.", host_tools=binding, persist_session=False):
            events.append(event)
    assert len(native_provider.requests) == 3
    results = [message["content"] for message in native_provider.requests[-1]["messages"]
               if message["role"] == "tool"]
    assert results[0] == "first read only\n" and "already charged" in results[1]
    assert binding.calls == 1
    assert [event.id for event in events if isinstance(event, ToolStartEvent)] == ["same-native-id"]
    assert [event.id for event in events if isinstance(event, ToolResultEvent)] == ["same-native-id"]
    conflicts = [event for event in events if isinstance(event, DiagnosticEvent)
                 and event.code == "pi_host_tool_identity_conflict"]
    assert conflicts
    assert any("already charged" in json.dumps(event.metadata) for event in conflicts)
    assert binding.native_settled and binding.cleanup_complete


async def test_native_cached_backend_cancellation_joins_callback_before_freeze_and_preserves_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider,
) -> None:
    """An active real MCP read child ends before evidence freezes while a cached-backend sibling continues.

    Existing CLI cancellation tests have no host callback and cannot prove writer joins, authority revocation
    or sibling resource ownership across two genuine native sessions.
    """
    root = tmp_path.resolve()
    blocked_path, surviving_path = root / "blocked", root / "surviving"
    blocked_path.write_text("CANCELLED_BYTES_MUST_NEVER_RETURN")
    surviving_path.write_text("independent sibling bytes\n")
    release_fifo = root / "release"
    os.mkfifo(release_fifo)
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_reader(read_fd, entered.set)
    survivor_waiting, finish_survivor = threading.Event(), threading.Event()
    spawn = asyncio.create_subprocess_exec
    children: list[asyncio.subprocess.Process] = []

    async def blocking_spawn(*args: Any, **options: Any) -> asyncio.subprocess.Process:
        if (len(args) >= 4 and args[1] == "-I" and Path(args[2]).name == "host_tools_read.py"
                and args[3] == str(blocked_path)):
            script = "import os,sys; os.write(int(sys.argv[1]),b'R'); open(sys.argv[2],'rb').read(1)"
            child = await spawn(sys.executable, "-c", script, str(write_fd), str(release_fifo),
                                **{**options, "pass_fds": (write_fd,)})
            children.append(child)
            return child
        return await spawn(*args, **options)

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        user = next(message for message in request["messages"] if message["role"] == "user")
        cancelled = "cancelled-invocation" in json.dumps(user)
        if any(message["role"] == "tool" for message in request["messages"]):
            if not cancelled:
                survivor_waiting.set()
                if not finish_survivor.wait(8):
                    raise RuntimeError("sibling release barrier was not reached")
            return []
        return [("cancel-read" if cancelled else "sibling-read", NATIVE_READ_FILE,
                 {"path": str(blocked_path if cancelled else surviving_path)})]

    native_provider.choose = choose
    monkeypatch.setattr("daydream.host_tools.asyncio.create_subprocess_exec", blocking_spawn)
    backend = PiBackend(model="output-model")
    cancelled_binding = HostToolBinding((blocked_path,), deadline=time.monotonic() + 20,
        limits=HostToolLimits(callback_timeout_s=10))
    sibling_binding = HostToolBinding((surviving_path,), deadline=time.monotonic() + 20)
    cancelled_recorder = make_recorder(root, path=root / "cancelled.json", backend_name="pi")
    sibling_recorder = make_recorder(root, path=root / "sibling.json", backend_name="pi")

    async def invoke(binding: HostToolBinding, recorder: Any, prompt: str) -> Any:
        async with recorder:
            return await run_agent(backend, root, prompt, phase=DaydreamPhase.REVIEW, host_tools=binding,
                                   persist_session=False, progress_callback=lambda _: None)

    cancelled = asyncio.create_task(invoke(cancelled_binding, cancelled_recorder, "cancelled-invocation"))
    sibling = asyncio.create_task(invoke(sibling_binding, sibling_recorder, "surviving-invocation"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert await asyncio.to_thread(survivor_waiting.wait, 5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert len(children) == 1 and children[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
        frozen = cancelled_recorder.path.read_bytes()
        cancelled_lifecycle = json.loads(frozen)["extra"]["subtrajectories"][0]["backend_lifecycle"]
        assert cancelled_lifecycle["host_authority_closed"] is True
        assert cancelled_lifecycle["host_native_settled"] is False
        assert cancelled_lifecycle["host_cleanup_complete"] is True
        assert not sibling.done()
        finish_survivor.set()
        result, _, reason = await asyncio.wait_for(sibling, 5)
        assert reason is None and "Native reasoning" in result
        assert cancelled_recorder.path.read_bytes() == frozen
        assert "CANCELLED_BYTES_MUST_NEVER_RETURN" not in frozen.decode()
        assert "independent sibling bytes" in sibling_recorder.path.read_text()
        assert not cancelled_binding.authority_open and cancelled_binding.cleanup_complete
        assert sibling_binding.native_settled and sibling_binding.cleanup_complete
        assert cancelled_binding.calls == sibling_binding.calls == 1
        assert backend._transports == []
    finally:
        finish_survivor.set()
        for task in (cancelled, sibling):
            if not task.done():
                task.cancel()
        await asyncio.gather(cancelled, sibling, return_exceptions=True)
        loop.remove_reader(read_fd)
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("missing", ["agent_settled", "tool_execution_end"])
async def test_native_bound_calls_without_output_schema_require_positive_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, missing: str,
) -> None:
    """Real callback success/final prose/zero exit cannot hide missing native settlement or an active tool.

    Historical structured-output replay checks miss strict settlement for host bindings without an output schema.
    Drop one actual transport event; every provider choice, callback and other native event remains real.
    """
    original = CliTransport.lines

    async def lose_event(transport: CliTransport, *args: Any, **kwargs: Any) -> Any:
        async for line in original(transport, *args, **kwargs):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if event.get("type") != missing:
                yield line

    monkeypatch.setattr(CliTransport, "lines", lose_event)
    source = tmp_path.resolve() / "source"
    source.write_text("actual callback bytes\n")
    native_provider.turns = [[("read", NATIVE_READ_FILE, {"path": str(source)})]]
    binding = HostToolBinding((source,), deadline=time.monotonic() + 15)
    recorder = make_recorder(tmp_path, backend_name="pi")
    async with recorder:
        with pytest.raises(PiError, match="did not settle"):
            await run_agent(PiBackend(model="output-model"), tmp_path, "Read and finish.",
                phase=DaydreamPhase.REVIEW, host_tools=binding, persist_session=False,
                progress_callback=lambda _: None)
    history = [message["content"] for message in native_provider.requests[-1]["messages"]
               if message["role"] == "tool"]
    assert history == ["actual callback bytes\n"]
    trajectory = json.loads(recorder.path.read_text())
    lifecycle = trajectory["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["host_authority_closed"] is True and lifecycle["host_cleanup_complete"] is True
    assert lifecycle["host_native_settled"] is (missing != "agent_settled")
    assert sum((step.get("metrics") or {}).get("prompt_tokens", 0) or 0 for step in trajectory["steps"]) == 20


@pytest.mark.parametrize('native', [True, False], ids=['native', 'validation-optout'])
async def test_installed_pi_submits_native_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool,
) -> None:
    native_pi = shutil.which("pi")
    if native_pi is None or shutil.which("node") is None:
        pytest.skip("The installed official Pi CLI and Node are required for native tool loading.")
    config = tmp_path / "isolated-pi-config"
    config.mkdir()
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(config))
    for name in ("PI_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert not any('Live native Pi attempt budget:' in json.dumps(message)
                           for message in request['messages'])
            output = {"verdict": "complete", "findings": []}
            deltas: list[dict[str, Any]] = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "native-output-001", "type": "function",
                    "function": {"name": "structured_output", "arguments": json.dumps(output)},
                }]}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ]
            if not native:
                deltas[0]['choices'][0]['delta'] = {'role': 'assistant', 'content': json.dumps(output)}
                deltas[1]['choices'][0]['finish_reason'] = 'stop'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for payload in deltas:
                self.wfile.write(("data: " + json.dumps({
                    "id": "native-output", "object": "chat.completion.chunk", "created": 1,
                    "model": "output-model", **payload}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / "models.json").write_text(json.dumps({"providers": {"output-fixture": {
            "baseUrl": f"{base_url}/v1", "api": "openai-completions",
            "apiKey": "synthetic-loopback-only", "models": [{"id": "output-model", "reasoning": False,
                "input": ["text"], "contextWindow": 32768, "maxTokens": 8192}]}}}))
        monkeypatch.setenv("PI_PROVIDER", "output-fixture")
        events = [event async for event in PiBackend(model="output-model").execute(
            tmp_path, "Return the requested result.", output_schema=_SCHEMA, read_only=True, persist_session=False,
            validate_structured_output=native, tool_call_budget=None if native else 8)]

    request = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request.config, PiRequestConfig)
    assert request.config.no_extensions is native
    assert request.output_schema == _SCHEMA
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [(event.id, event.name) for event in starts] == (
        [("native-output-001", "structured_output")] if native else [])
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == ({"verdict": "complete", "findings": []} if native else None)
    if not native:
        assert [event.text for event in events if isinstance(event, TextEvent)] == [
            '{"verdict": "complete", "findings": []}']


@pytest.mark.parametrize('allowance,submit', [(8, True), (7, True), (6, True), (2, True), (1, True), (None, True),
                                            (7, False), (6, False)],
                         ids=['ample', 'submission-only', 'batch-exhausted', 'batch-overdrawn',
                              'initial-only', 'unbounded', 'bounded-recovery', 'exhausted-no-recovery'])
async def test_native_pi_budget_warning_preserves_wire_roles_history_and_charged_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowance: int | None, submit: bool,
) -> None:
    """Actual wire requests catch custom notes becoming users or dropping genuine history.

    The immediate-submission journey misses threshold transitions, failed/invalid batch
    charges and missing-submission recovery at an unchanged remaining allowance.
    """
    if shutil.which('pi') is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    config = tmp_path / 'isolated-pi-config'
    config.mkdir()
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(config))
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    source = 'Captured source\n'
    (tmp_path / 'source.txt').write_text(source)
    prompt = 'Investigate the supplied source and submit the result.'
    assignment = {'objective': 'first_pass', 'target_ids': ['target-source'], 'candidate_ids': ['candidate-source']}
    policy = 'Stable original system policy: preserve this exact instruction.'
    instructions = policy + '\nDaydream final-submission assignment: ' + json.dumps(assignment)
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}
    batches: list[list[tuple[str, str, dict[str, Any]]]] = [
        [('read-a', 'read', {'path': 'source.txt'}), ('read-missing', 'read', {'path': 'missing.txt'}),
         ('read-b', 'read', {'path': 'source.txt'})],
        [('failed-submit', 'structured_output', {'verdict': 'incomplete', 'findings': False})],
        [('read-c', 'read', {'path': 'source.txt'}), ('read-d', 'read', {'path': 'source.txt'})],
        *([[]] if allowance in (8, 7, None) else []),  # Repeat threshold through bounded recovery.
        [('final-submit', 'structured_output', output)] if submit else [],
    ]

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            index = len(requests) - 1
            batch = batches[index]
            assistant_delta: dict[str, Any] = {
                'role': 'assistant', 'content': f'Original assistant reasoning summary {index}.'}
            if batch:
                assistant_delta['tool_calls'] = [
                    {'index': index, 'id': call_id, 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(batch)]
            deltas: list[dict[str, Any]] = [
                {'choices': [{'index': 0, 'delta': assistant_delta, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if batch else 'stop'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            ]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for delta in deltas:
                self.wfile.write(('data: ' + json.dumps({'id': 'pacing-output', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'output-model', **delta}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / 'models.json').write_text(json.dumps({'providers': {'output-fixture': {
            'baseUrl': f'{base_url}/v1', 'api': 'openai-completions', 'apiKey': 'synthetic-loopback-only',
            'models': [{'id': 'output-model', 'reasoning': False, 'input': ['text'],
                        'contextWindow': 32768, 'maxTokens': 8192}]}}}))
        monkeypatch.setenv('PI_PROVIDER', 'output-fixture')
        events = [event async for event in PiBackend(model='output-model').execute(
            tmp_path, prompt, output_schema=_SCHEMA, review_instructions=instructions,
            read_only=True, persist_session=False, tool_call_budget=allowance)]

    assert len(requests) == len(batches)
    completions: dict[str, str] = {}
    for event in events:
        if isinstance(event, ToolResultEvent):
            # The host retains native details as JSON; OpenAI's tool content carries text only.
            completions[event.id] = (''.join(block['text'] for block in json.loads(event.output)['content'])
                                     if event.output.startswith('{"content":') else event.output)
    warning_requests: list[int] = []
    recovery = 'Submit the final result now by calling structured_output alone. Assistant prose is not a submission.'
    warning_instruction = ('Final-submission budget warning: Only submission capacity remains. '
                           'Submit now using structured_output alone; report unfinished assigned work '
                           'with truthful incomplete decisions. Assigned objective and identities: '
                           + json.dumps(assignment, separators=(',', ':')))
    original_system: dict[str, Any] | None = None
    for index, request in enumerate(requests):
        messages = request['messages']
        system = [message for message in messages if message['role'] in {'system', 'developer'}]
        assert system and messages[0] == system[0]
        assert policy in json.dumps(system)
        assert 'Attempt dispatch-time snapshot:' in json.dumps(system)
        preserved_system = dict(system[0])
        preserved_system['content'] = (
            system[0]['content'].replace(warning_instruction, '').replace(recovery, '').strip())
        if original_system is None:
            original_system = preserved_system
        assert preserved_system == original_system
        users = [message for message in messages if message['role'] == 'user']
        assert len(users) == 1 and prompt in json.dumps(users[0])
        assert 'Live native Pi attempt budget:' not in json.dumps(messages)
        assert 'daydream_native_tool_budget' not in json.dumps(messages)
        assert request['tools'] == requests[0]['tools']
        assert next(tool for tool in request['tools'] if tool['function']['name'] == 'structured_output')[
            'function']['parameters'] == _SCHEMA
        assistants = [message for message in messages if message['role'] == 'assistant']
        assert len(assistants) == index
        for earlier, assistant in enumerate(assistants):
            assert assistant['content'] == f'Original assistant reasoning summary {earlier}.'
            assert [(call['id'], call['function']['name'], json.loads(call['function']['arguments']))
                    for call in assistant.get('tool_calls', [])] == batches[earlier]
        history = {message['tool_call_id']: message['content'] for message in messages if message['role'] == 'tool'}
        previous = [call for batch in batches[:index] for call in batch]
        assert set(history) == {call_id for call_id, _, _ in previous}
        for call_id, name, arguments in previous:
            assert history[call_id] == completions[call_id]
            if name == 'read' and arguments['path'] == 'source.txt':
                assert history[call_id] == source
        warnings = [message for message in messages if 'Final-submission budget warning:' in json.dumps(message)]
        if warnings:
            warning_requests.append(index)
            assert len(warnings) == 1 and warnings[0]['role'] in {'system', 'developer'}
            warning = json.dumps(warnings[0])
            assert 'Only submission capacity remains' in warning and 'truthful incomplete decisions' in warning
            assignment_text = warnings[0]['content'].split('Assigned objective and identities: ', 1)[1]
            assert json.JSONDecoder().raw_decode(assignment_text)[0] == assignment
        recovery_messages = [message for message in messages if recovery in json.dumps(message)]
        assert len(recovery_messages) == (1 if index == 4 else 0)
        assert all(message['role'] in {'system', 'developer'} for message in recovery_messages)
    assert warning_requests == ([3] if allowance == 7 else [0] if allowance == 1 else [])
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [event.id for event in starts] == [call_id for batch in batches for call_id, _, _ in batch]
    failures = [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert failures == ['read-missing', 'failed-submit']
    assert completions['read-missing'] == f"ENOENT: no such file or directory, access '{tmp_path / 'missing.txt'}'"
    assert completions['failed-submit'].startswith('Validation failed for tool "structured_output":\n')
    assert completions['failed-submit'].endswith(
        'Received arguments:\n{\n  "verdict": "incomplete",\n  "findings": false\n}')
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == (output if submit else None)


async def test_native_pi_retry_refreshes_attempt_count_without_renewing_host_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart warning state resets without refunding successful or failed tool starts.

    The batch/history journey never restarts the CLI, so it cannot detect a warning flag
    leaking between attempts or a retry packet renewing the cumulative host allowance.
    """
    if shutil.which('pi') is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    config = tmp_path / 'isolated-pi-config'
    config.mkdir()
    (config / 'settings.json').write_text(json.dumps({'retry': {'enabled': False, 'maxRetries': 0}}))
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(config))
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / 'source.txt').write_text('Captured source\n')
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(requests) == 2:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': {'message': '503 Service Unavailable'}}).encode())
                return
            batch = ([('attempt-1-a', 'read', {'path': 'source.txt'}),
                      ('attempt-1-b', 'read', {'path': 'missing.txt'})] if len(requests) == 1 else
                     [('attempt-2-submit', 'structured_output', output)])
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            deltas: list[dict[str, Any]] = [
                {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [
                    {'index': index, 'id': call_id, 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(batch)]}, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            ]
            for delta in deltas:
                self.wfile.write(('data: ' + json.dumps({'id': 'retry-output', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'output-model', **delta}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / 'models.json').write_text(json.dumps({'providers': {'output-fixture': {
            'baseUrl': f'{base_url}/v1', 'api': 'openai-completions', 'apiKey': 'synthetic-loopback-only',
            'models': [{'id': 'output-model', 'reasoning': False, 'input': ['text'],
                        'contextWindow': 32768, 'maxTokens': 8192}]}}}))
        monkeypatch.setenv('PI_PROVIDER', 'output-fixture')
        backend = PiBackend(model='output-model')
        backend.retry_attempts = 1
        backend.retry_base_delay_s = backend.retry_max_delay_s = 0
        budget = ReviewInvestigationBudget.from_limits(ReviewLimits(60, 3, 3))
        result, _, reason = await run_agent(
            backend, tmp_path, 'Investigate the supplied source and submit the result.',
            phase=DaydreamPhase.DEEP, output_schema=_SCHEMA, read_only=True, persist_session=False,
            investigation_budget=budget, tool_call_budget=3, progress_callback=lambda _: None)

    assert reason is None and result == output
    assert budget.observed_tool_starts == 3 and budget.remaining_tool_calls == 0
    assert len(requests) == 3
    for index, request in enumerate(requests):
        notes = [message for message in request['messages']
                 if 'Final-submission budget warning:' in json.dumps(message)]
        assert len(notes) == (0 if index == 0 else 1)
        assert all(message['role'] in {'system', 'developer'} for message in notes)
        assert len([message for message in request['messages'] if message['role'] == 'user']) == 1
        assert 'Live native Pi attempt budget:' not in json.dumps(request['messages'])
    assert 'tool-call allocation is 3 calls' in json.dumps(requests[0]['messages'])
    assert 'tool-call allocation is 1 calls' in json.dumps(requests[2]['messages'])
    assert 'attempt-1-a' not in json.dumps(requests[2]['messages'])

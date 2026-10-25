import hashlib
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from claude_agent_sdk._internal.query import Query
from claude_agent_sdk._internal.transport import Transport
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from daydream.atif import validate
from daydream.backends import (
    BackendExecutionInput,
    ClaudeRequestConfig,
    ContinuationToken,
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.claude import (
    ClaudeAgentError,
    ClaudeBackend,
    MaxTurnsError,
    _RunLocalClaudeSDKClient,
    _RunLocalSubprocessCLITransport,
)
from daydream.config import TEST_WALL_BUDGET_S
from daydream.trajectory import DaydreamPhase
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    MockThinkingBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_claude_sdk,
    scripted_client,
)
from tests.harness.trajectory import make_recorder


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _patch(client_class: Any) -> None:
        patch_claude_sdk(monkeypatch, client_class)
    return _patch


def _capturing_client(captured: dict[str, Any]) -> type:
    """A scripted client that records the ``ClaudeAgentOptions`` it was built with."""
    return scripted_client(
        [MockAssistantMessage(content=[MockTextBlock(text="OK")]), MockResultMessage(total_cost_usd=0.01)],
        captured=captured,
    )


def _capturing_backend(apply_patch: Any, **kwargs: Any) -> tuple[ClaudeBackend, dict[str, Any]]:
    """Patch the SDK with a capturing client and build a backend sharing its capture."""
    captured: dict[str, Any] = {}
    apply_patch(_capturing_client(captured))
    return ClaudeBackend(model="opus", **kwargs), captured


@pytest.mark.parametrize(
    "version_admission_delay_s", [pytest.param(0.0, id="version-completes"), pytest.param(2.1, id="version-times-out")],
)
async def test_injected_claude_uses_run_local_transport_for_version_and_main_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version_admission_delay_s: float,
) -> None:
    capture = tmp_path / "native-env.jsonl"
    cli = tmp_path / "claude"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_LOG'], 'a', encoding='utf-8') as out:\n"
        "    out.write(json.dumps({'kind': 'env', 'version': '-v' in sys.argv, "
        "'anthropic': os.environ.get('ANTHROPIC_API_KEY'), "
        "'ambient': os.environ.get('AMBIENT_ONLY'), "
        "'path': os.environ.get('PATH'), 'pwd': os.environ.get('PWD')}) + '\\n')\n"
        "if '-v' in sys.argv:\n"
        "    print('2.1.0', flush=True)\n"
        "else:\n"
        "    for line in sys.stdin:\n"
        "        message = json.loads(line)\n"
        "        with open(os.environ['CAPTURE_LOG'], 'a', encoding='utf-8') as out:\n"
        "            out.write(json.dumps({'kind': 'protocol', 'message': message}) + '\\n')\n"
        "        if message.get('type') == 'control_request':\n"
        "            print(json.dumps({'type': 'control_response', 'response': {"
        "'subtype': 'success', 'request_id': message['request_id'], "
        "'response': {}}}), flush=True)\n"
        "        elif message.get('type') == 'user':\n"
        "            print(json.dumps({'type': 'assistant', 'message': {"
        "'id': 'm1', 'model': 'fixture-model', 'role': 'assistant', "
        "'content': [{'type': 'text', 'text': 'OK'}], "
        "'stop_reason': 'end_turn', 'usage': {}}}), flush=True)\n"
        "            print(json.dumps({'type': 'result', 'subtype': 'success', "
        "'duration_ms': 1, 'duration_api_ms': 1, 'is_error': False, "
        "'num_turns': 1, 'session_id': 's1', 'total_cost_usd': 0, "
        "'usage': {}, 'result': 'OK'}), flush=True)\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    environment = {
        "PATH": str(tmp_path), "HOME": str(tmp_path / "run-home"), "CAPTURE_LOG": str(capture),
        "ANTHROPIC_API_KEY": "run-key", "PWD": "/run/pwd",
    }
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("AMBIENT_ONLY", "must-not-cross")
    monkeypatch.setenv("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "malformed-ambient")
    open_process = anyio.open_process
    spawn_attempts: list[tuple[bool, dict[str, str]]] = []
    async def _record_open_process(
        command: list[str], *args: Any, **kwargs: Any
    ) -> Any:
        native_environment = kwargs.get("env")
        assert isinstance(native_environment, dict)
        is_version = command == [str(cli), "-v"]
        spawn_attempts.append((is_version, dict(native_environment)))
        if is_version and version_admission_delay_s:
            # The SDK's version check is explicitly best-effort and capped at
            # two seconds. Model an overloaded host where process admission
            # misses that window; the main SDK session must still start.
            await anyio.sleep(version_admission_delay_s)
        return await open_process(command, *args, **kwargs)
    monkeypatch.setattr(anyio, "open_process", _record_open_process)
    async def _hook(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"continue": True}
    options = ClaudeAgentOptions(
        cli_path=str(cli), cwd=str(tmp_path), env=dict(environment),
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[cast(Any, _hook)])]},
    )
    transport = _RunLocalSubprocessCLITransport(options, environment=environment)
    async with _RunLocalClaudeSDKClient(options=options, transport=transport, initialize_timeout_s=60.0) as client:
        await client.query("review")
        messages = [message async for message in client.receive_response()]
    observations = [json.loads(line) for line in capture.read_text().splitlines()]
    native = [item for item in observations if item["kind"] == "env"]
    protocol = [item["message"] for item in observations if item["kind"] == "protocol"]
    assert len(spawn_attempts) == 2
    assert {is_version for is_version, _ in spawn_attempts} == {False, True}
    for _, attempted_environment in spawn_attempts:
        assert attempted_environment["ANTHROPIC_API_KEY"] == "run-key"
        assert "AMBIENT_ONLY" not in attempted_environment
        assert attempted_environment["PATH"] == str(tmp_path)
        assert attempted_environment["PWD"] == str(tmp_path)
        assert attempted_environment["HOME"] == str(tmp_path / "run-home")
    main_native = [item for item in native if not item["version"]]
    assert len(main_native) == 1
    assert all(item["anthropic"] == "run-key" for item in native)
    assert all(item["ambient"] is None for item in native)
    assert all(item["path"] == str(tmp_path) for item in native)
    assert main_native[0]["pwd"] == str(tmp_path)
    initialize = next(item for item in protocol if item["type"] == "control_request")
    assert initialize["request"]["hooks"]["PreToolUse"][0]["hookCallbackIds"]
    assert len(messages) == 2


async def test_claude_backend_injected_environment_reaches_sdk_options_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    base_client = scripted_client([MockAssistantMessage(content=[MockTextBlock(text="OK")]), MockResultMessage()])
    class CapturingClient(base_client):  # type: ignore[misc,valid-type]
        def __init__(self, options: Any = None, transport: Any = None) -> None:
            super().__init__(options=options)
            captured["options"] = options
            captured["transport"] = transport
    patch_claude_sdk(monkeypatch, CapturingClient)
    execution = BackendExecutionInput.from_environment(
        {
            "HOME": str(tmp_path / "home"), "PATH": "/run/bin", "ANTHROPIC_API_KEY": "run-key",
            "DAYDREAM_FANOUT_CONCURRENCY": "4", "DAYDREAM_PI_RETRY_ATTEMPTS": "3",
        }, backend="claude",
    )
    backend = ClaudeBackend(model="opus", execution_input=execution)
    _ = [event async for event in backend.execute(tmp_path, "review")]
    assert captured["options"].env["ANTHROPIC_API_KEY"] == "run-key"
    assert captured["options"].env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert backend.fanout_concurrency == 4
    assert backend.retry_policy == RetryPolicy(3, 2.0, 120.0)


async def test_execute_early_close_interrupts_and_drains_before_disconnect(patch_sdk: Any) -> None:
    lifecycle: list[str] = []
    class EarlyCloseClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options
        async def __aenter__(self) -> Any:
            return self
        async def __aexit__(self, *args: Any) -> None:
            lifecycle.append("disconnect")
        async def query(self, prompt: str) -> None:
            pass
        async def interrupt(self) -> None:
            lifecycle.append("interrupt")
        async def receive_response(self) -> AsyncIterator[Any]:
            yield MockAssistantMessage(
                content=[MockToolUseBlock(id="tool-1", name="Read", input={"file_path": "a.py"})]
            )
            lifecycle.append("terminal")
            yield MockResultMessage()
    patch_sdk(EarlyCloseClient)
    backend = ClaudeBackend(model="opus")
    event_stream = backend.execute(Path("/tmp"), "Review this")
    async for event in event_stream:
        if isinstance(event, ToolStartEvent):
            break
    await event_stream.aclose()
    assert lifecycle == ["interrupt", "terminal", "disconnect"]


async def test_error_result_raises_instead_of_clean_empty_result(patch_sdk: Any) -> None:
    """A billed SDK error raises after its native terminal and tool evidence."""
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 20,
             "cache_creation_input_tokens": 30}
    patch_sdk(
        scripted_client(
            [
                MockAssistantMessage(content=[MockTextBlock("Invalid API key · Fix external API key"),
                    MockToolUseBlock("read", "Read", {"file": "source.py"})], model="actual-model", usage=usage),
                MockUserMessage(content=[MockToolResultBlock("read", cast(Any, [
                    {"type": "text", "text": "structured tool output"},
                ]))]),
                MockResultMessage(total_cost_usd=0.02, is_error=True,
                    result="Invalid API key · Fix external API key", session_id="failed-session", usage=usage,
                    duration_ms=120, duration_api_ms=100, stop_reason="error"),
            ]
        )
    )
    backend = ClaudeBackend(model="opus")
    events = []
    with pytest.raises(ClaudeAgentError, match="Invalid API key"):
        async for event in backend.execute(Path("/tmp"), "Review this", persist_session=False):
            events.append(event)
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert (cost.input_tokens, cost.output_tokens, cost.cached_tokens, cost.cache_creation_tokens, cost.cost_usd) == (
        60, 5, 20, 30, 0.02,
    )
    tool = next(e for e in events if isinstance(e, ToolResultEvent))
    assert json.loads(tool.output) == [{"type": "text", "text": "structured tool output"}]
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert (result.session_id, result.finish_reason, result.continuation) == ("failed-session", "error", None)
    assert (result.duration_ms, result.duration_api_ms) == (120, 100)


async def test_max_turns_result_raises_typed_error(patch_sdk: Any) -> None:
    """A turn-cap error preserves its subtype even when SDK result text is absent."""
    patch_sdk(
        scripted_client(
            [
                MockAssistantMessage(content=[MockTextBlock(text="working on it")]),
                MockResultMessage(total_cost_usd=None, is_error=True, result=None, subtype="error_max_turns"),
            ]
        )
    )
    backend = ClaudeBackend(model="opus")
    with pytest.raises(MaxTurnsError) as excinfo:
        async for _ in backend.execute(Path("/tmp"), "Review this"):
            pass
    assert excinfo.value.subtype == "error_max_turns"
    assert isinstance(excinfo.value, ClaudeAgentError)


@pytest.mark.parametrize(
    ("read_only", "command", "allowed"),
    [
        (True, "git log --grep='fix|bug'", True),
        (True, "git blame -L 10,10 src.py", True),
        (True, "git show 648a327 -- src.py", True),
        (True, "git diff HEAD~1 HEAD -- --output", True),
        (True, "daydream search symbol src --expand deps/pkg --reason 'trace caller'", True),
        (True, "git ls-files -z --cached --others --exclude-standard -- src", True),
        (True, "cat foo(1).txt", True),
        (True, "ls -la", True),
        (True, "git commit -m x", False),
        (True, "daydream search-other symbol src", False),
        (True, "daydream search symbol src > evidence.json", False),
        (True, "git status && rm x", False),
        (True, "cat f | tee g", False),
        (True, "git log; rm x", False),
        (True, "echo $(rm x)", False),
        (True, "git log foo# > status.txt", False),
        (True, "cat < README.md", False),
        (True, "git diff --output=diff.patch", False),
        (True, "git log --output log.txt", False),
        (True, "(rm x)", False),
        (True, "git log\nrm x", False),
        (True, "git log 'unclosed", False),
        (True, "git logfoo", False),
        (True, "", False),
        (False, "find / -name y", False),
        (False, "grep -r pattern /", False),
        (False, "rm --force --recursive /", False),
        (False, "rm -f -r /", False),
        (False, "rm -Rf /", False),
        (False, "rm -i -r /*", False),
        (False, "rm -rf foo /", False),
        (False, "rm -rf /home/user/tmp", True),
        (False, "grep / file.txt", True),
        (False, "find core/osprey-tui -name agent.rs", True),
    ],
)
async def test_registered_bash_guards_enforce_the_execution_profile(
    patch_sdk: Any,
    read_only: bool,
    command: str,
    allowed: bool,
) -> None:
    """Native callback registration enforces readonly syntax and always-on catastrophic-command guards."""
    backend, captured = _capturing_backend(patch_sdk)
    _ = [event async for event in backend.execute(Path("/tmp"), "Go", read_only=read_only)]
    matcher = captured["options"].hooks["PreToolUse"][0]
    response = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"command": command}})
    assert (response.get("hookSpecificOutput", {}).get("permissionDecision") == "deny") is (not allowed)
    if not allowed:
        reason = response["hookSpecificOutput"]["permissionDecisionReason"]
        assert ("read-only guard" if read_only else "dangerous command blocked") in reason


@pytest.mark.parametrize(
    ("payload", "allowed"),
    [
        ({"tool_name": "Read", "tool_input": {"file_path": "x"}}, True),
        ({"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, False),
        ({"tool_name": "FutureMutator", "tool_input": {}}, False),
        ({"tool_name": "Bash"}, False),
        ({"tool_name": "Bash", "tool_input": "malformed"}, False),
        (None, False),
    ],
)
async def test_read_only_execute_registers_pretooluse_guard(patch_sdk: Any, payload: Any, allowed: bool) -> None:
    """The all-tool SDK matcher admits inspection and fails closed on mutating, unknown or malformed calls."""
    backend, captured = _capturing_backend(patch_sdk)
    _ = [event async for event in backend.execute(Path("/tmp"), "Go", read_only=True)]
    matchers = captured["options"].hooks["PreToolUse"]
    assert len(matchers) == 1 and matchers[0].matcher == ".*"
    response = await _decide(matchers[0], payload)
    assert (response.get("hookSpecificOutput", {}).get("permissionDecision") == "deny") is (not allowed)
    if not allowed:
        assert "read-only guard" in response["hookSpecificOutput"]["permissionDecisionReason"]


def test_unsupported_reasoning_effort_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        ClaudeBackend(model="opus", reasoning_effort="minimal")


async def _audit_decision(backend: ClaudeBackend, payload: Any) -> Any:
    guard = backend._audit_root_guard  # noqa: SLF001 - security contract seam
    assert guard is not None
    return await guard(payload, None, {"signal": None})


def _is_denied(decision: Any) -> bool:
    return bool(decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny")


_GIT_REDIRECT_VARS = (
    "PWD", "OLDPWD", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES", "GIT_PREFIX",
)


async def _decide(matcher: Any, payload: Any) -> Any:
    # Mirror the SDK: run every registered hook; first deny wins.
    for hook in matcher.hooks:
        out = await hook(payload, None, {})
        if out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny":
            return out
    return {}


async def test_audit_root_guard_allows_only_canonical_read_tools(tmp_path: Path) -> None:
    root = tmp_path / "audit root"
    clean = root / "clean"
    clean.mkdir(parents=True)
    inside = clean / "inside.py"
    inside.write_text("needle\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    secret = source / "secret.txt"
    secret.write_text("secret\n", encoding="utf-8")
    outward = root / "escape"
    outward.symlink_to(source, target_is_directory=True)
    linked = root / "sub"
    linked.mkdir()
    (linked / "loop").symlink_to("..", target_is_directory=True)
    backend = ClaudeBackend(model="opus", audit_root=root, audit_outward_symlinks=frozenset({outward}))
    allowed = [
        {"tool_name": "StructuredOutput", "tool_input": {"result": {}}},
        {"tool_name": "Read", "tool_input": {"file_path": "clean/inside.py"}},
        {
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "needle", "path": str(inside), "output_mode": "content", "-n": True, "head_limit": 5,
            },
        }, {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "**/*.py"}},
    ]
    for payload in allowed:
        assert not _is_denied(await _audit_decision(backend, payload)), payload
    denied = [
        {}, {"tool_name": "Read", "tool_input": {}}, {"tool_name": "Read", "tool_input": {"file_path": 7}},
        {"tool_name": "Read", "tool_input": {"file_path": "../source/secret.txt"}},
        {"tool_name": "Read", "tool_input": {"file_path": str(clean / ".." / "clean" / "inside.py")}},
        {"tool_name": "Read", "tool_input": {"file_path": str(secret)}},
        {"tool_name": "Read", "tool_input": {"file_path": "escape/secret.txt"}},
        {"tool_name": "Read", "tool_input": {"file_path": "clean/inside.py", "pages": "1"}},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle"}},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle", "path": "clean"}},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle", "path": str(secret)}},
        {"tool_name": "Grep", "tool_input": { "pattern": "needle", "path": str(inside), "output_mode": [], }},
        {"tool_name": "Grep", "tool_input": { "pattern": "needle", "path": str(inside), "output_mode": {}, }},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle", "path": str(inside), "glob": "*"}},
        {"tool_name": "Glob", "tool_input": {"pattern": "clean/*.py"}},
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "../*.py"}},
        {"tool_name": "Glob", "tool_input": {"path": "sub", "pattern": "loop/escape/*"}},
        {"tool_name": "Glob", "tool_input": {"path": "sub/loop/clean", "pattern": "*.py"}},
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "C:\\*"}},
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "\\\\server\\share"}},
        {"tool_name": "Bash", "tool_input": {"command": "cat ../source/secret.txt"}},
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "x"}},
        {"tool_name": "Task", "tool_input": {}}, {"tool_name": "Skill", "tool_input": {}},
        {"tool_name": "mcp__server__read", "tool_input": {}}, {"tool_name": "FutureTool", "tool_input": {}},
    ]
    for payload in denied:
        assert _is_denied(await _audit_decision(backend, payload)), payload


@pytest.mark.parametrize("bad_call", ["cwd", "read_only", "continuation", "agents"])
async def test_audit_execute_rejects_unsafe_invocation_before_client(
    patch_sdk: Any, tmp_path: Path, bad_call: str,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    backend, captured = _capturing_backend(patch_sdk, audit_root=root)
    kwargs: dict[str, Any] = {"read_only": True}
    cwd = root
    if bad_call == "cwd":
        cwd = other
    elif bad_call == "read_only":
        kwargs["read_only"] = False
    elif bad_call == "continuation":
        kwargs["continuation"] = ContinuationToken(backend="claude", data={"session_id": "old"})
    else:
        kwargs["agents"] = {"unsafe": object()}
    with pytest.raises(ClaudeAgentError, match="audit isolation"):
        async for _ in backend.execute(cwd, "audit", **kwargs):
            pass
    assert "options" not in captured


async def test_audit_guard_round_trips_through_real_sdk_query_protocol(tmp_path: Path) -> None:
    class MemoryTransport(Transport):
        def __init__(self) -> None:
            self.in_send, self.in_receive = anyio.create_memory_object_stream[ dict[str, Any] ](10)
            self.out_send, self.out_receive = anyio.create_memory_object_stream[ dict[str, Any] ](10)
            self.ready = False
        async def connect(self) -> None:
            self.ready = True
        async def write(self, data: str) -> None:
            message = cast(dict[str, Any], json.loads(data))
            await self.out_send.send(message)
            if message.get("type") == "control_request":
                request_id = message["request_id"]
                await self.in_send.send(
                    {
                        "type": "control_response",
                        "response": {"subtype": "success", "request_id": request_id, "response": {"commands": []}},
                    }
                )
        def read_messages(self) -> AsyncIterator[dict[str, Any]]:
            return self.in_receive.__aiter__()
        async def end_input(self) -> None:
            return None
        async def close(self) -> None:
            self.ready = False
            await self.in_send.aclose()
            await self.out_send.aclose()
        def is_ready(self) -> bool:
            return self.ready
    root = tmp_path / "audit"
    root.mkdir()
    inside = root / "inside.py"
    inside.write_text("ok\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    sub = root / "sub"
    sub.mkdir()
    (sub / "loop").symlink_to("..", target_is_directory=True)
    backend = ClaudeBackend(model="opus", audit_root=root, audit_outward_symlinks=frozenset({root / "escape"}))
    guard = backend._audit_root_guard  # noqa: SLF001 - pinned SDK adapter seam
    assert guard is not None
    transport = MemoryTransport()
    await transport.connect()
    query = Query(
        transport=transport, is_streaming_mode=True, hooks={"PreToolUse": [{"matcher": ".*", "hooks": [guard]}]},
    )
    await query.start()
    try:
        await query.initialize()
        initialize = await transport.out_receive.receive()
        hook_config = initialize["request"]["hooks"]["PreToolUse"][0]
        callback_id = hook_config["hookCallbackIds"][0]
        assert callback_id in query.hook_callbacks
        for request_id, payload, expected in (
            ("allow-1", {"tool_name": "Read", "tool_input": {"file_path": "inside.py"}}, None),
            ("deny-1", {"tool_name": "Bash", "tool_input": {"command": "cat /etc/passwd"}}, "deny"),
            (
                "deny-unhashable-list",
                {"tool_name": "Grep", "tool_input": { "pattern": "ok", "path": "inside.py", "output_mode": [], }},
                "deny",
            ),
            (
                "deny-unhashable-object",
                {"tool_name": "Grep", "tool_input": { "pattern": "ok", "path": "inside.py", "output_mode": {}, }},
                "deny",
            ),
            (
                "deny-inward-symlink-glob",
                {"tool_name": "Glob", "tool_input": {"path": "sub", "pattern": "loop/escape/*"}}, "deny",
            ),
        ):
            await transport.in_send.send(
                {
                    "type": "control_request", "request_id": request_id,
                    "request": {
                        "subtype": "hook_callback", "callback_id": callback_id, "input": payload,
                        "tool_use_id": "tool-1",
                    },
                }
            )
            response = await transport.out_receive.receive()
            assert response["response"]["subtype"] == "success"
            hook_output = response["response"]["response"]
            decision = hook_output.get("hookSpecificOutput", {}).get("permissionDecision")
            assert decision == expected
    finally:
        await query.close()
        query.close_receive_stream()
        await transport.close()


async def test_audit_options_reach_real_sdk_subprocess_transport(
    patch_sdk: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    backend, captured = _capturing_backend(patch_sdk, audit_root=root)
    async for _ in backend.execute(root, "audit", read_only=True):
        pass
    options = captured["options"]
    for variable in _GIT_REDIRECT_VARS:
        monkeypatch.setenv(variable, str(source))
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    spawned: dict[str, Any] = {}
    class FinishedProcess:
        stdin = None
        stdout = None
        stderr = None
        returncode = 0
    async def fake_open_process(command: list[str], **kwargs: Any) -> FinishedProcess:
        spawned["command"] = command
        spawned["kwargs"] = kwargs
        return FinishedProcess()
    monkeypatch.setattr(anyio, "open_process", fake_open_process)
    transport = SubprocessCLITransport(prompt="audit", options=options)
    transport._cli_path = "/usr/bin/true"  # noqa: SLF001 - pinned SDK seam
    await transport.connect()
    await transport.close()
    command = spawned["command"]
    assert command[command.index("--tools") + 1] == "Read,Grep,Glob,StructuredOutput"
    assert command[command.index("--allowedTools") + 1] == (
        "Read,Grep,Glob,StructuredOutput"
    )
    assert "--strict-mcp-config" in command
    assert "--setting-sources=" in command
    assert "--no-session-persistence" in command
    assert "--mcp-config" not in command
    assert "--plugin-dir" not in command
    assert not any(argument.startswith("--resume") for argument in command)
    child_env = spawned["kwargs"]["env"]
    for key, value in options.env.items():
        assert child_env[key] == value
    assert str(source) not in {child_env[key] for key in _GIT_REDIRECT_VARS}


# fanout_concurrency


# --- Session continuation via SDK resume --------------------------------------


@pytest.mark.parametrize(
    ("token", "expected_resume"),
    [
        (ContinuationToken(backend="claude", data={"session_id": "sess-123"}), "sess-123"),
        (ContinuationToken(backend="codex", data={"thread_id": "th-1"}), None),
    ], ids=["claude-minted", "foreign-backend"],
)
async def test_continuation_token_controls_resume(
    monkeypatch: pytest.MonkeyPatch, token: ContinuationToken, expected_resume: str | None,
) -> None:
    """A claude-minted token becomes ``options.resume``; a foreign token starts cold."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(
        monkeypatch,
        scripted_client([MockResultMessage(total_cost_usd=0.01, session_id="sess-123")], captured=captured),
    )
    backend = ClaudeBackend(model="opus")
    async for _ in backend.execute(Path("/tmp"), "p", continuation=token):
        pass
    assert captured["options"].resume == expected_resume


@pytest.mark.parametrize(("persist_session", "session_id"), [(False, "sess-9"), (True, None)])
async def test_result_continuation_requires_persisted_native_session(
    monkeypatch: pytest.MonkeyPatch, persist_session: bool, session_id: str | None,
) -> None:
    captured: dict[str, Any] = {}
    patch_claude_sdk(
        monkeypatch,
        scripted_client([MockResultMessage(total_cost_usd=0.01, session_id=session_id)], captured=captured),
    )
    backend = ClaudeBackend(model="opus")
    results = [event async for event in backend.execute(Path("/tmp"), "p", persist_session=persist_session)
               if isinstance(event, ResultEvent)]
    assert len(results) == 1
    assert results[0].session_id == session_id
    assert captured["options"].extra_args == ({} if persist_session else {"no-session-persistence": None})
    if persist_session and session_id:
        assert results[0].continuation is not None
        assert results[0].continuation.backend == "claude"
        assert results[0].continuation.data["session_id"] == session_id
    else:
        assert results[0].continuation is None


@pytest.mark.parametrize("read_only", [False, True])
async def test_execute_registers_background_bash_guard(patch_sdk: Any, read_only: bool) -> None:
    """Registered guards block background Bash in both profiles with foreground guidance.

    Foreground commands and non-Bash tools with the same key remain allowed.
    """
    backend, captured = _capturing_backend(patch_sdk)
    async for _ in backend.execute(Path("/tmp"), "Go", read_only=read_only):
        pass
    env = captured["options"].env
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    expected_ms = str(int(TEST_WALL_BUDGET_S * 1000))
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == expected_ms
    assert env["BASH_MAX_TIMEOUT_MS"] == expected_ms
    matcher = captured["options"].hooks["PreToolUse"][0]
    deny_bg = await _decide(
        matcher, {"tool_name": "Bash", "tool_input": {"command": "git status", "run_in_background": True}},
    )
    hook_out = deny_bg["hookSpecificOutput"]
    assert hook_out["permissionDecision"] == "deny"
    assert "background Bash blocked" in hook_out["permissionDecisionReason"]
    assert "foreground" in hook_out["permissionDecisionReason"]
    allow_fg = await _decide(matcher,
        {"tool_name": "Bash", "tool_input": {"command": "git status", "run_in_background": False}}
    )
    assert "hookSpecificOutput" not in allow_fg
    allow_no_key = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"command": "git status"}})
    assert "hookSpecificOutput" not in allow_no_key
    other_tool = await _decide(matcher,
        {"tool_name": "Read", "tool_input": {"file_path": "x", "run_in_background": True}}
    )
    assert "background Bash blocked" not in str(other_tool)
    missing_command = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"run_in_background": True}})
    assert "background Bash blocked" in str(missing_command)
    malformed = await _decide(matcher, {"tool_name": "Bash", "tool_input": "not-a-dict"})
    assert "background Bash blocked" not in str(malformed)
    assert "background Bash blocked" not in str(await _decide(matcher, "garbage"))
    assert "background Bash blocked" not in str(await _decide(matcher, None))


# --- P18 Task 1: effective request-config admission at the Claude SDK seam ---


async def test_sdk_conversation_records_native_tools_usage_and_resumed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual SDK stream owns suppression, partial telemetry and request provenance together."""
    monkeypatch.delenv("DAYDREAM_FANOUT_CONCURRENCY", raising=False)
    target = (tmp_path / "model cwd with spaces").resolve()
    target.mkdir()
    (target / "source.py").write_text("SOURCE_CANARY\n")
    captured: dict[str, Any] = {}
    client = scripted_client(
        [
            MockAssistantMessage(
                content=[
                    MockThinkingBlock("Inspect the file"),
                    MockTextBlock("Checking"),
                    MockToolUseBlock("read", "Read", {"file": "source.py"}),
                    MockToolUseBlock("schema", "StructuredOutput", {"issues": []}),
                ],
                model="sdk-model",
                message_id="initial",
            ),
            MockUserMessage(
                content=[MockToolResultBlock("read", "file contents"), MockToolResultBlock("schema", '{"issues":[]}')]
            ),
            MockAssistantMessage(
                content=[MockTextBlock("Partial")], model="sdk-model", message_id="partial", usage={"input_tokens": 100}
            ),
            MockResultMessage(
                total_cost_usd=None,
                usage={"input_tokens": 100, "output_tokens": 50},
                structured_output={"issues": []},
                session_id="native-session",
            ),
        ],
        captured=captured,
    )

    class ObservingClient(client):  # type: ignore[misc,valid-type]
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            assert Path(self.options.cwd) == target
            assert (Path(self.options.cwd) / "source.py").read_text() == "SOURCE_CANARY\n"
            captured["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()

    patch_claude_sdk(monkeypatch, ObservingClient)
    backend = ClaudeBackend(model="opus", reasoning_effort="max")
    assert backend.fanout_concurrency == 8
    recorder = make_recorder(tmp_path, agent_model_name="claude")
    first: list[Any] = []
    second: list[Any] = []
    prompt = "Inspect the committed source only."
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt=prompt)
            async for event in backend.execute(target, prompt, output_schema={"type": "object"}):
                first.append(event)
                inv.observe(event)
        options = captured["options"]
        assert options.effort == "max" and options.permission_mode == "bypassPermissions"
        assert options.allowed_tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        assert captured["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        first_result = next(event for event in first if isinstance(event, ResultEvent))
        assert first_result.structured_output == {"issues": []}
        token = first_result.continuation
        assert token is not None and token.data["session_id"] == "native-session"
        assert [event.text for event in first if isinstance(event, TextEvent)] == ["Checking", "Partial"]
        assert [(event.id, event.name, event.input) for event in first if isinstance(event, ToolStartEvent)] == [
            ("read", "Read", {"file": "source.py"}),
        ]
        assert [(event.id, event.output, event.is_error) for event in first if isinstance(event, ToolResultEvent)] == [
            ("read", "file contents", False),
        ]
        assert not any(isinstance(event, MetricsEvent) for event in first)
        cost = next(event for event in first if isinstance(event, CostEvent))
        assert (cost.input_tokens, cost.output_tokens, cost.cost_usd) == (100, 50, None)
        patch_claude_sdk(
            monkeypatch,
            scripted_client(
                [
                    MockAssistantMessage(
                        content=[MockTextBlock("Fixed")],
                        model="sdk-model",
                        message_id="billed",
                        usage={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cache_read_input_tokens": 20,
                            "cache_creation_input_tokens": 30,
                        },
                    ),
                    MockResultMessage(
                        total_cost_usd=0.02,
                        session_id="native-session",
                        duration_ms=120, duration_api_ms=100, stop_reason="end_turn",
                        usage={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cache_read_input_tokens": 20,
                            "cache_creation_input_tokens": 30,
                        },
                    ),
                ],
                captured=captured,
            ),
        )
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            inv.observe_user_step(prompt="Apply the fix")
            async for event in backend.execute(target, "Apply the fix", continuation=token):
                second.append(event)
                inv.observe(event)
    request = next(event for event in second if isinstance(event, RequestEvent))
    assert isinstance(request.config, ClaudeRequestConfig) and request.config.continuation_mode == "resume"
    assert request.session_id == "native-session" and request.session_source == "host_generated"
    assert request.model_source == "configured" and request.timestamp_source == "host_observed"
    assert captured["options"].resume == "native-session" and backend._active_clients == set()
    metric = next(event for event in second if isinstance(event, MetricsEvent))
    assert (metric.message_id, metric.prompt_tokens, metric.completion_tokens, metric.cache_creation_tokens) == (
        "billed",
        60,
        5,
        30,
    )
    cost = next(event for event in second if isinstance(event, CostEvent))
    assert (cost.input_tokens, cost.output_tokens, cost.cached_tokens, cost.cache_creation_tokens, cost.cost_usd) == (
        60, 5, 20, 30, 0.02,
    )
    terminal = next(event for event in second if isinstance(event, ResultEvent))
    assert (terminal.session_id, terminal.finish_reason, terminal.duration_ms, terminal.duration_api_ms) == (
        "native-session", "end_turn", 120, 100,
    )
    document = json.loads(recorder.path.read_text())
    steps = [step for step in document["steps"] if step["source"] == "agent"]
    assert steps[0]["message"] == "Checking" and steps[0]["reasoning_content"] == "Inspect the file"
    assert [(call["tool_call_id"], call["function_name"], call["arguments"]) for call in steps[0]["tool_calls"]] == [
        ("read", "Read", {"file": "source.py"}),
    ]
    assert [(result["source_call_id"], result["content"]) for result in steps[0]["observation"]["results"]] == [
        ("read", "file contents"),
    ]
    assert not any(
        call["function_name"] == "StructuredOutput" for step in steps for call in (step.get("tool_calls") or [])
    )
    fixed = next(step for step in steps if step["message"] == "Fixed")
    assert fixed["model_name"] == "sdk-model"
    assert document["final_metrics"]["total_prompt_tokens"] == 160
    assert document["final_metrics"]["total_completion_tokens"] == 55
    assert not any(isinstance(event, (GenerationStartEvent, GenerationEndEvent)) for event in first + second)
    assert validate(recorder.path, validate_images=False)

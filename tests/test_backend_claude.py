import asyncio
import hashlib
import json
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from claude_agent_sdk import ClaudeAgentOptions, CLINotFoundError, HookMatcher
from mcp import types as mcp_types

from daydream import agent, clock
from daydream.atif import validate
from daydream.backends import (
    AuditIsolationError,
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
    claude as claude_backend,
    create_backend,
)
from daydream.backends.claude import (
    ClaudeAgentError,
    ClaudeBackend,
    _RunLocalClaudeSDKClient,
    _RunLocalSubprocessCLITransport,
)
from daydream.config import TEST_WALL_BUDGET_S
from daydream.host_tools import HostTool, HostToolBinding, HostToolFatal
from daydream.review_budget import BudgetLedger
from daydream.trajectory import DaydreamPhase
from tests.harness.claude_provider import MODEL, LoopbackProvider, Turn, claude_cli_env, tool
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
from tests.harness.git_helpers import commit, init_repo, write_and_stage
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


async def test_injected_claude_uses_run_local_transport_for_version_and_main_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
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
        "    print('0.0.1', flush=True)\n"
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
        return await open_process(command, *args, **kwargs)
    monkeypatch.setattr(anyio, "open_process", _record_open_process)
    async def _hook(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"continue": True}
    # No explicit path: connect resolves the CLI lazily, and a bundled CLI wins over PATH discovery.
    monkeypatch.setattr(_RunLocalSubprocessCLITransport, "_find_bundled_cli", lambda _self: str(cli))
    options = ClaudeAgentOptions(
        cwd=str(tmp_path), env=dict(environment),
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[cast(Any, _hook)])]},
    )
    transport = _RunLocalSubprocessCLITransport(options, environment=environment)
    with caplog.at_level("WARNING"):
        client_context = _RunLocalClaudeSDKClient(options=options, transport=transport, initialize_timeout_s=60.0)
        async with client_context as client:
            await client.query("review")
            messages = [message async for message in client.receive_response()]
    # The best-effort version probe warns about an unsupported CLI without blocking the session.
    assert "Claude Code version 0.0.1" in caplog.text
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

    # The version probe is best effort: supported or unreadable output never warns or fails the session.
    for version_script in ("print('99.0.0')", "pass"):
        caplog.clear()
        cli.write_text(f"#!{sys.executable}\n{version_script}\n", encoding="utf-8")
        await transport._check_claude_version()  # noqa: SLF001 - best-effort probe seam
        assert "unsupported" not in caplog.text
    # A started transport never spawns again.
    monkeypatch.setattr(transport, "_process", object())
    spawned = len(spawn_attempts)
    await transport.connect()
    assert len(spawn_attempts) == spawned
    # A resolved CLI that disappeared fails as a missing CLI, not a hung or silent session.
    monkeypatch.setattr(transport, "_process", None)
    monkeypatch.setattr(transport, "_cli_path", str(tmp_path / "missing-claude"))
    with pytest.raises(CLINotFoundError, match="missing-claude"):
        await transport.connect()


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


def test_unsupported_reasoning_effort_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        ClaudeBackend(model="opus", reasoning_effort="minimal")


async def _audit_decision(backend: ClaudeBackend, payload: Any) -> Any:
    guard = backend._audit_root_guard  # noqa: SLF001 - security contract seam
    assert guard is not None
    return await guard(payload, None, {"signal": None})


def _is_denied(decision: Any) -> bool:
    return bool(decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny")


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


async def test_audit_execute_rejects_unsafe_invocation_before_client(patch_sdk: Any, tmp_path: Path) -> None:
    """Unsafe audit calls fail before any client; a safe one is bound to the snapshot root's Git identity."""
    root = tmp_path / "audit"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    backend, captured = _capturing_backend(patch_sdk, audit_root=root)
    unsafe_calls: list[tuple[Path, dict[str, Any]]] = [
        (other, {"read_only": True}),
        (root, {"read_only": False}),
        (root, {"read_only": True, "continuation": ContinuationToken(backend="claude", data={"session_id": "old"})}),
        (root, {"read_only": True, "agents": {"unsafe": object()}}),
    ]
    for cwd, kwargs in unsafe_calls:
        with pytest.raises(ClaudeAgentError, match="audit isolation"):
            async for _ in backend.execute(cwd, "audit", **kwargs):
                pass
        assert "options" not in captured
    # Host tools need a fresh, nonpersistent read-only call; Improve's audit root never serves them.
    ordinary = ClaudeBackend(model="opus")
    host_refusals: list[tuple[ClaudeBackend, dict[str, Any]]] = [
        (backend, {"read_only": True, "persist_session": False}),
        *((ordinary, {"read_only": True, "persist_session": False, **unsafe}) for unsafe in (
            {"finalization": True}, {"agents": {"helper": object()}},
            {"continuation": ContinuationToken(backend="claude", data={"session_id": "old"})})),
        (ordinary, {"read_only": False, "persist_session": False}), (ordinary, {"read_only": True}),
    ]
    for refused_backend, kwargs in host_refusals:
        binding = HostToolBinding(deadline=clock.monotonic() + 30, accept=dict)
        with pytest.raises(ValueError, match="Claude host tools require"):
            async for _ in refused_backend.execute(root, "audit", output_schema={"type": "object"},
                                                   host_tools=binding, **kwargs):
                pass
        assert "options" not in captured and binding.calls == 0 and binding.schema is None
    with pytest.raises(ValueError, match="review instructions are served only with host tools"):
        async for _ in ordinary.execute(root, "audit", review_instructions="Assignment"):
            pass
    # Only Claude has the snapshot-root tool boundary; the factory refuses every other audit backend.
    for name in ("codex", "pi", "osprey"):
        with pytest.raises(AuditIsolationError) as refused:
            create_backend(name, audit_root=root)
        assert (refused.value.backend_name, refused.value.reason) == (name, "unsupported_backend")
    with pytest.raises(ValueError, match=r"Unknown backend: 'invalid'\. Expected .*'pi'"):
        create_backend("invalid", audit_root=root)
    # A safe audit pins every Git redirect to the snapshot root instead of ambient or source values.
    async for _ in backend.execute(root, "audit", read_only=True):
        pass
    env = captured["options"].env
    git_dir = root / ".git"
    assert {key: env[key] for key in ("PWD", "OLDPWD", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                                      "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                                      "GIT_CEILING_DIRECTORIES", "GIT_PREFIX")} == {
        "PWD": str(root), "OLDPWD": str(root), "GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(root),
        "GIT_INDEX_FILE": str(git_dir / "index"), "GIT_OBJECT_DIRECTORY": str(git_dir / "objects"),
        "GIT_COMMON_DIR": str(git_dir), "GIT_ALTERNATE_OBJECT_DIRECTORIES": "",
        "GIT_CEILING_DIRECTORIES": str(tmp_path), "GIT_PREFIX": "",
    }


# --- Session continuation via SDK resume --------------------------------------


async def test_execute_registers_background_bash_guard(patch_sdk: Any) -> None:
    """Registered guards block background Bash in both profiles with foreground guidance.

    Foreground commands and non-Bash tools with the same key remain allowed; the read-only
    syntax guard and the always-on catastrophic-command guard enforce each execution profile.
    """
    profile_commands = {
        True: [
            ("git log --grep='fix|bug'", True), ("git log", True), ("git blame -L 10,10 src.py", True),
            ("git show 648a327 -- src.py", True), ("git diff HEAD~1 HEAD -- --output", True),
            ("daydream search symbol src --expand deps/pkg --reason 'trace caller'", True),
            ("git ls-files -z --cached --others --exclude-standard -- src", True),
            ("cat foo(1).txt", True), ("ls -la", True),
            ("git commit -m x", False), ("git reset --hard HEAD", False), ("rm -rf build", False),
            ("ls \nrm -rf /", False), ("cat foo\rrm x", False), ("daydream search-other symbol src", False),
            ("daydream search symbol src > evidence.json", False), ("git status && rm x", False),
            ("cat f | tee g", False), ("git log; rm x", False), ("echo $(rm x)", False),
            ("git log foo# > status.txt", False), ("cat < README.md", False),
            ("git diff --output=diff.patch", False), ("git log --output log.txt", False), ("(rm x)", False),
            ("git log\nrm x", False), ("git log 'unclosed", False), ("git logfoo", False), ("", False),
        ],
        False: [
            ("find / -name y", False), ("grep -r pattern /", False), ("rm --force --recursive /", False),
            ("rm -f -r /", False), ("rm -Rf /", False), ("rm -i -r /*", False), ("rm -rf foo /", False),
            ("rm -rf /home/user/tmp", True), ("grep / file.txt", True),
            ("find core/osprey-tui -name agent.rs", True),
        ],
    }
    # The read-only matcher covers every tool and fails closed on mutating, unknown or malformed calls.
    read_only_payloads: list[tuple[Any, bool]] = [
        ({"tool_name": "Read", "tool_input": {"file_path": "x"}}, True),
        ({"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, False),
        ({"tool_name": "FutureMutator", "tool_input": {}}, False), ({"tool_name": "Bash"}, False),
        ({"tool_name": "Bash", "tool_input": "malformed"}, False), (None, False),
    ]
    for read_only, commands in profile_commands.items():
        backend, captured = _capturing_backend(patch_sdk)
        async for _ in backend.execute(Path("/tmp"), "Go", read_only=read_only):
            pass
        env = captured["options"].env
        assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
        expected_ms = str(int(TEST_WALL_BUDGET_S * 1000))
        assert env["BASH_DEFAULT_TIMEOUT_MS"] == expected_ms
        assert env["BASH_MAX_TIMEOUT_MS"] == expected_ms
        matcher = captured["options"].hooks["PreToolUse"][0]
        payloads = [({"tool_name": "Bash", "tool_input": {"command": command}}, allowed)
                    for command, allowed in commands] + (read_only_payloads if read_only else [])
        for payload, allowed in payloads:
            response = await _decide(matcher, payload)
            denied = response.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
            assert denied is (not allowed), (read_only, payload)
            if not allowed:
                reason = response["hookSpecificOutput"]["permissionDecisionReason"]
                assert ("read-only guard" if read_only else "dangerous command blocked") in reason, payload
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
        assert token.backend == "claude"
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

    # A foreign token starts cold; an unpersisted session mints no continuation; partial terminal
    # usage keeps unknown output unknown and emits no per-message metrics.
    patch_claude_sdk(monkeypatch, scripted_client([
        MockAssistantMessage(content=[MockTextBlock("Partial")], usage={"input_tokens": 100}),
        MockResultMessage(total_cost_usd=0.001, session_id="sess-9", usage={"input_tokens": 100}),
    ], captured=captured))
    foreign = ContinuationToken(backend="codex", data={"thread_id": "th-1", "session_id": "foreign-session"})
    third = [event async for event in backend.execute(target, "Review", continuation=foreign,
                                                      persist_session=False)]
    assert captured["options"].resume is None
    assert next(event for event in third if isinstance(event, ResultEvent)).continuation is None
    assert not any(isinstance(event, MetricsEvent) for event in third)
    cost = next(event for event in third if isinstance(event, CostEvent))
    assert (cost.input_tokens, cost.output_tokens, cost.cost_usd) == (100, None, 0.001)
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


# --- Host tools through the real bundled CLI and an offline loopback provider ---

_LOOKUP = {"type": "object", "properties": {"path": {"type": "string", "maxLength": 64}},
           "required": ["path"], "additionalProperties": False}
_BATCH = {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"],
          "additionalProperties": False}
_VERDICT = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["clean", "concern"]}},
            "required": ["verdict"], "additionalProperties": False}
_FINAL = {"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]}
_HOST_NAMES = ["mcp__daydream__review_admit", "mcp__daydream__review_source", "mcp__daydream__structured_output"]


def _host_binding(repo: Path, ledger: BudgetLedger, installed: list[Any]) -> HostToolBinding:
    """A binding whose lookup serves exact committed Git bytes and whose admissions are recorded."""
    def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        shown = subprocess.run(["git", "show", f"HEAD:{arguments['path']}"], cwd=repo, capture_output=True, check=True)
        return {"path": arguments["path"], "content": shown.stdout.decode()}

    def admit(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return {"admitted": len(installed)}

    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return {"final": value["verdict"]}

    return HostToolBinding(budget=ledger, accept=accept, host_tools={
        "review_source": HostTool(_LOOKUP, lookup), "review_admit": HostTool(_BATCH, admit, mutating=True)})


def _tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    content = request["messages"][-1]["content"]
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]


def _result_text(block: dict[str, Any]) -> str:
    content = block["content"]
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


@pytest.mark.parametrize("route", ["ordinary", "run_local"])
async def test_host_tools_complete_through_both_sdk_startup_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    """Real CLI journeys: readiness, exact schemas, inline captured bytes, guards and corrected acceptance.

    A sibling is cancelled before the main conversation continues: closed by its consumer on
    the ordinary route, by task cancellation on the run-local route. The ordinary route
    settles after acceptance; the run-local route fails natively after acceptance, keeping
    the admission and terminal usage, stops on a host failure and refuses to start when its
    in-process server is not registered.
    """
    repo = tmp_path / "subject"
    init_repo(repo)
    # Near the bound result size: inline delivery needs the host's result-size hint.
    source = "".join(f"line {number:06d} \"quoted\" \\\\ value\n" for number in range(3300))
    write_and_stage(repo, "source.py", source)
    commit(repo, "source")
    conversations: dict[str, int] = {}
    sibling_started, sibling_closed = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def respond(body: dict[str, Any]) -> Turn:
        system = json.dumps(body.get("system"))
        (role,) = [role for role in ("MAIN", "SIBLING", "FAILING") if role in system]
        step = conversations[role] = conversations.get(role, 0) + 1
        if role == "SIBLING":
            loop.call_soon_threadsafe(sibling_started.set)
            return tool("other", "mcp__daydream__review_source", {"path": "source.py"}) if step == 1 else Turn(
                text="stopped")
        if role == "FAILING":
            # Without the host's interrupt the model would keep calling for eleven more turns.
            return tool(f"missing-{step}", "mcp__daydream__review_source", {"path": "missing.py"}) if (
                step < 12) else Turn(text="stopped")
        if step == 2:
            # The main conversation continues only after its sibling is fully cancelled.
            assert sibling_closed.wait(20)
        script = [
            tool("look", "mcp__daydream__review_source", {"path": "source.py"}),
            # The CLI refuses an unoffered tool before PreToolUse; a denied Bash reaches the guards.
            tool("write", "Write", {"file_path": "x", "content": "y"}) if route == "run_local"
            else tool("wipe", "Bash", {"command": "rm -rf /"}),
            tool("batch", "mcp__daydream__review_admit", {"note": "first batch"}),
            tool("invalid", "mcp__daydream__structured_output", {"verdict": "maybe"}),
            tool("final", "mcp__daydream__structured_output", {"verdict": "concern"}),
            *([Turn(error_status=400)] if route == "run_local" else [
                tool("late", "mcp__daydream__review_source", {"path": "source.py"}), Turn(text="done")]),
        ]
        return script[min(step, len(script)) - 1]

    provider = LoopbackProvider(respond)
    environment = claude_cli_env(monkeypatch, provider, tmp_path / "home")
    execution = BackendExecutionInput.from_environment(environment, backend="claude") if route == "run_local" else None
    backend = ClaudeBackend(model=MODEL, execution_input=execution)
    captured: list[ClaudeAgentOptions] = []
    authority_at_disconnect: list[bool] = []
    client_name = "_RunLocalClaudeSDKClient" if route == "run_local" else "ClaudeSDKClient"
    original = getattr(claude_backend, client_name)
    other = _host_binding(repo, BudgetLedger("sibling", deadline=clock.monotonic() + 60, limit=8), [])

    class Recording(original):  # type: ignore[misc,valid-type]
        def __init__(self, *, options: ClaudeAgentOptions, **kwargs: Any) -> None:
            captured.append(options)
            super().__init__(options=options, **kwargs)

        async def disconnect(self) -> None:
            if self.options.system_prompt == "SIBLING assignment":
                authority_at_disconnect.append(other.authority_open)
            await super().disconnect()

    monkeypatch.setattr(claude_backend, client_name, Recording)
    installed: list[Any] = []
    ledger = BudgetLedger("host", deadline=clock.monotonic() + 60, limit=8, investigation=True)
    binding = _host_binding(repo, ledger, installed)
    recorder = make_recorder(tmp_path)

    async def sibling() -> None:
        stream = backend.execute(repo, "Inspect", output_schema=_VERDICT, read_only=True, persist_session=False,
                                 review_instructions="SIBLING assignment", host_tools=other)
        if route == "ordinary":
            async for event in stream:
                if isinstance(event, ToolStartEvent):
                    break
            await stream.aclose()
        else:
            async def consume() -> None:
                async for _ in stream:
                    pass

            consumer = asyncio.create_task(consume())
            await sibling_started.wait()
            consumer.cancel()  # CancelledError reaches the adapter mid-stream, not GeneratorExit.
            with suppress(asyncio.CancelledError):
                await consumer
        sibling_closed.set()

    async def main() -> Any:
        async with recorder:
            return await agent.run_agent(
                backend, repo, "Review the captured source.", phase=DaydreamPhase.REVIEW, output_schema=_VERDICT,
                result_schema=_FINAL, read_only=True, persist_session=False, budget=ledger, host_tools=binding,
                review_system_instructions="MAIN assignment: review source.py with review_source.",
            )

    with provider.serving():
        if route == "ordinary":
            result, _ = await asyncio.gather(main(), sibling())
            assert result == ({"final": "concern"}, None, None)
            assert binding.native_settled
        else:
            outcomes = await asyncio.gather(main(), sibling(), return_exceptions=True)
            assert isinstance(outcomes[0], ClaudeAgentError) and "400" in str(outcomes[0])
            assert outcomes[1] is None and not binding.native_settled
            # An unexpected host callback failure is terminal: the conversation is interrupted
            # and the binding's own failure, not native completion, ends the invocation.
            failing = _host_binding(repo, BudgetLedger("failing", deadline=clock.monotonic() + 60), [])
            with pytest.raises(HostToolFatal) as failed:
                async for _ in backend.execute(repo, "Inspect", output_schema=_VERDICT, read_only=True,
                                               persist_session=False, review_instructions="FAILING assignment",
                                               host_tools=failing):
                    pass
            assert failed.value.code == "callback_failed" and failing.fatal_error is failed.value
            assert not failing.accepted and failing.cleanup_complete and conversations["FAILING"] < 12
            # A registration fault leaves the model unstarted: no provider request is made.
            before = len(provider.requests)
            monkeypatch.setattr(claude_backend, "_sdk_mcp_servers", lambda options: {})
            unregistered = _host_binding(repo, BudgetLedger("broken", deadline=clock.monotonic() + 60), [])
            with pytest.raises(HostToolFatal) as refused:
                async for _ in backend.execute(repo, "Inspect", output_schema=_VERDICT, read_only=True,
                                               persist_session=False, host_tools=unregistered):
                    pass
            assert refused.value.code == "native_readiness" and len(provider.requests) == before
            assert unregistered.cleanup_complete and unregistered.calls == 0
    # The cancelled sibling revoked only its own attempt, before its client disconnected.
    assert authority_at_disconnect == [False]
    assert (other.authority_open, other.accepted, other.native_settled, other.cleanup_complete) == (
        False, False, False, True)

    main_requests = [request for request in provider.requests
                     if "MAIN assignment" in json.dumps(request.get("system"))]
    first = main_requests[0]
    # The first provider request already offers exactly the bound host schemas beside the inspection tools.
    offered = {item["name"]: item.get("input_schema") for item in first["tools"]}
    assert set(offered) == {"Bash", "Glob", "Grep", "Read", *_HOST_NAMES}
    assert [offered[name] for name in _HOST_NAMES] == [_BATCH, _LOOKUP, _VERDICT]
    looked_up, refused_start, admitted, invalid, final = (_tool_results(request)[0]
                                                          for request in main_requests[1:6])
    assert source in _result_text(looked_up) and not looked_up.get("is_error")
    assert refused_start["is_error"] and ("No such tool" if route == "run_local" else "blocked") in _result_text(
        refused_start)
    assert '"admitted":1' in _result_text(admitted) and '"capacity"' in _result_text(admitted)
    assert invalid["is_error"] and 'at $["verdict"]: choose an allowed value' in _result_text(invalid)
    assert '"final":"concern"' in _result_text(final)
    # Every native start charged the one ledger: guard-denied, CLI-refused and rejected ones too.
    # A start after acceptance is denied and observed once as overshoot, never charged.
    assert installed == [{"note": "first batch"}, {"verdict": "concern"}]
    assert (binding.accepted_result, binding.calls, ledger.charged, binding.cleanup_complete) == (
        {"final": "concern"}, 5, 5, True)
    if route == "ordinary":
        (late,) = _tool_results(main_requests[6])
        assert late["is_error"] and "assignment has ended" in _result_text(late)
    assert ledger.observed == (6 if route == "ordinary" else 5)
    # Only exact host names are preapproved; strict MCP config excludes every other server.
    options = captured[0]
    assert set(options.allowed_tools) == {"Bash", "Glob", "Grep", "Read", *_HOST_NAMES}
    assert options.strict_mcp_config
    guards = options.hooks["PreToolUse"][0] if options.hooks else None
    for name, tool_input, allowed in [
        *((name, {}, True) for name in _HOST_NAMES), ("mcp__daydream__other", {}, False),
        ("mcp__external__review_source", {}, False), ("Write", {}, False), ("Task", {}, False),
        ("Bash", {"command": "rm -rf /"}, False), ("Bash", {"command": "git status", "run_in_background": True}, False),
    ]:
        decision = await _decide(guards, {"tool_name": name, "tool_input": tool_input})
        assert _is_denied(decision) is (not allowed), (name, tool_input)
    # Terminal usage and the host lifecycle survive a native failure after acceptance.
    document = json.loads(recorder.path.read_text())
    assert document["final_metrics"]["total_prompt_tokens"] > 0
    (invocation,) = document["extra"]["subtrajectories"]
    assert invocation["host_dispatch"]["system_disposition"] == "supplied"
    lifecycle = invocation["backend_lifecycle"]
    assert [lifecycle[key] for key in ("backend", "host_authority_closed", "host_native_settled",
                                       "host_cleanup_complete", "host_tool_call_count")] == [
        "claude", True, route == "ordinary", True, 5]


async def test_retired_host_callbacks_change_nothing_and_uncorrelated_calls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retired attempt's hook and MCP handler cannot charge, admit or revoke a newer attempt.

    This drives the exact callbacks the SDK invokes; the real CLI journeys above prove
    readiness. Acceptance ends the assignment, so a later start is observed overshoot.
    Unexpected host failures fail the binding closed instead of escaping: a raising SDK
    hook would let the start run.
    """
    installed: list[Any] = []
    ledger = BudgetLedger("stale", deadline=clock.monotonic() + 30)
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return {"final": value["verdict"]}

    binding = HostToolBinding(budget=ledger, accept=accept)
    binding.configure_schema(_VERDICT)
    start = {"tool_name": "mcp__daydream__structured_output", "tool_input": {"verdict": "clean"}}

    async def call(endpoint: Any, call_id: str | None) -> mcp_types.CallToolResult:
        meta = {"claudecode/toolUseId": call_id} if call_id is not None else None
        request = mcp_types.CallToolRequest(params=mcp_types.CallToolRequestParams.model_validate(
            {"name": "structured_output", "arguments": {"verdict": "clean"}, "_meta": meta}))
        result = await endpoint.server.request_handlers[mcp_types.CallToolRequest](request)
        return cast(mcp_types.CallToolResult, result.root)

    async with binding.attempt(claude_backend._ClaudeHostEndpoint) as first:
        assert await first.admit(start, "early", {}) == {}
    async with binding.attempt(claude_backend._ClaudeHostEndpoint) as second:
        assert _is_denied(await first.admit(start, "late", {}))
        assert (await call(first, "early")).isError
        assert (ledger.charged, ledger.observed, binding.fatal_error, binding.authority_open) == (1, 1, None, True)
        assert await second.admit(start, "fresh", {}) == {}
        assert not (await call(second, "fresh")).isError and binding.accepted and not binding.authority_open
        assert _is_denied(await second.admit(start, "overshoot", {}))
    assert (ledger.charged, ledger.observed, installed, binding.cleanup_complete) == (2, 3, [{"verdict": "clean"}],
                                                                                       True)
    # An MCP call without the CLI's tool identity cannot be correlated with its charged start.
    uncorrelated = HostToolBinding(deadline=clock.monotonic() + 30, accept=accept)
    uncorrelated.configure_schema(_VERDICT)
    async with uncorrelated.attempt(claude_backend._ClaudeHostEndpoint) as endpoint:
        assert await endpoint.admit(start, "admitted", {}) == {}
        assert (await call(endpoint, None)).isError
    assert uncorrelated.fatal_error is not None and uncorrelated.fatal_error.code == "native_correlation"
    assert len(installed) == 1 and not uncorrelated.accepted
    broken = HostToolBinding(deadline=clock.monotonic() + 30, accept=accept)
    broken.configure_schema(_VERDICT)

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("host defect")

    monkeypatch.setattr(broken, "submit", unexpected)
    monkeypatch.setattr(broken, "admit", unexpected)
    async with broken.attempt(claude_backend._ClaudeHostEndpoint) as endpoint:
        assert (await call(endpoint, "submitted")).isError
        assert broken.fatal_error is not None and broken.fatal_error.code == "callback_failed"
        assert _is_denied(await endpoint.admit(start, "next", {}))
    assert len(installed) == 1 and not broken.accepted

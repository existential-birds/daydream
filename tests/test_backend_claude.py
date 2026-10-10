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

from daydream.backends import (
    BackendExecutionInput,
    ClaudeRequestConfig,
    ContinuationToken,
    RequestEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    effective_fanout_concurrency,
)
from daydream.backends.claude import (
    ClaudeAgentError,
    ClaudeBackend,
    MaxTurnsError,
    _is_background_bash,
    _is_dangerous_command,
    _is_read_only_command,
    _RunLocalClaudeSDKClient,
    _RunLocalSubprocessCLITransport,
)
from daydream.config import TEST_WALL_BUDGET_S
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_claude_sdk,
    scripted_client,
)


async def test_artifact_visibility_protocol_sdk_query_observes_exact_options_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "model cwd with spaces").resolve()
    target.mkdir()
    (target / "source.py").write_text("SOURCE_CANARY\n", encoding="utf-8")
    observed: dict[str, Any] = {}
    base_client = scripted_client([
        MockAssistantMessage(content=[MockTextBlock(text="CURRENT_REASONING_CANARY")]),
        MockResultMessage(total_cost_usd=None),
    ])
    class ObservingClient(base_client):  # type: ignore[misc,valid-type]
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            cwd = Path(self.options.cwd)
            observed.update(
                cwd=str(cwd), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                source_seen="SOURCE_CANARY" in (cwd / "source.py").read_text(encoding="utf-8"),
                permission_mode=self.options.permission_mode, tools=self.options.allowed_tools,
            )
    patch_claude_sdk(monkeypatch, ObservingClient)
    backend = ClaudeBackend(model="fixture-model")
    prompt = "Inspect the committed source only."
    events = [event async for event in backend.execute(target, prompt)]
    assert observed["cwd"] == str(target)
    assert observed["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert observed["source_seen"] is True
    assert observed["permission_mode"] == "bypassPermissions"
    assert observed["tools"] == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    assert any(isinstance(event, TextEvent) and event.text == "CURRENT_REASONING_CANARY" for event in events)
    assert len([event for event in events if isinstance(event, ResultEvent)]) == 1
    assert backend._active_clients == set()

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

async def _drive_claude_backend_to_list(
    *, messages: list[Any], patch_sdk_fn: Any, output_schema: Any = None, prompt: str = "go",
) -> list[Any]:
    patch_sdk_fn(scripted_client(messages))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), prompt, output_schema=output_schema):
        events.append(event)
    return events


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
    """An SDK error raises after terminal metadata, never becoming a clean empty review."""
    patch_sdk(
        scripted_client(
            [
                MockAssistantMessage(content=[MockTextBlock(text="Invalid API key · Fix external API key")]),
                MockResultMessage(total_cost_usd=None, is_error=True, result="Invalid API key · Fix external API key"),
            ]
        )
    )
    backend = ClaudeBackend(model="opus")
    events = []
    with pytest.raises(ClaudeAgentError, match="Invalid API key"):
        async for event in backend.execute(Path("/tmp"), "Review this"):
            events.append(event)
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.continuation is None

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

@pytest.mark.parametrize("cmd,allowed", [
    ("git log --oneline -5", True), ("git blame -L 10,10 daydream/phases.py", True),
    ("git show 648a327 -- daydream/phases.py", True), ("git diff main..HEAD -- daydream/phases.py", True),
    ("cat README.md", True), ("git status", True), ("ls -la", True), ("git commit -m x", False), ("git add -A", False),
    ("git checkout -- f.py", False), ("rm -rf build", False), ("touch newfile", False),
    ("git status && rm x", False),       # chain into mutating token
    ("cat f | tee g", False),            # pipe into non-allowlisted
    ("git log; rm x", False),            # semicolon chain
    ("echo $(rm x)", False),             # command substitution
    ("git logfoo", False),               # prefix must be word-bounded
    ("git status > status.txt", False),  # output redirection creates/truncates a file
    ("cat README.md >> copy.txt", False),# append redirection
    ("cat < README.md", False),          # input redirection
    ("git log foo# > status.txt", False),# mid-word '#' must not blind the redirect scan
    ("cat a#b > f", False),               # mid-word '#' hidden in an arg, then redirect
    ("git log foo#; rm x", False),        # mid-word '#' then a chaining operator
    ("git diff --output=diff.patch", False),  # git output-file option, equals form
    ("git log --output log.txt", False), # git output-file option, separated form
    ("( rm x )", False),                 # command-leading subshell group executes
    ("(rm x)", False),                   # spacing-free subshell at command start still denies
    ("(git status > out.txt)", False),   # subshell wrapping a redirect still denies
    ("ls -la ( x )", True),              # mid-command parens: bash syntax error, nothing runs
    ("git log --format=%C(red)%h", True),# git color placeholder: parens glued to a word
    ("cat foo(1).txt", True),            # parens glued into a filename word
    ("git log\nrm x", False),           # newline command separator -> fail closed
    ("git log 'unclosed", False),        # malformed quoting -> fail closed
    ("git log --grep='fix|bug'", True),  # quoted | is argument content, not an operator
    ("git diff HEAD~1 HEAD -- --output", True),  # --output after -- is a path arg; scan stops at --
    ("", False),                          # empty → fail closed
])
def test_read_only_bash_guard_decision(cmd: Any, allowed: Any) -> None:
    assert _is_read_only_command(cmd) is allowed

@pytest.mark.parametrize("cmd, dangerous", [
    ("find / -path '*x*'", True), ("find / -name y", True), ("grep -r pattern /", True), ("rm -rf /", True),
    ("rm -fr /", True),                       # reversed flags
    ("rm -rf /*", True),                      # root glob
    ("rm --recursive --force /", True),       # long-form flags
    ("rm -Rf /", True),                       # capital-R recursive
    ("rm -rf foo /", True),                   # trailing root arg
    # Reordering recursive flags must not evade the ownership guard.
    ("rm --force --recursive /", True),       # long-form, recursive second
    ("rm -f -r /", True),                     # short-form, recursive second
    ("rm -i -r /*", True),                    # recursive second, root glob
    ("rm -rf /home/user/tmp", False),         # subpath under / is left alone
    ("rm -rf build", False),                  # relative path
    # F12: `/` as the grep pattern (not a root path) is no longer a false positive.
    ("grep / file.txt", False),               # searching for a literal slash
    ("find core/osprey-tui -name agent.rs", False), ("ls", False), ("rg foo src/", False),
])
def test_is_dangerous_command(cmd: Any, dangerous: Any) -> None:
    """The always-on dangerous-command predicate flags root-scans and catastrophic deletes."""
    assert _is_dangerous_command(cmd) is dangerous

async def test_read_only_execute_registers_pretooluse_guard(patch_sdk: Any) -> None:
    """Drive registered guards: bypassPermissions needs all-tool, deny-by-default enforcement.

    Mutation and unknown tools deny; inspection remains available.
    """
    backend, captured = _capturing_backend(patch_sdk)
    async for _ in backend.execute(Path("/tmp"), "Go", read_only=True):
        pass
    opts = captured["options"]
    assert opts is not None
    hooks = opts.hooks
    assert hooks is not None and "PreToolUse" in hooks
    matchers = hooks["PreToolUse"]
    assert len(matchers) == 1
    matcher = matchers[0]
    assert matcher.hooks  # callbacks registered
    deny_write = await _decide(matcher, {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}})
    assert deny_write["hookSpecificOutput"]["permissionDecision"] == "deny"
    write_reason = deny_write["hookSpecificOutput"]["permissionDecisionReason"]
    assert "read-only guard" in write_reason
    assert "read-only summarizer" not in write_reason
    deny_missing = await _decide(matcher, {"tool_name": "Bash"})
    assert deny_missing["hookSpecificOutput"]["permissionDecision"] == "deny"
    deny_delete = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}})
    assert deny_delete["hookSpecificOutput"]["permissionDecision"] == "deny"
    delete_reason = deny_delete["hookSpecificOutput"]["permissionDecisionReason"]
    assert "read-only guard" in delete_reason
    assert "read-only summarizer" not in delete_reason
    deny_bash = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"command": "git commit -m x"}})
    assert deny_bash["hookSpecificOutput"]["permissionDecision"] == "deny"
    deny_git_output = await _decide(matcher,
        {"tool_name": "Bash", "tool_input": {"command": "git diff --output=diff.patch"}}
    )
    assert deny_git_output["hookSpecificOutput"]["permissionDecision"] == "deny"
    allow_bash = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"command": "git log -n 5"}})
    assert "hookSpecificOutput" not in allow_bash
    allow_read = await _decide(matcher, {"tool_name": "Read", "tool_input": {"file_path": "x"}})
    assert "hookSpecificOutput" not in allow_read
    # Fail-closed: a narrow deny-list matcher would never present an unknown tool to the guard.
    deny_unknown = await _decide(matcher, {"tool_name": "FutureMutator", "tool_input": {}})
    assert deny_unknown["hookSpecificOutput"]["permissionDecision"] == "deny"
    deny_find_root = await _decide(matcher, {"tool_name": "Bash", "tool_input": {"command": "find / -name x"}})
    assert deny_find_root["hookSpecificOutput"]["permissionDecision"] == "deny"



async def test_structured_output_tool_result_is_suppressed(patch_sdk: Any) -> None:
    """StructuredOutput ToolUseBlocks are skipped, and the corresponding
    ToolResultBlock in the next UserMessage must also be skipped — otherwise
    the trajectory recorder logs it as an unmatched_tool_result."""
    events = await _drive_claude_backend_to_list(
        messages=[
            MockAssistantMessage(content=[
                MockToolUseBlock(id="tool-real", name="Read", input={"file": "a.py"}),
                MockToolUseBlock(id="tool-so", name="StructuredOutput", input={"result": "{}"}),
            ]),
            MockUserMessage(content=[
                MockToolResultBlock(tool_use_id="tool-real", content="file contents"),
                MockToolResultBlock(tool_use_id="tool-so", content='{"data": 1}'),
            ]), MockResultMessage(total_cost_usd=0.01, structured_output={"data": 1}),
        ], patch_sdk_fn=patch_sdk,
    )
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_starts) == 1
    assert tool_starts[0].name == "Read"
    assert tool_starts[0].id == "tool-real"
    assert len(tool_results) == 1
    assert tool_results[0].id == "tool-real"
    assert tool_results[0].output == "file contents"
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert result_events[0].structured_output == {"data": 1}


@pytest.mark.parametrize("effort", [None, "max"])
async def test_reasoning_effort_reaches_sdk_options(patch_sdk: Any, effort: str | None) -> None:
    backend, captured = _capturing_backend(patch_sdk, reasoning_effort=effort)
    async for _ in backend.execute(Path("/tmp"), "go"):
        pass
    assert captured["options"].effort == effort


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

def test_claude_fanout_concurrency_defaults_to_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_FANOUT_CONCURRENCY", raising=False)
    assert effective_fanout_concurrency(10, ClaudeBackend(model="opus")) == 8

def test_claude_fanout_concurrency_never_exceeds_workflow_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", "8")
    assert effective_fanout_concurrency(2, ClaudeBackend(model="opus")) == 2

@pytest.mark.parametrize(("env_value", "expected"), [ ("6", 6), ("0", 8), ("-1", 8), ("notanint", 8), ("", 8), ])
def test_claude_fanout_concurrency_env_validation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, env_value: Any, expected: int,
) -> None:
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", env_value)
    assert effective_fanout_concurrency(10, ClaudeBackend(model="opus")) == expected
    if expected == 8:
        assert "using default 8" in caplog.text

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

@pytest.mark.parametrize("persist_session", [False, True])
@pytest.mark.parametrize("session_id", [None, "sess-9"])
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

@pytest.mark.parametrize(
    ("payload", "background"),
    [
        ({"tool_name": "Bash", "tool_input": {"command": "make test", "run_in_background": True}}, True),
        ({"tool_name": "Bash", "tool_input": {"command": "make test", "run_in_background": False}}, False),
        ({"tool_name": "Bash", "tool_input": {"command": "make test"}}, False),
        ({"tool_name": "Bash", "tool_input": {"run_in_background": True}}, True),
        ({"tool_name": "Read", "tool_input": {"run_in_background": True}}, False),
        ({"tool_name": "Bash", "tool_input": "not-a-dict"}, False), ("garbage", False), (None, False),
    ],
)
def test_is_background_bash(payload: Any, background: bool) -> None:
    """Only a Bash payload with a truthy ``run_in_background`` is a background call."""
    assert _is_background_bash(payload) is background

# --- P18 Task 1: effective request-config admission at the Claude SDK seam ---

async def _request_event(apply_patch: Any, prompt: str, **execute_kwargs: Any) -> RequestEvent:
    backend, _ = _capturing_backend(apply_patch)
    events = [event async for event in backend.execute(Path("/tmp"), prompt, **execute_kwargs)]
    return next(e for e in events if isinstance(e, RequestEvent))


async def test_request_event_resume_provenance_is_host_generated(patch_sdk: Any) -> None:
    token = ContinuationToken(backend="claude", data={"session_id": "sess-42"})
    request = await _request_event(patch_sdk, "again", continuation=token)
    config = request.config
    assert isinstance(config, ClaudeRequestConfig)
    assert config.continuation_mode == "resume"
    assert request.session_id == "sess-42"
    assert request.session_source == "host_generated"

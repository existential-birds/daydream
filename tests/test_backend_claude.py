"""Tests for ClaudeBackend."""
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import (
    ClaudeRequestConfig,
    ContinuationToken,
    CostEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    effective_fanout_concurrency,
)
from daydream.backends.claude import ClaudeAgentError, ClaudeBackend
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


@pytest.mark.asyncio
async def test_artifact_visibility_protocol_sdk_query_observes_exact_options_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

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
                permission_mode=self.options.permission_mode,
                tools=self.options.allowed_tools,
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
    """Return a function that patches the SDK imports in claude.py."""
    def _patch(client_class: Any) -> None:
        patch_claude_sdk(monkeypatch, client_class)
    return _patch


def _capturing_client(captured: dict[str, Any]) -> type:
    """A scripted client that records the ``ClaudeAgentOptions`` it was built with."""
    return scripted_client(
        [
            MockAssistantMessage(content=[MockTextBlock(text="OK")]),
            MockResultMessage(total_cost_usd=0.01),
        ],
        captured=captured,
    )


@pytest.mark.parametrize(
    "version_admission_delay_s",
    [pytest.param(0.0, id="version-completes"), pytest.param(2.1, id="version-times-out")],
)
@pytest.mark.asyncio
async def test_injected_claude_uses_run_local_transport_for_version_and_main_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_admission_delay_s: float,
) -> None:
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    from daydream.backends.claude import (
        _RunLocalClaudeSDKClient,
        _RunLocalSubprocessCLITransport,
    )

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
        "PATH": str(tmp_path),
        "HOME": str(tmp_path / "run-home"),
        "CAPTURE_LOG": str(capture),
        "ANTHROPIC_API_KEY": "run-key",
        "PWD": "/run/pwd",
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
        cli_path=str(cli),
        cwd=str(tmp_path),
        env=dict(environment),
        hooks={
            "PreToolUse": [HookMatcher(matcher=None, hooks=[cast(Any, _hook)])]
        },
    )
    transport = _RunLocalSubprocessCLITransport(options, environment=environment)

    async with _RunLocalClaudeSDKClient(
        options=options,
        transport=transport,
        initialize_timeout_s=60.0,
    ) as client:
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


@pytest.mark.asyncio
async def test_claude_backend_injected_environment_reaches_sdk_options_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.backends import BackendExecutionInput, RetryPolicy

    captured: dict[str, Any] = {}
    base_client = scripted_client(
        [MockAssistantMessage(content=[MockTextBlock(text="OK")]), MockResultMessage()]
    )

    class CapturingClient(base_client):  # type: ignore[misc,valid-type]
        def __init__(self, options: Any = None, transport: Any = None) -> None:
            super().__init__(options=options)
            captured["options"] = options
            captured["transport"] = transport

    patch_claude_sdk(monkeypatch, CapturingClient)
    execution = BackendExecutionInput.from_environment(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": "/run/bin",
            "ANTHROPIC_API_KEY": "run-key",
            "DAYDREAM_FANOUT_CONCURRENCY": "4",
            "DAYDREAM_PI_RETRY_ATTEMPTS": "3",
        },
        backend="claude",
    )
    backend = ClaudeBackend(model="opus", execution_input=execution)

    _ = [event async for event in backend.execute(tmp_path, "review")]

    assert captured["options"].env["ANTHROPIC_API_KEY"] == "run-key"
    assert captured["options"].env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert backend.fanout_concurrency == 4
    assert backend.retry_policy == RetryPolicy(3, 2.0, 120.0)


async def _drive_claude_backend_to_list(
    *,
    messages: list[Any],
    patch_sdk_fn: Any,
    output_schema: Any = None,
    prompt: str = "go",
) -> list[Any]:
    """Drive ClaudeBackend.execute with a scripted SDK message sequence."""
    patch_sdk_fn(scripted_client(messages))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), prompt, output_schema=output_schema):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_execute_yields_text_and_result(patch_sdk: Any) -> None:
    events = await _drive_claude_backend_to_list(
        messages=[
            MockAssistantMessage(content=[MockTextBlock(text="Hello world")]),
            MockResultMessage(total_cost_usd=0.05, structured_output=None),
        ],
        patch_sdk_fn=patch_sdk,
        prompt="Say hello",
    )

    text_events = [e for e in events if isinstance(e, TextEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]

    assert len(text_events) == 1
    assert text_events[0].text == "Hello world"
    assert len(cost_events) == 1
    assert cost_events[0].cost_usd == 0.05
    assert len(result_events) == 1
    assert result_events[0].structured_output is None
    assert result_events[0].continuation is None


@pytest.mark.asyncio
async def test_execute_yields_tool_events(patch_sdk: Any) -> None:
    events = await _drive_claude_backend_to_list(
        messages=[
            MockAssistantMessage(content=[MockThinkingBlock(thinking="Let me think...")]),
            MockAssistantMessage(content=[MockTextBlock(text="I'll run a command.")]),
            MockAssistantMessage(content=[MockToolUseBlock(id="tool-1", name="Bash", input={"command": "ls"})]),
            MockUserMessage(content=[MockToolResultBlock(tool_use_id="tool-1", content="file.py", is_error=False)]),
            MockResultMessage(total_cost_usd=0.10),
        ],
        patch_sdk_fn=patch_sdk,
        prompt="Run ls",
    )

    thinking_events = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_start_events = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]

    assert len(thinking_events) == 1
    assert thinking_events[0].text == "Let me think..."
    assert len(tool_start_events) == 1
    assert tool_start_events[0].name == "Bash"
    assert tool_start_events[0].input == {"command": "ls"}
    assert len(tool_result_events) == 1
    assert tool_result_events[0].output == "file.py"
    assert tool_result_events[0].is_error is False
    assert tool_start_events[0].id == tool_result_events[0].id


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_execute_structured_output(patch_sdk: Any) -> None:
    events = await _drive_claude_backend_to_list(
        messages=[
            MockAssistantMessage(content=[MockTextBlock(text="Parsed.")]),
            MockResultMessage(
                total_cost_usd=0.02,
                structured_output={"issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]},
            ),
        ],
        patch_sdk_fn=patch_sdk,
        prompt="Parse",
        output_schema={"type": "object", "properties": {"issues": {"type": "array"}}},
    )

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]
    }


@pytest.mark.asyncio
async def test_error_result_raises_instead_of_clean_empty_result(patch_sdk: Any) -> None:
    """An is_error ResultMessage must raise after exposing terminal metadata.

    Regression guard for the sandbox acceptance failure: an invalid API key
    run streamed the error text and a ResultMessage(is_error=True), and the
    backend yielded a clean ResultEvent — the review then exited 0 with
    "no issues found" despite the agent never running.
    """
    patch_sdk(
        scripted_client(
            [
                MockAssistantMessage(content=[MockTextBlock(text="Invalid API key · Fix external API key")]),
                MockResultMessage(
                    total_cost_usd=None,
                    is_error=True,
                    result="Invalid API key · Fix external API key",
                ),
            ]
        )
    )
    backend = ClaudeBackend(model="opus")
    events = []
    with pytest.raises(ClaudeAgentError, match="Invalid API key"):
        async for event in backend.execute(Path("/tmp"), "Review this"):
            events.append(event)
    # A terminal metadata event does not turn the failed stream into success.
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.continuation is None


@pytest.mark.asyncio
async def test_max_turns_result_raises_typed_error(patch_sdk: Any) -> None:
    """error_max_turns must raise the typed MaxTurnsError carrying the subtype.

    A generic ClaudeAgentError left callers (and the trajectory) unable to
    distinguish a turn-cap failure from a real backend error. Mirrors the real
    SDK shape: a ResultMessage with ``is_error=True`` and
    ``subtype="error_max_turns"`` (``result`` is None, so the detail falls back
    to the subtype).
    """
    from daydream.backends.claude import MaxTurnsError

    patch_sdk(
        scripted_client(
            [
                MockAssistantMessage(content=[MockTextBlock(text="working on it")]),
                MockResultMessage(
                    total_cost_usd=None,
                    is_error=True,
                    result=None,
                    subtype="error_max_turns",
                ),
            ]
        )
    )
    backend = ClaudeBackend(model="opus")
    with pytest.raises(MaxTurnsError) as excinfo:
        async for _ in backend.execute(Path("/tmp"), "Review this"):
            pass
    # Subtype is carried for trajectory recording; still a ClaudeAgentError.
    assert excinfo.value.subtype == "error_max_turns"
    assert isinstance(excinfo.value, ClaudeAgentError)


@pytest.mark.parametrize("cmd,allowed", [
    ("git log --oneline -5", True),
    ("git blame -L 10,10 daydream/phases.py", True),
    ("git show 648a327 -- daydream/phases.py", True),
    ("git diff main..HEAD -- daydream/phases.py", True),
    ("cat README.md", True),
    ("git status", True),
    ("ls -la", True),
    ("git commit -m x", False),
    ("git add -A", False),
    ("git checkout -- f.py", False),
    ("rm -rf build", False),
    ("touch newfile", False),
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
    """The read-only Bash allowlist predicate allows inspection, denies mutation/chains."""
    from daydream.backends.claude import _is_read_only_command

    assert _is_read_only_command(cmd) is allowed


@pytest.mark.parametrize("cmd, dangerous", [
    ("find / -path '*x*'", True),
    ("find / -name y", True),
    ("grep -r pattern /", True),
    ("rm -rf /", True),
    # F2: catastrophic wipe shapes the old `-rf?` literal missed.
    ("rm -fr /", True),                       # reversed flags
    ("rm -rf /*", True),                      # root glob
    ("rm --recursive --force /", True),       # long-form flags
    ("rm -Rf /", True),                       # capital-R recursive
    ("rm -rf foo /", True),                   # trailing root arg
    # Recursive flag NOT first: a non-recursive option token preceding it must
    # not let the wipe slip past the guard (CodeRabbit #185).
    ("rm --force --recursive /", True),       # long-form, recursive second
    ("rm -f -r /", True),                     # short-form, recursive second
    ("rm -i -r /*", True),                    # recursive second, root glob
    ("rm -rf /home/user/tmp", False),         # subpath under / is left alone
    ("rm -rf build", False),                  # relative path
    # F12: `/` as the grep pattern (not a root path) is no longer a false positive.
    ("grep / file.txt", False),               # searching for a literal slash
    ("find core/osprey-tui -name agent.rs", False),
    ("ls", False),
    ("rg foo src/", False),
])
def test_is_dangerous_command(cmd: Any, dangerous: Any) -> None:
    """The always-on dangerous-command predicate flags root-scans and catastrophic deletes."""
    from daydream.backends.claude import _is_dangerous_command

    assert _is_dangerous_command(cmd) is dangerous


@pytest.mark.asyncio
async def test_read_only_execute_registers_pretooluse_guard(patch_sdk: Any) -> None:
    """read_only=True wires a fail-closed PreToolUse guard onto the SDK options.

    The contract is behavioral, not a matcher-string shape: under
    ``bypassPermissions`` the hook is the *only* enforcement, so the guard must
    fire for every tool and deny-by-default. We assert that by driving the
    callback that was actually registered on the options — denying Write and
    mutating Bash, allowing inspection (read-only Bash + allowlisted tools), and
    denying an unknown/future tool (the fail-closed property a narrow matcher
    would silently lose).
    """
    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")

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

    async def decide(payload: Any) -> Any:
        # Mirror the SDK: run every registered hook; first deny wins.
        for hook in matcher.hooks:
            out = await hook(payload, None, {})
            if out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny":
                return out
        return {}

    deny_write = await decide(
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}
    )
    assert deny_write["hookSpecificOutput"]["permissionDecision"] == "deny"

    deny_bash = await decide(
        {"tool_name": "Bash", "tool_input": {"command": "git commit -m x"}}
    )
    assert deny_bash["hookSpecificOutput"]["permissionDecision"] == "deny"

    deny_git_output = await decide(
        {"tool_name": "Bash", "tool_input": {"command": "git diff --output=diff.patch"}}
    )
    assert deny_git_output["hookSpecificOutput"]["permissionDecision"] == "deny"

    allow_bash = await decide(
        {"tool_name": "Bash", "tool_input": {"command": "git log -n 5"}}
    )
    assert "hookSpecificOutput" not in allow_bash
    allow_read = await decide({"tool_name": "Read", "tool_input": {"file_path": "x"}})
    assert "hookSpecificOutput" not in allow_read

    # Fail-closed: a narrow deny-list matcher would never present an unknown tool to the guard.
    deny_unknown = await decide({"tool_name": "FutureMutator", "tool_input": {}})
    assert deny_unknown["hookSpecificOutput"]["permissionDecision"] == "deny"

    # read_only=True also composes the always-on dangerous-command guard: a root scan denies.
    deny_find_root = await decide(
        {"tool_name": "Bash", "tool_input": {"command": "find / -name x"}}
    )
    assert deny_find_root["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_non_read_only_execute_registers_dangerous_command_hook(patch_sdk: Any) -> None:
    """read_only=False (default) still wires the always-on dangerous-command guard.

    The guard is registered unconditionally (all phases). Drive the callback that
    was actually built onto the production options: a ``find /`` root scan denies,
    a scoped ``find core/...`` allows.
    """
    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")

    async for _ in backend.execute(Path("/tmp"), "Go", read_only=False):
        pass

    opts = captured["options"]
    assert opts is not None
    hooks = opts.hooks
    assert hooks is not None and "PreToolUse" in hooks
    matchers = hooks["PreToolUse"]
    assert len(matchers) == 1
    guard = matchers[0].hooks[0]

    deny_find_root = await guard(
        {"tool_name": "Bash", "tool_input": {"command": "find / -name x"}}, None, {}
    )
    assert deny_find_root["hookSpecificOutput"]["permissionDecision"] == "deny"

    allow_find_scoped = await guard(
        {"tool_name": "Bash", "tool_input": {"command": "find core/osprey-tui -name agent.rs"}},
        None,
        {},
    )
    assert "hookSpecificOutput" not in allow_find_scoped


@pytest.mark.asyncio
async def test_read_only_guard_denies_mutation_allows_inspection() -> None:
    """The registered guard callback denies Write and non-read-only Bash, allows read-only Bash."""
    from daydream.backends.claude import _read_only_guard

    deny_write = cast(dict[str, Any], await _read_only_guard(
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, None, {},
    ))
    assert deny_write["hookSpecificOutput"]["permissionDecision"] == "deny"

    deny_bash = cast(dict[str, Any], await _read_only_guard(
        {"tool_name": "Bash", "tool_input": {"command": "git commit -m x"}}, None, {},
    ))
    assert deny_bash["hookSpecificOutput"]["permissionDecision"] == "deny"

    allow_bash = await _read_only_guard(
        {"tool_name": "Bash", "tool_input": {"command": "git log -n 5"}}, None, {},
    )
    assert "hookSpecificOutput" not in allow_bash

    # Malformed input → fail closed (deny)
    deny_malformed = cast(dict[str, Any], await _read_only_guard({"tool_name": "Bash"}, None, {}))
    assert deny_malformed["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_read_only_guard_deny_reason_uses_shared_guard_wording() -> None:
    """The guard is shared (diagnostic subagents, failure summarizer, exploration
    specialists), so its deny reasons must say 'read-only guard', not the stale
    'read-only summarizer'."""
    from daydream.backends.claude import _read_only_guard

    deny_bash = cast(
        dict[str, Any],
        await _read_only_guard(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}}, None, {},
        ),
    )
    bash_reason = deny_bash["hookSpecificOutput"]["permissionDecisionReason"]
    assert "read-only guard" in bash_reason
    assert "read-only summarizer" not in bash_reason

    deny_tool = cast(
        dict[str, Any],
        await _read_only_guard(
            {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, None, {},
        ),
    )
    tool_reason = deny_tool["hookSpecificOutput"]["permissionDecisionReason"]
    assert "read-only guard" in tool_reason
    assert "read-only summarizer" not in tool_reason


@pytest.mark.asyncio
async def test_execute_passes_agents_dict_to_options(patch_sdk: Any) -> None:
    """Agents dict must reach ClaudeAgentOptions with original keys preserved verbatim."""
    from claude_agent_sdk.types import AgentDefinition

    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")

    pattern_scanner = AgentDefinition(
        description="pattern scanner",
        prompt="scan patterns",
        tools=["Read", "Grep"],
        model="sonnet",
    )
    dependency_tracer = AgentDefinition(
        description="dependency tracer",
        prompt="trace deps",
        tools=["Read", "Grep"],
        model="sonnet",
    )

    agents = {
        "pattern-scanner": pattern_scanner,
        "dependency-tracer": dependency_tracer,
    }

    events = []
    async for event in backend.execute(Path("/tmp"), "Go", agents=agents):
        events.append(event)

    opts = captured["options"]
    assert opts is not None
    assert opts.agents == {
        "pattern-scanner": pattern_scanner,
        "dependency-tracer": dependency_tracer,
    }
    assert "explorer-0" not in opts.agents
    assert "explorer-1" not in opts.agents


@pytest.mark.asyncio
async def test_execute_passes_none_when_no_agents(patch_sdk: Any) -> None:
    """When agents=None, ClaudeAgentOptions should not carry an agents dict."""
    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")

    events = []
    async for event in backend.execute(Path("/tmp"), "Go"):
        events.append(event)

    opts = captured["options"]
    assert opts is not None
    agents_val = getattr(opts, "agents", None)
    assert agents_val is None


# Helpers for TurnEndEvent tests (Task 6)


def _assistant_message(*, text: str, message_id: str) -> MockAssistantMessage:
    """Build a MockAssistantMessage carrying one TextBlock + a message_id."""
    msg = MockAssistantMessage(content=[MockTextBlock(text=text)])
    msg.message_id = message_id  # type: ignore[attr-defined]
    return msg


def _result_message(*, cost: float | None = 0.0) -> MockResultMessage:
    return MockResultMessage(total_cost_usd=cost, structured_output=None)


@pytest.mark.asyncio
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
            ]),
            MockResultMessage(total_cost_usd=0.01, structured_output={"data": 1}),
        ],
        patch_sdk_fn=patch_sdk,
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


@pytest.mark.asyncio
async def test_claude_backend_emits_turn_end_per_assistant_message(patch_sdk: Any) -> None:
    """One TurnEndEvent per AssistantMessage, after that message's events."""
    from daydream.backends import TurnEndEvent

    events = await _drive_claude_backend_to_list(
        messages=[
            _assistant_message(text="turn-1", message_id="msg_1"),
            _assistant_message(text="turn-2", message_id="msg_2"),
            _result_message(cost=0.0),
        ],
        patch_sdk_fn=patch_sdk,
    )
    texts = [e for e in events if isinstance(e, TextEvent)]
    turn_ends = [(i, e) for i, e in enumerate(events) if isinstance(e, TurnEndEvent)]
    assert [e.text for e in texts] == ["turn-1", "turn-2"]
    assert len(turn_ends) == 2
    assert turn_ends[0][1].message_id == "msg_1"
    assert turn_ends[1][1].message_id == "msg_2"
    first_text_idx = events.index(texts[0])
    second_text_idx = events.index(texts[1])
    assert first_text_idx < turn_ends[0][0] < second_text_idx
    assert second_text_idx < turn_ends[1][0]


@pytest.mark.asyncio
async def test_reasoning_effort_reaches_sdk_options_as_effort(patch_sdk: Any) -> None:
    """The resolved per-phase effort arrives as ClaudeAgentOptions.effort."""
    captured: dict[str, Any] = {}

    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus", reasoning_effort="max")
    async for _ in backend.execute(Path("/tmp"), "go"):
        pass

    assert captured["options"].effort == "max"


@pytest.mark.asyncio
async def test_no_reasoning_effort_leaves_sdk_effort_unset(patch_sdk: Any) -> None:
    captured: dict[str, Any] = {}

    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")
    async for _ in backend.execute(Path("/tmp"), "go"):
        pass

    assert captured["options"].effort is None


def test_unsupported_reasoning_effort_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        ClaudeBackend(model="opus", reasoning_effort="minimal")


async def _audit_decision(backend: ClaudeBackend, payload: Any) -> Any:
    guard = backend._audit_root_guard  # noqa: SLF001 - security contract seam
    assert guard is not None
    return await guard(payload, None, {"signal": None})


def _is_denied(decision: Any) -> bool:
    return bool(
        decision.get("hookSpecificOutput", {}).get("permissionDecision")
        == "deny"
    )


@pytest.mark.asyncio
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
    backend = ClaudeBackend(
        model="opus",
        audit_root=root,
        audit_outward_symlinks=frozenset({outward}),
    )

    allowed = [
        {"tool_name": "StructuredOutput", "tool_input": {"result": {}}},
        {"tool_name": "Read", "tool_input": {"file_path": "clean/inside.py"}},
        {
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "needle",
                "path": str(inside),
                "output_mode": "content",
                "-n": True,
                "head_limit": 5,
            },
        },
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "**/*.py"}},
    ]
    for payload in allowed:
        assert not _is_denied(await _audit_decision(backend, payload)), payload

    denied = [
        {},
        {"tool_name": "Read", "tool_input": {}},
        {"tool_name": "Read", "tool_input": {"file_path": 7}},
        {"tool_name": "Read", "tool_input": {"file_path": "../source/secret.txt"}},
        {
            "tool_name": "Read",
            "tool_input": {"file_path": str(clean / ".." / "clean" / "inside.py")},
        },
        {"tool_name": "Read", "tool_input": {"file_path": str(secret)}},
        {"tool_name": "Read", "tool_input": {"file_path": "escape/secret.txt"}},
        {"tool_name": "Read", "tool_input": {"file_path": "clean/inside.py", "pages": "1"}},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle"}},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle", "path": "clean"}},
        {"tool_name": "Grep", "tool_input": {"pattern": "needle", "path": str(secret)}},
        {
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "needle",
                "path": str(inside),
                "output_mode": [],
            },
        },
        {
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "needle",
                "path": str(inside),
                "output_mode": {},
            },
        },
        {"tool_name": "Grep", "tool_input": {"pattern": "needle", "path": str(inside), "glob": "*"}},
        {"tool_name": "Glob", "tool_input": {"pattern": "clean/*.py"}},
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "../*.py"}},
        {
            "tool_name": "Glob",
            "tool_input": {"path": "sub", "pattern": "loop/escape/*"},
        },
        {
            "tool_name": "Glob",
            "tool_input": {"path": "sub/loop/clean", "pattern": "*.py"},
        },
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "C:\\*"}},
        {"tool_name": "Glob", "tool_input": {"path": "clean", "pattern": "\\\\server\\share"}},
        {"tool_name": "Bash", "tool_input": {"command": "cat ../source/secret.txt"}},
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "x"}},
        {"tool_name": "Task", "tool_input": {}},
        {"tool_name": "Skill", "tool_input": {}},
        {"tool_name": "mcp__server__read", "tool_input": {}},
        {"tool_name": "FutureTool", "tool_input": {}},
    ]
    for payload in denied:
        assert _is_denied(await _audit_decision(backend, payload)), payload


@pytest.mark.asyncio
async def test_audit_execute_builds_closed_sdk_options_and_environment(
    patch_sdk: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit root"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    captured: dict[str, Any] = {}
    patch_sdk(
        scripted_client(
            [MockResultMessage(total_cost_usd=0.01, session_id="must-not-persist")],
            captured=captured,
        )
    )
    for variable in (
        "PWD",
        "OLDPWD",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_PREFIX",
    ):
        monkeypatch.setenv(variable, str(source))
    backend = ClaudeBackend(model="opus", audit_root=root)

    results = [
        event
        async for event in backend.execute(root, "audit", read_only=True)
        if isinstance(event, ResultEvent)
    ]

    options = captured["options"]
    assert options.tools == ["Read", "Grep", "Glob", "StructuredOutput"]
    assert options.allowed_tools == ["Read", "Grep", "Glob", "StructuredOutput"]
    assert options.mcp_servers == {}
    assert options.strict_mcp_config is True
    assert options.setting_sources == []
    assert options.skills == []
    assert options.plugins == []
    assert options.agents is None
    assert options.resume is None
    assert options.extra_args == {"no-session-persistence": None}
    assert options.env == {
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "BASH_DEFAULT_TIMEOUT_MS": options.env["BASH_DEFAULT_TIMEOUT_MS"],
        "BASH_MAX_TIMEOUT_MS": options.env["BASH_MAX_TIMEOUT_MS"],
        "PWD": str(root.resolve()),
        "OLDPWD": str(root.resolve()),
        "GIT_DIR": str(root.resolve() / ".git"),
        "GIT_WORK_TREE": str(root.resolve()),
        "GIT_INDEX_FILE": str(root.resolve() / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(root.resolve() / ".git" / "objects"),
        "GIT_COMMON_DIR": str(root.resolve() / ".git"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "",
        "GIT_CEILING_DIRECTORIES": str(root.resolve().parent),
        "GIT_PREFIX": "",
    }
    assert results[0].continuation is None
    matchers = options.hooks["PreToolUse"]
    assert len(matchers) == 1
    assert matchers[0].matcher == ".*"
    assert matchers[0].hooks == [backend._audit_root_guard]  # noqa: SLF001


@pytest.mark.parametrize("bad_call", ["cwd", "read_only", "continuation", "agents"])
@pytest.mark.asyncio
async def test_audit_execute_rejects_unsafe_invocation_before_client(
    patch_sdk: Any,
    tmp_path: Path,
    bad_call: str,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus", audit_root=root)
    kwargs: dict[str, Any] = {"read_only": True}
    cwd = root
    if bad_call == "cwd":
        cwd = other
    elif bad_call == "read_only":
        kwargs["read_only"] = False
    elif bad_call == "continuation":
        kwargs["continuation"] = ContinuationToken(
            backend="claude", data={"session_id": "old"}
        )
    else:
        kwargs["agents"] = {"unsafe": object()}

    with pytest.raises(ClaudeAgentError, match="audit isolation"):
        async for _ in backend.execute(cwd, "audit", **kwargs):
            pass

    assert "options" not in captured


@pytest.mark.asyncio
async def test_audit_guard_round_trips_through_real_sdk_query_protocol(
    tmp_path: Path,
) -> None:
    import anyio
    from claude_agent_sdk._internal.query import Query
    from claude_agent_sdk._internal.transport import Transport

    class MemoryTransport(Transport):
        def __init__(self) -> None:
            self.in_send, self.in_receive = anyio.create_memory_object_stream[
                dict[str, Any]
            ](10)
            self.out_send, self.out_receive = anyio.create_memory_object_stream[
                dict[str, Any]
            ](10)
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
                        "response": {
                            "subtype": "success",
                            "request_id": request_id,
                            "response": {"commands": []},
                        },
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
    backend = ClaudeBackend(
        model="opus",
        audit_root=root,
        audit_outward_symlinks=frozenset({root / "escape"}),
    )
    guard = backend._audit_root_guard  # noqa: SLF001 - pinned SDK adapter seam
    assert guard is not None
    transport = MemoryTransport()
    await transport.connect()
    query = Query(
        transport=transport,
        is_streaming_mode=True,
        hooks={"PreToolUse": [{"matcher": ".*", "hooks": [guard]}]},
    )
    await query.start()
    try:
        await query.initialize()
        initialize = await transport.out_receive.receive()
        hook_config = initialize["request"]["hooks"]["PreToolUse"][0]
        callback_id = hook_config["hookCallbackIds"][0]
        assert callback_id in query.hook_callbacks

        for request_id, payload, expected in (
            (
                "allow-1",
                {"tool_name": "Read", "tool_input": {"file_path": "inside.py"}},
                None,
            ),
            (
                "deny-1",
                {"tool_name": "Bash", "tool_input": {"command": "cat /etc/passwd"}},
                "deny",
            ),
            (
                "deny-unhashable-list",
                {
                    "tool_name": "Grep",
                    "tool_input": {
                        "pattern": "ok",
                        "path": "inside.py",
                        "output_mode": [],
                    },
                },
                "deny",
            ),
            (
                "deny-unhashable-object",
                {
                    "tool_name": "Grep",
                    "tool_input": {
                        "pattern": "ok",
                        "path": "inside.py",
                        "output_mode": {},
                    },
                },
                "deny",
            ),
            (
                "deny-inward-symlink-glob",
                {
                    "tool_name": "Glob",
                    "tool_input": {"path": "sub", "pattern": "loop/escape/*"},
                },
                "deny",
            ),
        ):
            await transport.in_send.send(
                {
                    "type": "control_request",
                    "request_id": request_id,
                    "request": {
                        "subtype": "hook_callback",
                        "callback_id": callback_id,
                        "input": payload,
                        "tool_use_id": "tool-1",
                    },
                }
            )
            response = await transport.out_receive.receive()
            assert response["response"]["subtype"] == "success"
            hook_output = response["response"]["response"]
            decision = hook_output.get("hookSpecificOutput", {}).get(
                "permissionDecision"
            )
            assert decision == expected
    finally:
        await query.close()
        query.close_receive_stream()
        await transport.close()


@pytest.mark.asyncio
async def test_audit_options_reach_real_sdk_subprocess_transport(
    patch_sdk: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import anyio
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    root = tmp_path / "audit"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus", audit_root=root)
    async for _ in backend.execute(root, "audit", read_only=True):
        pass
    options = captured["options"]

    for variable in (
        "PWD",
        "OLDPWD",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_PREFIX",
    ):
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
    assert str(source) not in {
        child_env[key]
        for key in (
            "PWD",
            "OLDPWD",
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_COMMON_DIR",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CEILING_DIRECTORIES",
            "GIT_PREFIX",
        )
    }


# ---------------------------------------------------------------------------
# fanout_concurrency
# ---------------------------------------------------------------------------


def test_claude_fanout_concurrency_defaults_to_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_FANOUT_CONCURRENCY", raising=False)
    assert effective_fanout_concurrency(10, ClaudeBackend(model="opus")) == 8


def test_claude_fanout_concurrency_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", "3")
    assert effective_fanout_concurrency(10, ClaudeBackend(model="opus")) == 3


def test_claude_fanout_concurrency_never_exceeds_workflow_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", "8")
    assert effective_fanout_concurrency(2, ClaudeBackend(model="opus")) == 2


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("6", 6),
        ("0", 8),
        ("-1", 8),
        ("notanint", 8),
        ("", 8),
    ],
)
def test_claude_fanout_concurrency_env_validation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_value: Any,
    expected: int,
) -> None:
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", env_value)
    assert effective_fanout_concurrency(10, ClaudeBackend(model="opus")) == expected
    if expected == 8:
        assert "using default 8" in caplog.text


# --- Session continuation via SDK resume --------------------------------------


@pytest.mark.asyncio
async def test_continuation_token_sets_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """A claude-minted token becomes ``options.resume`` on the next call."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(
        monkeypatch,
        scripted_client(
            [MockResultMessage(total_cost_usd=0.01, session_id="sess-123")],
            captured=captured,
        ),
    )
    backend = ClaudeBackend(model="opus")
    token = ContinuationToken(backend="claude", data={"session_id": "sess-123"})

    async for _ in backend.execute(Path("/tmp"), "p", continuation=token):
        pass

    assert captured["options"].resume == "sess-123"


@pytest.mark.asyncio
async def test_result_event_mints_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The terminal ResultMessage's session_id is minted into the ResultEvent."""
    patch_claude_sdk(
        monkeypatch,
        scripted_client([MockResultMessage(total_cost_usd=0.01, session_id="sess-9")]),
    )
    backend = ClaudeBackend(model="opus")

    results = [
        e
        async for e in backend.execute(Path("/tmp"), "p")
        if isinstance(e, ResultEvent)
    ]

    assert len(results) == 1
    assert results[0].continuation is not None
    assert results[0].continuation.backend == "claude"
    assert results[0].continuation.data["session_id"] == "sess-9"


@pytest.mark.asyncio
async def test_no_token_without_persist_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_session=False disables SDK session persistence and suppresses the token."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(
        monkeypatch,
        scripted_client([MockResultMessage(total_cost_usd=0.01, session_id="sess-9")], captured=captured),
    )
    backend = ClaudeBackend(model="opus")

    results = [
        e
        async for e in backend.execute(Path("/tmp"), "p", persist_session=False)
        if isinstance(e, ResultEvent)
    ]

    assert results[0].continuation is None
    assert captured["options"].extra_args == {"no-session-persistence": None}


@pytest.mark.asyncio
async def test_foreign_backend_token_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token minted by another backend starts cold instead of raising."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(
        monkeypatch,
        scripted_client(
            [MockResultMessage(total_cost_usd=0.01, session_id="sess-1")],
            captured=captured,
        ),
    )
    backend = ClaudeBackend(model="opus")
    token = ContinuationToken(backend="codex", data={"thread_id": "th-1"})

    async for _ in backend.execute(Path("/tmp"), "p", continuation=token):
        pass

    assert captured["options"].resume is None


@pytest.mark.asyncio
async def test_no_session_id_mints_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ResultMessage without a session id yields continuation=None."""
    patch_claude_sdk(
        monkeypatch,
        scripted_client([MockResultMessage(total_cost_usd=0.01, session_id=None)]),
    )
    backend = ClaudeBackend(model="opus")

    results = [
        e
        async for e in backend.execute(Path("/tmp"), "p")
        if isinstance(e, ResultEvent)
    ]

    assert results[0].continuation is None


@pytest.mark.asyncio
async def test_execute_disables_cli_background_tasks_and_lifts_bash_ceiling(patch_sdk: Any) -> None:
    """The CLI subprocess env switches background tasks off and raises the Bash timeout ceiling.

    daydream reads a turn's final text as the phase result and stops consuming
    the session when the turn ends, at which point the CLI kills its background
    tasks -- so a backgrounded ``make test`` can never report. The ceiling is
    lifted to the host's largest per-turn wall budget so a slow suite has no
    reason to be backgrounded in the first place.
    """
    from daydream.config import TEST_WALL_BUDGET_S

    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")

    async for _ in backend.execute(Path("/tmp"), "Go"):
        pass

    env = captured["options"].env
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    expected_ms = str(int(TEST_WALL_BUDGET_S * 1000))
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == expected_ms
    assert env["BASH_MAX_TIMEOUT_MS"] == expected_ms


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [False, True])
async def test_execute_registers_background_bash_guard(patch_sdk: Any, read_only: bool) -> None:
    """Every execute() wires an always-on PreToolUse guard denying ``Bash(run_in_background=True)``.

    Drive the callbacks actually registered on the production options, in both
    the default and the read-only profile: a backgrounded read-only command is
    denied with a reason that tells the agent to rerun in the foreground; the
    same command in the foreground is allowed; a non-Bash tool carrying the key
    is not the guard's business.
    """
    captured: dict[str, Any] = {}
    patch_sdk(_capturing_client(captured))
    backend = ClaudeBackend(model="opus")

    async for _ in backend.execute(Path("/tmp"), "Go", read_only=read_only):
        pass

    matcher = captured["options"].hooks["PreToolUse"][0]

    async def decide(payload: Any) -> Any:
        for hook in matcher.hooks:
            out = await hook(payload, None, {})
            if out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny":
                return out
        return {}

    deny_bg = await decide(
        {"tool_name": "Bash", "tool_input": {"command": "git status", "run_in_background": True}}
    )
    hook_out = deny_bg["hookSpecificOutput"]
    assert hook_out["permissionDecision"] == "deny"
    assert "background Bash blocked" in hook_out["permissionDecisionReason"]
    assert "foreground" in hook_out["permissionDecisionReason"]

    allow_fg = await decide(
        {"tool_name": "Bash", "tool_input": {"command": "git status", "run_in_background": False}}
    )
    assert "hookSpecificOutput" not in allow_fg
    allow_no_key = await decide({"tool_name": "Bash", "tool_input": {"command": "git status"}})
    assert "hookSpecificOutput" not in allow_no_key

    other_tool = await decide(
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
        ({"tool_name": "Bash", "tool_input": "not-a-dict"}, False),
        ("garbage", False),
        (None, False),
    ],
)
def test_is_background_bash(payload: Any, background: bool) -> None:
    """Only a Bash payload with a truthy ``run_in_background`` is a background call."""
    from daydream.backends.claude import _is_background_bash

    assert _is_background_bash(payload) is background


# --- P18 Task 1: effective request-config admission at the Claude SDK seam ---


@pytest.mark.asyncio
async def test_request_event_carries_typed_config_from_exact_sdk_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RequestEvent.config mirrors the options the SDK client actually received."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(monkeypatch, _capturing_client(captured))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "capture config"):
        events.append(event)

    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, ClaudeRequestConfig)
    # Common subset: max_turns not passed -> None (never an effective claim).
    assert config.max_turns is None
    assert config.read_only is False
    assert config.persist_session is True
    assert config.continuation_mode == "fresh"
    # Claude main-model-only surface is single-model-capable.
    assert config.model_mode == "single"
    # Backend-specific: exact implemented facts (never guessed defaults).
    assert config.permission_mode == "bypassPermissions"
    assert config.allowed_tools_count == 6
    assert config.allowed_tools_present is True
    assert config.audit_tools_count is None
    assert config.audit_tools_present is None
    assert config.setting_sources_present is True
    assert config.native_output_format is False
    assert config.buffer_limit_bytes == 10 * 1024 * 1024
    assert config.hooks_enabled is True
    # Provenance: configured model, no claimed provider, host-observed stamp.
    assert request.model_name == "opus"
    assert request.model_source == "configured"
    assert request.provider_name is None
    assert request.provider_source is None
    assert request.timestamp_source == "host_observed"


@pytest.mark.asyncio
async def test_request_event_requires_multi_or_dynamic_for_nonempty_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonempty agents mapping makes the aggregate multi-model-capable."""
    events = await _drive_with_agents(monkeypatch)

    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, ClaudeRequestConfig)
    assert config.model_mode == "multi_or_dynamic"
    # Agent definitions are never inspected or exported: no prompt/model of
    # any specialist leaks into the admitted config.
    assert not hasattr(config, "agents")
    assert not hasattr(config, "agent_definitions")
    # The exact main-model Daydream provenance is preserved.
    assert request.model_name == "opus"
    assert request.model_source == "configured"


async def _drive_with_agents(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Drive execute with a nonempty agents mapping and collect the events."""
    from claude_agent_sdk.types import AgentDefinition

    captured: dict[str, Any] = {}
    patch_claude_sdk(monkeypatch, _capturing_client(captured))
    backend = ClaudeBackend(model="opus")
    specialists: dict[str, AgentDefinition] = {
        "pattern-scanner": AgentDefinition(
            description="Scan patterns", prompt="Scan", model="sonnet",
        ),
    }
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "fan out", agents=specialists):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_request_event_read_only_and_max_turns_are_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actually-passed max_turns/read_only appear; accepted-but-ignored do not exist."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(monkeypatch, _capturing_client(captured))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "go", max_turns=7, read_only=True):
        events.append(event)

    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, ClaudeRequestConfig)
    assert config.max_turns == 7  # passed -> admitted (never interpreted as max tokens)
    assert config.read_only is True
    assert config.allowed_tools_count == 6  # allowed_tools unchanged by read_only
    assert config.hooks_enabled is True


@pytest.mark.asyncio
async def test_request_event_resume_provenance_is_host_generated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed claude session is continuation_mode=resume with host provenance."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(monkeypatch, _capturing_client(captured))
    backend = ClaudeBackend(model="opus")
    token = ContinuationToken(backend="claude", data={"session_id": "sess-42"})
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "again", continuation=token):
        events.append(event)

    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, ClaudeRequestConfig)
    assert config.continuation_mode == "resume"
    assert request.session_id == "sess-42"
    assert request.session_source == "host_generated"


@pytest.mark.asyncio
async def test_request_event_output_schema_sets_native_output_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema request admits native_output_format=True at the SDK options."""
    captured: dict[str, Any] = {}
    patch_claude_sdk(monkeypatch, _capturing_client(captured))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "structured", output_schema={"type": "object"}):
        events.append(event)

    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, ClaudeRequestConfig)
    assert config.native_output_format is True
    assert request.output_schema == {"type": "object"}

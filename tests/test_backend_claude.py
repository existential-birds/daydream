# tests/test_backend_claude.py
"""Tests for ClaudeBackend."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.claude import ClaudeAgentError, ClaudeBackend

# Mock SDK types (same pattern as test_integration.py)


@dataclass
class MockTextBlock:
    text: str


@dataclass
class MockToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] | None = None


@dataclass
class MockToolResultBlock:
    tool_use_id: str
    content: str | None = None
    is_error: bool = False


@dataclass
class MockThinkingBlock:
    thinking: str


@dataclass
class MockAssistantMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class MockUserMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class MockResultMessage:
    total_cost_usd: float | None = 0.001
    structured_output: Any = None
    is_error: bool = False
    result: str | None = None
    subtype: str = "success"


@pytest.fixture
def patch_sdk(monkeypatch):
    """Return a function that patches the SDK imports in claude.py."""
    def _patch(client_class):
        monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", client_class)
        monkeypatch.setattr("daydream.backends.claude.AssistantMessage", MockAssistantMessage)
        monkeypatch.setattr("daydream.backends.claude.UserMessage", MockUserMessage)
        monkeypatch.setattr("daydream.backends.claude.ResultMessage", MockResultMessage)
        monkeypatch.setattr("daydream.backends.claude.TextBlock", MockTextBlock)
        monkeypatch.setattr("daydream.backends.claude.ThinkingBlock", MockThinkingBlock)
        monkeypatch.setattr("daydream.backends.claude.ToolUseBlock", MockToolUseBlock)
        monkeypatch.setattr("daydream.backends.claude.ToolResultBlock", MockToolResultBlock)
    return _patch


def _scripted_client(messages: list[Any]) -> type:
    """Build a one-off ClaudeSDKClient stand-in yielding *messages* verbatim."""

    class _ScriptedClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self._prompt: str = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            self._prompt = prompt

        async def receive_response(self):
            for m in messages:
                yield m

    return _ScriptedClient


async def _drive_claude_backend_to_list(
    *,
    messages: list[Any],
    patch_sdk_fn: Any,
    output_schema: Any = None,
    prompt: str = "go",
) -> list[Any]:
    """Drive ClaudeBackend.execute with a scripted SDK message sequence."""
    patch_sdk_fn(_scripted_client(messages))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), prompt, output_schema=output_schema):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_execute_yields_text_and_result(patch_sdk):
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
async def test_execute_yields_tool_events(patch_sdk):
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
async def test_execute_structured_output(patch_sdk):
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
async def test_error_result_raises_instead_of_clean_empty_result(patch_sdk):
    """An is_error ResultMessage must raise, never yield a normal ResultEvent.

    Regression guard for the sandbox acceptance failure: an invalid API key
    run streamed the error text and a ResultMessage(is_error=True), and the
    backend yielded a clean ResultEvent — the review then exited 0 with
    "no issues found" despite the agent never running.
    """
    patch_sdk(
        _scripted_client(
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
    # The error text still streamed as agent text, but no ResultEvent escaped.
    assert not [e for e in events if isinstance(e, ResultEvent)]


@pytest.mark.asyncio
async def test_max_turns_result_raises_typed_error(patch_sdk):
    """error_max_turns must raise the typed MaxTurnsError carrying the subtype.

    A generic ClaudeAgentError left callers (and the trajectory) unable to
    distinguish a turn-cap failure from a real backend error. Mirrors the real
    SDK shape: a ResultMessage with ``is_error=True`` and
    ``subtype="error_max_turns"`` (``result`` is None, so the detail falls back
    to the subtype).
    """
    from daydream.backends.claude import MaxTurnsError

    patch_sdk(
        _scripted_client(
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


@pytest.mark.parametrize(
    ("skill_key", "args", "expected"),
    [
        ("beagle-python:review-python", "", "/beagle-python:review-python"),
        ("beagle-core:fetch-pr-feedback", "--pr 42 --bot mybot", "/beagle-core:fetch-pr-feedback --pr 42 --bot mybot"),
    ],
    ids=["full-key", "with-args"],
)
def test_format_skill_invocation(skill_key, args, expected):
    backend = ClaudeBackend(model="fixture-model")
    result = backend.format_skill_invocation(skill_key, args) if args else backend.format_skill_invocation(skill_key)
    assert result == expected


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
    ("", False),                          # empty → fail closed
])
def test_read_only_bash_guard_decision(cmd, allowed):
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
def test_is_dangerous_command(cmd, dangerous):
    """The always-on dangerous-command predicate flags root-scans and catastrophic deletes."""
    from daydream.backends.claude import _is_dangerous_command

    assert _is_dangerous_command(cmd) is dangerous


@pytest.mark.asyncio
async def test_read_only_execute_registers_pretooluse_guard(patch_sdk):
    """read_only=True wires a fail-closed PreToolUse guard onto the SDK options.

    The contract is behavioral, not a matcher-string shape: under
    ``bypassPermissions`` the hook is the *only* enforcement, so the guard must
    fire for every tool and deny-by-default. We assert that by driving the
    callback that was actually registered on the options — denying Write and
    mutating Bash, allowing inspection (read-only Bash + allowlisted tools), and
    denying an unknown/future tool (the fail-closed property a narrow matcher
    would silently lose).
    """
    patch_sdk(MockClaudeSDKClientCapture)
    backend = ClaudeBackend(model="opus")

    async for _ in backend.execute(Path("/tmp"), "Go", read_only=True):
        pass

    opts = MockClaudeSDKClientCapture.captured_options
    assert opts is not None
    hooks = opts.hooks
    assert hooks is not None and "PreToolUse" in hooks
    matchers = hooks["PreToolUse"]
    assert len(matchers) == 1
    matcher = matchers[0]
    assert matcher.hooks  # callbacks registered

    async def decide(payload):
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
async def test_non_read_only_execute_registers_dangerous_command_hook(patch_sdk):
    """read_only=False (default) still wires the always-on dangerous-command guard.

    The guard is registered unconditionally (all phases). Drive the callback that
    was actually built onto the production options: a ``find /`` root scan denies,
    a scoped ``find core/...`` allows.
    """
    patch_sdk(MockClaudeSDKClientCapture)
    backend = ClaudeBackend(model="opus")

    async for _ in backend.execute(Path("/tmp"), "Go", read_only=False):
        pass

    opts = MockClaudeSDKClientCapture.captured_options
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
async def test_read_only_guard_denies_mutation_allows_inspection():
    """The registered guard callback denies Write and non-read-only Bash, allows read-only Bash."""
    from daydream.backends.claude import _read_only_guard

    deny_write = await _read_only_guard(
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, None, {},
    )
    assert deny_write["hookSpecificOutput"]["permissionDecision"] == "deny"

    deny_bash = await _read_only_guard(
        {"tool_name": "Bash", "tool_input": {"command": "git commit -m x"}}, None, {},
    )
    assert deny_bash["hookSpecificOutput"]["permissionDecision"] == "deny"

    allow_bash = await _read_only_guard(
        {"tool_name": "Bash", "tool_input": {"command": "git log -n 5"}}, None, {},
    )
    assert "hookSpecificOutput" not in allow_bash

    # Malformed input → fail closed (deny)
    deny_malformed = await _read_only_guard({"tool_name": "Bash"}, None, {})
    assert deny_malformed["hookSpecificOutput"]["permissionDecision"] == "deny"


class MockClaudeSDKClientCapture:
    """Mock client that captures the options it was constructed with."""

    captured_options = None

    def __init__(self, options: Any = None):
        MockClaudeSDKClientCapture.captured_options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[MockTextBlock(text="OK")])
        yield MockResultMessage(total_cost_usd=0.01)


@pytest.mark.asyncio
async def test_execute_passes_agents_dict_to_options(patch_sdk):
    """Agents dict must reach ClaudeAgentOptions with original keys preserved verbatim."""
    from claude_agent_sdk.types import AgentDefinition

    patch_sdk(MockClaudeSDKClientCapture)
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

    opts = MockClaudeSDKClientCapture.captured_options
    assert opts is not None
    assert opts.agents == {
        "pattern-scanner": pattern_scanner,
        "dependency-tracer": dependency_tracer,
    }
    assert "explorer-0" not in opts.agents
    assert "explorer-1" not in opts.agents


@pytest.mark.asyncio
async def test_execute_passes_none_when_no_agents(patch_sdk):
    """When agents=None, ClaudeAgentOptions should not carry an agents dict."""
    patch_sdk(MockClaudeSDKClientCapture)
    backend = ClaudeBackend(model="opus")

    events = []
    async for event in backend.execute(Path("/tmp"), "Go"):
        events.append(event)

    opts = MockClaudeSDKClientCapture.captured_options
    assert opts is not None
    agents_val = getattr(opts, "agents", None)
    assert agents_val is None


def test_backend_protocol_agents_param_is_dict_typed():
    """The Backend protocol's execute.agents annotation must be dict[str, AgentDefinition]."""
    from daydream.backends import Backend

    annotations = Backend.execute.__annotations__
    assert "agents" in annotations
    annotation = annotations["agents"]
    # Annotation may be a string (from __future__ annotations) or a real type
    annotation_str = annotation if isinstance(annotation, str) else repr(annotation)
    assert "dict[str, AgentDefinition]" in annotation_str


# Helpers for TurnEndEvent tests (Task 6)


def _assistant_message(*, text: str, message_id: str) -> MockAssistantMessage:
    """Build a MockAssistantMessage carrying one TextBlock + a message_id."""
    msg = MockAssistantMessage(content=[MockTextBlock(text=text)])
    msg.message_id = message_id  # type: ignore[attr-defined]
    return msg


def _result_message(*, cost: float | None = 0.0) -> MockResultMessage:
    return MockResultMessage(total_cost_usd=cost, structured_output=None)


@pytest.mark.asyncio
async def test_structured_output_tool_result_is_suppressed(patch_sdk) -> None:
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
async def test_claude_backend_emits_turn_end_per_assistant_message(patch_sdk) -> None:
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


# Skill guard tests


@pytest.mark.parametrize("prompt,expected", [
    (
        "/beagle-python:review-python\nWrite the full review output to .review-output.md",
        {"beagle-python:review-python"},
    ),
    ("/beagle-core:fetch-pr-feedback 42 --bot x", {"beagle-core:fetch-pr-feedback"}),
    ("Read the diff file at /Users/ka/proj/.daydream/diff.patch", set()),
    ("compare /path/to/base against /path/to/head", set()),
    ("Analyze the staged diff and report intent.", set()),
])
def test_prompt_skill_keys(prompt, expected):
    """_prompt_skill_keys extracts anchored /{key} invocations, never filesystem paths."""
    from daydream.backends.claude import _prompt_skill_keys

    assert _prompt_skill_keys(prompt) == frozenset(expected)


@pytest.mark.asyncio
async def test_skill_guard_allows_non_skill_tool_and_prompt_invoked_skill():
    """The skill guard passes non-Skill tools untouched and allows prompt-invoked keys."""
    from daydream.backends.claude import _make_skill_guard

    guard = _make_skill_guard(frozenset({"beagle-python:review-python"}))

    allow_bash = await guard({"tool_name": "Bash", "tool_input": {"command": "ls"}}, None, {})
    assert "hookSpecificOutput" not in allow_bash

    allow_invoked = await guard(
        {"tool_name": "Skill", "tool_input": {"skill": "beagle-python:review-python"}}, None, {},
    )
    assert "hookSpecificOutput" not in allow_invoked


@pytest.mark.asyncio
async def test_skill_guard_allows_unreferenced_beagle_namespaced_skill():
    """Beagle skills chain, so a beagle-* key stays invocable even with an empty allowed set."""
    from daydream.backends.claude import _make_skill_guard

    guard = _make_skill_guard(frozenset())

    allow_chained = await guard(
        {"tool_name": "Skill", "tool_input": {"skill": "beagle-core:review-verification-protocol"}},
        None,
        {},
    )
    assert "hookSpecificOutput" not in allow_chained


@pytest.mark.asyncio
async def test_skill_guard_denies_builtin_review():
    """Claude Code's built-in /review is denied when the prompt invoked no skill."""
    from daydream.backends.claude import _make_skill_guard

    guard = _make_skill_guard(frozenset())

    deny = await guard({"tool_name": "Skill", "tool_input": {"skill": "review"}}, None, {})
    assert deny["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "review" in deny["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.asyncio
async def test_skill_guard_fails_closed_on_malformed_input():
    """Missing tool_input or a non-string skill name denies (fail closed)."""
    from daydream.backends.claude import _make_skill_guard

    guard = _make_skill_guard(frozenset({"beagle-python:review-python"}))

    deny_no_input = await guard({"tool_name": "Skill"}, None, {})
    assert deny_no_input["hookSpecificOutput"]["permissionDecision"] == "deny"

    deny_non_string = await guard({"tool_name": "Skill", "tool_input": {"skill": 42}}, None, {})
    assert deny_non_string["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_execute_registers_skill_guard(patch_sdk):
    """execute() wires the always-on skill guard scoped to the prompt's invocations.

    With a prompt that invokes no skill, the PreToolUse hooks registered on the
    production options must deny Claude Code's built-in ``review`` skill while
    still allowing a beagle-namespaced key (skills chain within the plugin).
    """
    patch_sdk(MockClaudeSDKClientCapture)
    backend = ClaudeBackend(model="opus")

    async for _ in backend.execute(Path("/tmp"), "Analyze the staged diff and report intent."):
        pass

    opts = MockClaudeSDKClientCapture.captured_options
    assert opts is not None
    hooks = opts.hooks
    assert hooks is not None and "PreToolUse" in hooks
    matchers = hooks["PreToolUse"]
    assert len(matchers) == 1
    matcher = matchers[0]
    assert matcher.hooks  # callbacks registered

    async def decide(payload):
        # Mirror the SDK: run every registered hook; first deny wins.
        for hook in matcher.hooks:
            out = await hook(payload, None, {})
            if out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny":
                return out
        return {}

    deny_builtin = await decide({"tool_name": "Skill", "tool_input": {"skill": "review"}})
    assert deny_builtin["hookSpecificOutput"]["permissionDecision"] == "deny"

    allow_beagle = await decide(
        {"tool_name": "Skill", "tool_input": {"skill": "beagle-core:review-verification-protocol"}}
    )
    assert "hookSpecificOutput" not in allow_beagle



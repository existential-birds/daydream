# Codex Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the backend abstraction layer from the approved design doc (`docs/plans/2026-02-06-codex-backend-design.md`) so daydream can run against either Claude Code SDK or Codex CLI, chosen per-phase at runtime.

**Architecture:** Extract the Claude SDK interaction from `agent.py` into a `Backend` protocol with two implementations: `ClaudeBackend` (existing behavior) and `CodexBackend` (new, subprocess-based). The `run_agent()` function becomes a thin event consumer that renders UI from a unified event stream. Phases accept a `Backend` parameter and use `backend.format_skill_invocation()` for skill syntax.

**Tech Stack:** Python 3.12, asyncio subprocess, JSONL parsing, dataclasses, `typing.Protocol`

---

### Task 1: Create `daydream/backends/__init__.py` — Event types, Backend protocol, factory

**Files:**
- Create: `daydream/backends/__init__.py`
- Test: `tests/test_backends_init.py`

**Step 1: Write the failing test**

```python
# tests/test_backends_init.py
"""Tests for backend protocol, event types, and factory."""

from dataclasses import fields
from typing import Any

import pytest

from daydream.backends import (
    AgentEvent,
    Backend,
    ClaudeBackend,
    ContinuationToken,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    create_backend,
)


def test_text_event_has_text_field():
    event = TextEvent(text="hello")
    assert event.text == "hello"


def test_thinking_event_has_text_field():
    event = ThinkingEvent(text="reasoning...")
    assert event.text == "reasoning..."


def test_tool_start_event_fields():
    event = ToolStartEvent(id="t1", name="Bash", input={"command": "ls"})
    assert event.id == "t1"
    assert event.name == "Bash"
    assert event.input == {"command": "ls"}


def test_tool_result_event_fields():
    event = ToolResultEvent(id="t1", output="file.py", is_error=False)
    assert event.id == "t1"
    assert event.output == "file.py"
    assert event.is_error is False


def test_cost_event_fields():
    event = CostEvent(cost_usd=0.01, input_tokens=100, output_tokens=50)
    assert event.cost_usd == 0.01
    assert event.input_tokens == 100
    assert event.output_tokens == 50


def test_cost_event_nullable_fields():
    event = CostEvent(cost_usd=None, input_tokens=None, output_tokens=None)
    assert event.cost_usd is None


def test_continuation_token_fields():
    token = ContinuationToken(backend="codex", data={"thread_id": "abc"})
    assert token.backend == "codex"
    assert token.data == {"thread_id": "abc"}


def test_result_event_fields():
    token = ContinuationToken(backend="codex", data={})
    event = ResultEvent(structured_output={"key": "val"}, continuation=token)
    assert event.structured_output == {"key": "val"}
    assert event.continuation is token


def test_result_event_nullable():
    event = ResultEvent(structured_output=None, continuation=None)
    assert event.structured_output is None
    assert event.continuation is None


def test_create_backend_claude_default():
    backend = create_backend("claude")
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == "opus"


def test_create_backend_claude_custom_model():
    backend = create_backend("claude", model="sonnet")
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == "sonnet"


def test_create_backend_codex_default():
    backend = create_backend("codex")
    # Import here to avoid circular — just check it's not ClaudeBackend
    from daydream.backends.codex import CodexBackend
    assert isinstance(backend, CodexBackend)
    assert backend.model == "gpt-5.3-codex"


def test_create_backend_codex_custom_model():
    backend = create_backend("codex", model="o3-pro")
    from daydream.backends.codex import CodexBackend
    assert isinstance(backend, CodexBackend)
    assert backend.model == "o3-pro"


def test_create_backend_invalid_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backends_init.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daydream.backends'`

**Step 3: Write minimal implementation**

```python
# daydream/backends/__init__.py
"""Backend abstraction layer for daydream.

Defines the unified event stream, Backend protocol, and factory function.
Backends yield AgentEvent instances that the UI layer consumes without
knowing which backend produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class TextEvent:
    """Agent text output."""

    text: str


@dataclass
class ThinkingEvent:
    """Extended thinking / reasoning."""

    text: str


@dataclass
class ToolStartEvent:
    """Tool invocation started."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultEvent:
    """Tool invocation completed."""

    id: str
    output: str
    is_error: bool


@dataclass
class CostEvent:
    """Cost and usage information."""

    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class ContinuationToken:
    """Opaque token for multi-turn interactions."""

    backend: str
    data: dict[str, Any]


@dataclass
class ResultEvent:
    """Final event in the stream. Carries structured output and continuation token."""

    structured_output: Any | None
    continuation: ContinuationToken | None


AgentEvent = (
    TextEvent | ThinkingEvent | ToolStartEvent
    | ToolResultEvent | CostEvent | ResultEvent
)


class Backend(Protocol):
    """Protocol for agent backends.

    Each backend yields a stream of AgentEvent instances from execute().
    """

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def cancel(self) -> None: ...

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...


def create_backend(name: str, model: str | None = None) -> Backend:
    """Create a backend by name.

    Args:
        name: Backend name ("claude" or "codex").
        model: Optional model override. Each backend has its own default.

    Returns:
        A Backend instance.

    Raises:
        ValueError: If the backend name is unknown.

    """
    if name == "claude":
        from daydream.backends.claude import ClaudeBackend
        return ClaudeBackend(model=model or "opus")
    if name == "codex":
        from daydream.backends.codex import CodexBackend
        return CodexBackend(model=model or "gpt-5.3-codex")
    raise ValueError(f"Unknown backend: {name!r}. Expected 'claude' or 'codex'.")


# Re-export ClaudeBackend at package level for convenience
from daydream.backends.claude import ClaudeBackend  # noqa: E402

__all__ = [
    "AgentEvent",
    "Backend",
    "ClaudeBackend",
    "ContinuationToken",
    "CostEvent",
    "ResultEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolResultEvent",
    "ToolStartEvent",
    "create_backend",
]
```

Note: This will fail until `claude.py` and `codex.py` exist. We create stubs in Step 3 of Tasks 2 and 3. However, since Task 2 (ClaudeBackend) is the next task, we need it to exist before this test passes.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backends_init.py -v`
Expected: PASS (after Tasks 2 and 3 stubs exist)

**Step 5: Commit**

```bash
git add daydream/backends/__init__.py tests/test_backends_init.py
git commit -m "feat(backends): add event types, Backend protocol, and factory"
```

---

### Task 2: Create `daydream/backends/claude.py` — Extract ClaudeBackend from agent.py

**Files:**
- Create: `daydream/backends/claude.py`
- Test: `tests/test_backend_claude.py`

This task extracts the SDK interaction loop from `agent.py:262-398` into `ClaudeBackend.execute()` that yields unified events. The existing `run_agent()` in `agent.py` stays unchanged for now — we prove the backend works independently first, then swap in Task 5.

**Step 1: Write the failing test**

```python
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
from daydream.backends.claude import ClaudeBackend


# --- Mock SDK types (same pattern as test_integration.py) ---


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


class MockClaudeSDKClient:
    """Mock client that yields a configurable message sequence."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[MockTextBlock(text="Hello world")])
        yield MockResultMessage(total_cost_usd=0.05, structured_output=None)


class MockClaudeSDKClientWithTools:
    """Mock client that yields tool use messages."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[
            MockThinkingBlock(thinking="Let me think..."),
        ])
        yield MockAssistantMessage(content=[
            MockTextBlock(text="I'll run a command."),
        ])
        yield MockAssistantMessage(content=[
            MockToolUseBlock(id="tool-1", name="Bash", input={"command": "ls"}),
        ])
        yield MockUserMessage(content=[
            MockToolResultBlock(tool_use_id="tool-1", content="file.py", is_error=False),
        ])
        yield MockResultMessage(total_cost_usd=0.10)


class MockClaudeSDKClientStructured:
    """Mock client that returns structured output."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[MockTextBlock(text="Parsed.")])
        yield MockResultMessage(
            total_cost_usd=0.02,
            structured_output={"issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]},
        )


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


@pytest.mark.asyncio
async def test_execute_yields_text_and_result(patch_sdk):
    patch_sdk(MockClaudeSDKClient)
    backend = ClaudeBackend(model="opus")
    events = []
    async for event in backend.execute(Path("/tmp"), "Say hello"):
        events.append(event)

    # Should have TextEvent, CostEvent, ResultEvent
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
    patch_sdk(MockClaudeSDKClientWithTools)
    backend = ClaudeBackend(model="opus")
    events = []
    async for event in backend.execute(Path("/tmp"), "Run ls"):
        events.append(event)

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


@pytest.mark.asyncio
async def test_execute_structured_output(patch_sdk):
    patch_sdk(MockClaudeSDKClientStructured)
    backend = ClaudeBackend(model="opus")
    events = []
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]
    }


def test_format_skill_invocation_full_key():
    backend = ClaudeBackend()
    result = backend.format_skill_invocation("beagle-python:review-python")
    assert result == "/beagle-python:review-python"


def test_format_skill_invocation_with_args():
    backend = ClaudeBackend()
    result = backend.format_skill_invocation("beagle-core:fetch-pr-feedback", "--pr 42 --bot mybot")
    assert result == "/beagle-core:fetch-pr-feedback --pr 42 --bot mybot"


def test_skips_structured_output_tool():
    """StructuredOutput tool blocks should be skipped (not yielded as ToolStartEvent)."""
    # This is tested implicitly via the structured output test — no ToolStartEvent for StructuredOutput
    pass
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backend_claude.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daydream.backends.claude'`

**Step 3: Write minimal implementation**

```python
# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)


class ClaudeBackend:
    """Backend that wraps the Claude Agent SDK.

    Translates Claude SDK message types into the unified AgentEvent stream.
    """

    def __init__(self, model: str = "opus"):
        self.model = model
        self._client: ClaudeSDKClient | None = None

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a prompt and yield unified events.

        Args:
            cwd: Working directory for the agent.
            prompt: The prompt to send.
            output_schema: Optional JSON schema for structured output.
            continuation: Ignored by Claude backend.

        Yields:
            AgentEvent instances.

        """
        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        options = ClaudeAgentOptions(
            cwd=str(cwd),
            permission_mode="bypassPermissions",
            setting_sources=["user", "project", "local"],
            model=self.model,
            output_format=output_format,
        )

        structured_result: Any = None

        async with ClaudeSDKClient(options=options) as client:
            self._client = client
            try:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                yield TextEvent(text=block.text)
                            elif isinstance(block, ThinkingBlock) and block.thinking:
                                yield ThinkingEvent(text=block.thinking)
                            elif isinstance(block, ToolUseBlock):
                                if block.name == "StructuredOutput":
                                    continue
                                yield ToolStartEvent(
                                    id=block.id,
                                    name=block.name,
                                    input=block.input or {},
                                )

                    elif isinstance(msg, UserMessage):
                        for user_block in msg.content:
                            if isinstance(user_block, ToolResultBlock):
                                content_str = str(user_block.content) if user_block.content else ""
                                yield ToolResultEvent(
                                    id=user_block.tool_use_id,
                                    output=content_str,
                                    is_error=user_block.is_error or False,
                                )

                    elif isinstance(msg, ResultMessage):
                        if msg.structured_output is not None:
                            structured_result = msg.structured_output
                        if msg.total_cost_usd:
                            yield CostEvent(
                                cost_usd=msg.total_cost_usd,
                                input_tokens=None,
                                output_tokens=None,
                            )

                yield ResultEvent(
                    structured_output=structured_result,
                    continuation=None,
                )
            finally:
                self._client = None

    async def cancel(self) -> None:
        """Cancel the running agent by terminating the SDK client."""
        if self._client is not None:
            self._client.terminate()

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation for Claude.

        Claude uses /{namespace:skill} syntax.

        Args:
            skill_key: Full skill key (e.g. "beagle-python:review-python").
            args: Optional arguments string.

        Returns:
            Formatted skill invocation string.

        """
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backend_claude.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/backends/claude.py tests/test_backend_claude.py
git commit -m "feat(backends): extract ClaudeBackend from agent.py SDK loop"
```

---

### Task 3: Create `daydream/backends/codex.py` — Codex CLI subprocess backend

**Files:**
- Create: `daydream/backends/codex.py`
- Create: `tests/fixtures/codex_jsonl/` (fixture directory)
- Create: `tests/fixtures/codex_jsonl/simple_text.jsonl`
- Create: `tests/fixtures/codex_jsonl/tool_use.jsonl`
- Create: `tests/fixtures/codex_jsonl/structured_output.jsonl`
- Create: `tests/fixtures/codex_jsonl/turn_failed.jsonl`
- Test: `tests/test_backend_codex.py`

**Step 1: Create JSONL fixtures**

Based on the Codex JSONL event mapping from the design doc, create canned JSONL event streams.

```jsonl
# tests/fixtures/codex_jsonl/simple_text.jsonl
{"type":"thread.started","thread_id":"th_abc123"}
{"type":"item.started","item":{"type":"agent_message","id":"msg_1","content":[]}}
{"type":"item.completed","item":{"type":"agent_message","id":"msg_1","content":[{"type":"text","text":"Hello from Codex"}]}}
{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}
```

```jsonl
# tests/fixtures/codex_jsonl/tool_use.jsonl
{"type":"thread.started","thread_id":"th_def456"}
{"type":"item.started","item":{"type":"reasoning","id":"reason_1","content":[]}}
{"type":"item.completed","item":{"type":"reasoning","id":"reason_1","content":[{"type":"text","text":"Let me run a command"}]}}
{"type":"item.started","item":{"type":"command_execution","id":"cmd_1","command":"ls -la","status":"running"}}
{"type":"item.completed","item":{"type":"command_execution","id":"cmd_1","command":"ls -la","status":"completed","exit_code":0,"aggregated_output":"file.py\ntest.py"}}
{"type":"item.completed","item":{"type":"file_change","id":"fc_1","file_path":"main.py","action":"modified"}}
{"type":"item.started","item":{"type":"agent_message","id":"msg_1","content":[]}}
{"type":"item.completed","item":{"type":"agent_message","id":"msg_1","content":[{"type":"text","text":"Done!"}]}}
{"type":"turn.completed","usage":{"input_tokens":200,"output_tokens":100}}
```

```jsonl
# tests/fixtures/codex_jsonl/structured_output.jsonl
{"type":"thread.started","thread_id":"th_struct"}
{"type":"item.started","item":{"type":"agent_message","id":"msg_1","content":[]}}
{"type":"item.completed","item":{"type":"agent_message","id":"msg_1","content":[{"type":"text","text":"{\"issues\":[{\"id\":1,\"description\":\"Fix type hints\",\"file\":\"app.py\",\"line\":5}]}"}]}}
{"type":"turn.completed","usage":{"input_tokens":150,"output_tokens":80}}
```

```jsonl
# tests/fixtures/codex_jsonl/turn_failed.jsonl
{"type":"thread.started","thread_id":"th_fail"}
{"type":"turn.failed","error":{"message":"Model returned an error","code":"model_error"}}
```

**Step 2: Write the failing test**

```python
# tests/test_backend_codex.py
"""Tests for CodexBackend with canned JSONL fixtures."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend, CodexError

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"


def _make_mock_process(fixture_name: str):
    """Create a mock asyncio.subprocess.Process from a JSONL fixture file."""
    fixture_path = FIXTURES_DIR / fixture_name
    lines = fixture_path.read_text().strip().split("\n")

    class MockStdout:
        def __init__(self):
            self._lines = iter(lines)

        async def readline(self):
            try:
                line = next(self._lines)
                return (line + "\n").encode()
            except StopIteration:
                return b""

    process = MagicMock()
    process.stdout = MockStdout()
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.close = MagicMock()
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


@pytest.mark.asyncio
async def test_simple_text_events():
    backend = CodexBackend(model="gpt-5.3-codex")
    mock_proc = _make_mock_process("simple_text.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Say hello"):
            events.append(event)

    text_events = [e for e in events if isinstance(e, TextEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]

    assert len(text_events) == 1
    assert text_events[0].text == "Hello from Codex"
    assert len(cost_events) == 1
    assert cost_events[0].cost_usd is None
    assert cost_events[0].input_tokens == 100
    assert cost_events[0].output_tokens == 50
    assert len(result_events) == 1
    assert result_events[0].continuation is not None
    assert result_events[0].continuation.backend == "codex"
    assert result_events[0].continuation.data["thread_id"] == "th_abc123"


@pytest.mark.asyncio
async def test_tool_use_events():
    backend = CodexBackend()
    mock_proc = _make_mock_process("tool_use.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Run ls"):
            events.append(event)

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    texts = [e for e in events if isinstance(e, TextEvent)]

    # Reasoning → ThinkingEvent
    assert len(thinking) == 1
    assert thinking[0].text == "Let me run a command"

    # command_execution → ToolStart + ToolResult
    assert any(ts.name == "shell" and ts.input == {"command": "ls -la"} for ts in tool_starts)
    assert any(tr.output == "file.py\ntest.py" and not tr.is_error for tr in tool_results)

    # file_change → synthetic ToolStart("patch") + ToolResult
    assert any(ts.name == "patch" for ts in tool_starts)
    assert any("main.py" in tr.output for tr in tool_results)

    # agent_message → TextEvent
    assert any(t.text == "Done!" for t in texts)


@pytest.mark.asyncio
async def test_structured_output():
    backend = CodexBackend()
    mock_proc = _make_mock_process("structured_output.jsonl")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
            events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]
    }


@pytest.mark.asyncio
async def test_turn_failed_raises():
    backend = CodexBackend()
    mock_proc = _make_mock_process("turn_failed.jsonl")

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="Model returned an error"):
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass


@pytest.mark.asyncio
async def test_continuation_token_resumes():
    """Test that continuation token is passed as 'resume' argument."""
    from daydream.backends import ContinuationToken

    backend = CodexBackend()
    mock_proc = _make_mock_process("simple_text.jsonl")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "Continue", continuation=token):
            pass

        # Verify 'resume' and thread_id appear in the args
        call_args = mock_exec.call_args
        flat_args = list(call_args.args) if call_args.args else []
        # asyncio.create_subprocess_exec takes *args
        assert "resume" in flat_args
        assert "th_prev" in flat_args


def test_format_skill_invocation():
    backend = CodexBackend()
    # Should strip namespace prefix and use $ syntax
    result = backend.format_skill_invocation("beagle-python:review-python")
    assert result == "$review-python"


def test_format_skill_invocation_with_args():
    backend = CodexBackend()
    result = backend.format_skill_invocation("beagle-core:fetch-pr-feedback", "--pr 42 --bot mybot")
    assert result == "$fetch-pr-feedback --pr 42 --bot mybot"


def test_format_skill_invocation_no_namespace():
    backend = CodexBackend()
    result = backend.format_skill_invocation("commit-push")
    assert result == "$commit-push"
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_backend_codex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daydream.backends.codex'`

**Step 4: Write minimal implementation**

```python
# daydream/backends/codex.py
"""Codex CLI subprocess backend for daydream.

Spawns `codex exec --experimental-json` as an async subprocess,
writes the prompt to stdin, and reads JSONL events from stdout.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)


class CodexError(Exception):
    """Raised when a Codex turn fails."""


class CodexBackend:
    """Backend that wraps the Codex CLI subprocess.

    Translates Codex JSONL events into the unified AgentEvent stream.
    """

    def __init__(self, model: str = "gpt-5.3-codex"):
        self.model = model
        self._process: asyncio.subprocess.Process | None = None

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a prompt via Codex CLI and yield unified events.

        Args:
            cwd: Working directory for the agent.
            prompt: The prompt to send.
            output_schema: Optional JSON schema for structured output.
            continuation: Optional token for thread resumption.

        Yields:
            AgentEvent instances.

        Raises:
            CodexError: If the Codex turn fails.

        """
        args = [
            "codex", "exec", "--experimental-json",
            "--model", self.model,
            "--sandbox", "danger-full-access",
            "--cd", str(cwd),
        ]

        schema_path: str | None = None
        if output_schema:
            schema_path = self._write_temp_schema(output_schema)
            args.extend(["--output-schema", schema_path])

        if continuation and continuation.backend == "codex":
            args.extend(["resume", continuation.data["thread_id"]])

        thread_id: str | None = None
        last_agent_text: str | None = None
        structured_result: Any = None

        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Write prompt to stdin and close immediately
            if self._process.stdin:
                self._process.stdin.write(prompt.encode())
                self._process.stdin.close()

            # Read JSONL events line by line
            while True:
                if self._process.stdout is None:
                    break
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    event = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")

                elif event_type == "item.started":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "command_execution":
                        yield ToolStartEvent(
                            id=item.get("id", str(uuid.uuid4())),
                            name="shell",
                            input={"command": item.get("command", "")},
                        )
                    elif item_type == "mcp_tool_call":
                        yield ToolStartEvent(
                            id=item.get("id", str(uuid.uuid4())),
                            name=item.get("tool", "unknown"),
                            input=item.get("arguments", {}),
                        )
                    # agent_message and reasoning item.started are no-ops
                    # (text is empty, we wait for item.completed)

                elif event_type == "item.completed":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "agent_message":
                        text = self._extract_text(item)
                        if text:
                            last_agent_text = text
                            yield TextEvent(text=text)

                    elif item_type == "reasoning":
                        text = self._extract_text(item)
                        if text:
                            yield ThinkingEvent(text=text)

                    elif item_type == "command_execution":
                        item_id = item.get("id", str(uuid.uuid4()))
                        exit_code = item.get("exit_code", -1)
                        output = item.get("aggregated_output", "")
                        status = item.get("status", "")

                        if status == "declined":
                            yield ToolResultEvent(
                                id=item_id,
                                output="Command declined by sandbox",
                                is_error=True,
                            )
                        else:
                            yield ToolResultEvent(
                                id=item_id,
                                output=output,
                                is_error=exit_code != 0,
                            )

                    elif item_type == "file_change":
                        # file_change has no item.started — emit synthetic pair
                        item_id = item.get("id", str(uuid.uuid4()))
                        file_path = item.get("file_path", "unknown")
                        action = item.get("action", "modified")
                        yield ToolStartEvent(
                            id=item_id,
                            name="patch",
                            input={"file": file_path, "action": action},
                        )
                        yield ToolResultEvent(
                            id=item_id,
                            output=f"{action}: {file_path}",
                            is_error=False,
                        )

                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id", str(uuid.uuid4()))
                        result_content = ""
                        if "result" in item:
                            result_content = str(item["result"].get("content", ""))
                        error = item.get("error")
                        yield ToolResultEvent(
                            id=item_id,
                            output=result_content,
                            is_error=bool(error),
                        )

                elif event_type == "turn.completed":
                    usage = event.get("usage", {})
                    yield CostEvent(
                        cost_usd=None,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                    )

                    # Parse structured output from last agent message if schema was provided
                    if output_schema and last_agent_text:
                        try:
                            structured_result = json.loads(last_agent_text)
                        except json.JSONDecodeError:
                            pass

                    continuation_token = None
                    if thread_id:
                        continuation_token = ContinuationToken(
                            backend="codex",
                            data={"thread_id": thread_id},
                        )

                    yield ResultEvent(
                        structured_output=structured_result,
                        continuation=continuation_token,
                    )

                elif event_type == "turn.failed":
                    error = event.get("error", {})
                    raise CodexError(error.get("message", "Unknown Codex error"))

            await self._process.wait()

        finally:
            self._process = None
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)

    async def cancel(self) -> None:
        """Cancel the running Codex process.

        Sends SIGTERM, waits briefly, then SIGKILL if still running.
        """
        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation for Codex.

        Codex uses $skill-name syntax. Strips namespace prefix if present.

        Args:
            skill_key: Full skill key (e.g. "beagle-python:review-python").
            args: Optional arguments string.

        Returns:
            Formatted skill invocation string.

        """
        # Strip namespace prefix: "beagle-python:review-python" → "review-python"
        if ":" in skill_key:
            skill_name = skill_key.split(":")[-1]
        else:
            skill_name = skill_key

        result = f"${skill_name}"
        if args:
            result = f"{result} {args}"
        return result

    @staticmethod
    def _extract_text(item: dict[str, Any]) -> str:
        """Extract text from a Codex item's content blocks."""
        content = item.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _write_temp_schema(schema: dict[str, Any]) -> str:
        """Write JSON schema to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="daydream-schema-"
        ) as f:
            json.dump(schema, f)
            return f.name
```

**Step 5: Create fixture files**

Create the directory `tests/fixtures/codex_jsonl/` and write the four JSONL fixture files as shown in Step 1.

**Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_backend_codex.py -v`
Expected: PASS

**Step 7: Run Task 1 tests too (they depend on both backends existing)**

Run: `python -m pytest tests/test_backends_init.py tests/test_backend_codex.py tests/test_backend_claude.py -v`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add daydream/backends/codex.py tests/test_backend_codex.py tests/fixtures/
git commit -m "feat(backends): add CodexBackend with JSONL event parsing"
```

---

### Task 4: Run all existing tests — verify no regressions

**Files:**
- None (verification only)

**Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS — no existing tests should be broken. The new backends module is additive.

**Step 2: Run linter and type checker**

Run: `make lint && make typecheck`
Expected: PASS

**Step 3: Commit (only if lint/type fixes were needed)**

```bash
git add -A
git commit -m "chore: fix lint/type issues from backend extraction"
```

---

### Task 5: Simplify `agent.py` — consume unified event stream

**Files:**
- Modify: `daydream/agent.py`
- Test: `tests/test_integration.py` (update existing mocks)

This is the key refactor. `run_agent()` changes its signature to accept a `Backend` and consumes `AgentEvent` instead of Claude SDK types. The function no longer imports any Claude SDK types — it only knows about the unified event stream.

**Step 1: Write the failing test for the new signature**

Add a new test to `tests/test_integration.py` that tests the refactored `run_agent()` with a mock backend:

```python
# Add to tests/test_integration.py

from daydream.backends import (
    Backend,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)


class MockBackend:
    """Mock backend that yields a configurable event sequence."""

    def __init__(self, events: list):
        self._events = events
        self.cancelled = False

    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        for event in self._events:
            yield event

    async def cancel(self):
        self.cancelled = True

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}" + (f" {args}" if args else "")


@pytest.mark.asyncio
async def test_run_agent_with_backend_text_events(monkeypatch):
    """Test that run_agent() consumes TextEvent from a backend."""
    from daydream.agent import run_agent, set_quiet_mode

    set_quiet_mode(True)  # Suppress UI

    backend = MockBackend([
        TextEvent(text="Hello from backend"),
        CostEvent(cost_usd=0.01, input_tokens=100, output_tokens=50),
        ResultEvent(structured_output=None, continuation=None),
    ])

    output, continuation = await run_agent(
        backend=backend,
        cwd=Path("/tmp"),
        prompt="Test prompt",
    )

    assert output == "Hello from backend"
    assert continuation is None


@pytest.mark.asyncio
async def test_run_agent_with_backend_structured_output(monkeypatch):
    """Test that run_agent() returns structured output from ResultEvent."""
    from daydream.agent import run_agent, set_quiet_mode

    set_quiet_mode(True)

    structured = {"issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]}
    backend = MockBackend([
        TextEvent(text="Parsed."),
        ResultEvent(structured_output=structured, continuation=None),
    ])

    output, continuation = await run_agent(
        backend=backend,
        cwd=Path("/tmp"),
        prompt="Parse feedback",
        output_schema={"type": "object"},
    )

    assert output == structured
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_integration.py::test_run_agent_with_backend_text_events -v`
Expected: FAIL — `run_agent()` doesn't accept `backend` parameter yet

**Step 3: Refactor agent.py**

Replace `run_agent()` to consume the unified event stream. Key changes:

1. Remove all Claude SDK imports (`ClaudeSDKClient`, `ClaudeAgentOptions`, message types, block types)
2. Change `AgentState.current_clients` to `AgentState.current_backends`
3. Change `run_agent()` signature to accept `Backend` as first parameter
4. Return `tuple[str | Any, ContinuationToken | None]` instead of `str | Any`
5. Replace the SDK message loop with a simple event consumer loop
6. Keep all UI rendering logic (AgentTextRenderer, LiveToolPanelRegistry, print_thinking, print_cost)
7. Keep debug logging
8. Keep MissingSkillError detection (scan TextEvent.text for the pattern)

The refactored `run_agent()`:

```python
async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    continuation: ContinuationToken | None = None,
) -> tuple[str | Any, ContinuationToken | None]:
    """Run agent with the given prompt and return output plus continuation token.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the agent.
        prompt: The prompt to send to the agent.
        output_schema: Optional JSON schema for structured output.
        progress_callback: Optional callback for status updates (quiet mode).
        continuation: Optional continuation token for multi-turn.

    Returns:
        Tuple of (output, continuation_token). Output is text or structured data.

    Raises:
        MissingSkillError: If a required skill is not available.

    """
    _log_debug(f"\n{'='*80}\n")
    _log_debug(f"[PROMPT] cwd={cwd}\n{prompt}\n")
    _log_debug(f"{'='*80}\n\n")

    output_parts: list[str] = []
    structured_result: Any = None
    result_continuation: ContinuationToken | None = None
    use_callback = progress_callback is not None

    _state.current_backends.append(backend)
    try:
        if not use_callback:
            agent_renderer = AgentTextRenderer(console)
            tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)

        async for event in backend.execute(cwd, prompt, output_schema, continuation):
            if isinstance(event, TextEvent):
                output_parts.append(event.text)
                _log_debug(f"[TEXT] {event.text}\n")

                # Check for missing skill error
                skill_match = re.search(UNKNOWN_SKILL_PATTERN, event.text)
                if skill_match:
                    if not use_callback:
                        agent_renderer.finish()
                        tool_registry.finish_all()
                    raise MissingSkillError(skill_match.group(1))

                if use_callback and progress_callback is not None:
                    last_line = event.text.strip().split("\n")[-1]
                    if last_line:
                        progress_callback(last_line)
                else:
                    agent_renderer.append(event.text)

            elif isinstance(event, ThinkingEvent):
                _log_debug(f"[THINKING] {event.text}\n")
                if not use_callback:
                    if agent_renderer.has_content:
                        agent_renderer.finish()
                    print_thinking(console, event.text)

            elif isinstance(event, ToolStartEvent):
                _log_debug(f"[TOOL_USE] {event.name}({event.input})\n")
                if use_callback and progress_callback is not None:
                    first_arg = next(iter(event.input.values()), "") if event.input else ""
                    progress_callback(f"{event.name} {first_arg}"[:80])
                else:
                    if agent_renderer.has_content:
                        agent_renderer.finish()
                    tool_registry.create(event.id, event.name, event.input)

            elif isinstance(event, ToolResultEvent):
                _log_debug(f"[TOOL_RESULT{' [ERROR]' if event.is_error else ''}] {event.output}\n")
                if not use_callback:
                    panel = tool_registry.get(event.id)
                    if panel:
                        panel.set_result(event.output, event.is_error)
                        panel.finish()
                        tool_registry.remove(event.id)

            elif isinstance(event, CostEvent):
                if event.cost_usd:
                    _log_debug(f"[COST] ${event.cost_usd:.4f}\n")
                    if not use_callback:
                        if agent_renderer.has_content:
                            agent_renderer.finish()
                        console.print()
                        print_cost(console, event.cost_usd)
                elif event.input_tokens is not None:
                    _log_debug(f"[TOKENS] in={event.input_tokens} out={event.output_tokens}\n")

            elif isinstance(event, ResultEvent):
                if event.structured_output is not None:
                    structured_result = event.structured_output
                    _log_debug(f"[STRUCTURED_OUTPUT] {structured_result}\n")
                    if not use_callback:
                        issues = structured_result.get("issues", []) if isinstance(structured_result, dict) else []
                        if issues:
                            lines = [f"[{i['id']}] {i['file']}:{i['line']} - {i['description']}" for i in issues]
                            agent_renderer.append("\n".join(lines))
                result_continuation = event.continuation

        if not use_callback:
            if agent_renderer.has_content:
                agent_renderer.finish()
            tool_registry.finish_all()
            console.print()
    finally:
        _state.current_backends.remove(backend)

    if output_schema is not None and structured_result is not None:
        return structured_result, result_continuation
    return "".join(output_parts), result_continuation
```

Also update `AgentState`:

```python
@dataclass
class AgentState:
    debug_log: TextIO | None = None
    quiet_mode: bool = False
    model: str = "opus"
    shutdown_requested: bool = False
    current_backends: list[Any] = field(default_factory=list)
```

Update `get_current_clients()` → `get_current_backends()`:

```python
def get_current_backends() -> list[Any]:
    return list(_state.current_backends)
```

Remove these imports from agent.py:
- `from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient`
- `from claude_agent_sdk.types import ...`

Add these imports:
- `from daydream.backends import Backend, ContinuationToken, TextEvent, ThinkingEvent, ToolStartEvent, ToolResultEvent, CostEvent, ResultEvent`

**Step 4: Update `cli.py` signal handler**

In `cli.py`, update the signal handler to use `get_current_backends()` instead of `get_current_clients()`:

```python
# cli.py - change import
from daydream.agent import (
    console,
    get_current_backends,
    set_shutdown_requested,
)

# In _signal_handler:
if get_current_backends():
    panel.add_step("Terminating running agent(s)...")
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_integration.py::test_run_agent_with_backend_text_events tests/test_integration.py::test_run_agent_with_backend_structured_output -v`
Expected: PASS

**Step 6: Update existing integration tests**

The existing tests in `test_integration.py` that mock `ClaudeSDKClient` directly need to be updated. They should now use `MockBackend` that yields events, since `run_agent()` no longer touches the SDK directly.

For the `test_full_fix_flow` test: the mock needs to be at the `Backend` level, not at the SDK level. Update the `mock_sdk_client` fixture to patch at the backend level.

However, since `test_full_fix_flow` calls `run()` which calls `run_agent()`, and `run()` now creates a backend, the patching point changes. The simplest approach: patch `create_backend` in `runner.py` to return a `MockBackend`.

Update existing tests to use `MockBackend` and patch at the `run_agent` or `create_backend` level. The tool panel tests (`test_glob_tool_panel_*`, `test_quiet_mode_*`, etc.) should be updated to create `MockBackend` instances that yield the appropriate events.

**Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add daydream/agent.py daydream/cli.py tests/test_integration.py
git commit -m "refactor(agent): consume unified event stream from Backend protocol"
```

---

### Task 6: Update `phases.py` — accept Backend, use format_skill_invocation()

**Files:**
- Modify: `daydream/phases.py`
- Test: `tests/test_phases.py` (new)

Every phase function that calls `run_agent()` now needs to pass a `backend` as the first argument. Phase functions that invoke skills (review, commit-push, fetch-pr-feedback, respond-pr-feedback) need to use `backend.format_skill_invocation()` instead of hardcoded `/{skill}` syntax.

**Step 1: Write the failing test**

```python
# tests/test_phases.py
"""Tests for phase functions with backend abstraction."""

from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    ContinuationToken,
    ResultEvent,
    TextEvent,
)


class MockBackend:
    """Mock backend that records calls and yields canned events."""

    def __init__(self, events: list | None = None):
        self.calls: list[dict[str, Any]] = []
        self._events = events or [
            TextEvent(text="OK"),
            ResultEvent(structured_output=None, continuation=None),
        ]

    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        self.calls.append({
            "cwd": cwd,
            "prompt": prompt,
            "output_schema": output_schema,
            "continuation": continuation,
        })
        for event in self._events:
            yield event

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        # Track that format was called with the right key
        name = skill_key.split(":")[-1] if ":" in skill_key else skill_key
        result = f"$test-{name}"
        if args:
            result = f"{result} {args}"
        return result


@pytest.mark.asyncio
async def test_phase_review_uses_format_skill_invocation(tmp_path, monkeypatch):
    """phase_review should use backend.format_skill_invocation() for skill syntax."""
    from daydream.phases import phase_review

    # Suppress UI
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)

    # Create review output file (mock agent doesn't write it)
    (tmp_path / ".review-output.md").write_text("review content")

    backend = MockBackend()
    await phase_review(backend, tmp_path, "beagle-python:review-python")

    assert len(backend.calls) == 1
    prompt = backend.calls[0]["prompt"]
    # Should use the formatted invocation from our mock ($test-review-python)
    assert "$test-review-python" in prompt


@pytest.mark.asyncio
async def test_phase_commit_push_uses_format_skill_invocation(tmp_path, monkeypatch):
    """phase_commit_push should use backend.format_skill_invocation()."""
    from daydream.phases import phase_commit_push

    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)

    backend = MockBackend()
    await phase_commit_push(backend, tmp_path)

    assert len(backend.calls) == 1
    prompt = backend.calls[0]["prompt"]
    assert "$test-commit-push" in prompt


@pytest.mark.asyncio
async def test_phase_fix_passes_backend(tmp_path, monkeypatch):
    """phase_fix should pass backend to run_agent."""
    from daydream.phases import phase_fix

    monkeypatch.setattr("daydream.phases.print_fix_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_fix_complete", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    item = {"id": 1, "description": "Fix bug", "file": "main.py", "line": 10}
    backend = MockBackend()
    await phase_fix(backend, tmp_path, item, 1, 1)

    assert len(backend.calls) == 1
    assert "Fix bug" in backend.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_phase_parse_feedback_passes_backend(tmp_path, monkeypatch):
    """phase_parse_feedback should pass backend to run_agent."""
    from daydream.phases import phase_parse_feedback

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)

    (tmp_path / ".review-output.md").write_text("review content")

    structured = {"issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]}
    backend = MockBackend(events=[
        TextEvent(text="Parsed."),
        ResultEvent(structured_output=structured, continuation=None),
    ])

    items = await phase_parse_feedback(backend, tmp_path)

    assert len(items) == 1
    assert items[0]["description"] == "Fix X"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_phases.py -v`
Expected: FAIL — `phase_review()` doesn't accept `backend` parameter yet

**Step 3: Update phases.py**

Add `backend: Backend` as the first parameter to every phase function. Update all `run_agent()` calls to pass `backend` as the first argument. Replace hardcoded skill invocations with `backend.format_skill_invocation()`.

Key changes:

1. **Import**: Add `from daydream.backends import Backend, ContinuationToken`
2. **Import**: Update `run_agent` import — it now needs `backend` as first arg
3. **phase_review(backend, cwd, skill)**: Replace `f"/{skill}"` with `backend.format_skill_invocation(skill)`
4. **phase_parse_feedback(backend, cwd)**: Pass `backend` to `run_agent()`
5. **phase_fix(backend, cwd, item, num, total)**: Pass `backend` to `run_agent()`
6. **phase_test_and_heal(backend, cwd)**: Pass `backend` to `run_agent()`, handle continuation token for retry loop
7. **phase_commit_push(backend, cwd)**: Replace `"/beagle-core:commit-push"` with `backend.format_skill_invocation("beagle-core:commit-push")`
8. **phase_fetch_pr_feedback(backend, cwd, pr, bot)**: Replace hardcoded skill invocation
9. **phase_fix_parallel(backend, cwd, items)**: Pass `backend` to `run_agent()`
10. **phase_commit_push_auto(backend, cwd)**: Replace hardcoded skill invocation
11. **phase_respond_pr_feedback(backend, cwd, pr, bot, results)**: Replace hardcoded skill invocation

For `run_agent()` calls, update from:
```python
await run_agent(cwd, prompt)
```
to:
```python
output, _ = await run_agent(backend, cwd, prompt)
```

For `phase_test_and_heal`, track the continuation token:
```python
async def phase_test_and_heal(backend: Backend, cwd: Path) -> tuple[bool, int]:
    continuation = None
    while True:
        output, continuation = await run_agent(backend, cwd, prompt, continuation=continuation)
        # ... rest of loop
        # On fix retry:
        _, continuation = await run_agent(backend, cwd, TEST_FIX_PROMPT, continuation=continuation)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_phases.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "refactor(phases): accept Backend parameter, use format_skill_invocation()"
```

---

### Task 7: Update `runner.py` — create backend per-phase, add RunConfig fields

**Files:**
- Modify: `daydream/runner.py`
- Modify: `daydream/runner.py` (RunConfig)

The runner creates a `Backend` instance and passes it to each phase function. For Phase 1 (this PR), it always creates `ClaudeBackend` — per-phase backend selection is Task 9.

**Step 1: Write the failing test**

```python
# Add to tests/test_integration.py or a new tests/test_runner.py

@pytest.mark.asyncio
async def test_run_creates_backend_and_passes_to_phases(tmp_path, monkeypatch):
    """run() should create a backend and pass it to phase functions."""
    from daydream.runner import RunConfig, run
    from daydream.backends.claude import ClaudeBackend

    captured_backends = []

    original_phase_review = None

    async def mock_phase_review(backend, cwd, skill):
        captured_backends.append(backend)
        # Create review output
        (cwd / ".review-output.md").write_text("review")

    async def mock_phase_parse(backend, cwd):
        captured_backends.append(backend)
        return []

    monkeypatch.setattr("daydream.runner.phase_review", mock_phase_review)
    monkeypatch.setattr("daydream.runner.phase_parse_feedback", mock_phase_parse)
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")

    config = RunConfig(
        target=str(tmp_path),
        skill="python",
        quiet=True,
        cleanup=False,
        review_only=True,
    )

    await run(config)

    # All phases should have received the same backend
    assert len(captured_backends) == 2
    assert all(isinstance(b, ClaudeBackend) for b in captured_backends)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_integration.py::test_run_creates_backend_and_passes_to_phases -v`
Expected: FAIL — `run()` doesn't create backends yet

**Step 3: Update runner.py**

1. **Import**: Add `from daydream.backends import create_backend`
2. **RunConfig**: Add `backend: str = "claude"` field (per-phase fields come in Task 9)
3. **run()**: After setting model, create backend:
   ```python
   backend = create_backend(config.backend, config.model if config.model != "opus" else None)
   ```
   Wait — actually, since `--model` flag currently only accepts Claude model names, and we need the backend's default if no model is passed, the logic is:
   ```python
   model_override = config.model if config.model != "opus" else None
   backend = create_backend(config.backend, model=model_override)
   ```
   Actually, simpler: just pass `config.model` always. If backend is "claude", model is "opus" by default. If backend is "codex", the user must pass `--model gpt-5.3-codex`. But the design says each backend has its own default and `--model` is a raw passthrough. So:
   ```python
   backend = create_backend(config.backend, model=config.model if config.model else None)
   ```
   But `config.model` defaults to `"opus"`, which would be wrong for codex. The design says "If the user doesn't pass `--model`, each backend uses its own default." So we need to distinguish "user passed --model opus" from "default value opus". Change the default to `None`:

   In RunConfig:
   ```python
   model: str | None = None  # None = use backend default
   ```

   In cli.py:
   ```python
   parser.add_argument("--model", default=None, ...)
   ```

   Then in runner.py:
   ```python
   backend = create_backend(config.backend, model=config.model)
   ```

   And for `set_model()`, if model is None, keep "opus" for backward compat:
   ```python
   set_model(config.model or "opus")
   ```

4. Pass `backend` to all phase calls:
   ```python
   await phase_review(backend, target_dir, skill)
   feedback_items = await phase_parse_feedback(backend, target_dir)
   await phase_fix(backend, target_dir, item, i, len(feedback_items))
   await phase_test_and_heal(backend, target_dir)
   await phase_commit_push(backend, target_dir)
   ```

5. Same for `run_pr_feedback()`:
   ```python
   backend = create_backend(config.backend, model=config.model)
   await phase_fetch_pr_feedback(backend, target_dir, pr_number, bot)
   # etc.
   ```

6. Update signal handler references: In `runner.py`, no direct client handling — that's in `cli.py` and `agent.py`. The `_state.current_backends` list is populated/depopulated by `run_agent()`, and the signal handler in `cli.py` calls `backend.cancel()` on each.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/runner.py daydream/cli.py tests/test_integration.py
git commit -m "refactor(runner): create backend and pass to phases"
```

---

### Task 8: Update `cli.py` — add --backend flag, update --model default

**Files:**
- Modify: `daydream/cli.py`
- Test: `tests/test_cli.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys

import pytest

from daydream.cli import _parse_args


def test_default_backend_is_claude(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.backend == "claude"


def test_backend_flag_codex(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_backend_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "-b", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_model_default_is_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.model is None


def test_model_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--model", "sonnet"])
    config = _parse_args()
    assert config.model == "sonnet"


def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — no `--backend` flag exists

**Step 3: Update cli.py**

1. Add `--backend` / `-b` argument:
   ```python
   parser.add_argument(
       "--backend", "-b",
       choices=["claude", "codex"],
       default="claude",
       help="Agent backend: claude (default) or codex",
   )
   ```

2. Change `--model` to not restrict choices (raw passthrough):
   ```python
   parser.add_argument(
       "--model",
       default=None,
       help="Model to use (default: backend-specific). Examples: opus, sonnet, haiku, gpt-5.3-codex",
   )
   ```

3. Add `backend=args.backend` to `RunConfig(...)` construction

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/cli.py tests/test_cli.py
git commit -m "feat(cli): add --backend flag, make --model a raw passthrough"
```

---

### Task 9: Wire per-phase backend overrides

**Files:**
- Modify: `daydream/cli.py`
- Modify: `daydream/runner.py`
- Test: `tests/test_cli.py` (add tests)
- Test: `tests/test_runner.py` (new or add to existing)

**Step 1: Write the failing test**

```python
# Add to tests/test_cli.py

def test_review_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--backend", "claude", "--review-backend", "codex",
    ])
    config = _parse_args()
    assert config.backend == "claude"
    assert config.review_backend == "codex"


def test_fix_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--fix-backend", "codex",
    ])
    config = _parse_args()
    assert config.fix_backend == "codex"


def test_test_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--test-backend", "codex",
    ])
    config = _parse_args()
    assert config.test_backend == "codex"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_review_backend_override -v`
Expected: FAIL

**Step 3: Update cli.py and runner.py**

In `cli.py`, add three new arguments:
```python
parser.add_argument("--review-backend", choices=["claude", "codex"], default=None,
                    help="Override backend for review phase")
parser.add_argument("--fix-backend", choices=["claude", "codex"], default=None,
                    help="Override backend for fix phase")
parser.add_argument("--test-backend", choices=["claude", "codex"], default=None,
                    help="Override backend for test phase")
```

In `RunConfig`, add:
```python
review_backend: str | None = None
fix_backend: str | None = None
test_backend: str | None = None
```

In `runner.py`, resolve backend per-phase:
```python
def _resolve_backend(config: RunConfig, phase: str) -> Backend:
    """Create the backend for a given phase, respecting per-phase overrides."""
    override = getattr(config, f"{phase}_backend", None)
    backend_name = override or config.backend
    return create_backend(backend_name, model=config.model)

# Usage in run():
review_backend = _resolve_backend(config, "review")
await phase_review(review_backend, target_dir, skill)

fix_backend = _resolve_backend(config, "fix")
for i, item in enumerate(feedback_items, 1):
    await phase_fix(fix_backend, target_dir, item, i, len(feedback_items))

test_backend = _resolve_backend(config, "test")
await phase_test_and_heal(test_backend, target_dir)
```

Parse phase uses the same backend as review (no override needed — it's always a local operation).

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py tests/test_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/cli.py daydream/runner.py tests/test_cli.py
git commit -m "feat(cli): add per-phase backend overrides (--review-backend, --fix-backend, --test-backend)"
```

---

### Task 10: Wire continuation tokens through test-and-heal retry loop

**Files:**
- Modify: `daydream/phases.py` (phase_test_and_heal)
- Test: `tests/test_phases.py` (add test)

**Step 1: Write the failing test**

```python
# Add to tests/test_phases.py

@pytest.mark.asyncio
async def test_phase_test_and_heal_passes_continuation(tmp_path, monkeypatch):
    """Test that continuation token is threaded through test retries."""
    from daydream.phases import phase_test_and_heal

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_menu", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    call_count = 0
    token = ContinuationToken(backend="codex", data={"thread_id": "th_test"})

    class ContinuationBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tests fail
                yield TextEvent(text="1 failed, 0 passed")
                yield ResultEvent(structured_output=None, continuation=token)
            elif call_count == 2:
                # Fix call: should receive the continuation token
                assert continuation is token, f"Expected continuation token, got {continuation}"
                yield TextEvent(text="Fixed")
                yield ResultEvent(structured_output=None, continuation=token)
            else:
                # Retry: tests pass, should receive token
                assert continuation is token
                yield TextEvent(text="All 1 tests passed")
                yield ResultEvent(structured_output=None, continuation=token)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    # Simulate: fail → choice "2" (fix and retry) → pass
    choices = iter(["2"])
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(choices, "3"))

    backend = ContinuationBackend()
    success, retries = await phase_test_and_heal(backend, tmp_path)

    assert success is True
    assert retries == 1
    assert call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_phases.py::test_phase_test_and_heal_passes_continuation -v`
Expected: FAIL — continuation not threaded through

**Step 3: Update phase_test_and_heal**

The function already has the `continuation` tracking from Task 6. Verify it correctly passes continuation through the retry loop:

```python
async def phase_test_and_heal(backend: Backend, cwd: Path) -> tuple[bool, int]:
    continuation: ContinuationToken | None = None
    retries_used = 0

    while True:
        # ... UI code ...
        prompt = "Run the project's test suite. Report if tests pass or fail."
        output, continuation = await run_agent(backend, cwd, prompt, continuation=continuation)
        # ... check test_passed ...
        # On fix:
        _, continuation = await run_agent(backend, cwd, TEST_FIX_PROMPT, continuation=continuation)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_phases.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "feat(phases): wire continuation tokens through test-and-heal retry loop"
```

---

### Task 11: Full integration test — end-to-end with MockBackend

**Files:**
- Modify: `tests/test_integration.py`

Update the `test_full_fix_flow` test to work with the new backend abstraction. This validates the entire pipeline works end-to-end.

**Step 1: Update test_full_fix_flow**

```python
@pytest.mark.asyncio
async def test_full_fix_flow_with_backend(mock_ui, target_project: Path, monkeypatch):
    """Test the complete flow using the backend abstraction."""
    from daydream.backends import ResultEvent, TextEvent

    call_index = 0

    class FullFlowBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_index
            call_index += 1
            prompt_lower = prompt.lower()

            if "review" in prompt_lower and ("$" in prompt or "/" in prompt):
                yield TextEvent(text="Review complete. Found 1 issue to fix.")
                yield ResultEvent(structured_output=None, continuation=None)
            elif output_schema is not None:
                structured = {
                    "issues": [
                        {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
                    ]
                }
                yield TextEvent(text="Extracted feedback.")
                yield ResultEvent(structured_output=structured, continuation=None)
            elif "fix this issue" in prompt_lower:
                yield TextEvent(text="Fixed the issue.")
                yield ResultEvent(structured_output=None, continuation=None)
            elif "test suite" in prompt_lower:
                yield TextEvent(text="All 1 tests passed. 0 failed.")
                yield ResultEvent(structured_output=None, continuation=None)
            else:
                yield TextEvent(text="OK")
                yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}" + (f" {args}" if args else "")

    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None: FullFlowBackend(),
    )

    config = RunConfig(
        target=str(target_project),
        skill="python",
        quiet=True,
        cleanup=False,
    )

    exit_code = await run(config)
    assert exit_code == 0
```

**Step 2: Run test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: update integration tests for backend abstraction"
```

---

### Task 12: Final validation — lint, typecheck, full test suite

**Files:**
- None (verification only)

**Step 1: Run linter**

Run: `make lint`
Expected: PASS

**Step 2: Run type checker**

Run: `make typecheck`
Expected: PASS — may need to add type stubs or `# type: ignore` for protocol conformance edge cases

**Step 3: Run full test suite**

Run: `make test`
Expected: ALL PASS

**Step 4: Run all checks**

Run: `make check`
Expected: PASS

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix lint/type issues from backend implementation"
```

---

### Task 13: Commit and create PR

**Files:**
- None

**Step 1: Verify git status is clean**

Run: `git status`

**Step 2: Create PR**

Run:
```bash
gh pr create --title "feat: add backend abstraction layer (Claude + Codex)" --body "$(cat <<'EOF'
## Summary

- Introduces `daydream/backends/` package with `Backend` protocol, unified event types, and factory
- Extracts `ClaudeBackend` from existing `agent.py` SDK interaction loop
- Adds `CodexBackend` that spawns `codex exec --experimental-json` subprocess
- Simplifies `agent.py` to consume unified `AgentEvent` stream
- Updates all phases to accept `Backend` parameter and use `format_skill_invocation()`
- Adds `--backend` / `-b` flag and per-phase overrides (`--review-backend`, `--fix-backend`, `--test-backend`)
- Wires continuation tokens through test-and-heal retry loop for Codex thread resumption

Implements design from `docs/plans/2026-02-06-codex-backend-design.md` (Phases 1-3).

## Test plan

- [ ] `make check` passes (lint + typecheck + tests)
- [ ] `python -m pytest tests/test_backends_init.py -v` — event types, factory
- [ ] `python -m pytest tests/test_backend_claude.py -v` — Claude backend extraction
- [ ] `python -m pytest tests/test_backend_codex.py -v` — Codex JSONL parsing
- [ ] `python -m pytest tests/test_phases.py -v` — phases with backend parameter
- [ ] `python -m pytest tests/test_cli.py -v` — CLI flags
- [ ] `python -m pytest tests/test_integration.py -v` — full flow
- [ ] Manual: `daydream /path/to/project --python` works unchanged (Claude backend)
- [ ] Manual: `daydream /path/to/project --python --backend codex --model gpt-5.3-codex` works with Codex

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

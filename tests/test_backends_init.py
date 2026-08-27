# tests/test_backends_init.py
"""Tests for the backend protocol and the ``create_backend`` factory.

The ``AgentEvent`` dataclass field/default assertions live in
``tests/test_backends_events.py``; they used to be duplicated here.
"""
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import (
    ClaudeBackend,
    ContinuationToken,
    ResultEvent,
    create_backend,
)


def test_continuation_token_fields() -> None:
    token = ContinuationToken(backend="codex", data={"thread_id": "abc"})
    assert token.backend == "codex"
    assert token.data == {"thread_id": "abc"}


def test_create_backend_claude_default_uses_config_constant() -> None:
    from daydream.config import DEFAULT_CLAUDE_MODEL
    backend = create_backend("claude")
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == DEFAULT_CLAUDE_MODEL


def test_create_backend_claude_custom_model() -> None:
    backend = create_backend("claude", model="sonnet")
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == "sonnet"


def test_create_backend_codex_default_uses_config_constant() -> None:
    from daydream.backends.codex import CodexBackend
    from daydream.config import DEFAULT_CODEX_MODEL
    backend = create_backend("codex")
    assert isinstance(backend, CodexBackend)
    assert backend.model == DEFAULT_CODEX_MODEL


def test_create_backend_codex_custom_model() -> None:
    backend = create_backend("codex", model="o3-pro")
    from daydream.backends.codex import CodexBackend
    assert isinstance(backend, CodexBackend)
    assert backend.model == "o3-pro"


def test_create_backend_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")


def test_pi_backend_concise_fix_prompts_true() -> None:
    """PiBackend requests concise fix prompts by default (GLM verbosity suppression)."""
    from daydream.backends.pi import PiBackend
    backend = PiBackend(model="glm-5.2")
    assert backend.concise_fix_prompts is True


def test_claude_and_codex_backend_concise_fix_prompts_false() -> None:
    """Claude and Codex backends do not request concise fix prompts by default."""
    from daydream.backends.codex import CodexBackend
    assert ClaudeBackend(model="test").concise_fix_prompts is False
    assert CodexBackend(model="test").concise_fix_prompts is False


@pytest.mark.asyncio
async def test_backend_execute_accepts_agents_kwarg() -> None:
    """MockBackend (satisfying Backend protocol) should accept agents=None."""
    from collections.abc import AsyncGenerator

    from daydream.backends import AgentEvent

    class MockBackendWithAgents:
        model = "mock-model"

        async def execute(
            self,
            cwd: Any,
            prompt: Any,
            output_schema: Any=None,
            continuation: Any=None,
            agents: Any=None,
            max_turns: Any=None,
            read_only: Any=False,
        ) -> AsyncGenerator[AgentEvent, None]:
            yield ResultEvent(structured_output=None, continuation=None)
            return

        async def cancel(self) -> None:
            pass

    backend = MockBackendWithAgents()
    events = []
    async for event in backend.execute(Path("/tmp"), "test", agents=None):
        events.append(event)
    assert len(events) == 1


def test_create_backend_forwards_reasoning_effort_to_every_driver() -> None:
    """All three backends carry the resolved effort, not just Codex."""
    from daydream.backends import create_backend

    for name in ("claude", "codex", "pi"):
        backend: Any = create_backend(name, reasoning_effort="max")
        assert backend.reasoning_effort == "max", name


def test_create_backend_without_reasoning_effort_leaves_it_unset() -> None:
    from daydream.backends import create_backend

    for name in ("claude", "codex", "pi"):
        assert cast(Any, create_backend(name)).reasoning_effort is None, name

"""Tests for retry/backoff logic in run_agent (daydream/agent.py).

Every test drives run_agent — the production entrypoint — with a mock backend
that simulates retryable and non-retryable failures. Tests assert on observable
outcomes (returned output, call count) never on internal implementation details.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.backends.pi import PiError
from daydream.trajectory import DaydreamPhase


class _RetryableThenSuccessBackend:
    """Raises a retryable PiError on the first call, succeeds on the second."""

    model = "test-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.call_count += 1
        if self.call_count == 1:
            raise PiError("429 Too Many Requests - rate limit exceeded", retryable=True)
        yield TextEvent(text="Review complete")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
        return ""


class _NonRetryableBackend:
    """Always raises a non-retryable PiError."""

    model = "test-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.call_count += 1
        raise PiError("auth failed", retryable=False)
        yield  # make this an async generator

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
        return ""


class _AlwaysRetryableBackend:
    """Always raises a retryable PiError (used to test retry exhaustion)."""

    model = "test-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.call_count += 1
        raise PiError("429 rate limit", retryable=True)
        yield  # make this an async generator

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
        return ""


class _PartialThenRetryBackend:
    """Yields partial text then raises retryable on attempt 1; yields final text on attempt 2."""

    model = "test-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.call_count += 1
        if self.call_count == 1:
            yield TextEvent(text="partial text")
            raise PiError("429 overload", retryable=True)
        yield TextEvent(text="final text")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
        return ""


@pytest.mark.asyncio
async def test_run_agent_retries_on_retryable_error(monkeypatch, tmp_path: Path) -> None:
    """First call raises retryable PiError; second succeeds. Output is from the second call."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = _RetryableThenSuccessBackend()

    output, _, _ = await run_agent(
        backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW
    )

    assert output == "Review complete"
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_no_retry_on_non_retryable(monkeypatch, tmp_path: Path) -> None:
    """Non-retryable PiError propagates immediately without any retry."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = _NonRetryableBackend()

    with pytest.raises(PiError, match="auth failed"):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_run_agent_retry_exhausted(monkeypatch, tmp_path: Path) -> None:
    """Always-retryable backend is called max_attempts+1 times total, then raises."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "2")
    backend = _AlwaysRetryableBackend()

    with pytest.raises(PiError):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    # 1 original attempt + 2 retries = 3 total
    assert backend.call_count == 3


@pytest.mark.asyncio
async def test_run_agent_retry_resets_output(monkeypatch, tmp_path: Path) -> None:
    """Partial output from a failed attempt is discarded; only the final output is returned."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = _PartialThenRetryBackend()

    output, _, _ = await run_agent(
        backend, tmp_path, "review", phase=DaydreamPhase.REVIEW
    )

    assert output == "final text"
    assert backend.call_count == 2

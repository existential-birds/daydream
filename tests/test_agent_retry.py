"""Tests for retry/backoff logic in run_agent (daydream/agent.py).

Every test drives run_agent — the production entrypoint — with a mock backend
that simulates retryable and non-retryable failures. Tests assert on observable
outcomes (returned output, call count) never on internal implementation details.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.backends.pi import PiError, _is_retryable_error_message
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


class _RetryableThenSuccessBackend:
    """Raises a retryable PiError on the first call, succeeds on the second."""

    model = "test-model"
    fanout_concurrency = 4
    retry_attempts = 1
    retry_base_delay_s = 0.0
    retry_max_delay_s = 0.0

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
        persist_session: bool = True,
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
        persist_session: bool = True,
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
        persist_session: bool = True,
    ):
        self.call_count += 1
        raise PiError("429 rate limit", retryable=True)
        yield  # make this an async generator

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
        return ""


class _PlainErrorBackend:
    """Raises a non-retryable plain exception with NO ``.category`` (Claude/Codex-style)."""

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
        persist_session: bool = True,
    ):
        self.call_count += 1
        # A human-readable reason plus a secret-shaped substring, to prove the
        # message surfaces AND that secrets are scrubbed at the host boundary.
        raise RuntimeError("overloaded-502 ZAI_API_KEY=leaked-secret-abc123")
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
        persist_session: bool = True,
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
async def test_run_agent_ignores_malformed_retry_environment(monkeypatch, tmp_path: Path) -> None:
    """Malformed Pi retry environment values fall back without blocking a backend call."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "not-an-integer")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "nan")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "inf")
    backend = _RetryableThenSuccessBackend()
    backend.retry_attempts = 1
    backend.retry_base_delay_s = 0.0
    backend.retry_max_delay_s = 0.0

    output, _, _ = await run_agent(
        backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW
    )

    assert output == "Review complete"
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_surfaces_backend_error_message(monkeypatch, tmp_path: Path) -> None:
    """A categoryless backend error surfaces its MESSAGE to the user, not a bare class name."""
    from rich.console import Console

    rec = Console(record=True, force_terminal=True, width=200)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = _PlainErrorBackend()

    with pytest.raises(RuntimeError, match="overloaded-502"):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    out = rec.export_text()
    assert "Backend Execution Error" in out
    # The exception MESSAGE (not just "RuntimeError") must reach the user.
    assert "overloaded-502" in out
    # ...but a secret embedded in that message is redacted at the host boundary.
    assert "leaked-secret-abc123" not in out
    assert "[REDACTED_ENV_VAR]" in out


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


@pytest.mark.asyncio
async def test_concurrent_retry_does_not_kill_sibling_invocations(
    monkeypatch, tmp_path: Path
) -> None:
    """Shared-backend concurrency shape: a retryable failure on one concurrent invocation
    must not abort sibling invocations that share the same backend instance.

    This mirrors phases.phase_per_stack_reviews, where multiple run_agent calls share
    a single Backend under an anyio TaskGroup with a CapacityLimiter.

    The key contract under test: agent.py's retry path does NOT call backend.cancel()
    (which would kill all subprocesses on the shared backend, including siblings).
    It only closes the individual event iterator for the failing invocation.
    """
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")

    cancel_calls: list[str] = []

    class _SharedBackend:
        """Three named prompt → behaviour mappings on one shared instance.

        - prompt containing "fail-once": retryable PiError on first call, succeeds on retry.
        - prompt containing "ok-a" / "ok-b": always succeeds immediately.

        cancel() is tracked; the test asserts it is NOT called during retry so that
        sibling concurrent invocations are unaffected.
        """

        model = "test-model"
        fanout_concurrency = 3
        # retry_attempts read by agent.py via getattr(backend, "retry_attempts", 3)
        retry_attempts = 3
        retry_base_delay_s = 0.01

        def __init__(self) -> None:
            self.call_counts: dict[str, int] = {}

        async def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: Any = None,
            continuation: Any = None,
            agents: Any = None,
            max_turns: Any = None,
            read_only: bool = False,
            persist_session: bool = True,
        ):
            key = (
                "fail-once"
                if "fail-once" in prompt
                else "ok-a"
                if "ok-a" in prompt
                else "ok-b"
            )
            self.call_counts[key] = self.call_counts.get(key, 0) + 1
            if key == "fail-once" and self.call_counts[key] == 1:
                raise PiError("429 overload", retryable=True)
            yield TextEvent(text=f"done-{key}")
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self) -> None:
            cancel_calls.append("cancel")

        def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
            return ""

    backend = _SharedBackend()

    results: list[tuple[str, str]] = []

    async def _run(prompt: str) -> None:
        output, _, _ = await run_agent(
            backend, tmp_path, prompt, phase=DaydreamPhase.REVIEW
        )
        results.append((prompt, output))

    # Run all three concurrently — same shape as phase_per_stack_reviews TaskGroup.
    async with anyio.create_task_group() as tg:
        tg.start_soon(_run, "fail-once review")
        tg.start_soon(_run, "ok-a review")
        tg.start_soon(_run, "ok-b review")

    # All three invocations must have produced output — siblings must survive the retry.
    assert len(results) == 3, f"Expected 3 results, got {len(results)}: {results}"

    outputs = {prompt: out for prompt, out in results}
    assert outputs["fail-once review"] == "done-fail-once"
    assert outputs["ok-a review"] == "done-ok-a"
    assert outputs["ok-b review"] == "done-ok-b"

    # backend.cancel() must NOT have been called during retry — calling it would kill
    # all subprocesses on the shared backend, terminating sibling concurrent tasks.
    assert cancel_calls == [], (
        f"backend.cancel() was called {len(cancel_calls)} time(s) during retry; "
        "this would kill sibling concurrent invocations"
    )

    # The fail-once slot was called twice (fail + retry); others exactly once.
    assert backend.call_counts.get("fail-once", 0) == 2
    assert backend.call_counts.get("ok-a", 0) == 1
    assert backend.call_counts.get("ok-b", 0) == 1


class _StreamDropThenSuccessBackend:
    """Raises a stream-drop PiError on the first call, succeeds on the second.

    Uses the production classifier ``_is_retryable_error_message`` to set
    ``retryable``, mirroring how ``PiBackend`` constructs ``PiError`` in
    production. This ensures the test exercises the real classification path:
    if ``_is_retryable_error_message("terminated")`` ever returns ``False``,
    ``run_agent`` would NOT retry and the test would fail.
    """

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
        persist_session: bool = True,
    ):
        self.call_count += 1
        if self.call_count == 1:
            # Mirrors production: PiBackend raises PiError with retryable set
            # by the _is_retryable_error_message classifier. If the classifier
            # stops recognizing "terminated" as retryable, this raises with
            # retryable=False and run_agent does NOT retry — the test fails.
            raise PiError(
                "terminated",
                retryable=_is_retryable_error_message("terminated"),
            )
        yield TextEvent(text="Review complete after retry")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, *a: Any, **kw: Any) -> str:
        return ""


@pytest.mark.asyncio
async def test_run_agent_retries_on_stream_drop(monkeypatch, tmp_path: Path) -> None:
    """First call raises PiError('terminated') (stream-drop); second succeeds."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = _StreamDropThenSuccessBackend()

    output, _, _ = await run_agent(
        backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW
    )

    assert output == "Review complete after retry"
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_retry_exhausted_marks_trajectory_partial(
    monkeypatch, tmp_path: Path
) -> None:
    """Retry-exhaustion → trajectory ``partial`` composition (PR headline).

    When a retryable ``PiError`` exhausts all retries, ``run_agent`` re-raises
    and the exception propagates through the active ``TrajectoryRecorder``
    scope. The recorder stamps ``extra.partial = True`` on the emitted
    trajectory so downstream consumers can distinguish clean completions from
    aborted ones. Real-path test driving the PR's headline behavior through
    the production entrypoint (``run_agent``) with a real recorder on the real
    filesystem.
    """
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "2")
    backend = _AlwaysRetryableBackend()

    trajectory_path = tmp_path / ".daydream" / "trajectory.json"
    recorder = TrajectoryRecorder(
        path=trajectory_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="test",
    )

    with pytest.raises(PiError):
        async with recorder:
            await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    # 1 original attempt + 2 retries = 3 total, then re-raised through the
    # recorder scope (which stamps partial=true) and caught here.
    assert backend.call_count == 3

    # The trajectory was written and stamped partial=true by the recorder's
    # exception-exit path (TrajectoryRecorder._aborted → _write).
    assert trajectory_path.exists(), "trajectory.json was not written on retry exhaustion"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["extra"]["partial"] is True

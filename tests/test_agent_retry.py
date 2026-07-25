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
from tests.harness.backend import ScriptedBackend


def _fail_then_succeed(
    error: BaseException, *, text: str, partial: str | None = None, **attrs: Any
) -> ScriptedBackend:
    """Attempt 1 emits *partial* (when given) then raises *error*; attempt 2 yields *text*."""
    first: list[Any] = [TextEvent(text=partial)] if partial is not None else []
    first.append(error)
    return ScriptedBackend(
        script=[first, [TextEvent(text=text), ResultEvent(structured_output=None, continuation=None)]],
        **attrs,
    )


def _always_raises(error: BaseException) -> ScriptedBackend:
    """Every attempt raises *error* — the retry-exhaustion and no-retry shapes."""
    return ScriptedBackend(events=[error])


@pytest.mark.parametrize(
    ("make_backend", "expected_output"),
    [
        # First call raises a retryable PiError; second succeeds. Output is from the second call.
        pytest.param(
            lambda: _fail_then_succeed(
                PiError("429 Too Many Requests - rate limit exceeded", retryable=True),
                text="Review complete",
            ),
            "Review complete",
            id="rate-limit",
        ),
        # Partial output from a failed attempt is discarded; only the final output is returned.
        pytest.param(
            lambda: _fail_then_succeed(
                PiError("429 overload", retryable=True),
                text="final text",
                partial="partial text",
            ),
            "final text",
            id="partial-output-discarded",
        ),
        # Stream drop. ``retryable`` comes from the PRODUCTION classifier, mirroring how
        # PiBackend constructs PiError, so this param exercises the real classification
        # path: if ``_is_retryable_error_message("terminated")`` ever returns False,
        # run_agent does NOT retry and this fails.
        pytest.param(
            lambda: _fail_then_succeed(
                PiError("terminated", retryable=_is_retryable_error_message("terminated")),
                text="Review complete after retry",
            ),
            "Review complete after retry",
            id="stream-drop",
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_agent_retries_and_returns_the_successful_attempt(
    monkeypatch, tmp_path: Path, make_backend: Any, expected_output: str
) -> None:
    """A retryable first attempt is re-run; the second attempt's output is what returns."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = make_backend()

    output, _, _ = await run_agent(backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW)

    assert output == expected_output
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_no_retry_on_non_retryable(monkeypatch, tmp_path: Path) -> None:
    """Non-retryable PiError propagates immediately without any retry."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = _always_raises(PiError("auth failed", retryable=False))

    with pytest.raises(PiError, match="auth failed"):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_run_agent_ignores_malformed_retry_environment(monkeypatch, tmp_path: Path) -> None:
    """Malformed Pi retry environment values fall back without blocking a backend call."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "not-an-integer")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "nan")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "inf")
    backend = _fail_then_succeed(
        PiError("429 Too Many Requests - rate limit exceeded", retryable=True),
        text="Review complete",
        retry_attempts=1,
        retry_base_delay_s=0.0,
        retry_max_delay_s=0.0,
    )

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
    # A plain exception with NO ``.category`` (Claude/Codex-style): a human-readable
    # reason plus a secret-shaped substring, to prove the message surfaces AND that
    # secrets are scrubbed at the host boundary.
    backend = _always_raises(RuntimeError("overloaded-502 ZAI_API_KEY=leaked-secret-abc123"))

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
    backend = _always_raises(PiError("429 rate limit", retryable=True))

    with pytest.raises(PiError):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    # 1 original attempt + 2 retries = 3 total
    assert backend.call_count == 3


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
        # retry_attempts read by agent.py via getattr(backend, "retry_attempts", 20)
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
    backend = _always_raises(PiError("429 rate limit", retryable=True))

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

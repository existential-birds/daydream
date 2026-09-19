"""Tests for retry/backoff logic in run_agent (daydream/agent.py).

Every test drives run_agent — the production entrypoint — with a mock backend
that simulates retryable and non-retryable failures. Tests assert on observable
outcomes (returned output, call count) never on internal implementation details.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.pi import PiError, _is_retryable_error_message
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock, patch_retry_sleep


def _fail_then_succeed(
    error: BaseException,
    *,
    text: str,
    partial: str | None = None,
    **attrs: Any,
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_backend: Any,
    expected_output: str,
) -> None:
    """A retryable first attempt is re-run; the second attempt's output is what returns."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = make_backend()

    output, _, _ = await run_agent(backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW)

    assert output == expected_output
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_no_retry_on_non_retryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Non-retryable PiError propagates immediately without any retry."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    backend = _always_raises(PiError("auth failed", retryable=False))

    with pytest.raises(PiError, match="auth failed"):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_run_agent_ignores_malformed_retry_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
async def test_run_agent_uses_backend_retry_policy_without_reading_ambient_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An injected backend policy is complete; ambient retry values are untouched."""
    from daydream.backends import RetryPolicy

    backend = _fail_then_succeed(
        PiError("429 overloaded", retryable=True),
        text="done",
    )
    backend_with_policy: Any = backend
    backend_with_policy.retry_policy = RetryPolicy(
        attempts=1, base_delay_s=0.0, max_delay_s=0.0
    )

    class _ForbiddenEnvironment:
        def get(self, key: str, default: Any = None) -> Any:
            if key.startswith("DAYDREAM_PI_RETRY_"):
                raise AssertionError(f"ambient retry read: {key}")
            return default

    monkeypatch.setattr(
        "daydream.agent.os", SimpleNamespace(environ=_ForbiddenEnvironment()),
    )

    output, _, _ = await run_agent(
        backend, tmp_path, "review", phase=DaydreamPhase.REVIEW
    )

    assert output == "done"
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_surfaces_backend_error_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A categoryless backend error surfaces its MESSAGE to the user, not a bare class name."""
    from rich.console import Console

    rec = Console(file=StringIO(), record=True, force_terminal=True, width=200)
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
async def test_run_agent_retry_exhausted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Always-retryable backend is called max_attempts+1 times total, then raises."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0.01")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "2")
    backend = _always_raises(PiError("429 rate limit", retryable=True))

    with pytest.raises(PiError):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    # 1 original attempt + 2 retries = 3 total
    assert backend.call_count == 3


@pytest.mark.asyncio
async def test_stream_stall_gets_only_one_fresh_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repeated dead-air windows cannot multiply into hours of retries."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    backend = ScriptedBackend(
        events=[StreamStalledError("pi", 300)],
        retry_attempts=5,
        retry_base_delay_s=0,
        retry_max_delay_s=0,
    )

    with pytest.raises(StreamStalledError):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_retry_does_not_kill_sibling_invocations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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

    call_counts: dict[str, int] = {}

    def responder(cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Three named prompt → behaviour mappings on one shared instance.

        - prompt containing "fail-once": retryable PiError on first call, succeeds on retry.
        - prompt containing "ok-a" / "ok-b": always succeeds immediately.

        ``cancel()`` is tracked by the harness; the test asserts it is NOT called during
        retry so that sibling concurrent invocations are unaffected.
        """
        key = (
            "fail-once"
            if "fail-once" in prompt
            else "ok-a"
            if "ok-a" in prompt
            else "ok-b"
        )
        call_counts[key] = call_counts.get(key, 0) + 1
        if key == "fail-once" and call_counts[key] == 1:
            return [PiError("429 overload", retryable=True)]
        return [TextEvent(text=f"done-{key}"), ResultEvent(structured_output=None, continuation=None)]

    backend = ScriptedBackend(
        responder=responder,
        model="test-model",
        fanout_concurrency=3,
        # retry_attempts read by agent.py via getattr(backend, "retry_attempts", 20)
        retry_attempts=3,
        retry_base_delay_s=0.01,
    )

    results: list[tuple[str, str]] = []

    async def _run(prompt: str) -> None:
        output, _, _ = await run_agent(
            cast(Backend, backend), tmp_path, prompt, phase=DaydreamPhase.REVIEW
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
    assert backend.cancel_calls == 0, (
        f"backend.cancel() was called {backend.cancel_calls} time(s) during retry; "
        "this would kill sibling concurrent invocations"
    )

    # The fail-once slot was called twice (fail + retry); others exactly once.
    assert call_counts.get("fail-once", 0) == 2
    assert call_counts.get("ok-a", 0) == 1
    assert call_counts.get("ok-b", 0) == 1


@pytest.mark.asyncio
async def test_run_agent_retry_exhausted_marks_trajectory_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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


class _PermanentWithHigherCap(PiError):
    """A permanent failure that also advertises a per-failure retry cap."""

    max_retries = 5


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        pytest.param(
            PiError("model not found: gpt-5 (503)", retryable=True, category="SERVER_ERROR"),
            1,
            id="permanent-beats-transient-token",
        ),
        pytest.param(
            PiError("invalid api key: not configured", retryable=False, category="AUTH_CONFIG"),
            1,
            id="auth",
        ),
        pytest.param(
            PiError("response failed JSON schema validation", retryable=False, category="SCHEMA"),
            1,
            id="schema",
        ),
        pytest.param(
            _PermanentWithHigherCap("model not found: gpt-5", category="AUTH_CONFIG"),
            1,
            id="cap-override-cannot-raise",
        ),
        pytest.param(
            PiError("429 rate limit exceeded", retryable=True, category="RATE_LIMIT"),
            2,
            id="rate-limit",
        ),
    ],
)
@pytest.mark.asyncio
async def test_failure_classification_decides_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: PiError,
    expected_calls: int,
) -> None:
    """Permanent conditions get zero retries; transient ones still retry."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    backend = (
        ScriptedBackend(events=[error])
        if expected_calls == 1
        else _fail_then_succeed(error, text="done")
    )

    if expected_calls == 1:
        with pytest.raises(type(error)):
            await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)
    else:
        assert (await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW))[0] == "done"

    assert backend.call_count == expected_calls


class _HintError(RuntimeError):
    """A retryable transport failure that can carry a server-provided hint."""

    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@pytest.mark.parametrize(
    ("pinned", "expected"),
    [
        pytest.param("cap", 30.0, id="pinned-to-cap"),
        pytest.param(0.0, 0.0, id="pinned-to-zero"),
        pytest.param("hostile", 30.0, id="hostile-sample-is-clamped"),
    ],
)
@pytest.mark.asyncio
async def test_bounded_full_jitter_never_exceeds_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned: object, expected: float
) -> None:
    """The delay is ``min(sample(0, cap), cap)``: a hostile sampler cannot overshoot.

    ``retry_attempts=1`` pins the ladder to a single retry so the delay list is
    exactly the one computed backoff; ``base_delay_s``/``max_delay_s`` pin the
    exponential cap to 30 s.
    """
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    sampler = {"cap": lambda cap: cap, 0.0: lambda _cap: 0.0, "hostile": lambda cap: cap * 10}[pinned]
    monkeypatch.setattr("daydream.agent._sample_retry_delay", sampler)
    backend = ScriptedBackend(
        events=[_HintError("503 Service Unavailable")],
        retry_attempts=1,
        retry_base_delay_s=30.0,
        retry_max_delay_s=60.0,
    )

    with pytest.raises(_HintError):
        await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            wall_budget_s=10_000.0,
            retry_recovery_allowance_s=300.0,
        )

    assert slept == [pytest.approx(expected)]


@pytest.mark.parametrize(
    ("retry_after", "expected_slept", "stop"),
    [
        pytest.param(30.0, [30.0], None, id="valid-and-fits"),
        pytest.param(600.0, [], "retry_hint_exceeds_budget", id="valid-but-too-long"),
        pytest.param(0.0, [0.0], None, id="zero-is-honoured"),
        pytest.param(None, [30.0], None, id="absent-degrades-to-jitter"),
    ],
)
@pytest.mark.asyncio
async def test_server_retry_hint_is_honoured_and_capped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    retry_after: float | None,
    expected_slept: list[float],
    stop: str | None,
) -> None:
    """A numeric server hint replaces jitter but never extends the budget.

    ``retry_attempts=1`` pins the ladder to a single retry so a fitting hint is
    observable as exactly one sleep and two dispatches. The jitter seam is
    pinned to the cap so an absent hint is distinguishable from a hint.
    """
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    backend = ScriptedBackend(
        events=[_HintError("503 Service Unavailable", retry_after=retry_after)],
        retry_attempts=1,
        retry_base_delay_s=30.0,
        retry_max_delay_s=60.0,
    )

    with pytest.raises(_HintError):
        await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            wall_budget_s=10_000.0,
            retry_recovery_allowance_s=300.0,
        )

    assert slept == [pytest.approx(s) for s in expected_slept]
    if stop is None:
        assert backend.call_count == 2
    else:
        assert backend.call_count == 1  # hint longer than the budget extends nothing


class _MessageHintError(RuntimeError):
    """A retryable failure that carries its hint only in the message token."""

    retryable = True


@pytest.mark.asyncio
async def test_server_retry_hint_is_read_from_the_message_when_the_attribute_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``retry-after: N`` message token is honoured when no attribute is set."""
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    backend = ScriptedBackend(
        events=[_MessageHintError("503 Service Unavailable; retry-after: 45")],
        retry_attempts=1,
        retry_base_delay_s=60.0,
        retry_max_delay_s=60.0,
    )

    with pytest.raises(_MessageHintError):
        await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            wall_budget_s=10_000.0,
            retry_recovery_allowance_s=300.0,
        )

    assert slept == [pytest.approx(45.0)]
    assert backend.call_count == 2


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param("30", id="string"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(-5.0, id="negative"),
    ],
)
@pytest.mark.asyncio
async def test_a_malformed_retry_after_attribute_degrades_to_jitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, malformed: Any
) -> None:
    """A present-but-invalid ``retry_after`` is ignored, never coerced to a delay."""
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    backend = ScriptedBackend(
        events=[_HintError("503 Service Unavailable", retry_after=malformed)],
        retry_attempts=1,
        retry_base_delay_s=30.0,
        retry_max_delay_s=60.0,
    )

    with pytest.raises(_HintError):
        await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            wall_budget_s=10_000.0,
            retry_recovery_allowance_s=300.0,
        )

    assert slept == [pytest.approx(30.0)]  # jitter pinned to the cap, not the malformed value


@pytest.mark.asyncio
async def test_contradictory_retry_budgets_refuse_before_any_dispatch(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        events=[_HintError("503")],
        retry_attempts=3,
        retry_base_delay_s=120.0,
        retry_max_delay_s=2.0,
    )  # would fail retryably if dispatched

    with pytest.raises(ValueError, match="retry_base_delay_s"):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX)

    assert backend.call_count == 0  # refused before dispatch


@pytest.mark.asyncio
async def test_a_non_zero_allowance_with_retries_disabled_is_contradictory(
    tmp_path: Path,
) -> None:
    backend = ScriptedBackend(events=[_HintError("503")], retry_attempts=0)

    with pytest.raises(ValueError, match="retry_recovery_allowance_s"):
        await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            retry_recovery_allowance_s=60.0,
        )

    assert backend.call_count == 0


@pytest.mark.asyncio
async def test_a_zero_allowance_is_the_sanctioned_way_to_disable_retry_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    backend = ScriptedBackend(events=[_HintError("503")], retry_attempts=5)

    with pytest.raises(_HintError):
        await run_agent(
            backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            retry_recovery_allowance_s=0.0,
        )

    assert backend.call_count == 1 and slept == []


@pytest.mark.asyncio
async def test_retry_recovery_allowance_precedence_policy_then_attribute_then_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A superior declared source wins; a zero allowance ends the ladder immediately."""
    # Policy (a complete retry declaration) outranks the backend attribute and
    # the explicit argument: its zero allowance stops after one dispatch.
    policy_backend = ScriptedBackend(
        events=[_HintError("503")],
        retry_attempts=5,
        retry_recovery_allowance_s=300.0,
        retry_policy=SimpleNamespace(
            attempts=5,
            base_delay_s=0.0,
            max_delay_s=0.0,
            retry_recovery_allowance_s=0.0,
        ),
    )

    with pytest.raises(_HintError):
        await run_agent(
            policy_backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            retry_recovery_allowance_s=300.0,
        )
    assert policy_backend.call_count == 1

    # Backend attribute outranks the explicit argument.
    attribute_backend = ScriptedBackend(
        events=[_HintError("503")],
        retry_attempts=5,
        retry_recovery_allowance_s=0.0,
    )
    with pytest.raises(_HintError):
        await run_agent(
            attribute_backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            retry_recovery_allowance_s=300.0,
        )
    assert attribute_backend.call_count == 1


@pytest.mark.asyncio
async def test_retry_recovery_allowance_argument_outranks_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file-config argument beats the ambient env override; the env is the last resort."""
    monkeypatch.setenv("DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S", "0")
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)

    argument_backend = ScriptedBackend(
        events=[_HintError("503")],
        retry_attempts=1,
        retry_base_delay_s=0.0,
        retry_max_delay_s=0.0,
    )
    with pytest.raises(_HintError):
        await run_agent(
            argument_backend,
            tmp_path,
            "go",
            phase=DaydreamPhase.FIX,
            retry_recovery_allowance_s=300.0,
        )
    assert argument_backend.call_count == 2  # the argument's 300s outranked the env's 0s

    # Nothing declared but the env: the env's zero stops after one dispatch.
    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)
    env_backend = ScriptedBackend(events=[_HintError("503")], retry_attempts=5)
    with pytest.raises(_HintError):
        await run_agent(env_backend, tmp_path, "go", phase=DaydreamPhase.FIX)
    assert env_backend.call_count == 1


@pytest.mark.asyncio
async def test_the_retry_policy_allowance_field_governs_the_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The documented top precedence tier is a real field with real effect.

    A policy that declares ``retry_recovery_allowance_s = 0`` ends the ladder after
    one dispatch, and — because a declared ``RetryPolicy`` is complete — the ambient
    env override cannot re-grant recovery behind its back.
    """
    from daydream.backends import RetryPolicy

    monkeypatch.setenv("DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S", "600")
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    patch_retry_sleep(monkeypatch, fake)
    backend = ScriptedBackend(
        events=[_HintError("503")],
        retry_policy=RetryPolicy(
            attempts=5,
            base_delay_s=0.0,
            max_delay_s=0.0,
            retry_recovery_allowance_s=0.0,
        ),
    )

    with pytest.raises(_HintError):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX)

    assert backend.call_count == 1  # the policy's 0 ended the ladder before dispatch 2


@pytest.mark.asyncio
async def test_the_retry_hint_reader_never_raises_on_a_hostile_message() -> None:
    """``_retry_hint`` guards ``str(exc)`` exactly like the shared classifier does."""
    from daydream.agent import _retry_hint
    from daydream.retry_policy import classify_failure

    class _HostileHint(RuntimeError):
        retryable = True

        def __str__(self) -> str:
            raise RuntimeError("no string for you")

    assert _retry_hint(_HostileHint("503")) is None  # no fabricated hint, no raise
    assert classify_failure(_HostileHint()).retries_allowed is True  # still classified


def _resolver_backend(**attrs: Any) -> Any:
    """A minimal backend stand-in for the pure retry-settings resolver.

    Only the attributes the resolver reads are needed, so the extraction stays
    testable without a full ``Backend`` (which would have to be driven to observe
    the same values).
    """
    return SimpleNamespace(model="mock-model", **attrs)


def test_the_extracted_settings_resolver_keeps_the_documented_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_resolve_retry_settings`` is the precedence ladder, testable on its own.

    The retry branch's resolution used to be inline in ``_run_agent``; pinning it
    here keeps the documented order (policy field > backend attribute > argument >
    env > default) observable without driving a ladder.
    """
    from daydream.agent import _resolve_retry_settings
    from daydream.backends import RetryPolicy
    from daydream.config import DEFAULT_RETRY_RECOVERY_ALLOWANCE_S

    def _resolve(backend: Any, explicit: float | None = None) -> Any:
        return _resolve_retry_settings(cast(Backend, backend), explicit)

    monkeypatch.setenv("DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S", "10")

    assert _resolve(_resolver_backend()).allowance_s == 10.0          # env tier
    assert _resolve(_resolver_backend(), 20.0).allowance_s == 20.0    # argument wins
    assert _resolve(_resolver_backend(), 20.0).allowance_declared is True

    attributed = _resolver_backend(retry_recovery_allowance_s=30.0)
    assert _resolve(attributed, 20.0).allowance_s == 30.0             # attribute wins

    policed = _resolver_backend(
        retry_policy=RetryPolicy(
            attempts=3, base_delay_s=1.0, max_delay_s=2.0, retry_recovery_allowance_s=0.0
        )
    )
    resolved = _resolve(policed, 20.0)
    assert resolved.allowance_s == 0.0            # a declared policy is complete
    assert resolved.allowance_declared is True
    assert resolved.max_attempts == 3 and resolved.base_delay_s == 1.0

    # Nothing declared anywhere: the documented default applies, undeclared.
    monkeypatch.delenv("DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S")
    fallback = _resolve(_resolver_backend())
    assert fallback.allowance_s == DEFAULT_RETRY_RECOVERY_ALLOWANCE_S
    assert fallback.allowance_declared is False


def test_the_extracted_settings_resolver_refuses_contradictions() -> None:
    """Both documented contradictions are refused before any dispatch."""
    from daydream.agent import _resolve_retry_settings
    from daydream.backends import RetryPolicy

    inverted = _resolver_backend(
        retry_policy=RetryPolicy(attempts=3, base_delay_s=5.0, max_delay_s=1.0)
    )
    with pytest.raises(ValueError, match="must not exceed"):
        _resolve_retry_settings(cast(Backend, inverted), None)

    disabled = _resolver_backend(
        retry_policy=RetryPolicy(
            attempts=0, base_delay_s=1.0, max_delay_s=2.0, retry_recovery_allowance_s=60.0
        )
    )
    with pytest.raises(ValueError, match="cannot be non-zero while retries are disabled"):
        _resolve_retry_settings(cast(Backend, disabled), None)

    # The default allowance must not turn a legitimate "no retries" declaration into
    # an error: only a *declared* non-zero value contradicts `attempts = 0`.
    plain = _resolver_backend(
        retry_policy=RetryPolicy(attempts=0, base_delay_s=1.0, max_delay_s=2.0)
    )
    assert _resolve_retry_settings(cast(Backend, plain), None).max_attempts == 0


def test_the_extracted_retry_delay_planner_clamps_to_every_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_plan_retry_delay`` decides the delay or the hint stop, on its own."""
    from daydream.agent import _plan_retry_delay

    monkeypatch.setattr("daydream.agent._sample_retry_delay", lambda cap: cap)

    # Cap = min(exponential growth, max delay, remaining allowance, deadline time).
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=60.0, deadline_remaining_s=None, hint=None,
    ) == (10.0, None)
    assert _plan_retry_delay(
        attempt=4, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=60.0, deadline_remaining_s=None, hint=None,
    ) == (60.0, None)
    assert _plan_retry_delay(
        attempt=4, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=None, deadline_remaining_s=7.5, hint=None,
    ) == (7.5, None)
    # A spent deadline is a zero bound, not a negative one.
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=None, deadline_remaining_s=0.0, hint=None,
    ) == (0.0, None)

    # A hint replaces jitter but never extends a budget: over-large stops the ladder.
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=30.0, deadline_remaining_s=None, hint=31.0,
    ) == (0.0, "retry_hint_exceeds_budget")
    # A hint inside the budget still never exceeds the computed cap.
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=300.0, deadline_remaining_s=None, hint=7.0,
    ) == (7.0, None)
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=300.0, deadline_remaining_s=None, hint=30.0,
    ) == (10.0, None)
    # No declared bound at all: the hint is honoured inside the cap, and nothing is
    # fabricated when the server offers none.
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=None, deadline_remaining_s=None, hint=45.0,
    ) == (10.0, None)
    assert _plan_retry_delay(
        attempt=0, base_delay_s=10.0, max_delay_s=120.0,
        allowance_remaining_s=None, deadline_remaining_s=None, hint=None,
    ) == (10.0, None)

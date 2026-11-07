"""Agent retry/backoff behavior and settings precedence."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console

from daydream import agent_retry
from daydream.agent import run_agent
from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.pi import PiBackend, PiError
from daydream.review_budget import BudgetLedger
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.pi_replay import make_mock_process

# The sanitized provider 429 and the healthy follow-up, replayed through the real
# Pi transport by the timing proof below.
_ERROR_MESSAGE = (
    '429: {"message":"Temporary admission failure","code":429,'
    '"metadata":{"headers":{"Retry-After":"-5"," RETRY-AFTER ":10.5}}} retry-after: 99'
)
_ERROR_LINES = (
    '{"type":"session","sessionId":"pi_ses_429"}',
    '{"type":"agent_start"}',
    '{"type":"turn_start"}',
    json.dumps({
        "type": "turn_end",
        "message": {
            "role": "assistant", "content": [], "stopReason": "error", "errorMessage": _ERROR_MESSAGE,
        },
    }),
)
def _fail_then_succeed(error: BaseException, *, text: str, **attrs: Any) -> ScriptedBackend:
    """Attempt 1 raises *error*; attempt 2 yields *text*."""
    return ScriptedBackend(
        script=[[error], [TextEvent(text=text), ResultEvent(structured_output=None, continuation=None)]], **attrs,
    )

async def test_run_agent_ignores_malformed_retry_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "not-an-integer")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "nan")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "inf")
    backend = _fail_then_succeed(
        PiError("429 Too Many Requests - rate limit exceeded", retryable=True), text="Review complete",
        retry_attempts=1, retry_base_delay_s=0.0, retry_max_delay_s=0.0,
    )
    output, _, _ = await run_agent(backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW)
    assert output == "Review complete"
    assert backend.call_count == 2

class _HintError(RuntimeError):
    """A retryable transport failure that can carry a server-provided hint."""

    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

@pytest.mark.parametrize(
    ("pinned", "expected"),
    [
        pytest.param("hostile", 30.0, id="hostile-sample-is-clamped"),
    ],
)
async def test_bounded_full_jitter_never_exceeds_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned: object, expected: float
) -> None:
    """A hostile hint message still reaches one bounded retry with the original error type."""
    class _HostileHint(_HintError):
        def __str__(self) -> str:
            raise RuntimeError("no string for you")
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    sampler = {0.0: lambda _cap: 0.0, "hostile": lambda cap: cap * 10}[pinned]
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", sampler)
    backend = ScriptedBackend(
        events=[_HostileHint("503 Service Unavailable")], retry_attempts=1, retry_base_delay_s=30.0,
        retry_max_delay_s=60.0,
    )
    with pytest.raises(_HostileHint):
        await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX, budget=BudgetLedger("fix", wall_s=10_000.0),
        )
    assert slept == [pytest.approx(expected)]
    assert backend.call_count == 2
    monkeypatch.undo()
    # The real sampler is full jitter within the cap, and a nonpositive cap never sleeps.
    assert agent_retry._sample_retry_delay(0.0) == 0.0 and 0.0 <= agent_retry._sample_retry_delay(1.0) <= 1.0

async def test_zero_server_retry_hint_is_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An explicit zero server wait takes precedence over the nonzero jitter cap."""
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    backend = ScriptedBackend(
        events=[_HintError("503 Service Unavailable", retry_after=0.0)], retry_attempts=1,
        retry_base_delay_s=30.0, retry_max_delay_s=60.0,
    )
    with pytest.raises(_HintError):
        await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX, budget=BudgetLedger("fix", wall_s=10_000.0),
        )
    assert slept == [0.0]
    assert backend.call_count == 2

@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param("30", id="string"), pytest.param(float("nan"), id="nan"),
    ],
)
async def test_a_malformed_retry_after_attribute_degrades_to_jitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, malformed: Any
) -> None:
    """A present-but-invalid ``retry_after`` is ignored, never coerced to a delay."""
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: cap)
    backend = ScriptedBackend(
        events=[_HintError("503 Service Unavailable", retry_after=malformed)], retry_attempts=1,
        retry_base_delay_s=30.0, retry_max_delay_s=60.0,
    )
    with pytest.raises(_HintError):
        await run_agent(
            backend, tmp_path, "go", phase=DaydreamPhase.FIX, budget=BudgetLedger("fix", wall_s=10_000.0),
        )
    assert slept == [pytest.approx(30.0)]  # jitter pinned to the cap, not the malformed value

async def test_contradictory_retry_budgets_refuse_before_any_dispatch(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        events=[_HintError("503")], retry_attempts=3, retry_base_delay_s=120.0, retry_max_delay_s=2.0,
    )  # would fail retryably if dispatched
    with pytest.raises(ValueError, match="retry_base_delay_s"):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX)
    assert backend.call_count == 0  # refused before dispatch

@pytest.mark.parametrize(
    ("second_lines", "wall_s", "retries", "expected_sleep"),
    [
        # The next attempt begins no earlier than the advertised delay, even when the configured
        # jitter maximum (4s) is smaller than the hint (10.5s).
        # An unfittable hint (10.5 > 5): exactly one attempt, one stop reason, no second spawn.
        pytest.param(list(_ERROR_LINES), 5.0, False, [], id="hint-exceeds-budget-stops"),
    ],
)
async def test_run_agent_honours_or_stops_on_a_server_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, second_lines: list[str], wall_s: float,
    retries: bool, expected_sleep: list[Any],
) -> None:
    notices: list[str] = []
    clock = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, clock)
    procs = [make_mock_process(list(_ERROR_LINES)), make_mock_process(list(second_lines))]
    spawned: list[int] = []

    async def _spawn(*_args: Any, **_kwargs: Any) -> Any:
        spawned.append(1)
        return procs.pop(0)

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", _spawn)
    backend = PiBackend(model="glm-5.2")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = 1.0
    backend.retry_max_delay_s = 4.0
    if retries:
        out, _, _ = await run_agent(
            cast(Backend, backend), tmp_path, "p", phase=DaydreamPhase.FIX,
            budget=BudgetLedger("fix", wall_s=wall_s),
            progress_callback=lambda text: notices.append(text.plain.strip()),
        )
        assert out == "done"
        assert len(spawned) == 2  # exactly one retry after the admission failure
    else:
        with pytest.raises(PiError) as failure:
            await run_agent(
                cast(Backend, backend), tmp_path, "p", phase=DaydreamPhase.FIX,
                budget=BudgetLedger("fix", wall_s=wall_s),  # hint 10.5 > the 5s deadline
                progress_callback=lambda text: notices.append(text.plain.strip()),
            )
        assert len(spawned) == 1  # no further attempt dispatched after the stop
        assert failure.value.retry_after == 10.5
        assert failure.value.category == "RATE_LIMIT" and failure.value.retryable is True
    assert slept == expected_sleep  # the full hint, or no shortened wait in its place
    assert [notice for notice in notices if notice.startswith("[retry]")] == ([
        "[retry] Backend error (PiError), retrying attempt 2/2 after 10.5s (server-advertised wait)...",
    ] if retries else [])


class _PermanentWithHigherCap(PiError):
    max_retries = 5


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        pytest.param(StreamStalledError("pi", 1.0), 2, id="stall-cap-below-backend-budget"),
        pytest.param(PiError("model not found: gpt-5 (503)", retryable=True, category="SERVER_ERROR"), 1,
                     id="permanent-beats-transient-token"),
        pytest.param(_PermanentWithHigherCap("model not found: gpt-5", category="AUTH_CONFIG"), 1,
                     id="cap-override-cannot-raise"),
    ],
)
async def test_failure_classification_and_per_error_caps_decide_retries(
    tmp_path: Path, error: Exception, expected_calls: int,
) -> None:
    """A per-error cap bounds retries below the backend budget; permanent categories never retry."""
    backend = ScriptedBackend(events=[error], retry_attempts=3, retry_base_delay_s=0.0, retry_max_delay_s=0.0)
    with pytest.raises(type(error)):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)
    assert backend.call_count == expected_calls


async def test_backend_error_reaches_the_console_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A categoryless backend failure surfaces its message with embedded credentials redacted."""
    recording = Console(file=StringIO(), record=True, force_terminal=True, width=200)
    monkeypatch.setattr("daydream.agent.console", recording)
    backend = ScriptedBackend(events=[RuntimeError("overloaded-502 ZAI_API_KEY=leaked-secret-abc123")],
                              retry_attempts=0)
    context = RunContext(InteractionPolicy())
    with pytest.raises(RuntimeError, match="overloaded-502"):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW, run_context=context)
    out = recording.export_text()
    assert "Backend Execution Error" in out and "overloaded-502" in out
    assert "leaked-secret-abc123" not in out and "[REDACTED_ENV_VAR]" in out
    assert context.active_backends() == ()

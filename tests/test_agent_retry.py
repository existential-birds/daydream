"""Agent retry/backoff behavior and settings precedence."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rich.console import Console

from daydream import agent_retry
from daydream.agent import run_agent
from daydream.backends import Backend
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.pi import PiBackend, PiError
from daydream.extensions.registry import Registry
from daydream.observability.config import ObservabilityConfig
from daydream.observability.runtime import trace_run
from daydream.review_budget import BudgetLedger, BudgetStop
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.pi_replay import make_mock_process

# The sanitized provider 429, replayed through the real Pi transport by the hint proof below.
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

class _HintError(RuntimeError):
    """A retryable transport failure that can carry a server-provided hint."""

    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

class _HostileHint(_HintError):
    def __str__(self) -> str:
        raise RuntimeError("no string for you")


@pytest.mark.parametrize(
    ("error", "sampler", "expected"),
    [
        # A hostile message and an oversized jitter sample still reach one capped retry.
        pytest.param(_HostileHint("503 Service Unavailable"), lambda cap: cap * 10, 30.0, id="hostile-sample"),
        # An explicit zero server wait takes precedence over the nonzero jitter cap.
        pytest.param(_HintError("503", retry_after=0.0), lambda cap: cap, 0.0, id="zero-hint"),
        # A hint above the jitter cap and maximum delay is still waited in full.
        pytest.param(_HintError("503", retry_after=90.0), lambda cap: cap, 90.0, id="hint-over-cap"),
        # A present-but-invalid retry_after is ignored, never coerced to a delay.
        pytest.param(_HintError("503", retry_after="30"), lambda cap: cap, 30.0, id="string-hint"),  # type: ignore[arg-type]
        pytest.param(_HintError("503", retry_after=float("nan")), lambda cap: cap, 30.0, id="nan-hint"),
    ],
)
async def test_retry_delay_honours_valid_hints_and_caps_jitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: _HintError, sampler: Any, expected: float,
) -> None:
    """One retry waits a valid server hint in full, otherwise jitter within the cap."""
    fake = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, fake)
    monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", sampler)
    backend = ScriptedBackend(events=[error], retry_attempts=1, retry_base_delay_s=30.0, retry_max_delay_s=60.0)
    with pytest.raises(type(error)):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX)
    assert slept == [pytest.approx(expected)]
    assert backend.call_count == 2
    monkeypatch.undo()
    # The real sampler is full jitter within the cap, and a nonpositive cap never sleeps.
    assert agent_retry._sample_retry_delay(0.0) == 0.0 and 0.0 <= agent_retry._sample_retry_delay(1.0) <= 1.0

async def test_contradictory_retry_budgets_refuse_before_any_dispatch(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        events=[_HintError("503")], retry_attempts=3, retry_base_delay_s=120.0, retry_max_delay_s=2.0,
    )  # would fail retryably if dispatched
    with pytest.raises(ValueError, match="retry_base_delay_s"):
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX)
    assert backend.call_count == 0  # refused before dispatch

async def test_run_agent_stops_on_a_server_hint_longer_than_the_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An unfittable hint (10.5 s > the 5 s deadline) stops after one attempt with its error intact.

    The exception path still records its one stop on the agent span.
    """
    notices: list[str] = []
    clock = FakeClock(monotonic_value=0.0).install(monkeypatch)
    slept = patch_retry_sleep(monkeypatch, clock)
    spawned: list[int] = []

    async def _spawn(*_args: Any, **_kwargs: Any) -> Any:
        spawned.append(1)
        return make_mock_process(list(_ERROR_LINES))

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", _spawn)
    backend = PiBackend(model="glm-5.2")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = 1.0
    backend.retry_max_delay_s = 4.0
    ledger = BudgetLedger("fix", wall_s=5.0)
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)
    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        with pytest.raises(PiError) as failure:
            await run_agent(
                cast(Backend, backend), tmp_path, "p", phase=DaydreamPhase.FIX, budget=ledger,
                progress_callback=lambda text: notices.append(text.plain.strip()),
            )
        run.finish(1)
    agent_span = next(span for span in exporter.get_finished_spans()
                      if (span.attributes or {}).get("daydream.span.kind") == "agent")
    assert (agent_span.attributes or {})["daydream.budget.dimension"] == "time"
    assert ledger.stopped == BudgetStop("fix", "time", limit=5.0, spent=0.0, remaining_at_dispatch=5.0, reserved=0)
    assert len(spawned) == 1  # no further attempt dispatched after the stop
    assert failure.value.retry_after == 10.5
    assert failure.value.category == "RATE_LIMIT" and failure.value.retryable is True
    assert slept == []  # no shortened wait in place of the hint
    assert [notice for notice in notices if notice.startswith("[retry]")] == []


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

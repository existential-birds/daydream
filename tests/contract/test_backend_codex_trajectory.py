"""ATIF trajectory parity for CodexBackend — golden round-trip + D-04 fallback (#155).

Two contract tests over a real-shape multi-turn Codex JSONL fixture:

1. **Correlation (D-04 fallback):** a multi-turn Codex fixture produces exactly
   one ``MetricsEvent`` per ``turn.completed`` (turn-granular correlation),
   each with ``message_id=''`` (Codex emits no per-message id). This is the
   documented, tested limitation — no silent coarsening.

2. **Golden round-trip:** the recorded trajectory validates cleanly against
   the ATIF v1.6 schema, survives a load → re-validate cycle, and captures
   REASON spans (ThinkingEvent), ACT/tool spans (ToolStart/ToolResult paired
   via the item ``id``), ``Step.metrics`` with prompt/completion tokens, and
   ``cached_tokens`` surfaced from ``usage.cached_input_tokens``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.atif.validator import TrajectoryValidator
from daydream.backends import MetricsEvent
from daydream.backends.codex import CodexBackend
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.codex_replay import make_mock_process_from_fixture

FIXTURE = "multi_turn_with_metrics.jsonl"


async def _drive_codex_through_recorder(
    tmp_path: Path, *, fixture: str = FIXTURE
) -> tuple[list[Any], TrajectoryRecorder]:
    """Drive ``CodexBackend.execute`` through *fixture* while recording.

    Mirrors the recorder + invocation pattern in
    ``tests/contract/test_backend_step_parity.py`` /
    ``_run_backend_against_canonical``: one ``TrajectoryRecorder``, one
    ``Invocation`` scope, and the backend's ``AgentEvent`` stream fed to
    ``inv.observe``. Returns the raw event list (for stream assertions) and
    the recorder (for step assertions).
    """
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="codex-test-model",
        session_id="00000000-0000-0000-0000-000000000155",
    )
    backend = CodexBackend(model="codex-test-model")
    events: list[Any] = []
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            mock_proc = make_mock_process_from_fixture(fixture)
            with patch(
                "daydream.backends.codex.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                async for event in backend.execute(tmp_path, "review"):
                    inv.observe(event)
                    events.append(event)
    return events, recorder


@pytest.mark.asyncio
async def test_codex_emits_one_turn_granular_metrics_event_per_turn(
    tmp_path: Path,
) -> None:
    """D-04 fallback: one MetricsEvent per turn.completed, message_id always ''.

    Codex has no per-message id surface (spike: agent_message items carry
    ``id`` but no ``message_id``; ``turn.completed`` carries only ``usage``),
    so correlation is turn-granular — coarser than Claude's per-message
    correlation. The limitation is named here in code, not silently applied.
    """
    events, _ = await _drive_codex_through_recorder(tmp_path)

    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics_events) == 2, (
        f"expected one MetricsEvent per turn (2 turns), got {len(metrics_events)}"
    )
    for idx, mev in enumerate(metrics_events, start=1):
        assert mev.message_id == "", (
            f"turn {idx}: Codex MetricsEvent.message_id must be '' (D-04), "
            f"got {mev.message_id!r}"
        )
        assert mev.prompt_tokens > 0, (
            f"turn {idx}: prompt_tokens non-positive ({mev.prompt_tokens})"
        )
        assert mev.completion_tokens > 0, (
            f"turn {idx}: completion_tokens non-positive ({mev.completion_tokens})"
        )
    # Distinct token counts prove the two MetricsEvents originate from the two
    # separate turn.completed events rather than a duplicated emission.
    assert metrics_events[0].prompt_tokens != metrics_events[1].prompt_tokens


@pytest.mark.asyncio
async def test_codex_trajectory_golden_round_trip(tmp_path: Path) -> None:
    """Recorded trajectory validates, round-trips, and spans REASON + ACT.

    The fixture carries reasoning items (→ ThinkingEvent → reasoning_content =
    REASON span), ``command_execution`` items with ``id`` fields (→ paired
    ToolStart/ToolResult = ACT/tool span), and ``usage.cached_input_tokens``
    (→ ``Step.metrics.cached_tokens`` non-None).
    """
    _, recorder = await _drive_codex_through_recorder(tmp_path)

    traj_path = tmp_path / "trajectory.json"
    assert traj_path.exists(), "recorder.__aexit__ must write trajectory.json"

    # First validation pass on the freshly-written file.
    validator = TrajectoryValidator()
    first_ok = validator.validate(traj_path)
    assert first_ok, validator.get_errors() or "first validation failed"

    # Round-trip: re-load the written JSON, re-validate from dict form
    # (validate_images=False — no filesystem anchor for in-memory dict).
    raw = json.loads(traj_path.read_text())
    rt_validator = TrajectoryValidator()
    rt_ok = rt_validator.validate(raw, validate_images=False)
    assert rt_ok, rt_validator.get_errors() or "round-trip validation failed"

    agent_steps = [s for s in recorder.steps if s.source == "agent"]
    assert agent_steps, "no agent steps recorded"

    # REASON span: at least one step carries reasoning_content.
    reason_steps = [s for s in agent_steps if s.reasoning_content]
    assert reason_steps, "no REASON span (reasoning_content) captured"

    # ACT/tool span: at least one step carries paired tool_calls + observation.
    act_steps = [
        s
        for s in agent_steps
        if s.tool_calls and s.observation and s.observation.results
    ]
    assert act_steps, "no ACT/tool span (tool_calls + observation) captured"
    # ToolStart/ToolResult paired via the item 'id' field: each observation
    # result's source_call_id must match a tool_call's tool_call_id on the
    # same step (CORE-06).
    for act_step in act_steps:
        observation = act_step.observation
        assert observation is not None and observation.results, (
            f"ACT span on step {act_step.step_id}: observation/results missing"
        )
        call_ids = {tc.tool_call_id for tc in (act_step.tool_calls or [])}
        result_ids = {r.source_call_id for r in observation.results}
        assert result_ids & call_ids, (
            f"ACT span on step {act_step.step_id}: tool result id {result_ids} "
            f"not paired with tool call id {call_ids}"
        )

    # Step.metrics present with prompt/completion tokens (turn-granular, D-04).
    metric_steps = [s for s in agent_steps if s.metrics is not None]
    assert metric_steps, "no step carries metrics"
    for ms in metric_steps:
        metrics = ms.metrics
        assert metrics is not None, f"step {ms.step_id}: metrics is None"
        assert metrics.prompt_tokens is not None, (
            f"step {ms.step_id}: metrics.prompt_tokens is None"
        )
        assert metrics.completion_tokens is not None, (
            f"step {ms.step_id}: metrics.completion_tokens is None"
        )

    # cached_tokens surfaced from usage.cached_input_tokens (non-None).
    cached_steps = [s for s in metric_steps if s.metrics is not None and s.metrics.cached_tokens is not None]
    assert cached_steps, "no step carries non-None cached_tokens"
    for cs in cached_steps:
        metrics = cs.metrics
        assert metrics is not None, f"step {cs.step_id}: metrics is None"
        assert metrics.cached_tokens and metrics.cached_tokens > 0, (
            f"step {cs.step_id}: cached_tokens not positive "
            f"({metrics.cached_tokens})"
        )

"""Codex JSONL must produce one MetricsEvent per turn.completed with empty message_id because no
per-message identity is available. Recorded ATIF v1.7 trajectories must validate after reload and
preserve reasoning, item-ID tool correlation, per-step usage, and cached input tokens.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.atif.validator import TrajectoryValidator
from daydream.backends import MetricsEvent
from daydream.backends._subprocess import StreamStalledError
from daydream.backends.codex import CodexBackend
from daydream.trajectory import DaydreamPhase, TrajectoryRecorder
from tests.harness.codex_replay import GapThenBlockingStdout as _GapThenBlockingStdout, make_mock_process_from_fixture
from tests.harness.trajectory import make_recorder

FIXTURE = "multi_turn_with_metrics.jsonl"


async def _drive_codex_through_recorder(
    tmp_path: Path,
    *,
    fixture: str = FIXTURE,
) -> tuple[list[Any], TrajectoryRecorder]:
    """Drive the fixture through Backend.execute inside one recorder/invocation. Return raw events and
    the recorder for stream and step assertions.
    """
    recorder = make_recorder(
        tmp_path, path=tmp_path / "trajectory.json",
        agent_model_name="codex-test-model", session_id="00000000-0000-0000-0000-000000000155",
    )
    backend = CodexBackend(model="codex-test-model")
    events: list[Any] = []
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            mock_proc = make_mock_process_from_fixture(fixture)
            with patch(
                "daydream.backends._transport.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                async for event in backend.execute(tmp_path, "review"):
                    inv.observe(event)
                    events.append(event)
    return events, recorder


@pytest.mark.asyncio
async def test_codex_trajectory_golden_round_trip(tmp_path: Path) -> None:
    """The fixture covers reasoning_content, command_execution tool/result pairs, and
    cached_input_tokens mapped into Step.metrics.
    """
    events, recorder = await _drive_codex_through_recorder(tmp_path)
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
    # Distinct token counts distinguish separate turns from duplicate MetricsEvent emission.
    assert metrics_events[0].prompt_tokens != metrics_events[1].prompt_tokens


    traj_path = tmp_path / "trajectory.json"
    assert traj_path.exists(), "recorder.__aexit__ must write trajectory.json"

    validator = TrajectoryValidator()  # type: ignore[no-untyped-call]  # vendored atif (untyped)
    first_ok = validator.validate(traj_path)
    assert first_ok, validator.get_errors() or "first validation failed"

    # Revalidate the loaded dict without image checks because it has no filesystem anchor.
    raw = json.loads(traj_path.read_text())
    step_prompt = sum(
        s["metrics"]["prompt_tokens"] for s in raw["steps"]
        if s.get("metrics") and s["metrics"].get("prompt_tokens")
    )
    assert step_prompt == 34594 + 36000
    assert raw["final_metrics"]["total_prompt_tokens"] == step_prompt
    rt_validator = TrajectoryValidator()  # type: ignore[no-untyped-call]  # vendored atif (untyped)
    rt_ok = rt_validator.validate(raw, validate_images=False)
    assert rt_ok, rt_validator.get_errors() or "round-trip validation failed"

    agent_steps = [s for s in recorder.steps if s.source == "agent"]
    assert agent_steps, "no agent steps recorded"

    reason_steps = [s for s in agent_steps if s.reasoning_content]
    assert reason_steps, "no REASON span (reasoning_content) captured"

    act_steps = [
        s
        for s in agent_steps
        if s.tool_calls and s.observation and s.observation.results
    ]
    assert act_steps, "no ACT/tool span (tool_calls + observation) captured"
    # Every linked observation must reference a tool call on the same step.
    for act_step in act_steps:
        observation = act_step.observation
        assert observation is not None and observation.results, (
            f"ACT span on step {act_step.step_id}: observation/results missing"
        )
        call_ids = {tc.tool_call_id for tc in (act_step.tool_calls or [])}
        # Interrupted markers use null source_call_id so coverage consumers cannot count interrupted
        # reads as completed results.
        result_ids = {
            r.source_call_id for r in observation.results if r.source_call_id is not None
        }
        # Require every result ID to match; a nonempty intersection would allow unmatched results.
        assert result_ids.issubset(call_ids), (
            f"ACT span on step {act_step.step_id}: unpaired tool result ids "
            f"{sorted(str(r) for r in (result_ids - call_ids))} "
            f"not present in tool call ids {sorted(str(c) for c in call_ids)}"
        )

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

    cached_steps = [s for s in metric_steps if s.metrics is not None and s.metrics.cached_tokens is not None]
    assert cached_steps, "no step carries non-None cached_tokens"
    for cs in cached_steps:
        metrics = cs.metrics
        assert metrics is not None, f"step {cs.step_id}: metrics is None"
        assert metrics.cached_tokens and metrics.cached_tokens > 0, (
            f"step {cs.step_id}: cached_tokens not positive "
            f"({metrics.cached_tokens})"
        )
    # Reasoning tokens use Metrics.extra because ATIF has no dedicated field. They are a subset of
    # completion tokens, never additive.
    reasoning_steps = [
        s
        for s in metric_steps
        if s.metrics is not None
        and s.metrics.extra is not None
        and s.metrics.extra.get("reasoning_tokens") is not None
    ]
    assert reasoning_steps, "no step carries reasoning_tokens via Metrics.extra"
    for rs in reasoning_steps:
        metrics = rs.metrics
        assert metrics is not None and metrics.extra is not None, (
            f"step {rs.step_id}: metrics/extra missing"
        )
        rt = metrics.extra["reasoning_tokens"]
        assert isinstance(rt, int) and rt > 0, (
            f"step {rs.step_id}: reasoning_tokens not a positive int ({rt})"
        )
        assert metrics.completion_tokens is not None and rt <= metrics.completion_tokens, (
            f"step {rs.step_id}: reasoning_tokens ({rt}) exceeds "
            f"completion_tokens ({metrics.completion_tokens}) — subset invariant"
        )


@pytest.mark.asyncio
async def test_replayable_shell_commands_survive_codex_to_atif_exactly(
    tmp_path: Path,
) -> None:
    """Decoded shell argv are archived exactly and only syntax-checked, never run."""
    expected = [
        'printf "%s\\n" "$HOME"',
        '''sed -n '1,3p' "a file.txt"''',
        'printf "%s" "$(uname -s)"',
        "printf one\nprintf two",
        "cat <<'EOF'\n$HOME\nEOF",
    ]

    _, recorder = await _drive_codex_through_recorder(
        tmp_path,
        fixture="replayable_shell_commands.jsonl",
    )
    archived = [
        call.arguments["command"]
        for step in recorder.steps
        for call in (step.tool_calls or [])
        if call.function_name == "shell"
    ]

    assert archived == expected
    for command in archived:
        # Syntax-check POSIX bodies without requiring the captured wrapper shell on this host.
        checked = subprocess.run(
            ["/bin/sh", "-n", "-c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr


@pytest.mark.asyncio
async def test_codex_diagnostics_survive_atif_validation_and_json_round_trip(
    tmp_path: Path,
) -> None:
    """Conditional transport/parser evidence reaches normalized Step.extra."""
    await _drive_codex_through_recorder(
        tmp_path,
        fixture="parser_coverage_gaps.jsonl",
    )
    trajectory_path = tmp_path / "trajectory.json"
    raw = json.loads(trajectory_path.read_text())
    diagnostics = [
        diagnostic
        for step in raw["steps"]
        for diagnostic in (step.get("extra") or {}).get("backend_diagnostics", [])
    ]

    assert [diagnostic["code"] for diagnostic in diagnostics] == [
        "codex_transport_coverage",
        "codex_parser_coverage",
        "codex_parser_coverage",
    ]
    assert all(set(diagnostic) == {"code", "message", "metadata"} for diagnostic in diagnostics)
    assert diagnostics[0]["metadata"]["occurrences"] == 1
    assert diagnostics[-1]["metadata"]["unknown_event_types"]["total"] == 35
    serialized = json.dumps(raw)
    assert "opaque-parser-secret" not in serialized
    assert "/Users/private-person" not in serialized

    validator = TrajectoryValidator()  # type: ignore[no-untyped-call]
    assert validator.validate(raw, validate_images=False), validator.get_errors()


@pytest.mark.asyncio
async def test_parser_gap_survives_in_partial_trajectory_before_stream_stall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0.01")

    recorder = make_recorder(
        tmp_path, path=tmp_path / "trajectory.json",
        agent_model_name="codex-test-model", session_id="00000000-0000-0000-0000-000000001128",
    )
    backend = CodexBackend(model="codex-test-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    mock_proc.stdout = _GapThenBlockingStdout()

    async with recorder:
        with pytest.raises(StreamStalledError):
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                with patch(
                    "daydream.backends._transport.asyncio.create_subprocess_exec",
                    return_value=mock_proc,
                ):
                    async for event in backend.execute(tmp_path, "review"):
                        invocation.observe(event)

    raw = json.loads((tmp_path / "trajectory.json").read_text())
    diagnostics = [
        diagnostic
        for step in raw["steps"]
        for diagnostic in (step.get("extra") or {}).get("backend_diagnostics", [])
    ]
    assert [diagnostic["code"] for diagnostic in diagnostics] == [
        "codex_parser_coverage"
    ]
    assert diagnostics[0]["metadata"]["unknown_event_types"]["total"] == 1


@pytest.mark.asyncio
async def test_codex_final_metrics_equal_step_sum(tmp_path: Path) -> None:
    """Usage appears in both MetricsEvent and CostEvent; final_metrics must count it once."""
    await _drive_codex_through_recorder(
        tmp_path, fixture="turn_completed_with_usage.jsonl"
    )

    traj = json.loads((tmp_path / "trajectory.json").read_text())
    step_prompt = sum(
        s["metrics"]["prompt_tokens"]
        for s in traj["steps"]
        if s.get("metrics") and s["metrics"].get("prompt_tokens")
    )
    step_completion = sum(
        s["metrics"]["completion_tokens"]
        for s in traj["steps"]
        if s.get("metrics") and s["metrics"].get("completion_tokens")
    )
    assert step_prompt == 200, f"fixture step-sum drifted: {step_prompt}"

    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == step_prompt  # not 400
    assert final["total_completion_tokens"] == step_completion  # not 200


@pytest.mark.asyncio
async def test_issue_1126_failure_status_and_incomplete_marker(tmp_path: Path) -> None:
    """Nonzero exits retain failure metadata. Uncompleted calls receive schema-valid interruption
    markers, and the trajectory validates as ATIF v1.7.
    """
    _, recorder = await _drive_codex_through_recorder(
        tmp_path, fixture="command_failures_issue1126.jsonl"
    )
    steps = [
        s
        for s in recorder.steps
        if s.source == "agent" and s.observation and s.observation.results
    ]
    assert steps, "no agent steps with observation results"

    failed = steps[0].observation.results[0]  # type: ignore[union-attr]  # filtered above
    assert failed.extra == {"is_error": True, "exit_code": 128, "status": "completed"}
    ok = steps[0].observation.results[1]  # type: ignore[union-attr]  # filtered above
    assert ok.extra == {"is_error": False, "exit_code": 0, "status": "completed"}

    dangling = steps[-1].observation.results[-1]  # type: ignore[union-attr]  # filtered above
    assert dangling.source_call_id is None
    assert dangling.content == "[interrupted: call did not complete before invocation ended]"
    assert dangling.extra == {"is_error": True, "status": "interrupted"}

    traj_path = tmp_path / "trajectory.json"
    assert traj_path.exists()
    validator = TrajectoryValidator()  # type: ignore[no-untyped-call]  # vendored atif (untyped)
    assert validator.validate(traj_path), validator.get_errors() or "validation failed"

"""Codex JSONL must produce one MetricsEvent per turn.completed with empty message_id because no
per-message identity is available. Recorded ATIF v1.7 trajectories must validate after reload and
preserve reasoning, item-ID tool correlation, per-step usage, and cached input tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.atif.validator import TrajectoryValidator
from daydream.backends.codex import CodexBackend
from daydream.trajectory import DaydreamPhase, TrajectoryRecorder
from tests.harness.codex_replay import make_mock_process_from_fixture
from tests.harness.trajectory import make_recorder


async def _drive_codex_through_recorder(
    tmp_path: Path,
    *,
    fixture: str,
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
    parser = diagnostics[-1]["metadata"]
    assert parser["unknown_event_types"] == {
        "total": 35, "labels": {f"unknown.{index:02d}": (2 if index == 0 else 1) for index in range(32)}, "overflow": 2,
    }
    assert parser["unknown_item_types"] == {
        "total": 3, "labels": {"mystery.item": 2, 'token="[REDACTED_CREDENTIAL]"': 1}, "overflow": 0,
    }
    assert parser["malformed_shapes"] == {"event_not_object": 2, "event_type_not_scalar": 1, "item_not_object": 1}


async def test_issue_1126_failure_status_and_incomplete_marker(tmp_path: Path) -> None:
    """Nonzero command exits retain failure metadata; uncompleted calls receive schema-valid interruption markers."""
    _, recorder = await _drive_codex_through_recorder(tmp_path, fixture="command_failures_issue1126.jsonl")
    steps = [s for s in recorder.steps if s.source == "agent" and s.observation and s.observation.results]
    assert steps, "no agent steps with observation results"
    failed, ok = steps[0].observation.results[:2]  # type: ignore[union-attr]  # filtered above
    assert failed.extra == {"is_error": True, "exit_code": 128, "status": "completed"}
    assert ok.extra == {"is_error": False, "exit_code": 0, "status": "completed"}
    dangling = steps[-1].observation.results[-1]  # type: ignore[union-attr]  # filtered above
    assert dangling.source_call_id is None
    assert dangling.content == "[interrupted: call did not complete before invocation ended]"
    assert dangling.extra == {"is_error": True, "status": "interrupted"}
    validator = TrajectoryValidator()  # type: ignore[no-untyped-call]  # vendored atif (untyped)
    assert validator.validate(tmp_path / "trajectory.json"), validator.get_errors() or "validation failed"

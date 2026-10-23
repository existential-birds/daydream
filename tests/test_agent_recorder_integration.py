"""Agent/recorder integration: validate persisted trajectories and observable behavior.

The public agent boundary requires an explicit keyword-only phase.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    Backend,
    CostEvent,
    MaxTurnsError,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    RunWriteSnapshot,
    dispatch_scope,
    flush_active_signal_recorders,
    phase_scope,
    run_directory,
    run_document_path,
)
from tests.harness.backend import ScriptedBackend
from tests.harness.codex_lifecycle_cli import install_codex_lifecycle_cli
from tests.harness.processes import wait_for_process_group_gone
from tests.harness.trajectory import make_recorder


async def _run_with_recorder(
    backend: Backend, tmp_path: Path, *, phase: DaydreamPhase = DaydreamPhase.REVIEW,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, prompt: str = "hello",
) -> tuple[dict[str, Any] | None, tuple[Any, Any, Any]]:
    """Drive run_agent inside a TrajectoryRecorder. Return (trajectory_dict, return_value)."""
    recorder = make_recorder(tmp_path, run_flow=run_flow)
    target_path = recorder.path
    async with recorder:
        result = await run_agent(backend, tmp_path, prompt, phase=phase)
    if target_path.exists():
        return json.loads(target_path.read_text()), result
    return None, result


def _scripted(events: list[AgentEvent]) -> ScriptedBackend:
    """ScriptedBackend whose turn ends with an empty structured result."""
    return ScriptedBackend(events=[*events, ResultEvent(structured_output=None, continuation=None)], model="mock-model")


def _single_agent_step(traj: dict[str, Any] | None) -> dict[str, Any]:
    """Validate *traj* and return its sole agent step. Every recorder test that inspects one agent step asserts the
    same two invariants first: the document is schema-valid and exactly one step has ``source == "agent"``."""
    assert traj is not None
    assert atif_validate(traj) is True
    steps: list[dict[str, Any]] = traj["steps"]
    agent_steps = [s for s in steps if s["source"] == "agent"]
    assert len(agent_steps) == 1
    return agent_steps[0]


async def test_agent_events_and_invocations_share_one_recording(tmp_path: Path) -> None:
    """A review then fix preserves event correlation, per-call labels and both usage sources."""
    backend = ScriptedBackend(
        model="mock-model",
        script=[
            [
                ThinkingEvent(text="plan the review"),
                TextEvent(text="reviewing"),
                ToolStartEvent(id="t1", name="Bash", input={"command": "pytest"}),
                ToolResultEvent(id="t1", output="OK", is_error=False),
                MetricsEvent(
                    message_id="msg-1", prompt_tokens=100, completion_tokens=20, cached_tokens=5, cost_usd=0.001
                ),
                ResultEvent(structured_output=None, continuation=None),
            ],
            [
                TextEvent(text="fixing"),
                CostEvent(cost_usd=0.002, input_tokens=200, output_tokens=40, cached_tokens=15),
                ResultEvent(structured_output=None, continuation=None),
            ],
        ],
    )
    recorder = make_recorder(tmp_path, run_flow=DaydreamRunFlow.PR)
    async with recorder:
        assert (await run_agent(backend, tmp_path, "review please", phase=DaydreamPhase.REVIEW))[0] == "reviewing"
        assert (await run_agent(backend, tmp_path, "fix please", phase=DaydreamPhase.FIX))[0] == "fixing"
    trajectory = json.loads(recorder.path.read_text())
    assert atif_validate(trajectory) is True
    users = [step for step in trajectory["steps"] if step["source"] == "user"]
    agents = [step for step in trajectory["steps"] if step["source"] == "agent"]
    assert [step["message"] for step in users] == ["review please", "fix please"]
    assert all(
        step.get("tool_calls") is None
        and step.get("metrics") is None
        and step.get("model_name") is None
        and step.get("reasoning_content") is None
        for step in users
    )
    assert [step["message"] for step in agents] == ["reviewing", "fixing"]
    assert [step["extra"]["daydream_phase"] for step in trajectory["steps"]] == ["review", "review", "fix", "fix"]
    assert all(step["extra"]["daydream_run_flow"] == "pr" for step in trajectory["steps"])
    review = agents[0]
    assert review["reasoning_content"] == "plan the review"
    assert review["tool_calls"][0]["tool_call_id"] == "t1"
    assert review["observation"]["results"][0]["source_call_id"] == "t1"
    assert review["observation"]["results"][0]["content"] == "OK"
    assert [step["metrics"]["prompt_tokens"] for step in agents] == [100, 200]
    assert [step["metrics"]["completion_tokens"] for step in agents] == [20, 40]
    assert [step["metrics"]["cached_tokens"] for step in agents] == [5, 15]
    assert [step["metrics"]["cost_usd"] for step in agents] == [0.001, 0.002]
    final = trajectory["final_metrics"]
    assert (final["total_prompt_tokens"], final["total_completion_tokens"], final["total_cached_tokens"]) == (
        300,
        60,
        20,
    )
    assert final["total_cost_usd"] == pytest.approx(0.003)


@pytest.mark.parametrize("supports_system", [True, False], ids=["absent", "unsupported"])
async def test_host_dispatch_counts_final_user_bytes_without_inventing_unsent_system_bytes(
    tmp_path: Path, supports_system: bool,
) -> None:
    """Persist pre-redaction UTF-8 sizes; unsupported staged policy never becomes sent system text."""
    recorder = make_recorder(tmp_path)
    backend = ScriptedBackend(
        events=[TextEvent(text="done"), ResultEvent(structured_output=None, continuation=None)],
        supports_review_instructions=supports_system,
    )
    options: dict[str, Any] = {} if supports_system else {
        "investigation_budget": ReviewInvestigationBudget.from_limits(ReviewLimits(17, 3, 4)),
        "review_system_instructions": "Unsent system-only marker é🙂",
    }
    async with recorder:
        result = await run_agent(
            backend, tmp_path, "é🙂 API_KEY=private-dispatch-credential", phase=DaydreamPhase.FIX, **options,
        )
    assert result == ("done", None, None)
    trajectory_text = recorder.path.read_text(encoding="utf-8")
    trajectory = json.loads(trajectory_text)
    invocation = trajectory["extra"]["subtrajectories"][0]
    assert invocation["host_dispatch"] == {
        "attempt": 1,
        "user_utf8_bytes": len(backend.calls[0]["prompt"].encode("utf-8")),
        "system_disposition": "absent" if supports_system else "unsupported",
    }
    assert len(backend.calls[0]["prompt"]) < invocation["host_dispatch"]["user_utf8_bytes"]
    assert "private-dispatch-credential" not in trajectory_text
    assert "Unsent system-only marker" not in trajectory_text
    assert "[REDACTED_ENV_VAR]" in trajectory_text


async def test_unencodable_dispatch_measurement_does_not_change_agent_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Encoding failure keeps the backend outcome and never prints failed input or exception text."""
    class UnencodablePrompt(str):
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            raise UnicodeError("PRIVATE_MEASUREMENT_FAILURE")

    backend = _scripted([TextEvent(text="done")])
    trajectory, result = await _run_with_recorder(
        backend, tmp_path, prompt=UnencodablePrompt("API_KEY=private-unencodable-credential"), phase=DaydreamPhase.FIX,
    )
    assert result == ("done", None, None)
    assert trajectory is not None
    assert "host_dispatch" not in trajectory["extra"]["subtrajectories"][0]
    assert _single_agent_step(trajectory)["message"] == "done"
    output = capsys.readouterr().out
    assert "Trajectory dispatch measurement failed: UnicodeError" in output
    assert "PRIVATE_MEASUREMENT_FAILURE" not in output
    assert "private-unencodable-credential" not in output + json.dumps(trajectory)


async def test_calling_run_agent_without_phase_raises_typeerror(tmp_path: Path) -> None:
    backend = _scripted([ TextEvent(text="ok"), ])
    with pytest.raises(TypeError) as excinfo:
        await run_agent(backend, tmp_path, "hi")  # type: ignore[call-arg]
    assert "phase" in str(excinfo.value).lower()
    assert inspect.signature(run_agent).parameters["phase"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_calling_run_agent_with_positional_phase_raises_typeerror(tmp_path: Path) -> None:
    backend = ScriptedBackend(events=[], model="mock-model")
    with pytest.raises(TypeError):
        await run_agent(backend, tmp_path, "hi", DaydreamPhase.REVIEW)  # type: ignore[call-arg]


def _max_turns_backend(pre_events: list[AgentEvent]) -> ScriptedBackend:
    """Replay partial output then MaxTurnsError, matching a mid-turn SDK limit failure."""
    return ScriptedBackend(
        events=[*pre_events, MaxTurnsError("Claude agent run failed: error_max_turns", subtype="error_max_turns")],
        model="mock-model",
    )


async def test_max_turns_error_is_recorded_in_trajectory(tmp_path: Path) -> None:
    """Propagate MaxTurnsError and persist its error marker/subtype on the in-flight step."""
    recorder = make_recorder(tmp_path)
    target_path = recorder.path
    backend = _max_turns_backend([
        TextEvent(text="applying fix"), ToolStartEvent(id="t1", name="Edit", input={"path": "a.py"}),
        ToolResultEvent(id="t1", output="ok", is_error=False),
    ])
    # (a) typed exception propagates through the production entrypoint.
    with pytest.raises(MaxTurnsError) as excinfo:
        async with recorder:
            await run_agent(backend, tmp_path, "fix this", phase=DaydreamPhase.FIX)
    assert excinfo.value.subtype == "error_max_turns"
    # (b) the emitted trajectory carries the error marker + subtype.
    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert atif_validate(traj) is True
    errored = [s for s in traj["steps"] if s.get("extra", {}).get("error_subtype")]
    assert len(errored) == 1
    assert errored[0]["extra"]["error"] is True
    assert errored[0]["extra"]["error_subtype"] == "error_max_turns"
    # The marker lands on the in-flight Step that already held the turn's
    # output — not a synthetic empty step.
    assert errored[0]["message"] == "applying fix"
    assert errored[0]["tool_calls"][0]["tool_call_id"] == "t1"

async def test_active_native_fork_checkpoints_retain_live_lifecycle_without_finishing_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked native attempt must be diagnosable before any finalizer runs.

    Existing process-cancellation tests inspect joined terminal exports, and
    recorder-only partial tests never bind a live backend collector. This real
    CLI/agent/fork boundary detects lifecycle retention deferred until finalization,
    absent active sibling references, and checkpoint sampling that freezes the
    live collector or changes a previous immutable common-cutoff snapshot.
    """
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", mode="preterminal_silence")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    snapshots: list[RunWriteSnapshot] = []
    session = "bdfc09ea-3f5a-4a58-88c9-d24d82742d32"
    root = make_recorder(tmp_path, session_id=session,
                         path=run_document_path(run_directory(tmp_path / ".daydream", session)),
                         on_write=lambda _recorder, snapshot: snapshots.append(snapshot))
    async with root:
        async with dispatch_scope(root, phase=DaydreamPhase.REVIEW, descriptors=["active-native"]) as dispatch:
            assert dispatch is not None

            async def review() -> None:
                async with root.fork("active-native", dispatch=dispatch):
                    async with phase_scope(DaydreamPhase.REVIEW, stage="first-pass"):
                        await run_agent(CodexBackend(model="fixture-model", reasoning_effort="high"),
                                        tmp_path, "synthetic lifecycle request", phase=DaydreamPhase.REVIEW,
                                        wall_budget_s=30, review_stage_scope_id="assigned-scope")

            task = asyncio.create_task(review())
            try:
                observation = await fixture.wait_for_stage("initial")
                # The readiness file belongs to the external process. Wait also
                # for its known public commentary to cross the real host adapter.
                async with asyncio.timeout(5):
                    while True:
                        flush_active_signal_recorders()
                        if snapshots and any(b"Synthetic initial commentary" in document.json_bytes
                                             for document in snapshots[-1].documents):
                            break
                        await asyncio.sleep(0.01)
                first = snapshots[-1]
                first.validate(session)
                first_bytes = tuple(document.json_bytes for document in first.documents)
                documents = {document.trajectory_id: json.loads(document.json_bytes)
                             for document in first.documents}
                parent = documents[session]
                fork = next(row for row in parent["extra"]["subtrajectories"]
                            if row.get("descriptor") == "active-native")
                child = documents[fork["trajectory_id"]]
                assert fork["ended_at"] is None and fork["dispatch_id"] == dispatch.dispatch_id
                assert fork["fork_id"] == child["trajectory_id"]
                assert fork["sibling_trajectory_ref"] == next(
                    document.path.relative_to(tmp_path / ".daydream").as_posix().removesuffix(".partial")
                    for document in first.documents if document.trajectory_id == fork["trajectory_id"])
                invocation = child["extra"]["subtrajectories"][0]
                assert invocation == fork["invocations"][0]
                assert invocation["ended_at"] is None and "generation_lifecycle" not in invocation
                assert invocation["host_dispatch"]["stage_scope_id"] == "assigned-scope"
                lifecycle = invocation["backend_lifecycle"]
                assert lifecycle["backend"] == "codex" and lifecycle["attempt"] == 1
                assert lifecycle["last_native_event_kind"] == "item_completed"
                assert lifecycle["last_native_event_s"] is not None
                assert lifecycle["spawn_completed_s"] is not None and lifecycle["elapsed_s"] > 0
                assert lifecycle["wall_allowance_s"] > 0 and lifecycle["remaining_wall_s"] > 0
                assert lifecycle["terminal_received_s"] is None and lifecycle["stdout_eof_s"] is None
                assert lifecycle["process_returncode"] is None and lifecycle["teardown_started_s"] is None
                assert child["extra"]["phase_events"][0]["metadata"] == {"stage": "first-pass"}
                assert child["final_metrics"].get("total_prompt_tokens") is None
                assert all(atif_validate(payload) for payload in documents.values())
                assert {payload["extra"]["snapshot_at"] for payload in documents.values()} == {first.cutoff_at}
                assert not task.done()

                flush_active_signal_recorders()
                second = snapshots[-1]
                assert second.cutoff_at != first.cutoff_at
                second_child = next(json.loads(document.json_bytes) for document in second.documents
                                    if document.trajectory_id == child["trajectory_id"])
                current = second_child["extra"]["subtrajectories"][0]["backend_lifecycle"]
                assert current["elapsed_s"] > lifecycle["elapsed_s"]
                assert current["last_native_event_s"] == lifecycle["last_native_event_s"]
                assert tuple(document.json_bytes for document in first.documents) == first_bytes
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await wait_for_process_group_gone(observation["pid"], timeout_s=5)

    final = json.loads(root.path.read_text())
    completed = next(row for row in final["extra"]["subtrajectories"] if row.get("descriptor") == "active-native")
    lifecycle = completed["invocations"][0]["backend_lifecycle"]
    assert lifecycle["teardown_completed_s"] is not None
    assert lifecycle["process_exit_s"] is not None and lifecycle["process_returncode"] is not None
    assert lifecycle["elapsed_s"] >= current["elapsed_s"]
    assert completed["invocations"][0]["ended_at"] is not None

"""Tests for daydream/trajectory.py — TrajectoryRecorder + Invocation + Redactor.

Per D-18, tests follow schema-validity + behavior-predicate patterns. Full-tree
snapshot equality is banned (Pitfall 11). Most assertions go through
``daydream.atif.validate()`` plus one or two specific behavioral predicates.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import pytest

import daydream.trajectory as trajectory_module
from daydream.atif import validate as atif_validate
from daydream.atif.models import Step
from daydream.backends import (
    AgentEvent,
    DiagnosticEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    Redactor,
    TrajectoryRecorder,
    _safe_descriptor,
    get_current_recorder,
    now_iso,
    redact_text,
)
from tests.harness.trajectory import (
    make_recorder,
    observe_text_and_result,
    read_trajectory,
)


def test_no_pr_feedback_phase_member() -> None:
    """M2: DaydreamPhase.PR_FEEDBACK is removed; PR flow is kept."""
    assert not hasattr(DaydreamPhase, "PR_FEEDBACK")
    assert DaydreamRunFlow.PR.value == "pr"


async def _drive(tmp_path: Path, *events: AgentEvent, phase: DaydreamPhase = DaydreamPhase.REVIEW) -> dict[str, Any]:
    """Observe *events* in one invocation; return the schema-valid trajectory dict."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=phase) as inv:
            for event in events:
                inv.observe(event)

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    return traj


def _agent_steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """The trajectory's agent-sourced steps, in order."""
    return [s for s in traj["steps"] if s["source"] == "agent"]


def only_dispatch(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Return the sole identified deterministic dispatch step."""
    dispatches = [
        step
        for step in trajectory["steps"]
        if step.get("llm_call_count") == 0 and isinstance(step.get("extra"), dict) and "dispatch_id" in step["extra"]
    ]
    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert isinstance(dispatch, dict)
    return dispatch


def dispatch_refs(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a dispatch's ordered ATIF sibling references."""
    observation = step.get("observation")
    assert isinstance(observation, dict)
    refs: list[dict[str, Any]] = []
    for result in observation["results"]:
        refs.extend(result.get("subagent_trajectory_ref", []))
    return refs


def children_for_dispatch(
    step: dict[str, Any],
    target_dir: Path,
) -> list[dict[str, Any]]:
    """Read only the sibling documents named by one dispatch."""
    return [read_trajectory(target_dir / ".daydream" / ref["trajectory_path"]) for ref in dispatch_refs(step)]


# Behavior 1: TextEvent + ResultEvent → exactly one agent Step with that text


async def test_text_event_then_result_produces_one_agent_step(tmp_path: Path) -> None:
    """Behavior 1: One agent Step from a single TextEvent + ResultEvent."""
    traj = await _drive(
        tmp_path,
        TextEvent(text="Hello world"),
        ResultEvent(structured_output=None, continuation=None),
    )
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "Hello world"


async def test_diagnostic_is_json_safe_redacted_and_persisted_in_arrival_order(
    tmp_path: Path,
) -> None:
    """Diagnostics cross one fail-closed recorder privacy/type boundary."""
    secret = "sk-diagnostic123456"
    unsupported = object()
    _, steps = await _record_events(
        tmp_path,
        DiagnosticEvent(
            code=secret,
            message=f"API_KEY={secret}",
            metadata={
                "nested": {secret: f"TOKEN={secret}", "api_key": secret},
                "unsupported": unsupported,
                "unsupported_key": {42: "retained safely"},
                "non_finite": float("nan"),
            },
        ),
        DiagnosticEvent(code="second", message="safe", metadata={"count": 2}),
        ResultEvent(structured_output=None, continuation=None),
    )

    assert len(steps) == 1
    records = (steps[0].extra or {})["backend_diagnostics"]
    assert [record["code"] for record in records] == ["[REDACTED_API_KEY]", "second"]
    encoded = json.dumps(records, allow_nan=False)
    assert secret not in encoded
    assert "[REDACTED" in encoded
    assert records[0]["metadata"]["unsupported"] == "[UNSUPPORTED_DIAGNOSTIC_VALUE]"
    assert records[0]["metadata"]["unsupported_key"] == {"[UNSUPPORTED_DIAGNOSTIC_KEY]": "retained safely"}
    assert records[0]["metadata"]["non_finite"] == "[UNSUPPORTED_DIAGNOSTIC_VALUE]"
    assert json.loads(encoded) == records


async def test_diagnostic_redaction_failure_persists_fixed_safe_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-neverpersist123456"

    def fail_redaction(value: Any, sensitive: bool = False) -> Any:
        raise RuntimeError("redaction unavailable")

    monkeypatch.setattr("daydream.trajectory.redact_value", fail_redaction)
    _, steps = await _record_events(
        tmp_path,
        DiagnosticEvent(code=secret, message=secret, metadata={"raw": secret}),
        ResultEvent(structured_output=None, continuation=None),
    )

    records = (steps[0].extra or {})["backend_diagnostics"]
    assert records == [
        {
            "code": "diagnostic_redaction_failed",
            "message": "[DIAGNOSTIC_REDACTION_FAILED]",
            "metadata": {},
        }
    ]
    assert secret not in json.dumps(records)


# Behavior 2: Two consecutive TextEvent chunks coalesce into one step (D-03)


async def test_text_event_chunks_coalesce_into_one_step(tmp_path: Path) -> None:
    """Behavior 2: Two TextEvents concatenate into one Step.message (D-03)."""
    traj = await _drive(
        tmp_path,
        TextEvent(text="Hello "),
        TextEvent(text="world"),
        ResultEvent(structured_output=None, continuation=None),
    )
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "Hello world"


# Behavior 2b: ResultEvent flushes accumulated text; new text starts a new step
# (TEST-02 gap fill: explicit flush-on-result-boundary)


async def test_result_event_flushes_text_and_starts_new_step(tmp_path: Path) -> None:
    """TEST-02: ResultEvent terminates the current step; subsequent text starts a new one."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="first chunk"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            inv.observe(TextEvent(text="second chunk"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 2
    assert agent_steps[0]["message"] == "first chunk"
    assert agent_steps[1]["message"] == "second chunk"


# Behavior 3: ToolStart + ToolResult → tool_call & observation in SAME step
# (CORE-06, Pitfall 3)


async def test_tool_call_and_result_land_on_same_step(tmp_path: Path) -> None:
    """Behavior 3: ToolStartEvent + ToolResultEvent both land on same Step."""
    traj = await _drive(
        tmp_path,
        TextEvent(text="Running a tool"),
        ToolStartEvent(id="tool-1", name="Bash", input={"command": "ls"}),
        ToolResultEvent(id="tool-1", output="file1\nfile2", is_error=False),
        ResultEvent(structured_output=None, continuation=None),
    )
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    step = agent_steps[0]
    assert step["tool_calls"][0]["tool_call_id"] == "tool-1"
    assert step["observation"]["results"][0]["source_call_id"] == "tool-1"


# Behavior: ToolResultEvent failure metadata round-trips into
# ObservationResult.extra (issue #1126, Task 3).


async def _record_events(tmp_path: Path, *events: AgentEvent) -> tuple[Invocation, list[Step]]:
    """Observe *events* in one invocation; return it with its materialized steps."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for event in events:
                inv.observe(event)
    return inv, inv.steps


def _single_observation_result(steps: list[Step], index: int) -> Any:
    """The single ObservationResult of *steps[index]*, asserting the shape."""
    step = steps[index]
    assert step.observation is not None
    assert len(step.observation.results) == 1
    return step.observation.results[0]


async def test_result_metadata_round_trips_into_observation_extra(
    tmp_path: Path,
) -> None:
    """exit_code/status on an error ToolResultEvent persist into result extra."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="c1", name="shell", input={"command": "false"}),
        ToolResultEvent(id="c1", output="fatal", is_error=True, exit_code=128, status="completed"),
    )
    result = _single_observation_result(steps, 0)
    assert result.source_call_id == "c1"
    assert result.extra == {"is_error": True, "exit_code": 128, "status": "completed"}


async def test_success_metadata_round_trips_not_just_errors(tmp_path: Path) -> None:
    """Metadata persists on success too — is_error is False, not omitted."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="c2", name="shell", input={"command": "true"}),
        ToolResultEvent(id="c2", output="ok", is_error=False, exit_code=0, status="completed"),
    )
    assert _single_observation_result(steps, 0).extra == {
        "is_error": False,
        "exit_code": 0,
        "status": "completed",
    }


async def test_is_error_only_metadata_round_trips_for_scarce_backends(
    tmp_path: Path,
) -> None:
    """Backends with no structured fields still round-trip is_error."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="c3", name="bash", input={"command": "x"}),
        ToolResultEvent(id="c3", output="err", is_error=True),
    )
    assert _single_observation_result(steps, 0).extra == {"is_error": True}


async def test_late_result_on_closed_step_carries_extra(tmp_path: Path) -> None:
    """The closed-step amendment path persists extra too, not just content."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="c4", name="shell", input={}),
        TurnEndEvent(message_id="m1"),
        ToolResultEvent(id="c4", output="late", is_error=True, exit_code=1, status="completed"),
        ResultEvent(structured_output=None, continuation=None),
    )
    result = _single_observation_result(steps, 0)
    assert result.extra == {"is_error": True, "exit_code": 1, "status": "completed"}


# Behavior: finish() emits incomplete-call markers for in-flight tools
# (issue #1126, Task 4).

INCOMPLETE_CONTENT = "[interrupted: call did not complete before invocation ended]"


async def test_finish_marks_in_flight_tool_on_open_step(tmp_path: Path) -> None:
    """A tool call still in flight when the invocation ends gets a marker observation."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="d1", name="shell", input={"command": "sleep 999"}),
        ResultEvent(structured_output=None, continuation=None),
    )
    result = _single_observation_result(steps, 0)
    # No source_call_id: a marker must never derive as a completed tool call
    # (deep/coverage._completed_read_paths keys on results' source_call_id).
    assert result.source_call_id is None
    assert result.content == INCOMPLETE_CONTENT
    assert result.extra == {"is_error": True, "status": "interrupted"}


async def test_finish_marks_in_flight_tool_on_closed_step(tmp_path: Path) -> None:
    """An in-flight call whose host step closed still gets its marker, amended onto it."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="d2", name="shell", input={}),
        TurnEndEvent(message_id="m1"),
        ResultEvent(structured_output=None, continuation=None),
    )
    result = _single_observation_result(steps, 0)
    assert result.source_call_id is None
    assert result.content == INCOMPLETE_CONTENT
    assert result.extra == {"is_error": True, "status": "interrupted"}


async def test_late_result_before_finish_still_amends_normally(tmp_path: Path) -> None:
    """A result arriving before the final flush amends its step and suppresses the marker."""
    inv, steps = await _record_events(
        tmp_path,
        ToolStartEvent(id="d3", name="shell", input={}),
        TurnEndEvent(message_id="m1"),
        ToolResultEvent(id="d3", output="made", is_error=False, exit_code=0, status="completed"),
        ResultEvent(structured_output=None, continuation=None),
    )
    result = _single_observation_result(steps, 0)
    assert result.content == "made"
    assert result.extra == {"is_error": False, "exit_code": 0, "status": "completed"}


# Regression: interruption markers vs the deep-flow completed-read derivation.
# deep/coverage._completed_read_paths -- shared by the uncovered-file sweep,
# the per-stack verdict evidence gate and diagram-grounding receipts -- treats
# every observation result with a STRING source_call_id as a completed tool
# call, so an interrupted read's marker must carry NO source_call_id: otherwise
# a diff file mid-read at interruption would derive as covered/reviewed and the
# documented fail-open invariant ("an interrupted read must NOT count as
# coverage") flips to fail-closed.


async def test_completed_read_derivation_sees_finished_read(tmp_path: Path) -> None:
    """Positive control: a Read paired with its result IS a completed read."""
    from daydream.deep.coverage import _completed_read_paths

    traj = await _drive(
        tmp_path,
        ToolStartEvent(id="done-read", name="Read", input={"file_path": "src/app.py"}),
        ToolResultEvent(id="done-read", output="print('ok')", is_error=False),
        ResultEvent(structured_output=None, continuation=None),
    )
    assert "src/app.py" in _completed_read_paths(traj)


async def test_interrupted_read_never_completes_in_fork_review_trajectory(
    tmp_path: Path,
) -> None:
    """A Read still in flight at finish() stays derivable-uncovered in a deep-<stack> fork.

    Recording an in-flight Read through the recorder (the exact failure shape
    of budget truncation / backend cancel / CLI death mid-read) and deriving
    completed reads the way the deep-flow consumers do must NOT yield the file:
    fail-open means the sweep still sees it.
    """
    from daydream.deep.coverage import _completed_read_paths

    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as inv:
                inv.observe(ToolStartEvent(id="hung-read", name="Read", input={"file_path": "src/app.py"}))
    # finish() marked the in-flight read interrupted when the invocation exited.
    fork_traj = read_trajectory(child.path)
    assert "src/app.py" not in _completed_read_paths(fork_traj)
    agent_steps = [s for s in fork_traj["steps"] if s["source"] == "agent"]
    results = [r for s in agent_steps for r in (s.get("observation") or {}).get("results") or []]
    assert results == [
        {
            "content": INCOMPLETE_CONTENT,
            "extra": {"is_error": True, "status": "interrupted"},
        }
    ]
    # The marker serializes with NO source_call_id key (null excludes from
    # JSON), so no consumer can derive it as a completed tool call.
    assert "source_call_id" not in results[0]


# Behavior: mark_aborted stamps extra["stop_reason"] on the closing step
# and the trajectory stays schema-valid (Task 2).


async def test_mark_aborted_stamps_stop_reason_on_closing_step(tmp_path: Path) -> None:
    """An aborted invocation closes cleanly with extra['stop_reason'] set."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            inv.observe(ToolStartEvent(id="tool-1", name="Bash", input={"command": "ls"}))
            inv.mark_aborted("budget_exceeded")

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    assert agent_steps[0]["extra"]["stop_reason"] == "budget_exceeded"


# Behavior 4: User step has source="user" and NO agent-only fields
# (Pitfall 4)


async def test_user_step_omits_agent_only_fields(tmp_path: Path) -> None:
    """Behavior 4: observe_user_step produces source='user' with NO agent fields."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="What is the answer?")
            inv.observe(TextEvent(text="42"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    user_steps = [s for s in traj["steps"] if s["source"] == "user"]
    assert len(user_steps) == 1
    user = user_steps[0]
    assert user["message"] == "What is the answer?"
    # Agent-only fields must be absent (not empty / not None — absent from JSON)
    for forbidden in ("model_name", "tool_calls", "metrics", "reasoning_content"):
        assert forbidden not in user, f"User step must not carry agent-only field '{forbidden}'; got {user.keys()}"


# Behavior 5: MetricsEvent attaches Metrics; cached_tokens is a SUBSET (D-15)


async def test_metrics_event_cached_tokens_is_subset_not_added(tmp_path: Path) -> None:
    """Behavior 5: MetricsEvent.cached_tokens is a SUBSET of prompt_tokens (D-15)."""
    metrics_event = MetricsEvent(
        message_id="msg-1",
        prompt_tokens=500,
        completion_tokens=80,
        cached_tokens=100,
        cost_usd=0.01,
    )
    traj = await _drive(
        tmp_path,
        TextEvent(text="thinking..."),
        metrics_event,
        ResultEvent(structured_output=None, continuation=None),
    )
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    metrics = agent_steps[0]["metrics"]
    # Critical: prompt_tokens stays 500 (NOT 600). cached_tokens is reported alongside.
    assert metrics["prompt_tokens"] == 500
    assert metrics["cached_tokens"] == 100
    assert metrics["completion_tokens"] == 80


# Behavior 6: dispatch exception is caught at the recorder boundary; run continues
# (Architecture Q7)


async def test_dispatch_failure_is_caught_and_run_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Behavior 6: Recorder boundary catches dispatch exceptions; run continues."""
    recorder = make_recorder(tmp_path)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="before-failure"))

            # Force a single dispatch failure on the next observe().
            original_dispatch = inv._dispatch
            call_count = {"n": 0}

            def boom(event: Any) -> None:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("simulated dispatch failure")
                original_dispatch(event)

            monkeypatch.setattr(inv, "_dispatch", boom)

            # This call's dispatch raises; observe() must catch it.
            inv.observe(TextEvent(text="will-fail"))
            inv.observe(TextEvent(text="after-failure"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    # "before-failure" + "after-failure" both make it; "will-fail" was dropped.
    assert "before-failure" in agent_steps[0]["message"]
    assert "after-failure" in agent_steps[0]["message"]
    assert "will-fail" not in agent_steps[0]["message"]


# Recorder-level Behavior A: clean exit writes a schema-valid JSON file


async def test_recorder_writes_schema_valid_trajectory_on_clean_exit(
    tmp_path: Path,
) -> None:
    """Behavior A: clean __aexit__ writes a JSON file passing daydream.atif.validate."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hello")
            inv.observe(TextEvent(text="world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert recorder.path.exists()
    assert atif_validate(recorder.path, validate_images=False) is True


# Recorder-level Behavior C: write failure on __aexit__ degrades with warning
# (D-11)


async def test_write_failure_degrades_with_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Behavior C: PermissionError on write emits warning; run does not raise."""
    recorder = make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.trajectory.print_warning", fake_print_warning)
    monkeypatch.setattr(
        "os.replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    # Exit MUST NOT raise even though _write fails inside __aexit__
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert any("Trajectory write failed" in m for m in warnings_emitted)


# Recorder-level Behavior D: FinalMetrics totals are sum of per-step Metrics


async def test_final_metrics_totals_match_per_step_sum(tmp_path: Path) -> None:
    """Behavior D: FinalMetrics totals equal the sum of MetricsEvent values."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="step-one-text"))
            inv.observe(
                MetricsEvent(
                    message_id="m-1",
                    prompt_tokens=100,
                    completion_tokens=20,
                    cached_tokens=10,
                    cost_usd=0.001,
                )
            )
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv2:
            inv2.observe(TextEvent(text="step-two-text"))
            inv2.observe(
                MetricsEvent(
                    message_id="m-2",
                    prompt_tokens=200,
                    completion_tokens=40,
                    cached_tokens=20,
                    cost_usd=0.002,
                )
            )
            inv2.observe(ResultEvent(structured_output=None, continuation=None))

    traj = read_trajectory(recorder.path)
    fm = traj["final_metrics"]
    assert fm["total_prompt_tokens"] == 300
    assert fm["total_completion_tokens"] == 60
    assert fm["total_cached_tokens"] == 30
    assert abs(fm["total_cost_usd"] - 0.003) < 1e-9
    assert fm["total_steps"] == len(traj["steps"])


# Recorder-level Behavior E: ContextVar is set inside, cleared after


async def test_context_var_set_inside_and_cleared_after(tmp_path: Path) -> None:
    """Behavior E: get_current_recorder is the recorder inside, None after."""
    assert get_current_recorder() is None
    recorder = make_recorder(tmp_path)
    async with recorder:
        assert get_current_recorder() is recorder
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    assert get_current_recorder() is None


# Recorder-level Behavior F (CORE-08): Trajectory.agent identity baked in


async def test_trajectory_agent_identity_is_daydream(tmp_path: Path) -> None:
    """Behavior F (CORE-08): agent.name='daydream', version non-empty, model_name passed-in."""
    traj = await _drive(
        tmp_path,
        TextEvent(text="hi"),
        ResultEvent(structured_output=None, continuation=None),
    )
    assert traj["agent"]["name"] == "daydream"
    assert isinstance(traj["agent"]["version"], str) and traj["agent"]["version"]
    assert traj["agent"]["model_name"] == "opus"


# Bonus: schema_version + session_id present and well-formed


async def test_schema_version_and_session_id_present(tmp_path: Path) -> None:
    """schema_version pinned to ATIF-v1.7; session_id and trajectory_id present."""
    traj = await _drive(
        tmp_path,
        TextEvent(text="hi"),
        ResultEvent(structured_output=None, continuation=None),
    )
    assert traj["schema_version"] == "ATIF-v1.7"
    assert isinstance(traj["session_id"], str)
    assert len(traj["session_id"]) > 0
    # v1.7: root trajectory carries a per-document trajectory_id. With no fork
    # descriptor it equals the run-scoped session_id.
    assert traj["trajectory_id"] == traj["session_id"]


async def test_agent_step_carries_llm_call_count_one(tmp_path: Path) -> None:
    """v1.7: a real assistant turn records llm_call_count == 1."""
    traj = await _drive(
        tmp_path,
        TextEvent(text="Hello world"),
        ResultEvent(structured_output=None, continuation=None),
    )
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    assert agent_steps[0]["llm_call_count"] == 1


async def test_dispatch_step_is_deterministic_zero_llm_calls(tmp_path: Path) -> None:
    """v1.7 no-LLM-orchestration rule: the fan-out dispatch step is a
    deterministic (non-LLM) step — llm_call_count == 0 and no metrics /
    reasoning_content. The vendored validator enforces this, so a passing
    ``atif_validate`` plus the field assertions prove the constraint holds.
    """
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with trajectory_module.dispatch_scope(
            recorder, phase=DaydreamPhase.FIX, descriptors=["fix-0"],
        ) as dispatch:
            async with recorder.fork("fix-0", dispatch=dispatch) as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    observe_text_and_result(inv)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    dispatch_steps = [s for s in traj["steps"] if s["source"] == "agent" and "Dispatching" in s.get("message", "")]
    assert len(dispatch_steps) == 1
    dispatch = dispatch_steps[0]
    assert dispatch["llm_call_count"] == 0
    assert "metrics" not in dispatch
    assert "reasoning_content" not in dispatch


async def _run_scoped_dispatch_child(
    recorder: TrajectoryRecorder,
    dispatch: Any,
    descriptor: str,
    entered: anyio.Event,
    release: anyio.Event,
    completed: anyio.Event,
) -> None:
    async with trajectory_module.maybe_fork(
        recorder,
        descriptor,
        dispatch=dispatch,
    ) as child:
        entered.set()
        await release.wait()
        async with child.invocation(phase=DaydreamPhase.FIX) as inv:
            observe_text_and_result(inv, descriptor)
    completed.set()


async def test_dispatch_interval_encloses_children_in_declared_order(
    tmp_path: Path,
) -> None:
    """A post-materialized dispatch retains its start and declared child order."""
    assert hasattr(trajectory_module, "dispatch_scope")
    recorder = make_recorder(tmp_path)
    entered = {name: anyio.Event() for name in ("alpha", "beta")}
    release = {name: anyio.Event() for name in entered}
    completed = {name: anyio.Event() for name in entered}

    async with recorder:
        async with trajectory_module.dispatch_scope(
            recorder,
            phase=DaydreamPhase.FIX,
            descriptors=("alpha", "beta"),
        ) as dispatch:
            assert dispatch is not None
            async with anyio.create_task_group() as task_group:
                for name in ("alpha", "beta"):
                    task_group.start_soon(
                        _run_scoped_dispatch_child,
                        recorder,
                        dispatch,
                        name,
                        entered[name],
                        release[name],
                        completed[name],
                    )
                await entered["alpha"].wait()
                await entered["beta"].wait()
                release["beta"].set()
                await completed["beta"].wait()
                release["alpha"].set()

    root = read_trajectory(recorder.path)
    dispatch_step = only_dispatch(root)
    children = children_for_dispatch(dispatch_step, recorder.target_dir)
    refs = dispatch_refs(dispatch_step)
    assert [result["content"] for result in dispatch_step["observation"]["results"]] == [
        "Dispatched to alpha",
        "Dispatched to beta",
    ]
    assert [ref["trajectory_id"] for ref in refs] == [child["trajectory_id"] for child in children]
    assert dispatch_step["timestamp"] <= min(child["extra"]["run_started_at"] for child in children)
    assert dispatch_step["extra"]["dispatch_completed_at"] >= max(child["extra"]["run_ended_at"] for child in children)
    assert dispatch_step["extra"] == {
        "daydream_phase": "fix",
        "daydream_run_flow": "normal",
        "dispatch_id": f"{recorder.session_id}:dispatch:1",
        "dispatch_started_at": dispatch_step["timestamp"],
        "dispatch_completed_at": dispatch_step["extra"]["dispatch_completed_at"],
        "planned_count": 2,
        "attempted_count": 2,
        "completed_count": 2,
        "dispatch_status": "succeeded",
    }
    assert root["extra"]["run_started_at"] <= dispatch_step["timestamp"]
    assert root["extra"]["run_ended_at"] >= dispatch_step["extra"]["dispatch_completed_at"]
    assert atif_validate(root, validate_images=False) is True
    assert all(atif_validate(child, validate_images=False) for child in children)


async def test_dispatch_repeated_descriptor_scopes_keep_distinct_children(
    tmp_path: Path,
) -> None:
    """Repeated semantic labels cannot overwrite or migrate across dispatches."""
    assert hasattr(trajectory_module, "dispatch_scope")
    recorder = make_recorder(tmp_path)
    child_paths: list[Path] = []
    async with recorder:
        for output in ("first", "second"):
            async with trajectory_module.dispatch_scope(
                recorder,
                phase=DaydreamPhase.FIX,
                descriptors=("repeat",),
            ) as dispatch:
                assert dispatch is not None
                async with trajectory_module.maybe_fork(
                    recorder,
                    "repeat",
                    dispatch=dispatch,
                ) as child:
                    child_paths.append(child.path)
                    async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                        observe_text_and_result(inv, output)

    root = read_trajectory(recorder.path)
    dispatches = [
        step for step in root["steps"] if step.get("llm_call_count") == 0 and "dispatch_id" in step.get("extra", {})
    ]
    assert len(dispatches) == 2
    assert len(set(child_paths)) == 2
    first_ref, second_ref = (dispatch_refs(step)[0] for step in dispatches)
    assert first_ref["trajectory_path"] != second_ref["trajectory_path"]
    assert first_ref["trajectory_id"] != second_ref["trajectory_id"]
    assert [step["extra"]["dispatch_id"] for step in dispatches] == [
        f"{recorder.session_id}:dispatch:1",
        f"{recorder.session_id}:dispatch:2",
    ]
    assert "first" in json.dumps(read_trajectory(child_paths[0]))
    assert "second" in json.dumps(read_trajectory(child_paths[1]))


async def test_dispatch_cancellation_overrides_optimistic_terminal(
    tmp_path: Path,
) -> None:
    """Cancellation propagates while closing the dispatch with a stable code."""
    assert hasattr(trajectory_module, "dispatch_scope")
    recorder = make_recorder(tmp_path)
    async with recorder:
        with anyio.CancelScope() as cancel_scope:
            async with trajectory_module.dispatch_scope(
                recorder,
                phase=DaydreamPhase.FIX,
                descriptors=("cancelled-child",),
            ) as dispatch:
                assert dispatch is not None
                dispatch.finish(trajectory_module.LifecycleStatus.SUCCEEDED)
                async with trajectory_module.maybe_fork(
                    recorder,
                    "cancelled-child",
                    dispatch=dispatch,
                ) as child:
                    async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                        observe_text_and_result(inv, "before cancellation")
                    cancel_scope.cancel()
                    await anyio.sleep_forever()

    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["dispatch_status"] == "cancelled"
    assert step["extra"]["reason_code"] == "cancelled"


async def test_dispatch_cancellation_overrides_empty_child_write_failure(
    tmp_path: Path,
) -> None:
    """Cancellation remains authoritative when an empty child cannot be written."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        with anyio.CancelScope() as cancel_scope:
            async with trajectory_module.dispatch_scope(
                recorder,
                phase=DaydreamPhase.FIX,
                descriptors=("empty-child",),
            ) as dispatch:
                assert dispatch is not None
                async with trajectory_module.maybe_fork(
                    recorder,
                    "empty-child",
                    dispatch=dispatch,
                ):
                    pass
                cancel_scope.cancel()
                await anyio.sleep_forever()

    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["dispatch_status"] == "cancelled"
    assert step["extra"]["reason_code"] == "cancelled"
    assert step["extra"]["attempted_count"] == 1
    assert step["extra"]["completed_count"] == 0


async def test_dispatch_exception_overrides_empty_child_write_failure(
    tmp_path: Path,
) -> None:
    """An escaping exception retains its cause when an empty child also failed."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        with pytest.raises(RuntimeError, match="dispatch body failed"):
            async with trajectory_module.dispatch_scope(
                recorder,
                phase=DaydreamPhase.FIX,
                descriptors=("empty-child",),
            ) as dispatch:
                assert dispatch is not None
                async with trajectory_module.maybe_fork(
                    recorder,
                    "empty-child",
                    dispatch=dispatch,
                ):
                    pass
                raise RuntimeError("dispatch body failed")

    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["dispatch_status"] == "failed"
    assert step["extra"]["reason_code"] == "uncaught_exception"
    assert step["extra"]["attempted_count"] == 1
    assert step["extra"]["completed_count"] == 0


async def test_recursive_invocation_identity_has_no_fork_wrapper_call(
    tmp_path: Path,
) -> None:
    """A completed fork propagates every document-qualified nested invocation."""
    assert hasattr(trajectory_module, "dispatch_scope")
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with trajectory_module.dispatch_scope(
            recorder,
            phase=DaydreamPhase.REVIEW,
            descriptors=("outer",),
        ) as outer_dispatch:
            assert outer_dispatch is not None
            async with trajectory_module.maybe_fork(
                recorder,
                "outer",
                dispatch=outer_dispatch,
            ) as child:
                async with trajectory_module.phase_scope(DaydreamPhase.REVIEW):
                    for text in ("outer-one", "outer-two"):
                        async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                            observe_text_and_result(inv, text)
                async with trajectory_module.dispatch_scope(
                    child,
                    phase=DaydreamPhase.FIX,
                    descriptors=("nested",),
                ) as nested_dispatch:
                    assert nested_dispatch is not None
                    async with trajectory_module.maybe_fork(
                        child,
                        "nested",
                        dispatch=nested_dispatch,
                    ) as grandchild:
                        async with trajectory_module.phase_scope(DaydreamPhase.FIX):
                            async with grandchild.invocation(phase=DaydreamPhase.FIX) as inv:
                                observe_text_and_result(inv, "nested-one")

    root = read_trajectory(recorder.path)
    outer_summary = root["extra"]["subtrajectories"][0]
    invocations = outer_summary["invocations"]
    identities = {(invocation["trajectory_id"], invocation["invocation_id"]) for invocation in invocations}
    assert len(invocations) == 3
    assert len(identities) == 3
    assert "invocation_id" not in outer_summary
    assert outer_summary["dispatch_id"] == f"{recorder.session_id}:dispatch:1"
    assert {event["trajectory_id"] for event in outer_summary["phase_events"]} == {
        child.trajectory_id,
        grandchild.trajectory_id,
    }
    assert all(event["scope_id"] for event in outer_summary["phase_events"])


async def test_lifecycle_reason_redaction_omits_exception_details(
    tmp_path: Path,
) -> None:
    """Lifecycle evidence stores closed codes, never backend exception text."""
    assert hasattr(trajectory_module, "dispatch_scope")
    secret = "token=p07-secret https://user:pass@example.invalid"
    recorder = make_recorder(tmp_path)
    child_path: Path
    async with recorder:
        with pytest.raises(RuntimeError, match="p07-secret"):
            async with trajectory_module.phase_scope(DaydreamPhase.DEEP) as phase:
                phase.finish(trajectory_module.LifecycleStatus.SUCCEEDED)
                async with trajectory_module.dispatch_scope(
                    recorder,
                    phase=DaydreamPhase.DEEP,
                    descriptors=("safe-child",),
                ) as dispatch:
                    assert dispatch is not None
                    dispatch.finish(trajectory_module.LifecycleStatus.SUCCEEDED)
                    async with trajectory_module.maybe_fork(
                        recorder,
                        "safe-child",
                        dispatch=dispatch,
                    ) as child:
                        child_path = child.path
                        async with child.invocation(phase=DaydreamPhase.DEEP) as inv:
                            observe_text_and_result(inv, "safe output")
                    raise RuntimeError(secret)

    root = read_trajectory(recorder.path)
    child_data = read_trajectory(child_path)
    serialized = json.dumps({"root": root, "child": child_data}, sort_keys=True)
    assert "p07-secret" not in serialized
    assert "user:pass" not in serialized
    dispatch_step = only_dispatch(root)
    assert dispatch_step["extra"]["dispatch_status"] == "failed"
    assert dispatch_step["extra"]["reason_code"] == "uncaught_exception"
    terminal = root["extra"]["phase_events"][-1]
    assert terminal["status"] == "failed"
    assert terminal["reason_code"] == "uncaught_exception"


def test_lifecycle_snapshot_value_types_are_frozen(tmp_path: Path) -> None:
    """Task 1 freezes the immutable payload types consumed by Task 5."""
    assert hasattr(trajectory_module, "TrajectoryDocumentSnapshot")
    document = trajectory_module.TrajectoryDocumentSnapshot(
        trajectory_id="root",
        path=tmp_path / "trajectory.json",
        json_bytes=b"{}",
    )
    snapshot = trajectory_module.RunWriteSnapshot(
        status="partial",
        cutoff_at="2026-01-01T00:00:00Z",
        root_trajectory_id="root",
        documents=(document,),
    )
    assert snapshot.documents == (document,)
    with pytest.raises(AttributeError):
        setattr(snapshot, "cutoff_at", "changed")


async def test_fork_child_trajectory_id_distinct_from_root(tmp_path: Path) -> None:
    """v1.7: a fork's per-document trajectory_id is descriptor-qualified so it
    is distinct from the shared run-scoped session_id, and the sibling ref on
    the parent carries that canonical trajectory_id as its resolution key (not
    just the run-scoped session_id) alongside the external trajectory_path.
    """
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                observe_text_and_result(inv)
            child_path = child.path
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    child_traj = read_trajectory(child_path)
    assert atif_validate(child_traj, validate_images=False) is True
    assert child_traj["session_id"] == recorder.session_id
    assert child_traj["trajectory_id"] == f"{recorder.session_id}:fix-0"

    parent_traj = read_trajectory(recorder.path)
    ref = parent_traj["extra"]["subtrajectories"][0]
    assert ref["sibling_trajectory_ref"].startswith("runs/")
    # v1.7 resolution key: the ref points at the sibling's canonical
    # per-document trajectory_id, and session_id stays as informational run
    # identity only (shared with the parent, not a matching key).
    assert ref["trajectory_id"] == child_traj["trajectory_id"]
    assert ref["invocations"][0]["trajectory_id"] == child_traj["trajectory_id"]
    assert "invocation_id" not in ref
    assert (tmp_path / ".daydream" / ref["sibling_trajectory_ref"]) == child_path


# Sanity: now_iso, Redactor, Invocation public surface


def test_now_iso_ends_with_z() -> None:
    """now_iso returns ISO 8601 string ending in 'Z' (Pitfall 2)."""
    ts = now_iso()
    assert ts.endswith("Z")


def test_redactor_is_passthrough() -> None:
    """Redactor.redact_step preserves semantic equality on clean inputs.

    Phase 4: Redactor returns a fresh model_copy whenever ANY scannable text
    field is present. Identity (`out is step`) is NO LONGER guaranteed; the
    contract is field-by-field semantic equality on inputs containing no
    secret patterns.
    """
    from daydream.atif import Step as AtifStep

    step = AtifStep(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="hello",
    )
    out = Redactor().redact_step(step)
    assert out.message == step.message
    assert out.reasoning_content == step.reasoning_content
    assert out.tool_calls == step.tool_calls
    assert out.observation == step.observation
    assert out.extra == step.extra


def test_empty_secret_assignment_does_not_consume_the_following_line() -> None:
    """Redaction never deletes a line it mistook for a secret's value.

    ``API_KEY=`` at end of line used to match the *next* line as its value and
    the replacement dropped the newline, silently deleting real content.
    """
    block = (
        "CLERK_SECRET_KEY=\n"
        "CLOUDFLARE_ACCOUNT_ID=\n"
        "CLOUDFLARE_API_TOKEN=\n"
        "CLOUDFLARE_ACCOUNT_HASH=\n"
        "INTERNAL_SERVICE_SECRET="
    )

    assert redact_text(block) == block


def test_secret_value_on_the_same_line_is_still_redacted() -> None:
    assert redact_text("API_KEY= sk-live-abc123") == "API_KEY=[REDACTED_ENV_VAR]"
    assert redact_text("TOKEN=abc\nPLAIN_SETTING=1") == "TOKEN=[REDACTED_ENV_VAR]\nPLAIN_SETTING=1"


def test_public_text_redactor_is_fail_closed() -> None:
    secret = "OPENAI_API_KEY=sk-secret123456"

    assert redact_text(secret) == "OPENAI_API_KEY=[REDACTED_ENV_VAR]"

    class _FailingPattern:
        def sub(self, replacement: str, value: str) -> str:
            raise RuntimeError("synthetic redaction failure")

    with patch(
        "daydream.trajectory._REDACTION_RULES",
        ((_FailingPattern(), "[REDACTED]"),),
    ):
        assert redact_text(secret) == "[REDACTION_FAILED]"


def test_invocation_has_no_parent_field() -> None:
    """D-08: Invocation does not carry parent; parent linkage is on TrajectoryRecorder."""
    fields = {f.name for f in Invocation.__dataclass_fields__.values()}
    assert "parent" not in fields, f"Invocation must not carry a parent field in Phase 2 (D-08); got {fields}"


# No-recorder no-op (CORE-09)


def test_no_recorder_no_op_get_current_returns_none() -> None:
    """CORE-09: outside any recorder context, get_current_recorder is None."""
    assert get_current_recorder() is None


# compute_wall_clock_seconds: derived from step timestamps, no --eval needed


def _append_step_at(recorder: TrajectoryRecorder, ts: str) -> None:
    """Append a minimal agent step with an explicit ISO-8601 timestamp."""
    from daydream.atif import Step as AtifStep

    recorder.steps.append(
        AtifStep(
            step_id=recorder._next_step_id(),
            timestamp=ts,
            source="agent",
            message="x",
        )
    )


def test_compute_wall_clock_seconds_spans_first_to_last(tmp_path: Path) -> None:
    """Span is max(timestamp) - min(timestamp), in seconds, regardless of insertion order."""
    recorder = make_recorder(tmp_path)
    _append_step_at(recorder, "2026-05-31T10:00:00.000000Z")
    _append_step_at(recorder, "2026-05-31T10:00:07.500000Z")
    _append_step_at(recorder, "2026-05-31T10:00:03.000000Z")

    assert recorder.compute_wall_clock_seconds() == 7.5


def test_compute_wall_clock_seconds_single_step_is_none(tmp_path: Path) -> None:
    """Fewer than two timestamped steps means no measurable span."""
    recorder = make_recorder(tmp_path)
    _append_step_at(recorder, "2026-05-31T10:00:00.000000Z")

    assert recorder.compute_wall_clock_seconds() is None


def test_compute_wall_clock_seconds_no_steps_is_none(tmp_path: Path) -> None:
    """An empty recorder yields None rather than raising."""
    recorder = make_recorder(tmp_path)

    assert recorder.compute_wall_clock_seconds() is None


# Fork / Sibling / Continuation tests (Phase 3, SUBA-01..09)


# SUBA-07: ContextVar isolation inside fork


async def test_fork_contextvar_isolation(tmp_path: Path) -> None:
    """SUBA-07: Inside fork scope get_current_recorder() returns child; outside returns parent."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        assert get_current_recorder() is recorder
        async with recorder.fork("fix-0") as child:
            assert get_current_recorder() is child
            assert get_current_recorder() is not recorder
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                observe_text_and_result(inv)
        assert get_current_recorder() is recorder


# SUBA-06: Sibling inherits session_id


async def test_sibling_inherits_session_id(tmp_path: Path) -> None:
    """SUBA-06: Child trajectory file has same session_id as parent."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                observe_text_and_result(inv)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    parent_traj = read_trajectory(recorder.path)
    sibling_path = child.path
    sibling_traj = read_trajectory(sibling_path)
    assert parent_traj["session_id"] == sibling_traj["session_id"]
    assert parent_traj["session_id"] == recorder.session_id


# SUBA-06: Sibling file path format


async def test_sibling_file_path_format(tmp_path: Path) -> None:
    """SUBA-06: Sibling path is <target>/.daydream/runs/<session_id>/trajectories/<descriptor>.json."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as inv:
                observe_text_and_result(inv)

    expected = tmp_path / ".daydream" / "runs" / recorder.session_id / "trajectories" / "deep-python.json"
    assert child.path == expected
    assert expected.exists()


# SUBA-08: Step ID isolation across siblings


async def test_step_id_isolation_across_siblings(tmp_path: Path) -> None:
    """SUBA-08: Parent and child step_ids both start from 1 independently."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "parent-step")
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                observe_text_and_result(inv, "child-step")

    parent_traj = read_trajectory(recorder.path)
    child_traj = read_trajectory(child.path)

    parent_ids = [s["step_id"] for s in parent_traj["steps"]]
    child_ids = [s["step_id"] for s in child_traj["steps"]]

    assert parent_ids == list(range(1, len(parent_ids) + 1))
    assert child_ids == list(range(1, len(child_ids) + 1))
    assert child_ids[0] == 1


# SUBA-09 (superseded): parent FinalMetrics are fork-inclusive


async def test_parent_metrics_include_children(tmp_path: Path) -> None:
    """Parent FinalMetrics totals fold in child totals; the child file keeps its own.

    Supersedes the original SUBA-09 expectation (parent excludes children): the
    root trajectory is now whole-run truth so manifest and eval consumers read
    one number instead of re-summing sibling files. The parent's own share stays
    recoverable from its per-step metrics.
    """
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent-text"))
            inv.observe(
                MetricsEvent(
                    message_id="m-parent",
                    prompt_tokens=100,
                    completion_tokens=10,
                    cached_tokens=5,
                    cost_usd=0.001,
                )
            )
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe(TextEvent(text="child-text"))
                inv.observe(
                    MetricsEvent(
                        message_id="m-child",
                        prompt_tokens=200,
                        completion_tokens=20,
                        cached_tokens=10,
                        cost_usd=0.002,
                    )
                )
                inv.observe(ResultEvent(structured_output=None, continuation=None))

    parent_traj = read_trajectory(recorder.path)
    child_traj = read_trajectory(child.path)

    assert parent_traj["final_metrics"]["total_prompt_tokens"] == 300
    assert child_traj["final_metrics"]["total_prompt_tokens"] == 200

    # The parent's own share is still distinguishable from the folded total.
    own = sum(
        s["metrics"]["prompt_tokens"]
        for s in parent_traj["steps"]
        if s.get("metrics") and s["metrics"].get("prompt_tokens")
    )
    assert own == 100


# Dispatch step uses relative path (starts with "trajectories/")


async def test_dispatch_step_uses_relative_path(tmp_path: Path) -> None:
    """Dispatch step subagent_trajectory_ref.trajectory_path is relative to .daydream."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with trajectory_module.dispatch_scope(
            recorder, phase=DaydreamPhase.FIX, descriptors=["fix-0"],
        ) as dispatch:
            async with recorder.fork("fix-0", dispatch=dispatch) as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    observe_text_and_result(inv)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    parent_traj = read_trajectory(recorder.path)
    dispatch_steps = [
        s for s in parent_traj["steps"] if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    ref = dispatch_steps[0]["observation"]["results"][0]["subagent_trajectory_ref"][0]
    assert ref["trajectory_path"].startswith("runs/")
    assert ref["trajectory_path"].endswith(".json")


# Dispatch step no-op when no siblings


async def test_dispatch_step_noop_when_no_siblings(tmp_path: Path) -> None:
    """An empty declared fan-out adds no dispatch step."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)
        steps_before = len(recorder.steps)
        async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX, descriptors=[]) as dispatch:
            assert dispatch is None
        assert len(recorder.steps) == steps_before


# _safe_descriptor slugification


def test_safe_descriptor_slugification() -> None:
    """Various inputs correctly slugified by _safe_descriptor."""
    assert _safe_descriptor("fix-0") == "fix-0"
    assert _safe_descriptor("deep-python") == "deep-python"
    assert _safe_descriptor("explore-pattern-scanner") == "explore-pattern-scanner"
    assert _safe_descriptor("Fix_Issue (3)") == "fix-issue-3"
    assert _safe_descriptor("UPPER--CASE") == "upper-case"
    assert _safe_descriptor("-leading-trailing-") == "leading-trailing"
    assert _safe_descriptor("../etc/passwd") == "etc-passwd"


def test_safe_descriptor_rejects_degenerate_inputs() -> None:
    """Degenerate inputs that produce empty slugs raise ValueError (CR-01)."""
    with pytest.raises(ValueError, match="empty slug"):
        _safe_descriptor("")
    with pytest.raises(ValueError, match="empty slug"):
        _safe_descriptor("...")
    with pytest.raises(ValueError, match="empty slug"):
        _safe_descriptor("   ")


# SUBA-01: Sequential phases produce single file


async def test_sequential_phases_single_file(tmp_path: Path) -> None:
    """SUBA-01: Three sequential invocations produce one file with continuous step_ids."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        for phase in (DaydreamPhase.REVIEW, DaydreamPhase.PARSE, DaydreamPhase.FIX):
            async with recorder.invocation(phase=phase) as inv:
                observe_text_and_result(inv, f"{phase.value}-output")

    assert recorder.path.exists()
    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True

    step_ids = [s["step_id"] for s in traj["steps"]]
    assert step_ids == list(range(1, len(step_ids) + 1))

    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0


# Fork write failure degrades gracefully


async def test_fork_write_failure_degrades(tmp_path: Path) -> None:
    """If child _write() raises, parent ContextVar is restored, no crash."""
    recorder = make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)
        with patch("daydream.trajectory.print_warning", fake_print_warning):
            async with recorder.fork("fail-child") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    observe_text_and_result(inv)
                original_path = child.path
                # Sabotage the write path: make the parent an existing
                # regular file so atomic_write_json's parent mkdir fails no
                # matter the filesystem permissions (a fixed
                # /nonexistent-dir-xyz path silently succeeds on containers
                # with a writable root).
                blocker = tmp_path / "not-a-dir"
                blocker.write_text("x")
                child.path = blocker / "child.json"

        assert get_current_recorder() is recorder
        child.path = original_path
        from daydream.trajectory import flush_active_signal_recorders

        flush_active_signal_recorders()
        assert recorder.path.with_suffix(".json.partial").exists()
        assert not child.path.with_suffix(".json.partial").exists()

    assert any("Sibling trajectory write failed" in m for m in warnings_emitted)
    assert recorder.path.exists()


# Pitfall 6: Fork child with no steps produces no file


async def test_fork_child_no_steps_no_file(tmp_path: Path) -> None:
    """Pitfall 6: Child with 0 steps writes no sibling file; parent has no registration."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("empty-child"):
            pass
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0
    root = read_trajectory(recorder.path)
    assert not any("sibling_trajectory_ref" in item for item in root["extra"]["subtrajectories"])


# Multiple forks all registered


async def test_multiple_forks_all_registered(tmp_path: Path) -> None:
    """Three sequential forks all register with parent; dispatch step has 3 refs."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        descriptors = [f"fix-{i}" for i in range(3)]
        async with trajectory_module.dispatch_scope(
            recorder, phase=DaydreamPhase.FIX, descriptors=descriptors,
        ) as dispatch:
            for i, descriptor in enumerate(descriptors):
                async with recorder.fork(descriptor, dispatch=dispatch) as child:
                    async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                        observe_text_and_result(inv, f"child-{i}")
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    parent_traj = read_trajectory(recorder.path)
    assert atif_validate(parent_traj, validate_images=False) is True

    dispatch_steps = [
        s for s in parent_traj["steps"] if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    assert len(dispatch_steps) == 1
    results = dispatch_steps[0]["observation"]["results"]
    assert len(results) == 3
    for r in results:
        assert len(r["subagent_trajectory_ref"]) == 1


# Fork validator accepts both parent and child


async def test_fork_validator_accepts_both(tmp_path: Path) -> None:
    """Both parent and child trajectories pass atif_validate."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                observe_text_and_result(inv)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    parent_traj = read_trajectory(recorder.path)
    child_traj = read_trajectory(child.path)

    assert atif_validate(parent_traj, validate_images=False) is True
    assert atif_validate(child_traj, validate_images=False) is True


# write_partial tests (CLI-03, D-07 SIGINT partial flush)


async def test_write_partial_writes_partial_file_with_partial_flag(
    tmp_path: Path,
) -> None:
    """CLI-03: write_partial writes <path>.partial with extra.partial=true."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="in-flight"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        recorder.write_partial()

        partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
        assert partial_path.exists()
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        assert partial.get("extra", {}).get("partial") is True
        assert atif_validate(partial, validate_images=False) is True


def test_write_partial_no_op_when_steps_empty(tmp_path: Path) -> None:
    """write_partial skips disk write when steps list is empty (matches _write)."""
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )
    recorder.write_partial()
    partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
    assert not partial_path.exists()


async def test_write_partial_is_idempotent(tmp_path: Path) -> None:
    """Calling write_partial twice yields a single .partial file with latest contents."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="first"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        recorder.write_partial()
        partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
        first = partial_path.read_text(encoding="utf-8")
        recorder.write_partial()
        second = partial_path.read_text(encoding="utf-8")
        assert first == second


async def test_write_partial_failure_emits_warning_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk-write failure during partial flush degrades with warning, never raises."""
    recorder = make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.trajectory.print_warning", fake_print_warning)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

        monkeypatch.setattr(
            Path,
            "write_text",
            lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
        )
        recorder.write_partial()

    assert any("Partial trajectory write failed" in m for m in warnings_emitted)


# Regression: WR-01 — write_partial must capture Invocation in-flight steps


async def test_write_partial_captures_in_flight_invocation_steps(
    tmp_path: Path,
) -> None:
    """WR-01 regression: SIGINT mid-run_agent() must include in-flight Invocation steps.

    Pre-fix bug: write_partial read recorder.steps directly, but Invocation
    accumulates steps in its own list and only flushes to recorder.steps on
    __aexit__. A partial flush mid-invocation lost every step from the
    in-flight invocation.
    """
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            # Steps observed inside the invocation but BEFORE __aexit__
            inv.observe_user_step(prompt="hello")
            inv.observe(TextEvent(text="response"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

            assert recorder.steps == [], "recorder.steps should be empty mid-invocation"
            assert len(inv.steps) >= 2, "Invocation should have accumulated steps"

            recorder.write_partial()

    partial_path = tmp_path / ".daydream" / "trajectory.json.partial"
    assert partial_path.exists(), "Partial trajectory file should be written"

    data = json.loads(partial_path.read_text(encoding="utf-8"))
    assert data.get("extra", {}).get("partial") is True
    assert len(data["steps"]) >= 2, f"Partial trajectory missing in-flight steps: {data['steps']!r}"
    messages = [s.get("message") for s in data["steps"] if isinstance(s, dict)]
    assert any("hello" in (m or "") for m in messages), f"User step prompt not in partial: {messages!r}"


async def test_write_partial_records_in_flight_tool_like_finish(tmp_path: Path) -> None:
    """A partial flush mid-invocation carries the same interrupted marker the
    final flush (finish()) emits, and the snapshot leaves live state untouched."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(ToolStartEvent(id="ip1", name="Read", input={"file_path": "a.py"}))
            recorder.write_partial()
            # Signal-safe snapshot: nothing consumed; finish() still marks the call.
            assert "ip1" in inv._in_flight_tools

    partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    final = read_trajectory(recorder.path)

    def marker_results(traj: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            r
            for s in traj["steps"]
            if s["source"] == "agent"
            for r in (s.get("observation") or {}).get("results") or []
            if r.get("extra", {}).get("status") == "interrupted"
        ]

    expected = [
        {
            "content": INCOMPLETE_CONTENT,
            "extra": {"is_error": True, "status": "interrupted"},
        }
    ]
    assert marker_results(partial) == expected
    assert marker_results(final) == expected


async def test_write_partial_preserves_in_flight_diagnostic_once(
    tmp_path: Path,
) -> None:
    """Signal-safe snapshots include normalized diagnostics without consuming them."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(DiagnosticEvent(code="parser_gap", message="bounded", metadata={"count": 1}))
            recorder.write_partial()

    partial = json.loads(recorder.path.with_suffix(recorder.path.suffix + ".partial").read_text(encoding="utf-8"))
    final = read_trajectory(recorder.path)

    for trajectory in (partial, final):
        diagnostics = [
            diagnostic
            for step in trajectory["steps"]
            if step["source"] == "agent"
            for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
        ]
        assert diagnostics == [{"code": "parser_gap", "message": "bounded", "metadata": {"count": 1}}]


async def test_write_partial_no_double_count_after_invocation_exit(
    tmp_path: Path,
) -> None:
    """After Invocation.__aexit__, write_partial should NOT double-count steps.

    Steps moved from invocation.steps to recorder.steps; write_partial reads
    recorder.steps + active_invocations. After the invocation exits,
    active_invocations is empty so we read only recorder.steps.
    """
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hi")
            inv.observe(TextEvent(text="ok"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

        # Now invocation has exited, steps flushed
        assert recorder._active_invocations == []
        flushed_count = len(recorder.steps)
        assert flushed_count >= 2

        recorder.write_partial()

    partial_path = tmp_path / ".daydream" / "trajectory.json.partial"
    data = json.loads(partial_path.read_text(encoding="utf-8"))
    assert len(data["steps"]) == flushed_count, f"Expected {flushed_count} steps post-exit, got {len(data['steps'])}"


def _partial_path(recorder: TrajectoryRecorder) -> Path:
    return recorder.path.with_suffix(recorder.path.suffix + ".partial")


def _trajectory_text(path: Path) -> str:
    return json.dumps(read_trajectory(path), sort_keys=True)


async def _hold_fork(
    parent: TrajectoryRecorder,
    descriptor: str,
    marker: str,
    entered: anyio.Event,
    release: anyio.Event,
    children: dict[str, TrajectoryRecorder],
) -> None:
    """Hold one public fork invocation open across a signal-flush barrier."""
    async with parent.fork(descriptor) as child:
        children[descriptor] = child
        async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
            assert get_current_recorder() is child
            observe_text_and_result(inv, marker)
            entered.set()
            await release.wait()


@pytest.mark.parametrize(
    "entry_order",
    [("signal-a", "signal-b"), ("signal-b", "signal-a")],
    ids=["a-then-b", "b-then-a"],
)
async def test_signal_flushes_concurrent_siblings(tmp_path: Path, entry_order: tuple[str, str]) -> None:
    """One run flush writes root and every live sibling, independent of entry order."""
    from daydream.trajectory import flush_active_signal_recorders

    markers = {"signal-a": "SIBLING_A_ONLY", "signal-b": "SIBLING_B_ONLY"}
    entered = {name: anyio.Event() for name in markers}
    release = {name: anyio.Event() for name in markers}
    children: dict[str, TrajectoryRecorder] = {}
    root = make_recorder(tmp_path)

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        async with anyio.create_task_group() as tg:
            for name in entry_order:
                tg.start_soon(
                    _hold_fork,
                    root,
                    name,
                    markers[name],
                    entered[name],
                    release[name],
                    children,
                )
                await entered[name].wait()

            flush_active_signal_recorders()
            paths = [
                _partial_path(root),
                *(_partial_path(children[n]) for n in markers),
            ]
            assert all(path.exists() for path in paths)
            trajectories = [read_trajectory(path) for path in paths]
            assert all(atif_validate(item, validate_images=False) for item in trajectories)
            assert len({item["trajectory_id"] for item in trajectories}) == 3
            assert "ROOT_ONLY" in _trajectory_text(paths[0])
            for name, marker in markers.items():
                text = _trajectory_text(_partial_path(children[name]))
                assert marker in text
                assert "ROOT_ONLY" not in text
                assert all(other == marker or other not in text for other in markers.values())

            for name in reversed(entry_order):
                release[name].set()


async def test_signal_flush_freezes_all_documents_before_one_callback(
    tmp_path: Path,
) -> None:
    """A run-wide partial snapshot has one cutoff and immutable written bytes."""
    from daydream.trajectory import RunWriteSnapshot, flush_active_signal_recorders

    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    entered = {name: anyio.Event() for name in ("signal-a", "signal-b")}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, TrajectoryRecorder] = {}

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        async with anyio.create_task_group() as tg:
            for name in entered:
                tg.start_soon(
                    _hold_fork,
                    root,
                    name,
                    name,
                    entered[name],
                    release[name],
                    children,
                )
                await entered[name].wait()

            flush_active_signal_recorders()
            assert len(snapshots) == 1
            snapshot = snapshots[0]
            assert snapshot.status == "partial"
            assert len(snapshot.documents) == 3
            payloads = [json.loads(document.json_bytes) for document in snapshot.documents]
            assert {payload["extra"]["snapshot_at"] for payload in payloads} == {snapshot.cutoff_at}
            assert all("run_ended_at" not in payload["extra"] for payload in payloads)
            assert all(document.path.read_bytes() == document.json_bytes for document in snapshot.documents)
            timing = root.compute_timing_summary(snapshot)
            assert timing is not None
            assert timing.agent_completeness["total"] == 3
            assert timing.diagnostics["malformed_invocation"] == 2

            frozen = tuple(document.json_bytes for document in snapshot.documents)
            for event in release.values():
                event.set()

        assert tuple(document.json_bytes for document in snapshot.documents) == frozen


async def test_signal_flush_with_child_evidence_freezes_schema_valid_empty_root(
    tmp_path: Path,
) -> None:
    """An early fan-out signal retains root lifecycle evidence without an LLM call."""
    from daydream.trajectory import RunWriteSnapshot, flush_active_signal_recorders

    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    entered = anyio.Event()
    release = anyio.Event()
    children: dict[str, TrajectoryRecorder] = {}

    async with root:
        assert root.steps == []
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                _hold_fork,
                root,
                "initial-exploration",
                "CHILD_ONLY",
                entered,
                release,
                children,
            )
            await entered.wait()

            flush_active_signal_recorders()
            assert len(snapshots) == 1
            snapshot = snapshots[0]
            assert [document.trajectory_id for document in snapshot.documents] == [
                root.trajectory_id,
                children["initial-exploration"].trajectory_id,
            ]
            root_payload = json.loads(snapshot.documents[0].json_bytes)
            assert atif_validate(root_payload, validate_images=False)
            assert root_payload["steps"] == [
                {
                    "step_id": 1,
                    "timestamp": snapshot.cutoff_at,
                    "source": "system",
                    "message": "Daydream run snapshot",
                    "extra": {
                        "daydream_run_flow": root.run_flow.value,
                        "host_event": "partial_snapshot",
                    },
                }
            ]
            assert root.steps == []
            assert root.compute_timing_summary(snapshot) is not None
            release.set()


async def test_signal_flush_reuses_cutoff_until_any_document_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged run snapshot reuses bytes; child progress advances its cutoff."""
    from daydream.trajectory import RunWriteSnapshot, flush_active_signal_recorders

    ticks = iter(f"2026-09-06T00:00:{second:02d}.000000Z" for second in range(60))
    monkeypatch.setattr(trajectory_module, "now_iso", lambda: next(ticks))
    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))

    async with root:
        async with trajectory_module.maybe_fork(root, "active-child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe(TextEvent(text="first state"))
                inv.observe(ResultEvent(structured_output=None, continuation=None))
                flush_active_signal_recorders()
                first = snapshots[-1]
                first_bytes = tuple(document.json_bytes for document in first.documents)

                flush_active_signal_recorders()
                second = snapshots[-1]
                assert second.cutoff_at == first.cutoff_at
                assert tuple(document.json_bytes for document in second.documents) == first_bytes

                inv.observe_user_step(prompt="later context")
                flush_active_signal_recorders()
                third = snapshots[-1]
                assert third.cutoff_at != first.cutoff_at
                third_bytes = tuple(document.json_bytes for document in third.documents)
                assert third_bytes != first_bytes
                assert {
                    json.loads(document.json_bytes)["extra"]["snapshot_at"]
                    for document in third.documents
                } == {third.cutoff_at}
                assert tuple(document.json_bytes for document in first.documents) == first_bytes


async def test_signal_flush_root_prepare_failure_never_publishes_rootless_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root freeze failure writes no child-only snapshot and a retry recovers."""
    from daydream.trajectory import RunWriteSnapshot, flush_active_signal_recorders

    snapshots: list[RunWriteSnapshot] = []
    warnings: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    original_prepare = root._prepare_document
    failed_once = False

    def fail_first_root_prepare(**kwargs: Any) -> Any:
        nonlocal failed_once
        if kwargs["status"] == "partial" and not failed_once:
            failed_once = True
            raise RuntimeError("root preparation failed")
        return original_prepare(**kwargs)

    monkeypatch.setattr(root, "_prepare_document", fail_first_root_prepare)
    monkeypatch.setattr(
        trajectory_module,
        "print_warning",
        lambda _console, message: warnings.append(message),
    )

    async with root:
        async with root.fork("active-child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe(TextEvent(text="child evidence"))

                flush_active_signal_recorders()
                assert len(warnings) == 1
                assert snapshots == []
                assert not _partial_path(root).exists()
                assert not _partial_path(child).exists()

                flush_active_signal_recorders()
                assert len(warnings) == 1
                assert len(snapshots) == 1
                snapshot = snapshots[0]
                assert [document.trajectory_id for document in snapshot.documents] == [
                    root.trajectory_id,
                    child.trajectory_id,
                ]
                assert all(
                    atif_validate(json.loads(document.json_bytes), validate_images=False)
                    for document in snapshot.documents
                )
                assert {
                    json.loads(document.json_bytes)["extra"]["snapshot_at"]
                    for document in snapshot.documents
                } == {snapshot.cutoff_at}
                assert all(document.path.exists() for document in snapshot.documents)


@pytest.mark.parametrize("exit_kind", ["normal", "runtime", "cancel", "system-exit"])
async def test_signal_flush_excludes_exited_child(tmp_path: Path, exit_kind: str) -> None:
    """Every child exit shape unregisters before a later root-only flush."""
    from daydream.trajectory import flush_active_signal_recorders

    root = make_recorder(tmp_path)

    async def child_body() -> TrajectoryRecorder:
        async with root.fork(f"exited-{exit_kind}") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                observe_text_and_result(inv, "EXITED_CHILD_ONLY")
                if exit_kind == "runtime":
                    raise RuntimeError("child body")
                if exit_kind == "system-exit":
                    raise SystemExit(17)
            return child

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")

        if exit_kind == "runtime":
            with pytest.raises(RuntimeError, match="child body"):
                await child_body()
        elif exit_kind == "system-exit":
            with pytest.raises(SystemExit) as exc:
                await child_body()
            assert exc.value.code == 17
        elif exit_kind == "cancel":
            with anyio.CancelScope() as scope:
                async with root.fork("exited-cancel") as active_child:
                    async with active_child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                        observe_text_and_result(inv, "EXITED_CHILD_ONLY")
                        scope.cancel()
                        await anyio.sleep_forever()
        else:
            await child_body()

        flush_active_signal_recorders()
        assert _partial_path(root).exists()
        child_path = root._sibling_path_for(f"exited-{exit_kind}")
        assert not child_path.with_suffix(child_path.suffix + ".partial").exists()


async def test_signal_flush_excludes_child_after_final_write_system_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BaseException from the child final write cannot leak registry membership."""
    from daydream.trajectory import flush_active_signal_recorders

    root = make_recorder(tmp_path)
    child: TrajectoryRecorder
    child_path: Path

    def fail_final_write() -> None:
        raise SystemExit(23)

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        with pytest.raises(SystemExit) as exc:
            async with root.fork("final-system-exit") as child:
                child_path = child.path
                async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                    observe_text_and_result(inv, "STALE_CHILD_ONLY")
                monkeypatch.setattr(child, "_write", fail_final_write)
        assert exc.value.code == 23
        child.path = child_path
        flush_active_signal_recorders()
        assert _partial_path(root).exists()
        assert not _partial_path(child).exists()


async def test_signal_flush_selects_latest_independent_root(tmp_path: Path) -> None:
    """A nested independent root is targeted until it exits, then outer resumes."""
    from daydream.trajectory import flush_active_signal_recorders

    writes: list[tuple[str, str]] = []
    outer = make_recorder(
        tmp_path / "outer",
        on_write=lambda _rec, snapshot: writes.append(("outer", snapshot.status)),
    )
    inner = make_recorder(
        tmp_path / "inner",
        on_write=lambda _rec, snapshot: writes.append(("inner", snapshot.status)),
    )
    async with outer:
        async with outer.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "OUTER_ONLY")
        async with inner:
            async with inner.invocation(phase=DaydreamPhase.REVIEW) as inv:
                observe_text_and_result(inv, "INNER_ONLY")
            flush_active_signal_recorders()
            assert writes == [("inner", "partial")]
        writes.clear()
        flush_active_signal_recorders()
        assert writes == [("outer", "partial")]


def _finish_shutdown_panel() -> None:
    from daydream.ui import get_shutdown_panel, set_shutdown_panel

    panel = get_shutdown_panel()
    if panel is not None:
        panel.finish()
        set_shutdown_panel(None)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
async def test_signal_handler_flushes_all_siblings_once(tmp_path: Path, signum: signal.Signals) -> None:
    """The real handler flushes root and both siblings without parent recursion."""
    from daydream.cli import _signal_handler

    root_statuses: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: root_statuses.append(snapshot.status))
    entered = {name: anyio.Event() for name in ("signal-a", "signal-b")}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, TrajectoryRecorder] = {}
    markers = {"signal-a": "SIBLING_A_ONLY", "signal-b": "SIBLING_B_ONLY"}

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        async with anyio.create_task_group() as tg:
            for name in ("signal-a", "signal-b"):
                tg.start_soon(
                    _hold_fork,
                    root,
                    name,
                    markers[name],
                    entered[name],
                    release[name],
                    children,
                )
                await entered[name].wait()
            try:
                with pytest.raises(KeyboardInterrupt):
                    _signal_handler(signum, None)
                assert root_statuses == ["partial"]
                paths = [
                    _partial_path(root),
                    *(_partial_path(children[n]) for n in entered),
                ]
                assert all(path.exists() for path in paths)
            finally:
                _finish_shutdown_panel()
                for event in release.values():
                    event.set()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
async def test_signal_handler_isolates_sibling_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: signal.Signals,
) -> None:
    """One denied sibling partial cannot prevent healthy siblings or shutdown setup."""
    from daydream.cli import _signal_handler

    root = make_recorder(tmp_path)
    entered = {name: anyio.Event() for name in ("signal-a", "signal-b")}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, TrajectoryRecorder] = {}
    warnings: list[str] = []
    real_write_text = Path.write_text

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        async with anyio.create_task_group() as tg:
            for name, marker in (
                ("signal-a", "SIBLING_A_ONLY"),
                ("signal-b", "SIBLING_B_ONLY"),
            ):
                tg.start_soon(
                    _hold_fork,
                    root,
                    name,
                    marker,
                    entered[name],
                    release[name],
                    children,
                )
                await entered[name].wait()

            denied_path = _partial_path(children["signal-a"])

            def selective_write(path: Path, *args: Any, **kwargs: Any) -> int:
                if path == denied_path:
                    raise PermissionError("denied sibling A")
                return real_write_text(path, *args, **kwargs)

            monkeypatch.setattr(Path, "write_text", selective_write)
            monkeypatch.setattr(
                "daydream.trajectory.print_warning",
                lambda _console, message: warnings.append(message),
            )
            try:
                with pytest.raises(KeyboardInterrupt):
                    _signal_handler(signum, None)
                assert not denied_path.exists()
                for recorder in (root, children["signal-b"]):
                    path = _partial_path(recorder)
                    assert path.exists()
                    assert atif_validate(read_trajectory(path), validate_images=False)
                matching_warnings = [
                    warning for warning in warnings if "Partial trajectory write failed: PermissionError" in warning
                ]
                assert len(matching_warnings) == 1
            finally:
                _finish_shutdown_panel()
                for event in release.values():
                    event.set()


async def test_forked_child_write_partial_captures_in_flight_steps(
    tmp_path: Path,
) -> None:
    """A direct child write keeps the established child-to-parent cascade."""

    parent = make_recorder(tmp_path)
    async with parent:
        async with parent.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "parent-before-child")
        async with parent.fork("child-branch") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="forked-prompt")
                inv.observe(TextEvent(text="forked-response"))
                inv.observe(ResultEvent(structured_output=None, continuation=None))

                child.write_partial()

    partial_path = child.path.with_suffix(child.path.suffix + ".partial")
    parent_partial_path = parent.path.with_suffix(parent.path.suffix + ".partial")
    assert partial_path.exists(), "Child partial trajectory should be written"
    assert parent_partial_path.exists(), "Direct child partial should cascade to parent"

    data = json.loads(partial_path.read_text(encoding="utf-8"))
    parent_data = json.loads(parent_partial_path.read_text(encoding="utf-8"))
    assert data.get("extra", {}).get("partial") is True
    assert parent_data.get("extra", {}).get("partial") is True
    assert "parent-before-child" in json.dumps(parent_data, sort_keys=True)
    assert len(data["steps"]) >= 2, f"Child partial missing in-flight steps: {data['steps']!r}"


async def test_recorder_marks_partial_on_exception_exit(tmp_path: Path) -> None:
    """When __aexit__ receives an exception, the trajectory is marked partial."""
    recorder = make_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="hello")
                inv.observe(TextEvent(text="partial output"))
                raise RuntimeError("boom")

    assert recorder.path.exists()
    traj = read_trajectory(recorder.path)
    assert traj.get("extra", {}).get("partial") is True


async def test_forked_child_marks_partial_on_exception_exit(tmp_path: Path) -> None:
    """A sibling that dies mid-flight is marked partial on exception exit.

    Regression for the fork path: _ForkCM.__aexit__ must mirror the top-level
    recorder and set child._aborted when an exception escapes the fork scope,
    so the sibling trajectory's extra.partial reflects that it was aborted —
    not silently written as if it completed cleanly.
    """
    recorder = make_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        async with recorder:
            async with recorder.fork("fix-0") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    inv.observe_user_step(prompt="hello")
                    inv.observe(TextEvent(text="partial sibling output"))
                    raise RuntimeError("boom")

    assert child.path.exists(), "Sibling trajectory should be written on exception exit"
    sibling_traj = read_trajectory(child.path)
    assert sibling_traj.get("extra", {}).get("partial") is True, (
        "Forked child trajectory must be marked partial when an exception escapes the fork"
    )


async def test_forked_child_does_not_mark_partial_on_clean_exit(tmp_path: Path) -> None:
    """A sibling that exits cleanly is NOT marked partial."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe_user_step(prompt="hello")
                inv.observe(TextEvent(text="sibling output"))
                inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert child.path.exists()
    sibling_traj = read_trajectory(child.path)
    assert "partial" not in sibling_traj.get("extra", {})


async def test_recorder_does_not_mark_partial_on_clean_exit(tmp_path: Path) -> None:
    """Clean exit does NOT mark the trajectory as partial."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hello")
            inv.observe(TextEvent(text="world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert recorder.path.exists()
    traj = read_trajectory(recorder.path)
    assert "partial" not in traj.get("extra", {})


def test_custom_run_flow_member_exists() -> None:
    assert DaydreamRunFlow.CUSTOM.value == "custom"
    # str-Enum: the value round-trips as a plain string for metadata serialization
    assert DaydreamRunFlow("custom") is DaydreamRunFlow.CUSTOM


# Fork totals fold into the parent — the root file is whole-run truth


async def test_fork_totals_fold_into_parent(tmp_path: Path) -> None:
    """Root final_metrics includes fork totals; the fork file keeps its own share."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent-text"))
            inv.observe(
                MetricsEvent(
                    message_id="m-1",
                    prompt_tokens=100,
                    completion_tokens=20,
                    cached_tokens=10,
                    cost_usd=1.0,
                )
            )
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as cinv:
                cinv.observe(TextEvent(text="child-text"))
                cinv.observe(
                    MetricsEvent(
                        message_id="m-2",
                        prompt_tokens=40,
                        completion_tokens=8,
                        cached_tokens=4,
                        cost_usd=0.5,
                    )
                )
                cinv.observe(ResultEvent(structured_output=None, continuation=None))

    parent = read_trajectory(recorder.path)["final_metrics"]
    assert parent["total_prompt_tokens"] == 140
    assert parent["total_completion_tokens"] == 28
    assert parent["total_cached_tokens"] == 14
    assert parent["total_cost_usd"] == pytest.approx(1.5)
    assert parent["total_steps"] == len(read_trajectory(recorder.path)["steps"])
    assert parent["extra"] == {
        "daydream_metric_scope": "whole_run_including_forks",
        "total_steps_scope": "local_trajectory",
    }

    child_fm = read_trajectory(child.path)["final_metrics"]
    assert child_fm["total_prompt_tokens"] == 40
    assert child_fm["total_cost_usd"] == pytest.approx(0.5)


async def test_empty_fork_folds_nothing_into_parent(tmp_path: Path) -> None:
    """A fork whose write produced no steps contributes no totals."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent-text"))
            inv.observe(
                MetricsEvent(
                    message_id="m-1",
                    prompt_tokens=100,
                    completion_tokens=20,
                    cached_tokens=10,
                    cost_usd=1.0,
                )
            )
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("empty-child"):
            pass

    parent = read_trajectory(recorder.path)["final_metrics"]
    assert parent["total_prompt_tokens"] == 100
    assert parent["total_cost_usd"] == pytest.approx(1.0)


async def test_nested_fork_totals_reach_the_root(tmp_path: Path) -> None:
    """Fold is transitive: a fork of a fork reaches the root's final_metrics."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="root"))
            inv.observe(
                MetricsEvent(
                    message_id="m-1",
                    prompt_tokens=10,
                    completion_tokens=1,
                    cached_tokens=0,
                    cost_usd=0.1,
                )
            )
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("outer") as outer:
            async with outer.invocation(phase=DaydreamPhase.DEEP) as oinv:
                oinv.observe(TextEvent(text="outer"))
                oinv.observe(
                    MetricsEvent(
                        message_id="m-2",
                        prompt_tokens=20,
                        completion_tokens=2,
                        cached_tokens=0,
                        cost_usd=0.2,
                    )
                )
                oinv.observe(ResultEvent(structured_output=None, continuation=None))
            async with outer.fork("inner") as inner:
                async with inner.invocation(phase=DaydreamPhase.DEEP) as iinv:
                    iinv.observe(TextEvent(text="inner"))
                    iinv.observe(
                        MetricsEvent(
                            message_id="m-3",
                            prompt_tokens=30,
                            completion_tokens=3,
                            cached_tokens=0,
                            cost_usd=0.3,
                        )
                    )
                    iinv.observe(ResultEvent(structured_output=None, continuation=None))

    root = read_trajectory(recorder.path)["final_metrics"]
    assert root["total_prompt_tokens"] == 60
    assert root["total_cost_usd"] == pytest.approx(0.6)


async def test_analyze_costs_total_comes_from_root_only(tmp_path: Path) -> None:
    """Root final_metrics is fork-inclusive, so analyze_costs must not re-sum forks."""
    from daydream.eval.analyzer import analyze_costs, load_trajectories

    session = "sess-fold-0001"
    daydream_dir = tmp_path / ".daydream"
    recorder = TrajectoryRecorder(
        path=daydream_dir / "runs" / session / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id=session,
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent"))
            inv.observe(
                MetricsEvent(
                    message_id="m-1",
                    prompt_tokens=100,
                    completion_tokens=20,
                    cached_tokens=10,
                    cost_usd=1.0,
                )
            )
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as cinv:
                cinv.observe(TextEvent(text="child"))
                cinv.observe(
                    MetricsEvent(
                        message_id="m-2",
                        prompt_tokens=40,
                        completion_tokens=8,
                        cached_tokens=4,
                        cost_usd=0.5,
                    )
                )
                cinv.observe(ResultEvent(structured_output=None, continuation=None))

    costs = analyze_costs(load_trajectories(daydream_dir, session))
    assert costs["total_cost_usd"] == pytest.approx(1.5)  # not 2.0 (root 1.5 + fork 0.5)
    assert costs["total_prompt_tokens_raw"] == 140  # not 180
    assert costs["total_completion_tokens"] == 28

    # by_agent rows keep fork-level detail, and the main row is the root's
    # folded totals minus the fork totals, so sum(by_agent) == total (no
    # double-count of the fork).
    by_agent = {a["agent"]: a for a in costs["by_agent"]}
    assert len(by_agent) == 2
    assert any(a["cost_usd"] == pytest.approx(0.5) for a in costs["by_agent"])
    assert sum(a["cost_usd"] for a in costs["by_agent"]) == pytest.approx(costs["total_cost_usd"])


async def test_build_trajectory_extra_records_backend_identity(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
        backend_name="codex",
        review_backend_name="pi",
        fix_backend_name="claude",
        test_backend_name="osprey",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hello"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    extra = read_trajectory(recorder.path)["extra"]
    assert extra["backend"] == "codex"
    assert extra["review_backend"] == "pi"
    assert extra["fix_backend"] == "claude"
    assert extra["test_backend"] == "osprey"
    assert extra["target_dir"] == str(tmp_path)


async def test_build_trajectory_omits_backend_when_unset(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hello"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    extra = read_trajectory(recorder.path)["extra"]
    assert "backend" not in extra
    assert "review_backend" not in extra
    assert "fix_backend" not in extra
    assert "test_backend" not in extra


async def test_build_trajectory_omits_empty_per_phase_backend_keys(
    tmp_path: Path,
) -> None:
    """Per-phase backend keys are omitted when their name is empty (improve flow)."""
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.IMPROVE,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
        backend_name="codex",
        review_backend_name="codex",
        fix_backend_name="",
        test_backend_name="",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hello"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    extra = read_trajectory(recorder.path)["extra"]
    assert extra["backend"] == "codex"
    assert extra["review_backend"] == "codex"
    assert "fix_backend" not in extra
    assert "test_backend" not in extra


async def test_fork_child_inherits_backend_identity(tmp_path: Path) -> None:
    parent = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
        backend_name="codex",
        review_backend_name="codex",
        fix_backend_name="pi",
        test_backend_name="codex",
    )
    async with parent:
        async with parent.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with parent.fork("deep") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as cinv:
                cinv.observe(TextEvent(text="child"))
                cinv.observe(ResultEvent(structured_output=None, continuation=None))
    child_extra = read_trajectory(child.path)["extra"]
    assert child_extra["backend"] == "codex"
    assert child_extra["review_backend"] == "codex"
    assert child_extra["fix_backend"] == "pi"
    assert child_extra["test_backend"] == "codex"


# Issue #726 task 12: host-side phases (test-execution / hook-run / commit /
# push) are bracketed by phase events carrying duration_ms + stop_reason.


async def test_host_phase_scope_records_duration_and_stop_reason(
    tmp_path: Path,
) -> None:
    from daydream.trajectory import DaydreamPhase, host_phase_scope

    rec = make_recorder(tmp_path)
    async with rec:
        async with host_phase_scope(DaydreamPhase.COMMIT) as handle:
            handle.stop_reason = "timed_out"
        with pytest.raises(RuntimeError):
            async with host_phase_scope(DaydreamPhase.PUSH):
                raise RuntimeError("push failed")
        async with host_phase_scope(DaydreamPhase.HOOK_RUN):
            pass

    events = rec.phase_event_dicts()
    ends = {(e["phase"], e["event"]): e for e in events if e["event"] == "phase_end"}
    commit = ends[(DaydreamPhase.COMMIT.value, "phase_end")]
    assert commit["status"] == "timed_out"
    assert commit["reason_code"] == "timed_out"
    assert commit["session_id"] == rec.session_id
    assert commit["scope_id"]
    assert commit["metadata"]["stop_reason"] == "timed_out"
    assert commit["metadata"]["duration_ms"] >= 0
    push = ends[(DaydreamPhase.PUSH.value, "phase_end")]
    assert push["status"] == "failed"
    assert push["reason_code"] == "uncaught_exception"
    assert push["metadata"]["stop_reason"] == "failed"
    assert push["metadata"]["duration_ms"] >= 0
    hook = ends[(DaydreamPhase.HOOK_RUN.value, "phase_end")]
    assert hook["status"] == "succeeded"
    assert hook["metadata"]["stop_reason"] == "completed"
    # The original four host phases remain distinct; remote CI has its own
    # focused assertion below.
    assert {
        p.value
        for p in (
            DaydreamPhase.TEST_EXECUTION,
            DaydreamPhase.HOOK_RUN,
            DaydreamPhase.COMMIT,
            DaydreamPhase.PUSH,
        )
    } == {"test-execution", "hook-run", "commit", "push"}


async def test_host_phase_scope_noop_without_recorder() -> None:
    from daydream.trajectory import (
        DaydreamPhase,
        _reset_recorder_for_tests,
        host_phase_scope,
    )

    _reset_recorder_for_tests()
    async with host_phase_scope(DaydreamPhase.COMMIT):
        pass  # must not raise when no recorder is active


@pytest.mark.parametrize(
    ("stop_reason", "expected_status", "expected_reason"),
    [
        ("completed", "succeeded", None),
        ("passed", "succeeded", None),
        ("no_ci", "succeeded", None),
        ("timed_out", "timed_out", "timed_out"),
        ("cancelled", "cancelled", "cancelled"),
        ("interrupted", "cancelled", "cancelled"),
        ("failed", "failed", "domain_failure"),
        ("missing", "failed", "domain_failure"),
        ("unavailable", "failed", "domain_failure"),
        ("superseded", "failed", "domain_failure"),
        ("pending", "failed", "domain_failure"),
    ],
)
async def test_remote_ci_host_phases_record_exact_terminal_reasons(
    tmp_path: Path,
    stop_reason: str,
    expected_status: str,
    expected_reason: str | None,
) -> None:
    """Every admitted remote-CI reason has one closed lifecycle projection."""
    from daydream.trajectory import DaydreamPhase, host_phase_scope

    rec = make_recorder(tmp_path)
    async with rec:
        async with host_phase_scope(DaydreamPhase.REMOTE_CI) as phase:
            await anyio.sleep(0)
            phase.stop_reason = stop_reason

    remote_events = [
        event
        for event in rec.phase_event_dicts()
        if event["phase"] == DaydreamPhase.REMOTE_CI.value
    ]
    assert [event["event"] for event in remote_events] == ["phase_start", "phase_end"]
    ends = [event for event in remote_events if event["event"] == "phase_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == expected_status
    assert ends[0].get("reason_code") == expected_reason
    assert ends[0]["metadata"]["stop_reason"] == stop_reason
    assert all(event["metadata"]["duration_ms"] >= 0 for event in ends)


async def test_artifact_document_writer_precedes_root_capture_and_is_inherited_by_fork(
    tmp_path: Path,
) -> None:
    """The host sink owns root/child bytes while P07 keeps one pure root callback."""
    private_run = tmp_path / "private" / "runs" / "test"
    writes: list[tuple[str, str, Path, bytes]] = []
    callbacks: list[Any] = []
    callback_contexts: list[TrajectoryRecorder | None] = []

    def writer(document: Any, status: str) -> None:
        writes.append((document.trajectory_id, status, document.path, document.json_bytes))

    def capture(recorder: TrajectoryRecorder, snapshot: Any) -> None:
        callback_contexts.append(get_current_recorder())
        callbacks.append((recorder, snapshot, tuple(writes)))

    recorder = TrajectoryRecorder(
        path=private_run / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        artifact_run_dir=private_run,
        document_writer=writer,
        agent_model_name="test",
        session_id="test",
        on_write=capture,
    )
    async with recorder:
        async with recorder.fork("child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, "child")
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, "root")

    assert [item[:2] for item in writes] == [
        ("test:child", "complete"),
        ("test", "complete"),
    ]
    assert writes[0][2].parent == private_run / "trajectories"
    assert callbacks and callbacks[0][0] is recorder
    assert callback_contexts == [recorder]
    assert get_current_recorder() is None
    snapshot = callbacks[0][1]
    assert snapshot.status == "complete"
    assert [document.trajectory_id for document in snapshot.documents] == [
        "test",
        "test:child",
    ]
    assert [item[0] for item in callbacks[0][2]] == ["test:child", "test"]
    assert not recorder.path.exists()


async def test_artifact_partial_writer_failure_still_delivers_immutable_capture(
    tmp_path: Path,
) -> None:
    """A failed live partial write cannot erase the already prepared P07 bytes."""
    captured: list[Any] = []

    def fail_writer(_document: Any, status: str) -> None:
        assert status == "partial"
        raise OSError("injected partial output failure")

    recorder = TrajectoryRecorder(
        path=tmp_path / "private" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        artifact_run_dir=tmp_path / "private",
        document_writer=fail_writer,
        agent_model_name="test",
        session_id="test",
        on_write=lambda _recorder, snapshot: captured.append(snapshot),
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, "partial")
        recorder.write_partial()
        recorder.document_writer = None

    assert len(captured) == 2
    assert captured[0].status == "partial"
    assert json.loads(captured[0].documents[0].json_bytes)["extra"]["partial"] is True


async def test_artifact_final_writer_failure_preserves_existing_primary_exception(
    tmp_path: Path,
) -> None:
    """A secondary explicit-output failure cannot replace the active body error."""
    primary = RuntimeError("authoritative body failure")

    def fail_writer(_document: Any, _status: str) -> None:
        raise OSError("secondary output failure")

    recorder = TrajectoryRecorder(
        path=tmp_path / "explicit.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        document_writer=fail_writer,
        agent_model_name="test",
        session_id="test",
        explicit_path=True,
    )

    with pytest.raises(RuntimeError) as raised:
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, "body")
            raise primary

    assert raised.value is primary
    assert any("trajectory finalization" in note for note in primary.__notes__)


def test_remote_ci_artifact_paths_are_named_under_deep_dir(tmp_path: Path) -> None:
    from daydream.deep.artifacts import (
        push_verdict_path,
        remote_ci_handoff_path,
        remote_ci_verdict_path,
    )

    assert push_verdict_path(tmp_path) == tmp_path / "push-verdict.json"
    assert remote_ci_verdict_path(tmp_path) == tmp_path / "remote-ci-verdict.json"
    assert remote_ci_handoff_path(tmp_path) == tmp_path / "remote-ci-handoff.json"


async def test_do_commit_records_commit_phase_event(
    git_repo: Path,
    make_work: Any,
) -> None:
    """Real-path: _do_commit's host-native commit emits a distinct ``commit``
    phase event with duration_ms + stop_reason (issue #726 task 12)."""
    from collections.abc import AsyncGenerator

    from daydream.backends import ResultEvent, TextEvent
    from daydream.phases import _do_commit

    class _Backend:
        model = "mock-model"

        async def cancel(self) -> None:
            return None

        async def execute(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
            yield TextEvent(text="unused on the host commit path")
            yield ResultEvent(structured_output=None, continuation=None)

    work = make_work(git_repo)
    (git_repo / "app.py").write_text("x = 0\n")
    _git_add_commit(git_repo)
    (git_repo / "app.py").write_text("x = 1\n")

    rec = make_recorder(git_repo)
    async with rec:
        ok = await _do_commit(_Backend(), work, push=False, preexisting_untracked=set())
    assert ok.committed is True
    assert ok.push is None

    commit_ends = [e for e in rec.phase_event_dicts() if e["phase"] == "commit" and e["event"] == "phase_end"]
    assert len(commit_ends) == 1
    assert commit_ends[0]["metadata"]["stop_reason"] == "completed"
    assert commit_ends[0]["metadata"]["duration_ms"] >= 0


def _git_add_commit(repo: Path) -> None:
    import subprocess

    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "base"],
        cwd=repo,
        check=True,
    )

"""Recorder closes one Step per TurnEndEvent — never collapses multi-turn invocations."""
from pathlib import Path

from daydream.backends import TextEvent, ThinkingEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


async def test_two_text_turns_produce_two_steps(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="00000000-0000-0000-0000-000000000001",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Turn one body."))
            inv.observe(TurnEndEvent(message_id="msg_1"))
            inv.observe(TextEvent(text="Turn two body."))
            inv.observe(TurnEndEvent(message_id="msg_2"))
    agent_steps = [s for s in recorder.steps if s.source == "agent"]
    assert [s.message for s in agent_steps] == ["Turn one body.", "Turn two body."]


async def test_reasoning_is_isolated_per_turn(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="00000000-0000-0000-0000-000000000002",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(ThinkingEvent(text="thought-1"))
            inv.observe(TextEvent(text="say-1"))
            inv.observe(TurnEndEvent())
            inv.observe(ThinkingEvent(text="thought-2"))
            inv.observe(TextEvent(text="say-2"))
            inv.observe(TurnEndEvent())
    agent = [s for s in recorder.steps if s.source == "agent"]
    assert [s.reasoning_content for s in agent] == ["thought-1", "thought-2"]
    assert [s.message for s in agent] == ["say-1", "say-2"]


async def test_tool_call_spans_turn_boundary_stays_with_its_turn(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="00000000-0000-0000-0000-000000000003",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Calling tool..."))
            inv.observe(ToolStartEvent(id="tool_1", name="Read", input={"path": "/x"}))
            inv.observe(TurnEndEvent())
            inv.observe(ToolResultEvent(id="tool_1", output="contents", is_error=False))
            inv.observe(TextEvent(text="Done."))
            inv.observe(TurnEndEvent())
    agent = [s for s in recorder.steps if s.source == "agent"]
    assert len(agent) == 2
    assert agent[0].tool_calls is not None and agent[0].tool_calls[0].tool_call_id == "tool_1"
    assert agent[0].observation is not None
    assert agent[0].observation.results[0].source_call_id == "tool_1"

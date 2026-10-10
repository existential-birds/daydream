"""Recorder closes one Step per TurnEndEvent — never collapses multi-turn invocations."""
from pathlib import Path

from daydream.backends import TextEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder


async def test_tool_call_spans_turn_boundary_stays_with_its_turn(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path, agent_model_name="test-model")
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

"""Recorder closes one Step per TurnEndEvent — never collapses multi-turn invocations."""
from pathlib import Path

from daydream.backends import MetricsEvent, TextEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder


async def test_tool_calls_spanning_turn_boundaries_stay_with_their_turns(tmp_path: Path) -> None:
    """Delayed results find their closed host step, and a late native model relabels closed steps."""
    recorder = make_recorder(tmp_path, agent_model_name="claude")
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Calling tool..."))
            inv.observe(ToolStartEvent(id="tool_1", name="Read", input={"path": "/x"}))
            inv.observe(TurnEndEvent())
            inv.observe(ToolStartEvent(id="tool_2", name="Read", input={"path": "/y"}))
            inv.observe(TurnEndEvent())
            inv.observe(ToolResultEvent(id="tool_2", output="second", is_error=False))
            inv.observe(ToolResultEvent(id="tool_1", output="first", is_error=False))
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(MetricsEvent(message_id="", prompt_tokens=1, completion_tokens=1, cached_tokens=None,
                                     cost_usd=None, model_name="native-model", usage_scope="invocation"))
            inv.observe(TextEvent(text="Done."))
            inv.observe(TurnEndEvent())
    agent = [s for s in recorder.steps if s.source == "agent"]
    assert len(agent) == 3
    for step, call_id, output in ((agent[0], "tool_1", "first"), (agent[1], "tool_2", "second")):
        assert step.tool_calls is not None and [call.tool_call_id for call in step.tool_calls] == [call_id]
        assert step.observation is not None
        assert [(result.source_call_id, result.content) for result in step.observation.results] == [(call_id, output)]
    assert [step.model_name for step in agent] == ["native-model"] * 3

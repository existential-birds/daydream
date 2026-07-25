"""Tests for the shared ``ScriptedBackend`` harness.

The harness is depended on by ~30 test modules, so its own semantics — turn
sequencing, last-turn repeat, mid-stream raise, argument recording — are pinned
here rather than left to be inferred from its consumers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend


async def _drain(backend: ScriptedBackend, prompt: str = "go", **kwargs: Any) -> list[AgentEvent]:
    return [event async for event in backend.execute(Path("/tmp"), prompt, **kwargs)]


def _texts(events: list[AgentEvent]) -> list[str]:
    return [event.text for event in events if isinstance(event, TextEvent)]


@pytest.mark.asyncio
async def test_turns_advance_then_the_last_turn_repeats() -> None:
    """Each call consumes the next turn; calls past the script re-serve the final one."""
    backend = ScriptedBackend(
        script=[
            [TextEvent(text="first")],
            [TextEvent(text="second")],
        ]
    )

    assert _texts(await _drain(backend)) == ["first"]
    assert _texts(await _drain(backend)) == ["second"]
    assert _texts(await _drain(backend)) == ["second"]
    assert backend.call_count == 3


@pytest.mark.asyncio
async def test_default_script_yields_a_bare_result_event() -> None:
    """The no-argument backend satisfies run_agent without a caller-supplied script."""
    events = await _drain(ScriptedBackend())

    assert len(events) == 1
    assert isinstance(events[0], ResultEvent)


@pytest.mark.asyncio
async def test_an_exception_in_a_turn_raises_after_earlier_events_are_yielded() -> None:
    """A turn can emit partial output and then fail — the retry path's real shape."""
    backend = ScriptedBackend(
        script=[
            [TextEvent(text="partial"), RuntimeError("boom"), TextEvent(text="unreached")],
            [TextEvent(text="recovered")],
        ]
    )

    seen: list[AgentEvent] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for event in backend.execute(Path("/tmp"), "go"):
            seen.append(event)

    assert _texts(seen) == ["partial"], "events before the exception must still reach the consumer"
    assert _texts(await _drain(backend)) == ["recovered"]


@pytest.mark.asyncio
async def test_events_shorthand_replays_the_same_stream_every_call() -> None:
    backend = ScriptedBackend(
        events=[TextEvent(text="same"), ResultEvent(structured_output=None, continuation=None)]
    )

    assert _texts(await _drain(backend)) == ["same"]
    assert _texts(await _drain(backend)) == ["same"]


def test_script_and_events_together_is_rejected() -> None:
    """Two sources of truth for the script would silently drop one."""
    with pytest.raises(ValueError, match="not both"):
        ScriptedBackend(script=[[TextEvent(text="a")]], events=[TextEvent(text="b")])


@pytest.mark.asyncio
async def test_every_execute_argument_is_recorded() -> None:
    """The recorders replace the per-test ``captured_*`` nonlocal lists."""
    backend = ScriptedBackend()

    await _drain(backend, "first", max_turns=7, output_schema={"type": "object"}, read_only=True)
    await _drain(backend, "second")

    assert backend.prompts == ["first", "second"]
    assert backend.last_prompt == "second"
    assert backend.max_turns == [7, None]
    assert backend.schemas == [{"type": "object"}, None]
    assert backend.continuations == [None, None]
    assert backend.calls[0]["read_only"] is True
    assert backend.calls[0]["persist_session"] is True


def test_extra_attrs_are_set_for_the_optional_protocol_extensions() -> None:
    """Backends advertise opt-in hints as plain attributes, read via ``getattr``."""
    backend = ScriptedBackend(model="pi-glm", retry_attempts=1, reasoning_effort="high")

    assert backend.model == "pi-glm"
    assert getattr(backend, "retry_attempts", None) == 1
    assert getattr(backend, "reasoning_effort", None) == "high"
    assert backend.fanout_concurrency == 4


@pytest.mark.asyncio
async def test_scripted_backend_drives_run_agent_end_to_end(tmp_path: Path) -> None:
    """The harness satisfies the real ``run_agent`` seam, not just its own drain helper."""
    backend = ScriptedBackend(
        events=[
            TextEvent(text="Review complete"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )

    output, _, _ = await run_agent(backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW)

    assert output == "Review complete"
    assert backend.last_prompt == "review this"

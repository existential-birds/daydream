"""Pi-compatible finite-review responses without launching a model process."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from typing import Any

from daydream.backends import AgentEvent, ResultEvent
from daydream.backends.pi import PiBackend
from tests.harness.fake_clock import FakeClock


class PacketBackend(PiBackend):
    """Exercise run_agent, native Pi capabilities, and shared deadline accounting."""

    def __init__(self, responses: list[dict[str, Any]], clock: FakeClock | None = None,
                 durations: tuple[float, ...] = ()) -> None:
        super().__init__(model="fixture-model", reasoning_effort="high")
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.clock = clock
        self.durations = durations

    async def execute(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
        # Bind the actual Pi signature so its capability parameters cannot drift.
        bound = inspect.signature(PiBackend.execute).bind(self, *args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        call.pop("self")
        call["schema"] = call.pop("output_schema")
        index = len(self.calls)
        self.calls.append(call)
        if self.clock is not None and index < len(self.durations):
            self.clock.advance(self.durations[index])
        yield ResultEvent(structured_output=self.responses[min(index, len(self.responses) - 1)], continuation=None)

"""Recording Backend fake for orchestration tests.

Each execute consumes one scripted turn; the final turn repeats. Turns contain
events and exceptions, allowing partial output before failure. Schema matching
and responders support fan-out independent of completion order. Prompt-based
review/fix routing belongs to PhaseDispatchBackend."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from daydream.backends import AgentEvent, ResultEvent

Turn = Sequence[AgentEvent | BaseException]

# A per-call hook: returns a ``Turn`` to yield, an async iterator to stream,
# ``None`` to fall through to schema/script selection, or an awaitable of any
# of those (awaited before the turn — the rendezvous shape).
Responder = Callable[..., "Turn | AsyncIterator[AgentEvent] | None | Awaitable[Any]"]

# The default turn: a bare terminal ResultEvent. This is what the ~20 fakes that
# only existed to satisfy the protocol (model-line spies, minimal runner stubs)
# yielded.
_DEFAULT_TURN: Turn = (ResultEvent(structured_output=None, continuation=None),)


class ScriptedBackend:
    """Record execute arguments in calls and expose ordered prompt, continuation,
    turn-limit, schema and read-only projections. Count cancellation separately."""

    def __init__(self, script: Sequence[Turn] | None = None, *, events: Turn | None = None,
        responses_by_schema: Sequence[tuple[dict[str, Any] | None, Turn]] | None = None,
        responder: Responder | None = None, model: str | None = "test-model", fanout_concurrency: int = 4, **attrs: Any,
    ) -> None:
        """Choose either per-call script turns or an events turn repeated on every call.

        Selection order: responder, first equal non-None schema, None-schema fallback,
        then script. Schemas compare by equality without hashing. A responder receives
        all eight execute arguments and may return a turn, async iterator, None, or an
        awaitable of those. Async iterators close with their consumer. With no script,
        emit a bare ResultEvent. Extra attrs advertise optional Backend capabilities."""
        if script is not None and events is not None:
            raise ValueError("pass either script= or events=, not both")
        if events is not None:
            script = [events]
        self._script: list[Turn] = [list(turn) for turn in script] if script else [list(_DEFAULT_TURN)]
        self._responses_by_schema: list[tuple[dict[str, Any] | None, Turn]] = [
            (schema, list(turn)) for schema, turn in (responses_by_schema or [])
        ]
        self._responder = responder
        self.model: Any = model
        self.fanout_concurrency = fanout_concurrency
        for name, value in attrs.items():
            setattr(self, name, value)
        self.calls: list[dict[str, Any]] = []
        self.cancel_calls = 0

    # --- Observables ---------------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompts(self) -> list[str]:
        return [call["prompt"] for call in self.calls]

    @property
    def last_prompt(self) -> str:
        """The most recent prompt, or ``""`` if never called."""
        return self.calls[-1]["prompt"] if self.calls else ""

    @property
    def continuations(self) -> list[Any]:
        return [call["continuation"] for call in self.calls]

    @property
    def max_turns(self) -> list[int | None]:
        return [call["max_turns"] for call in self.calls]

    @property
    def schemas(self) -> list[dict[str, Any] | None]:
        return [call["output_schema"] for call in self.calls]

    @property
    def read_only_calls(self) -> list[bool]:
        return [call["read_only"] for call in self.calls]

    # --- Backend surface -----------------------------------------------------

    async def execute(
        self, cwd: Path, prompt: str, output_schema: dict[str, Any] | None = None, continuation: Any = None,
        agents: Any = None, max_turns: int | None = None, read_only: bool = False, persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.calls.append({"cwd": cwd, "prompt": prompt, "output_schema": output_schema, "continuation": continuation,
                "agents": agents, "max_turns": max_turns, "read_only": read_only, "persist_session": persist_session,
            }
        )
        index = min(len(self.calls) - 1, len(self._script) - 1)
        responded: Any = None
        if self._responder is not None:
            responded = self._responder(
                cwd, prompt, output_schema, continuation, agents, max_turns, read_only, persist_session
            )
            if inspect.isawaitable(responded):
                responded = await responded
        if responded is None:
            turn = self._turn_for_schema(output_schema)
            if turn is None:
                turn = self._script[index]
            async for item in self._raise_or_yield(turn):
                yield item
            return
        if isinstance(responded, AsyncIterator):
            try:
                async for event in responded:
                    yield event
            finally:
                aclose = getattr(responded, "aclose", None)
                if aclose is not None:
                    await aclose()
            return
        async for item in self._raise_or_yield(responded):
            yield item

    @staticmethod
    async def _raise_or_yield(turn: Iterable[AgentEvent | BaseException]) -> AsyncIterator[AgentEvent]:
        """Raise scripted exceptions at their exact position in either selected or
        responder-provided turns."""
        for item in turn:
            if isinstance(item, BaseException):
                raise item
            yield item

    def _turn_for_schema(self, output_schema: dict[str, Any] | None) -> Turn | None:
        """Select the first equal schema, then the None-key fallback; otherwise use the script."""
        for schema, turn in self._responses_by_schema:
            if schema is not None and schema == output_schema:
                return turn
        for schema, turn in self._responses_by_schema:
            if schema is None:
                return turn
        return None

    async def cancel(self) -> None:
        self.cancel_calls += 1

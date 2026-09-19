"""Shared scripted fake ``Backend`` for tests that mock the agent seam.

``ScriptedBackend`` yields pre-built events and records its calls so tests can
exercise orchestration through the production seams.

``ScriptedBackend`` is the *scripted* fake: it yields a pre-built turn script,
records what it was called with, can key responses by the call's
``output_schema`` (a constructor seam) so a parallel fan-out whose completion
order is not fixed still gets the right turn, and can hand a call to a
per-call ``responder`` that returns a turn, streams its own async iterator, or
declines (``None``) so the schema/script selection decides. Prompt-heuristic
routing for the shallow review-fix-test loop stays
``tests.harness.phase_backend.PhaseDispatchBackend``'s job, and phase-keyed
replay of real driver output stays ``tests.harness.phase_replay``'s.

A *script* is a list of turns, one per ``execute`` call. A turn is a sequence of
items, each either an ``AgentEvent`` to yield or a ``BaseException`` to raise at
that point in the stream (so "yield partial text, then fail" is expressible).
Once the script is exhausted the final turn repeats, which is what the
``if call_count == 1: ... else: ...`` fakes were all encoding by hand.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
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
    """Recording fake ``Backend`` driven by a per-call turn script.

    Attributes:
        model: Model name reported to the recorder and the ``Model:`` UI line.
        calls: One ``dict`` per ``execute`` call, capturing every argument
            (``cwd``, ``prompt``, ``output_schema``, ``continuation``,
            ``agents``, ``max_turns``, ``read_only``, ``persist_session``).
        prompts: Prompts in call order.
        continuations: ``continuation`` arguments in call order — the observable
            that fresh-context tests assert on.
        max_turns: ``max_turns`` arguments in call order.
        schemas: ``output_schema`` arguments in call order.
        call_count: Total ``execute`` invocations.
        cancel_calls: Total ``cancel`` invocations.
    """

    def __init__(
        self,
        script: Sequence[Turn] | None = None,
        *,
        events: Turn | None = None,
        responses_by_schema: Sequence[tuple[dict[str, Any] | None, Turn]] | None = None,
        responder: Responder | None = None,
        model: str = "test-model",
        fanout_concurrency: int = 4,
        **attrs: Any,
    ) -> None:
        """Configure the fake.

        Args:
            script: One turn per ``execute`` call; the last turn repeats once
                exhausted. Defaults to a single bare ``ResultEvent`` turn.
            events: Shorthand for a one-turn script (``script=[events]``) — the
                every-call-yields-the-same-stream mode. Mutually exclusive with
                ``script``.
            responses_by_schema: Ordered pairs, each a ``(output_schema, turn)``
                whose first pair comparing ``==`` to a call's ``output_schema``
                supplies that call's turn. A ``None`` key is the fallback for
                any unmatched call. With no match (and no fallback) the normal
                per-call script selection applies. Matching never hashes the
                schema, so the real dict schemas work as keys.
            responder: Optional per-call hook, called with the same eight
                ``execute`` arguments. Returns a ``Turn`` (yielded, so an
                exception inside it raises mid-stream), an async iterator
                (streamed and closed with the consumer), or ``None`` to fall
                through to ``responses_by_schema`` and then the script. An
                awaitable result is awaited first (a rendezvous before the
                turn).
            model: Value of the ``model`` attribute.
            fanout_concurrency: The optional ``Backend`` scheduling hint.
            **attrs: Extra instance attributes, for the optional protocol
                extensions a given test needs the backend to advertise
                (``retry_attempts``, ``reasoning_effort``,
                ``concise_fix_prompts``, ...).

        Raises:
            ValueError: If both ``script`` and ``events`` are given.
        """
        if script is not None and events is not None:
            raise ValueError("pass either script= or events=, not both")
        if events is not None:
            script = [events]
        self._script: list[Turn] = [list(turn) for turn in script] if script else [list(_DEFAULT_TURN)]
        self._responses_by_schema: list[tuple[dict[str, Any] | None, Turn]] = [
            (schema, list(turn)) for schema, turn in (responses_by_schema or [])
        ]
        self._responder = responder
        self.model = model
        self.fanout_concurrency = fanout_concurrency
        for name, value in attrs.items():
            setattr(self, name, value)
        self.calls: list[dict[str, Any]] = []
        self.cancel_calls = 0

    # --- Observables ---------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total ``execute`` invocations."""
        return len(self.calls)

    @property
    def prompts(self) -> list[str]:
        """Prompts in call order."""
        return [call["prompt"] for call in self.calls]

    @property
    def last_prompt(self) -> str:
        """The most recent prompt, or ``""`` if never called."""
        return self.calls[-1]["prompt"] if self.calls else ""

    @property
    def continuations(self) -> list[Any]:
        """``continuation`` arguments in call order."""
        return [call["continuation"] for call in self.calls]

    @property
    def max_turns(self) -> list[int | None]:
        """``max_turns`` arguments in call order."""
        return [call["max_turns"] for call in self.calls]

    @property
    def schemas(self) -> list[dict[str, Any] | None]:
        """``output_schema`` arguments in call order."""
        return [call["output_schema"] for call in self.calls]

    @property
    def read_only_calls(self) -> list[bool]:
        """``read_only`` arguments in call order (the fix-gate write-guard observable)."""
        return [call["read_only"] for call in self.calls]

    # --- Backend surface -----------------------------------------------------

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.calls.append(
            {
                "cwd": cwd,
                "prompt": prompt,
                "output_schema": output_schema,
                "continuation": continuation,
                "agents": agents,
                "max_turns": max_turns,
                "read_only": read_only,
                "persist_session": persist_session,
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
            for item in turn:
                if isinstance(item, BaseException):
                    raise item
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
        for item in responded:
            if isinstance(item, BaseException):
                raise item
            yield item

    def _turn_for_schema(self, output_schema: dict[str, Any] | None) -> Turn | None:
        """The turn for ``output_schema``, or ``None`` to fall through to the script.

        The first pair comparing ``==`` wins; a ``None``-keyed pair is the
        fallback for any unmatched schema.
        """
        for schema, turn in self._responses_by_schema:
            if schema is not None and schema == output_schema:
                return turn
        for schema, turn in self._responses_by_schema:
            if schema is None:
                return turn
        return None

    async def cancel(self) -> None:
        self.cancel_calls += 1

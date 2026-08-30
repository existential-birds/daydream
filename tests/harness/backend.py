"""Shared scripted fake ``Backend`` for tests that mock the agent seam.

``ScriptedBackend`` yields pre-built events and records its calls so tests can
exercise orchestration through the production seams.

``ScriptedBackend`` is the *scripted* fake: it yields a pre-built turn script
and records what it was called with. It is deliberately not a dispatch fake —
prompt-heuristic routing for the shallow review-fix-test loop lives in
``tests.harness.phase_backend.PhaseDispatchBackend``, and phase-keyed replay of
real driver output lives in ``tests.harness.phase_replay``.

A *script* is a list of turns, one per ``execute`` call. A turn is a sequence of
items, each either an ``AgentEvent`` to yield or a ``BaseException`` to raise at
that point in the stream (so "yield partial text, then fail" is expressible).
Once the script is exhausted the final turn repeats, which is what the
``if call_count == 1: ... else: ...`` fakes were all encoding by hand.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any

from daydream.backends import AgentEvent, ResultEvent

Turn = Sequence[AgentEvent | BaseException]

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
    """

    def __init__(
        self,
        script: Sequence[Turn] | None = None,
        *,
        events: Turn | None = None,
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
        self.model = model
        self.fanout_concurrency = fanout_concurrency
        for name, value in attrs.items():
            setattr(self, name, value)
        self.calls: list[dict[str, Any]] = []

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
        for item in self._script[index]:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def cancel(self) -> None:
        pass

"""Phase-keyed replay context managers — the harness core.

These context managers inject a *real* backend through the
``daydream.runner.create_backend`` seam and stub only that backend's external
boundary (the Codex subprocess / the Claude SDK client). The stub is keyed on
the *firing* phase, read live from ``TrajectoryRecorder.current_phase()``: when
``CodexBackend.execute`` (or the Claude analog) iterates, the active invocation
opened by ``run_agent`` (``daydream/agent.py:432-436``) makes the phase readable
via ``get_current_recorder().current_phase()`` (see the ordering note in
``agent.py:415-436``). The factory serves that phase's synthesized fixture.

Limitation (matches the spec's rejected per-phase response *queue* decision):
ONE response per phase per firing. A phase that fires more than once replays the
SAME fixture each time — there is no per-iteration variation. The single place a
loop legitimately needs a per-iteration sequence (the shallow loop's parse
queue) is served by ``PhaseDispatchBackend`` (Task 10), deliberately kept out of
this phase-keyed *replay* harness.

A phase that fires with no entry in the map is a TEST BUG, not a silent no-op:
the factory raises ``AssertionError`` naming the missing phase rather than
serving an empty stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from daydream.backends import Backend
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend
from daydream.trajectory import DaydreamPhase, get_current_recorder
from tests.contract._loaders import (
    _MockAssistantMessage,
    _MockResultMessage,
    _MockTextBlock,
    _MockThinkingBlock,
    _MockToolResultBlock,
    _MockToolUseBlock,
    _MockUserMessage,
)
from tests.harness.codex_replay import make_mock_process
from tests.harness.scripts import (
    PhaseScripts,
    build_claude_messages_for_phase,
    render_codex,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


def _firing_phase() -> DaydreamPhase:
    """Read the firing phase from the active recorder, or fail loudly.

    Reads ``get_current_recorder().current_phase()`` — the public read-seam set
    by the ``run_agent`` invocation scope. Raises ``AssertionError`` when no
    recorder/phase is active, since the replay context managers are only valid
    inside an open recorder + invocation.
    """
    recorder = get_current_recorder()
    if recorder is None:
        raise AssertionError(
            "no active TrajectoryRecorder — phase-keyed replay requires the boundary "
            "stub to fire inside an open recorder invocation"
        )
    phase = recorder.current_phase()
    if phase is None:
        raise AssertionError(
            "no active invocation phase — the boundary stub fired outside a run_agent "
            "invocation scope"
        )
    return phase


def codex_subprocess_for_phases(phase_scripts: PhaseScripts) -> AbstractContextManager[Any]:
    """Patch the Codex subprocess boundary to serve per-phase fixtures.

    Patches ``daydream.backends.codex.asyncio.create_subprocess_exec`` with a
    ``side_effect`` factory that, on each subprocess launch, reads the firing
    phase via ``current_phase()`` and returns ``make_mock_process`` over that
    phase's rendered Codex JSONL lines.

    Args:
        phase_scripts: ``{DaydreamPhase: script}`` map. Each script is rendered
            to Codex JSONL once, up front.

    Raises:
        AssertionError: At launch time, when the firing phase is absent from
            *phase_scripts* (a phase firing with no fixture is a test bug, not a
            silent empty stream).
    """
    rendered = render_codex(phase_scripts)

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        phase = _firing_phase()
        if phase not in rendered:
            raise AssertionError(
                f"Codex subprocess fired for phase {phase.name} with no fixture in "
                f"phase_scripts (have: {sorted(p.name for p in rendered)})"
            )
        return make_mock_process(rendered[phase])

    return patch(
        "daydream.backends.codex.asyncio.create_subprocess_exec",
        side_effect=factory,
    )


def claude_messages_for_phases(phase_scripts: PhaseScripts) -> AbstractContextManager[Any]:
    """Patch the Claude SDK client to serve per-phase message streams.

    The SDK-patch analog of :func:`codex_subprocess_for_phases`, mirroring
    ``tests/contract/_loaders.py:167-176``. The scripted client's
    ``receive_response`` reads the firing phase via ``current_phase()`` and
    replays that phase's synthesized Claude messages.

    Args:
        phase_scripts: ``{DaydreamPhase: script}`` map. Each script is rendered
            to Claude SDK messages once, up front.

    Raises:
        AssertionError: When the firing phase is absent from *phase_scripts*.
    """
    rendered = {
        phase: build_claude_messages_for_phase(script)
        for phase, script in phase_scripts.items()
    }

    class _ScriptedClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options

        async def __aenter__(self) -> "_ScriptedClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self) -> Any:
            phase = _firing_phase()
            if phase not in rendered:
                raise AssertionError(
                    f"Claude SDK fired for phase {phase.name} with no fixture in "
                    f"phase_scripts (have: {sorted(p.name for p in rendered)})"
                )
            for message in rendered[phase]:
                yield message

    return patch.multiple(
        "daydream.backends.claude",
        ClaudeSDKClient=_ScriptedClient,
        AssistantMessage=_MockAssistantMessage,
        UserMessage=_MockUserMessage,
        ResultMessage=_MockResultMessage,
        TextBlock=_MockTextBlock,
        ThinkingBlock=_MockThinkingBlock,
        ToolUseBlock=_MockToolUseBlock,
        ToolResultBlock=_MockToolResultBlock,
    )


def replay_through_runner(
    driver: str, phase_scripts: PhaseScripts
) -> AbstractContextManager[Any]:
    """Replay *phase_scripts* through ``runner.run`` for the given *driver*.

    Combines the driver's boundary patch with a patch of
    ``daydream.runner.create_backend`` so ``runner.run`` builds the real backend
    while its external boundary is stubbed per phase. Composes as a single
    context manager so a ``with replay_through_runner(...):`` block wraps a
    ``run(config)`` call end-to-end.

    Args:
        driver: ``"codex"`` or ``"claude"``.
        phase_scripts: ``{DaydreamPhase: script}`` map served per firing phase.

    Returns:
        A context manager combining the create_backend seam and the boundary
        stub.

    Raises:
        ValueError: When *driver* is not ``"codex"`` or ``"claude"``.
    """
    if driver == "codex":
        boundary = codex_subprocess_for_phases(phase_scripts)

        def make_backend(name: str, model: Any = None) -> Backend:
            return CodexBackend(model=model)
    elif driver == "claude":
        boundary = claude_messages_for_phases(phase_scripts)

        def make_backend(name: str, model: Any = None) -> Backend:
            return ClaudeBackend(model=model)
    else:
        raise ValueError(f"unknown driver {driver!r}; expected 'codex' or 'claude'")

    create_backend_patch = patch(
        "daydream.runner.create_backend",
        side_effect=make_backend,
    )

    class _Combined:
        def __enter__(self) -> None:
            boundary.__enter__()
            create_backend_patch.__enter__()

        def __exit__(self, *exc: Any) -> None:
            create_backend_patch.__exit__(*exc)
            boundary.__exit__(*exc)

    return _Combined()

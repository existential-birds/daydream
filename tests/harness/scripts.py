"""Multi-phase canonical-script synthesis with structured-output support.

Extends the single-script synthesis in ``tests/contract/_loaders.py`` (reused,
not copied) to:

1. Render a ``{phase: script}`` map to per-phase native message streams
   (``render_codex``), so the replay harness can serve each firing phase its
   own fixture.
2. Thread a script-level ``structured_output`` dict onto the final turn so the
   REAL backend parser (``codex.py:348-368`` / Claude ``ResultMessage``)
   extracts it — the parity path for the PARSE phase's schema-constrained
   return.

The structured-output path emits the final ``agent_message`` text as
``json.dumps(structured_output)`` for Codex (the real parser ``json.loads`` the
last agent text) and sets ``ResultMessage.structured_output`` for Claude. No
``structured_output`` key => behaviour identical to today's single-script
synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from daydream.backends import AgentEvent
from daydream.backends.codex import CodexBackend
from daydream.trajectory import DaydreamPhase
from tests.contract._loaders import _build_claude_messages, _build_codex_jsonl
from tests.harness.codex_replay import make_mock_process

PhaseScripts = dict[DaydreamPhase, dict[str, Any]]


def _with_structured_output(script: dict[str, Any]) -> dict[str, Any]:
    """Return a script whose final turn carries ``json.dumps(structured_output)``.

    The real Codex parser extracts structured output by ``json.loads`` of the
    last ``agent_message`` text (``codex.py:349-353``). To satisfy it we
    overwrite the final turn's ``text`` with the serialized structured payload.
    Scripts without a ``structured_output`` key are returned unchanged.
    """
    structured = script.get("structured_output")
    if structured is None:
        return script
    turns = script["turns"]
    if not turns:
        raise AssertionError("structured_output requested but script has no turns to carry it")
    serialized = json.dumps(structured)
    new_turns = [dict(t) for t in turns]
    new_turns[-1]["text"] = serialized
    return {**script, "turns": new_turns}


def build_codex_jsonl_for_phase(script: dict[str, Any]) -> list[str]:
    """Synthesize Codex JSONL lines for one phase's script.

    Reuses ``tests/contract/_loaders.py:_build_codex_jsonl``. When
    ``script["structured_output"]`` is set, the final ``agent_message``
    ``item.completed`` text is ``json.dumps(structured_output)`` so the real
    backend parser extracts it on ``ResultEvent.structured_output``.

    A script carrying ``raw_lines`` is a RECORDED-REAL passthrough: the lines
    are replayed verbatim (no synthesis, no structured-output rewrite) so the
    genuine parser sees the captured real-CLI bytes. ``raw_lines`` and the
    synthesized keys (``turns``/``structured_output``) are mutually exclusive.
    """
    raw_lines = script.get("raw_lines")
    if raw_lines is not None:
        if "turns" in script:
            raise AssertionError(
                "a raw_lines passthrough script must not also carry synthesized 'turns'"
            )
        return list(raw_lines)
    return _build_codex_jsonl(_with_structured_output(script))


def build_claude_messages_for_phase(script: dict[str, Any]) -> list[Any]:
    """Synthesize Claude SDK messages for one phase's script (parity path).

    Reuses ``tests/contract/_loaders.py:_build_claude_messages`` and threads
    ``script["structured_output"]`` onto the trailing ``ResultMessage`` so the
    real Claude backend yields it on ``ResultEvent.structured_output``.
    """
    messages = _build_claude_messages(script)
    structured = script.get("structured_output")
    if structured is not None:
        # The trailing message is the ResultMessage (see _build_claude_messages).
        messages[-1].structured_output = structured
    return messages


def render_codex(phase_scripts: PhaseScripts) -> dict[DaydreamPhase, list[str]]:
    """Render a ``{phase: script}`` map to ``{phase: codex_jsonl_lines}``."""
    return {phase: build_codex_jsonl_for_phase(script) for phase, script in phase_scripts.items()}


async def drive_codex(
    lines: list[str], output_schema: dict[str, Any] | None = None
) -> list[AgentEvent]:
    """Drive a real ``CodexBackend`` over *lines*, collecting emitted events.

    Thin helper: ``make_mock_process`` (the consolidated builder from Task 3) +
    a patch of the Codex subprocess boundary + a real ``CodexBackend``. No
    swallowing — events flow straight from the genuine parser, so a missing or
    malformed structured output surfaces as a failed assertion in the caller.

    Args:
        lines: JSONL lines to replay through the mocked subprocess stdout.
        output_schema: Optional schema passed to ``execute`` so the backend
            attempts structured-output extraction.

    Returns:
        The list of ``AgentEvent`` instances the backend emitted.
    """
    mock_proc = make_mock_process(lines)
    backend = CodexBackend(model="codex-test-model")
    events: list[AgentEvent] = []
    with patch(
        "daydream.backends.codex.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        async for event in backend.execute(Path("/tmp"), "go", output_schema=output_schema):
            events.append(event)
    return events

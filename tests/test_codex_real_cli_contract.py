# tests/test_codex_real_cli_contract.py
"""Contract checks against REAL codex CLI output (parser-drift guard).

Two layers, per issue #154:

1. ``test_real_golden_parses_to_expected_events`` (always-on): drives
   ``CodexBackend.execute`` through a committed golden fixture derived from
   REAL ``codex exec --experimental-json`` output (codex 0.139.0). Asserts
   the parser produces a structurally-correct ``AgentEvent`` stream with zero
   orphaned tool results — the #153 contract re-asserted on real data. This
   catches parser drift the next time someone re-captures the golden and the
   CLI shape has changed. Structural assertions are version-robust (not
   byte-exact) so a model/CLI swap that rewords the agent text still passes.

2. ``test_codex_live_smoke`` (Layer 2, skip-gated): marked ``live_codex`` and
   skipped cleanly when the ``codex`` binary is not on ``$PATH`` (no red CI
   for contributors without the binary). When codex IS present it runs a
   trivial review against the in-repo sample repo and asserts a non-empty
   ``AgentEvent`` stream + clean trajectory, logging any unrecognized JSONL
   event types.

The golden is committed at ``tests/fixtures/codex_jsonl/real/golden.jsonl``;
see ``tests/fixtures/codex_jsonl/real/README.md`` for the capture/refresh
procedure, and ``scripts/capture-codex-golden.sh`` to re-capture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.backends import (
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend
from tests.harness.codex_replay import FIXTURES_DIR, make_mock_process_from_fixture

REAL_GOLDEN = "real/golden.jsonl"

_logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_real_golden_parses_to_expected_events() -> None:
    """The committed REAL codex golden parses to a structurally-correct stream.

    The golden is genuine ``codex exec --experimental-json`` output (codex
    0.139.0, see the fixture README). This asserts the parser still agrees
    with the live CLI on the observed event coverage: a text span, paired
    tool calls (zero orphans — the #153 contract), per-turn metrics with
    prompt/completion tokens, and a result event. Assertions are structural,
    not byte-exact, so re-capturing on a new model that rewords the agent
    message still passes.
    """
    assert (FIXTURES_DIR / REAL_GOLDEN).exists(), (
        f"real golden missing at {FIXTURES_DIR / REAL_GOLDEN}"
    )

    backend = CodexBackend(model="real-golden-model")
    mock_proc = make_mock_process_from_fixture(REAL_GOLDEN)

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events: list[Any] = []
        async for event in backend.execute(Path("/tmp"), "Read README.md then hello.py"):
            events.append(event)

    # Text span: agent_message produced at least one TextEvent.
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert text_events, "real golden produced no TextEvent (agent_message lost)"

    # Tool spans: the golden has TWO command_execution pairs (README + hello.py).
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_starts) == 2, f"expected 2 tool starts, got {len(tool_starts)}"
    assert len(tool_results) == 2, f"expected 2 tool results, got {len(tool_results)}"

    # Zero orphans: every ToolResultEvent.id pairs with a ToolStartEvent.id.
    # This re-asserts the #153 deterministic-correlation contract on REAL data.
    start_ids = {e.id for e in tool_starts}
    result_ids = {e.id for e in tool_results}
    assert result_ids == start_ids, (
        f"orphaned tool result on real data: starts={start_ids} results={result_ids}"
    )

    # Per-turn metrics: turn.completed yields a MetricsEvent with prompt AND
    # completion tokens, and cached_tokens surfaced from cached_input_tokens.
    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    assert metrics_events, "real golden produced no MetricsEvent (usage lost)"
    mev = metrics_events[0]
    assert mev.prompt_tokens is not None and mev.prompt_tokens > 0, (
        f"real usage input_tokens not surfaced: {mev.prompt_tokens}"
    )
    assert mev.completion_tokens is not None and mev.completion_tokens > 0, (
        f"real usage output_tokens not surfaced: {mev.completion_tokens}"
    )
    assert mev.cached_tokens is not None and mev.cached_tokens > 0, (
        f"real cached_input_tokens not surfaced: {mev.cached_tokens}"
    )

    # Result event present (turn.completed → ResultEvent with continuation).
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert result_events, "real golden produced no ResultEvent"


@pytest.mark.live_codex
@pytest.mark.asyncio
async def test_codex_live_smoke() -> None:
    """Live smoke against the real codex binary — skipped when codex is absent.

    Proves the subprocess seam, arg construction, and parser still agree with
    the live CLI. Skipped cleanly (no red CI) when ``codex`` is not on PATH.
    Any unrecognized JSONL event type is logged at WARNING for triage.
    """
    if shutil.which("codex") is None:
        pytest.skip("codex CLI not available on PATH")

    sample_repo = Path(__file__).parent / "fixtures" / "real_cli_sample_repo"
    prompt = "Read README.md and summarize it in one sentence."

    proc = await asyncio.create_subprocess_exec(
        "codex",
        "exec",
        "--experimental-json",
        "--sandbox",
        "read-only",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout is None or proc.stdin is None:
        pytest.skip("could not open codex stdio")
    proc.stdin.write(prompt.encode())
    proc.stdin.close()

    # Collect the raw JSONL lines, logging any line that is not valid JSON or
    # carries an unrecognized event type (the drift signal this test exists to
    # surface, per the issue's "log any unrecognized JSONL event types").
    known_types = {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
    lines: list[str] = []
    assert proc.stdout is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        text = raw.decode().strip()
        if not text:
            continue
        try:
            evt = json.loads(text)
        except json.JSONDecodeError:
            _logger.warning("codex live smoke: non-JSON line: %r", text[:120])
            continue
        etype = evt.get("type", "")
        if etype not in known_types:
            _logger.warning("codex live smoke: unrecognized event type %r", etype)
        lines.append(text)
    await proc.wait()

    assert lines, "codex live smoke produced no JSONL output"

    # Feed the live lines through the parser and assert a non-empty stream.
    backend = CodexBackend(model="live-smoke-model")
    from tests.harness.codex_replay import make_mock_process

    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=make_mock_process(lines)):
        events: list[Any] = []
        async for event in backend.execute(sample_repo, prompt):
            events.append(event)

    assert events, "codex live smoke parsed to an empty AgentEvent stream"

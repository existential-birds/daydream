"""Tests for multi-phase canonical synthesis with structured-output support.

These exercise the harness's per-phase Codex/Claude script synthesis. The
structured-output roundtrip drives the synthesized JSONL through the REAL
``CodexBackend`` parser and asserts the parser extracts the requested dict on
``ResultEvent.structured_output`` — never that a builder was merely called.
"""

from daydream.backends import ResultEvent
from daydream.phases import FEEDBACK_SCHEMA
from tests.harness.scripts import build_codex_jsonl_for_phase, drive_codex


async def test_parse_phase_structured_output_roundtrip():
    script = {
        "turns": [{"message_id": "m1", "text": ""}],
        "structured_output": {"issues": [{"id": 1, "description": "x", "file": "a.py", "line": 1}]},
    }
    lines = build_codex_jsonl_for_phase(script)
    events = await drive_codex(lines, output_schema=FEEDBACK_SCHEMA)
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.structured_output == {
        "issues": [{"id": 1, "description": "x", "file": "a.py", "line": 1}]
    }

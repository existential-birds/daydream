"""Tests for multi-phase canonical synthesis with structured-output support.

These exercise the harness's per-phase Codex/Claude script synthesis. The
structured-output roundtrip drives the synthesized JSONL through the REAL
``CodexBackend`` parser and asserts the adapter forwards the requested JSON as assistant text
for host resolution — never that a builder was merely called.
"""

import json

from daydream.backends import ResultEvent, TextEvent
from daydream.phases import FEEDBACK_SCHEMA
from tests.harness.scripts import build_codex_jsonl_for_phase, drive_codex


async def test_parse_phase_structured_output_roundtrip() -> None:
    script = {"turns": [{"message_id": "m1", "text": ""}],
        "structured_output": {"issues": [{"id": 1, "description": "x", "file": "a.py", "line": 1}]},
    }
    lines = build_codex_jsonl_for_phase(script)
    events = await drive_codex(lines, output_schema=FEEDBACK_SCHEMA)
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.structured_output is None
    text = next(event.text for event in events if isinstance(event, TextEvent))
    assert json.loads(text) == {"issues": [{"id": 1, "description": "x", "file": "a.py", "line": 1}]}

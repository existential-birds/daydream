"""Tests for the shared :func:`daydream.json_utils.extract_json` helper.

This helper is shared by the backends (structured-output extraction) and
``run_agent`` (raw-text fallback) — it is not a Pi-specific concern. The tests
target the canonical function directly so they remain valid regardless of any
backend-private wrapper aliases.
"""

from typing import Any

import pytest

from daydream.json_utils import extract_json


class TestExtractJson:
    """Verify extract_json handles clean JSON, fenced JSON, and prose-wrapped JSON."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                '{"findings": [], "ok": true}',
                {"findings": [], "ok": True},
                id="clean-json-object",
            ),
            pytest.param("[1, 2, 3]", [1, 2, 3], id="clean-json-array"),
            pytest.param(
                '```json\n{"findings": [{"arb_id": 1, "keep": true}]}\n```',
                {"findings": [{"arb_id": 1, "keep": True}]},
                id="markdown-fenced-json",
            ),
            pytest.param('```\n{"x": 1}\n```', {"x": 1}, id="markdown-fenced-bare"),
            pytest.param(
                "Based on my analysis of all findings, here are my verdicts:\n"
                '{"findings": [{"arb_id": 1, "keep": false}]}',
                {"findings": [{"arb_id": 1, "keep": False}]},
                id="prose-wrapped-json",
            ),
            pytest.param(
                'Here are the issues:\n[{"id": 1, "severity": "high"}]\nThat concludes the review.',
                [{"id": 1, "severity": "high"}],
                id="prose-wrapped-array",
            ),
            pytest.param("", None, id="empty-string"),
            pytest.param("   \n  ", None, id="whitespace-only"),
            pytest.param(
                "This is just prose with no JSON whatsoever.",
                None,
                id="no-json-at-all",
            ),
            pytest.param(
                '{"msg": "contains a } brace", "ok": true}',
                {"msg": "contains a } brace", "ok": True},
                id="json-with-nested-braces-in-strings",
            ),
            pytest.param(
                "[1, bad] then [3,4]",
                [3, 4],
                id="unparseable-array-then-valid-array",
            ),
            pytest.param(
                '{bad} then {"issues":[1,2]}',
                {"issues": [1, 2]},
                id="unparseable-object-then-valid-object",
            ),
            pytest.param(
                'prefix {bad {"findings": []}} suffix',
                {"findings": []},
                id="nested-valid-json-inside-balanced-invalid-span",
            ),
            pytest.param(
                'note {"k": 1} then {"findings": [{"arb_id": 1, "keep": false, "x": "y"}]}',
                {"findings": [{"arb_id": 1, "keep": False, "x": "y"}]},
                id="largest-object-wins-over-smaller-earlier-object",
            ),
        ],
    )
    def test_extract_json(self, text: str, expected: Any) -> None:
        """Extract supported JSON wrappers while preserving expected Python values."""
        assert extract_json(text) == expected

    def test_stray_prose_bracket_does_not_beat_the_real_object(self):
        # Regression for the sentry-67876 arbiter crash. The model's prose
        # referenced a code snippet `metadata["sender"]` BEFORE its real fenced
        # answer. The earliest-bracket rule parsed `["sender"]` (a valid 1-element
        # list) and returned that bare list, which crashed the arbiter with
        # "Arbiter returned no findings list (got list)". The largest-span rule
        # must instead return the substantial `{"findings": [...]}` object.
        text = (
            "**Finding 2 (arb_id=2):** Confirmed. Line 503 does "
            '`integration.metadata["sender"]["login"]` with direct subscripting.\n\n'
            "```json\n"
            '{"findings": [{"arb_id": 1, "keep": true}, {"arb_id": 2, "keep": true}]}\n'
            "```"
        )
        result = extract_json(text)
        assert isinstance(result, dict)
        assert [f["arb_id"] for f in result["findings"]] == [1, 2]

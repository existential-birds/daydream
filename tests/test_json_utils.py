"""Tests for the shared :func:`daydream.json_utils.extract_json` helper.

This helper is shared by the backends (structured-output extraction) and
``run_agent`` (raw-text fallback) — it is not a Pi-specific concern. The tests
target the canonical function directly so they remain valid regardless of any
backend-private wrapper aliases.
"""

from daydream.json_utils import extract_json


class TestExtractJson:
    """Verify extract_json handles clean JSON, fenced JSON, and prose-wrapped JSON."""

    def test_clean_json_object(self):
        assert extract_json('{"findings": [], "ok": true}') == {"findings": [], "ok": True}

    def test_clean_json_array(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_markdown_fenced_json(self):
        text = '```json\n{"findings": [{"arb_id": 1, "keep": true}]}\n```'
        result = extract_json(text)
        assert result == {"findings": [{"arb_id": 1, "keep": True}]}

    def test_markdown_fenced_bare(self):
        text = '```\n{"x": 1}\n```'
        assert extract_json(text) == {"x": 1}

    def test_prose_wrapped_json(self):
        text = (
            "Based on my analysis of all findings, here are my verdicts:\n"
            '{"findings": [{"arb_id": 1, "keep": false}]}'
        )
        result = extract_json(text)
        assert result == {"findings": [{"arb_id": 1, "keep": False}]}

    def test_prose_wrapped_array(self):
        text = 'Here are the issues:\n[{"id": 1, "severity": "high"}]\nThat concludes the review.'
        result = extract_json(text)
        assert result == [{"id": 1, "severity": "high"}]

    def test_empty_string(self):
        assert extract_json("") is None

    def test_whitespace_only(self):
        assert extract_json("   \n  ") is None

    def test_no_json_at_all(self):
        assert extract_json("This is just prose with no JSON whatsoever.") is None

    def test_json_with_nested_braces_in_strings(self):
        text = '{"msg": "contains a } brace", "ok": true}'
        result = extract_json(text)
        assert result == {"msg": "contains a } brace", "ok": True}

    def test_unparseable_array_then_valid_array(self):
        # First balanced [...] span is unparseable; must scan forward to the
        # next valid span of the same brace type instead of giving up.
        assert extract_json('[1, bad] then [3,4]') == [3, 4]

    def test_unparseable_object_then_valid_object(self):
        # First balanced {...} span is unparseable; must scan forward to the
        # next valid object of the same brace type (not fall back to the inner
        # array, which would return the wrong type).
        assert extract_json('{bad} then {"issues":[1,2]}') == {"issues": [1, 2]}

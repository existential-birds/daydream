# tests/test_phases.py
"""Tests for phase functions with backend abstraction."""

import pytest

from daydream.backends import (
    ContinuationToken,
    ResultEvent,
    TextEvent,
)
from daydream.config import REVIEW_OUTPUT_FILE


@pytest.mark.asyncio
async def test_phase_test_and_heal_passes_continuation(tmp_path, monkeypatch):
    """Test that continuation token is threaded through test retries."""
    from daydream.phases import phase_test_and_heal

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_menu", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    call_count = 0
    token = ContinuationToken(backend="codex", data={"thread_id": "th_test"})

    class ContinuationBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tests fail
                yield TextEvent(text="1 failed, 0 passed")
                yield ResultEvent(structured_output=None, continuation=token)
            elif call_count == 2:
                # Fix call: should receive the continuation token
                assert continuation is token, f"Expected continuation token on fix call, got {continuation}"
                yield TextEvent(text="Fixed")
                yield ResultEvent(structured_output=None, continuation=token)
            else:
                # Retry: tests pass, should receive token
                assert continuation is token, f"Expected continuation token on retry, got {continuation}"
                yield TextEvent(text="All 1 tests passed")
                yield ResultEvent(structured_output=None, continuation=token)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    # Simulate: fail -> choice "2" (fix and retry) -> pass
    choices = iter(["2"])
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(choices, "3"))

    backend = ContinuationBackend()
    success, retries = await phase_test_and_heal(backend, tmp_path)

    assert success is True
    assert retries == 1
    assert call_count == 3


@pytest.mark.asyncio
async def test_phase_parse_feedback_empty_response_returns_empty_list(tmp_path, monkeypatch):
    """When the agent returns empty text (schema miss), treat as no issues."""
    from daydream.phases import phase_parse_feedback

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    # Write a review file so the prompt references a real path
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Verdict\n\nReady: Yes\n")

    class EmptyResponseBackend:
        """Simulates a schema miss: no structured output, no text."""

        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            # Only yield a result with no structured output (schema miss)
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    result = await phase_parse_feedback(EmptyResponseBackend(), tmp_path)
    assert result == []


@pytest.mark.asyncio
async def test_phase_parse_feedback_json_fallback(tmp_path, monkeypatch):
    """When structured output fails but raw text is valid JSON, parse it."""
    from daydream.phases import phase_parse_feedback

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. [foo.py:10] Bug\n")

    class JsonTextBackend:
        """Simulates a schema miss where the model outputs JSON as plain text."""

        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text='{"issues": [{"id": 1, "description": "Bug", "file": "foo.py", "line": 10}]}')
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    result = await phase_parse_feedback(JsonTextBackend(), tmp_path)
    assert len(result) == 1
    assert result[0]["file"] == "foo.py"

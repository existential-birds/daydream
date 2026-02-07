# tests/test_phases.py
"""Tests for phase functions with backend abstraction."""

import pytest

from daydream.backends import (
    ContinuationToken,
    ResultEvent,
    TextEvent,
)


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

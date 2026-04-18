"""Regression + behavior tests for phase_parse_feedback input_path kwarg (D-21, D-40)."""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.config import REVIEW_OUTPUT_FILE
from daydream.phases import phase_parse_feedback


class _SpyBackend:
    def __init__(self) -> None:
        self.last_prompt: str = ""

    async def execute(self, cwd, prompt, output_schema=None, continuation=None, agents=None):
        from daydream.backends import ResultEvent, TextEvent
        self.last_prompt = prompt
        yield TextEvent(text="parsing")
        yield ResultEvent(
            structured_output={"issues": []},
            continuation=None,
        )

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


@pytest.fixture(autouse=True)
def _silence_ui(monkeypatch):
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )


async def test_input_path_default_uses_review_output_file(tmp_path: Path) -> None:
    """D-40 regression: default call (no kwarg) reads cwd / REVIEW_OUTPUT_FILE."""
    backend = _SpyBackend()
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("# Issues\n1. [a.py:1] x\n")
    await phase_parse_feedback(backend, tmp_path)
    assert str(tmp_path / REVIEW_OUTPUT_FILE) in backend.last_prompt


async def test_input_path_override_used_when_provided(tmp_path: Path) -> None:
    """D-21: input_path overrides the default, enabling per-stack iteration."""
    backend = _SpyBackend()
    custom = tmp_path / ".daydream" / "deep" / "stack-python-review.md"
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_text("# Issues\n1. [api.py:1] x\n")
    await phase_parse_feedback(backend, tmp_path, input_path=custom)
    assert str(custom) in backend.last_prompt
    # Default path is NOT in the prompt when override is used
    assert str(tmp_path / REVIEW_OUTPUT_FILE) not in backend.last_prompt


async def test_input_path_is_keyword_only(tmp_path: Path) -> None:
    """input_path cannot be passed positionally (signature guard)."""
    backend = _SpyBackend()
    with pytest.raises(TypeError):
        await phase_parse_feedback(backend, tmp_path, tmp_path / "other.md")  # type: ignore[misc]

"""Regression + behavior tests for phase_parse_feedback input_path kwarg (D-21, D-40)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.phases import phase_parse_feedback
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend


def _spy_backend() -> ScriptedBackend:
    return ScriptedBackend(
        events=[
            TextEvent(text="parsing"),
            ResultEvent(structured_output={"issues": []}, continuation=None),
        ],
        model="test-model",
    )


@pytest.fixture(autouse=True)
def _silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )


async def test_input_path_default_uses_review_output_file(
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
) -> None:
    """D-40 regression: default call (no kwarg) reads work.repo / REVIEW_OUTPUT_FILE."""
    backend = _spy_backend()
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("# Issues\n1. [a.py:1] x\n")
    await phase_parse_feedback(cast(Backend, backend), make_work(tmp_path), allow_standalone=True)
    assert str(tmp_path / REVIEW_OUTPUT_FILE) in backend.last_prompt


async def test_input_path_override_used_when_provided(tmp_path: Path, make_work: Callable[..., WorkContext]) -> None:
    """D-21: input_path overrides the default, enabling per-stack iteration."""
    backend = _spy_backend()
    custom = tmp_path / ".daydream" / "deep" / "stack-python-review.md"
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_text("# Issues\n1. [api.py:1] x\n")
    await phase_parse_feedback(
        cast(Backend, backend), make_work(tmp_path), input_path=custom, allow_standalone=True
    )
    assert str(custom) in backend.last_prompt
    # Default path is NOT in the prompt when override is used
    assert str(tmp_path / REVIEW_OUTPUT_FILE) not in backend.last_prompt


async def test_input_path_is_keyword_only(tmp_path: Path, make_work: Callable[..., WorkContext]) -> None:
    """input_path cannot be passed positionally (signature guard)."""
    backend = _spy_backend()
    with pytest.raises(TypeError):
        await phase_parse_feedback(backend, make_work(tmp_path), tmp_path / "other.md")  # type: ignore[call-overload]

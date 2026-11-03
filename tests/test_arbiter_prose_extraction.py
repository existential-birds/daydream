"""Prose brackets must not hijack arbiter JSON extraction.

A Pi response can mention metadata["sender"]["login"] before its fenced
findings object. Extraction must select the candidate the requested schema admits.
Tests drive phase_arbiter_review through run_agent with native-shaped text
events; the host chooses schema-valid fallback JSON, while Pi forwards prose.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.phases import phase_arbiter_review
from daydream.phases.review import ReviewOutputError
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend

SELECTED_RECORDS: list[dict[str, Any]] = [{
        "id": "py-1", "description": "OAuth `state` is a deterministic md5; CSRF is defeated.",
        "file": "src/sentry/integrations/github/integration.py", "line": 402, "severity": "high", "confidence": "HIGH",
        "rationale": "signature is md5 over view FQNs, knowable a priori.",
    }, {"id": "py-2", "description": "Unchecked metadata['sender']['login'] raises KeyError -> 500.",
        "file": "src/sentry/integrations/github/integration.py", "line": 502, "severity": "high", "confidence": "HIGH",
        "rationale": "metadata is JSONField(default=dict); sender may be absent.",
    },
]

MALFORMED_MESSAGE = "Sorry, I was unable to complete the adjudication. No JSON here."


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    diff_path = tmp_path / "diff.patch"
    intent_path = tmp_path / "intent.md"
    alternatives_path = tmp_path / "alternatives.json"
    diff_path.write_text("diff --git a/x b/x\n+changed\n")
    intent_path.write_text("# Intent\n")
    alternatives_path.write_text('{"alternatives": []}\n')
    return diff_path, intent_path, alternatives_path


def _prose_only_backend(message: str) -> ScriptedBackend:
    """Pi forwards prose; actual run_agent fallback selects the requested schema."""
    return ScriptedBackend(events=[TextEvent(text=message),
                                  ResultEvent(structured_output=None, continuation=None)], model="glm-5.2")


async def test_arbiter_still_raises_on_genuinely_unparseable_output(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """A message with no JSON yields no findings object; the phase raises, not papers over."""
    diff_path, intent_path, alternatives_path = _write_inputs(tmp_path)
    with pytest.raises(ReviewOutputError) as failure:
        await phase_arbiter_review(
            cast(Backend, _prose_only_backend(MALFORMED_MESSAGE)), make_work(tmp_path),
            selected_records=SELECTED_RECORDS,
            diff_path=diff_path, intent_path=intent_path, alternatives_path=alternatives_path, allow_standalone=True,
        )

    assert failure.value.reason_code.value == "malformed_output"

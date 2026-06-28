"""Regression: prose brackets must not hijack the arbiter's JSON extraction.

Real failure (getsentry/sentry#67876, pi/glm reviewer): the arbiter agent
returned a correct fenced ``{"findings": [...]}`` answer, but its prose first
referenced a code snippet ``integration.metadata["sender"]["login"]``. The pi
backend extracts structured output with ``extract_json(last_assistant_text)``,
whose old "earliest bracket wins" rule parsed ``["sender"]`` — a valid
one-element list — and returned that bare list. ``phase_arbiter_review`` then
crashed at its dict-shape check with
``ValueError("Arbiter returned no findings list (got list)")``.

The fix makes ``extract_json`` return the LARGEST balanced JSON span (the real
answer dwarfs an incidental prose bracket), so the arbiter receives the proper
findings object.

These tests drive the real production path
(``phase_arbiter_review -> run_agent -> backend ResultEvent``). The mock backend
reproduces the pi contract faithfully: ``structured_output = extract_json(text)``
over the model's actual prose-wrapped message. Only the backend is mocked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent, TextEvent
from daydream.json_utils import extract_json
from daydream.phases import phase_arbiter_review

SELECTED_RECORDS: list[dict[str, Any]] = [
    {
        "id": "py-1",
        "description": "OAuth `state` is a deterministic md5; CSRF is defeated.",
        "file": "src/sentry/integrations/github/integration.py",
        "line": 402,
        "severity": "high",
        "confidence": "HIGH",
        "rationale": "signature is md5 over view FQNs, knowable a priori.",
    },
    {
        "id": "py-2",
        "description": "Unchecked metadata['sender']['login'] raises KeyError -> 500.",
        "file": "src/sentry/integrations/github/integration.py",
        "line": 502,
        "severity": "high",
        "confidence": "HIGH",
        "rationale": "metadata is JSONField(default=dict); sender may be absent.",
    },
]

# The model's actual message shape: prose that mentions a bracketed code snippet
# (the `["sender"]` that hijacked the old extractor) BEFORE the fenced answer.
ARBITER_MESSAGE = (
    "I have the intent and the two findings to adjudicate. Both findings are confirmed.\n\n"
    "**Finding 2 (arb_id=2):** Line 503 does "
    '`integration.metadata["sender"]["login"]` with direct subscripting, and '
    "`Integration.metadata` is a JSONField(default=dict), so a missing sender 500s.\n\n"
    "```json\n"
    '{"findings": ['
    '{"arb_id": 1, "keep": true, "severity": "high", "confidence": "HIGH",'
    ' "description": "OAuth state is a constant md5; CSRF defeated.",'
    ' "rationale": "Reproduced the hardcoded signature from open-source FQNs."},'
    '{"arb_id": 2, "keep": true, "severity": "high", "confidence": "HIGH",'
    ' "description": "Unchecked metadata sender subscript 500s.",'
    ' "rationale": "Fail closed via .get() instead of subscripting."}'
    "]}\n"
    "```"
)

MALFORMED_MESSAGE = "Sorry, I was unable to complete the adjudication. No JSON here."


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    diff_path = tmp_path / "diff.patch"
    intent_path = tmp_path / "intent.md"
    alternatives_path = tmp_path / "alternatives.json"
    diff_path.write_text("diff --git a/x b/x\n+changed\n")
    intent_path.write_text("# Intent\n")
    alternatives_path.write_text('{"alternatives": []}\n')
    return diff_path, intent_path, alternatives_path


class _PiLikeBackend:
    """Mirrors the pi backend: structured_output = extract_json(final text)."""

    model = "glm-5.2"
    fanout_concurrency = 4

    def __init__(self, message: str) -> None:
        self._message = message

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        yield TextEvent(text=self._message)
        structured = extract_json(self._message) if output_schema else None
        yield ResultEvent(structured_output=structured, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_arbiter_extracts_findings_from_prose_wrapped_message(tmp_path: Path, make_work) -> None:
    """The fenced findings object wins over the stray prose bracket; verdicts are produced."""
    diff_path, intent_path, alternatives_path = _write_inputs(tmp_path)
    verdicts = await phase_arbiter_review(
        _PiLikeBackend(ARBITER_MESSAGE),
        make_work(tmp_path),
        selected_records=SELECTED_RECORDS,
        diff_path=diff_path,
        intent_path=intent_path,
        alternatives_path=alternatives_path,
    )
    assert set(verdicts) == {1, 2}
    assert verdicts[1]["keep"] is True
    assert verdicts[2]["keep"] is True
    # The crash signature was a one-element ['sender'] list; ensure we did NOT
    # silently coerce that into a bogus finding.
    assert verdicts[1]["description"].startswith("OAuth state")


async def test_arbiter_still_raises_on_genuinely_unparseable_output(tmp_path: Path, make_work) -> None:
    """A message with no JSON yields no findings object; the phase raises, not papers over."""
    diff_path, intent_path, alternatives_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError):
        await phase_arbiter_review(
            _PiLikeBackend(MALFORMED_MESSAGE),
            make_work(tmp_path),
            selected_records=SELECTED_RECORDS,
            diff_path=diff_path,
            intent_path=intent_path,
            alternatives_path=alternatives_path,
        )

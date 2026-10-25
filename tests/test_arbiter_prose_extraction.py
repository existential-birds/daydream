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

from daydream.agent import StructuredOutputFailure, run_agent
from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.phases import phase_arbiter_review
from daydream.phases.review import ReviewOutputError
from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA
from daydream.trajectory import DaydreamPhase
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
    ' "rationale": "Reproduced the hardcoded signature from open-source FQNs.", "evidence": "integration.py:402"},'
    '{"arb_id": 2, "keep": true, "severity": "high", "confidence": "HIGH",'
    ' "description": "Unchecked metadata sender subscript 500s.",'
    ' "rationale": "Fail closed via .get() instead of subscripting.", "evidence": "integration.py:502"}'
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



def _prose_only_backend(message: str) -> ScriptedBackend:
    """Pi forwards prose; actual run_agent fallback selects the requested schema."""
    return ScriptedBackend(events=[TextEvent(text=message),
                                  ResultEvent(structured_output=None, continuation=None)], model="glm-5.2")


# The issue-1445 message shape: incidental prose JSON (the dependency-impact list
# the old prompt asked for) followed by the real answer, an empty issues array.
PROSE_WITH_INCIDENTAL_JSON = (
    'The dependency impact: ["module", "moduleVersion", "surface"] were inspected. '
    'No defects established. {"issues": []}'
)


async def test_host_fallback_returns_empty_result_under_strict_gate() -> None:
    """A completed review that established no defect persists the empty result, not the prose list."""
    result, _, _ = await run_agent(
        cast(Backend, _prose_only_backend(PROSE_WITH_INCIDENTAL_JSON)), Path("/tmp"), "review",
        phase=DaydreamPhase.DEEP, output_schema=PER_STACK_RECORD_SCHEMA,
        require_full_schema=True, persist_session=False, tool_call_budget=4, wall_budget_s=60,
    )
    assert result == {"issues": []}


async def test_arbiter_extracts_findings_from_prose_wrapped_message(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """The fenced findings object wins over the stray prose bracket; verdicts are produced."""
    diff_path, intent_path, alternatives_path = _write_inputs(tmp_path)
    verdicts, _ = await phase_arbiter_review(
        cast(Backend, _prose_only_backend(ARBITER_MESSAGE)), make_work(tmp_path), selected_records=SELECTED_RECORDS,
        diff_path=diff_path, intent_path=intent_path, alternatives_path=alternatives_path, allow_standalone=True,
    )
    assert set(verdicts) == {1, 2}
    assert verdicts[1]["keep"] is True
    assert verdicts[2]["keep"] is True
    # The crash signature was a one-element ['sender'] list; ensure we did NOT
    # silently coerce that into a bogus finding.
    assert verdicts[1]["description"].startswith("OAuth state")

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


def test_rejection_diagnostic_names_type_and_content_free_reason() -> None:
    failure = StructuredOutputFailure("prose", "malformed_output",
                                      detail="candidate type list failed type at $")
    error = ReviewOutputError(failure)
    assert error.reason_code.value == "malformed_output"      # req 15: no new ReasonCode
    assert "reviewer response did not satisfy its schema" in str(error)
    assert "list" in str(error) and "type at $" in str(error)
    assert '"module"' not in str(error) and "module" not in str(error)  # req 14: no content


def test_rejection_diagnostic_without_detail_is_unchanged() -> None:
    assert str(ReviewOutputError(StructuredOutputFailure("prose", "malformed_output"))) == \
           "malformed_output: reviewer response did not satisfy its schema"

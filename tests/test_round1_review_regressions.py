"""Round-1 deep-review regressions for daydream issue #1445.

Pins the schema-aware selection contract at every structured-output boundary
that combines span extraction with a strict gate.  Each test names the merged
finding it closes:

* ``test_strict_gate_has_one_implementation`` -- finding item:3 (LOW); the
  private alias it originally pinned was removed by round 2's item:1.
* ``test_review_evidence_*``                      -- finding item:2 (MEDIUM).
* ``test_agent_prefers_a_fully_valid_*``          -- finding item:5 (LOW).
"""

import json
import pathlib
from pathlib import Path

import pytest

import daydream.agent as agent_module
import daydream.review_evidence as review_evidence_module
from daydream.backends import TextEvent, TurnEndEvent
from daydream.review_evidence import ReviewEvidence
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend

REVIEW_STRICT_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
    "required": ["findings"],
    "additionalProperties": False,
}

_INLINE_GATE_BODIES = (
    "Draft202012Validator(schema).iter_errors",
    "Draft202012Validator(self.schema).iter_errors",
)


def test_strict_gate_has_one_implementation() -> None:
    """item:3 -- the strict gate is implemented once, in the lowest layer.

    Round 2 (item:1) removed the private alias this test used to pin, so the
    invariant is now asserted directly: no module carries a second spelling and
    no module re-implements the gate inline.
    """
    import daydream.json_utils as json_utils

    assert json_utils.validates_schema({"a": 1}, {"type": "object"})
    assert not hasattr(agent_module, "_validates_schema"), (
        "the alias is a second spelling; consumers import the public helper"
    )
    for module in (agent_module, review_evidence_module):
        assert module.__file__ is not None, f"{module.__name__} has no source file"
        source = pathlib.Path(module.__file__).read_text()
        for body in _INLINE_GATE_BODIES:
            assert body not in source, f"{module.__name__} re-implements the strict gate inline: {body}"


def test_review_evidence_keeps_a_schema_valid_result_embedded_in_prose() -> None:
    """item:2 -- a smaller-but-valid span beats a larger incidental one."""
    incidental = json.dumps({"findings": ["x" * 400] * 40})
    authoritative = json.dumps({"findings": []})
    text = f"Some prose mentioning metadata.\n\n{incidental}\n\nand the answer:\n{authoritative}"

    evidence = ReviewEvidence(REVIEW_STRICT_SCHEMA)
    evidence.observe(TextEvent(text=text))
    evidence.observe(TurnEndEvent())

    assert evidence.checkpoint == {"findings": []}
    assert evidence.valid(evidence.checkpoint)


def test_review_evidence_does_not_checkpoint_an_invalid_span() -> None:
    """item:2 -- the schema gate still rejects; this is selection, not admission."""
    evidence = ReviewEvidence(REVIEW_STRICT_SCHEMA)
    evidence.observe(TextEvent(text=json.dumps({"other": ["y" * 500]})))
    evidence.observe(TurnEndEvent())
    assert evidence.checkpoint is None


def test_review_evidence_without_a_schema_never_checkpoints() -> None:
    """item:2 -- the schema-is-None guard survives the migration."""
    evidence = ReviewEvidence(None)
    evidence.observe(TextEvent(text=json.dumps({"findings": []})))
    evidence.observe(TurnEndEvent())
    assert evidence.checkpoint is None


@pytest.mark.parametrize("require_full_schema", [True, False])
async def test_agent_prefers_a_fully_valid_candidate_over_a_later_salvageable_one(
    tmp_path: Path, require_full_schema: bool,
) -> None:
    """item:5 -- strict-first ordering under the salvage-tolerant gate."""
    authoritative = json.dumps({"findings": ["real finding"]})
    # A later, merely salvageable object: it carries the required key with a list
    # value but violates additionalProperties, so only _salvageable admits it.
    incidental = json.dumps({"findings": [], "evidence_digest": "incidental"})
    raw = f"{authoritative}\n\nAlso, quoting a snippet: {incidental}"

    result, _, _ = await agent_module.run_agent(
        ScriptedBackend(events=[TextEvent(text=raw)]), tmp_path, "review", phase=DaydreamPhase.REVIEW,
        output_schema=REVIEW_STRICT_SCHEMA, require_full_schema=require_full_schema,
    )
    assert result == {"findings": ["real finding"]}


async def test_agent_salvage_gate_still_admits_a_shape_when_nothing_is_fully_valid(tmp_path: Path) -> None:
    """item:5 -- the strict-first scan must not narrow the salvage path."""
    salvageable = json.dumps({"findings": ["a"], "evidence_digest": "d"})
    result, _, _ = await agent_module.run_agent(
        ScriptedBackend(events=[TextEvent(text=salvageable)]), tmp_path, "review", phase=DaydreamPhase.REVIEW,
        output_schema=REVIEW_STRICT_SCHEMA,
    )
    assert result == {"findings": ["a"], "evidence_digest": "d"}


def test_extract_json_and_schema_selection_enumerate_the_same_candidates() -> None:
    """item:4 -- extract_json delegates fence stripping, so the sets agree."""
    import daydream.json_utils as json_utils

    fenced = '```json\n{"a": 1}\n```'
    assert json_utils.extract_json(fenced) == {"a": 1}
    assert json_utils._strip_json_fences(fenced) == '{"a": 1}'
    assert json_utils._strip_json_fences("plain") == "plain"

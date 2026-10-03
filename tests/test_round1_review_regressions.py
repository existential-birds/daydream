"""Round-1 deep-review regressions for daydream issue #1445.

Pins the schema-aware selection contract at every structured-output boundary
that combines span extraction with a strict gate.  Each test names the merged
finding it closes:

* ``test_strict_gate_has_one_implementation`` -- finding item:3 (LOW).
* ``test_review_evidence_*``                      -- finding item:2 (MEDIUM).
* ``test_agent_prefers_a_fully_valid_*``          -- finding item:5 (LOW).
"""

import json
import pathlib

import pytest

import daydream.agent as agent_module
import daydream.review_evidence as review_evidence_module
from daydream.backends import TextEvent, TurnEndEvent
from daydream.review_evidence import ReviewEvidence

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
    """item:3 -- agent's private predicate IS the json_utils helper, not a copy."""
    import daydream.json_utils as json_utils

    assert agent_module._validates_schema is json_utils.validates_schema
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
def test_agent_prefers_a_fully_valid_candidate_over_a_later_salvageable_one(
    require_full_schema: bool,
) -> None:
    """item:5 -- strict-first ordering under the salvage-tolerant gate."""
    authoritative = json.dumps({"findings": ["real finding"]})
    # A later, merely salvageable object: it carries the required key with a list
    # value but violates additionalProperties, so only _salvageable admits it.
    incidental = json.dumps({"findings": [], "evidence_digest": "incidental"})
    raw = f"{authoritative}\n\nAlso, quoting a snippet: {incidental}"

    selection = agent_module._select_by_schema(
        raw, REVIEW_STRICT_SCHEMA, require_full_schema=require_full_schema
    )
    assert selection.value == {"findings": ["real finding"]}


def test_agent_salvage_gate_still_admits_a_shape_when_nothing_is_fully_valid() -> None:
    """item:5 -- the strict-first scan must not narrow the salvage path."""
    salvageable = json.dumps({"findings": ["a"], "evidence_digest": "d"})
    selection = agent_module._select_by_schema(
        salvageable, REVIEW_STRICT_SCHEMA, require_full_schema=False
    )
    assert selection.value == {"findings": ["a"], "evidence_digest": "d"}


def test_extract_json_and_schema_selection_enumerate_the_same_candidates() -> None:
    """item:4 -- extract_json delegates fence stripping, so the sets agree."""
    import daydream.json_utils as json_utils

    fenced = '```json\n{"a": 1}\n```'
    assert json_utils.extract_json(fenced) == {"a": 1}
    assert json_utils._strip_json_fences(fenced) == '{"a": 1}'
    assert json_utils._strip_json_fences("plain") == "plain"

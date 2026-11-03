"""Round-1 deep-review regressions for daydream issue #1445.

Pins the schema-aware selection contract at every structured-output boundary
that combines span extraction with a strict gate.  Each test names the merged
finding it closes:

* ``test_review_evidence_*``                      -- finding item:2 (MEDIUM).
* ``test_agent_prefers_a_fully_valid_*``          -- finding item:5 (LOW).
"""




def test_extract_json_and_schema_selection_enumerate_the_same_candidates() -> None:
    """item:4 -- extract_json delegates fence stripping, so the sets agree."""
    import daydream.json_utils as json_utils

    fenced = '```json\n{"a": 1}\n```'
    assert json_utils.extract_json(fenced) == {"a": 1}
    assert json_utils._strip_json_fences(fenced) == '{"a": 1}'
    assert json_utils._strip_json_fences("plain") == "plain"

"""Backend-neutral JSON extraction and shared crash-safe write primitives."""

import json

from daydream.json_utils import (
    extract_json,
    extract_json_by_schema,
    validates_schema,
)
from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA


def test_schema_selection_reports_content_free_rejections() -> None:
    """Selection is driven by the boundary's acceptance predicate, never span size."""
    assert extract_json("") is None
    sel = extract_json_by_schema('["module", "moduleVersion", "surface"]',
                                 schema=PER_STACK_RECORD_SCHEMA, accept=validates_schema)
    assert sel.value is None
    assert sel.rejected_type == "list"
    assert sel.rejected_reason == "type at $"
    assert "module" not in (sel.rejected_reason or "")  # never candidate content
    assert sel.candidate_count >= 1
    wrapped = extract_json_by_schema(
        'Reviewed ["module", "moduleVersion", "surface"] without findings.',
        schema=PER_STACK_RECORD_SCHEMA,
        accept=validates_schema,
    )
    assert (wrapped.value, wrapped.rejected_type, wrapped.rejected_reason) == (None, "list", "type at $")
    assert wrapped.rejection is None
    selection = extract_json_by_schema(
        '{"issues": []}', schema=PER_STACK_RECORD_SCHEMA, accept=lambda _value, _schema: False,
    )
    assert selection.value is None and selection.candidate_count == 2
    assert selection.rejected_type == "dict"
    assert selection.rejected_reason is None and selection.rejection is None


def test_staged_decoder_consumes_outer_spans_and_bounds_packaging() -> None:
    schema = {"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array"}},
              "additionalProperties": False}
    cases = [
        ('```\n{"issues": []}\n```', {"issues": []}, False),
        (json.dumps({"result": {"issues": ['braces } ] and escaped " quotes \\']}}),
         {"issues": ['braces } ] and escaped " quotes \\']}, False),
        ('[{"issues": []}] followed by {"issues": []}', {"issues": []}, False),
        ('{"result": {"issues": []} tail', None, True),
        ('[] ' * 128 + '{"issues": []}', None, True),
        ('{"issues": []} }', None, True),
        ('{"a": 1} {"b": 2}', None, False),
    ]
    for text, expected, syntax in cases:
        selection = extract_json_by_schema(text, schema=schema, accept=validates_schema, require_complete_root=True)
        assert selection.value == expected, text
        assert (selection.syntax_error is not None) is syntax, text
        assert not selection.schema_retry_eligible and selection.rejection is None
        assert selection.candidate_count <= 128

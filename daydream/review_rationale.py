"""Bounded persisted reviewer assertions, distinct from host coverage evidence."""
from __future__ import annotations

from typing import Any

from daydream.json_utils import validates_schema
from daydream.output_schema import strict_object
from daydream.redaction import redact_value

_TEXT = {'type': 'string', 'maxLength': 512}
_FILES = {'type': 'array', 'maxItems': 32, 'items': _TEXT}
NOTE_ORIGIN_SCHEMA = strict_object({
    'scope_id': _TEXT, 'stage': _TEXT,
    'sequence': {'type': 'integer', 'minimum': 1}, 'judgment_origin': {'const': 'reviewer'},
    'analyzed_revision': strict_object({name: _TEXT
        for name in ('head_sha', 'merge_base_sha', 'diff_key', 'pr_base_sha')}),
})
NOTE_ORIGIN_SCHEMA['properties']['analyzed_revision']['required'].remove('pr_base_sha')
SOURCE_RATIONALE_SCHEMA = strict_object({
    'admitted_stages': {'type': 'integer', 'minimum': 0},
    'target_decisions': strict_object({name: {'type': 'integer', 'minimum': 0}
                                      for name in ('required', 'reviewed', 'not_reviewed')}),
    'notes': {'type': 'array', 'maxItems': 8, 'items': strict_object({
        'text': _TEXT, 'files': _FILES, 'supporting_files': _FILES,
        'origin': {'anyOf': [NOTE_ORIGIN_SCHEMA, {'type': 'null'}]},
    })},
    'notes_omitted': {'type': 'integer', 'minimum': 0},
})
_SCOPE_RATIONALE_SCHEMA = strict_object({
    'scope_id': {'type': 'string', 'minLength': 1, 'maxLength': 512},
    'originating_run_id': {'type': 'string', 'minLength': 1, 'maxLength': 512},
    'analyzed_revision': NOTE_ORIGIN_SCHEMA['properties']['analyzed_revision'],
    'review_context': SOURCE_RATIONALE_SCHEMA,
})


def valid_review_context(value: Any) -> bool:
    """Validate optional persisted assertions without treating them as semantic proof."""
    if not validates_schema(value, SOURCE_RATIONALE_SCHEMA) or redact_value(value) != value:
        return False
    decisions = value['target_decisions']
    return bool(decisions['reviewed'] + decisions['not_reviewed'] <= decisions['required'])


def valid_scope_review_context(
    value: Any, *, scope_id: str, analyzed_revision: dict[str, Any], originating_run_id: str,
) -> bool:
    """Admit an optional host sidecar only for the exact current source records."""
    return bool(validates_schema(value, _SCOPE_RATIONALE_SCHEMA)
            and value['scope_id'] == scope_id and value['analyzed_revision'] == analyzed_revision
            and value['originating_run_id'] == originating_run_id
            and valid_review_context(value['review_context']))

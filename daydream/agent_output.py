"""Host-owned selection and validation of final agent output."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from daydream.json_utils import (
    SchemaAwareSelection,
    SchemaRejection,
    extract_json,
    extract_json_by_schema,
    schema_rejection,
    validates_schema,
)


class StructuredOutputFailure(str):
    """Text fallback carrying the host's rejected structured-output witness.

    ``detail`` is an optional, already content-free diagnostic fragment: the
    selected candidate's Python type name and its first schema error as
    ``"<validator> at <json_path>"``. It never carries candidate content, so it
    stays safe through the redaction/bounding path that surfaces it.
    """

    reason: str
    detail: str | None
    rejection: SchemaRejection | None
    schema_retry_eligible: bool
    syntax_error: dict[str, int] | None

    def __new__(cls, text: str, reason: str, detail: str | None = None, *,
        rejection: SchemaRejection | None = None, schema_retry_eligible: bool = False,
        syntax_error: dict[str, int] | None = None) -> "StructuredOutputFailure":
        value = super().__new__(cls, text)
        value.reason = reason
        value.detail = detail
        value.rejection = rejection
        value.schema_retry_eligible = schema_retry_eligible
        value.syntax_error = syntax_error
        return value


def _select_by_schema(text: str, schema: dict[str, Any], *, require_full_schema: bool) -> SchemaAwareSelection:
    """Select one candidate with the run's own gate, strictly preferring full validity.

    ``_salvageable`` is deliberately loose — any object carrying the schema's
    required keys qualifies — so under a "last admitted candidate wins" rule it
    would let a later incidental object (e.g. an evidence digest that merely
    lists issues) displace the real answer. Scanning with the strict gate first
    keeps a fully valid candidate authoritative; the salvage scan only runs when
    nothing validates, so this never widens what is accepted, only reorders it.
    """
    strict = extract_json_by_schema(text, schema=schema, accept=validates_schema)
    if require_full_schema or strict.value is not None:
        return strict
    return extract_json_by_schema(text, schema=schema, accept=_salvageable)


def _salvageable(value: Any, schema: dict[str, Any]) -> bool:
    """Accept full schema validity or a shape downstream consumers can salvage.

    Objects must contain required keys, with lists in required array slots;
    nested records are validated downstream by callers using this capability.
    """
    if validates_schema(value, schema):
        return True
    if not isinstance(value, dict):
        return False
    required = schema.get("required")
    if not isinstance(required, list):
        return False
    properties = schema.get("properties", {})
    for key in required:
        if key not in value:
            return False
        prop = properties.get(key)
        if isinstance(prop, dict) and prop.get("type") == "array":
            if not isinstance(value[key], list):
                return False
    return True


def resolve_output(
    structured_result: Any, raw: str, *, native_output: bool, text_overflow: bool,
    output_schema: dict[str, Any] | None, validate_structured_output: bool, require_full_schema: bool,
    staged: bool, schema_rejection_guard: Callable[[Any], bool] | None,
) -> Any:
    """Resolve native authority or final text using the caller's existing validation policy."""
    def _usable(value: Any) -> bool:
        """Accept explicit validation opt-out or a downstream-salvageable value."""
        return not validate_structured_output or (
            output_schema is not None and (
                validates_schema(value, output_schema) if require_full_schema or native_output
                else _salvageable(value, output_schema)
            )
        )

    if output_schema is not None and structured_result is not None and _usable(structured_result):
        return structured_result
    selection: SchemaAwareSelection | None = None
    if output_schema is not None:
        # Fallback: robust extraction (prose-wrapped JSON, markdown fences) when
        # structured output failed. Selection is schema-driven rather than
        # size-driven — the last candidate this same gate admits wins — so
        # incidental prose JSON ahead of a real answer (e.g. a trailing
        # `{"issues": []}` after a bracket list) resolves to the answer, never to
        # the largest span. The selected value must still pass the same
        # salvage-tolerant gate as the success path (see _salvageable) — and, under
        # the last-admitted-wins rule, a fully schema-valid candidate outranks any
        # merely salvageable one (see _select_by_schema).
        #
        # The selector discriminates only when that gate discriminates. A caller
        # that opts out via validate_structured_output=False has a vacuous gate,
        # where "last admitted candidate" degenerates to the innermost span and
        # would hand back a nested record instead of the intended object; those
        # callers keep largest-span extraction. Everything else narrows to the
        # last candidate its own gate admits, which never widens what is
        # accepted.
        # Staged native output is authoritative; rejection forbids salvaging text fragments from the same invocation.
        if not native_output and raw.strip() and not text_overflow and not (
            staged and structured_result is not None):
            selected: Any = None
            if validate_structured_output:
                selection = (extract_json_by_schema(raw, schema=output_schema, accept=validates_schema,
                                                    require_complete_root=True, rejection_guard=schema_rejection_guard)
                             if staged else
                             _select_by_schema(raw, output_schema, require_full_schema=require_full_schema))
                selected = selection.value
            else:
                selected = extract_json(raw)
            if selected is not None and _usable(selected):
                return selected
    if output_schema is not None and (require_full_schema or native_output):
        reason = ("malformed_output" if structured_result is not None or raw.strip() or text_overflow
                  else "missing_output")
        # Content-free rejection trace: the selected candidate's Python type name,
        # its first schema error as "<validator> at <json_path>" (never the
        # jsonschema message, which embeds candidate content), and how many spans
        # the scan enumerated. Nothing here can inject model-authored text.
        reject_detail: str | None = None
        if reason == "malformed_output" and selection is not None and selection.rejected_type is not None:
            reject_detail = (f"candidate type {selection.rejected_type} failed {selection.rejected_reason}"
                             f"; {selection.candidate_count} candidate(s)")
        rejection = (schema_rejection(structured_result, output_schema) if structured_result is not None else
                     selection.rejection if selection is not None else None)
        if rejection is not None:
            reject_detail = (f"schema {rejection.category} at {rejection.schema_path}; "
                             f"{rejection.error_count} error(s); {rejection.candidate_count} candidate(s)")
        eligible = not native_output and (
            rejection is not None and (schema_rejection_guard is None or schema_rejection_guard(structured_result))
            if structured_result is not None else selection.schema_retry_eligible if selection is not None else False
        )
        # Preserve the domain guard's rejection diagnostics even when a selected native payload cannot retry.
        if staged and structured_result is not None:
            eligible = False
        failure = StructuredOutputFailure(raw, reason, reject_detail, rejection=rejection,
                                          schema_retry_eligible=eligible,
                                          syntax_error=selection.syntax_error if selection is not None else None)
        return failure
    return raw

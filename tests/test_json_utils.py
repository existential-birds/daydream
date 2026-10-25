"""Backend-neutral JSON extraction and shared crash-safe write primitives."""

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from daydream.json_utils import (
    atomic_write_bytes,
    atomic_write_json,
    extract_json,
    extract_json_by_schema,
    validates_schema,
)
from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA


class TestExtractJson:
    @pytest.mark.parametrize(("text", "expected"),
        [
        pytest.param("", None, id="empty-string"),
        pytest.param('prefix [1] then { }', {}, id="equal-size-object-before-array"),
        pytest.param('prefix [1] then [2]', [1], id="equal-size-first-array"),
        pytest.param("This is just prose with no JSON whatsoever.", None, id="no-json-at-all",),
        pytest.param('note {"k": 1} then {"findings": [{"arb_id": 1, "keep": false, "x": "y"}]}',
                {"findings": [{"arb_id": 1, "keep": False, "x": "y"}]},
                id="largest-object-wins-over-smaller-earlier-object",
            ),
    ],
    )
    def test_extract_json(self, text: str, expected: Any) -> None:
        assert extract_json(text) == expected

    def test_stray_prose_bracket_does_not_beat_the_real_object(self) -> None:
        # Regression for the sentry-67876 arbiter crash. The model's prose
        # referenced a code snippet `metadata["sender"]["login"]` BEFORE its real fenced
        # answer. The earliest-bracket rule parsed `["sender"]` (a valid 1-element
        # list) and returned that bare list, which crashed the arbiter with
        # "Arbiter returned no findings list (got list)". The largest-span rule
        # must instead return the substantial `{"findings": [...]}` object.
        text = ("**Finding 2 (arb_id=2):** Confirmed. Line 503 does "
            '`integration.metadata["sender"]["login"]` with direct subscripting.\n\n'
            "```json\n"
            '{"findings": [{"arb_id": 1, "keep": true}, {"arb_id": 2, "keep": true}]}\n'
            "```"
        )
        result = extract_json(text)
        assert isinstance(result, dict)
        assert [f["arb_id"] for f in result["findings"]] == [1, 2]


class TestExtractJsonBySchema:
    """Selection is driven by the boundary's acceptance predicate, never span size."""

    def test_larger_incidental_object_does_not_win(self) -> None:
        text = 'prefix {"a": [1,2,3,4,5,6,7,8]} trailing {"issues": []}'
        sel = extract_json_by_schema(text, schema=PER_STACK_RECORD_SCHEMA, accept=validates_schema)
        assert sel.value == {"issues": []}

    def test_whole_text_that_parses_but_fails_schema_does_not_win(self) -> None:
        sel = extract_json_by_schema('{"issues": [], "extra": 1}',
                                     schema=PER_STACK_RECORD_SCHEMA, accept=validates_schema)
        assert sel.value is None

    def test_nested_valid_fragment_inside_invalid_root_still_competes(self) -> None:
        sel = extract_json_by_schema('prefix {bad {"issues": []}} suffix',
                                     schema=PER_STACK_RECORD_SCHEMA, accept=validates_schema)
        assert sel.value == {"issues": []}

    def test_rejection_reports_content_free_type_and_reason(self) -> None:
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

    def test_no_candidates_returns_none(self) -> None:
        sel = extract_json_by_schema("This is just prose with no JSON whatsoever.",
                                     schema=PER_STACK_RECORD_SCHEMA, accept=validates_schema)
        assert sel.value is None
        assert sel.candidate_count == 0

    def test_custom_acceptance_can_reject_valid_json_without_a_schema_error(self) -> None:
        selection = extract_json_by_schema(
            '{"issues": []}', schema=PER_STACK_RECORD_SCHEMA, accept=lambda _value, _schema: False,
        )
        assert selection.value is None and selection.candidate_count == 2
        assert selection.rejected_type == "dict"
        assert selection.rejected_reason is None and selection.rejection is None


@pytest.mark.parametrize(("text", "expected", "syntax"), [
    ('```\n{"issues": []}\n```', {"issues": []}, False),
    (json.dumps({"result": {"issues": ['braces } ] and escaped " quotes \\']}}),
     {"issues": ['braces } ] and escaped " quotes \\']}, False),
    ('[{"issues": []}] followed by {"issues": []}', {"issues": []}, False),
    ('{"result": {"issues": []} tail', None, True),
    ('{"issues": []} "literal [ unfinished"', None, True),
    ('[] ' * 128 + '{"issues": []}', None, True),
])
def test_staged_decoder_consumes_outer_spans_and_bounds_packaging(
    text: str, expected: Any, syntax: bool,
) -> None:
    schema = {"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array"}},
              "additionalProperties": False}
    selection = extract_json_by_schema(text, schema=schema, accept=validates_schema, require_complete_root=True)
    assert selection.value == expected
    assert (selection.syntax_error is not None) is syntax
    assert not selection.schema_retry_eligible and selection.rejection is None
    assert selection.candidate_count <= 128


class TestAtomicWritePrimitives:
    def test_bytes_roundtrip_and_replaces_prior_content(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "data.bin"
        atomic_write_bytes(target, b"first")
        assert target.read_bytes() == b"first"
        atomic_write_bytes(target, b"second")
        assert target.read_bytes() == b"second"

    def test_mode_knob_is_umask_immune_and_survives_replacement(self, tmp_path: Path) -> None:
        target = tmp_path / "private.json"
        umask = os.umask(0o077)
        try:
            atomic_write_bytes(target, b"{}", mode=0o600)
        finally:
            os.umask(umask)
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        atomic_write_bytes(target, b"[1]", mode=0o640)
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    def test_no_mode_keeps_default_permissions(self, tmp_path: Path) -> None:
        target = tmp_path / "plain.txt"
        # Permissive umask: a umask-derived open() writer would yield 0o644
        # here, so asserting 0o600 actually discriminates the mkstemp-based
        # default (0o600) from umask-derived permissions.
        umask = os.umask(0o022)
        try:
            atomic_write_bytes(target, b"x")
        finally:
            os.umask(umask)
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_json_knob_byte_format_is_stable(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.json"
        atomic_write_json(target, {"b": 1, "a": [1, 2]}, sort_keys=True)
        assert json.loads(target.read_text(encoding="utf-8")) == {"b": 1, "a": [1, 2]}
        # The default remains pretty-printed, two-space indent, no trailing newline.
        assert target.read_text(encoding="utf-8") == '{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}'

    def test_failure_cleans_temp_and_propagates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "out" / "boom.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"prior")
        real_replace = os.replace
        def failing_replace(src: object, dst: object, **kwargs: object) -> None:
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError, match="No space left"):
            atomic_write_bytes(target, b"new")
        monkeypatch.setattr(os, "replace", real_replace)
        assert target.read_bytes() == b"prior"
        assert list(target.parent.iterdir()) == [target], "temp file must be cleaned up"

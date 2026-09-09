"""Tests for the shared :mod:`daydream.json_utils` helpers.

``extract_json`` is shared by the backends (structured-output extraction) and
``run_agent`` (raw-text fallback) — it is not a Pi-specific concern. The tests
target the canonical functions directly so they remain valid regardless of any
backend-private wrapper aliases. ``atomic_write_bytes``/``atomic_write_json``
are the shared crash-safe write primitives that the #1162 consolidation
routes the previously duplicated copies through.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from daydream.json_utils import atomic_write_bytes, atomic_write_json, extract_json


class TestExtractJson:
    """Verify extract_json handles clean JSON, fenced JSON, and prose-wrapped JSON."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                '{"findings": [], "ok": true}',
                {"findings": [], "ok": True},
                id="clean-json-object",
            ),
            pytest.param("[1, 2, 3]", [1, 2, 3], id="clean-json-array"),
            pytest.param(
                '```json\n{"findings": [{"arb_id": 1, "keep": true}]}\n```',
                {"findings": [{"arb_id": 1, "keep": True}]},
                id="markdown-fenced-json",
            ),
            pytest.param('```\n{"x": 1}\n```', {"x": 1}, id="markdown-fenced-bare"),
            pytest.param(
                "Based on my analysis of all findings, here are my verdicts:\n"
                '{"findings": [{"arb_id": 1, "keep": false}]}',
                {"findings": [{"arb_id": 1, "keep": False}]},
                id="prose-wrapped-json",
            ),
            pytest.param(
                'Here are the issues:\n[{"id": 1, "severity": "high"}]\nThat concludes the review.',
                [{"id": 1, "severity": "high"}],
                id="prose-wrapped-array",
            ),
            pytest.param("", None, id="empty-string"),
            pytest.param("   \n  ", None, id="whitespace-only"),
            pytest.param(
                "This is just prose with no JSON whatsoever.",
                None,
                id="no-json-at-all",
            ),
            pytest.param(
                '{"msg": "contains a } brace", "ok": true}',
                {"msg": "contains a } brace", "ok": True},
                id="json-with-nested-braces-in-strings",
            ),
            pytest.param(
                "[1, bad] then [3,4]",
                [3, 4],
                id="unparseable-array-then-valid-array",
            ),
            pytest.param(
                '{bad} then {"issues":[1,2]}',
                {"issues": [1, 2]},
                id="unparseable-object-then-valid-object",
            ),
            pytest.param(
                'prefix {bad {"findings": []}} suffix',
                {"findings": []},
                id="nested-valid-json-inside-balanced-invalid-span",
            ),
            pytest.param(
                'note {"k": 1} then {"findings": [{"arb_id": 1, "keep": false, "x": "y"}]}',
                {"findings": [{"arb_id": 1, "keep": False, "x": "y"}]},
                id="largest-object-wins-over-smaller-earlier-object",
            ),
        ],
    )
    def test_extract_json(self, text: str, expected: Any) -> None:
        """Extract supported JSON wrappers while preserving expected Python values."""
        assert extract_json(text) == expected

    def test_stray_prose_bracket_does_not_beat_the_real_object(self) -> None:
        # Regression for the sentry-67876 arbiter crash. The model's prose
        # referenced a code snippet `metadata["sender"]["login"]` BEFORE its real fenced
        # answer. The earliest-bracket rule parsed `["sender"]` (a valid 1-element
        # list) and returned that bare list, which crashed the arbiter with
        # "Arbiter returned no findings list (got list)". The largest-span rule
        # must instead return the substantial `{"findings": [...]}` object.
        text = (
            "**Finding 2 (arb_id=2):** Confirmed. Line 503 does "
            '`integration.metadata["sender"]["login"]` with direct subscripting.\n\n'
            "```json\n"
            '{"findings": [{"arb_id": 1, "keep": true}, {"arb_id": 2, "keep": true}]}\n'
            "```"
        )
        result = extract_json(text)
        assert isinstance(result, dict)
        assert [f["arb_id"] for f in result["findings"]] == [1, 2]


class TestAtomicWritePrimitives:
    """The shared crash-safe write primitive and its knobs (#1162 Item A)."""

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

    def test_fsync_false_skips_sync_but_still_atomic(self, tmp_path: Path) -> None:
        target = tmp_path / "fast.json"
        atomic_write_bytes(target, b"ok", fsync=False, dir_fsync=False)
        assert target.read_bytes() == b"ok"

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

    def test_dir_fsync_publishes_rename_durably(self, tmp_path: Path) -> None:
        target = tmp_path / "durable.json"
        atomic_write_bytes(target, b"durable", dir_fsync=True)
        assert target.read_bytes() == b"durable"

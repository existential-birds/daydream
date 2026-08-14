"""Acceptance tests for scripts/redrive_post.py (canonical merged-items posting).

Drives the real CLI entrypoint (scripts.redrive_post.main) through real argument
parsing, with the fake gh harness only at the external subprocess boundary. The
delegation functions (post_review_to_pr_from_report / parsed_issues_from_items /
classify / build_payload / _post) are NOT monkeypatched — the real shared
canonical posting path must run end-to-end (spec M3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.redrive_post as redrive
from tests.test_extension_seam_integration import _serve_pr_view

CANONICAL_KEEP_INLINE = "CANONICAL_KEEP_INLINE_SENTINEL"
CANONICAL_KEEP_STRUCTURAL = "CANONICAL_KEEP_STRUCTURAL_SENTINEL"
LEGACY_DROPPED = "LEGACY_DROPPED_SENTINEL"


def _run_cli(*argv: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["redrive_post", *argv])
    redrive.main()


def _write_canonical(target: Path) -> None:
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "merged-items.json").write_text(
        json.dumps(
            {
                "items": [
                    {"id": 1, "lens": "generic", "file": "api.py", "line": 1,
                     "description": CANONICAL_KEEP_INLINE, "severity": "high",
                     "confidence": "HIGH", "rationale": "r"},
                    {"id": 2, "lens": "structural", "file": "README.md", "line": 1,
                     "description": CANONICAL_KEEP_STRUCTURAL, "severity": "high",
                     "confidence": "HIGH", "rationale": "r"},
                ]
            }
        )
    )
    # Legacy pre-merge artifacts the OLD redrive would have read. Redrive must
    # never source findings from these (spec M2).
    (deep / "alternatives.json").write_text(
        json.dumps([{"id": 99, "files": ["api.py"], "description": LEGACY_DROPPED}])
    )
    (deep / "stack-a-records.json").write_text(
        json.dumps([{"file": "api.py", "id": 1, "line": 1, "description": LEGACY_DROPPED}])
    )
    (deep / "dedup-candidates.json").write_text("[]")


def test_redrive_posts_only_canonical_merged_items(
    multi_stack_target: Path, fake_gh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_pr_view(fake_gh, multi_stack_target)
    _write_canonical(multi_stack_target)

    _run_cli(str(multi_stack_target), "--pr", "7", "--yes", monkeypatch=monkeypatch)

    posted = json.dumps(
        [
            call.payload
            for call in (
                *fake_gh.calls("POST", "repos/acme/widgets/pulls/7/reviews"),
                *fake_gh.calls("POST", "repos/acme/widgets/pulls/7/comments"),
            )
        ]
    )
    assert CANONICAL_KEEP_INLINE in posted
    assert CANONICAL_KEEP_STRUCTURAL in posted
    assert LEGACY_DROPPED not in posted
    view_calls = fake_gh.pr_view_calls()
    assert view_calls and "7" in view_calls[0].argv          # explicit `gh pr view 7`
    assert fake_gh.command_calls("pr list") == []            # no current-branch discovery
    assert not (multi_stack_target / ".review-output.md").exists()


def test_redrive_requires_canonical_merged_items(
    multi_stack_target: Path,
    fake_gh: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)  # exists, but no merged-items.json

    with pytest.raises(SystemExit) as exc:
        _run_cli(str(multi_stack_target), "--pr", "7", "--yes", monkeypatch=monkeypatch)

    assert exc.value.code == 1
    missing = deep / "merged-items.json"
    assert f"No canonical merged-items.json found at {missing}" in capsys.readouterr().err
    assert fake_gh.calls("POST") == []       # zero GitHub requests
    assert not (multi_stack_target / ".review-output.md").exists()


def test_redrive_corrupt_merged_items_skips_gracefully(
    multi_stack_target: Path,
    fake_gh: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _serve_pr_view(fake_gh, multi_stack_target)
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "merged-items.json").write_text("{not valid json}")

    _run_cli(str(multi_stack_target), "--pr", "7", "--yes", monkeypatch=monkeypatch)

    out = capsys.readouterr().out
    assert "Could not read merged-items.json" in out
    assert fake_gh.calls("POST") == []
    assert not (multi_stack_target / ".review-output.md").exists()

"""Severity vocabulary reaches the two schema owners and actual publication consumers."""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream import pr_comment_renderer, severity
from daydream.pr_review import (
    ClassifiedIssues,
    ParsedIssue,
    ReviewRenderers,
    default_render_finding,
    default_render_summary,
)
from tests.harness.review_payload import payload_for
from tests.harness.review_profile import sample_pr

REPO = Path(__file__).resolve().parents[1]


def test_pr_review_severity_breakdown_follows_the_declaration(monkeypatch: pytest.MonkeyPatch,) -> None:
    # Declaration reversed: the model-facing order must follow it, so a hand-written
    # tuple at the call site renders "1 high, 1 low" and fails here.
    monkeypatch.setattr(severity, "CANONICAL_LEVELS", ("high", "medium", "low"))
    classified = ClassifiedIssues(body_only=[
            ParsedIssue(path="a.py", line=10, title="t1", body="b", confidence="HIGH", severity="high"),
            ParsedIssue(path="a.py", line=12, title="t2", body="b", confidence="LOW", severity="low"),
        ]
    )
    body = payload_for(
        sample_pr(), classified, renderers=ReviewRenderers(default_render_finding, default_render_summary),
        run_info=pr_comment_renderer._render_fallback(),
    )["body"]
    assert "- **Severity:** 1 low, 1 high" in body

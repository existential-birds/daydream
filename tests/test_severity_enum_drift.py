"""Severity vocabulary reaches the two schema owners and actual publication consumers."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream import pr_comment_renderer, severity
from daydream.benchmark.harbor import verifier_core
from daydream.improve.prompts import VET_SCHEMA
from daydream.phases import SUPERVISE_SCHEMA
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
FROZEN_MODEL_FACING = ["high", "medium", "low"]
_SCHEMAS = [
    ("supervise", SUPERVISE_SCHEMA["properties"]["verdicts"]["items"]["properties"]["severity"]),
    ("vet", VET_SCHEMA["properties"]["verdicts"]["items"]["properties"]["severity"]),
]


@pytest.mark.parametrize("site,fragment", _SCHEMAS)
def test_site_emits_the_frozen_model_facing_order(site: str, fragment: dict[str, Any]) -> None:
    assert fragment == {"anyOf": [{"type": "string", "enum": FROZEN_MODEL_FACING}, {"type": "null"}]}, site


@pytest.fixture(scope="module")
def rebuilt_under_patched_declaration() -> dict[str, dict[str, Any]]:
    script = """
import json
import daydream.severity as severity
severity.CANONICAL_LEVELS = ("high", "medium", "low", "critical")
from daydream.phases import SUPERVISE_SCHEMA
from daydream.improve.prompts import VET_SCHEMA
print(json.dumps({
    "supervise": SUPERVISE_SCHEMA["properties"]["verdicts"]["items"]["properties"]["severity"],
    "vet": VET_SCHEMA["properties"]["verdicts"]["items"]["properties"]["severity"],
}))
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    rebuilt: dict[str, dict[str, Any]] = json.loads(proc.stdout)
    return rebuilt


@pytest.mark.parametrize("site,fragment", _SCHEMAS)
def test_site_follows_the_declaration_when_the_declaration_moves(
    site: str, fragment: dict[str, Any], rebuilt_under_patched_declaration: dict[str, dict[str, Any]],
) -> None:
    assert fragment["anyOf"][0]["enum"] == FROZEN_MODEL_FACING
    assert rebuilt_under_patched_declaration[site] == {
        "anyOf": [{"type": "string", "enum": ["critical", "low", "medium", "high"]}, {"type": "null"}]}


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

def test_fenced_verifier_accepts_exactly_the_canonical_vocabulary() -> None:
    """verifier_core deploys byte-for-byte into a daydream-free image, so it keeps its
    literal; this test is the drift protection instead of an import (spec requirement 11)."""
    base = {"candidate_id": "a" * 64, "title": "t", "body": "b", "path": "src/a.py", "start_line": 1, "end_line": 1}
    for level in severity.CANONICAL_LEVELS:
        assert verifier_core.parse_candidate_finding({**base, "severity": level}).severity == level
    with pytest.raises(verifier_core.VerifierError):
        verifier_core.parse_candidate_finding({**base, "severity": "critical"})

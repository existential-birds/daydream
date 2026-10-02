"""Incomplete reviewer evidence remains visible in partial-result warnings."""

import json
from pathlib import Path

import pytest

from daydream.deep.artifacts import DeepArtifact
from daydream.review_budget import review_warnings
from tests.harness.review_result import review_coverage


@pytest.mark.parametrize("reason", [
    "evidence incomplete: client.py unavailable",
    "budget exhausted: review turn budget exhausted",
    "RuntimeError: review provider unavailable",
    "PermissionError: reviewer input inaccessible",
])
def test_failed_reviewer_remains_a_reported_warning(tmp_path: Path, reason: str) -> None:
    coverage = review_coverage(scope_ids=("python",), phases=("arbiter",))
    coverage.record_scope("python", "failed", reasons=("backend_failure",), diagnostic=reason)
    coverage.record_phase("arbiter", "incomplete", reasons=("host_wall_budget_exhaustion",),
                          diagnostic="wall budget exhausted")
    path = DeepArtifact.REVIEW_COVERAGE.at(tmp_path)
    path.write_text(json.dumps(coverage.to_dict()))

    assert review_warnings(tmp_path) == ("arbiter: wall budget exhausted", f"python: {reason}")

    coverage.record_scope("python", "complete")
    path.write_text(json.dumps(coverage.to_dict()))
    assert review_warnings(tmp_path) == ("arbiter: wall budget exhausted",)


def test_synthesis_failure_is_a_distinct_phase_warning(tmp_path: Path) -> None:
    coverage = review_coverage(scope_ids=('python',))
    coverage.record_scope('python', 'failed', reasons=('backend_failure',), diagnostic='provider unavailable')
    coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',), diagnostic='prior synthesis failed')
    DeepArtifact.REVIEW_COVERAGE.at(tmp_path).write_text(json.dumps(coverage.to_dict()))
    assert review_warnings(tmp_path) == ('merge: prior synthesis failed', 'python: provider unavailable')


def test_failed_reviewer_warning_redacts_provider_credentials(tmp_path: Path) -> None:
    reason = "RuntimeError: OPENROUTER_API_KEY=not-a-real-credential"
    coverage = review_coverage(scope_ids=("python",), phases=())
    coverage.record_scope("python", "failed", reasons=("backend_failure",), diagnostic=reason)
    path = DeepArtifact.REVIEW_COVERAGE.at(tmp_path)
    path.write_text(json.dumps(coverage.to_dict()))

    warnings = review_warnings(tmp_path)

    assert len(warnings) == 1
    assert "not-a-real-credential" not in warnings[0]
    assert "[REDACTED_ENV_VAR]" in warnings[0]
    assert "not-a-real-credential" not in path.read_text()

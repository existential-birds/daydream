"""Incomplete reviewer evidence remains visible in partial-result warnings."""

import json
from pathlib import Path

from daydream.deep.artifacts import DeepArtifact
from daydream.review_budget import review_warnings
from tests.harness.review_result import review_coverage


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

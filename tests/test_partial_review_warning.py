"""Incomplete reviewer evidence remains visible in partial-result warnings."""

import json
from pathlib import Path

import pytest

from daydream.deep.artifacts import DeepArtifact
from daydream.review_budget import record_review_budget_stop, review_warnings


@pytest.mark.parametrize("reason", [
    "evidence incomplete: client.py unavailable",
    "budget exhausted: review turn budget exhausted",
    "RuntimeError: review provider unavailable",
    "PermissionError: reviewer input inaccessible",
])
def test_failed_reviewer_remains_a_reported_warning(tmp_path: Path, reason: str) -> None:
    DeepArtifact.PER_STACK_FAILURES.at(tmp_path).write_text(json.dumps({
        "python": reason, "__merge__": {"message": "old merge failed", "result_type": "str"},
    }))
    record_review_budget_stop(tmp_path, "Arbiter", "wall budget exhausted")

    assert review_warnings(tmp_path) == ("Arbiter: wall budget exhausted", f"python: {reason}")

    DeepArtifact.PER_STACK_FAILURES.at(tmp_path).write_text("{}")
    assert review_warnings(tmp_path) == ("Arbiter: wall budget exhausted",)


def test_reserved_merge_failure_is_not_a_reviewer_warning(tmp_path: Path) -> None:
    DeepArtifact.PER_STACK_FAILURES.at(tmp_path).write_text(json.dumps({
        "__merge__": "corrupt", "python": "provider unavailable",
    }))
    assert review_warnings(tmp_path) == ("python: provider unavailable",)


def test_failed_reviewer_warning_redacts_provider_credentials(tmp_path: Path) -> None:
    reason = "RuntimeError: OPENROUTER_API_KEY=not-a-real-credential"
    DeepArtifact.PER_STACK_FAILURES.at(tmp_path).write_text(json.dumps({"python": reason}))

    warnings = review_warnings(tmp_path)

    assert len(warnings) == 1
    assert "not-a-real-credential" not in warnings[0]
    assert "[REDACTED_ENV_VAR]" in warnings[0]
    assert json.loads(DeepArtifact.PER_STACK_FAILURES.at(tmp_path).read_text()) == {"python": reason}

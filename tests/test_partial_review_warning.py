"""Incomplete reviewer evidence remains visible in partial-result warnings."""

import json
from pathlib import Path

import pytest

from daydream.deep.artifacts import per_stack_failures_path
from daydream.review_budget import record_review_budget_stop, review_warnings


@pytest.mark.parametrize("reason", [
    "evidence incomplete: client.py unavailable",
    "budget exhausted: review turn budget exhausted",
    "RuntimeError: review provider unavailable",
    "PermissionError: reviewer input inaccessible",
])
def test_failed_reviewer_remains_a_reported_warning(tmp_path: Path, reason: str) -> None:
    per_stack_failures_path(tmp_path).write_text(json.dumps({
        "python": reason, "__merge__": {"message": "old merge failed", "result_type": "str"},
    }))
    record_review_budget_stop(tmp_path, "Arbiter", "wall budget exhausted")

    assert review_warnings(tmp_path) == ("Arbiter: wall budget exhausted", f"python: {reason}")

    per_stack_failures_path(tmp_path).write_text("{}")
    assert review_warnings(tmp_path) == ("Arbiter: wall budget exhausted",)

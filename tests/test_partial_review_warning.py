"""Incomplete reviewer evidence remains visible in partial-result warnings."""

import json
from pathlib import Path

from daydream.deep.artifacts import per_stack_failures_path
from daydream.review_budget import review_warnings


def test_unavailable_evidence_remains_a_reported_warning(tmp_path: Path) -> None:
    per_stack_failures_path(tmp_path).write_text(json.dumps({"python": "evidence incomplete: client.py unavailable"}))
    assert review_warnings(tmp_path) == ("python: evidence incomplete: client.py unavailable",)

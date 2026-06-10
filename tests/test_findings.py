"""Tests for the findings artifact (build/write/load) in `daydream/findings.py`."""

import json

from daydream import pr_review
from daydream.findings import (
    FINDINGS_SCHEMA_VERSION,
    build_findings_artifact,
    write_findings_artifact,
)
from daydream.pr_review import ParsedIssue, PRInfo


def test_build_artifact_declares_target_and_placed_findings(tmp_path, monkeypatch) -> None:
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main",
                owner="o", repo="r", url="u")
    issues = [ParsedIssue(path="a.py", line=None, title="T", body="B", severity="high",
                          confidence="HIGH", fingerprint="f" * 64)]
    monkeypatch.setattr(pr_review, "resolve_line", lambda *_a: 12)
    monkeypatch.setattr(pr_review, "file_hunks", lambda *_a, **_k: [(10, 14)])
    artifact = build_findings_artifact(tmp_path, pr, issues, run_info=None)
    assert (artifact["repo"], artifact["pr_number"], artifact["head_sha"]) == ("o/r", 7, "h" * 40)
    f = artifact["findings"][0]
    assert (f["fingerprint"], f["placement"], f["line"]) == ("f" * 64, "inline", 12)


def test_write_artifact_round_trips(tmp_path) -> None:
    path = tmp_path / "findings.json"
    write_findings_artifact(path, {"schema_version": FINDINGS_SCHEMA_VERSION, "repo": "o/r",
                                   "pr_number": 7, "head_sha": "h" * 40, "findings": []})
    assert json.loads(path.read_text())["schema_version"] == FINDINGS_SCHEMA_VERSION

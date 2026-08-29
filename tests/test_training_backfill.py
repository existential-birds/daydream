"""Tests for the backfill pass — re-annotate indexed runs through the harvest
annotation path and append a fresh ``label_observations`` generation, with a
machine-readable migration report.

Covers:
* Append-only backfill: new generations carry the current policy version (M17).
* Idempotent re-run: unchanged evidence + unchanged versions dedup to a no-op (M18).
* Fail-closed on deleted/inaccessible PR comments (M19/M22).
* Machine-readable report keys (M20).
* Legacy ``label_observations`` rows are never mutated or deleted (M17).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.index import upsert_run
from daydream.archive.manifest import Manifest
from daydream.git_ops import GitError
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.training import backfill
from daydream.training.backfill import run_backfill

_FP_A = "a" * 64


def _reply(body: str, assoc: str = "OWNER") -> dict[str, Any]:
    """A qualifying human reply to the daydream finding comment (id 1)."""
    return {
        "id": 2,
        "in_reply_to_id": 1,
        "user": {"login": "amelia"},
        "author_association": assoc,
        "body": body,
    }


def _fake_gh(
    *,
    comments: list[dict[str, Any]] | None = None,
    gh_error: int | None = None,
) -> Any:
    """``gh_api(repo, endpoint, **kw)`` responder; ``gh_error`` makes every
    endpoint fail with that HTTP status (deleted repo/PR/comments)."""

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if gh_error is not None:
            raise GitError(f"gh api {endpoint} failed (HTTP {gh_error})")
        if endpoint.endswith("/comments"):
            return list(comments or [])
        if endpoint.endswith("/reviews"):
            return []
        return {"merged": True, "merged_at": None}

    return responder


def _archive_with_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replies: list[dict[str, Any]] | None,
    gh_error: int | None = None,
) -> Path:
    """Archive with one PR-linked session whose reply evidence is ``replies``.

    ``gh_api`` is monkeypatched to return the comment payloads without network.
    """
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        json.dumps({"verdicts": [{"issue_id": 1, "verdict": "consistent"}]})
    )
    (run_dir / "diff.patch").write_text("+new_line\n")
    (run_dir / "findings.json").write_text(
        json.dumps({"findings": [{"fingerprint": _FP_A}]})
    )
    daydream_comment = {
        "id": 1,
        "in_reply_to_id": None,
        "user": {"login": "daydream-runner"},
        "body": f"finding\n\n{finding_marker(_FP_A)}\n\n{DAYDREAM_FOOTER}",
    }
    comments = [daydream_comment, *(replies or [])]
    monkeypatch.setattr(backfill, "_gh_api", _fake_gh(comments=comments, gh_error=gh_error))
    upsert_run(
        archive,
        Manifest(
            session_id="s1",
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            repo_slug="o/r",
            pr_repo="o/r",
            pr_number=7,
            head_sha="h",
            base_branch="main",
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(run_dir),
        ),
    )
    return archive


def _observations(archive: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(archive / "index.db")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM label_observations ORDER BY observed_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def all_observations(archive: Path) -> list[dict[str, Any]]:
    return _observations(archive)


def observation_count(archive: Path) -> int:
    return len(_observations(archive))


def snapshot_rows(archive: Path) -> list[dict[str, Any]]:
    return _observations(archive)


def test_backfill_appends_new_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    summary = run_backfill(archive, dry_run=False)
    assert summary["sessions_reprocessed"] == 1
    rows = all_observations(archive)
    assert rows[-1]["labeler_policy_version"] is not None
    assert rows[-1]["labels"] == '["accepted"]'


def test_backfill_rerun_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unchanged GitHub evidence + unchanged versions ⇒ second run appends nothing (M18)."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    run_backfill(archive, dry_run=False)
    before = observation_count(archive)
    summary = run_backfill(archive, dry_run=False)
    assert observation_count(archive) == before
    assert summary["appended"] == 0 and summary["skipped"] >= 1


def test_backfill_fails_closed_on_deleted_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleted/inaccessible comments ⇒ unknown, never decisive (M19/M22)."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=None, gh_error=404)
    run_backfill(archive, dry_run=False)
    rows = all_observations(archive)
    assert rows[-1]["labels"] == '["unknown"]'


def test_backfill_report_is_machine_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("False positive", assoc="OWNER")])
    summary = run_backfill(archive, dry_run=False, report_path=tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    for key in ("run_label_transitions", "disposition_counts", "parser_rule_count",
                "ambiguous_manual_review", "bot_self_reply_exclusions", "pr_state_counts", "class_balance"):
        assert key in report
    assert summary is not None


def test_backfill_historical_rows_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backfill never mutates or deletes existing label_observations (M17)."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    legacy = snapshot_rows(archive)
    run_backfill(archive, dry_run=False)
    assert snapshot_rows(archive)[: len(legacy)] == legacy

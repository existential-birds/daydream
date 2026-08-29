"""Tests for the backfill pass — re-annotate indexed runs through the harvest
annotation path and append a fresh ``label_observations`` generation, with a
machine-readable migration report.

Covers:
* Append-only backfill: new generations carry the current policy version (M17).
* Idempotent re-run: unchanged evidence + unchanged versions dedup to a no-op (M18).
* Fail-closed on deleted/inaccessible PR comments (M19/M22), including a benign
  404 that escapes ``build_annotation`` into backfill's own benign-escape handler.
* ``dry_run=True`` (headline mode) builds annotations without writing.
* ``RateLimitError`` aborts cleanly; transient errors isolate per-row.
* ``non_pr_skipped``, ``session_filter``, and ``valid_at_override``.
* Machine-readable report keys and values (M20).
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
from daydream.git_ops import GitError, RateLimitError
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.training import backfill
from daydream.training.backfill import run_backfill
from daydream.training.reply_classifier import (
    _ACCEPT_RULES,
    _DISPUTE_RULES,
    _FACTUAL_DISAGREEMENT_RULES,
    _REJECT_RULES,
)

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


_DAYDREAM_COMMENT = {
    "id": 1,
    "in_reply_to_id": None,
    "user": {"login": "daydream-runner"},
    "body": f"finding\n\n{finding_marker(_FP_A)}\n\n{DAYDREAM_FOOTER}",
}


def _thread_comments(replies: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The PR's daydream comment plus the qualifying human replies under it."""
    return [_DAYDREAM_COMMENT, *(replies or [])]


def _add_run(
    tmp_path: Path,
    *,
    session_id: str = "s1",
    run_dir_name: str = "run",
    repo_slug: str = "o/r",
    pr_repo: str | None = "o/r",
    pr_number: int | None = 7,
) -> Path:
    """Index one archived run (no ``gh_api`` patching); returns the archive root.

    Reused by multi-session tests so several runs share one archive.
    """
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / run_dir_name
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        json.dumps({"verdicts": [{"issue_id": 1, "verdict": "consistent"}]})
    )
    (run_dir / "diff.patch").write_text("+new_line\n")
    (run_dir / "findings.json").write_text(
        json.dumps({"findings": [{"fingerprint": _FP_A}]})
    )
    upsert_run(
        archive,
        Manifest(
            session_id=session_id,
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            repo_slug=repo_slug,
            pr_repo=pr_repo,
            pr_number=pr_number,
            head_sha="h",
            base_branch="main",
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(run_dir),
        ),
    )
    return archive


def _archive_with_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replies: list[dict[str, Any]] | None,
    gh_error: int | None = None,
    **run_kwargs: Any,
) -> Path:
    """Archive with one PR-linked session whose reply evidence is ``replies``.

    ``gh_api`` is monkeypatched to return the comment payloads without network.
    """
    archive = _add_run(tmp_path, **run_kwargs)
    monkeypatch.setattr(
        backfill, "_gh_api", _fake_gh(comments=_thread_comments(replies), gh_error=gh_error)
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
    assert rows[-1]["labels"] == '[]'


def test_backfill_report_is_machine_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("False positive", assoc="OWNER")])
    summary = run_backfill(archive, dry_run=False, report_path=tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    for key in ("run_label_transitions", "disposition_counts", "parser_rule_count",
                "ambiguous_manual_review", "bot_self_reply_exclusions", "pr_state_counts", "class_balance"):
        assert key in report
    assert summary is not None
    assert report["summary"] == summary
    assert report["dry_run"] is False
    assert report["run_label_transitions"] == {"s1": {"old_labels": None, "new_labels": ["rejected"]}}
    assert report["disposition_counts"] == {"rejected": 1}
    assert report["pr_state_counts"] == {"merged": 1}
    assert report["class_balance"] == {"rejected": 1}
    assert report["ambiguous_manual_review"] == 0
    assert report["bot_self_reply_exclusions"] == 0
    assert report["parser_rule_count"] == (
        len(_ACCEPT_RULES) + len(_REJECT_RULES) + len(_DISPUTE_RULES) + len(_FACTUAL_DISAGREEMENT_RULES)
    )


def test_backfill_dry_run_builds_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dry_run=True (the module's headline mode) builds annotations but writes nothing."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    summary = run_backfill(archive, dry_run=True, report_path=tmp_path / "report.json")
    assert summary["appended"] == 0
    assert summary["skipped"] == 0
    assert summary["sessions_reprocessed"] == 1
    assert observation_count(archive) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["dry_run"] is True
    assert report["summary"] == summary


def test_backfill_fails_closed_on_benign_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A benign 404 escaping build_annotation (merge OK, comment fetch 404) lands in
    backfill's own fail-closed handler: unknown with the current policy version."""
    _add_run(tmp_path)

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            raise GitError(f"gh api {endpoint} failed (HTTP 404)")
        return {"merged": True, "merged_at": None}

    monkeypatch.setattr(backfill, "_gh_api", responder)
    summary = run_backfill(tmp_path / "archive", dry_run=False)
    assert summary["sessions_reprocessed"] == 1
    assert summary["appended"] == 1
    assert summary["errors"] == 0
    rows = all_observations(tmp_path / "archive")
    assert rows[-1]["labels"] == '[]'
    assert rows[-1]["pr_state"] is None


def test_backfill_aborts_on_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RateLimitError aborts the sweep cleanly, preserving the remaining queue."""
    _add_run(tmp_path)

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        raise RateLimitError(f"gh api {endpoint} failed (HTTP 429)", retry_after=5)

    monkeypatch.setattr(backfill, "_gh_api", responder)
    summary = run_backfill(tmp_path / "archive", dry_run=False)
    assert summary["aborted"] == 1
    assert summary["appended"] == 0
    assert observation_count(tmp_path / "archive") == 0


def test_backfill_non_pr_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with no PR link is skipped and counted, never annotated."""
    _add_run(tmp_path, pr_repo=None, pr_number=None)
    monkeypatch.setattr(backfill, "_gh_api", _fake_gh(comments=_thread_comments(None)))
    summary = run_backfill(tmp_path / "archive", dry_run=False)
    assert summary["non_pr_skipped"] == 1
    assert summary["sessions_reprocessed"] == 0
    assert observation_count(tmp_path / "archive") == 0


def test_backfill_session_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """session_filter restricts the queue; filtered-out sessions stay untouched."""
    _archive_with_session(
        tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")],
        session_id="s1", run_dir_name="run_a",
    )
    _archive_with_session(
        tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")],
        session_id="s2", run_dir_name="run_b",
    )
    summary = run_backfill(tmp_path / "archive", dry_run=False, session_filter="s2")
    assert summary["sessions_reprocessed"] == 1
    assert summary["non_pr_skipped"] == 0
    rows = all_observations(tmp_path / "archive")
    assert [row["session_id"] for row in rows] == ["s2"]


def test_backfill_valid_at_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """valid_at_override beats the derived decisive-evidence timestamp."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    run_backfill(archive, dry_run=False, valid_at_override="2026-02-01T00:00:00Z")
    rows = all_observations(archive)
    assert rows[-1]["valid_at"] == "2026-02-01T00:00:00+00:00"


def test_backfill_per_row_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure (HTTP 500) on one row isolates; the next row survives."""
    _add_run(tmp_path, session_id="s1", run_dir_name="run_a", repo_slug="o/fail", pr_repo="o/fail")
    _add_run(tmp_path, session_id="s2", run_dir_name="run_b", repo_slug="o/ok", pr_repo="o/ok")

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if repo == "o/fail":
            raise GitError(f"gh api {endpoint} failed (HTTP 500)")
        if endpoint.endswith("/comments"):
            return _thread_comments([_reply("Fixed in abc123", assoc="OWNER")])
        if endpoint.endswith("/reviews"):
            return []
        return {"merged": True, "merged_at": None}

    monkeypatch.setattr(backfill, "_gh_api", responder)
    summary = run_backfill(tmp_path / "archive", dry_run=False)
    assert summary["errors"] == 1
    assert summary["appended"] == 1
    assert summary["sessions_reprocessed"] == 1
    rows = all_observations(tmp_path / "archive")
    assert [row["session_id"] for row in rows] == ["s2"]
    assert rows[-1]["labels"] == '["accepted"]'


def test_backfill_historical_rows_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backfill never mutates or deletes existing label_observations (M17)."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    legacy = snapshot_rows(archive)
    run_backfill(archive, dry_run=False)
    assert snapshot_rows(archive)[: len(legacy)] == legacy


def test_backfill_old_labels_human_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_label_transitions['old_labels'] reflects the archive's winner
    projection: a human override beats a newer auto row (human-first)."""
    archive = _archive_with_session(tmp_path, monkeypatch, replies=[_reply("Fixed in abc123", assoc="OWNER")])
    conn = sqlite3.connect(archive / "index.db")
    try:
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, source) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "2026-01-02T00:00:00Z", '["auto"]', "v1", "auto"),
        )
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, source) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "2026-01-01T00:00:00Z", '["human"]', "v1", "human"),
        )
        conn.commit()
    finally:
        conn.close()
    summary = run_backfill(archive, dry_run=True, report_path=tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    transition = report["run_label_transitions"]["s1"]
    assert transition["old_labels"] == ["human"]
    assert summary["sessions_reprocessed"] == 1

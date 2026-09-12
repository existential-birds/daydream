"""Corpus harvest composition through real git, archive, and resume storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from daydream import cli, git_ops
from daydream.archive.index import label_observation_history, query_runs, upsert_run
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.training import labeler_versions, reward
from tests.harness.git_helpers import git
from tests.harness.trajectory import make_manifest


@pytest.mark.parametrize("outcome", ["accepted", "unanswered", "malformed", "rate_limited"])
def test_corpus_harvest_archive_and_resume_journey(
    tmp_path: Path,
    archive_dir: Path,
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    session = "cli-harvest-session"
    head = git(git_repo, "rev-parse", "HEAD")
    run_dir = archive_dir / session
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        json.dumps({"verdicts": [{"verdict": "consistent"}]}), encoding="utf-8"
    )
    fingerprint = "a" * 64
    (run_dir / "findings.json").write_text(
        json.dumps({"findings": [{"fingerprint": fingerprint}]}), encoding="utf-8"
    )
    manifest = make_manifest(
        session_id=session,
        archive_path=str(run_dir),
        source_path=str(git_repo),
        repo_slug="org/repo",
        branch="main",
        base_branch="main",
        head_sha=head,
        grounding_rate=1.0,
        pr_number=None,
        pr_repo=None,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    original_manifest = manifest_path.read_bytes()
    upsert_run(archive_dir, manifest)
    if outcome == "malformed":
        with closing(sqlite3.connect(archive_dir / "index.db")) as connection:
            connection.execute("UPDATE runs SET archive_path = '' WHERE session_id = ?", (session,))
            connection.commit()

    endpoints: list[str] = []
    rate_limited = outcome == "rate_limited"
    comments: list[dict[str, Any]] = [{
        "id": 1,
        "user": {"login": "daydream-runner"},
        "body": f"finding\n\n{finding_marker(fingerprint)}\n\n{DAYDREAM_FOOTER}",
    }]
    if outcome != "unanswered":
        comments.append({
            "id": 2,
            "in_reply_to_id": 1,
            "user": {"login": "reviewer"},
            "author_association": "OWNER",
            "body": "Good catch, applied.",
            "created_at": "2026-09-10T10:00:00Z",
        })

    def github_boundary(
        repo: Path,
        endpoint: str,
        *,
        auth: git_ops.GitHubAuth,
        **kwargs: Any,
    ) -> Any:
        assert repo == Path(".")
        assert auth is git_ops.INHERIT_GITHUB_AUTH
        endpoints.append(endpoint)
        if rate_limited:
            raise git_ops.RateLimitError("rate limit", retry_after=0)
        if endpoint == f"repos/org/repo/commits/{head}/pulls":
            assert kwargs["paginate"] is True
            return [{"number": 7, "head": {"sha": head, "repo": {"full_name": "org/repo"}}}]
        if endpoint == "repos/org/repo/pulls/7":
            return {"merged": True, "merged_at": "2026-09-10T11:00:00Z", "user": {"login": "author"}}
        assert kwargs["paginate"] is True
        if endpoint == "repos/org/repo/pulls/7/reviews":
            return [{"user": {"login": "reviewer"}}]
        if endpoint == "repos/org/repo/pulls/7/comments":
            return comments
        pytest.fail(f"unexpected GitHub endpoint: {endpoint}")

    monkeypatch.setattr(git_ops, "gh_api", github_boundary)
    cache_dir = tmp_path / "cache"

    def harvest(cache: Path) -> int:
        with pytest.raises(SystemExit) as exit_info:
            cli.main([
                "corpus", "harvest", "--archive-dir", str(archive_dir),
                "--cache-dir", str(cache), "--gh-spacing-sec", "0",
            ])
        return int(exit_info.value.code or 0)

    first_exit = harvest(cache_dir)
    if outcome == "malformed":
        assert first_exit == 1
        diagnostic = capsys.readouterr().out
        assert session in diagnostic
        assert "archive_path" in diagnostic
        assert endpoints == []
        assert not cache_dir.exists()
        assert manifest_path.read_bytes() == original_manifest
        assert label_observation_history(archive_dir, session) == []
        return

    progress_path = cache_dir / "progress.jsonl"
    if rate_limited:
        assert first_exit == 1
        assert not progress_path.exists()
        assert label_observation_history(archive_dir, session) == []
        rate_limited = False
        assert harvest(cache_dir) == 0
    else:
        assert first_exit == 0

    row, = query_runs(archive_dir)
    assert (row["pr_number"], row["pr_repo"]) == (7, "org/repo")
    assert json.loads(manifest_path.read_text())["code_context"]["base_sha"] == head
    annotation, = label_observation_history(archive_dir, session)
    assert annotation["reward_version"] == reward.REWARD_VERSION
    assert json.loads(annotation["labels"]) == ([] if outcome == "unanswered" else ["accepted"])
    assert annotation["valid_at"] == (
        "2026-09-10T11:00:00+00:00" if outcome == "unanswered" else "2026-09-10T10:00:00+00:00"
    )
    marker, = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert marker["session_id"] == session
    assert marker["labeler_policy_version"] == labeler_versions.LABELER_POLICY_VERSION
    assert set(endpoints) == {
        f"repos/org/repo/commits/{head}/pulls",
        "repos/org/repo/pulls/7",
        "repos/org/repo/pulls/7/reviews",
        "repos/org/repo/pulls/7/comments",
    }

    previous_endpoints = list(endpoints)
    previous_progress = progress_path.read_bytes()
    assert harvest(cache_dir) == 0
    assert endpoints == previous_endpoints
    assert progress_path.read_bytes() == previous_progress
    assert label_observation_history(archive_dir, session) == [annotation]
    # A fresh resume cache re-acquires evidence, while SQLite still deduplicates it.
    next_cache = tmp_path / "next-cache"
    assert harvest(next_cache) == 0
    assert label_observation_history(archive_dir, session) == [annotation]
    assert json.loads((next_cache / "progress.jsonl").read_text())["session_id"] == session

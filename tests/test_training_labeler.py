"""Tests for :mod:`daydream.training.labeler`.

Covers:
* Merged PR with clean comments → ``accepted`` label written + observation row.
* PR-less run (no ``pr_repo``/``pr_number``) routed through the local-branch
  signal path; ``posterior_source == "local_branch"``.
* ``--dry-run`` mode short-circuits the writes but still considers the row.
* Per-row exception is caught and counted in ``errors``; other rows complete.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.archive.index import label_observation_history, query_runs, upsert_run
from daydream.archive.manifest import Manifest
from daydream.training.labeler import LabelerConfig, run_label


async def test_labeler_writes_accepted_for_merged_pr_clean_comments(tmp_path: Path, monkeypatch) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(exist_ok=True)
    upsert_run(archive_dir, Manifest(
        session_id="00000000-0000-0000-0000-000000000099",
        archived_at="2026-01-01T00:00:00Z", run_flow="normal", backend="claude",
        repo_slug="org/repo", pr_repo="org/repo", pr_number=42,
        head_sha="abc", base_branch="main", changed_files=["daydream/x.py"],
        archive_path=str(tmp_path / "run-99"),
    ))
    def fake_gh(repo, endpoint, **kw):
        if endpoint == "repos/org/repo/pulls/42":
            return {"merged": True, "merged_at": "2026-01-02T00:00:00Z"}
        if endpoint == "repos/org/repo/pulls/42/comments":
            return []
        raise AssertionError(f"unexpected endpoint {endpoint}")
    monkeypatch.setattr("daydream.training.labeler._gh_api", fake_gh)
    monkeypatch.setattr("daydream.training.labeler._diff_name_only",
                        lambda repo, base, head: ["daydream/x.py"])
    monkeypatch.setattr("daydream.training.labeler._commits_in_window",
                        lambda repo, head, base, days: ["c1"])
    monkeypatch.setattr("daydream.training.labeler._file_at",
                        lambda repo, path, sha: "")

    config = LabelerConfig(archive_dir=archive_dir, dry_run=False, cache_dir=tmp_path / "cache")
    summary = await run_label(config)

    rows = query_runs(archive_dir, "session_id = ?", ("00000000-0000-0000-0000-000000000099",))
    assert json.loads(rows[0]["outcome_labels"]) == ["accepted"]
    hist = label_observation_history(archive_dir, "00000000-0000-0000-0000-000000000099")
    assert len(hist) == 1
    assert summary["labeled"] == 1


async def test_labeler_uses_local_branch_signal_for_no_pr_run(tmp_path: Path, monkeypatch) -> None:
    """No pr_repo/pr_number → labeler uses local_commit_applied_signal."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(exist_ok=True)
    archive_path = tmp_path / "run-no-pr"
    archive_path.mkdir()
    (archive_path / "diff.patch").write_text("+ new_line\n")  # minimal valid patch
    upsert_run(archive_dir, Manifest(
        session_id="00000000-0000-0000-0000-0000000000aa",
        archived_at="2026-01-01T00:00:00Z", run_flow="normal", backend="claude",
        repo_slug="org/repo", pr_repo=None, pr_number=None,
        head_sha="abc", branch="feat/x", base_branch="main",
        archive_path=str(archive_path),
    ))
    monkeypatch.setattr("daydream.training.labeler._commits_since",
                        lambda repo, branch, since: ["c1"])
    monkeypatch.setattr("daydream.training.labeler._file_at",
                        lambda repo, path, sha: "new_line\n")

    config = LabelerConfig(archive_dir=archive_dir, dry_run=False, cache_dir=tmp_path / "cache")
    summary = await run_label(config)  # noqa: F841 - verbatim from plan; assertions below cover behavior

    rows = query_runs(archive_dir, "session_id = ?", ("00000000-0000-0000-0000-0000000000aa",))
    assert json.loads(rows[0]["outcome_labels"]) == ["accepted"]
    hist = label_observation_history(archive_dir, "00000000-0000-0000-0000-0000000000aa")
    assert hist[0]["pr_state"] is None  # no PR
    rub = json.loads(hist[0]["rubric_json"])
    assert rub["posterior_source"] == "local_branch"


async def test_labeler_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(exist_ok=True)
    upsert_run(archive_dir, Manifest(
        session_id="00000000-0000-0000-0000-000000000098",
        archived_at="2026-01-01T00:00:00Z", run_flow="normal", backend="claude",
        repo_slug="org/repo", pr_repo="org/repo", pr_number=43,
        head_sha="abc", base_branch="main",
        archive_path=str(tmp_path / "run-98"),
    ))
    monkeypatch.setattr("daydream.training.labeler._gh_api",
                        lambda r, e, **k: {"merged": True, "merged_at": "2026-01-02T00:00:00Z"}
                        if "pulls/43" in e and not e.endswith("comments") else [])
    monkeypatch.setattr("daydream.training.labeler._diff_name_only", lambda *a: [])
    monkeypatch.setattr("daydream.training.labeler._commits_in_window", lambda *a, **k: [])
    monkeypatch.setattr("daydream.training.labeler._file_at", lambda *a: "")

    config = LabelerConfig(archive_dir=archive_dir, dry_run=True, cache_dir=tmp_path / "cache")
    summary = await run_label(config)

    rows = query_runs(archive_dir, "session_id = ?", ("00000000-0000-0000-0000-000000000098",))
    assert json.loads(rows[0]["outcome_labels"]) == []
    assert summary["would_label"] == 1
    assert summary["labeled"] == 0


async def test_labeler_per_row_exception_does_not_derail_run(tmp_path: Path, monkeypatch) -> None:
    """One bad row counts in errors; subsequent rows still process."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(exist_ok=True)
    for i, sess in enumerate(["s-bad", "s-good"]):
        upsert_run(archive_dir, Manifest(
            session_id=sess, archived_at="2026-01-01T00:00:00Z",
            run_flow="normal", backend="claude",
            repo_slug="org/repo", pr_repo="org/repo", pr_number=100 + i,
            head_sha="abc", base_branch="main",
            archive_path=str(tmp_path / f"run-{sess}"),
        ))
    def flaky_gh(repo, endpoint, **kw):
        if "100" in endpoint:
            raise RuntimeError("simulated rate-limit")
        if endpoint.endswith("pulls/101"):
            return {"merged": True, "merged_at": "2026-01-02T00:00:00Z"}
        return []
    monkeypatch.setattr("daydream.training.labeler._gh_api", flaky_gh)
    monkeypatch.setattr("daydream.training.labeler._diff_name_only", lambda *a: [])
    monkeypatch.setattr("daydream.training.labeler._commits_in_window", lambda *a, **k: [])
    monkeypatch.setattr("daydream.training.labeler._file_at", lambda *a: "")

    config = LabelerConfig(archive_dir=archive_dir, dry_run=False, cache_dir=tmp_path / "cache")
    summary = await run_label(config)
    assert summary["errors"] == 1
    assert summary["labeled"] == 1  # s-good still labeled despite s-bad failing

"""Tests for the harvest pass — bronze signal assembly + per-run annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.index import label_observation_history, latest_label_observation, upsert_run
from daydream.archive.manifest import Manifest
from daydream.git_ops import GitError
from daydream.training.harvest import (
    HarvestConfig,
    _resolve_repo_for_row,
    assemble_scoring_inputs,
    build_annotation,
    run_harvest,
)


def _seed_deep_bronze(tmp_path: Path, *, verdict: str, grounding: float) -> Path:
    """Write a deep-run bronze bundle and return its run directory.

    Mirrors the seeding shape used across the labeler tests
    (``deep/recommendation-verdicts.json`` + ``diff.patch``); ``grounding``
    is accepted for parity with the indexed-row grounding signal supplied
    separately to :func:`build_annotation`.
    """
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        json.dumps({"verdicts": [{"issue_id": 1, "verdict": verdict}]})
    )
    (run_dir / "diff.patch").write_text(
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        "+new_line\n"
    )
    return run_dir


def _fake_gh_merged(merged_at: str):
    """Return a ``gh_api(repo, endpoint, **kw)`` responder for a merged PR.

    Reuses the labeler-test responder shape: keyed on the endpoint, the
    ``pulls/<n>`` endpoint reports merged with the given timestamp and the
    ``comments`` endpoint reports no comments (clean → ``accepted``).
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return []
        return {"merged": True, "merged_at": merged_at}

    return responder


def _unused_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """A ``gh_api`` responder the local-branch path must never call."""
    raise AssertionError(f"gh_api should not be called for a local row (endpoint={endpoint})")


def test_build_annotation_pr_row_carries_label_reward_and_merge_valid_at(tmp_path):
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s1", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir,
                           gh_api=_fake_gh_merged("2026-02-01T00:00:00+00:00"),
                           repo_clone=tmp_path, window_days=30)
    assert ann.labels == ["accepted"]
    assert ann.valid_at == "2026-02-01T00:00:00+00:00"        # PR merge time (Q2)
    assert ann.composite_reward == json.loads(ann.reward_json)["composite"]


def test_build_annotation_shallow_local_row_null_valid_at_reward_present(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()           # no deep/ → shallow
    row = {"session_id": "s2", "pr_repo": None, "pr_number": None, "branch": "feat",
           "head_sha": "h", "archive_path": str(run_dir), "grounding_rate": None,
           "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir, gh_api=_unused_gh,
                           repo_clone=tmp_path, window_days=30)
    assert ann.valid_at is None                               # collapses to observed_at on write
    rb = json.loads(ann.reward_json)
    assert rb["axes_present"]["correctness"] is False         # shallow: no verdicts


def test_assemble_reads_verdicts_and_grounding_from_bronze(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        '{"verdicts":[{"issue_id":1,"verdict":"consistent"}]}'
    )
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 0.75})
    assert inputs.verifier_verdicts == [{"issue_id": 1, "verdict": "consistent"}]
    assert inputs.grounding_rate == 0.75 and inputs.format_valid is True


def test_assemble_shallow_run_has_null_verdicts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": None})
    assert inputs.verifier_verdicts is None


def test_assemble_malformed_verdicts_flags_format_invalid(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text("{not json")
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 1.0})
    assert inputs.format_valid is False


def _seed_archived_deep_run(archive_dir: Path, session_id: str, *, merged_at: str) -> Path:
    """Seed a deep-run bronze bundle and index it under ``archive_dir``.

    ``_seed_deep_bronze`` + ``upsert_run`` (plan note): writes the bronze
    artifacts beside the archive and registers the indexed manifest row that
    :func:`run_harvest` walks. Returns the run directory.
    """
    run_dir = _seed_deep_bronze(archive_dir, verdict="consistent", grounding=1.0)
    upsert_run(
        archive_dir,
        Manifest(
            session_id=session_id,
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            repo_slug="org/repo",
            pr_repo="org/repo",
            pr_number=42,
            head_sha="abc",
            base_branch="main",
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(run_dir),
        ),
    )
    return run_dir


async def test_harvest_writes_one_annotation(tmp_path, archive_dir, monkeypatch):
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _fake_gh_merged("2026-02-01T00:00:00+00:00"))
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    obs = latest_label_observation(archive_dir, "s1")
    assert summary["annotated"] == 1
    assert obs["valid_at"] == "2026-02-01T00:00:00+00:00" and obs["composite_reward"] is not None


async def test_re_harvest_appends_new_generation(tmp_path, archive_dir, monkeypatch):
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _fake_gh_merged("2026-02-01T00:00:00+00:00"))
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1"))
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2"))
    assert len(label_observation_history(archive_dir, "s1")) == 2  # append-only, re-runnable


# ---------------------------------------------------------------------------
# _resolve_repo_for_row
# ---------------------------------------------------------------------------


def test_resolve_repo_for_row_prefers_source_path(tmp_path: Path):
    """source_path is preferred when it exists and contains .git."""
    source = tmp_path / "source_repo"
    source.mkdir()
    (source / ".git").mkdir()
    row = {"source_path": str(source), "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=tmp_path / "cache")
    assert result == source


def test_resolve_repo_for_row_clones_when_source_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Falls through to clone when source_path is absent."""
    cache = tmp_path / "cache"

    def fake_clone(url: str, target: Path, **kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / ".git").mkdir()

    monkeypatch.setattr("daydream.training.harvest.git_ops.clone", fake_clone)
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result == cache / "org" / "repo"


def test_resolve_repo_for_row_fetches_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When the cache clone already exists, fetch instead of clone."""
    cache = tmp_path / "cache"
    cached_repo = cache / "org" / "repo"
    cached_repo.mkdir(parents=True)
    (cached_repo / ".git").mkdir()

    fetched = []
    monkeypatch.setattr("daydream.training.harvest.git_ops.fetch", lambda repo, remote="origin": fetched.append(repo))
    monkeypatch.setattr("daydream.training.harvest.git_ops.clone", lambda *a, **k: pytest.fail("should not clone"))
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result == cached_repo
    assert fetched == [cached_repo]


def test_resolve_repo_for_row_returns_none_when_no_remote(tmp_path: Path):
    """Returns None when neither source_path nor remote_url is available."""
    row = {"source_path": None, "remote_url": None, "repo_slug": None}
    result = _resolve_repo_for_row(row, clone_cache=tmp_path / "cache")
    assert result is None


def test_resolve_repo_for_row_clone_failure_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Clone failure is swallowed and None is returned (no .git left on disk)."""
    cache = tmp_path / "cache"

    monkeypatch.setattr(
        "daydream.training.harvest.git_ops.clone",
        lambda url, target, **kwargs: (_ for _ in ()).throw(GitError("network error")),
    )
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result is None


def test_resolve_repo_for_row_fetch_failure_returns_cached_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fetch failure is swallowed and the existing cached repo path is returned."""
    cache = tmp_path / "cache"
    cached_repo = cache / "org" / "repo"
    cached_repo.mkdir(parents=True)
    (cached_repo / ".git").mkdir()

    monkeypatch.setattr(
        "daydream.training.harvest.git_ops.fetch",
        lambda repo, remote="origin": (_ for _ in ()).throw(GitError("fetch failed")),
    )
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result == cached_repo

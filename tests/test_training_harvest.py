"""Tests for the harvest pass — bronze signal assembly + per-run annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.index import label_observation_history, latest_label_observation, upsert_run
from daydream.archive.manifest import Manifest
from daydream.git_ops import GitError
from daydream.training import harvest
from daydream.training.harvest import (
    HarvestConfig,
    _resolve_repo_for_row,
    assemble_scoring_inputs,
    build_annotation,
    run_harvest,
)
from daydream.training.reward import score_trajectory


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
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            return []
        return {"merged": True, "merged_at": merged_at}

    return responder


def _fake_gh_not_merged():
    """Return a ``gh_api`` responder for an unmerged (closed) PR.

    The ``pulls/<n>`` endpoint reports ``merged: False`` so
    :func:`derive_outcome_label` yields ``"rejected"``; ``comments`` is empty.
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            return []
        return {"merged": False, "merged_at": None}

    return responder


def _fake_gh_not_merged_with_reviewer(login: str = "alice"):
    """Return a ``gh_api`` responder for an unmerged PR with one human reviewer.

    The ``pulls/<n>`` endpoint reports ``merged: False``; the ``reviews``
    endpoint returns a single review authored by *login* so that
    :func:`reviewer_logins_signal` yields a non-empty list.  This lets
    tests drive the production :func:`reviewer_set_penalty_prior` DB query
    (rather than monkeypatching it) to exercise the empty-pool path.
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/reviews"):
            return [{"user": {"login": login}}]
        if endpoint.endswith("/comments"):
            return []
        return {"merged": False, "merged_at": None}

    return responder


def _unused_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """A ``gh_api`` responder the local-branch path must never call."""
    raise AssertionError(f"gh_api should not be called for a local row (endpoint={endpoint})")


def test_build_annotation_pr_row_carries_label_reward_and_merge_valid_at(tmp_path):
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s1", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                           gh_api=_fake_gh_merged("2026-02-01T00:00:00+00:00"),
                           repo_clone=tmp_path, window_days=30)
    assert ann.labels == ["accepted"]
    assert ann.valid_at == "2026-02-01T00:00:00+00:00"        # PR merge time (Q2)
    assert ann.composite_reward == json.loads(ann.reward_json)["composite"]


def test_build_annotation_applies_posterior_penalty_for_rejected_pr(tmp_path):
    # A not-merged PR row → derive_outcome_label == "rejected". Its bronze
    # (consistent verdict, full grounding) yields a positive intrinsic composite.
    # Under C5 the posterior reject penalty is a SIBLING field — it does NOT
    # deduct from the stored composite, which stays pure intrinsic. The penalty
    # surfaces via false_positive_penalty / posterior_cost in reward_json.
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}

    # Intrinsic-only baseline: same production inputs scored with no posterior.
    intrinsic_inputs = assemble_scoring_inputs(run_dir, row)
    intrinsic_only_composite = score_trajectory(intrinsic_inputs).composite

    payload = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                               gh_api=_fake_gh_not_merged(),
                               repo_clone=tmp_path, window_days=30)

    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    breakdown = json.loads(payload.reward_json)
    assert breakdown["false_positive_penalty"] == 1.0
    assert breakdown["posterior_cost"] == 0.5  # sibling: max(0, 1.0 − 0.5 default prior)
    # Composite is pure intrinsic — the reject label does not fold into it.
    assert payload.composite_reward == intrinsic_only_composite


def test_build_annotation_rejected_pr_empty_pool_uses_default_prior(tmp_path, archive_dir):
    # Exercises the production wiring of reviewer_set_penalty_prior (not monkeypatched).
    # The gh responder returns a real reviewer login ("alice") so reviewer_logins_signal
    # yields ["alice"] and the DB query runs.  The archive is fresh (no prior history),
    # so reviewer_set_penalty_prior returns (None, 0) — the empty-pool path — and the
    # reducer applies the 0.5 default prior.
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej_prod", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    payload = build_annotation(row, run_dir=run_dir, archive_dir=archive_dir,
                               gh_api=_fake_gh_not_merged_with_reviewer("alice"),
                               repo_clone=tmp_path, window_days=30)
    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    rb = json.loads(payload.reward_json)
    # Empty pool → reviewer_set_penalty_prior returns (None, 0) → outcome_prior stays None,
    # prior_n == 0, and the 0.5 default is used: posterior_cost == max(0, 1.0 − 0.5) == 0.5.
    assert rb["outcome_prior"] is None
    assert rb["outcome_prior_n"] == 0
    assert rb["posterior_cost"] == 0.5
    assert payload.reviewer_logins == ["alice"]


def test_build_annotation_shallow_local_row_null_valid_at_reward_present(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()           # no deep/ → shallow
    row = {"session_id": "s2", "pr_repo": None, "pr_number": None, "branch": "feat",
           "head_sha": "h", "archive_path": str(run_dir), "grounding_rate": None,
           "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh,
                           repo_clone=tmp_path, window_days=30)
    assert ann.valid_at is None                               # collapses to observed_at on write
    rb = json.loads(ann.reward_json)
    assert rb["axes_present"]["correctness"] is False         # shallow: no verdicts


def test_build_annotation_pr_uses_pooled_prior_and_persists_reviewers(tmp_path, monkeypatch):
    monkeypatch.setattr(harvest, "reviewer_set_penalty_prior", lambda *a, **k: (0.8, 12))  # >=10 -> empirical
    monkeypatch.setattr(harvest, "reviewer_logins_signal", lambda *a, **k: ["alice", "carol"])
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    p = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh_not_merged(), repo_clone=tmp_path, window_days=30)
    rb = json.loads(p.reward_json)
    assert rb["posterior_cost"] == pytest.approx(0.2)   # max(0, 1.0 - 0.8)
    assert rb["outcome_prior"] == 0.8 and rb["outcome_prior_n"] == 12
    assert p.composite_reward == rb["composite"]        # stored composite is pure intrinsic (C5)
    assert p.has_posterior is True and p.reviewer_logins == ["alice", "carol"]


def test_build_annotation_below_threshold_falls_back_to_default_prior(tmp_path, monkeypatch):
    monkeypatch.setattr(harvest, "reviewer_set_penalty_prior", lambda *a, **k: (0.9, 4))  # n<10
    monkeypatch.setattr(harvest, "reviewer_logins_signal", lambda *a, **k: ["alice"])
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    rb = json.loads(
        build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh_not_merged(), repo_clone=tmp_path, window_days=30).reward_json
    )
    assert rb["outcome_prior"] is None and rb["outcome_prior_n"] == 4  # n recorded; prior None -> 0.5
    assert rb["posterior_cost"] == 0.5


def test_build_annotation_local_row_has_no_reviewer_prior(tmp_path, monkeypatch):
    # PR-less row -> reviewer_logins == [], outcome_prior None, prior query never
    # consulted; still a PosteriorBreakdown when the local verdict maps to a label.
    from daydream.training.labeler_signals import LocalCommitAppliedSignal

    # Force a mapped local verdict ("rejected") so the posterior axis is present.
    monkeypatch.setattr(
        harvest, "local_commit_applied_signal", lambda *a, **k: LocalCommitAppliedSignal(verdict="rejected")
    )
    # The pooled-prior query must never run for a PR-less row.
    monkeypatch.setattr(
        harvest, "reviewer_set_penalty_prior",
        lambda *a, **k: pytest.fail("reviewer_set_penalty_prior must not be called for a local row"),
    )
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_local", "pr_repo": None, "pr_number": None, "branch": "feat",
           "head_sha": "h", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    p = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh,
                         repo_clone=tmp_path, window_days=30)
    assert p.reviewer_logins == []
    assert p.labels == ["rejected"]
    assert p.has_posterior is True
    rb = json.loads(p.reward_json)
    assert rb["outcome_prior"] is None and rb["outcome_prior_n"] == 0
    assert rb["posterior_cost"] == 0.5  # max(0, 1.0 - 0.5 default prior)


def test_build_annotation_asserts_canonical_version(tmp_path, monkeypatch):
    # Force a non-canonical reward_version into the breakdown -> write must be refused.
    def _custom_version_score(*args, **kwargs):
        from daydream.training.reward import RewardWeights
        return score_trajectory(*args, **{**kwargs, "weights": RewardWeights(w_correctness=0.99)})

    monkeypatch.setattr(harvest, "score_trajectory", _custom_version_score)
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_custom", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    with pytest.raises(AssertionError, match="canonical"):
        build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh_not_merged(), repo_clone=tmp_path, window_days=30)


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

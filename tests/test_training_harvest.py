"""Tests for the harvest pass — bronze signal assembly + per-run annotation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    latest_label_observation,
    pr_attached_label_coverage,
    query_runs,
    upsert_run,
)
from daydream.archive.manifest import Manifest
from daydream.git_ops import GitError
from daydream.pr_review import DAYDREAM_FOOTER
from daydream.training import harvest
from daydream.training.backfill_cache import BackfillCache
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


def _fake_gh_orphan_relink(merged_at: str):
    """Return a ``gh_api`` responder that re-links an orphan run to PR 7.

    The ``commits/{sha}/pulls`` probe resolves the row's ``head_sha`` to PR 7
    (head sha ``orphsha``); PR 7 is then a merged PR whose only top-level
    comment is daydream's footer-marked, unresolved finding — so once the run
    is re-linked the rubric must label it ``contested``.
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/pulls") and "/commits/" in endpoint:
            return [{"number": 7, "head": {"sha": "orphsha"}}]
        if endpoint.endswith("/comments"):
            return [
                {
                    "id": 1,
                    "in_reply_to_id": None,
                    "user": {"login": "kevin"},
                    "body": f"finding\n\n{DAYDREAM_FOOTER}",
                }
            ]
        if endpoint.endswith("/reviews"):
            return []
        return {"merged": True, "merged_at": merged_at}

    return responder


def _fake_gh_merged_unresolved_daydream(merged_at: str):
    """Return a ``gh_api`` responder: a merged PR whose only top-level comment
    is daydream's footer-marked comment with NO reply.

    Mirrors :func:`_fake_gh_merged` but the ``comments`` endpoint returns a
    single unresolved daydream finding (identified by ``DAYDREAM_FOOTER``,
    authored as a normal human user — not a ``[bot]``). The rubric must read
    this as one unresolved daydream issue → ``contested``, not ``accepted``.
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return [
                {
                    "id": 1,
                    "in_reply_to_id": None,
                    "user": {"login": "kevin"},
                    "body": f"finding\n\n{DAYDREAM_FOOTER}",
                }
            ]
        if endpoint.endswith("/reviews"):
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
    ann = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                           gh_api=_fake_gh_merged("2026-02-01T00:00:00+00:00"),
                           repo_clone=tmp_path)
    assert ann.labels == ["accepted"]
    assert ann.valid_at == "2026-02-01T00:00:00+00:00"        # PR merge time (Q2)
    assert ann.composite_reward == json.loads(ann.reward_json)["composite"]


def test_build_annotation_applies_posterior_penalty_for_rejected_pr(tmp_path):
    # Not-merged PR -> "rejected". Under C5 the reject penalty is a SIBLING field
    # (false_positive_penalty / posterior_cost), not a deduction from the stored
    # composite, which stays pure intrinsic.
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}

    # Intrinsic-only baseline: same inputs scored with no posterior.
    intrinsic_inputs = assemble_scoring_inputs(run_dir, row)
    intrinsic_only_composite = score_trajectory(intrinsic_inputs).composite

    payload = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                               gh_api=_fake_gh_not_merged(),
                               repo_clone=tmp_path)

    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    breakdown = json.loads(payload.reward_json)
    assert breakdown["false_positive_penalty"] == 1.0
    assert breakdown["posterior_cost"] == 0.5  # sibling: max(0, 1.0 − 0.5 default prior)
    assert payload.composite_reward == intrinsic_only_composite  # pure intrinsic


def test_build_annotation_rejected_pr_empty_pool_uses_default_prior(tmp_path, archive_dir):
    # Production wiring of reviewer_set_penalty_prior (not monkeypatched): reviewer
    # "alice" makes the DB query run, but the fresh archive yields the empty-pool
    # path (None, 0), so the reducer applies the 0.5 default prior.
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej_prod", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    payload = build_annotation(row, run_dir=run_dir, archive_dir=archive_dir,
                               gh_api=_fake_gh_not_merged_with_reviewer("alice"),
                               repo_clone=tmp_path)
    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    rb = json.loads(payload.reward_json)
    # Empty pool -> (None, 0) -> default 0.5: posterior_cost == max(0, 1.0 − 0.5).
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
                           repo_clone=tmp_path)
    assert ann.valid_at is None                               # collapses to observed_at on write
    rb = json.loads(ann.reward_json)
    assert rb["axes_present"]["correctness"] is False         # shallow: no verdicts


def test_build_annotation_rejected_pr_populated_prior_drives_pool(tmp_path, archive_dir):
    # Seed a past label_observation for "alice", then annotate a rejected PR with
    # the same reviewer: the DB query must find the seeded row (prior_n >= 1, pool
    # non-empty) even at n < 10. Exercises before_valid_at filtering against a
    # non-empty archive — the gap the empty-pool test cannot detect.
    prior_session_id = "s_prior_alice"
    upsert_run(
        archive_dir,
        Manifest(
            session_id=prior_session_id,
            archived_at="2025-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            repo_slug="org/repo",
            pr_repo="org/repo",
            pr_number=1,
            head_sha="aaa",
            base_branch="main",
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(tmp_path),
        ),
    )
    append_label_observation(
        archive_dir,
        prior_session_id,
        labels=["rejected"],
        pr_state="closed",
        labeler_version="test",
        evidence_sha=None,
        valid_at="2025-06-01T00:00:00Z",   # strictly in the past
        reviewer_logins=["alice"],
        has_posterior=True,
    )

    run_dir = _seed_deep_bronze(tmp_path / "current_run", verdict="consistent", grounding=1.0)
    row = {
        "session_id": "s_rej_populated",
        "pr_repo": "o/r",
        "pr_number": 9,
        "head_sha": "h",
        "base_branch": "main",
        "archive_path": str(run_dir),
        "grounding_rate": 1.0,
        "changed_files": "[]",
    }
    payload = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=archive_dir,
        gh_api=_fake_gh_not_merged_with_reviewer("alice"),
        repo_clone=tmp_path,
    )
    assert payload.labels == ["rejected"]
    rb = json.loads(payload.reward_json)
    # Seeded prior row found (prior_n >= 1); n < 10 so outcome_prior is None, but
    # prior_n reflecting the pool proves the query ran against real history.
    assert rb["outcome_prior_n"] >= 1, (
        f"expected prior_n >= 1 from seeded archive, got {rb['outcome_prior_n']}"
    )
    assert rb["outcome_prior"] is None   # n < 10 threshold → fallback to default
    assert rb["posterior_cost"] == 0.5   # default prior applied


def test_build_annotation_pr_uses_pooled_prior_and_persists_reviewers(tmp_path, monkeypatch):
    monkeypatch.setattr(harvest, "reviewer_set_penalty_prior", lambda *a, **k: (0.8, 12))  # n>=10 -> empirical
    monkeypatch.setattr(harvest, "reviewer_logins_signal", lambda *a, **k: ["alice", "carol"])
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    p = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh_not_merged(), repo_clone=tmp_path)
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
                         gh_api=_fake_gh_not_merged(), repo_clone=tmp_path).reward_json
    )
    assert rb["outcome_prior"] is None and rb["outcome_prior_n"] == 4  # n recorded; prior None -> 0.5
    assert rb["posterior_cost"] == 0.5


def test_build_annotation_local_row_has_no_reviewer_prior(tmp_path, monkeypatch):
    # PR-less row -> reviewer_logins == [], outcome_prior None, prior query never
    # consulted; still a PosteriorBreakdown when the local verdict maps to a label.
    from daydream.training.labeler_signals import LocalCommitAppliedSignal

    # Mapped local verdict "rejected" so the posterior axis is present.
    monkeypatch.setattr(
        harvest, "local_commit_applied_signal", lambda *a, **k: LocalCommitAppliedSignal(verdict="rejected")
    )
    monkeypatch.setattr(
        harvest, "reviewer_set_penalty_prior",
        lambda *a, **k: pytest.fail("reviewer_set_penalty_prior must not be called for a local row"),
    )
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_local", "pr_repo": None, "pr_number": None, "branch": "feat",
           "head_sha": "h", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    p = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh,
                         repo_clone=tmp_path)
    assert p.reviewer_logins == []
    assert p.labels == ["rejected"]
    assert p.has_posterior is True
    rb = json.loads(p.reward_json)
    assert rb["outcome_prior"] is None and rb["outcome_prior_n"] == 0
    assert rb["posterior_cost"] == 0.5  # max(0, 1.0 - 0.5 default prior)


def test_build_annotation_asserts_canonical_version(tmp_path, monkeypatch):
    # Non-canonical reward_version -> the write must be refused.
    def _custom_version_score(*args, **kwargs):
        from daydream.training.reward import RewardWeights
        return score_trajectory(*args, **{**kwargs, "weights": RewardWeights(w_correctness=0.99)})

    monkeypatch.setattr(harvest, "score_trajectory", _custom_version_score)
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_custom", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    with pytest.raises((AssertionError, RuntimeError), match="canonical"):
        build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh_not_merged(), repo_clone=tmp_path)


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


def test_assemble_declined_deep_run_null_verdicts_keeps_format_valid(tmp_path: Path):
    # Verify relocation: a declined deep run writes the deep bundle (records,
    # merged report) but skips recommendation verification, so no
    # recommendation-verdicts.json exists. Harvest must treat the absent
    # verdicts as expected (verifier_verdicts=None, format gate intact), not
    # floor format_valid.
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    # stack records are present and valid (as on a real declined deep run)
    (run_dir / "deep" / "stack-python-records.json").write_text(
        json.dumps({"records": [{"id": "i1"}]})
    )
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 0.5})
    assert inputs.verifier_verdicts is None          # declined ⇒ no verdicts
    assert inputs.format_valid is True                # absence is expected, not malformed


def test_assemble_malformed_verdicts_flags_format_invalid(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text("{not json")
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 1.0})
    assert inputs.format_valid is False


def test_score_trajectory_grounding_axis_present_on_default_run(tmp_path: Path):
    """AC2: a default-run row (eval-by-default populated grounding_rate) yields a
    non-absent grounding axis; a --no-eval row (grounding_rate=None) omits it.

    Drives the real harvest → reward path: assemble_scoring_inputs reads the
    indexed ``grounding_rate`` and score_trajectory flags the axis present.
    """
    from daydream.training.reward import score_trajectory

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Default run: eval ran by default, so the manifest row carries grounding_rate.
    default_inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 1.0})
    default_breakdown = score_trajectory(default_inputs)
    assert default_breakdown.axes_present["grounding"] is True
    assert default_breakdown.grounding == 1.0

    # --no-eval run: no grounding_rate, so the axis is absent.
    no_eval_inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": None})
    no_eval_breakdown = score_trajectory(no_eval_inputs)
    assert no_eval_breakdown.axes_present["grounding"] is False
    assert no_eval_breakdown.grounding is None


def _seed_archived_deep_run(
    archive_dir: Path, session_id: str, *, merged_at: str, source_path: Path | None = None
) -> Path:
    """Seed a deep-run bronze bundle and index it under ``archive_dir``.

    ``_seed_deep_bronze`` + ``upsert_run`` (plan note): writes the bronze
    artifacts beside the archive and registers the indexed manifest row that
    :func:`run_harvest` walks. Returns the run directory.

    When ``source_path`` is supplied it is recorded on the manifest so
    :func:`_resolve_repo_for_row` resolves a working tree for the row (the
    caller seeds a ``.git`` dir there), making ``clone_resolved`` True.
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
            source_path=str(source_path) if source_path else None,
        ),
    )
    return run_dir


def _seed_orphan_run(
    archive_dir: Path, bronze_parent: Path, *, session_id: str, head_sha: str = "orphsha"
) -> Path:
    """Seed an orphan deep run (no PR linkage) and index it under ``archive_dir``.

    Mirrors :func:`_seed_archived_deep_run` but writes the orphan manifest shape
    the re-link path consumes: ``pr_number``/``pr_repo`` are ``None`` and the row
    carries only a ``branch``/``head_sha``. Bronze artifacts go under
    ``bronze_parent``; returns the run directory.
    """
    run_dir = _seed_deep_bronze(bronze_parent, verdict="consistent", grounding=1.0)
    upsert_run(
        archive_dir,
        Manifest(
            session_id=session_id,
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            repo_slug="org/repo",
            branch="feat/x",
            head_sha=head_sha,
            base_branch="main",
            pr_number=None,
            pr_repo=None,
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(run_dir),
        ),
    )
    return run_dir


def _seed_pr_runs(archive_dir: Path, bronze_parent: Path, count: int) -> None:
    """Seed ``count`` PR-attached deep runs (sessions ``s1``..``sN``, ``pr_number`` 1..N).

    Each run gets its own bronze dir under ``bronze_parent/<session_id>`` so the
    index carries ``count`` distinct rows; ``pr_number`` matches the session index
    so a fake ``_gh_api`` can identify the row from the PR endpoint.
    """
    for pr_number in range(1, count + 1):
        sid = f"s{pr_number}"
        run_dir = _seed_deep_bronze(bronze_parent / sid, verdict="consistent", grounding=1.0)
        upsert_run(
            archive_dir,
            Manifest(
                session_id=sid,
                archived_at="2026-01-01T00:00:00Z",
                run_flow="normal",
                backend="claude",
                repo_slug="org/repo",
                pr_repo="org/repo",
                pr_number=pr_number,
                head_sha="abc",
                base_branch="main",
                grounding_rate=1.0,
                changed_files=["app.py"],
                archive_path=str(run_dir),
            ),
        )


async def test_harvest_writes_one_annotation(tmp_path, archive_dir, monkeypatch):
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _fake_gh_merged("2026-02-01T00:00:00+00:00"))
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    obs = latest_label_observation(archive_dir, "s1")
    assert summary["annotated"] == 1
    assert obs["valid_at"] == "2026-02-01T00:00:00+00:00" and obs["composite_reward"] is not None


async def test_harvest_labels_unresolved_daydream_comment_contested(tmp_path, archive_dir, monkeypatch):
    """Real-path: a merged PR whose daydream comment is unresolved → ``contested``.

    Bug 1 regression guard. daydream posts review comments as a normal
    authenticated (non-``[bot]``) user, so the old ``[bot]`` author check made
    its findings invisible → every merged PR was mislabeled ``accepted``. This
    drives ``run_harvest`` end-to-end and asserts the persisted run label is
    ``contested``, the rubric's correct verdict for an unresolved finding.
    """
    _seed_archived_deep_run(archive_dir, "s-contest", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh_merged_unresolved_daydream("2026-02-01T00:00:00+00:00"),
    )
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    row = query_runs(archive_dir, "session_id = ?", ("s-contest",))[0]
    assert json.loads(row["outcome_labels"]) == ["contested"]


async def test_harvest_relinks_orphan_run_and_labels_it(tmp_path, archive_dir, monkeypatch):
    """Real-path: an orphan run (PR opened after launch) is re-linked at harvest.

    Bug 2. The default deep loop archives before the PR exists, freezing
    ``pr_number=None``. Harvest must re-link the orphan row by its ``head_sha``
    (via ``commits/{sha}/pulls``), persist the linkage, and then label the run
    through the PR path. Drives ``run_harvest`` end-to-end and asserts both the
    persisted linkage and the resulting ``contested`` label.
    """
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh_orphan_relink("2026-02-01T00:00:00+00:00"),
    )
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    row = query_runs(archive_dir, "session_id = ?", ("s-orph",))[0]
    assert row["pr_number"] == 7 and row["pr_repo"] == "org/repo"  # linkage persisted
    assert json.loads(row["outcome_labels"]) == ["contested"]  # now labelable (was orphan)


@pytest.mark.parametrize("with_clone", [False, True])
async def test_harvest_fork_pr_404_degrades_not_drops(tmp_path, archive_dir, monkeypatch, with_clone):
    """Real-path: a benign ``pulls/<n>`` 404 degrades to the local posterior.

    A fork-PR (or deleted-PR) fetch raises ``GitError`` with an HTTP 404. The
    fix catches benign ``GitError`` in ``build_annotation``'s PR-posterior block
    and falls back to ``_build_rubric_local`` rather than dropping the run as a
    hard error. Drives ``run_harvest`` end-to-end and asserts the fix's central
    contract: the run is still ANNOTATED (not dropped) and the benign 404 is NOT
    counted as an error (``errors == 0``).

    The degrade is proven observably on two axes:
      1. ``pr_state`` is ``None`` on the persisted observation — a real PR rubric
         stamps ``"merged"``/``"closed"``; the benign 404 instead routes through
         ``_build_rubric_local`` (``posterior_source="local_branch"``), so no
         fabricated ``merged=False`` PR rubric ever drove the label.
      2. The persisted label is EMPTY (``"unknown"``), NOT ``"rejected"``.

    Parametrized over ``with_clone``:
      * ``False`` — no working tree resolves; ``clone_resolved=False`` already
        forces ``"unknown"`` (the original no-clone case).
      * ``True`` — a git working tree IS resolved (``source_path`` carries a
        ``.git`` dir), so ``clone_resolved=True``. The #166 invariant must STILL
        force ``"unknown"``: a PR-shaped row whose merge evidence was merely
        unavailable is ineligible for the PR-less commit walk, which on this
        branch-less row would otherwise emit the false-negative ``"rejected"``.
    """
    source_path = None
    if with_clone:
        source_path = tmp_path / "clone"
        (source_path / ".git").mkdir(parents=True)
    _seed_archived_deep_run(
        archive_dir, "s-fork", merged_at="2026-02-01T00:00:00+00:00", source_path=source_path
    )

    def _gh_fork_404(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if re.search(r"/pulls/\d+", endpoint):
            raise GitError("gh: Not Found (HTTP 404)")
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            return []
        return {}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_fork_404)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0  # benign 404 degraded; not a hard error
    obs = latest_label_observation(archive_dir, "s-fork")
    assert obs is not None  # still annotated (not dropped)
    # pr_state is None on the local-branch rubric (not "merged"/"closed"):
    # observable proof the fix degraded to the local path rather than
    # fabricating a merged=False PR rubric.
    assert obs["pr_state"] is None
    # The central #166 contract: NOT mislabeled "rejected". With no resolvable
    # clone the local posterior is "unknown" → empty outcome_labels.
    row = query_runs(archive_dir, "session_id = ?", ("s-fork",))[0]
    assert json.loads(row["outcome_labels"]) == []


async def test_harvest_orphan_422_degrades_not_drops(tmp_path, archive_dir, monkeypatch):
    """Real-path: a benign ``commits/<sha>/pulls`` 422 degrades to local.

    An orphan run whose head SHA was never pushed yields an HTTP 422 from the
    ``commits/{sha}/pulls`` link probe. The fix catches benign ``GitError`` at
    the orphan re-link site and degrades to the local-branch posterior (the row
    stays ``pr_number=None``) instead of dropping the run. Drives ``run_harvest``
    end-to-end and asserts the run is still annotated (``errors == 0``, a label
    observation exists), the linkage was NOT applied, and — with no resolvable
    clone — the label degrades to ``"unknown"`` (empty), never ``"rejected"``.
    """
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph-422")

    def _gh_unpushed_422(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if "/commits/" in endpoint and endpoint.endswith("/pulls"):
            raise GitError("gh: No commit found for SHA (HTTP 422)")
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            return []
        return {}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0  # benign 422 degraded; not a hard error
    assert latest_label_observation(archive_dir, "s-orph-422") is not None  # annotated via local path
    row = query_runs(archive_dir, "session_id = ?", ("s-orph-422",))[0]
    assert row["pr_number"] is None and row["pr_repo"] is None  # linkage NOT applied
    # No resolvable clone → local posterior is "unknown", never "rejected".
    assert json.loads(row["outcome_labels"]) == []


async def test_harvest_dry_run_mutates_row_in_memory_but_suppresses_set_run_pr_link(
    tmp_path, archive_dir, monkeypatch
):
    """dry_run=True: in-memory linkage preview is applied but not persisted.

    The contract (harvest.py lines 832-836): when an orphan run re-links to a
    PR, ``row['pr_number']`` and ``row['pr_repo']`` are mutated unconditionally
    so ``build_annotation`` sees the linked PR and produces a PR-path annotation.
    The ``set_run_pr_link`` DB write is guarded by ``if not config.dry_run`` and
    must not fire.

    This test exercises the real ``set_run_pr_link`` code path (no spy/patch):
    if the guard is broken the function will actually write to the DB and the
    DB-state assertions below will catch it.
    """
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph-dry")

    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh_orphan_relink("2026-02-01T00:00:00+00:00"),
    )

    summary = await run_harvest(
        HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c", dry_run=True)
    )

    # In-memory linkage drove build_annotation through the PR path:
    assert summary["would_annotate"] == 1
    assert summary["annotated"] == 0

    # DB row stays unlinked (real set_run_pr_link was not called):
    row = query_runs(archive_dir, "session_id = ?", ("s-orph-dry",))[0]
    assert row["pr_number"] is None
    assert row["pr_repo"] is None

    # No label observation written:
    assert latest_label_observation(archive_dir, "s-orph-dry") is None


async def test_harvest_leaves_true_local_run_unlinked(tmp_path, archive_dir, monkeypatch):
    """Real-path: a local-only run (no PR ever opened) flows the local path.

    Bug 2 guard. The ``commits/{sha}/pulls`` probe returns no PR, so the row
    stays unlinked and must not be force-linked or errored — it flows the
    existing local-branch posterior path unchanged.
    """
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-local", head_sha="localsha")

    def _gh_no_pr(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/pulls") and "/commits/" in endpoint:
            return []  # no PR ever opened
        raise AssertionError(f"PR endpoints must not be hit for an unlinked local run ({endpoint})")

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_no_pr)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    row = query_runs(archive_dir, "session_id = ?", ("s-local",))[0]
    assert row["pr_number"] is None
    assert summary["errors"] == 0


async def test_re_harvest_is_idempotent(tmp_path, archive_dir, monkeypatch):
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _fake_gh_merged("2026-02-01T00:00:00+00:00"))
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1"))
    second = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2"))
    assert len(label_observation_history(archive_dir, "s1")) == 1  # deduped
    assert second["skipped"] == 1 and second["annotated"] == 0


async def test_re_harvest_appends_on_version_bump(tmp_path, archive_dir, monkeypatch):
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _fake_gh_merged("2026-02-01T00:00:00+00:00"))
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1"))
    monkeypatch.setattr("daydream.training.harvest.reward.REWARD_VERSION", "9999.99.99-bump")
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2"))
    assert len(label_observation_history(archive_dir, "s1")) == 2


async def test_harvest_aborts_cleanly_on_rate_limit_and_preserves_resume(tmp_path, archive_dir, monkeypatch):
    # Two PR rows; PR 1 succeeds, PR 2 hits an exhausted rate-limit on every gh
    # call so the harvest loop must abort cleanly.
    _seed_pr_runs(archive_dir, tmp_path, 2)
    merged = _fake_gh_merged("2026-02-01T00:00:00+00:00")

    def _gh(repo, endpoint, **kw):
        if "/pulls/2" in endpoint or "/2/" in endpoint or endpoint.endswith("/2"):
            raise git_ops.RateLimitError("exhausted")
        return merged(repo, endpoint, **kw)

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh)
    cache_dir = tmp_path / "c"
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=cache_dir))
    assert summary["aborted"] == 1
    # The completed session is preserved for resume; the failed one is not:
    done = BackfillCache(cache_dir=cache_dir, inner=_gh).completed_sessions()
    assert "s1" in done and "s2" not in done


def test_gh_api_backoff_retries_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(harvest, "_rate_limit_sleep", lambda s: slept.append(s))
    seq = [git_ops.RateLimitError("x"), git_ops.RateLimitError("x"), {"ok": True}]

    def _inner(*a, **k):
        v = seq.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(harvest.git_ops, "gh_api", _inner)
    assert harvest._gh_api("o/r", "endpoint") == {"ok": True}
    assert len(slept) == 2


# _resolve_repo_for_row


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


# pr_attached_label_coverage


def _make_manifest(session_id: str = "sess-0001", **overrides: Any) -> Manifest:
    """Build a minimal indexed manifest, mirroring ``test_archive._make_manifest``.

    ``pr_number``/``pr_repo`` are plain ``Manifest`` fields (see
    ``daydream/archive/manifest.py``), so PR-attached rows are produced by
    passing them through ``overrides``.
    """
    defaults: dict[str, Any] = {
        "session_id": session_id,
        "archived_at": "2026-04-29T00:00:00+00:00",
        "status": "complete",
        "run_flow": "normal",
        "skill": "python",
        "model": "opus",
        "backend": "claude",
        "archive_path": "/tmp/archive/runs/sess-0001",
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def test_pr_coverage_helper_counts_decisive(tmp_path: Path):
    for i, label in [(1, "accepted"), (2, "rejected"), (3, "unknown")]:
        upsert_run(tmp_path, _make_manifest(session_id=f"p{i}", pr_number=i, pr_repo="o/r"))
        append_label_observation(
            tmp_path,
            f"p{i}",
            labels=[label],
            pr_state=None,
            labeler_version="auto",
            evidence_sha=f"s{i}",
            source="auto",
        )
    upsert_run(tmp_path, _make_manifest(session_id="local1"))  # no pr_number — excluded
    cov = pr_attached_label_coverage(tmp_path)
    assert cov["pr_attached"] == 3 and cov["decisive"] == 2  # accepted+rejected, not unknown


async def test_harvest_propagates_transient_giterror_for_retry(tmp_path, archive_dir, monkeypatch):
    """Real-path: a transient GitError (HTTP 500) on /comments propagates, not degrades.

    ``/pulls/{n}`` succeeds (merged) but the ``/comments`` fetch raises a
    transient HTTP 500. Unlike a benign 404/422 (PR genuinely absent), a server
    error is *recoverable*: degrading it to the local posterior and caching the
    row "done" would permanently lose the merge evidence. Per #166 the row must
    instead surface as a hard error and stay un-cached so a later resume retries
    it. Drives ``run_harvest`` end-to-end and asserts the row is counted in
    ``errors``, is NOT annotated, and is NOT marked done in the resume cache.
    """
    _seed_archived_deep_run(archive_dir, "s-transient-500", merged_at="2026-02-01T00:00:00+00:00")
    cache_dir = tmp_path / "c"

    def _gh_merge_ok_comments_500(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if re.search(r"/pulls/\d+$", endpoint):
            return {"merged": True, "merged_at": "2026-02-01T00:00:00+00:00"}
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            raise GitError("gh: Internal Server Error (HTTP 500)")
        return {}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_merge_ok_comments_500)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=cache_dir))

    assert summary["errors"] == 1  # transient 500 propagated, not degraded
    assert latest_label_observation(archive_dir, "s-transient-500") is None  # not annotated
    # Resume contract: the failed row is NOT cached "done", so a later run retries it.
    done = BackfillCache(cache_dir=cache_dir, inner=_gh_merge_ok_comments_500).completed_sessions()
    assert "s-transient-500" not in done


async def test_harvest_does_not_discard_confirmed_merge_on_benign_comment_error(
    tmp_path, archive_dir, monkeypatch
):
    """Real-path: a 404 on /comments after a confirmed merge propagates, not degrades.

    ``/pulls/{n}`` succeeds (merged) but the ``/comments`` fetch raises a benign
    HTTP 404. Once the merge status is confirmed the PR provably exists, so a
    404 on its comments sub-resource cannot mean "PR absent" — it is transient.
    The old behavior degraded such a row to a local-branch posterior, discarding
    the confirmed ``PRMergeSignal`` (and its ``valid_at``) and caching an
    ``unknown`` label. The row must instead surface as a hard error and stay
    un-cached so a later resume retries it and recovers the merge evidence.
    """
    _seed_archived_deep_run(archive_dir, "s-merge-comments-404", merged_at="2026-02-01T00:00:00+00:00")
    cache_dir = tmp_path / "c"

    def _gh_merge_ok_comments_404(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if re.search(r"/pulls/\d+$", endpoint):
            return {"merged": True, "merged_at": "2026-02-01T00:00:00+00:00"}
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            raise GitError("gh: Not Found (HTTP 404)")
        return {}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_merge_ok_comments_404)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=cache_dir))

    assert summary["errors"] == 1  # benign comment 404 propagated, merge evidence not discarded
    assert latest_label_observation(archive_dir, "s-merge-comments-404") is None  # not annotated
    done = BackfillCache(cache_dir=cache_dir, inner=_gh_merge_ok_comments_404).completed_sessions()
    assert "s-merge-comments-404" not in done


async def test_harvest_keeps_labeled_row_when_reviewer_lookup_errors(tmp_path, archive_dir, monkeypatch):
    """Real-path: a GitError on the ``/reviews`` lookup must not drop a labeled row.

    ``/pulls/{n}`` resolves the PR outcome (merged → ``accepted``) and
    ``/comments`` succeeds, but the auxiliary ``/reviews`` lookup feeding the
    reviewer-set prior raises ``GitError``. Because the PR outcome is already
    decided, the row must still be annotated with its resolved label (degrading
    only the optional prior), not dropped as a hard error through the per-row
    catch-all. Drives ``run_harvest`` end-to-end.
    """
    _seed_archived_deep_run(archive_dir, "s-reviews-err", merged_at="2026-02-01T00:00:00+00:00")

    def _gh_reviews_fail(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/reviews"):
            raise GitError("gh: Internal Server Error (HTTP 500)")
        if endpoint.endswith("/comments"):
            return []
        return {"merged": True, "merged_at": "2026-02-01T00:00:00+00:00"}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_reviews_fail)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0  # reviewer-lookup failure degraded, row not dropped
    obs = latest_label_observation(archive_dir, "s-reviews-err")
    assert obs is not None  # still annotated
    assert obs["pr_state"] == "merged"  # resolved PR outcome preserved (not degraded to local)
    row = query_runs(archive_dir, "session_id = ?", ("s-reviews-err",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]  # merged-PR label kept


async def test_harvest_degrades_benign_giterror_rows_instead_of_dropping(tmp_path, archive_dir, monkeypatch):
    # 10 PR rows: PRs 1..8 merged ("accepted"); PRs 9..10 raise a benign GitError
    # (e.g. fork-PR 404). The fix DEGRADES 9..10 to the local-branch posterior
    # instead of dropping them: errors == 0, annotated == 10. Degraded rows route
    # through _build_rubric_local, so their pr_state is None (NOT a fabricated PR
    # rubric) — observable proof no merged=False rubric drove their label.
    _seed_pr_runs(archive_dir, tmp_path, 10)

    merged = _fake_gh_merged("2026-02-01T00:00:00+00:00")

    def _gh(repo: str, endpoint: str, **kw: Any) -> Any:
        match = re.search(r"/pulls/(\d+)", endpoint)
        number = int(match.group(1)) if match else 0
        if number >= 9:
            # Benign PR-fetch failure: degraded to local posterior, sweep continues.
            raise GitError(f"gh: Not Found (HTTP 404) for PR {number}")
        return merged(repo, endpoint, **kw)

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh)

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["aborted"] == 0  # the GitError rows did NOT abort the sweep
    # All 10 annotate: 8 PR-path "accepted", 2 degraded to local-branch.
    assert summary["annotated"] == 10 and summary["errors"] == 0

    # 8 PR-path rows: decisive "accepted", merged pr_state; 2 degraded rows:
    # pr_state None (local rubric) — proof they took the local path.
    accepted_pr_state = latest_label_observation(archive_dir, "s1")["pr_state"]
    assert accepted_pr_state == "merged"
    assert json.loads(query_runs(archive_dir, "session_id = ?", ("s1",))[0]["outcome_labels"]) == ["accepted"]
    for sid in ("s9", "s10"):
        degraded = latest_label_observation(archive_dir, sid)
        assert degraded is not None  # annotated, not dropped
        assert degraded["pr_state"] is None  # local-branch rubric, not a PR rubric
        # No resolvable clone -> "unknown", NOT the false-negative "rejected" #166 eliminates.
        assert json.loads(query_runs(archive_dir, "session_id = ?", (sid,))[0]["outcome_labels"]) == []

    # Coverage stays honest at 8/10: 8 merged decisive, 2 degraded "unknown"
    # non-decisive — the 80% bar holds without a bogus "rejected".
    cov = pr_attached_label_coverage(archive_dir)
    assert cov["pr_attached"] == 10  # every row stays PR-attached and annotated
    assert cov["decisive"] == 8  # only the 8 merged rows are decisive; "unknown" is not
    assert cov["coverage"] == 0.8

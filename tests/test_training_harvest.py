"""Tests for the harvest pass — bronze signal assembly + per-run annotation."""

from __future__ import annotations

import base64
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.training import harvest, labeler_versions, reward
from daydream.training.backfill_cache import BackfillCache
from daydream.training.harvest import (
    HarvestConfig,
    _resolve_repo_for_row,
    assemble_scoring_inputs,
    build_annotation,
    run_harvest,
)
from daydream.training.reward import score_trajectory
from tests.conftest import _make_repo_with_main
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.trajectory import diff_adding, make_manifest


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
    (run_dir / "diff.patch").write_text(diff_adding("new_line"))
    return run_dir


# Fingerprints recorded at review time (findings.json) and embedded as markers
# in the daydream comment bodies, so the per-finding join can resolve them.
_FP_A = "a" * 64
_FP_B = "b" * 64
_FP_C = "c" * 64


def _finding_comments(
    fp: str,
    *,
    reply: str | None = None,
    reply_created_at: str | None = None,
    reply_author: str = "amelia",
) -> list[dict[str, Any]]:
    """A footer-marked daydream finding comment for ``fp``, optionally replied.

    The reply (when supplied) is a qualifying OWNER human, so it is decisive
    evidence the classifier can label. ``reply_created_at`` feeds the
    decisive-evidence ``valid_at`` derivation.
    """
    comments: list[dict[str, Any]] = [
        {
            "id": 1,
            "in_reply_to_id": None,
            "user": {"login": "daydream-runner"},
            "body": f"finding\n\n{finding_marker(fp)}\n\n{DAYDREAM_FOOTER}",
        }
    ]
    if reply is not None:
        entry: dict[str, Any] = {
            "id": 2,
            "in_reply_to_id": 1,
            "user": {"login": reply_author},
            "author_association": "OWNER",
            "body": reply,
        }
        if reply_created_at is not None:
            entry["created_at"] = reply_created_at
        comments.append(entry)
    return comments


def _write_findings(run_dir: Path, *fps: str) -> None:
    """Record the given fingerprints in the run's ``findings.json`` artifact."""
    (run_dir / "findings.json").write_text(
        json.dumps({"findings": [{"fingerprint": fp} for fp in fps]})
    )


# One footer-marked daydream finding plus a human reply to it — used by tests
# that only exercise plumbing (idempotent re-harvest, resume markers) where the
# label itself is irrelevant; it carries no fingerprint marker, so the scoped
# per-finding join sees nothing and the run labels ``unknown``. A merged PR
# with *no* tracked comments is deliberately NOT an accepted shape either:
# ``comment_resolution`` is ``(0, 0, 0)`` and a merge alone is not evidence
# daydream contributed.
_REPLIED_FINDING: list[dict[str, Any]] = [
    {
        "id": 1,
        "in_reply_to_id": None,
        "user": {"login": "daydream-runner"},
        "body": f"finding\n\n{DAYDREAM_FOOTER}",
    },
    {"id": 2, "in_reply_to_id": 1, "user": {"login": "human"}, "body": "fixed"},
]

# Daydream's footer-marked comment with NO reply, authored as a normal human
# user (not a ``[bot]``): one unresolved daydream issue, which the rubric must
# read as ``unanswered``, never ``accepted``.
_UNRESOLVED_FINDING: list[dict[str, Any]] = [
    {
        "id": 1,
        "in_reply_to_id": None,
        "user": {"login": "kevin"},
        "body": f"finding\n\n{DAYDREAM_FOOTER}",
    }
]

# Re-links an orphan run to PR 7 (head sha ``orphsha``) via the commit->pulls probe.
_ORPHAN_COMMIT_PULLS: list[dict[str, Any]] = [{"number": 7, "head": {"sha": "orphsha"}}]


def _fake_gh(
    *,
    merged: bool = True,
    merged_at: str | None = None,
    comments: Sequence[dict[str, Any]] = (),
    reviews: Sequence[dict[str, Any]] = (),
    commit_pulls: Sequence[dict[str, Any]] | None = None,
) -> Callable[..., Any]:
    """Return a ``gh_api(repo, endpoint, **kw)`` responder keyed on the endpoint.

    ``pulls/<n>`` reports ``merged``/``merged_at`` (an unmerged PR makes
    :func:`derive_outcome_label` yield ``"rejected"``); ``comments`` and
    ``reviews`` serve their lists; ``commit_pulls``, when given, serves the
    ``commits/{sha}/pulls`` probe that re-links an orphan run to a PR.
    A non-empty ``reviews`` list makes :func:`reviewer_logins_signal` non-empty,
    which drives the production :func:`reviewer_set_penalty_prior` DB query
    rather than a monkeypatch.
    """

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if commit_pulls is not None and "/commits/" in endpoint and endpoint.endswith("/pulls"):
            return list(commit_pulls)
        if endpoint.endswith("/comments"):
            return list(comments)
        if endpoint.endswith("/reviews"):
            return list(reviews)
        return {"merged": merged, "merged_at": merged_at}

    return responder


def _fake_gh_merged(merged_at: str) -> Callable[..., Any]:
    """The evidenced-merge shape, kept as a name for ``tests/test_corpus_reproducibility.py``."""
    return _fake_gh(merged_at=merged_at, comments=_REPLIED_FINDING)


def _unused_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """A ``gh_api`` responder the local-branch path must never call."""
    raise AssertionError(f"gh_api should not be called for a local row (endpoint={endpoint})")


def test_build_annotation_pr_row_labels_from_decisive_reply_evidence(tmp_path: Path) -> None:
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    _write_findings(run_dir, _FP_A)
    row = {"session_id": "s1", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=tmp_path,
        gh_api=_fake_gh(
            merged_at="2026-02-05T00:00:00+00:00",
            comments=_finding_comments(_FP_A, reply="applied", reply_created_at="2026-02-01T10:00:00Z"),
        ),
        repo_clone=tmp_path,
    )
    assert ann.labels == ["accepted"]
    # valid_at is the decisive evidence time (the qualifying accept reply), not
    # the later merge timestamp (M12).
    assert ann.valid_at == "2026-02-01T10:00:00Z"
    assert ann.composite_reward == json.loads(ann.reward_json)["composite"]


def _fake_gh_merged_per_finding(merged_at: str, fp_replied: str, fp_unreplied: str) -> Any:
    """A merged PR with two daydream finding comments: one replied, one not.

    Reuses the labeler responder shape but the ``comments`` endpoint carries
    two footer-marked daydream comments whose bodies embed the finding
    markers; a human reply targets only the first.
    """

    return _fake_gh(
        merged_at=merged_at,
        comments=[
            {
                "id": 1,
                "in_reply_to_id": None,
                "user": {"login": "daydream-runner"},
                "body": f"finding\n\n{finding_marker(fp_replied)}\n\n{DAYDREAM_FOOTER}",
            },
            {
                "id": 2,
                "in_reply_to_id": None,
                "user": {"login": "daydream-runner"},
                "body": f"finding\n\n{finding_marker(fp_unreplied)}\n\n{DAYDREAM_FOOTER}",
            },
            {"id": 3, "in_reply_to_id": 1, "user": {"login": "human"},
             "author_association": "MEMBER", "body": "applied"},
        ],
    )


def test_build_annotation_pr_row_carries_per_finding_outcomes(tmp_path: Path) -> None:
    """A merged PR with a findings.json in the archive yields per-finding labels
    joined by fingerprint: the replied finding is accepted, the unreplied one unanswered."""
    fp_a = "a" * 64
    fp_b = "b" * 64
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    (run_dir / "findings.json").write_text(
        json.dumps({"findings": [{"fingerprint": fp_a}, {"fingerprint": fp_b}]})
    )
    row = {"session_id": "s_pf", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                           gh_api=_fake_gh_merged_per_finding("2026-02-01T00:00:00+00:00", fp_a, fp_b),
                           repo_clone=tmp_path)
    assert ann.rubric_json is not None
    assert json.loads(ann.rubric_json)["per_finding_outcomes"] == ["accepted", "unanswered"]


def test_harvest_626_shape_yields_both_polarities(tmp_path: Path) -> None:
    """Three findings on a merged PR: one OWNER 'Fixed in <sha>', one OWNER 'False positive',
    one qualifying question. Run label contested; per-finding exact (M22 final case)."""
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    _write_findings(run_dir, _FP_A, _FP_B, _FP_C)
    row = {"session_id": "s_626", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=tmp_path,
        gh_api=_fake_gh(
            merged_at="2026-02-05T00:00:00+00:00",
            comments=[
                {
                    "id": 1,
                    "in_reply_to_id": None,
                    "user": {"login": "daydream-runner"},
                    "body": f"finding\n\n{finding_marker(_FP_A)}\n\n{DAYDREAM_FOOTER}",
                },
                {
                    "id": 2,
                    "in_reply_to_id": None,
                    "user": {"login": "daydream-runner"},
                    "body": f"finding\n\n{finding_marker(_FP_B)}\n\n{DAYDREAM_FOOTER}",
                },
                {
                    "id": 3,
                    "in_reply_to_id": None,
                    "user": {"login": "daydream-runner"},
                    "body": f"finding\n\n{finding_marker(_FP_C)}\n\n{DAYDREAM_FOOTER}",
                },
                {
                    "id": 4,
                    "in_reply_to_id": 1,
                    "user": {"login": "amelia"},
                    "author_association": "OWNER",
                    "body": "Fixed in abc123",
                    "created_at": "2026-02-01T10:00:00Z",
                },
                {
                    "id": 5,
                    "in_reply_to_id": 2,
                    "user": {"login": "amelia"},
                    "author_association": "OWNER",
                    "body": "False positive",
                    "created_at": "2026-02-01T11:00:00Z",
                },
                {
                    "id": 6,
                    "in_reply_to_id": 3,
                    "user": {"login": "bob"},
                    "author_association": "MEMBER",
                    "body": "Which branch is this against?",
                    "created_at": "2026-02-01T09:00:00Z",
                },
            ],
        ),
        repo_clone=tmp_path,
    )
    assert ann.labels == ["contested"]
    assert ann.rubric_json is not None
    rubric = json.loads(ann.rubric_json)
    assert rubric["per_finding_outcomes"] == ["accepted", "rejected", "ambiguous"]
    # valid_at is the earliest decisive evidence across the joined findings; the
    # ambiguous MEMBER question is not decisive (M22).
    assert ann.valid_at == "2026-02-01T10:00:00Z"


def test_build_annotation_applies_posterior_penalty_for_rejected_pr(tmp_path: Path) -> None:
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

    _write_findings(run_dir, _FP_A)
    payload = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=tmp_path,
        gh_api=_fake_gh(merged=False, comments=_finding_comments(_FP_A, reply="not applicable")),
        repo_clone=tmp_path,
    )

    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    breakdown = json.loads(payload.reward_json)
    assert breakdown["false_positive_penalty"] == 1.0
    assert breakdown["posterior_cost"] == 0.5  # sibling: max(0, 1.0 − 0.5 default prior)
    assert payload.composite_reward == intrinsic_only_composite  # pure intrinsic


def test_build_annotation_rejected_pr_empty_pool_uses_default_prior(tmp_path: Path, archive_dir: Any) -> None:
    # Production wiring of reviewer_set_penalty_prior (not monkeypatched): reviewer
    # "alice" makes the DB query run, but the fresh archive yields the empty-pool
    # path (None, 0), so the reducer applies the 0.5 default prior.
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej_prod", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    _write_findings(run_dir, _FP_A)
    payload = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=archive_dir,
        gh_api=_fake_gh(
            merged=False,
            reviews=[{"user": {"login": "alice"}}],
            comments=_finding_comments(_FP_A, reply="not applicable"),
        ),
        repo_clone=tmp_path,
    )
    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    rb = json.loads(payload.reward_json)
    # Empty pool -> (None, 0) -> default 0.5: posterior_cost == max(0, 1.0 − 0.5).
    assert rb["outcome_prior"] is None
    assert rb["outcome_prior_n"] == 0
    assert rb["posterior_cost"] == 0.5
    # The qualifying reply author joins the reviewer set alongside the review author.
    assert payload.reviewer_logins == ["alice", "amelia"]


def test_build_annotation_fork_pr_author_reply_is_decisive(tmp_path: Path) -> None:
    """M6 wiring (issues #2/#4): a fork-PR author's reply is decisive.

    A fork/contributor author (``author_association`` ``NONE``) whose login
    matches the pull's author must be able to cast the decisive vote through
    the production harvest path — ``pr_author_logins`` is now threaded into
    ``per_finding_resolution_signal`` via ``build_annotation``. Without the
    wiring the reply is ``excluded:non-qualifying``, the finding stays
    ``unanswered``, and the run would resolve to ``unknown`` instead.
    """
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    _write_findings(run_dir, _FP_A)
    row = {"session_id": "s_fork_auth", "pr_repo": "o/r", "pr_number": 13, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    reply_created = "2026-08-02T10:00:00Z"
    comments = [
        {
            "id": 1,
            "in_reply_to_id": None,
            "user": {"login": "daydream-runner"},
            "body": f"finding\n\n{finding_marker(_FP_A)}\n\n{DAYDREAM_FOOTER}",
        },
        {
            "id": 2,
            "in_reply_to_id": 1,
            "user": {"login": "prfiona", "type": "User"},
            "author_association": "NONE",  # fork contributor
            "body": "Fixed in abc123",      # decisive accept
            "created_at": reply_created,
        },
    ]

    def fork_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return comments
        if endpoint.endswith("/reviews"):
            return []
        # The pull payload carries the PR author == the fork reply author.
        return {
            "merged": True,
            "merged_at": "2026-08-03T00:00:00Z",
            "state": "merged",
            "user": {"login": "prfiona"},
        }

    payload = build_annotation(
        row, run_dir=run_dir, archive_dir=tmp_path, gh_api=fork_gh, repo_clone=tmp_path,
    )
    assert payload.labels == ["accepted"]
    # The PR-author reply is the decisive evidence, so it sets valid_at (M12).
    assert payload.valid_at == reply_created
    rubric = json.loads(payload.rubric_json or "{}")
    assert rubric.get("per_finding_outcomes") == ["accepted"]


def test_build_annotation_formal_review_author_reply_is_decisive(tmp_path: Path) -> None:
    """M6 wiring (issues #2/#4): a formal-review author's reply is decisive.

    A reviewer author (not OWNER/MEMBER/COLLABORATOR) whose login comes back
    from the ``/reviews`` lookup counts under the M6 gate through the same
    ``review_author_logins`` plumbing in ``build_annotation``.
    """
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    _write_findings(run_dir, _FP_A)
    row = {"session_id": "s_review_auth", "pr_repo": "o/r", "pr_number": 14, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    comments = [
        {
            "id": 1,
            "in_reply_to_id": None,
            "user": {"login": "daydream-runner"},
            "body": f"finding\n\n{finding_marker(_FP_A)}\n\n{DAYDREAM_FOOTER}",
        },
        {
            "id": 2,
            "in_reply_to_id": 1,
            "user": {"login": "revbob", "type": "User"},
            "author_association": "NONE",  # reviewed the PR but not a maintainer
            "body": "False positive, the linter is wrong here",
            "created_at": "2026-08-02T09:00:00Z",
        },
    ]

    def review_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return comments
        if endpoint.endswith("/reviews"):
            return [{"user": {"login": "revbob"}}]
        return {"merged": True, "merged_at": "2026-08-03T00:00:00Z", "state": "merged",
                "user": {"login": "someone-else"}}

    payload = build_annotation(
        row, run_dir=run_dir, archive_dir=tmp_path, gh_api=review_gh, repo_clone=tmp_path,
    )
    assert payload.labels == ["rejected"]
    rubric = json.loads(payload.rubric_json or "{}")
    assert rubric.get("per_finding_outcomes") == ["rejected"]


def test_build_annotation_shallow_local_row_null_valid_at_reward_present(tmp_path: Path) -> None:
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


def test_build_annotation_rejected_pr_populated_prior_drives_pool(tmp_path: Path, archive_dir: Any) -> None:
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
    _write_findings(run_dir, _FP_A)
    payload = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=archive_dir,
        gh_api=_fake_gh(
            merged=False,
            reviews=[{"user": {"login": "alice"}}],
            comments=_finding_comments(_FP_A, reply="not applicable"),
        ),
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


def test_build_annotation_pr_uses_pooled_prior_and_persists_reviewers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harvest, "reviewer_set_penalty_prior", lambda *a, **k: (0.8, 12))  # n>=10 -> empirical
    monkeypatch.setattr(harvest, "reviewer_logins_signal", lambda *a, **k: ["alice", "carol"])
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    _write_findings(run_dir, _FP_A)
    p = build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh(merged=False, comments=_finding_comments(_FP_A, reply="not applicable")),
                         repo_clone=tmp_path)
    rb = json.loads(p.reward_json)
    assert rb["posterior_cost"] == pytest.approx(0.2)   # max(0, 1.0 - 0.8)
    assert rb["outcome_prior"] == 0.8 and rb["outcome_prior_n"] == 12
    assert p.composite_reward == rb["composite"]        # stored composite is pure intrinsic (C5)
    assert p.has_posterior is True and p.reviewer_logins == ["alice", "carol"]


def test_build_annotation_below_threshold_falls_back_to_default_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harvest, "reviewer_set_penalty_prior", lambda *a, **k: (0.9, 4))  # n<10
    monkeypatch.setattr(harvest, "reviewer_logins_signal", lambda *a, **k: ["alice"])
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_rej", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    _write_findings(run_dir, _FP_A)
    rb = json.loads(
        build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh(merged=False, comments=_finding_comments(_FP_A, reply="not applicable")),
                         repo_clone=tmp_path).reward_json
    )
    assert rb["outcome_prior"] is None and rb["outcome_prior_n"] == 4  # n recorded; prior None -> 0.5
    assert rb["posterior_cost"] == 0.5


def test_build_annotation_local_row_has_no_reviewer_prior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PR-less row -> reviewer_logins == [], prior query never consulted, and the
    # local verdict is withheld from the posterior axis: a local commit is not a
    # maintainer acting in a PR, so the label is kept but has_posterior is False.
    from daydream.training.labeler_signals import LocalCommitAppliedSignal

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
    assert p.labels == ["rejected"]  # label kept — consumers may still want it
    assert p.has_posterior is False  # ...but it is not maintainer-PR evidence
    rb = json.loads(p.reward_json)
    assert "posterior_cost" not in rb and "outcome_prior" not in rb
    assert rb["composite"] is not None  # the intrinsic axes are unaffected


def test_build_annotation_asserts_canonical_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-canonical reward_version -> the write must be refused.
    def _custom_version_score(*args: Any, **kwargs: Any) -> Any:
        from daydream.training.reward import RewardWeights
        return score_trajectory(*args, **{**kwargs, "weights": RewardWeights(w_correctness=0.99)})

    monkeypatch.setattr(harvest, "score_trajectory", _custom_version_score)
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    row = {"session_id": "s_custom", "pr_repo": "o/r", "pr_number": 9, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir), "grounding_rate": 1.0,
           "changed_files": "[]"}
    with pytest.raises((AssertionError, RuntimeError), match="canonical"):
        build_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh(merged=False), repo_clone=tmp_path)


def test_assemble_reads_verdicts_and_grounding_from_bronze(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        '{"verdicts":[{"issue_id":1,"verdict":"consistent"}]}'
    )
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 0.75})
    assert inputs.verifier_verdicts == [{"issue_id": 1, "verdict": "consistent"}]
    assert inputs.grounding_rate == 0.75 and inputs.format_valid is True


def test_assemble_shallow_run_has_null_verdicts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": None})
    assert inputs.verifier_verdicts is None


def test_assemble_declined_deep_run_null_verdicts_keeps_format_valid(tmp_path: Path) -> None:
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


def test_assemble_malformed_verdicts_flags_format_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text("{not json")
    inputs = assemble_scoring_inputs(run_dir, {"grounding_rate": 1.0})
    assert inputs.format_valid is False


def test_score_trajectory_grounding_axis_present_on_default_run(tmp_path: Path) -> None:
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
    archive_dir: Path,
    session_id: str,
    *,
    merged_at: str,
    source_path: Path | None = None,
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
    archive_dir: Path,
    bronze_parent: Path,
    *,
    session_id: str,
    head_sha: str = "orphsha",
    branch: str = "feat/x",
    source_path: Path | None = None,
) -> Path:
    """Seed an orphan deep run (no PR linkage) and index it under ``archive_dir``.

    Mirrors :func:`_seed_archived_deep_run` but writes the orphan manifest shape
    the re-link path consumes: ``pr_number``/``pr_repo`` are ``None`` and the row
    carries only a ``branch``/``head_sha``. Bronze artifacts go under
    ``bronze_parent``; returns the run directory.

    ``source_path`` records a working tree on the manifest so
    :func:`_resolve_repo_for_row` resolves a clone for the row
    (``clone_resolved`` True), enabling the local-commit walk.
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
            branch=branch,
            head_sha=head_sha,
            base_branch="main",
            pr_number=None,
            pr_repo=None,
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(run_dir),
            source_path=str(source_path) if source_path else None,
        ),
    )
    return run_dir


def _seed_pr_runs(
    archive_dir: Path,
    bronze_parent: Path,
    count: int,
    *,
    fingerprints: Sequence[str] | None = None,
) -> None:
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
        if fingerprints:
            _write_findings(run_dir, *fingerprints)


async def test_harvest_writes_one_annotation(tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING),
    )
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    obs = latest_label_observation(archive_dir, "s1")
    assert obs is not None
    assert summary["annotated"] == 1
    assert obs["valid_at"] == "2026-02-01T00:00:00+00:00" and obs["composite_reward"] is not None


async def test_harvest_stores_github_z_merge_timestamp_canonically(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path writer convergence: GitHub reports merged_at with a 'Z' suffix;
    the stored valid_at must be the canonical '+00:00' spelling."""
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00Z")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(merged_at="2026-02-01T00:00:00Z", comments=_REPLIED_FINDING),
    )
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    obs = latest_label_observation(archive_dir, "s1")
    assert obs is not None
    assert summary["annotated"] == 1
    assert obs["valid_at"] == "2026-02-01T00:00:00+00:00"


async def test_harvest_unresolved_daydream_comment_stays_unknown(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: a merged PR whose only finding has no qualifying reply is ``unknown``.

    An unresolved daydream comment is *non-decisive* evidence (M9): nobody said
    accept or reject, so the conservative rubric persists ``unknown`` (empty
    labels) — bare reply absence and merge state never fabricate a label.
    """
    run_dir = _seed_archived_deep_run(archive_dir, "s-contest", merged_at="2026-02-01T00:00:00+00:00")
    _write_findings(run_dir, _FP_A)
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_finding_comments(_FP_A)),
    )
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    row = query_runs(archive_dir, "session_id = ?", ("s-contest",))[0]
    assert json.loads(row["outcome_labels"]) == []  # unknown, never "accepted"


async def test_harvest_relinks_orphan_run_and_labels_it(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: an orphan run (PR opened after launch) is re-linked at harvest.

    Bug 2. The default deep loop archives before the PR exists, freezing
    ``pr_number=None``. Harvest must re-link the orphan row by its ``head_sha``
    (via ``commits/{sha}/pulls``), persist the linkage, and then label the run
    through the PR path. Drives ``run_harvest`` end-to-end and asserts both the
    persisted linkage and the resulting ``contested`` label (decisive accept
    beside an unresolved finding — mixed evidence, M9).
    """
    run_dir = _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph")
    _write_findings(run_dir, _FP_A, _FP_B)
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(
            merged_at="2026-02-01T00:00:00+00:00",
            comments=[
                *_finding_comments(_FP_A, reply="applied"),
                {
                    "id": 3,
                    "in_reply_to_id": None,
                    "user": {"login": "daydream-runner"},
                    "body": f"finding\n\n{finding_marker(_FP_B)}\n\n{DAYDREAM_FOOTER}",
                },
            ],
            commit_pulls=_ORPHAN_COMMIT_PULLS,
        ),
    )
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))
    row = query_runs(archive_dir, "session_id = ?", ("s-orph",))[0]
    assert row["pr_number"] == 7 and row["pr_repo"] == "org/repo"  # linkage persisted
    assert json.loads(row["outcome_labels"]) == ["contested"]  # now labelable (was orphan)


@pytest.mark.parametrize("with_clone", [False, True])
async def test_harvest_fork_pr_404_degrades_not_drops(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
    with_clone: Any,
) -> None:
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
    assert obs is not None
    # pr_state is None on the local-branch rubric (not "merged"/"closed"):
    # observable proof the fix degraded to the local path rather than
    # fabricating a merged=False PR rubric.
    assert obs["pr_state"] is None
    # The central #166 contract: NOT mislabeled "rejected". With no resolvable
    # clone the local posterior is "unknown" → empty outcome_labels.
    row = query_runs(archive_dir, "session_id = ?", ("s-fork",))[0]
    assert json.loads(row["outcome_labels"]) == []


async def test_harvest_orphan_422_degrades_not_drops(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0  # benign 422 degraded; not a hard error
    assert latest_label_observation(archive_dir, "s-orph-422") is not None  # annotated via local path
    row = query_runs(archive_dir, "session_id = ?", ("s-orph-422",))[0]
    assert row["pr_number"] is None and row["pr_repo"] is None  # linkage NOT applied
    # No resolvable clone → local posterior is "unknown", never "rejected".
    assert json.loads(row["outcome_labels"]) == []


def _gh_unpushed_422(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """gh stub for a squash-merged head SHA: the link probe 422s, nothing else."""
    if "/commits/" in endpoint and endpoint.endswith("/pulls"):
        raise GitError("gh: No commit found for SHA (HTTP 422)")
    if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
        return []
    return {}


async def test_harvest_deleted_branch_ref_labels_unknown_not_rejected(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: a squash-merged run whose branch ref is gone must NOT be "rejected".

    This is the shape that mislabeled 286 archived runs. After a squash merge
    GitHub deletes the branch and rewrites the commit, so at harvest time:
      * ``commits/<sha>/pulls`` 422s (the recorded SHA was never on the remote), and
      * ``git log <sha>..<branch>`` exits 128 (the branch ref no longer resolves).

    Crucially a working tree IS resolved here (``clone_resolved`` True), so the
    existing no-clone guard does not apply — the local-commit walk runs for real
    against a real repo and really fails. ``log_shas`` used to swallow that into
    ``[]``, which ``local_commit_applied_signal`` read as "no follow-up commit"
    and labeled ``"rejected"``: a false negative fed straight into the corpus.

    Drives ``run_harvest`` end-to-end and asserts the observable label.
    """
    clone = _make_repo_with_main(tmp_path, name="clone")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    # The branch the run recorded was squash-merged and deleted — never created here.
    _seed_orphan_run(
        archive_dir,
        tmp_path / "bronze",
        session_id="s-gone",
        head_sha=head_sha,
        branch="feat/squash-merged-and-deleted",
        source_path=clone,
    )
    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0  # benign 422 + unreadable window degrade, not error
    assert latest_label_observation(archive_dir, "s-gone") is not None  # still annotated
    row = query_runs(archive_dir, "session_id = ?", ("s-gone",))[0]
    # The central contract: unreadable commit window → "unknown" (empty), NOT "rejected".
    # The recommended change is absent from main here, so the base-branch
    # fallback cannot upgrade it — see the squash-merge test below for that arm.
    assert json.loads(row["outcome_labels"]) == []


async def test_harvest_squash_merged_branch_recovers_accepted_from_base_branch(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: a squash-merged run is recovered as "accepted", not lost to "unknown".

    The full production shape of the 286-row mislabel. The branch was squash-
    merged and deleted, so ``git log <sha>..<branch>`` exits 128 and the
    per-commit walk is impossible — but the recommended change IS on ``main``,
    carried there by the squash commit. The posterior's real question ("did the
    change land?") is therefore still answerable, and the run is a genuine
    positive that must not be discarded.

    Uses real git throughout: the branch is created, committed, merged with
    ``--squash``, then deleted, exactly as GitHub leaves the repo.
    """
    clone = _make_repo_with_main(tmp_path, name="clone")
    head_sha_holder: dict[str, str] = {}
    _git(clone, "checkout", "-b", "feat/squash-me")
    (clone / "app.py").write_text("existing\nguarded = True\n")
    _git(clone, "add", "app.py")
    _commit(clone, "add the guard")
    head_sha_holder["sha"] = _git(clone, "rev-parse", "HEAD").strip()
    # Squash-merge into main and delete the branch — the GitHub end state.
    _git(clone, "checkout", "main")
    _git(clone, "merge", "--squash", "feat/squash-me")
    _commit(clone, "add the guard (#220)")
    _git(clone, "branch", "-D", "feat/squash-me")

    run_dir = _seed_orphan_run(
        archive_dir,
        tmp_path / "bronze",
        session_id="s-squash",
        head_sha=head_sha_holder["sha"],
        branch="feat/squash-me",
        source_path=clone,
    )
    (run_dir / "recommended.patch").write_text(
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        "+guarded = True\n"
    )
    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0
    row = query_runs(archive_dir, "session_id = ?", ("s-squash",))[0]
    # Recovered as a real positive — not "rejected" (the bug) and not "unknown"
    # (giving up on evidence that is plainly there on the base branch).
    assert json.loads(row["outcome_labels"]) == ["accepted"]


async def test_harvest_live_branch_with_no_followup_commits_still_labels_rejected(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path counterpart: a readable window with no follow-up IS "rejected".

    Same code path as the deleted-ref test, differing only in that the recorded
    branch still exists and is readable. This pins the distinction the fix
    introduces — "could not look" is now ``unknown``, but "looked and found
    nothing applied" must remain a genuine negative label, or the fix would have
    silently erased every true rejection from the corpus.
    """
    clone = _make_repo_with_main(tmp_path, name="clone")
    _git(clone, "checkout", "-b", "feat/still-here")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    _seed_orphan_run(
        archive_dir,
        tmp_path / "bronze",
        session_id="s-live",
        head_sha=head_sha,
        branch="feat/still-here",
        source_path=clone,
    )
    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0
    row = query_runs(archive_dir, "session_id = ?", ("s-live",))[0]
    # Branch resolves and carries no follow-up commit → a real negative, preserved.
    assert json.loads(row["outcome_labels"]) == ["rejected"]


async def test_harvest_live_branch_with_applied_fix_labels_accepted(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path counterpart: a follow-up commit carrying the fix IS "accepted".

    Completes the three-way distinction (unknown / rejected / accepted) through
    the same real-git path, so the fix is pinned on the positive arm too.
    """
    clone = _make_repo_with_main(tmp_path, name="clone")
    _git(clone, "checkout", "-b", "feat/fixed")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    run_dir = _seed_orphan_run(
        archive_dir,
        tmp_path / "bronze",
        session_id="s-applied",
        head_sha=head_sha,
        branch="feat/fixed",
        source_path=clone,
    )
    # The recommended patch adds a line; a later commit on the branch lands it.
    (run_dir / "recommended.patch").write_text(
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        "+guarded = True\n"
    )
    (clone / "app.py").write_text("existing\nguarded = True\n")
    _git(clone, "add", "app.py")
    _commit(clone, "apply the recommended fix")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)

    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    row = query_runs(archive_dir, "session_id = ?", ("s-applied",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]


async def test_harvest_merged_pr_with_zero_comments_is_not_labeled_accepted(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: a merged PR where daydream tracked NO comments is unlabeled.

    The vacuous-accept bug. ``comment_resolution == (0, 0, 0)`` made
    ``unresolved == 0`` trivially true, so "merged, daydream contributed
    nothing observable" scored identically to "merged, every finding
    addressed" — 152 of 170 ``pr_review`` accepts in the real archive were this
    shape. Such a run carries no evidence either way, so it must persist as
    ``unknown`` (empty labels) and stay out of the posterior population.
    """
    _seed_archived_deep_run(archive_dir, "s-vacuous", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(merged_at="2026-02-01T00:00:00+00:00"),
    )

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0 and summary["annotated"] == 1
    row = query_runs(archive_dir, "session_id = ?", ("s-vacuous",))[0]
    assert json.loads(row["outcome_labels"]) == []  # NOT ["accepted"]
    obs = latest_label_observation(archive_dir, "s-vacuous")
    assert obs is not None
    assert obs["pr_state"] == "merged"  # merge state still recorded, just not decisive
    assert json.loads(obs["rubric_json"])["comment_resolution"]["total"] == 0
    # "unknown" maps to no penalty, so the row is excluded from the posterior population.
    assert obs["has_posterior"] == 0
    assert "posterior_cost" not in json.loads(obs["reward_json"])


async def test_harvest_merged_pr_with_reject_reply_is_contested(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The amelia#626 shape: merged PR + explicit human rejection ⇒ contested, never accepted (M10/M22).

    One daydream finding is explicitly rejected by a qualifying OWNER reply
    ("False positive…") while a second finding goes unanswered. The decisive
    reject beside non-decisive evidence aggregates to ``contested`` — a merge
    can never upgrade an explicit rejection to ``accepted``.
    """
    run_dir = _seed_archived_deep_run(archive_dir, "s-reject", merged_at="2026-08-10T00:00:00Z")
    _write_findings(run_dir, _FP_A, _FP_B)
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(
            merged_at="2026-08-10T00:00:00Z",
            comments=[
                *_finding_comments(_FP_A, reply="False positive — the code already handles this"),
                {
                    "id": 3,
                    "in_reply_to_id": None,
                    "user": {"login": "daydream-runner"},
                    "body": f"finding\n\n{finding_marker(_FP_B)}\n\n{DAYDREAM_FOOTER}",
                },
            ],
        ),
    )

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0 and summary["annotated"] == 1
    row = query_runs(archive_dir, "session_id = ?", ("s-reject",))[0]
    assert json.loads(row["outcome_labels"]) == ["contested"]  # NEVER ["accepted"]
    obs = latest_label_observation(archive_dir, "s-reject")
    assert obs is not None
    assert json.loads(obs["rubric_json"])["per_finding_outcomes"] == ["rejected", "unanswered"]


async def test_harvest_unmerged_pr_with_no_semantic_reply_is_unknown(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open PR, bot-only reply ⇒ unknown; pr_state == 'open' preserved (M11/M22).

    The only reply to the finding is from a bot, which never qualifies as human
    judgment, so the finding is ``unanswered`` and the run stays ``unknown``.
    The PR's open state is preserved as context, not read as a rejection.
    """
    run_dir = _seed_archived_deep_run(archive_dir, "s-open", merged_at="2026-08-10T00:00:00Z")
    _write_findings(run_dir, _FP_A)

    def _gh_open(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return [
                *_finding_comments(_FP_A),
                {
                    "id": 2,
                    "in_reply_to_id": 1,
                    "user": {"login": "dependabot[bot]", "type": "Bot"},
                    "body": "applied",
                },
            ]
        if endpoint.endswith("/reviews"):
            return []
        return {"merged": False, "merged_at": None, "state": "open"}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_open)

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0 and summary["annotated"] == 1
    row = query_runs(archive_dir, "session_id = ?", ("s-open",))[0]
    assert json.loads(row["outcome_labels"]) == []  # unknown, NOT ["rejected"]
    obs = latest_label_observation(archive_dir, "s-open")
    assert obs is not None
    assert obs["pr_state"] == "open"


def test_valid_at_is_decisive_evidence_time(tmp_path: Path) -> None:
    """valid_at = qualifying reply timestamp, not merged_at (M12/M22)."""
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    _write_findings(run_dir, _FP_A)
    row = {"session_id": "s-val", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=tmp_path,
        gh_api=_fake_gh(
            merged_at="2026-08-10T00:00:00Z",
            comments=_finding_comments(_FP_A, reply="applied", reply_created_at="2026-08-02T10:00:00Z"),
        ),
        repo_clone=tmp_path,
    )
    assert ann.valid_at == "2026-08-02T10:00:00Z"


def test_valid_at_override_respected(tmp_path: Path) -> None:
    """An explicit override beats derived evidence time (M12)."""
    run_dir = _seed_deep_bronze(tmp_path, verdict="consistent", grounding=1.0)
    _write_findings(run_dir, _FP_A)
    row = {"session_id": "s-val-ovr", "pr_repo": "o/r", "pr_number": 7, "head_sha": "h",
           "base_branch": "main", "archive_path": str(run_dir),
           "grounding_rate": 1.0, "changed_files": "[]"}
    ann = build_annotation(
        row,
        run_dir=run_dir,
        archive_dir=tmp_path,
        gh_api=_fake_gh(
            merged_at="2026-08-10T00:00:00Z",
            comments=_finding_comments(_FP_A, reply="applied", reply_created_at="2026-08-02T10:00:00Z"),
        ),
        repo_clone=tmp_path,
        valid_at_override="2026-09-01T00:00:00Z",
    )
    assert ann.valid_at == "2026-09-01T00:00:00Z"


async def test_labeler_version_is_not_reward_version(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """append_label_observation receives labeler_versions.LABELER_POLICY_VERSION (M13/M22)."""
    run_dir = _seed_archived_deep_run(archive_dir, "s-lv", merged_at="2026-08-10T00:00:00Z")
    _write_findings(run_dir, _FP_A)
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(
            merged_at="2026-08-10T00:00:00Z",
            comments=_finding_comments(_FP_A, reply="applied", reply_created_at="2026-08-02T10:00:00Z"),
        ),
    )
    captured: dict[str, Any] = {}

    def _capture(_archive_dir: Path, _session_id: str, **kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr("daydream.training.harvest.append_label_observation", _capture)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["annotated"] == 1
    assert captured["labeler_version"] == labeler_versions.LABELER_POLICY_VERSION
    assert captured["labeler_version"] != reward.REWARD_VERSION
    assert captured["reply_classifier_version"] == labeler_versions.REPLY_CLASSIFIER_VERSION
    assert captured["reply_evidence_digest"]


def test_resume_cache_invalidated_on_policy_bump(tmp_path: Path) -> None:
    """A policy-version change re-fetches sessions previously marked done (M15/M22)."""
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    cache.mark_session_done("sess-1")
    with patch.object(labeler_versions, "LABELER_POLICY_VERSION", "980-r2"):
        assert "sess-1" not in cache.completed_sessions()
    # And without the bump the session is still resumable:
    assert "sess-1" in cache.completed_sessions()


async def test_harvest_local_branch_accept_keeps_label_but_is_not_posterior_evidence(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: a local-commit "accepted" is a weaker evidence tier.

    A commit on the recorded branch containing the recommended lines is not a
    maintainer acting in a PR, so it must not enter the reward-model posterior
    population. The label is still persisted (consumers may want it), but
    ``has_posterior`` is 0 and no ``posterior_cost`` is written — letting
    consumers split on the column instead of silently mixing evidence tiers
    with real merged-PR outcomes.
    """
    clone = _make_repo_with_main(tmp_path, name="clone")
    _git(clone, "checkout", "-b", "feat/local-tier")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    run_dir = _seed_orphan_run(
        archive_dir,
        tmp_path / "bronze",
        session_id="s-local-tier",
        head_sha=head_sha,
        branch="feat/local-tier",
        source_path=clone,
    )
    (run_dir / "recommended.patch").write_text(
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        "+guarded = True\n"
    )
    (clone / "app.py").write_text("existing\nguarded = True\n")
    _git(clone, "add", "app.py")
    _commit(clone, "apply the recommended fix")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_unpushed_422)

    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0
    row = query_runs(archive_dir, "session_id = ?", ("s-local-tier",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]  # label kept
    obs = latest_label_observation(archive_dir, "s-local-tier")
    assert obs is not None
    assert json.loads(obs["rubric_json"])["posterior_source"] == "local_branch"
    # Weaker tier: labeled, but withheld from the posterior population.
    assert obs["has_posterior"] == 0
    assert "posterior_cost" not in json.loads(obs["reward_json"])


async def test_harvest_dry_run_mutates_row_in_memory_but_suppresses_set_run_pr_link(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry_run=True: in-memory linkage preview is applied but not persisted.

    When an orphan run re-links to a PR, ``row['pr_number']`` and
    ``row['pr_repo']`` are mutated unconditionally
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
        _fake_gh(
            merged_at="2026-02-01T00:00:00+00:00",
            comments=_UNRESOLVED_FINDING,
            commit_pulls=_ORPHAN_COMMIT_PULLS,
        ),
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


async def test_harvest_leaves_true_local_run_unlinked(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


async def test_re_harvest_is_idempotent(tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING),
    )
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1"))
    second = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2"))
    assert len(label_observation_history(archive_dir, "s1")) == 1  # deduped
    assert second["skipped"] == 1 and second["annotated"] == 0


async def test_re_harvest_appends_on_version_bump(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr(
        "daydream.training.harvest._gh_api",
        _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING),
    )
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1"))
    monkeypatch.setattr("daydream.training.labeler_versions.LABELER_POLICY_VERSION", "980-policy-bump")
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2"))
    assert len(label_observation_history(archive_dir, "s1")) == 2


async def test_harvest_aborts_cleanly_on_rate_limit_and_preserves_resume(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two PR rows; PR 1 succeeds, PR 2 hits an exhausted rate-limit on every gh
    # call so the harvest loop must abort cleanly.
    _seed_pr_runs(archive_dir, tmp_path, 2)
    merged = _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING)

    def _gh(repo: Any, endpoint: str, **kw: Any) -> Any:
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


def test_gh_api_backoff_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    slept = []
    monkeypatch.setattr(harvest, "_rate_limit_sleep", lambda s: slept.append(s))
    seq = [git_ops.RateLimitError("x"), git_ops.RateLimitError("x"), {"ok": True}]

    def _inner(*a: Any, **k: Any) -> Any:
        v = seq.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr("daydream.git_ops.gh_api", _inner)
    assert harvest._gh_api("o/r", "endpoint") == {"ok": True}
    assert len(slept) == 2


# _resolve_repo_for_row


def test_resolve_repo_for_row_prefers_source_path(tmp_path: Path) -> None:
    """source_path is preferred when it exists and contains .git."""
    source = tmp_path / "source_repo"
    source.mkdir()
    (source / ".git").mkdir()
    row = {"source_path": str(source), "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=tmp_path / "cache")
    assert result == source


def test_resolve_repo_for_row_clones_when_source_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls through to clone when source_path is absent."""
    cache = tmp_path / "cache"

    def fake_clone(url: str, target: Path, token: str | None = None, **kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / ".git").mkdir()

    monkeypatch.setattr("daydream.training.harvest.git_ops.clone_with_token", fake_clone)
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result == cache / "org" / "repo"


def test_resolve_repo_for_row_fetches_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the cache clone already exists, fetch instead of clone."""
    cache = tmp_path / "cache"
    cached_repo = cache / "org" / "repo"
    cached_repo.mkdir(parents=True)
    (cached_repo / ".git").mkdir()

    fetched = []
    monkeypatch.setattr("daydream.training.harvest.git_ops.fetch", lambda repo, remote="origin": fetched.append(repo))
    monkeypatch.setattr(
        "daydream.training.harvest.git_ops.clone_with_token",
        lambda *a, **k: pytest.fail("should not clone"),
    )
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result == cached_repo
    assert fetched == [cached_repo]


def test_resolve_repo_for_row_returns_none_when_no_remote(tmp_path: Path) -> None:
    """Returns None when neither source_path nor remote_url is available."""
    row = {"source_path": None, "remote_url": None, "repo_slug": None}
    result = _resolve_repo_for_row(row, clone_cache=tmp_path / "cache")
    assert result is None


def test_resolve_repo_for_row_clone_failure_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clone failure is swallowed and None is returned (no .git left on disk)."""
    cache = tmp_path / "cache"

    monkeypatch.setattr(
        "daydream.training.harvest.git_ops.clone_with_token",
        lambda url, target, token=None, **kwargs: (_ for _ in ()).throw(GitError("network error")),
    )
    row = {"source_path": None, "remote_url": "https://github.com/org/repo.git", "repo_slug": "org/repo"}
    result = _resolve_repo_for_row(row, clone_cache=cache)
    assert result is None


def test_resolve_repo_for_row_fetch_failure_returns_cached_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_pr_coverage_helper_counts_decisive(tmp_path: Path) -> None:
    for i, label in [(1, "accepted"), (2, "rejected"), (3, "unknown")]:
        upsert_run(tmp_path, make_manifest(session_id=f"p{i}", pr_number=i, pr_repo="o/r"))
        append_label_observation(
            tmp_path,
            f"p{i}",
            labels=[label],
            pr_state=None,
            labeler_version="auto",
            evidence_sha=f"s{i}",
            source="auto",
        )
    upsert_run(tmp_path, make_manifest(session_id="local1"))  # no pr_number — excluded
    cov = pr_attached_label_coverage(tmp_path)
    assert cov["pr_attached"] == 3 and cov["decisive"] == 2  # accepted+rejected, not unknown


async def test_harvest_propagates_transient_giterror_for_retry(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


async def test_harvest_keeps_labeled_row_when_reviewer_lookup_errors(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path: a GitError on the ``/reviews`` lookup must not drop a labeled row.

    ``/pulls/{n}`` resolves the PR outcome (merged → ``accepted``) and
    ``/comments`` succeeds, but the auxiliary ``/reviews`` lookup feeding the
    reviewer-set prior raises ``GitError``. Because the PR outcome is already
    decided, the row must still be annotated with its resolved label (degrading
    only the optional prior), not dropped as a hard error through the per-row
    catch-all. Drives ``run_harvest`` end-to-end.
    """
    run_dir = _seed_archived_deep_run(archive_dir, "s-reviews-err", merged_at="2026-02-01T00:00:00+00:00")
    _write_findings(run_dir, _FP_A)

    def _gh_reviews_fail(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/reviews"):
            raise GitError("gh: Internal Server Error (HTTP 500)")
        if endpoint.endswith("/comments"):
            # A resolved daydream thread: the evidence that makes the merge decisive.
            return _finding_comments(_FP_A, reply="applied")
        return {"merged": True, "merged_at": "2026-02-01T00:00:00+00:00"}

    monkeypatch.setattr("daydream.training.harvest._gh_api", _gh_reviews_fail)
    summary = await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c"))

    assert summary["errors"] == 0  # reviewer-lookup failure degraded, row not dropped
    obs = latest_label_observation(archive_dir, "s-reviews-err")
    assert obs is not None
    assert obs["pr_state"] == "merged"  # resolved PR outcome preserved (not degraded to local)
    row = query_runs(archive_dir, "session_id = ?", ("s-reviews-err",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]  # merged-PR label kept


async def test_harvest_degrades_benign_giterror_rows_instead_of_dropping(
    tmp_path: Path,
    archive_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 10 PR rows: PRs 1..8 merged ("accepted"); PRs 9..10 raise a benign GitError
    # (e.g. fork-PR 404). The fix DEGRADES 9..10 to the local-branch posterior
    # instead of dropping them: errors == 0, annotated == 10. Degraded rows route
    # through _build_rubric_local, so their pr_state is None (NOT a fabricated PR
    # rubric) — observable proof no merged=False rubric drove their label.
    _seed_pr_runs(archive_dir, tmp_path, 10, fingerprints=(_FP_A,))

    merged = _fake_gh(
        merged_at="2026-02-01T00:00:00+00:00",
        comments=_finding_comments(_FP_A, reply="applied"),
    )

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
    obs_pr = latest_label_observation(archive_dir, "s1")
    assert obs_pr is not None
    accepted_pr_state = obs_pr["pr_state"]
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


def _raise_git_error_with_url(*args: object, **kwargs: object) -> None:
    raise GitError("git clone https://user:ghp_canaryfake123@github.com/o/r failed: boom")


def test_repo_resolution_warning_is_value_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The harvest repo-resolution warning must never contain the remote URL (issue #981)."""
    row = {
        "source_path": None,
        "remote_url": "https://user:ghp_canaryfake123@github.com/o/r",
        "repo_slug": "o/r",
    }
    monkeypatch.setattr("daydream.training.harvest.git_ops.clone_with_token", _raise_git_error_with_url)
    _resolve_repo_for_row(row, clone_cache=tmp_path / "cache")
    out = capsys.readouterr().out
    assert "ghp_canaryfake123" not in out
    assert "o/r" in out  # slug IS present


# identity-based repo resolution (issue #981): never clone the archived raw URL


def test_resolve_repo_never_clones_raw_archived_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_clone(url: str, target: Path, token: str | None = None, **kw: object) -> None:
        seen.append(url)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".git").mkdir()

    monkeypatch.setattr("daydream.training.harvest.git_ops.clone_with_token", fake_clone)
    row = {
        "source_path": None,
        "remote_url": "https://user:ghp_canaryfake123@github.com/o/r",
        "repo_slug": "o/r",
    }
    assert _resolve_repo_for_row(row, clone_cache=tmp_path / "cache") == tmp_path / "cache" / "o" / "r"
    assert seen == ["https://github.com/o/r"]  # reconstructed identity, never raw


def test_resolve_repo_fails_closed_on_untrusted_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "daydream.training.harvest.git_ops.clone_with_token",
        lambda url, t, **kw: calls.append(url),
    )
    row = {"source_path": None, "remote_url": "https://evil.example.com/o/r", "repo_slug": "o/r"}
    assert _resolve_repo_for_row(row, clone_cache=tmp_path / "cache") is None  # M7: no clone attempt
    assert calls == []


def test_resolve_repo_fails_closed_on_file_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "daydream.training.harvest.git_ops.clone_with_token",
        lambda url, t, **kw: calls.append(url),
    )
    row = {"source_path": None, "remote_url": "file:///tmp/evil", "repo_slug": "o/r"}
    assert _resolve_repo_for_row(row, clone_cache=tmp_path / "cache") is None
    assert calls == []


def test_token_never_in_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAYDREAM_GIT_TOKEN", "ghp_envtokfake123")
    seen: list[list[str]] = []

    def _recording_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", _recording_run)
    _resolve_repo_for_row(
        {"source_path": None, "remote_url": "https://github.com/o/r", "repo_slug": "o/r"},
        clone_cache=tmp_path / "cache",
    )
    assert seen
    # The base64 Authorization header is trivially recoverable, so the token's
    # absence on argv must hold for both the raw token and its encoded form.
    basic = base64.b64encode(b"x-access-token:ghp_envtokfake123").decode()
    for c in seen:
        joined = " ".join(c)
        assert "ghp_envtokfake123" not in joined  # raw token never on argv
        assert basic not in joined  # recoverable base64 never on argv either (M8)


def test_hub_import_rejects_unsanitized_affected_bundle(tmp_path: Path) -> None:
    """M18: an affected (credential-bearing) incoming bundle with no released
    derivative is quarantined locally and skipped — never imported raw."""
    from daydream.archive import sanitize

    archive_dir = tmp_path / "archive"
    incoming = archive_dir / "incoming" / "s1"
    incoming.mkdir(parents=True)
    (incoming / "manifest.json").write_text(
        json.dumps(
            {"session_id": "s1", "git": {"remote_url": "https://user:ghp_canaryfake123@github.com/o/r"}}
        )
    )
    result = sanitize.import_bundle(incoming, archive_dir)
    assert result.imported is False or result.quarantined
    assert (archive_dir / "quarantine" / "s1").is_dir()
    assert not incoming.exists()


def test_hub_import_accepts_clean_bundle_in_place(tmp_path: Path) -> None:
    from daydream.archive import sanitize

    archive_dir = tmp_path / "archive"
    incoming = archive_dir / "incoming" / "s2"
    incoming.mkdir(parents=True)
    (incoming / "manifest.json").write_text(
        json.dumps({"session_id": "s2", "git": {"remote_url": "https://github.com/o/r"}})
    )
    result = sanitize.import_bundle(incoming, archive_dir)
    assert result.imported is True
    assert result.quarantined is False
    assert incoming.exists()


def test_per_finding_resolution_round_trips_through_canonical_dict() -> None:
    from daydream.training.labeler_signals import (
        PerFindingResolution,
        resolution_from_dict,
        resolution_to_dict,
    )

    r = PerFindingResolution(
        fingerprint="fp-1", comment_id=7, disposition="accepted",
        evidence=[{"reply_id": 1, "body_sha256": "abc"}], evidence_digest="d" * 32,
    )
    restored = resolution_from_dict(resolution_to_dict(r))
    assert restored == r  # canonical dict is the one round-trip shape

def test_per_finding_resolution_from_dict_fails_closed() -> None:
    from daydream.training.labeler_signals import resolution_from_dict

    with pytest.raises(ValueError, match="fingerprint"):
        resolution_from_dict({"disposition": "accepted", "evidence_digest": "d" * 32})
    with pytest.raises(ValueError, match="disposition"):
        resolution_from_dict({"fingerprint": "fp-1", "disposition": "banana"})
    with pytest.raises(ValueError, match="evidence_digest"):
        resolution_from_dict({"fingerprint": "fp-1", "disposition": "accepted"})


def test_rubric_to_dict_carries_full_per_finding_resolutions() -> None:
    """Req 1/3: the SQLite blob must carry fingerprints + evidence + digests,
    not labels only — the materializer's sole per-finding source."""
    from daydream.training.labeler_signals import (
        CommentResolutionSignal,
        FixAppliedSignal,
        PerFindingResolution,
        PRMergeSignal,
    )
    from daydream.training.rubric import Rubric

    r = PerFindingResolution(fingerprint="fp-1", comment_id=7, disposition="accepted",
                             evidence=[{"reply_id": 1}], evidence_digest="d" * 32)
    rubric = Rubric(
        pr_merge=PRMergeSignal(True, None, "closed", False),
        fix_applied=FixAppliedSignal("applied", 1, 1, []),
        comment_resolution=CommentResolutionSignal(1, 1, 0),
        local_commit_applied=None, posterior_source="pr_review",
        per_finding_resolutions=[r],
    )
    d = rubric.to_dict()
    stored = d["per_finding_resolutions"]
    assert stored[0]["fingerprint"] == "fp-1"
    assert stored[0]["disposition"] == "accepted"
    assert stored[0]["evidence_digest"] == "d" * 32
    assert stored[0]["evidence"] == [{"reply_id": 1}]
    # backward-compat consumers keep their labels-only view
    assert d["per_finding_outcomes"] == ["accepted"]

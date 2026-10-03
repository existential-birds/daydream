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
from daydream.archive import sanitize, scan
from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    latest_label_observation,
    query_runs,
    upsert_run,
)
from daydream.archive.manifest import Manifest
from daydream.git_ops import GitError
from daydream.reviews.identity import DAYDREAM_FOOTER, finding_marker
from daydream.training import harvest, labeler_versions, reward
from daydream.training.backfill_cache import BackfillCache
from daydream.training.harvest import (
    AnnotationPayload,
    HarvestConfig,
    HarvestPass,
    assemble_scoring_inputs,
    build_annotation as _build_annotation,
)
from daydream.training.harvest_types import HarvestEvidence, HarvestRow
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingResolution,
    PRMergeSignal,
    resolution_from_dict,
    resolution_to_dict,
)
from daydream.training.reward import RewardWeights, ScoringInputs, score_trajectory
from daydream.training.rubric import Rubric
from daydream.ui import create_console
from tests.conftest import _make_repo_with_main
from tests.harness.git_helpers import commit as _commit, git as _git
from tests.harness.harvest_services import HarvestTestServices
from tests.harness.trajectory import diff_adding


def _seed_deep_bronze(tmp_path: Path) -> Path:
    """Seed a deep bronze run with a consistent verdict and diff; return its directory."""
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        json.dumps({"verdicts": [{"issue_id": 1, "verdict": "consistent"}]})
    )
    (run_dir / "diff.patch").write_text(diff_adding("new_line"))
    return run_dir

# Review-time fingerprints join findings.json to markers in Daydream comments.
_FP_A = "a" * 64
_FP_B = "b" * 64
_FP_C = "c" * 64


def _bot_comment(fp: str | None = None, *, id: int = 1) -> dict[str, Any]:
    """A footer-marked daydream finding comment, optionally carrying ``fp``'s marker."""
    marker = f"{finding_marker(fp)}\n\n" if fp is not None else ""
    return {"id": id, "in_reply_to_id": None, "user": {"login": "daydream-runner"},
        "body": f"finding\n\n{marker}{DAYDREAM_FOOTER}",
    }


def _finding_comments(
    fp: str, *, reply: str | None = None, reply_created_at: str | None = None, reply_author: str = "amelia",
) -> list[dict[str, Any]]:
    """Build one fingerprinted finding and an optional qualifying OWNER reply with its evidence time."""
    comments: list[dict[str, Any]] = [_bot_comment(fp)]
    if reply is not None:
        entry: dict[str, Any] = {
            "id": 2, "in_reply_to_id": 1, "user": {"login": reply_author}, "author_association": "OWNER", "body": reply,
        }
        if reply_created_at is not None:
            entry["created_at"] = reply_created_at
        comments.append(entry)
    return comments


def _write_findings(run_dir: Path, *fps: str) -> None:
    (run_dir / "findings.json").write_text(json.dumps({"findings": [{"fingerprint": fp} for fp in fps]}))

# This replied finding lacks a fingerprint, so the scoped join labels it unknown. A merged PR with
# no tracked comments also lacks evidence that Daydream contributed.
_REPLIED_FINDING: list[dict[str, Any]] = [
    _bot_comment(), {"id": 2, "in_reply_to_id": 1, "user": {"login": "human"}, "body": "fixed"},
]

# A footer-marked finding authored by a human still counts as unanswered when no reply exists.
_UNRESOLVED_FINDING: list[dict[str, Any]] = [
    {"id": 1, "in_reply_to_id": None, "user": {"login": "kevin"}, "body": f"finding\n\n{DAYDREAM_FOOTER}"}
]

_ORPHAN_COMMIT_PULLS: list[dict[str, Any]] = [{"number": 7, "head": {"sha": "orphsha"}}]


def _fake_gh(*, merged: bool = True, merged_at: str | None = None, comments: Sequence[dict[str, Any]] = (),
    reviews: Sequence[dict[str, Any]] = (), commit_pulls: Sequence[dict[str, Any]] | None = None,
    pr_author: str | None = None,
) -> Callable[..., Any]:
    """Serve PR state, comments, reviews and optional commit-to-PR linkage.

    Review logins exercise the real reviewer-prior database query."""

    def responder(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if commit_pulls is not None and "/commits/" in endpoint and endpoint.endswith("/pulls"):
            return list(commit_pulls)
        if endpoint.endswith("/comments"):
            return list(comments)
        if endpoint.endswith("/reviews"):
            return list(reviews)
        pull: dict[str, Any] = {"merged": merged, "merged_at": merged_at}
        if pr_author is not None:
            pull.update(state="merged" if merged else "open", user={"login": pr_author})
        return pull

    return responder


def _applied_finding_gh() -> Callable[..., Any]:
    return _fake_gh(merged_at="2026-08-10T00:00:00Z",
        comments=_finding_comments(_FP_A, reply="applied", reply_created_at="2026-08-02T10:00:00Z"),
    )


def _unused_gh(repo: str, endpoint: str, **kwargs: Any) -> Any:
    raise AssertionError(f"gh_api should not be called for a local row (endpoint={endpoint})")


def _typed_row(raw: dict[str, Any], *, row_number: int = 1) -> HarvestRow:
    return HarvestRow.from_mapping(raw, row_number=row_number)


def _services(config: HarvestConfig, *, github: Callable[..., Any]) -> HarvestTestServices:
    """Build an explicit per-run service with only the GitHub boundary replaced."""
    return HarvestTestServices(HarvestPass(config), github=github)


def _pr_row(run_dir: Path, session_id: str, *, pr_number: int = 7) -> dict[str, Any]:
    return {"session_id": session_id, "pr_repo": "o/r", "pr_number": pr_number, "head_sha": "h", "base_branch": "main",
        "archive_path": str(run_dir), "changed_files": "[]",
    }


def _local_row(run_dir: Path, session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "pr_repo": None, "pr_number": None, "branch": "feat", "head_sha": "h",
        "archive_path": str(run_dir), "changed_files": "[]",
    }


def _write_recommended_patch(run_dir: Path) -> None:
    (run_dir / "recommended.patch").write_text(diff_adding("guarded = True"))


def _acquire_annotation(
    raw: dict[str, Any], *, run_dir: Path | None = None, archive_dir: Path, gh_api: Callable[..., Any],
    repo_clone: Path | None = None, services: HarvestPass | None = None, valid_at_override: str | None = None,
) -> AnnotationPayload:
    """Acquire through an explicit provider, then exercise the pure reducer."""
    row = _typed_row(raw)
    if run_dir is not None:
        assert row.archive_path == run_dir
    config = HarvestConfig(archive_dir=archive_dir)
    provider = services or HarvestTestServices(HarvestPass(config), github=gh_api)
    evidence = provider.acquire_harvest_evidence(row, repo_resolution=repo_clone,
        base_sha_status="available" if repo_clone is not None else "unavailable", valid_at_override=valid_at_override,
    )
    return _build_annotation(row, evidence)


def test_build_annotation_pr_row_labels_from_decisive_reply_evidence(tmp_path: Path) -> None:
    run_dir = _seed_deep_bronze(tmp_path)
    _write_findings(run_dir, _FP_A)
    row = _pr_row(run_dir, "s1")
    ann = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
        gh_api=_fake_gh(merged_at="2026-02-05T00:00:00+00:00",
            comments=_finding_comments(_FP_A, reply="applied", reply_created_at="2026-02-01T10:00:00Z"),
        ), repo_clone=tmp_path,
    )
    assert ann.labels == ["accepted"]
    # valid_at follows the qualifying accept reply, not the later merge.
    assert ann.valid_at == "2026-02-01T10:00:00Z"
    assert ann.composite_reward == json.loads(ann.reward_json)["composite"]


def _fake_gh_merged_per_finding(merged_at: str, fp_replied: str, fp_unreplied: str) -> Any:
    """Return a merged PR with two marked findings and one qualifying reply to the first."""

    return _fake_gh(merged_at=merged_at,
        comments=[_bot_comment(fp_replied), _bot_comment(fp_unreplied, id=2),
            {"id": 3, "in_reply_to_id": 1, "user": {"login": "human"},
             "author_association": "MEMBER", "body": "applied"},
        ],
    )


def test_build_annotation_pr_row_carries_per_finding_outcomes(tmp_path: Path) -> None:
    """Join findings by fingerprint: the replied finding is accepted and the other unanswered."""
    fp_a = "a" * 64
    fp_b = "b" * 64
    run_dir = _seed_deep_bronze(tmp_path)
    (run_dir / "findings.json").write_text(json.dumps({"findings": [{"fingerprint": fp_a}, {"fingerprint": fp_b}]}))
    row = _pr_row(run_dir, "s_pf")
    ann = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                           gh_api=_fake_gh_merged_per_finding("2026-02-01T00:00:00+00:00", fp_a, fp_b),
                           repo_clone=tmp_path)
    assert ann.rubric_json is not None
    assert json.loads(ann.rubric_json)["per_finding_outcomes"] == ["accepted", "unanswered"]

def test_harvest_626_shape_yields_both_polarities(tmp_path: Path) -> None:
    """A qualifying acceptance, rejection, and question preserve per-finding polarity and a contested run
    label.
    """
    run_dir = _seed_deep_bronze(tmp_path)
    _write_findings(run_dir, _FP_A, _FP_B, _FP_C)
    row = _pr_row(run_dir, "s_626")
    ann = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
        gh_api=_fake_gh(merged_at="2026-02-05T00:00:00+00:00",
            comments=[_bot_comment(_FP_A), _bot_comment(_FP_B, id=2), _bot_comment(_FP_C, id=3),
                {"id": 4, "in_reply_to_id": 1, "user": {"login": "amelia"}, "author_association": "OWNER",
                    "body": "Fixed in abc123", "created_at": "2026-02-01T10:00:00Z",
                }, {"id": 5, "in_reply_to_id": 2, "user": {"login": "amelia"}, "author_association": "OWNER",
                    "body": "False positive", "created_at": "2026-02-01T11:00:00Z",
                }, {"id": 6, "in_reply_to_id": 3, "user": {"login": "bob"}, "author_association": "MEMBER",
                    "body": "Which branch is this against?", "created_at": "2026-02-01T09:00:00Z",
                },
            ],
        ), repo_clone=tmp_path,
    )
    assert ann.labels == ["contested"]
    assert ann.rubric_json is not None
    rubric = json.loads(ann.rubric_json)
    assert rubric["per_finding_outcomes"] == ["accepted", "rejected", "ambiguous"]
    # Use the earliest decisive evidence; the ambiguous MEMBER question does not qualify.
    assert ann.valid_at == "2026-02-01T10:00:00Z"

def test_build_annotation_applies_posterior_penalty_for_rejected_pr(tmp_path: Path) -> None:
    # Rejection adds posterior_cost without deducting from the intrinsic composite.
    run_dir = _seed_deep_bronze(tmp_path)
    row = _pr_row(run_dir, "s_rej", pr_number=9)

    intrinsic_inputs = assemble_scoring_inputs(run_dir)
    intrinsic_only_composite = score_trajectory(intrinsic_inputs).composite

    _write_findings(run_dir, _FP_A)
    payload = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
        gh_api=_fake_gh(merged=False, comments=_finding_comments(_FP_A, reply="not applicable")), repo_clone=tmp_path,
    )

    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    breakdown = json.loads(payload.reward_json)
    assert breakdown["false_positive_penalty"] == 1.0
    assert breakdown["posterior_cost"] == 0.5  # sibling: max(0, 1.0 − 0.5 default prior)
    assert payload.composite_reward == intrinsic_only_composite  # pure intrinsic

def test_build_annotation_rejected_pr_empty_pool_uses_default_prior(tmp_path: Path, archive_dir: Any) -> None:
    # Reviewer alice triggers the real prior query; the empty archive returns (None, 0), selecting
    # the default 0.5 prior.
    run_dir = _seed_deep_bronze(tmp_path)
    row = _pr_row(run_dir, "s_rej_prod", pr_number=9)
    _write_findings(run_dir, _FP_A)
    payload = _acquire_annotation(row, run_dir=run_dir, archive_dir=archive_dir,
        gh_api=_fake_gh(merged=False, reviews=[{"user": {"login": "alice"}}],
            comments=_finding_comments(_FP_A, reply="not applicable"),
        ), repo_clone=tmp_path,
    )
    assert payload.labels == ["rejected"]
    assert payload.has_posterior is True
    rb = json.loads(payload.reward_json)
    assert rb["outcome_prior"] is None
    assert rb["outcome_prior_n"] == 0
    assert rb["posterior_cost"] == 0.5
    assert payload.reviewer_logins == ["alice", "amelia"]

@pytest.mark.parametrize(("session", "pr_number", "author", "pr_author", "body", "created", "label"), [
    pytest.param("s_fork_auth", 13, "prfiona", "prfiona", "Fixed in abc123", "2026-08-02T10:00:00Z", "accepted",
        id="fork-pr-author"),
    pytest.param("s_review_auth", 14, "revbob", "someone-else", "False positive, the linter is wrong here",
        "2026-08-02T09:00:00Z", "rejected", id="formal-review-author"),
])
def test_build_annotation_author_reply_is_decisive(
    tmp_path: Path, session: str, pr_number: int, author: str, pr_author: str, body: str, created: str, label: str,
) -> None:
    """A PR author or formal reviewer qualifies even with author_association NONE."""
    run_dir = _seed_deep_bronze(tmp_path)
    _write_findings(run_dir, _FP_A)
    row = _pr_row(run_dir, session, pr_number=pr_number)
    comments = [_bot_comment(_FP_A), {
        "id": 2, "in_reply_to_id": 1, "user": {"login": author, "type": "User"},
        "author_association": "NONE", "body": body, "created_at": created,
    }]
    github = _fake_gh(merged_at="2026-08-03T00:00:00Z", comments=comments, pr_author=pr_author,
        reviews=[] if author == pr_author else [{"user": {"login": author}}],
    )

    payload = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=github, repo_clone=tmp_path)

    assert payload.labels == [label]
    if author == pr_author:
        assert payload.valid_at == created
    rubric = json.loads(payload.rubric_json or "{}")
    assert rubric.get("per_finding_outcomes") == [label]

def test_build_annotation_shallow_local_row_null_valid_at_reward_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()           # no deep/ → shallow
    row = _local_row(run_dir, "s2")
    ann = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh, repo_clone=tmp_path)
    assert ann.valid_at is None                               # collapses to observed_at on write
    rb = json.loads(ann.reward_json)
    assert rb["axes_present"]["correctness"] is False         # shallow: no verdicts

def test_build_annotation_rejected_pr_populated_prior_drives_pool(tmp_path: Path, archive_dir: Any) -> None:
    # Seed real reviewer history to exercise before_valid_at filtering. Even below ten observations,
    # prior_n must reveal the nonempty pool.
    prior_session_id = "s_prior_alice"
    upsert_run(archive_dir,
        Manifest(session_id=prior_session_id, archived_at="2025-01-01T00:00:00Z", run_flow="normal", backend="claude",
            repo_slug="org/repo", pr_repo="org/repo", pr_number=1, head_sha="aaa", base_branch="main",

            changed_files=["app.py"], archive_path=str(tmp_path),
        ),
    )
    append_label_observation(
        archive_dir, prior_session_id, labels=["rejected"], pr_state="closed", labeler_version="test",
        evidence_sha=None, valid_at="2025-06-01T00:00:00Z",   # strictly in the past
        reviewer_logins=["alice"], has_posterior=True,
    )

    run_dir = _seed_deep_bronze(tmp_path / "current_run")
    row = _pr_row(run_dir, "s_rej_populated", pr_number=9)
    _write_findings(run_dir, _FP_A)
    payload = _acquire_annotation(row, run_dir=run_dir, archive_dir=archive_dir,
        gh_api=_fake_gh(merged=False, reviews=[{"user": {"login": "alice"}}],
            comments=_finding_comments(_FP_A, reply="not applicable"),
        ), repo_clone=tmp_path,
    )
    assert payload.labels == ["rejected"]
    rb = json.loads(payload.reward_json)
    # Below ten observations, outcome_prior stays None while prior_n proves the real history query
    # found the seed.
    assert rb["outcome_prior_n"] >= 1, (f"expected prior_n >= 1 from seeded archive, got {rb['outcome_prior_n']}"
    )
    assert rb["outcome_prior"] is None   # n < 10 threshold → fallback to default
    assert rb["posterior_cost"] == 0.5   # default prior applied

def test_build_annotation_pr_uses_pooled_prior_and_persists_reviewers(tmp_path: Path,) -> None:
    run_dir = _seed_deep_bronze(tmp_path)
    row = _pr_row(run_dir, "s_rej", pr_number=9)
    _write_findings(run_dir, _FP_A)
    config = HarvestConfig(archive_dir=tmp_path)
    p = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh, repo_clone=tmp_path,
        services=HarvestTestServices(HarvestPass(config),
            github=_fake_gh(merged=False, comments=_finding_comments(_FP_A, reply="not applicable"),
                reviews=[{"user": {"login": "alice"}}, {"user": {"login": "carol"}}],
            ), reviewer_prior=lambda *_args, **_kwargs: (0.8, 12),
        ),
    )
    rb = json.loads(p.reward_json)
    assert rb["posterior_cost"] == pytest.approx(0.2)   # max(0, 1.0 - 0.8)
    assert rb["outcome_prior"] == 0.8 and rb["outcome_prior_n"] == 12
    assert p.composite_reward == rb["composite"]        # stored composite is pure intrinsic (C5)
    assert p.has_posterior is True and p.reviewer_logins == ["alice", "amelia", "carol"]

def test_build_annotation_below_threshold_falls_back_to_default_prior(tmp_path: Path,) -> None:
    run_dir = _seed_deep_bronze(tmp_path)
    row = _pr_row(run_dir, "s_rej", pr_number=9)
    _write_findings(run_dir, _FP_A)
    rb = json.loads(_acquire_annotation(
            row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh, repo_clone=tmp_path,
            services=HarvestTestServices(HarvestPass(HarvestConfig(archive_dir=tmp_path)),
                github=_fake_gh(merged=False, comments=_finding_comments(_FP_A, reply="not applicable"),
                    reviews=[{"user": {"login": "alice"}}],
                ), reviewer_prior=lambda *_args, **_kwargs: (0.9, 4),
            ),
        ).reward_json
    )
    assert rb["outcome_prior"] is None and rb["outcome_prior_n"] == 4  # n recorded; prior None -> 0.5
    assert rb["posterior_cost"] == 0.5

def test_build_annotation_local_row_has_no_reviewer_prior(tmp_path: Path) -> None:
    # A local commit is not PR maintainer evidence: retain its label but withhold the posterior axis
    # and reviewer prior.
    run_dir = _seed_deep_bronze(tmp_path)
    row = _local_row(run_dir, "s_local")
    config = HarvestConfig(archive_dir=tmp_path)
    p = _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_unused_gh, repo_clone=tmp_path,
        services=HarvestTestServices(HarvestPass(config),
            local_commit_applied=lambda *_args, **_kwargs: LocalCommitAppliedSignal("rejected"),
            reviewer_prior=lambda *_args, **_kwargs: pytest.fail("local rows have no reviewer prior"),
        ),
    )
    assert p.reviewer_logins == []
    assert p.labels == ["rejected"]  # label kept — consumers may still want it
    assert p.has_posterior is False  # ...but it is not maintainer-PR evidence
    rb = json.loads(p.reward_json)
    assert "posterior_cost" not in rb and "outcome_prior" not in rb
    assert rb["composite"] is not None  # the intrinsic axes are unaffected

def test_build_annotation_asserts_canonical_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Weights that differ from the canonical constant mark a score custom and must block
    # publication.
    monkeypatch.setattr(reward, "DEFAULT_WEIGHTS", RewardWeights())
    run_dir = _seed_deep_bronze(tmp_path)
    row = _pr_row(run_dir, "s_custom", pr_number=9)
    with pytest.raises((AssertionError, RuntimeError), match="canonical"):
        _acquire_annotation(row, run_dir=run_dir, archive_dir=tmp_path,
                         gh_api=_fake_gh(merged=False), repo_clone=tmp_path)

def test_assemble_reads_verdicts_from_bronze(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text(
        '{"verdicts":[{"issue_id":1,"verdict":"consistent"}]}'
    )
    inputs = assemble_scoring_inputs(run_dir)
    assert inputs.verifier_verdicts == [{"issue_id": 1, "verdict": "consistent"}]
    assert inputs.format_valid is True

def test_assemble_shallow_run_has_null_verdicts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs = assemble_scoring_inputs(run_dir)
    assert inputs.verifier_verdicts is None

def test_assemble_declined_deep_run_null_verdicts_keeps_format_valid(tmp_path: Path) -> None:
    # Declined deep runs retain the deep bundle but skip recommendation verification. Absent
    # verdicts are expected and must not invalidate format.
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "stack-python-records.json").write_text(json.dumps({"records": [{"id": "i1"}]}))
    inputs = assemble_scoring_inputs(run_dir)
    assert inputs.verifier_verdicts is None          # declined ⇒ no verdicts
    assert inputs.format_valid is True                # absence is expected, not malformed

def test_assemble_malformed_verdicts_flags_format_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "deep").mkdir(parents=True)
    (run_dir / "deep" / "recommendation-verdicts.json").write_text("{not json")
    inputs = assemble_scoring_inputs(run_dir)
    assert inputs.format_valid is False


def _seed_run_manifest(archive_dir: Path, run_dir: Path, session_id: str, *, pr_number: int | None, pr_repo: str | None,
    head_sha: str = "abc", base_branch: str | None = "main", branch: str | None = None, source_path: Path | None = None,
) -> None:
    """Index a bronze bundle with the requested PR linkage, branch identity and source path."""
    upsert_run(archive_dir,
        Manifest(session_id=session_id, archived_at="2026-01-01T00:00:00Z", run_flow="normal", backend="claude",
            repo_slug="org/repo", branch=branch, head_sha=head_sha, base_branch=base_branch, pr_number=pr_number,
            pr_repo=pr_repo,

            changed_files=["app.py"], archive_path=str(run_dir), source_path=str(source_path) if source_path else None,
        ),
    )


def _seed_archived_deep_run(archive_dir: Path, session_id: str, *, source_path: Path | None = None,) -> Path:
    """Seed and index deep bronze; source_path enables real working-tree clone resolution."""
    run_dir = _seed_deep_bronze(archive_dir)
    _seed_run_manifest(archive_dir, run_dir, session_id, pr_number=42, pr_repo="org/repo", source_path=source_path)
    return run_dir


def _seed_orphan_run(
    archive_dir: Path, bronze_parent: Path, *, session_id: str, head_sha: str = "orphsha", branch: str = "feat/x",
    base_branch: str | None = "main", source_path: Path | None = None,
) -> Path:
    """Seed an unlinked deep run whose source_path enables the local commit walk."""
    run_dir = _seed_deep_bronze(bronze_parent)
    _seed_run_manifest(
        archive_dir, run_dir, session_id, pr_number=None, pr_repo=None, head_sha=head_sha, base_branch=base_branch,
        branch=branch, source_path=source_path,
    )
    return run_dir


def _seed_pr_runs(archive_dir: Path, bronze_parent: Path, count: int, *, fingerprints: Sequence[str] | None = None,
) -> None:
    """Seed s1..sN with distinct bronze directories and matching PR numbers for endpoint routing."""
    for pr_number in range(1, count + 1):
        sid = f"s{pr_number}"
        run_dir = _seed_deep_bronze(bronze_parent / sid)
        _seed_run_manifest(archive_dir, run_dir, sid, pr_number=pr_number, pr_repo="org/repo")
        if fingerprints:
            _write_findings(run_dir, *fingerprints)


@pytest.mark.parametrize("merged_at", ["2026-02-01T00:00:00+00:00", "2026-02-01T00:00:00Z"])
async def test_harvest_writes_one_annotation_with_canonical_merge_time(tmp_path: Path, archive_dir: Any, merged_at: str
) -> None:
    _seed_archived_deep_run(archive_dir, "s1")
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_fake_gh(merged_at=merged_at, comments=_REPLIED_FINDING)).run()
    obs = latest_label_observation(archive_dir, "s1")
    assert obs is not None
    assert summary["annotated"] == 1
    assert obs["valid_at"] == "2026-02-01T00:00:00+00:00"
    assert obs["composite_reward"] is not None

@pytest.mark.parametrize(("malformed", "session_label"),
    [({"session_id": "bad/session", "archive_path": "/tmp/bronze"}, "bad/session"),
        ({"archive_path": "/tmp/bronze"}, "<unknown>"), ({"session_id": "missing-archive"}, "missing-archive"),
        ({"session_id": "unsafe-slug", "archive_path": "/tmp/bronze", "pr_repo": "org/repo/extra"}, "unsafe-slug"),
    ],
)
async def test_harvest_rejects_malformed_index_row_before_any_side_effect(
    tmp_path: Path, archive_dir: Any, capsys: pytest.CaptureFixture[str], malformed: dict[str, Any], session_label: str,
) -> None:
    effects: list[str] = []

    def _completed_sessions() -> set[str]:
        effects.append("cache-read")
        return set()

    def _resolve_repo(*_args: Any, **_kwargs: Any) -> None:
        effects.append("clone")
        return None

    def _github(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        effects.append("github")
        return {}

    def _append_annotation(*_args: Any, **_kwargs: Any) -> bool:
        effects.append("write")
        return True

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "cache")
    services = HarvestTestServices(
        HarvestPass(config), rows=[malformed], completed_sessions=_completed_sessions,
        resolve_repo=_resolve_repo, github=_github, append_annotation=_append_annotation,
    )
    summary = await services.run()

    assert summary["considered"] == 1
    assert summary["errors"] == 1
    assert effects == []
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "row 1" in output and session_label in output

async def test_harvest_validates_completed_rows_before_resume_filtering(tmp_path: Path, archive_dir: Any,) -> None:
    _seed_archived_deep_run(archive_dir, "done")
    fresh_dir = _seed_deep_bronze(tmp_path / "fresh")
    _seed_run_manifest(archive_dir, fresh_dir, "fresh", pr_number=42, pr_repo="org/repo")
    completed = query_runs(archive_dir, "session_id = ?", ("done",))[0]
    fresh = query_runs(archive_dir, "session_id = ?", ("fresh",))[0]
    malformed_completed = {"session_id": "done", "archive_path": "relative/bronze"}
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "cache")
    services = HarvestTestServices(
        HarvestPass(config), rows=[malformed_completed, completed, fresh], completed={"done"},
        github=_fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING),
    )

    summary = await services.run()

    assert summary == {"considered": 2, "annotated": 1, "would_annotate": 0, "skipped": 0, "errors": 1, "aborted": 0}
    assert latest_label_observation(archive_dir, "done") is None
    assert latest_label_observation(archive_dir, "fresh") is not None

async def test_harvest_unresolved_daydream_comment_stays_unknown(tmp_path: Path, archive_dir: Any,) -> None:
    """A merge without a qualifying reply supplies no decisive label evidence."""
    run_dir = _seed_archived_deep_run(archive_dir, "s-contest")
    _write_findings(run_dir, _FP_A)
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    await _services(
            config, github=_fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_finding_comments(_FP_A)),
        ).run()
    row = query_runs(archive_dir, "session_id = ?", ("s-contest",))[0]
    assert json.loads(row["outcome_labels"]) == []  # unknown, never "accepted"

async def test_harvest_relinks_orphan_run_and_labels_it(tmp_path: Path, archive_dir: Any,) -> None:
    """Re-link a run archived before its PR existed, then persist its PR label.

    A decisive accept beside an unanswered finding remains contested."""
    run_dir = _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph")
    _write_findings(run_dir, _FP_A, _FP_B)
    github = _fake_gh(merged_at="2026-02-01T00:00:00+00:00",
            comments=[*_finding_comments(_FP_A, reply="applied"), _bot_comment(_FP_B, id=3)],
            commit_pulls=_ORPHAN_COMMIT_PULLS,
        )
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    await _services(config, github=github).run()
    row = query_runs(archive_dir, "session_id = ?", ("s-orph",))[0]
    assert row["pr_number"] == 7 and row["pr_repo"] == "org/repo"  # linkage persisted
    assert json.loads(row["outcome_labels"]) == ["contested"]  # now labelable (was orphan)

@pytest.mark.parametrize("with_clone", [False, True])
async def test_harvest_fork_pr_404_degrades_not_drops(
    tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch, with_clone: Any,
) -> None:
    """A benign PR 404 preserves the annotation with local provenance and no error.

    Missing PR evidence stays unknown even when a clone resolves: a PR-shaped
    row is ineligible for the PR-less commit walk. This prevents an unavailable
    merge from becoming a fabricated rejection."""
    source_path = None
    if with_clone:
        source_path = tmp_path / "clone"
        (source_path / ".git").mkdir(parents=True)
    _seed_archived_deep_run(archive_dir, "s-fork", source_path=source_path)

    def _gh_fork_404(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if re.search(r"/pulls/\d+", endpoint):
            raise GitError("gh: Not Found (HTTP 404)")
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            return []
        return {}

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_fork_404).run()

    assert summary["errors"] == 0  # benign 404 degraded; not a hard error
    obs = latest_label_observation(archive_dir, "s-fork")
    assert obs is not None
    # pr_state=None proves local-rubric fallback instead of a fabricated unmerged PR rubric.
    assert obs["pr_state"] is None
    # Without a resolvable clone, the local posterior is unknown and outcome_labels stays empty.
    row = query_runs(archive_dir, "session_id = ?", ("s-fork",))[0]
    assert json.loads(row["outcome_labels"]) == []

async def test_harvest_orphan_422_degrades_not_drops(tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpushed head's link-probe 422 preserves the unlinked annotation.

    With no clone, the label is unknown; the row is neither dropped nor rejected."""
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph-422")
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_unpushed_422).run()

    assert summary["errors"] == 0  # benign 422 degraded; not a hard error
    assert latest_label_observation(archive_dir, "s-orph-422") is not None  # annotated via local path
    row = query_runs(archive_dir, "session_id = ?", ("s-orph-422",))[0]
    assert row["pr_number"] is None and row["pr_repo"] is None  # linkage NOT applied
    assert json.loads(row["outcome_labels"]) == []


def _gh_unpushed_422(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """gh stub for a squash-merged head SHA: the link probe 422s, nothing else."""
    if "/commits/" in endpoint and endpoint.endswith("/pulls"):
        raise GitError("gh: No commit found for SHA (HTTP 422)")
    if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
        return []
    return {}


async def test_harvest_deleted_branch_ref_labels_unknown_not_rejected(
    tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable deleted-branch window yields unknown even with a resolved clone.

    Exercise real Git failure after the recorded SHA's link probe returns 422;
    a failed log must remain distinct from a readable empty commit window."""
    clone = _make_repo_with_main(tmp_path, name="clone")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    # The branch the run recorded was squash-merged and deleted — never created here.
    _seed_orphan_run(archive_dir, tmp_path / "bronze", session_id="s-gone", head_sha=head_sha,
        branch="feat/squash-merged-and-deleted", source_path=clone,
    )
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_unpushed_422).run()

    assert summary["errors"] == 0  # benign 422 + unreadable window degrade, not error
    assert latest_label_observation(archive_dir, "s-gone") is not None  # still annotated
    row = query_runs(archive_dir, "session_id = ?", ("s-gone",))[0]
    # The unreadable commit window stays unknown; main lacks the fix, so base-branch fallback cannot
    # upgrade it.
    assert json.loads(row["outcome_labels"]) == []

async def test_harvest_squash_merged_branch_recovers_accepted_from_base_branch(
    tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover an applied change from main after a real squash merge and branch deletion.

    The branch window is unreadable, but the recommended lines at the base tip
    still provide positive local evidence."""
    clone = _make_repo_with_main(tmp_path, name="clone")
    head_sha_holder: dict[str, str] = {}
    _git(clone, "checkout", "-b", "feat/squash-me")
    (clone / "app.py").write_text("existing\nguarded = True\n")
    _git(clone, "add", "app.py")
    _commit(clone, "add the guard")
    head_sha_holder["sha"] = _git(clone, "rev-parse", "HEAD").strip()
    _git(clone, "checkout", "main")
    _git(clone, "merge", "--squash", "feat/squash-me")
    _commit(clone, "add the guard (#220)")
    _git(clone, "branch", "-D", "feat/squash-me")

    run_dir = _seed_orphan_run(archive_dir, tmp_path / "bronze", session_id="s-squash", head_sha=head_sha_holder["sha"],
        branch="feat/squash-me", source_path=clone,
    )
    _write_recommended_patch(run_dir)
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_unpushed_422).run()

    assert summary["errors"] == 0
    row = query_runs(archive_dir, "session_id = ?", ("s-squash",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]

async def test_harvest_live_branch_with_no_followup_commits_still_labels_rejected(
    tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable empty commit window remains a rejection; only an unreadable window is unknown."""
    clone = _make_repo_with_main(tmp_path, name="clone")
    _git(clone, "checkout", "-b", "feat/still-here")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    _seed_orphan_run(archive_dir, tmp_path / "bronze", session_id="s-live", head_sha=head_sha, branch="feat/still-here",
        source_path=clone,
    )
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_unpushed_422).run()

    assert summary["errors"] == 0
    row = query_runs(archive_dir, "session_id = ?", ("s-live",))[0]
    assert json.loads(row["outcome_labels"]) == ["rejected"]

@pytest.mark.parametrize("branch", ["feat/fixed", ""])
async def test_harvest_live_branch_with_applied_fix_labels_accepted(tmp_path: Path, archive_dir: Any, branch: str,
) -> None:
    """Exercise the accepted arm through real Git, complementing unknown and rejected commit-window cases."""
    clone = _make_repo_with_main(tmp_path, name="clone")
    if branch:
        _git(clone, "checkout", "-b", branch)
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    session_id = "s-applied-empty-branch" if not branch else "s-applied"
    run_dir = _seed_orphan_run(
        archive_dir, tmp_path / "bronze", session_id=session_id, head_sha=head_sha, branch=branch, base_branch=None,
        source_path=clone,
    )
    # The recommended patch adds a line; a later commit on the branch lands it.
    _write_recommended_patch(run_dir)
    (clone / "app.py").write_text("existing\nguarded = True\n")
    _git(clone, "add", "app.py")
    _commit(clone, "apply the recommended fix")
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    await _services(config, github=_gh_unpushed_422).run()

    row = query_runs(archive_dir, "session_id = ?", (session_id,))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]

async def test_harvest_merged_pr_with_zero_comments_is_not_labeled_accepted(tmp_path: Path, archive_dir: Any,) -> None:
    """A merge with no tracked findings remains unknown and outside the posterior population.

    Zero unresolved comments is vacuous when there were no comments at all."""
    _seed_archived_deep_run(archive_dir, "s-vacuous")
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_fake_gh(merged_at="2026-02-01T00:00:00+00:00")).run()

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

async def test_harvest_merged_pr_with_reject_reply_is_contested(tmp_path: Path, archive_dir: Any,) -> None:
    """A qualifying rejection beside an unanswered finding stays contested despite a merge."""
    run_dir = _seed_archived_deep_run(archive_dir, "s-reject")
    _write_findings(run_dir, _FP_A, _FP_B)
    github = _fake_gh(merged_at="2026-08-10T00:00:00Z",
            comments=[*_finding_comments(_FP_A, reply="False positive — the code already handles this"),
                _bot_comment(_FP_B, id=3),
            ],
        )
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=github).run()

    assert summary["errors"] == 0 and summary["annotated"] == 1
    row = query_runs(archive_dir, "session_id = ?", ("s-reject",))[0]
    assert json.loads(row["outcome_labels"]) == ["contested"]  # NEVER ["accepted"]
    obs = latest_label_observation(archive_dir, "s-reject")
    assert obs is not None
    assert json.loads(obs["rubric_json"])["per_finding_outcomes"] == ["rejected", "unanswered"]

async def test_harvest_unmerged_pr_with_no_semantic_reply_is_unknown(
    tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bot-only reply supplies no human judgment; preserve open state without inferring rejection."""
    run_dir = _seed_archived_deep_run(archive_dir, "s-open")
    _write_findings(run_dir, _FP_A)

    def _gh_open(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/comments"):
            return [*_finding_comments(_FP_A),
                {"id": 2, "in_reply_to_id": 1, "user": {"login": "dependabot[bot]", "type": "Bot"}, "body": "applied"},
            ]
        if endpoint.endswith("/reviews"):
            return []
        return {"merged": False, "merged_at": None, "state": "open"}

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_open).run()

    assert summary["errors"] == 0 and summary["annotated"] == 1
    row = query_runs(archive_dir, "session_id = ?", ("s-open",))[0]
    assert json.loads(row["outcome_labels"]) == []  # unknown, NOT ["rejected"]
    obs = latest_label_observation(archive_dir, "s-open")
    assert obs is not None
    assert obs["pr_state"] == "open"

@pytest.mark.parametrize(
    ("override", "expected"), [(None, "2026-08-02T10:00:00Z"), ("2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z")],
    ids=["decisive-evidence", "explicit-override"],
)
def test_valid_at_uses_decisive_evidence_unless_overridden(tmp_path: Path, override: str | None, expected: str) -> None:
    run_dir = _seed_deep_bronze(tmp_path)
    _write_findings(run_dir, _FP_A)
    row = _pr_row(run_dir, "s-val")
    ann = _acquire_annotation(
        row, run_dir=run_dir, archive_dir=tmp_path, gh_api=_applied_finding_gh(), repo_clone=tmp_path,
        valid_at_override=override,
    )
    assert ann.valid_at == expected

async def test_labeler_version_is_not_reward_version(tmp_path: Path, archive_dir: Any,) -> None:
    run_dir = _seed_archived_deep_run(archive_dir, "s-lv")
    _write_findings(run_dir, _FP_A)
    captured: dict[str, Any] = {}

    def _capture(_row: HarvestRow, payload: AnnotationPayload) -> bool:
        captured.update(labeler_version=labeler_versions.LABELER_POLICY_VERSION,
            reply_classifier_version=payload.reply_classifier_version,
            reply_evidence_digest=payload.reply_evidence_digest,
        )
        return True

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await HarvestTestServices(
            HarvestPass(config), github=_applied_finding_gh(), append_annotation=_capture,
        ).run()

    assert summary["annotated"] == 1
    assert captured["labeler_version"] == labeler_versions.LABELER_POLICY_VERSION
    assert captured["labeler_version"] != reward.REWARD_VERSION
    assert captured["reply_classifier_version"] == labeler_versions.REPLY_CLASSIFIER_VERSION
    assert captured["reply_evidence_digest"]

def test_resume_cache_invalidated_on_policy_bump(tmp_path: Path) -> None:
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    cache.mark_session_done("sess-1")
    with patch.object(labeler_versions, "LABELER_POLICY_VERSION", "980-r2"):
        assert "sess-1" not in cache.completed_sessions()
    assert "sess-1" in cache.completed_sessions()

async def test_harvest_local_branch_accept_keeps_label_but_is_not_posterior_evidence(
    tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist a local applied label without posterior credit.

    A local commit is weaker evidence than maintainer action on a PR; preserve
    has_posterior=0 and omit posterior_cost so consumers can distinguish them."""
    clone = _make_repo_with_main(tmp_path, name="clone")
    _git(clone, "checkout", "-b", "feat/local-tier")
    head_sha = _git(clone, "rev-parse", "HEAD").strip()
    run_dir = _seed_orphan_run(
        archive_dir, tmp_path / "bronze", session_id="s-local-tier", head_sha=head_sha, branch="feat/local-tier",
        source_path=clone,
    )
    _write_recommended_patch(run_dir)
    (clone / "app.py").write_text("existing\nguarded = True\n")
    _git(clone, "add", "app.py")
    _commit(clone, "apply the recommended fix")
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_unpushed_422).run()

    assert summary["errors"] == 0
    row = query_runs(archive_dir, "session_id = ?", ("s-local-tier",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]  # label kept
    obs = latest_label_observation(archive_dir, "s-local-tier")
    assert obs is not None
    assert json.loads(obs["rubric_json"])["posterior_source"] == "local_branch"
    assert obs["has_posterior"] == 0
    assert "posterior_cost" not in json.loads(obs["reward_json"])

async def test_harvest_dry_run_mutates_row_in_memory_but_suppresses_set_run_pr_link(tmp_path: Path, archive_dir: Any,
) -> None:
    """Preview linked PR evidence in memory without persisting the linkage.

    The real database writer is active, so a broken dry-run guard changes the
    asserted database state."""
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-orph-dry")

    github = _fake_gh(
            merged_at="2026-02-01T00:00:00+00:00", comments=_UNRESOLVED_FINDING, commit_pulls=_ORPHAN_COMMIT_PULLS,
        )
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c", dry_run=True)
    summary = await _services(config, github=github).run()

    assert summary["would_annotate"] == 1
    assert summary["annotated"] == 0

    row = query_runs(archive_dir, "session_id = ?", ("s-orph-dry",))[0]
    assert row["pr_number"] is None
    assert row["pr_repo"] is None

    assert latest_label_observation(archive_dir, "s-orph-dry") is None

async def test_harvest_leaves_true_local_run_unlinked(tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_orphan_run(archive_dir, tmp_path, session_id="s-local", head_sha="localsha")

    def _gh_no_pr(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/pulls") and "/commits/" in endpoint:
            return []  # no PR ever opened
        raise AssertionError(f"PR endpoints must not be hit for an unlinked local run ({endpoint})")

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_no_pr).run()
    row = query_runs(archive_dir, "session_id = ?", ("s-local",))[0]
    assert row["pr_number"] is None
    assert summary["errors"] == 0

async def test_re_harvest_is_idempotent(tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_archived_deep_run(archive_dir, "s1")
    github = _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING)
    first_config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1")
    await _services(first_config, github=github).run()
    second_config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2")
    second = await _services(second_config, github=github).run()
    assert len(label_observation_history(archive_dir, "s1")) == 1  # deduped
    assert second["skipped"] == 1 and second["annotated"] == 0

async def test_re_harvest_appends_on_version_bump(tmp_path: Path, archive_dir: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_archived_deep_run(archive_dir, "s1")
    github = _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING)
    first_config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1")
    await _services(first_config, github=github).run()
    monkeypatch.setattr("daydream.training.labeler_versions.LABELER_POLICY_VERSION", "980-policy-bump")
    second_config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2")
    await _services(second_config, github=github).run()
    assert len(label_observation_history(archive_dir, "s1")) == 2

async def test_harvest_aborts_cleanly_on_rate_limit_and_preserves_resume(tmp_path: Path, archive_dir: Any,) -> None:
    # PR 1 succeeds; exhausted rate limits on PR 2 must abort while preserving completed work.
    _seed_pr_runs(archive_dir, tmp_path, 2)
    merged = _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_REPLIED_FINDING)

    def _gh(repo: Any, endpoint: str, **kw: Any) -> Any:
        if "/pulls/2" in endpoint or "/2/" in endpoint or endpoint.endswith("/2"):
            raise git_ops.RateLimitError("exhausted")
        return merged(repo, endpoint, **kw)

    cache_dir = tmp_path / "c"
    config = HarvestConfig(archive_dir=archive_dir, cache_dir=cache_dir)
    summary = await _services(config, github=_gh).run()
    assert summary["aborted"] == 1
    done = BackfillCache(cache_dir=cache_dir, inner=_gh).completed_sessions()
    assert "s1" in done and "s2" not in done

def test_github_retry_uses_explicit_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    sequence: list[Any] = [git_ops.RateLimitError("x"), git_ops.RateLimitError("x"), {"ok": True}]

    def _inner(*args: Any, **kwargs: Any) -> Any:
        value = sequence.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(git_ops, "gh_api", _inner)
    assert harvest._github_with_retry("o/r", "endpoint", auth=git_ops.INHERIT_GITHUB_AUTH, backoff_sleep=slept.append,
    ) == {"ok": True}
    assert slept == [30.0, 30.0]

def test_harvest_services_binds_explicit_github_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    auth = git_ops.StaticGitHubAuth({"GH_TOKEN": "test-token"})
    seen_auth: list[git_ops.GitHubAuth] = []

    def _github(*args: Any, **kwargs: Any) -> dict[str, bool]:
        seen_auth.append(kwargs["auth"])
        return {"ok": True}

    monkeypatch.setattr(git_ops, "gh_api", _github)
    config = HarvestConfig(archive_dir=tmp_path / "archive")
    services = HarvestPass(config, github_auth=auth)

    assert services.github("o/r", "endpoint") == {"ok": True}
    assert seen_auth == [auth]


def _resolve_repo_with_services(
    tmp_path: Path, *, clone_cache: Path, source_path: Path | None = None, remote_url: str | None = None,
    repo_slug: str | None = None,
) -> Path | None:
    row = HarvestRow.from_mapping({
            "session_id": "adapter-session", "archive_path": str(tmp_path / "archive" / "runs" / "adapter-session"),
            "source_path": str(source_path) if source_path is not None else None, "remote_url": remote_url,
            "repo_slug": repo_slug,
        }, row_number=1,
    )
    services = HarvestPass(HarvestConfig(archive_dir=tmp_path / "archive", repo_clone_root=clone_cache))
    return services.resolve_repo(row, console=create_console())


def _resolve_org_repo(tmp_path: Path, *, clone_cache: Path | None = None, source_path: Path | None = None,
) -> Path | None:
    """Resolve a row pointing at the canonical ``org/repo`` remote via the standard cache."""
    return _resolve_repo_with_services(tmp_path, clone_cache=clone_cache or tmp_path / "cache", source_path=source_path,
        remote_url="https://github.com/org/repo.git", repo_slug="org/repo",
    )


def test_resolve_repo_for_row_prefers_source_path(tmp_path: Path) -> None:
    source = tmp_path / "source_repo"
    source.mkdir()
    (source / ".git").mkdir()
    result = _resolve_org_repo(tmp_path, source_path=source)
    assert result == source

def test_resolve_repo_for_row_clones_when_source_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    cache = tmp_path / "cache"

    def fake_clone(url: str, target: Path, token: str | None = None, **kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / ".git").mkdir()

    monkeypatch.setattr(git_ops, "clone_with_token", fake_clone)
    result = _resolve_org_repo(tmp_path)
    assert result == cache / "org" / "repo"

def test_resolve_repo_for_row_fetches_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    cache = tmp_path / "cache"
    cached_repo = cache / "org" / "repo"
    cached_repo.mkdir(parents=True)
    (cached_repo / ".git").mkdir()

    fetched: list[Path] = []
    monkeypatch.setattr(git_ops, "fetch", lambda repo, remote="origin": fetched.append(repo))
    monkeypatch.setattr(git_ops, "clone_with_token", lambda *args, **kwargs: pytest.fail("should not clone"))
    result = _resolve_org_repo(tmp_path)
    assert result == cached_repo
    assert fetched == [cached_repo]

def test_resolve_repo_for_row_returns_none_when_no_remote(tmp_path: Path) -> None:
    result = _resolve_repo_with_services(tmp_path, clone_cache=tmp_path / "cache")
    assert result is None

def test_resolve_repo_for_row_clone_failure_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    monkeypatch.setattr(git_ops, "clone_with_token",
        lambda url, target, token=None, **kwargs: (_ for _ in ()).throw(GitError("network error")),
    )
    result = _resolve_org_repo(tmp_path)
    assert result is None

def test_resolve_repo_for_row_fetch_failure_returns_cached_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    cached_repo = cache / "org" / "repo"
    cached_repo.mkdir(parents=True)
    (cached_repo / ".git").mkdir()

    monkeypatch.setattr(git_ops, "fetch", lambda repo, remote="origin": (_ for _ in ()).throw(GitError("fetch failed")))
    result = _resolve_org_repo(tmp_path)
    assert result == cached_repo

@pytest.mark.parametrize(("session_id", "status_message"),
    [("s-transient-500", "gh: Internal Server Error (HTTP 500)"), ("s-merge-comments-404", "gh: Not Found (HTTP 404)")],
)
async def test_harvest_propagates_confirmed_merge_comment_fetch_error(
    tmp_path: Path, archive_dir: Any, session_id: str, status_message: str,
) -> None:
    """After PR existence is confirmed, comment-fetch errors must remain retryable.

    Even a comments 404 cannot mean the PR is absent. Dropping into local scoring
    and caching done would lose merge evidence and valid_at; keep the row in
    errors with no annotation or resume marker."""
    _seed_archived_deep_run(archive_dir, session_id)
    cache_dir = tmp_path / "c"

    def _gh_merge_ok_comments_fail(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if re.search(r"/pulls/\d+$", endpoint):
            return {"merged": True, "merged_at": "2026-02-01T00:00:00+00:00"}
        if endpoint.endswith("/comments") or endpoint.endswith("/reviews"):
            raise GitError(status_message)
        return {}

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=cache_dir)
    summary = await _services(config, github=_gh_merge_ok_comments_fail).run()

    assert summary["errors"] == 1  # comment-fetch error propagated, merge evidence not discarded
    assert latest_label_observation(archive_dir, session_id) is None  # not annotated
    # Resume contract: the failed row is NOT cached "done", so a later run retries it.
    done = BackfillCache(cache_dir=cache_dir, inner=_gh_merge_ok_comments_fail).completed_sessions()
    assert session_id not in done

async def test_harvest_keeps_labeled_row_when_reviewer_lookup_errors(tmp_path: Path, archive_dir: Any,) -> None:
    """Keep a resolved PR label when the auxiliary reviewer-prior lookup fails.

    Only the optional prior degrades; successful PR and comment evidence survives."""
    run_dir = _seed_archived_deep_run(archive_dir, "s-reviews-err")
    _write_findings(run_dir, _FP_A)

    def _gh_reviews_fail(repo: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/reviews"):
            raise GitError("gh: Internal Server Error (HTTP 500)")
        if endpoint.endswith("/comments"):
            # A resolved daydream thread: the evidence that makes the merge decisive.
            return _finding_comments(_FP_A, reply="applied")
        return {"merged": True, "merged_at": "2026-02-01T00:00:00+00:00"}

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh_reviews_fail).run()

    assert summary["errors"] == 0  # reviewer-lookup failure degraded, row not dropped
    obs = latest_label_observation(archive_dir, "s-reviews-err")
    assert obs is not None
    assert obs["pr_state"] == "merged"  # resolved PR outcome preserved (not degraded to local)
    row = query_runs(archive_dir, "session_id = ?", ("s-reviews-err",))[0]
    assert json.loads(row["outcome_labels"]) == ["accepted"]  # merged-PR label kept

async def test_harvest_degrades_benign_giterror_rows_instead_of_dropping(tmp_path: Path, archive_dir: Any,) -> None:
    # Eight PRs resolve as accepted; two benign GitError rows must degrade to local evidence and
    # still be annotated. pr_state=None proves they did not receive a fabricated unmerged PR rubric.
    _seed_pr_runs(archive_dir, tmp_path, 10, fingerprints=(_FP_A,))

    merged = _fake_gh(merged_at="2026-02-01T00:00:00+00:00", comments=_finding_comments(_FP_A, reply="applied"))

    def _gh(repo: str, endpoint: str, **kw: Any) -> Any:
        match = re.search(r"/pulls/(\d+)", endpoint)
        number = int(match.group(1)) if match else 0
        if number >= 9:
            raise GitError(f"gh: Not Found (HTTP 404) for PR {number}")
        return merged(repo, endpoint, **kw)

    config = HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c")
    summary = await _services(config, github=_gh).run()

    assert summary["aborted"] == 0  # the GitError rows did NOT abort the sweep
    assert summary["annotated"] == 10 and summary["errors"] == 0

    obs_pr = latest_label_observation(archive_dir, "s1")
    assert obs_pr is not None
    accepted_pr_state = obs_pr["pr_state"]
    assert accepted_pr_state == "merged"
    assert json.loads(query_runs(archive_dir, "session_id = ?", ("s1",))[0]["outcome_labels"]) == ["accepted"]
    for sid in ("s9", "s10"):
        degraded = latest_label_observation(archive_dir, sid)
        assert degraded is not None  # annotated, not dropped
        assert degraded["pr_state"] is None  # local-branch rubric, not a PR rubric
        # Without a resolvable clone, retain unknown instead of inventing a rejection.
        assert json.loads(query_runs(archive_dir, "session_id = ?", (sid,))[0]["outcome_labels"]) == []


def _raise_git_error_with_url(*args: object, **kwargs: object) -> None:
    raise GitError("git clone https://user:ghp_canaryfake123@github.com/o/r failed: boom")


def test_repo_resolution_warning_is_value_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(git_ops, "clone_with_token", _raise_git_error_with_url)
    _resolve_repo_with_services(
        tmp_path, clone_cache=tmp_path / "cache", remote_url="https://user:ghp_canaryfake123@github.com/o/r",
        repo_slug="o/r",
    )
    out = capsys.readouterr().out
    assert "ghp_canaryfake123" not in out
    assert "o/r" in out

def test_resolve_repo_never_clones_raw_archived_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    seen: list[str] = []

    def fake_clone(url: str, target: Path, token: str | None = None, **kwargs: object) -> None:
        seen.append(url)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".git").mkdir()

    monkeypatch.setattr(git_ops, "clone_with_token", fake_clone)
    assert _resolve_repo_with_services(
        tmp_path, clone_cache=tmp_path / "cache", remote_url="https://user:ghp_canaryfake123@github.com/o/r",
        repo_slug="o/r",
    ) == tmp_path / "cache" / "o" / "r"
    assert seen == ["https://github.com/o/r"]

@pytest.mark.parametrize(
    "remote_url", ["https://evil.example.com/o/r", "file:///tmp/evil"], ids=["untrusted-host", "file-scheme"],
)
def test_resolve_repo_fails_closed_before_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_url: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(git_ops, "clone_with_token", lambda url, target, **kwargs: calls.append(url))
    assert _resolve_repo_with_services(tmp_path, clone_cache=tmp_path / "cache", remote_url=remote_url, repo_slug="o/r",
    ) is None
    assert calls == []

def test_token_never_in_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAYDREAM_GIT_TOKEN", "ghp_envtokfake123")
    seen: list[list[str]] = []

    def _recording_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _recording_run)
    _resolve_repo_with_services(
        tmp_path, clone_cache=tmp_path / "cache", remote_url="https://github.com/o/r", repo_slug="o/r",
    )
    assert seen
    basic = base64.b64encode(b"x-access-token:ghp_envtokfake123").decode()
    for command in seen:
        joined = " ".join(command)
        assert "ghp_envtokfake123" not in joined
        assert basic not in joined

def test_hub_import_rejects_unsanitized_affected_bundle(tmp_path: Path) -> None:
    """M18: an affected (credential-bearing) incoming bundle with no released
    derivative is quarantined locally and skipped — never imported raw."""

    archive_dir = tmp_path / "archive"
    incoming = archive_dir / "incoming" / "s1"
    incoming.mkdir(parents=True)
    (incoming / "manifest.json").write_text(
        json.dumps({"session_id": "s1", "git": {"remote_url": "https://user:ghp_canaryfake123@github.com/o/r"}})
    )
    result = sanitize.import_bundle(incoming, archive_dir)
    assert result.imported is False or result.quarantined
    assert (archive_dir / "quarantine" / "s1").is_dir()
    assert not incoming.exists()

def test_hub_import_accepts_clean_bundle_in_place(tmp_path: Path) -> None:
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

def test_hub_import_accepts_advisory_only_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Import an advisory-only uppercase KEY flag without quarantine.

    The name shape is not a credential; diagnostics must remain value-free."""

    archive_dir = tmp_path / "archive"
    incoming = archive_dir / "incoming" / "s3"
    incoming.mkdir(parents=True)
    (incoming / "manifest.json").write_text(json.dumps({
                "session_id": "s3", "git": {"remote_url": "https://github.com/o/r"},
                "notes": 'FEATURE_FLAG_OVERRIDE_KEY = "override_flag"',
            }
        )
    )
    pre = scan.scan_run_dir(incoming)
    assert pre.clean is False and pre.blocking is False  # advisory-only bundle

    result = sanitize.import_bundle(incoming, archive_dir)

    assert result.imported is True
    assert result.quarantined is False
    assert incoming.exists()
    assert not (archive_dir / "quarantine" / "s3").exists()
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "advisory" in out
    assert "env_var" in out
    assert "override_flag" not in out  # M11: never a matched value

def test_per_finding_resolution_round_trips_through_canonical_dict() -> None:
    r = PerFindingResolution(fingerprint="fp-1", comment_id=7, disposition="accepted",
        evidence=[{"reply_id": 1, "body_sha256": "abc"}], evidence_digest="d" * 32,
    )
    restored = resolution_from_dict(resolution_to_dict(r))
    assert restored == r  # canonical dict is the one round-trip shape

def test_harvest_row_preserves_absent_and_empty_fingerprints() -> None:

    common = {"session_id": "s1", "archive_path": "/tmp/archive"}
    absent = HarvestRow.from_mapping(common, row_number=1)
    empty = HarvestRow.from_mapping({**common, "findings_fingerprints": []}, row_number=2)

    assert "findings_fingerprints" not in absent.as_signal_row()
    assert empty.as_signal_row()["findings_fingerprints"] == []

def test_harvest_evidence_detaches_nested_signals_with_canonical_rubric() -> None:

    verdicts: list[dict[str, Any]] = [{"verdict": "consistent", "detail": {"source": "original"}}]
    commits = ["a1"]
    replies: list[dict[str, Any]] = [{"reply_id": 7, "context": {"state": "original"}}]
    rubric = Rubric(pr_merge=PRMergeSignal(True, "2026-02-01T00:00:00Z", "merged"),
        fix_applied=FixAppliedSignal("applied", 1, 1, commits), comment_resolution=CommentResolutionSignal(1, 1, 0),
        local_commit_applied=None, posterior_source="pr_review",
        per_finding_resolutions=[PerFindingResolution("f" * 64, 7, "accepted", replies, "d" * 32)],
    )
    expected = rubric.to_dict()
    evidence = HarvestEvidence(
        scoring_inputs=ScoringInputs(verdicts, True, 12), rubric=rubric, reviewer_logins=("reviewer",),
        pooled_prior=0.75, prior_n=2,
    )

    verdicts[0]["detail"]["source"] = "mutated"
    commits.append("b2")
    replies[0]["context"]["state"] = "mutated"

    stored_verdicts = evidence.scoring_inputs.verifier_verdicts
    assert stored_verdicts is not None
    assert stored_verdicts[0]["detail"]["source"] == "original"
    assert list(evidence.rubric.fix_applied.window_commits) == ["a1"]
    resolution = evidence.rubric.per_finding_resolutions
    assert resolution is not None
    assert resolution_to_dict(resolution[0])["evidence"] == [{"reply_id": 7, "context": {"state": "original"}}]
    assert evidence.rubric.to_dict() == expected

def test_per_finding_resolution_from_dict_fails_closed() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        resolution_from_dict({"disposition": "accepted", "evidence_digest": "d" * 32})
    with pytest.raises(ValueError, match="disposition"):
        resolution_from_dict({"fingerprint": "fp-1", "disposition": "banana"})
    with pytest.raises(ValueError, match="evidence_digest"):
        resolution_from_dict({"fingerprint": "fp-1", "disposition": "accepted"})

def test_rubric_to_dict_carries_full_per_finding_resolutions() -> None:
    """Persist full per-finding identity, evidence, and digests: materialization reads them from the SQLite
    rubric.
    """

    r = PerFindingResolution(fingerprint="fp-1", comment_id=7, disposition="accepted",
                             evidence=[{"reply_id": 1}], evidence_digest="d" * 32)
    rubric = Rubric(
        pr_merge=PRMergeSignal(True, None, "closed", False), fix_applied=FixAppliedSignal("applied", 1, 1, []),
        comment_resolution=CommentResolutionSignal(1, 1, 0), local_commit_applied=None, posterior_source="pr_review",
        per_finding_resolutions=[r],
    )
    d = rubric.to_dict()
    stored = d["per_finding_resolutions"]
    assert stored[0]["fingerprint"] == "fp-1"
    assert stored[0]["disposition"] == "accepted"
    assert stored[0]["evidence_digest"] == "d" * 32
    assert stored[0]["evidence"] == [{"reply_id": 1}]
    assert d["per_finding_outcomes"] == ["accepted"]

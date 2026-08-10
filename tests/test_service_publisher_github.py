"""Hermetic tests for the trusted GitHub Checks publisher (Plan 008 Step 5).

Covers: the publisher is the only Checks-write holder; before success it
revalidates live identity per ``target_kind`` (current PR head for ``pr_head``;
caller-supplied current exact candidate for ``merge_group``); a changed PR head
or replaced merge-group candidate fails closed; ``external_id`` binds to the
immutable job; a publisher failure is never turned into success; and repo-scope
mismatch is refused.

The GitHub API is mocked at the ``git_ops.gh_api`` seam (a fake publisher port);
no real ``gh`` CLI, model provider, or network is touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream import git_ops
from daydream.github_app import (
    GitHubChecksPublisher,
    LiveIdentity,
    github_pr_head_live_identity,
)
from daydream.service.models import ReviewTarget, SourceOfTruth, TargetKind
from daydream.service.publisher import PublishError, PublishReceipt, PublishRequest

CANDIDATE_SHA = "a" * 40
TREE = "b" * 40
BASE_SHA = "c" * 40
DIFF_DIGEST = "d" * 64
CONFIG_DIGEST = "e" * 64


def _pr_target(pr_number: int = 77, candidate_sha: str = CANDIDATE_SHA) -> ReviewTarget:
    return ReviewTarget(
        repo="acme/widgets",
        kind=TargetKind.PR_HEAD,
        candidate_sha=candidate_sha,
        candidate_tree=TREE,
        base_sha=BASE_SHA,
        pr_number=pr_number,
        merge_group_id=None,
        diff_digest=DIFF_DIGEST,
        config_source=SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=CONFIG_DIGEST),
        invalidation_id="job-1",
    )


def _merge_group_target(candidate_sha: str = CANDIDATE_SHA) -> ReviewTarget:
    return ReviewTarget(
        repo="acme/widgets",
        kind=TargetKind.MERGE_GROUP,
        candidate_sha=candidate_sha,
        candidate_tree=TREE,
        base_sha=BASE_SHA,
        pr_number=None,
        merge_group_id="mg-1",
        diff_digest=DIFF_DIGEST,
        config_source=SourceOfTruth(ref="refs/heads/main", sha=BASE_SHA, digest=CONFIG_DIGEST),
        invalidation_id="job-1",
    )


_DEFAULT = "__default_target__"


def _req(
    *,
    conclusion: str = "success",
    target: ReviewTarget | None | str = _DEFAULT,
    repo: str = "acme/widgets",
    external_id: str = "job-1",
    target_sha: str | None = None,
) -> PublishRequest:
    if target == _DEFAULT:
        target = _pr_target()
    assert target is None or isinstance(target, ReviewTarget)
    resolved_sha: str = (
        target_sha
        if target_sha is not None
        else (target.candidate_sha if target is not None else "") or ""
    )
    return PublishRequest(
        external_id=external_id,
        conclusion=conclusion,  # type: ignore[arg-type]
        summary="bounded summary",
        repo=repo,
        target_sha=resolved_sha,
        check_name="daydream/review",
        target=target,
    )


def _publisher(**overrides: Any) -> GitHubChecksPublisher:
    kwargs: dict[str, Any] = {
        "repo_dir": Path("/tmp"),
        "owner": "acme",
        "repo": "widgets",
        "check_name": "daydream/review",
    }
    kwargs.update(overrides)
    return GitHubChecksPublisher(**kwargs)


def _live_resolver(sha: str) -> object:
    def resolve(target: ReviewTarget) -> LiveIdentity:
        return LiveIdentity(sha=sha)

    return resolve


# --- pr_head live-identity revalidation --------------------------------------


def test_pr_head_success_revalidates_live_head_and_publishes() -> None:
    """Success revalidates the live PR head; matching head posts the check."""
    seen = {}

    def fake_gh_api(repo, endpoint, **kw):
        if "pulls" in endpoint:
            assert kw.get("idempotent") is True
            return {"head": {"sha": CANDIDATE_SHA}}
        assert "check-runs" in endpoint
        assert kw["method"] == "POST"
        seen["payload"] = kw["input_data"]
        return {"id": 42}

    publisher = _publisher()
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        receipt = publisher.publish(_req())

    assert receipt.external_id == "job-1"
    assert receipt.check_run_id == 42
    payload = seen["payload"]
    assert payload["name"] == "daydream/review"
    assert payload["head_sha"] == CANDIDATE_SHA
    assert payload["conclusion"] == "success"
    assert payload["external_id"] == "job-1"
    assert payload["output"]["summary"] == "bounded summary"


def test_pr_head_changed_fails_closed_no_publish() -> None:
    """A changed PR head cancels success; nothing is written."""
    gh_calls = []

    def fake_gh_api(repo, endpoint, **kw):
        gh_calls.append(endpoint)
        if "pulls" in endpoint:
            return {"head": {"sha": "9" * 40}}  # live head moved
        return {"id": 999}

    publisher = _publisher()
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        with pytest.raises(PublishError, match="stale"):
            publisher.publish(_req())

    # only the live-identity read happened; no check-run write was attempted
    assert gh_calls == ["/repos/acme/widgets/pulls/77"]


def test_pr_head_success_requires_target() -> None:
    """A success without a ReviewTarget is refused (cannot revalidate identity)."""
    with pytest.raises(PublishError, match="ReviewTarget"):
        _publisher().publish(_req(target=None, target_sha=CANDIDATE_SHA))


def test_pr_head_live_resolver_rejects_non_pr_head() -> None:
    resolver = github_pr_head_live_identity(Path("/tmp"), "acme", "widgets")
    with pytest.raises(PublishError, match="merg"):
        resolver(_merge_group_target())


# --- merge_group live-identity revalidation ----------------------------------


def test_merge_group_success_revalidates_supplied_candidate() -> None:
    """merge_group success uses the caller-supplied current exact candidate."""
    seen = {}

    def fake_gh_api(repo, endpoint, **kw):
        assert "check-runs" in endpoint
        seen["payload"] = kw["input_data"]
        return {"id": 7}

    publisher = _publisher(live_identity=_live_resolver(CANDIDATE_SHA))
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        receipt = publisher.publish(_req(target=_merge_group_target()))

    assert receipt.check_run_id == 7
    assert seen["payload"]["head_sha"] == CANDIDATE_SHA


def test_merge_group_replaced_candidate_fails_closed() -> None:
    """A replaced merge-group candidate means M1 can never authorize M2."""
    def fake_gh_api(repo, endpoint, **kw):
        raise AssertionError("must not reach check-run write")

    publisher = _publisher(live_identity=_live_resolver("1" * 40))  # live candidate replaced
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        with pytest.raises(PublishError, match="stale"):
            publisher.publish(_req(target=_merge_group_target()))


def test_merge_group_without_resolver_fails_closed() -> None:
    """merge_group requires a caller-supplied live identity; refusal fails closed."""
    with pytest.raises(PublishError, match="resolver"):
        _publisher().publish(_req(target=_merge_group_target()))


# --- external_id binding and failure handling ---------------------------------


def test_external_id_bound_to_immutable_job() -> None:
    """The check carries the immutable job id as external_id, never the candidate."""
    seen = {}

    def fake_gh_api(repo, endpoint, **kw):
        if "pulls" in endpoint:
            return {"head": {"sha": CANDIDATE_SHA}}
        seen["payload"] = kw["input_data"]
        return {"id": 1}

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        _publisher().publish(_req(external_id="job-immutable-9"))

    assert seen["payload"]["external_id"] == "job-immutable-9"


def test_publisher_failure_never_becomes_success() -> None:
    """A gh write failure surfaces as PublishError; retrying does not invent success."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 422")):
        with pytest.raises(PublishError):
            _publisher().publish(_req(conclusion="failure", target=_pr_target()))


def test_repo_scope_mismatch_refused() -> None:
    """Publishing to a repo outside the publisher's scope is refused fail-closed."""
    def fake_gh_api(*args, **kwargs):
        raise AssertionError("must not write check for a foreign repo")

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        with pytest.raises(PublishError, match="scope"):
            _publisher().publish(_req(repo="evil/widgets"))


def test_failure_conclusion_does_not_require_live_identity_validation() -> None:
    """Only SUCCESS revalidates identity; a failure conclusion just posts the failure."""
    seen = []

    def fake_gh_api(repo, endpoint, **kw):
        seen.append(endpoint)
        return {"id": 3}

    publisher = _publisher()
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        receipt = publisher.publish(_req(conclusion="failure", target=_pr_target()))

    assert receipt.check_run_id == 3
    # no live PR-head GET was issued for a failure conclusion
    assert "/pulls/" not in seen


def test_missing_repo_refused() -> None:
    with pytest.raises(PublishError, match="repo"):
        _publisher().publish(_req(repo=""))


def test_missing_candidate_sha_refused() -> None:
    done = {}

    def fake_gh_api(*args, **kwargs):
        done["called"] = True
        return {}

    target = _pr_target(candidate_sha="")
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        with pytest.raises(PublishError):
            _publisher().publish(_req(target=target, conclusion="failure"))
    assert "called" not in done  # refused before any write


def test_receipt_type_is_used() -> None:
    def fake_gh_api(repo, endpoint, **kw):
        if "pulls" in endpoint:
            return {"head": {"sha": CANDIDATE_SHA}}
        return {"id": 5}

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        receipt = _publisher().publish(_req())
    assert isinstance(receipt, PublishReceipt)
    assert receipt.external_id == "job-1"

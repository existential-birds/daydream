"""Tests for the immutable service job model (daydream/service/models.py)."""

from dataclasses import FrozenInstanceError
from typing import Literal

import pytest

from daydream.service.models import ReviewJobV1, ReviewTargetV1


def _target(
    *,
    target_kind: Literal["pr_head", "merge_group"] = "pr_head",
    repo: str = "acme/demo",
    candidate_sha: str = "a" * 40,
    candidate_tree_digest: str = "b" * 40,
    base_sha: str = "c" * 40,
    pr_numbers: tuple[int, ...] = (7,),
    merge_group_id: str | None = None,
    full_diff_digest: str = "d" * 64,
    protected_config_ref: str | None = None,
    protected_config_digest: str | None = None,
    invalidation_id: str = "inv-1",
) -> ReviewTargetV1:
    return ReviewTargetV1(
        target_kind=target_kind,
        repo=repo,
        candidate_sha=candidate_sha,
        candidate_tree_digest=candidate_tree_digest,
        base_sha=base_sha,
        pr_numbers=pr_numbers,
        merge_group_id=merge_group_id,
        full_diff_digest=full_diff_digest,
        protected_config_ref=protected_config_ref,
        protected_config_digest=protected_config_digest,
        invalidation_id=invalidation_id,
    )


def _job(
    *,
    target: ReviewTargetV1 | None = None,
    required_lenses: tuple[str, ...] = ("python",),
    round_num: int = 1,
    attempt: int = 1,
    deadline: str = "2030-01-01T00:00:00Z",
    created_at: str = "2030-01-01T00:00:00Z",
    effective_config_digest: str = "e" * 64,
    reviewer_bundle_digest: str = "f" * 64,
) -> ReviewJobV1:
    return ReviewJobV1(
        job_id="job-1",
        idempotency_key="idem-1",
        target=target or _target(),
        effective_config_digest=effective_config_digest,
        reviewer_bundle_digest=reviewer_bundle_digest,
        required_lenses=required_lenses,
        round=round_num,
        attempt=attempt,
        deadline=deadline,
        created_at=created_at,
    )


def test_pr_head_requires_pr_numbers() -> None:
    with pytest.raises(ValueError):
        _target(pr_numbers=())


def test_pr_head_rejects_merge_group_id() -> None:
    with pytest.raises(ValueError):
        _target(pr_numbers=(7,), merge_group_id="mg-1")


def test_merge_group_requires_merge_group_id() -> None:
    with pytest.raises(ValueError):
        _target(target_kind="merge_group", merge_group_id=None)


def test_merge_group_rejects_pr_numbers() -> None:
    with pytest.raises(ValueError):
        _target(target_kind="merge_group", merge_group_id="mg-1", pr_numbers=(7,))


def test_unknown_target_kind_rejected() -> None:
    with pytest.raises(ValueError):
        _target(target_kind="nightly")  # type: ignore[arg-type]


def test_candidate_sha_must_be_40_hex() -> None:
    with pytest.raises(ValueError):
        _target(candidate_sha="short")
    with pytest.raises(ValueError):
        _target(candidate_sha="z" * 40)


def test_full_diff_digest_must_be_64_hex() -> None:
    with pytest.raises(ValueError):
        _target(full_diff_digest="abc")


def test_valid_pr_head_target_builds() -> None:
    t = _target()
    assert t.target_kind == "pr_head"
    assert t.pr_numbers == (7,)
    assert t.merge_group_id is None


def test_valid_merge_group_target_builds() -> None:
    t = _target(target_kind="merge_group", merge_group_id="mg-1", pr_numbers=())
    assert t.merge_group_id == "mg-1"


def test_review_job_validates_round_and_attempt() -> None:
    with pytest.raises(ValueError):
        _job(round_num=0)
    with pytest.raises(ValueError):
        _job(attempt=0)


def test_review_job_requires_non_empty_lenses() -> None:
    with pytest.raises(ValueError):
        _job(required_lenses=())


def test_review_job_deadline_must_be_iso_utc() -> None:
    with pytest.raises(ValueError):
        _job(deadline="tomorrow")
    with pytest.raises(ValueError):
        _job(deadline="2030-01-01T00:00:00")  # naive, not UTC


def test_targets_and_jobs_are_immutable() -> None:
    t = _target()
    with pytest.raises(FrozenInstanceError):
        t.candidate_sha = "c" * 40  # type: ignore[misc]
    j = _job()
    with pytest.raises(FrozenInstanceError):
        j.round = 2  # type: ignore[misc]


def test_job_to_dict_and_from_dict_round_trip() -> None:
    j = _job()
    assert ReviewJobV1.from_dict(j.to_dict()) == j


def test_from_dict_rejects_unknown_fields() -> None:
    data = _job().to_dict()
    data["surprise"] = True
    with pytest.raises(ValueError):
        ReviewJobV1.from_dict(data)


def test_from_dict_rejects_missing_required_field() -> None:
    data = _job().to_dict()
    del data["target"]
    with pytest.raises(ValueError):
        ReviewJobV1.from_dict(data)


def test_from_dict_target_rejects_unknown_fields() -> None:
    data = _job().to_dict()
    data["target"]["surprise"] = True
    with pytest.raises(ValueError):
        ReviewJobV1.from_dict(data)

"""Immutable service job model (REVIEW_TARGET_V1 / DAYDREAM_SERVICE_V1 contract).

Frozen dataclasses only. No mutable state, no unexpected fields on
deserialization, no worktree/executor/lease identity — a job names what to
review and under which frozen bundle, never where to run it (the controller
owns execution identity in its separate ``ExecutionRef``).

Contract invariants (HARD, enforced in ``__post_init__`` and in the strict
``from_dict`` deserializers):

* ``pr_head`` targets require non-empty ``pr_numbers`` and no
  ``merge_group_id``; ``merge_group`` targets require a ``merge_group_id``
  and empty ``pr_numbers``; any other ``target_kind`` is rejected.
* ``candidate_sha`` / ``candidate_tree_digest`` / ``base_sha`` are exact
  40-hex git identifiers; ``full_diff_digest`` is the canonical full-diff
  SHA-256 (64-hex).
* ``round``/``attempt`` are >= 1; ``required_lenses`` is non-empty;
  ``deadline``/``created_at`` are ISO 8601 UTC timestamps.

``from_dict`` implements ``additionalProperties=False`` semantics: unknown
fields raise ``ValueError`` at every nesting level, and missing required
fields raise ``ValueError`` (never a bare ``KeyError``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

TargetKind = Literal["pr_head", "merge_group"]

_HEX_RE = re.compile(r"[0-9a-fA-F]+")


def _validate_hex(value: object, length: int, name: str) -> None:
    """Reject *value* unless it is a *length*-character hex string."""
    if not isinstance(value, str) or len(value) != length or not _HEX_RE.fullmatch(value):
        raise ValueError(f"{name} must be a {length}-character hex string")


def _validate_iso_utc(value: object, name: str) -> None:
    """Reject *value* unless it parses as a timezone-aware UTC ISO 8601 string."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO 8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 UTC string: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC: {value!r}")


def _required(data: dict[str, Any], key: str) -> Any:
    """Return ``data[key]``, raising ``ValueError`` when the field is absent."""
    if key not in data:
        raise ValueError(f"missing required field {key!r}")
    return data[key]


@dataclass(frozen=True)
class ReviewTargetV1:
    """The exact, immutable review target.

    Attributes:
        target_kind: ``"pr_head"`` (a pull-request head) or ``"merge_group"``
            (an opaque merge-group identity).
        repo: ``owner/repo`` slug.
        candidate_sha: Exact 40-hex candidate commit SHA (the review target).
        candidate_tree_digest: Exact tree digest of the candidate commit.
        base_sha: Exact base commit SHA the diff is computed against.
        pr_numbers: PR numbers relevant to a ``pr_head`` target; must be
            non-empty for ``pr_head`` and empty for ``merge_group``.
        merge_group_id: Opaque merge-group identity; required for
            ``merge_group`` and forbidden for ``pr_head``.
        full_diff_digest: Canonical SHA-256 (64-hex) of the full
            ``base_sha..candidate_sha`` diff.
        protected_config_ref: Ref/SHA the trusted review config came from, or
            None when no protected config source applies.
        protected_config_digest: Digest of the protected config content.
        invalidation_id: Opaque invalidate-on-next-job identifier so a fresh
            target can provably supersede a stale one.
    """

    target_kind: TargetKind
    repo: str
    candidate_sha: str
    candidate_tree_digest: str
    base_sha: str
    full_diff_digest: str
    invalidation_id: str
    pr_numbers: tuple[int, ...] = ()
    merge_group_id: str | None = None
    protected_config_ref: str | None = None
    protected_config_digest: str | None = None

    def __post_init__(self) -> None:
        if self.target_kind not in ("pr_head", "merge_group"):
            raise ValueError(
                f"unknown target_kind {self.target_kind!r}; expected 'pr_head' or 'merge_group'"
            )
        if not isinstance(self.repo, str) or "/" not in self.repo or self.repo.startswith("/"):
            raise ValueError(f"repo must be an 'owner/repo' slug, got {self.repo!r}")
        _validate_hex(self.candidate_sha, 40, "candidate_sha")
        _validate_hex(self.candidate_tree_digest, 40, "candidate_tree_digest")
        _validate_hex(self.base_sha, 40, "base_sha")
        _validate_hex(self.full_diff_digest, 64, "full_diff_digest")
        if not isinstance(self.invalidation_id, str) or not self.invalidation_id:
            raise ValueError("invalidation_id must be a non-empty string")

        if self.target_kind == "pr_head":
            if not isinstance(self.pr_numbers, tuple) or not self.pr_numbers:
                raise ValueError("pr_head target requires non-empty pr_numbers")
            if not all(isinstance(n, int) and n > 0 for n in self.pr_numbers):
                raise ValueError("pr_numbers must be positive integers")
            if self.merge_group_id is not None:
                raise ValueError("pr_head target must not carry a merge_group_id")
        else:
            if not isinstance(self.merge_group_id, str) or not self.merge_group_id:
                raise ValueError("merge_group target requires a non-empty merge_group_id")
            if self.pr_numbers:
                raise ValueError("merge_group target must have empty pr_numbers")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (lists for tuples)."""
        return {
            "target_kind": self.target_kind,
            "repo": self.repo,
            "candidate_sha": self.candidate_sha,
            "candidate_tree_digest": self.candidate_tree_digest,
            "base_sha": self.base_sha,
            "pr_numbers": list(self.pr_numbers),
            "merge_group_id": self.merge_group_id,
            "full_diff_digest": self.full_diff_digest,
            "protected_config_ref": self.protected_config_ref,
            "protected_config_digest": self.protected_config_digest,
            "invalidation_id": self.invalidation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewTargetV1:
        """Deserialize strictly: unknown fields and missing required fields raise."""
        unknown = set(data) - _TARGET_KEYS
        if unknown:
            raise ValueError(f"unexpected fields on ReviewTargetV1: {sorted(unknown)}")
        pr_numbers = _required(data, "pr_numbers")
        if isinstance(pr_numbers, list):
            pr_numbers = tuple(pr_numbers)
        return cls(
            target_kind=_required(data, "target_kind"),
            repo=_required(data, "repo"),
            candidate_sha=_required(data, "candidate_sha"),
            candidate_tree_digest=_required(data, "candidate_tree_digest"),
            base_sha=_required(data, "base_sha"),
            pr_numbers=pr_numbers,
            merge_group_id=data.get("merge_group_id"),
            full_diff_digest=_required(data, "full_diff_digest"),
            invalidation_id=_required(data, "invalidation_id"),
            protected_config_ref=data.get("protected_config_ref"),
            protected_config_digest=data.get("protected_config_digest"),
        )


@dataclass(frozen=True)
class ReviewJobV1:
    """An immutable, validated service review job wrapping one target.

    Attributes:
        job_id: Controller-side job identifier.
        idempotency_key: Dedup key; identical jobs re-run to one artifact.
        target: The exact review target (see :class:`ReviewTargetV1`).
        effective_config_digest: Digest of the trusted effective config the
            reviewer bundle ran under.
        reviewer_bundle_digest: Digest of the immutable reviewer bundle.
        required_lenses: Every lens that MUST complete; non-empty.
        round: Logical full-review round, >= 1.
        attempt: Logical attempt within the round, >= 1.
        deadline: ISO 8601 UTC deadline.
        created_at: ISO 8601 UTC creation timestamp.
    """

    job_id: str
    idempotency_key: str
    target: ReviewTargetV1
    effective_config_digest: str
    reviewer_bundle_digest: str
    required_lenses: tuple[str, ...]
    round: int
    attempt: int
    deadline: str
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        if not isinstance(self.effective_config_digest, str) or not self.effective_config_digest:
            raise ValueError("effective_config_digest must be a non-empty string")
        if not isinstance(self.reviewer_bundle_digest, str) or not self.reviewer_bundle_digest:
            raise ValueError("reviewer_bundle_digest must be a non-empty string")
        if (
            not isinstance(self.required_lenses, tuple)
            or not self.required_lenses
            or not all(isinstance(lens, str) and lens for lens in self.required_lenses)
        ):
            raise ValueError("required_lenses must be a non-empty tuple of non-empty strings")
        if not isinstance(self.round, int) or self.round < 1:
            raise ValueError("round must be an integer >= 1")
        if not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be an integer >= 1")
        _validate_iso_utc(self.deadline, "deadline")
        _validate_iso_utc(self.created_at, "created_at")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (target nested, lenses as a list)."""
        return {
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "target": self.target.to_dict(),
            "effective_config_digest": self.effective_config_digest,
            "reviewer_bundle_digest": self.reviewer_bundle_digest,
            "required_lenses": list(self.required_lenses),
            "round": self.round,
            "attempt": self.attempt,
            "deadline": self.deadline,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewJobV1:
        """Deserialize strictly: unknown fields and missing required fields raise.

        ``additionalProperties=False`` semantics apply at the job level and,
        via :meth:`ReviewTargetV1.from_dict`, at the nested target level.
        """
        unknown = set(data) - _JOB_KEYS
        if unknown:
            raise ValueError(f"unexpected fields on ReviewJobV1: {sorted(unknown)}")
        target = _required(data, "target")
        if not isinstance(target, dict):
            raise ValueError("target must be an object")
        required_lenses = _required(data, "required_lenses")
        if isinstance(required_lenses, list):
            required_lenses = tuple(required_lenses)
        return cls(
            job_id=_required(data, "job_id"),
            idempotency_key=_required(data, "idempotency_key"),
            target=ReviewTargetV1.from_dict(target),
            effective_config_digest=_required(data, "effective_config_digest"),
            reviewer_bundle_digest=_required(data, "reviewer_bundle_digest"),
            required_lenses=required_lenses,
            round=_required(data, "round"),
            attempt=_required(data, "attempt"),
            deadline=_required(data, "deadline"),
            created_at=_required(data, "created_at"),
        )


_TARGET_KEYS = frozenset(ReviewTargetV1.__dataclass_fields__)
_JOB_KEYS = frozenset(ReviewJobV1.__dataclass_fields__)

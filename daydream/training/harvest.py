"""Harvest pass — assemble immutable bronze signals into reward inputs.

The harvest pass is the single deferred *annotate* step of the corpus
pipeline: it reads an archived run's immutable bronze artifacts, reduces
them to a :class:`~daydream.training.reward.ScoringInputs`, derives the
outcome label, scores a :class:`~daydream.training.reward.RewardBreakdown`
or :class:`~daydream.training.reward.PosteriorBreakdown`, and appends one
bitemporal annotation. The stored ``composite_reward`` is the *pure intrinsic*
composite (C5): the posterior false-positive axis is a sibling field carried on
:class:`~daydream.training.reward.PosteriorBreakdown`, never folded into the
composite. There is no separate "labeling" step: a single annotate pass writes
label + reward together.

This module carries the bronze-signal assembly step, the per-run annotation
builder, and the :func:`run_harvest` orchestrator that walks the archive index
and appends one fresh annotation generation per run.

Signal sources (all under the archived run directory):

* ``deep/recommendation-verdicts.json`` — the ``verdicts`` list produced by
  the recommendation-verification stage (verdict shape mirrors
  :data:`daydream.phases.RECOMMENDATION_VERDICTS_SCHEMA`).
* ``deep/stack-*-records.json`` — per-stack finding records (shape mirrors
  the reader in :mod:`daydream.eval.analyzer`).
* ``review-output.md`` (root) falling back to ``deep/review-output.md`` —
  the char-count length proxy (matching the back-compat fallback in the
  former exporter).

Failure-propagation rules:

* Absent verdicts ⇒ ``verifier_verdicts=None`` with the format gate intact.
  Expected for a shallow run and, after the verify relocation, a declined
  deep run that skipped recommendation verification at the apply-fixes gate.
* A *present* verdicts/records file that is malformed JSON ⇒ caught as
  :class:`json.JSONDecodeError` and surfaced as ``format_valid=False``;
  assembly never crashes on bad data.
* ``grounding_rate`` is read from the indexed manifest row
  (``row["grounding_rate"]``), never re-derived here.
* ``length`` is the documented review-output char-count proxy, ``None`` when
  no review output exists.

Evidence acquisition and annotation reduction:

* :func:`acquire_harvest_evidence` performs posterior, reviewer, prior, and
  bronze acquisition through one explicit :class:`HarvestServices` value. A
  benign PR-merge-status failure (fork PR 404, unpushed-SHA 422) degrades to the
  local posterior; exhausted rate limits reach the orchestrator.
* :func:`build_annotation` consumes only a validated row and frozen evidence,
  derives the outcome label, applies prior sufficiency, scores the reward, and
  returns a frozen :class:`AnnotationPayload` without I/O.
* ``valid_at`` is the earliest qualifying decisive-evidence timestamp — the
  ``created_at`` of a reply supporting an ``accepted``/``rejected`` disposition
  (M12) — falling back to the PR merge timestamp when no decisive evidence
  exists, and ``None`` for non-PR/local rows (the write layer collapses
  ``None`` → ``observed_at``).

Orchestrator (:func:`run_harvest`):

* **Idempotent and re-runnable:** every indexed run is considered on every
  pass, but the write layer dedups on ``(evidence_sha, labeler_policy_version,
  reply_evidence_digest, labels, has_posterior)`` — a re-harvest with unchanged
  evidence is a no-op (counted in ``skipped``). A ``LABELER_POLICY_VERSION`` or
  reply-evidence change alters the dedup key and so appends a fresh
  generation, letting older ``as_of`` pins still resolve their original
  generation. Only the ``cache``/``dry_run`` paths otherwise suppress writes.
* **Per-row error isolation:** an exception on one row counts in ``errors`` and
  does not derail subsequent rows. Configuration errors (missing
  ``archive_dir``) raise before the loop begins.
* **Capture-time ``base_sha``:** materialized into the manifest when missing
  (the only fallible git I/O of the annotate pass lives here, not in the pure
  build-corpus projection).

The rubric-assembly helpers remain harvest-owned; stateful operations cross the
per-run services boundary.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.archive.git_safe import _DEFAULT_HOSTS, normalize_remote_url
from daydream.archive.index import (
    append_label_observation,
    query_runs,
    reviewer_set_penalty_prior,
    set_run_pr_link,
)
from daydream.git_ops import GitError, GitHubAuth, RateLimitError
from daydream.training import labeler_versions, reward
from daydream.training._immutable_json import thaw_json
from daydream.training.backfill_cache import BackfillCache
from daydream.training.base_sha import materialize_base_sha
from daydream.training.harvest_types import BaseShaStatus, HarvestEvidence, HarvestRow
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    index_pr_review_comments,
    local_commit_applied_signal,
    per_finding_resolution_signal,
    pr_link_signal,
    pr_merge_signal,
    reviewer_logins_signal,
)
from daydream.training.reward import ScoringInputs, score_trajectory
from daydream.training.rubric import Rubric, derive_outcome_label
from daydream.trajectory import redact_text as _redact_text
from daydream.ui import create_console, print_warning

_VERDICTS_FILE = "recommendation-verdicts.json"
"""Bronze artifact (under ``deep/``) carrying the ``verdicts`` list."""

_RECORDS_GLOB = "stack-*-records.json"
"""Bronze per-stack finding-record artifacts (under ``deep/``)."""

_REVIEW_OUTPUT_FILE = "review-output.md"
"""Length-proxy artifact; at the run root for shallow runs, under ``deep/`` for deep runs."""

_PRIOR_SUFFICIENCY_THRESHOLD = 10
"""Minimum pooled prior-run count for the empirical reviewer-set mean penalty to
graduate from the ``0.5`` maximum-entropy default to the observed pooled mean
(spec C4). Below this, ``outcome_prior`` is left ``None`` (the reducer applies the
``0.5`` default), though ``outcome_prior_n`` still records the pooled count for audit."""


class HarvestServices(Protocol):
    """All stateful acquisition and persistence used by one harvest pass."""

    @property
    def archive_dir(self) -> Path: ...

    @property
    def progress_path(self) -> Path | None: ...

    def query_rows(self, session_filter: str | None) -> Sequence[Mapping[str, Any]]: ...

    def completed_sessions(self) -> set[str]: ...

    def resolve_repo(self, row: HarvestRow, *, console: Console) -> Path | None: ...

    def materialize_base_sha(
        self,
        row: HarvestRow,
        repo_clone: Path | None,
        *,
        console: Console,
    ) -> BaseShaStatus: ...

    def github(self, repo: str, endpoint: str, **kwargs: Any) -> Any: ...

    def reviewer_prior(
        self,
        logins: tuple[str, ...],
        *,
        before_valid_at: str,
        exclude_session: str,
        repo_slug: str | None,
    ) -> tuple[float | None, int]: ...

    def set_pr_link(self, row: HarvestRow, number: int, repo: str) -> None: ...

    def read_scoring_inputs(self, row: HarvestRow) -> ScoringInputs: ...

    def read_recorded_fingerprints(self, row: HarvestRow) -> tuple[str, ...]: ...

    def fix_applied(
        self,
        row: HarvestRow,
        *,
        changed_files: tuple[str, ...],
        repo_clone: Path,
    ) -> FixAppliedSignal: ...

    def local_commit_applied(
        self,
        row: HarvestRow,
        *,
        repo_clone: Path,
    ) -> LocalCommitAppliedSignal: ...

    def append_annotation(self, row: HarvestRow, payload: AnnotationPayload) -> bool: ...

    def mark_session_done(self, session_id: str) -> None: ...

    def now_iso(self) -> str: ...

    def backoff_sleep(self, seconds: float) -> None: ...

    async def sleep_between_rows(self, seconds: float) -> None: ...


def _read_review_output(run_dir: Path) -> str | None:
    r"""Return the review-output text, or ``None`` when absent.

    Tries ``review-output.md`` at the run root first (shallow-loop layout),
    then ``deep/review-output.md`` (deep-mode layout), mirroring the former
    exporter's back-compat fallback order.

    Returns:
        The text of the first review-output file found, or ``None`` when
        neither location exists. Non-``FileNotFoundError`` ``OSError``\s
        propagate to the caller.
    """
    for candidate in (run_dir / _REVIEW_OUTPUT_FILE, run_dir / "deep" / _REVIEW_OUTPUT_FILE):
        try:
            return candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return None


def _read_review_output_length(run_dir: Path) -> int | None:
    """Return the review-output char count, or ``None`` when absent.

    Delegates to :func:`_read_review_output`; see that function for the
    fallback order and error semantics.
    """
    text = _read_review_output(run_dir)
    return len(text) if text is not None else None


def assemble_scoring_inputs(run_dir: Path, row: HarvestRow) -> ScoringInputs:
    """Reduce one run's bronze artifacts to intrinsic :class:`ScoringInputs`.

    Reads the structured bronze artifacts under ``run_dir/deep`` and the
    review-output length proxy, combining them with the indexed
    ``grounding_rate`` into the capture-time signals the reward reducer
    consumes. Absent verdicts yield ``verifier_verdicts=None`` and leave the
    format gate intact — expected for a shallow run and, after the verify
    relocation, a declined deep run whose recommendation verification was
    skipped at the apply-fixes gate. A present-but-malformed structured
    artifact sets ``format_valid=False`` without raising.

    Args:
        run_dir: The archived run directory (bronze bundle root).
        row: The indexed manifest row; ``row["grounding_rate"]`` supplies the
            grounding axis (``None`` when unavailable).

    Returns:
        A :class:`ScoringInputs` with the verdicts list (or ``None``), the
        passed-through grounding rate, the format-validity gate, and the
        char-count length proxy (or ``None``).
    """
    deep_dir = run_dir / "deep"

    verifier_verdicts: list[dict[str, Any]] | None = None
    format_valid = True

    verdicts_path = deep_dir / _VERDICTS_FILE
    try:
        data = json.loads(verdicts_path.read_text(encoding="utf-8"))
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if isinstance(verdicts, list):
            verifier_verdicts = verdicts
    except FileNotFoundError:
        # No structured verdicts; nothing failed to parse. Expected for a
        # shallow run and, after the verify relocation, a declined deep run
        # that skipped recommendation verification at the apply-fixes gate.
        pass
    except json.JSONDecodeError:
        # Present but malformed ⇒ format gate floors.
        format_valid = False

    # A present-but-malformed records file also trips the format gate.
    if deep_dir.is_dir():
        for records_path in sorted(deep_dir.glob(_RECORDS_GLOB)):
            try:
                json.loads(records_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except json.JSONDecodeError:
                format_valid = False

    return ScoringInputs(
        verifier_verdicts=verifier_verdicts,
        grounding_rate=row.grounding_rate,
        format_valid=format_valid,
        length=_read_review_output_length(run_dir),
    )


# Bounded rate-limit backoff for the gh seam (honors parsed Retry-After, capped
# at _MAX_BACKOFF_SEC, falling back to _DEFAULT_BACKOFF_SEC when absent).
_DEFAULT_BACKOFF_SEC = 30.0
_MAX_BACKOFF_SEC = 120.0
_MAX_RATE_LIMIT_RETRIES = 5


def _github_with_retry(
    repo: str,
    endpoint: str,
    *,
    auth: GitHubAuth,
    backoff_sleep: Callable[[float], None],
    **kwargs: Any,
) -> Any:
    """Proxy to :func:`daydream.git_ops.gh_api` keyed by ``repo`` slug.

    The PR posterior signal extractors call ``gh_api(repo, endpoint, **kwargs)``
    with ``repo`` as a slug string (``"owner/name"``). :func:`git_ops.gh_api`
    takes a ``Path`` as its first argument because it uses ``cwd=repo`` for the
    shell-out. We adapt by using ``Path(".")`` — ``gh api`` works from any cwd
    because it authenticates against the GitHub host configured in ``gh auth``,
    not the local repo.

    On a :class:`~daydream.git_ops.RateLimitError`, we sleep with bounded
    backoff (``min(retry_after, _MAX_BACKOFF_SEC)``, falling back to
    ``_DEFAULT_BACKOFF_SEC`` only when ``retry_after`` is absent, so an explicit
    ``Retry-After: 0`` hint is preserved) and
    retry up to ``_MAX_RATE_LIMIT_RETRIES`` times; if the limit is still
    exhausted, the :class:`RateLimitError` propagates so the orchestrator can
    abort cleanly while preserving its resume marker.

    Limitation: ``repo`` is accepted for API compatibility but is not used to
    resolve the GitHub host; all requests go to the single host configured in
    ``gh auth`` (typically ``github.com``). Mixing repos from different GitHub
    hosts in a single harvest run would silently use the wrong host.
    """
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        try:
            return git_ops.gh_api(Path("."), endpoint, **kwargs, auth=auth)
        except RateLimitError as exc:
            if attempt == _MAX_RATE_LIMIT_RETRIES - 1:
                raise
            retry_after = exc.retry_after if exc.retry_after is not None else _DEFAULT_BACKOFF_SEC
            backoff = min(retry_after, _MAX_BACKOFF_SEC)
            print_warning(
                create_console(),
                f"harvest: GitHub rate limit hit; retrying in {backoff:.0f}s "
                f"(attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES - 1})",
            )
            backoff_sleep(backoff)
    # Unreachable: the loop either returns or raises on the final attempt.
    raise RuntimeError("rate-limit retry loop exited without returning")  # pragma: no cover


# Rubric assembly — PR vs local branch.


_FIX_APPLIED_STUB = FixAppliedSignal(
    verdict="unknown",
    hunks_applied=0,
    hunks_total=0,
    window_commits=[],
)
"""Returned when the fix-applied cascade cannot run (missing recommended.patch
/ diff.patch, empty changed_files, or any subprocess error). The rubric still
carries the field for schema stability; outcome derivation does not depend on
it for the PR-review path."""


def _safe_fix_applied(
    row: HarvestRow,
    *,
    services: HarvestServices,
    changed_files: tuple[str, ...],
    repo_clone: Path,
) -> FixAppliedSignal:
    """Run :func:`fix_applied_signal`, swallowing missing-data errors.

    The cascade reads ``recommended.patch`` (falling back to ``diff.patch``
    for legacy archives). When the archive directory is missing (older runs,
    dry fixtures), or when ``changed_files`` is empty, return
    :data:`_FIX_APPLIED_STUB`.
    """
    if not changed_files:
        return _FIX_APPLIED_STUB
    try:
        return services.fix_applied(
            row,
            changed_files=changed_files,
            repo_clone=repo_clone,
        )
    except (FileNotFoundError, OSError, GitError):
        return _FIX_APPLIED_STUB


# HTTP statuses meaning the PR/commit is genuinely absent (fork/deleted PR 404,
# unpushed-SHA 422) so a row may degrade to its local posterior; every other gh
# failure is transient and must propagate so resume retries, not mislabel (#166).
_BENIGN_PR_ABSENCE_STATUSES = (404, 422)


def _is_benign_pr_absence(exc: GitError) -> bool:
    """Return ``True`` when a ``gh`` ``GitError`` means the PR is genuinely absent.

    Classification is on the HTTP status embedded in the ``gh`` failure message
    (``... (HTTP 404)``); a failure with no recognizable status is treated as
    transient (not benign), so it propagates rather than silently degrading.
    """
    match = re.search(r"\bHTTP (\d{3})\b", str(exc))
    return match is not None and int(match.group(1)) in _BENIGN_PR_ABSENCE_STATUSES


def _build_rubric_pr(
    row: HarvestRow,
    *,
    services: HarvestServices,
    github: Callable[..., Any],
    repo_clone: Path,
    pr_merge: PRMergeSignal | None = None,
    changed_files: tuple[str, ...],
    pr_author_logins: frozenset[str] = frozenset(),
    review_author_logins: frozenset[str] = frozenset(),
) -> Rubric:
    """Compose all four signals for a row that originated from a PR.

    Args:
        pr_merge: Pre-fetched :class:`PRMergeSignal`. When supplied
            (already resolved by the caller before the catch boundary),
            ``pr_merge_signal`` is not called again. When ``None`` it is
            fetched here as before.
        changed_files: Pre-decoded ``changed_files`` for ``row`` (the caller
            already decoded the JSON column via ``_row_changed_files``).
        pr_author_logins: Logins whose replies count as PR-author judgment
            under the M6 gate (passed through to
            :func:`~daydream.training.labeler_signals.per_finding_resolution_signal`).
        review_author_logins: Logins whose replies count as formal-review
            judgment under the M6 gate.
    """
    signal_row = row.as_signal_row()
    if pr_merge is None:
        pr_merge = pr_merge_signal(signal_row, gh_api=github)
    # Fetch + index the PR's review comments once; both resolution signals
    # consume this index instead of each hitting the /comments endpoint.
    recorded_fingerprints = services.read_recorded_fingerprints(row)
    comment_threads = index_pr_review_comments(
        signal_row,
        gh_api=github,
        session_fingerprints=list(recorded_fingerprints),
    )
    comments = comment_resolution_signal(signal_row, gh_api=github, threads=comment_threads)
    fix = _safe_fix_applied(
        row,
        services=services,
        changed_files=changed_files,
        repo_clone=repo_clone,
    )
    rubric = Rubric(
        pr_merge=pr_merge,
        fix_applied=fix,
        comment_resolution=comments,
        local_commit_applied=None,
        posterior_source="pr_review",
    )
    if recorded_fingerprints:
        per_finding = per_finding_resolution_signal(
            signal_row,
            recorded_fingerprints=list(recorded_fingerprints),
            gh_api=github,
            threads=comment_threads,
            pr_author_logins=pr_author_logins,
            review_author_logins=review_author_logins,
        )
        rubric = replace(rubric, per_finding_resolutions=list(per_finding))
    return rubric


def _build_rubric_local(
    row: HarvestRow,
    *,
    services: HarvestServices,
    repo_clone: Path,
    clone_resolved: bool = False,
) -> Rubric:
    """Compose signals for a PR-less row (local-branch posterior).

    When ``clone_resolved`` is ``False`` no real git working tree was
    obtained for the row (the orchestrator passes the archive dir as a
    placeholder), so the local-commit check cannot distinguish "no follow-up
    commit applied the fix" from "we could not look". Forcing ``"unknown"``
    avoids mislabeling such a row ``"rejected"``.
    """
    # Invariant: the local-commit posterior is valid ONLY for PR-less runs. A
    # degraded PR row's merge evidence was merely unavailable, so emit "unknown"
    # rather than risk a "rejected" false negative.
    if row.is_pr or not clone_resolved:
        local = LocalCommitAppliedSignal(verdict="unknown")
    else:
        try:
            local = services.local_commit_applied(
                row,
                repo_clone=repo_clone,
            )
        except (FileNotFoundError, OSError):
            local = LocalCommitAppliedSignal(verdict="unknown")
    pr_merge = PRMergeSignal(merged=False, merged_at=None)
    comments = CommentResolutionSignal(total=0, replied=0, unresolved=0)
    return Rubric(
        pr_merge=pr_merge,
        fix_applied=_FIX_APPLIED_STUB,
        comment_resolution=comments,
        local_commit_applied=local,
        posterior_source="local_branch",
    )


def _pr_state_for_rubric(rubric: Rubric) -> str | None:
    """Map a PR-review rubric to a sqlite ``pr_state`` discriminator.

    For local-branch rubrics returns ``None`` so the column reflects "no PR
    associated". An unmerged PR preserves its live GitHub ``state`` (M11) —
    ``open`` stays ``open`` rather than being collapsed to ``closed``.
    """
    if rubric.posterior_source != "pr_review":
        return None
    if rubric.pr_merge.merged:
        return "merged"
    return rubric.pr_merge.state if rubric.pr_merge.state in ("open", "closed") else "closed"


# Per-run annotation builder


@dataclass(frozen=True)
class AnnotationPayload:
    """One run's bitemporal annotation, ready to persist (no DB writes here).

    Attributes:
        labels: Outcome labels (``[]`` when the derived label is
            ``"unknown"``, else a single-element list).
        pr_state: sqlite ``pr_state`` discriminator (``"merged"``/``"closed"``
            for PR rows, ``None`` for local-branch rows).
        valid_at: The valid-time of the posterior outcome — the earliest
            qualifying decisive-evidence timestamp (M12), falling back to the
            PR merge timestamp when no decisive evidence exists, and ``None``
            for non-PR/local rows (the write layer collapses ``None`` →
            ``observed_at``).
        reward_version: The :data:`daydream.training.reward.REWARD_VERSION`
            observed at scoring time.
        reward_json: ``json.dumps`` of the full
            :meth:`~daydream.training.reward.RewardBreakdown.to_dict` (the
            :class:`~daydream.training.reward.PosteriorBreakdown` variant on the
            mapped-label path) so re-projection has every axis, including the
            posterior sibling fields when present.
        composite_reward: The cached *pure intrinsic* composite scalar
            (correctness + grounding − length penalty); the posterior penalty is
            never folded in (C5). ``None`` when uncomputable.
        evidence_sha: The run's ``head_sha`` (the evidence anchor for the
            posterior signals), or ``None``.
        rubric_json: ``json.dumps`` of the posterior rubric, or ``None``.
        reviewer_logins: The human GitHub accounts whose review/reply outcomes
            seeded the posterior axis — captured at harvest time (irreproducible
            later) and persisted. ``[]`` for local/non-PR rows.
        has_posterior: Population discriminator — ``True`` when the scored
            breakdown is a :class:`~daydream.training.reward.PosteriorBreakdown`,
            i.e. a ``pr_review`` row whose maintainer outcome label mapped to a
            penalty. ``local_branch`` rows keep their label but are **not**
            posterior evidence (a local commit is not a maintainer acting in a
            PR), so they carry ``False`` and no ``posterior_cost``.
        reply_classifier_version: The
            :data:`~daydream.training.labeler_versions.REPLY_CLASSIFIER_VERSION`
            that produced the per-finding dispositions (M13 version axis).
        reply_evidence_digest: Stable digest over the session's combined reply
            evidence (M14 dedup input), or ``None`` when no reply evidence
            exists.
    """

    labels: list[str]
    pr_state: str | None
    valid_at: str | None
    reward_version: str
    reward_json: str
    composite_reward: float | None
    evidence_sha: str | None
    rubric_json: str | None
    reviewer_logins: list[str]
    has_posterior: bool
    reply_classifier_version: str | None = None
    reply_evidence_digest: str | None = None


def _degrade_to_local(
    row: HarvestRow,
    *,
    services: HarvestServices,
    repo_clone: Path,
    clone_resolved: bool,
) -> tuple[Any, None, list[str], None, int]:
    """Build a local-branch rubric and return the degraded posterior state.

    Used when a PR-path fetch fails benignly (fork PR 404, unpushed-SHA 422)
    or when the row has no PR at all.  Returns a 5-tuple
    ``(rubric, valid_at, reviewer_logins, outcome_prior, prior_n)`` with the
    non-PR defaults so callers can unpack uniformly.
    """
    rubric = _build_rubric_local(
        row,
        services=services,
        repo_clone=repo_clone,
        clone_resolved=clone_resolved,
    )
    return rubric, None, [], None, 0


def acquire_harvest_evidence(
    row: HarvestRow,
    *,
    services: HarvestServices,
    repo_resolution: Path | None,
    base_sha_status: BaseShaStatus,
    valid_at_override: str | None = None,
    github: Callable[..., Any] | None = None,
) -> HarvestEvidence:
    """Acquire one row's complete external evidence in established order."""
    github_api = services.github if github is None else github
    repo_clone = repo_resolution or services.archive_dir
    if row.is_pr:
        changed_files = row.changed_files
        try:
            pr_merge = pr_merge_signal(
                row.as_signal_row(),
                gh_api=github_api,
            )
        except RateLimitError:
            raise
        except GitError as exc:
            if not _is_benign_pr_absence(exc):
                raise
            rubric, _valid_at, reviewer_logins, pooled_prior, prior_n = _degrade_to_local(
                row,
                services=services,
                repo_clone=repo_clone,
                clone_resolved=repo_resolution is not None,
            )
        else:
            try:
                reviewer_logins = reviewer_logins_signal(
                    row.as_signal_row(),
                    gh_api=github_api,
                )
            except RateLimitError:
                raise
            except GitError:
                reviewer_logins = []
            rubric = _build_rubric_pr(
                row,
                services=services,
                github=github_api,
                repo_clone=repo_clone,
                pr_merge=pr_merge,
                changed_files=changed_files,
                pr_author_logins=(
                    frozenset({pr_merge.author_login}) if pr_merge.author_login else frozenset()
                ),
                review_author_logins=frozenset(reviewer_logins),
            )
            valid_at = _decisive_evidence_valid_at(rubric)
            if valid_at is None and rubric.pr_merge.merged:
                valid_at = rubric.pr_merge.merged_at
            pooled_prior, prior_n = services.reviewer_prior(
                tuple(reviewer_logins),
                before_valid_at=valid_at or services.now_iso(),
                exclude_session=row.session_id,
                repo_slug=row.repo_slug,
            )
    else:
        rubric, _valid_at, reviewer_logins, pooled_prior, prior_n = _degrade_to_local(
            row,
            services=services,
            repo_clone=repo_clone,
            clone_resolved=repo_resolution is not None,
        )

    # Preserve the established boundary: bronze reads occur after posterior,
    # reviewer, and prior acquisition.
    scoring_inputs = services.read_scoring_inputs(row)
    return HarvestEvidence(
        scoring_inputs=scoring_inputs,
        rubric=rubric,
        reviewer_logins=tuple(reviewer_logins),
        pooled_prior=pooled_prior,
        prior_n=prior_n,
        repo_resolution=repo_resolution,
        base_sha_status=base_sha_status,
        valid_at_override=valid_at_override,
    )


def build_annotation(row: HarvestRow, evidence: HarvestEvidence) -> AnnotationPayload:
    """Purely reduce validated row and immutable evidence into an annotation."""
    rubric = evidence.rubric
    outcome_label = derive_outcome_label(rubric)
    labels = [outcome_label] if outcome_label != "unknown" else []
    valid_at = None
    if rubric.posterior_source == "pr_review":
        valid_at = _decisive_evidence_valid_at(rubric)
        if valid_at is None and rubric.pr_merge.merged:
            valid_at = rubric.pr_merge.merged_at
    if evidence.valid_at_override is not None:
        valid_at = evidence.valid_at_override

    # Only a maintainer acting on a real PR is posterior evidence; a local commit
    # containing the recommended lines is a weaker tier and must not enter the
    # posterior population (the label is still recorded on `labels`).
    posterior_feedback = outcome_label if rubric.posterior_source == "pr_review" else None
    outcome_prior = (
        evidence.pooled_prior if evidence.prior_n >= _PRIOR_SUFFICIENCY_THRESHOLD else None
    )
    rb = score_trajectory(
        evidence.scoring_inputs,
        pr_feedback=posterior_feedback,
        outcome_prior=outcome_prior,
        outcome_prior_n=evidence.prior_n,
    )

    if rb.reward_version != reward.REWARD_VERSION:
        raise RuntimeError(
            f"non-canonical reward_version {rb.reward_version!r} cannot be written to canonical storage"
            f" (expected {reward.REWARD_VERSION!r})"
        )

    return AnnotationPayload(
        labels=labels,
        pr_state=_pr_state_for_rubric(rubric),
        valid_at=valid_at,
        reward_version=rb.reward_version,
        reward_json=json.dumps(rb.to_dict()),
        composite_reward=rb.composite,
        evidence_sha=row.head_sha,
        rubric_json=json.dumps(rubric.to_dict()),
        reviewer_logins=list(evidence.reviewer_logins),
        has_posterior=isinstance(rb, reward.PosteriorBreakdown),
        reply_classifier_version=labeler_versions.REPLY_CLASSIFIER_VERSION,
        reply_evidence_digest=_reply_evidence_digest(rubric),
    )

def _decisive_evidence_valid_at(rubric: Rubric) -> str | None:
    """Earliest qualifying decisive-evidence timestamp from the rubric (M12).

    Scans the per-finding resolutions' persisted evidence for ``created_at``
    stamps of replies that support the resolution's ``accepted``/``rejected``
    disposition: an entry counts only when its author qualified (the
    ``reason`` is not ``excluded:*``) AND the reply's own ``classifier_label``
    matches the disposition — so an earlier qualifying-but-ambiguous reply
    never pulls ``valid_at`` before the reply that actually decided the
    finding. Malformed entries (missing/blank ``created_at``) are skipped for
    the min computation; when no decisive timestamp exists the caller falls
    back to the merge time or ``None`` — never a fabricated timestamp.
    """
    stamps: list[str] = []
    for resolution in rubric.per_finding_resolutions or []:
        if resolution.disposition not in ("accepted", "rejected"):
            continue
        for entry in resolution.evidence:
            if not isinstance(entry, Mapping):
                continue
            reason = entry.get("reason")
            if not isinstance(reason, str) or reason.startswith("excluded:"):
                continue
            if entry.get("classifier_label") != resolution.disposition:
                continue
            created_at = entry.get("created_at")
            if isinstance(created_at, str) and created_at:
                stamps.append(created_at)
    return min(stamps) if stamps else None


def _reply_evidence_digest(rubric: Rubric) -> str | None:
    """Stable digest over the session's combined reply evidence (M14).

    ``None`` when no reply evidence was collected, so a digest-less row never
    collides with a digested one under the versioned dedup key.
    """
    evidence = [
        thaw_json(entry)
        for resolution in rubric.per_finding_resolutions or []
        for entry in resolution.evidence
    ]
    return labeler_versions.reply_evidence_digest(evidence) if evidence else None


# Repo resolution — source_path first, then identity-based clone (issue #981):
# the archived remote_url is only ever normalized to a credential-free HTTPS
# identity; hosts outside the allowlist fail closed.


def _resolve_repo_for_row(
    row: HarvestRow,
    clone_cache: Path | None,
    *,
    fetched_repos: set[Path] | None = None,
    console: Console | None = None,
) -> Path | None:
    """Resolve a local repo working tree for a manifest row.

    Priority:
        1. ``row["source_path"]`` when it exists on disk with a ``.git`` dir.
        2. Clone cache: ``clone_cache/<owner>/<repo>/`` — fetch if present, clone if not.
        3. ``None`` when no source is available.

    Clone/fetch failures are caught and logged; they never block harvest.

    Args:
        row: An indexed manifest row (supplies ``source_path``, ``remote_url``, ``repo_slug``).
        clone_cache: Root directory for cached clones, or ``None`` to skip cloning.
    """
    source_path = row.source_path
    if source_path and (source_path / ".git").exists():
        return source_path

    remote_url = row.remote_url
    repo_slug = row.repo_slug
    if (
        not isinstance(remote_url, str)
        or not isinstance(repo_slug, str)
        or not remote_url
        or not repo_slug
        or clone_cache is None
    ):
        return None

    parts = repo_slug.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None

    cached_repo = clone_cache / parts[0] / parts[1]
    try:
        if (cached_repo / ".git").exists():
            if fetched_repos is None or cached_repo not in fetched_repos:
                git_ops.fetch(cached_repo)
                if fetched_repos is not None:
                    fetched_repos.add(cached_repo)
        else:
            cached_repo.parent.mkdir(parents=True, exist_ok=True)
            # Issue #981: never clone the archived raw URL. Normalize it to a
            # credential-free HTTPS identity and fail closed on untrusted
            # hosts or unparseable input. Token, if any, travels out-of-band.
            identity, canonical = normalize_remote_url(
                remote_url, allowed_hosts=_DEFAULT_HOSTS
            )
            if identity is None or canonical is None:
                return None
            token = os.environ.get("DAYDREAM_GIT_TOKEN")
            git_ops.clone_with_token(canonical, cached_repo, token, blobless=True)
    except (GitError, OSError) as exc:
        redacted = _redact_text(str(exc))
        print_warning(
            console or create_console(),
            f"harvest: repo resolution failed for {repo_slug}: {type(exc).__name__}: {redacted}",
        )
        if not (cached_repo / ".git").exists():
            return None
    return cached_repo


def _materialize_base_sha_if_missing(
    row: HarvestRow,
    repo_clone: Path | None,
    *,
    console: Console | None = None,
) -> BaseShaStatus:
    """Opportunistically backfill ``code_context.base_sha`` into the manifest.

    Only acts when ``manifest.json`` exists AND ``repo_clone`` is available.
    Any failure is swallowed (opportunistic), leaving ``base_sha`` as ``None``.
    """
    manifest_path = row.archive_path / "manifest.json"
    if not manifest_path.exists():
        return "unavailable"
    if repo_clone is None:
        return "unavailable"
    try:
        resolved = materialize_base_sha(manifest_path, repo_clone=repo_clone)
    except (OSError, json.JSONDecodeError, GitError) as exc:
        print_warning(
            console or create_console(),
            f"harvest: base_sha backfill failed for {manifest_path}: {type(exc).__name__}: {exc}",
        )
        return "failed"
    if resolved is None:
        return "unavailable"
    return "available"


# Orchestrator — idempotent (evidence-hash dedup), re-runnable, per-row isolation


@dataclass(frozen=True)
class HarvestConfig:
    """Configuration for a single :func:`run_harvest` invocation.

    Attributes:
        archive_dir: Path to the daydream archive root (contains ``index.db``).
        dry_run: When ``True``, the loop builds annotations but suppresses the
            write to ``label_observations`` and the resume log.
        cache_dir: Optional directory backing
            :class:`~daydream.training.backfill_cache.BackfillCache`. When
            ``None``, ``gh_api`` calls hit the network on every row.
        repo_clone_root: Optional root under which per-repo clones live (used
            by the fix-applied / local-commit cascades). Falls back to
            ``cache_dir / 'repos'`` when unset (or ``None`` if ``cache_dir``
            is also unset).
        session_filter: Optional ``session_id`` prefix to restrict the queue.
        gh_request_spacing_sec: Sleep duration between rows to spread
            ``gh api`` calls under GitHub's secondary rate limits.
    """

    archive_dir: Path
    dry_run: bool = False
    cache_dir: Path | None = None
    repo_clone_root: Path | None = None
    session_filter: str | None = None
    gh_request_spacing_sec: float = 0.8


class _ProductionHarvestServices:
    """Per-run adapters for archive, repository, GitHub, cache, and clocks."""

    def __init__(self, config: HarvestConfig, github_auth: GitHubAuth) -> None:
        self._config = config
        self._github_auth = github_auth
        self._cache: BackfillCache | None = None
        self._fetched_repos: set[Path] = set()

    @property
    def archive_dir(self) -> Path:
        return self._config.archive_dir

    @property
    def progress_path(self) -> Path | None:
        return (
            self._config.cache_dir / "progress.jsonl"
            if self._config.cache_dir is not None
            else None
        )

    def _uncached_github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        return _github_with_retry(
            repo,
            endpoint,
            auth=self._github_auth,
            backoff_sleep=self.backoff_sleep,
            **kwargs,
        )

    def _cache_instance(self) -> BackfillCache | None:
        if self._config.cache_dir is None:
            return None
        if self._cache is None:
            self._cache = BackfillCache(
                cache_dir=self._config.cache_dir,
                inner=self._uncached_github,
            )
        return self._cache

    def query_rows(self, session_filter: str | None) -> Sequence[Mapping[str, Any]]:
        if not self.archive_dir.exists():
            raise FileNotFoundError(f"archive_dir does not exist: {self.archive_dir}")
        if session_filter:
            return query_runs(
                self.archive_dir,
                "session_id LIKE ? || '%'",
                (session_filter,),
            )
        return query_runs(self.archive_dir)

    def completed_sessions(self) -> set[str]:
        cache = self._cache_instance()
        return cache.completed_sessions() if cache is not None else set()

    def resolve_repo(self, row: HarvestRow, *, console: Console) -> Path | None:
        clone_cache = self._config.repo_clone_root or (
            self._config.cache_dir / "repos" if self._config.cache_dir else None
        )
        return _resolve_repo_for_row(
            row,
            clone_cache,
            fetched_repos=self._fetched_repos,
            console=console,
        )

    def materialize_base_sha(
        self,
        row: HarvestRow,
        repo_clone: Path | None,
        *,
        console: Console,
    ) -> BaseShaStatus:
        return _materialize_base_sha_if_missing(row, repo_clone, console=console)

    def github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        cache = self._cache_instance()
        if cache is not None:
            return cache(repo, endpoint, **kwargs)
        return self._uncached_github(repo, endpoint, **kwargs)

    def reviewer_prior(
        self,
        logins: tuple[str, ...],
        *,
        before_valid_at: str,
        exclude_session: str,
        repo_slug: str | None,
    ) -> tuple[float | None, int]:
        return reviewer_set_penalty_prior(
            self.archive_dir,
            list(logins),
            before_valid_at=before_valid_at,
            exclude_session=exclude_session,
            repo_slug=repo_slug,
        )

    def set_pr_link(self, row: HarvestRow, number: int, repo: str) -> None:
        set_run_pr_link(self.archive_dir, row.session_id, number, repo)

    def read_scoring_inputs(self, row: HarvestRow) -> ScoringInputs:
        return assemble_scoring_inputs(row.archive_path, row)

    def read_recorded_fingerprints(self, row: HarvestRow) -> tuple[str, ...]:
        if row.findings_fingerprints is not None:
            return row.findings_fingerprints
        try:
            data = json.loads((row.archive_path / "findings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ()
        findings = data.get("findings") if isinstance(data, dict) else None
        if not isinstance(findings, list):
            return ()
        return tuple(
            str(finding["fingerprint"])
            for finding in findings
            if isinstance(finding, dict) and "fingerprint" in finding
        )

    @staticmethod
    def _file_at(repo: Path, path: str, sha: str) -> str:
        try:
            return git_ops.show(repo, sha, path).decode("utf-8", errors="replace")
        except GitError:
            return ""

    def fix_applied(
        self,
        row: HarvestRow,
        *,
        changed_files: tuple[str, ...],
        repo_clone: Path,
    ) -> FixAppliedSignal:
        return fix_applied_signal(
            row.as_signal_row(),
            changed_files=list(changed_files),
            repo_clone=repo_clone,
            diff_fetcher=git_ops.diff_name_only,
            commits_in_window_fetcher=lambda repo, head, base: list(
                reversed(git_ops.log_shas_since(repo, head, base))
            ),
            file_at_fetcher=self._file_at,
        )

    def local_commit_applied(
        self,
        row: HarvestRow,
        *,
        repo_clone: Path,
    ) -> LocalCommitAppliedSignal:
        return local_commit_applied_signal(
            row.as_signal_row(),
            repo_clone=repo_clone,
            commits_since_fetcher=lambda repo, branch, since: git_ops.log_shas(
                repo,
                branch,
                since=since,
            ),
            file_at_fetcher=self._file_at,
        )

    def append_annotation(self, row: HarvestRow, payload: AnnotationPayload) -> bool:
        return append_label_observation(
            self.archive_dir,
            row.session_id,
            labels=payload.labels,
            pr_state=payload.pr_state,
            labeler_version=labeler_versions.LABELER_POLICY_VERSION,
            evidence_sha=payload.evidence_sha,
            rubric_json=payload.rubric_json,
            valid_at=payload.valid_at,
            reward_version=payload.reward_version,
            reward_json=payload.reward_json,
            composite_reward=payload.composite_reward,
            reviewer_logins=payload.reviewer_logins,
            has_posterior=payload.has_posterior,
            reply_classifier_version=payload.reply_classifier_version,
            reply_evidence_digest=payload.reply_evidence_digest,
        )

    def mark_session_done(self, session_id: str) -> None:
        cache = self._cache_instance()
        if cache is not None:
            cache.mark_session_done(session_id)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def backoff_sleep(seconds: float) -> None:
        time.sleep(seconds)

    @staticmethod
    async def sleep_between_rows(seconds: float) -> None:
        await anyio.sleep(seconds)


def make_harvest_services(
    config: HarvestConfig,
    *,
    github_auth: GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> HarvestServices:
    """Create side-effect-free production adapters for one harvest pass."""
    return _ProductionHarvestServices(config, github_auth)


async def run_harvest(
    config: HarvestConfig,
    *,
    services: HarvestServices,
) -> dict[str, int]:
    """Validate the archive queue, acquire evidence, and persist annotations."""
    requested_archive = config.archive_dir.resolve()
    services_archive = services.archive_dir.resolve()
    if requested_archive != services_archive:
        raise ValueError(
            f"harvest archive ownership mismatch: requested {requested_archive}, services {services_archive}"
        )
    raw_queue = services.query_rows(config.session_filter)
    console = create_console()
    queue: list[HarvestRow] = []
    invalid_rows = 0
    for row_number, raw in enumerate(raw_queue, start=1):
        try:
            queue.append(HarvestRow.from_mapping(raw, row_number=row_number))
        except ValueError as exc:
            invalid_rows += 1
            print_warning(console, f"harvest: {exc}")

    summary = {
        "considered": invalid_rows,
        "annotated": 0,
        "would_annotate": 0,
        "skipped": 0,
        "errors": invalid_rows,
        "aborted": 0,
    }
    if not queue:
        return summary

    # BackfillCache creation and progress reads remain lazy until every raw row
    # has crossed the validation boundary. Invalid rows still count even when a
    # valid sibling is removed by the resume filter.
    done = services.completed_sessions()
    queue = [row for row in queue if row.session_id not in done]
    summary["considered"] += len(queue)

    for row in queue:
        try:
            repo_resolution = services.resolve_repo(row, console=console)
            base_sha_status = services.materialize_base_sha(
                row,
                repo_resolution,
                console=console,
            )
            # Re-link orphan runs before acquiring their posterior. Persistence
            # remains at this exact boundary, so a later row failure keeps the
            # durable link as before.
            if not row.is_pr:
                try:
                    link = pr_link_signal(row.as_signal_row(), gh_api=services.github)
                except RateLimitError:
                    raise
                except GitError as exc:
                    if not _is_benign_pr_absence(exc):
                        raise
                    print_warning(
                        console,
                        f"harvest: PR link lookup failed for session {row.session_id}; "
                        f"degrading to local-branch posterior: {type(exc).__name__}: {exc}",
                    )
                    link = None
                if link is not None:
                    number, slug = link
                    if not config.dry_run:
                        services.set_pr_link(row, number, slug)
                    row = replace(row, pr_number=number, pr_repo=slug)

            evidence = acquire_harvest_evidence(
                row,
                services=services,
                repo_resolution=repo_resolution,
                base_sha_status=base_sha_status,
            )
            payload = build_annotation(row, evidence)
            if config.dry_run:
                summary["would_annotate"] += 1
            else:
                if services.append_annotation(row, payload):
                    summary["annotated"] += 1
                else:
                    summary["skipped"] += 1
                # A successful append or dedup is complete for resume. A raised
                # write never advances the marker.
                services.mark_session_done(row.session_id)
        except RateLimitError:
            summary["aborted"] = 1
            resume_marker = services.progress_path
            abort_msg = (
                "harvest: GitHub rate limit exhausted; aborting cleanly. "
                f"Resume from {resume_marker} by re-running with the same --cache-dir."
                if resume_marker is not None
                else "harvest: GitHub rate limit exhausted; aborting cleanly."
            )
            print_warning(console, abort_msg)
            break
        except Exception as exc:  # noqa: BLE001 - per-row isolation by design
            summary["errors"] += 1
            print_warning(
                console,
                f"harvest: session {row.session_id} failed: {type(exc).__name__}: {exc}",
            )
            continue

        if not config.dry_run:
            await services.sleep_between_rows(config.gh_request_spacing_sec)

    return summary

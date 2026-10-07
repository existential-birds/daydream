"""Harvest immutable bronze signals into bitemporal label and reward annotations.

HarvestServices owns acquisition and persistence; build_annotation reduces a
validated row and frozen evidence without I/O. Posterior false-positive cost
stays beside the pure intrinsic composite. Qualifying decisive reply time pins
PR outcomes, falling back to merge time; local outcomes have no valid-time pin.

Each pass reads selected frozen run records. The record store deduplicates
unchanged evidence and policy; changed evidence appends immutable observations.
Dry-run suppresses all writes. A disposable response cache can skip completed
rows; exhausted rate limits abort without losing already persisted evidence.
PR/base/license enrichment is append-only and leaves captured records sealed.
Exact reply text is retained separately from semantic evidence and its digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.archive.git_safe import _DEFAULT_HOSTS, normalize_remote_url
from daydream.dataset import LocalRecordStore
from daydream.git_ops import GitError, GitHubAuth, RateLimitError
from daydream.json_utils import canonical_json
from daydream.training import labeler_versions, reward
from daydream.training.adjudication.snapshot import record_evidence_digest
from daydream.training.backfill_cache import BackfillCache
from daydream.training.harvest_types import HarvestEvidence, HarvestRow
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingResolution,
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
from daydream.training.license_evidence import GithubLicenseResolver, LicenseEvidenceError
from daydream.training.record_evidence import section_value, validate_output_path
from daydream.training.reward import FP_PENALTY_MAP, ScoringInputs, score_trajectory
from daydream.training.rubric import Rubric, derive_outcome_label
from daydream.trajectory import redact_text as _redact_text
from daydream.ui import create_console, print_warning

_PRIOR_SUFFICIENCY_THRESHOLD = 10
"""Minimum pooled prior-run count for the empirical reviewer-set mean penalty to
graduate from the ``0.5`` maximum-entropy default to the observed pooled mean
(spec C4). Below this, ``outcome_prior`` is left ``None`` (the reducer applies the
``0.5`` default), though ``outcome_prior_n`` still records the pooled count for audit."""


class HarvestServices(Protocol):
    """All stateful acquisition and persistence used by one harvest pass."""

    @property
    def store_dir(self) -> Path: ...

    @property
    def snapshot_id(self) -> str: ...

    @property
    def dry_run(self) -> bool: ...

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
    ) -> None: ...

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
    """Call gh_api from the current directory with bounded rate-limit backoff.

    Honor Retry-After, including zero; use the default only when absent and cap
    delays at _MAX_BACKOFF_SEC. Exhaustion propagates for resumable abort.
    repo is the fetcher-interface slug, not host selection: gh uses its configured
    host, so one harvest cannot mix repositories from different GitHub hosts."""
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
"""Returned when the fix-applied cascade cannot run (missing captured recommendation,
empty changed_files, or any subprocess error). The rubric still
carries the field for schema stability; outcome derivation does not depend on
it for the PR-review path."""


def _safe_fix_applied(
    row: HarvestRow,
    *,
    services: HarvestServices,
    changed_files: tuple[str, ...],
    repo_clone: Path,
) -> FixAppliedSignal:
    """Check recommended hunks, using the legacy diff fallback when required.

    Missing archive data or an empty changed-file set returns _FIX_APPLIED_STUB.
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
    """Recognize HTTP 404 as absent; unknown statuses propagate as transient failures."""
    match = re.search(r"\bHTTP (\d{3})\b", str(exc))
    return match is not None and int(match.group(1)) in _BENIGN_PR_ABSENCE_STATUSES


def _build_rubric_pr(
    row: HarvestRow,
    *,
    services: HarvestServices,
    github: Callable[..., Any],
    repo_clone: Path,
    pr_merge: PRMergeSignal,
    pr_author_logins: frozenset[str] = frozenset(),
    review_author_logins: frozenset[str] = frozenset(),
) -> Rubric:
    """Combine a fetched PR state with scoped comment, fix, and finding signals.

    Fetch comments once; PR/review author sets govern decisive reply eligibility."""
    signal_row = row.as_signal_row()
    # Fetch + index the PR's review comments once; both resolution signals
    # consume this index instead of each hitting the /comments endpoint.
    recorded_fingerprints = row.findings_fingerprints
    comment_threads = index_pr_review_comments(
        signal_row,
        gh_api=github,
        session_fingerprints=list(recorded_fingerprints),
    )
    comments = comment_resolution_signal(signal_row, gh_api=github, threads=comment_threads)
    fix = _safe_fix_applied(
        row,
        services=services,
        changed_files=row.changed_files,
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
    """Build local-branch signals, forcing unknown when no clone was resolved.

    An unresolved repository cannot establish that a fix was rejected.
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
        per_finding_resolutions=[
            PerFindingResolution(
                fingerprint=fingerprint,
                comment_id=None,
                disposition="missing",
                evidence_digest=labeler_versions.reply_evidence_digest([]),
            )
            for fingerprint in row.findings_fingerprints
        ],
    )


def _pr_state_for_rubric(rubric: Rubric) -> str | None:
    """Return None for local runs; preserve the live open/closed state for unmerged PRs."""
    if rubric.posterior_source != "pr_review":
        return None
    if rubric.pr_merge.merged:
        return "merged"
    return rubric.pr_merge.state if rubric.pr_merge.state in ("open", "closed") else "closed"


# Per-run annotation builder


@dataclass(frozen=True)
class AnnotationPayload:
    """One run's complete annotation, retained in a typed observation.

    Unknown labels become []; evidence_sha is the captured head. valid_at is
    decisive PR reply time, then merge time, or None for local outcomes (the
    writer maps None to observed_at). reward_json retains every score axis;
    composite_reward is intrinsic only and may be uncomputable. rubric_json and
    reviewer_logins preserve acquisition-time provenance. has_posterior is true
    only for mapped maintainer feedback on a PR, never a local applied commit.
    Reply classifier version and combined evidence digest pin the dedup axis;
    no reply evidence yields a None digest."""

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


def acquire_harvest_evidence(
    row: HarvestRow,
    *,
    services: HarvestServices,
    repo_resolution: Path | None,
) -> HarvestEvidence:
    """Acquire one row's complete external evidence in established order."""
    repo_clone = repo_resolution or services.store_dir
    pr_merge = None
    if row.is_pr:
        try:
            pr_merge = pr_merge_signal(row.as_signal_row(), gh_api=services.github)
        except RateLimitError:
            raise
        except GitError as exc:
            if not _is_benign_pr_absence(exc):
                raise

    reviewer_logins: list[str] = []
    pooled_prior = None
    prior_n = 0
    if pr_merge is None:
        rubric = _build_rubric_local(
            row,
            services=services,
            repo_clone=repo_clone,
            clone_resolved=repo_resolution is not None,
        )
    else:
        try:
            reviewer_logins = reviewer_logins_signal(row.as_signal_row(), gh_api=services.github)
        except RateLimitError:
            raise
        except GitError:
            pass
        rubric = _build_rubric_pr(
            row,
            services=services,
            github=services.github,
            repo_clone=repo_clone,
            pr_merge=pr_merge,
            pr_author_logins=frozenset({pr_merge.author_login}) if pr_merge.author_login else frozenset(),
            review_author_logins=frozenset(reviewer_logins),
        )
        pooled_prior, prior_n = services.reviewer_prior(
            tuple(reviewer_logins),
            before_valid_at=_rubric_valid_at(rubric) or services.now_iso(),
            exclude_session=row.session_id,
            repo_slug=row.repo_slug,
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
    )


def build_annotation(row: HarvestRow, evidence: HarvestEvidence) -> AnnotationPayload:
    """Purely reduce validated row and immutable evidence into an annotation."""
    rubric = evidence.rubric
    outcome_label = derive_outcome_label(rubric)
    labels = [outcome_label] if outcome_label != "unknown" else []
    valid_at = _rubric_valid_at(rubric)

    # Only a maintainer acting on a real PR is posterior evidence; a local commit
    # containing the recommended lines is a weaker tier and must not enter the
    # posterior population (the label is still recorded on `labels`).
    posterior_feedback = outcome_label if rubric.posterior_source == "pr_review" else None
    outcome_prior = evidence.pooled_prior if evidence.prior_n >= _PRIOR_SUFFICIENCY_THRESHOLD else None
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
        reply_evidence_digest=record_evidence_digest(
            [resolution.evidence for resolution in rubric.per_finding_resolutions or []]
        ),
    )


def _rubric_valid_at(rubric: Rubric) -> str | None:
    """PR evidence time, falling back to merge time; local labels have no pin."""
    if rubric.posterior_source != "pr_review":
        return None
    decisive = _decisive_evidence_valid_at(rubric)
    if decisive is not None:
        return decisive
    return rubric.pr_merge.merged_at if rubric.pr_merge.merged else None


def _decisive_evidence_valid_at(rubric: Rubric) -> str | None:
    """Return the earliest qualifying reply timestamp supporting a decisive disposition.

    Exclude excluded:* authors and classifier labels that differ from the finding's
    accepted/rejected outcome. Skip missing/blank timestamps; absent decisive time
    lets the caller fall back to merge time or None.
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
    """Prefer an existing source_path Git tree, then fetch/clone owner/repo in the cache.

    Return None without a source/cache. Log clone/fetch failures without blocking harvest.
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
            identity, canonical = normalize_remote_url(remote_url, allowed_hosts=_DEFAULT_HOSTS)
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


# Orchestrator — idempotent (evidence-hash dedup), re-runnable, per-row isolation


@dataclass(frozen=True)
class HarvestConfig:
    """Settings for one frozen-record harvest pass.

    Dry-run acquires fresh evidence without mutating records or clones,
    caches, or completion markers. cache_dir enables response caching/resume;
    repo_clone_root defaults to its repos/ child, or None without a cache.
    session_filter is a session-id prefix; gh_request_spacing_sec separates rows."""

    store_dir: Path
    snapshot_id: str
    dry_run: bool = False
    cache_dir: Path | None = None
    repo_clone_root: Path | None = None
    session_filter: str | None = None
    gh_request_spacing_sec: float = 0.8

    def __post_init__(self) -> None:
        for output in (self.cache_dir, self.repo_clone_root):
            if output is not None:
                validate_output_path(self.store_dir, output)


class _ProductionHarvestServices:
    """Per-run adapters for records, repository, GitHub, cache, and clocks."""

    def __init__(self, config: HarvestConfig, github_auth: GitHubAuth) -> None:
        self._config = config
        self._github_auth = github_auth
        self._cache: BackfillCache | None = None
        self._fetched_repos: set[Path] = set()
        self._records = LocalRecordStore(config.store_dir).read_snapshot(config.snapshot_id)
        self._runs = {run["run_id"]: run for run in self._records.runs}

    @property
    def store_dir(self) -> Path:
        return self._config.store_dir

    @property
    def snapshot_id(self) -> str:
        return self._config.snapshot_id

    @property
    def dry_run(self) -> bool:
        return self._config.dry_run

    @property
    def progress_path(self) -> Path | None:
        return self._config.cache_dir / "progress.jsonl" if self._config.cache_dir is not None else None

    def _uncached_github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        return _github_with_retry(
            repo,
            endpoint,
            auth=self._github_auth,
            backoff_sleep=self.backoff_sleep,
            **kwargs,
        )

    def _cache_instance(self) -> BackfillCache | None:
        if self._config.cache_dir is None or self._config.dry_run:
            return None
        if self._cache is None:
            self._cache = BackfillCache(
                cache_dir=self._config.cache_dir,
                inner=self._uncached_github,
            )
        return self._cache

    def query_rows(self, session_filter: str | None) -> Sequence[Mapping[str, Any]]:
        rows = []
        for run in sorted(self._records.runs, key=lambda r: r["run_id"]):
            if session_filter and not run["run_id"].startswith(session_filter):
                continue
            task = section_value(run, "original_task") or {}
            repo = task.get("repository") or {}
            revision = task.get("analyzed_revision") or {}
            pr = task.get("pr") or {}
            base = revision.get("pr_base_sha") or revision.get("merge_base_sha")
            context = run.get("provenance", {}).get("repository_context", {})
            for observation in sorted(
                self._records.eligible_observations,
                key=lambda o: (datetime.fromisoformat(o["observed_at"]), o["observation_id"]),
            ):
                if observation["run_id"] != run["run_id"] or observation["payload"]["type"] != "enrichment":
                    continue
                payload = observation["payload"]
                value = payload["evidence"]["value"] if payload["evidence"]["status"] == "available" else None
                if value and payload["kind"] == "pr":
                    pr = value
                if value and payload["kind"] == "base":
                    base = value["base_sha"]
            rows.append(
                {
                    "session_id": run["run_id"],
                    "repo_slug": repo.get("repo_slug"),
                    "remote_url": repo.get("remote_url"),
                    "pr_repo": pr.get("repo"),
                    "pr_number": pr.get("number"),
                    "branch": context.get("branch"),
                    "base_branch": context.get("base_branch"),
                    "source_path": context.get("source_path"),
                    "head_sha": revision.get("head_sha"),
                    "base_sha": base,
                    "changed_files": task.get("changed_files", []),
                    "findings_fingerprints": [
                        item["fingerprint"] for item in (section_value(run, "findings") or {}).get("items", [])
                    ],
                    "recommended_patch": (section_value(run, "recommended_patch") or {}).get("patch", ""),
                }
            )
        return rows

    def completed_sessions(self) -> set[str]:
        cache = self._cache_instance()
        return cache.completed_sessions() if cache is not None else set()

    def resolve_repo(self, row: HarvestRow, *, console: Console) -> Path | None:
        clone_cache = self._config.repo_clone_root or (
            self._config.cache_dir / "repos" if self._config.cache_dir else None
        )
        if self._config.dry_run:
            candidates = [row.source_path]
            if clone_cache is not None and row.repo_slug:
                candidates.append(clone_cache / row.repo_slug)
            return next((p for p in candidates if p is not None and (p / ".git").exists()), None)
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
    ) -> None:
        if not self._config.dry_run and row.repo_slug and row.head_sha:
            try:
                license_evidence = GithubLicenseResolver().resolve(row.repo_slug, repo_commit=row.head_sha)
                value = asdict(license_evidence) if license_evidence is not None else None
                evidence = {
                    "status": "available" if value is not None else "unproduced",
                    "value": value,
                    "reason": None if value is not None else "license_unavailable",
                }
            except LicenseEvidenceError:
                value = None
                evidence = {"status": "failed", "value": None, "reason": "license_acquisition_failed"}
                print_warning(console, "harvest: license acquisition failed; corpus admission remains unavailable")
            self._append_typed(row.session_id, {"type": "enrichment", "kind": "license", "evidence": evidence}, value)
        if self._config.dry_run or row.base_sha:
            return
        if repo_clone is None or not row.base_branch or not row.head_sha:
            return
        try:
            resolved = git_ops.merge_base(repo_clone, row.base_branch, row.head_sha)
        except GitError as exc:
            print_warning(console, f"harvest: base revision enrichment failed: {type(exc).__name__}")
            return
        if resolved is not None:
            self._append_enrichment(row.session_id, "base", {"base_sha": resolved})

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
        if not logins:
            return None, 0
        winners: dict[str, dict[str, Any]] = {}
        cutoff = datetime.fromisoformat(before_valid_at)
        for observation in self._records.eligible_observations:
            if observation["payload"]["type"] != "harvest-annotation" or observation["run_id"] == exclude_session:
                continue
            annotation = observation["payload"]["annotation"]
            task = section_value(self._runs[observation["run_id"]], "original_task") or {}
            if repo_slug is not None and (task.get("repository") or {}).get("repo_slug") != repo_slug:
                continue
            if datetime.fromisoformat(observation["valid_at"]) >= cutoff or not set(logins).intersection(
                annotation["reviewer_logins"]
            ):
                continue
            previous = winners.get(observation["run_id"])
            if previous is None or (
                datetime.fromisoformat(observation["observed_at"]),
                observation["observation_id"],
            ) > (
                datetime.fromisoformat(previous["observed_at"]),
                previous["observation_id"],
            ):
                winners[observation["run_id"]] = observation
        penalties = [
            FP_PENALTY_MAP[a["labels"][0]]
            for o in winners.values()
            if (a := o["payload"]["annotation"])["labels"] and a["labels"][0] in FP_PENALTY_MAP
        ]
        return (sum(penalties) / len(penalties), len(penalties)) if penalties else (None, 0)

    def _append_enrichment(self, run_id: str, kind: str, value: dict[str, Any]) -> None:
        self._append_typed(
            run_id,
            {"type": "enrichment", "kind": kind, "evidence": {"status": "available", "value": value, "reason": None}},
            value,
        )

    def set_pr_link(self, row: HarvestRow, number: int, repo: str) -> None:
        self._append_enrichment(row.session_id, "pr", {"number": number, "repo": repo})

    def read_scoring_inputs(self, row: HarvestRow) -> ScoringInputs:
        scoring = section_value(self._runs[row.session_id], "scoring")
        if scoring is None:
            return ScoringInputs(verifier_verdicts=None, format_valid=False, length=None)
        return ScoringInputs(
            verifier_verdicts=scoring["verifier_verdicts"],
            format_valid=scoring["format_valid"],
            length=scoring["length"],
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
            commits_in_window_fetcher=lambda repo, head, base: list(reversed(git_ops.log_shas_since(repo, head, base))),
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

    def _append_typed(
        self,
        run_id: str,
        payload: dict[str, Any],
        semantic_evidence: Any,
        *,
        item_uid: str | None = None,
        digest: str | None = None,
        valid_at: str | None = None,
        scheme: str = "canonical-json-v1",
        reply_captures: list[dict[str, Any]] | None = None,
    ) -> bool:
        digest = digest or hashlib.sha256(canonical_json(semantic_evidence).encode()).hexdigest()
        identity = hashlib.sha256(
            canonical_json(
                {
                    "run_id": run_id,
                    "item_uid": item_uid,
                    "payload": payload,
                    "evidence_digest": digest,
                    "policy": labeler_versions.LABELER_POLICY_VERSION,
                    **({"reply_captures": reply_captures} if reply_captures is not None else {}),
                }
            ).encode()
        ).hexdigest()
        now = self.now_iso()
        # Preserve the original observation timestamps on idempotent retries.
        existing = LocalRecordStore(self.store_dir).read_records()["observations"]
        if any(o["observation_id"] == identity for o in existing):
            return False
        return (
            LocalRecordStore(self.store_dir)
            .append_observation(
                {
                    "schema_version": "daydream.observation.v2"
                    if payload["type"] == "harvest-annotation"
                    else "daydream.observation.v1",
                    "observation_id": identity,
                    "run_id": run_id,
                    "item_uid": item_uid,
                    "valid_at": valid_at or now,
                    "observed_at": now,
                    "source": "harvest",
                    "author": labeler_versions.LABELER_POLICY_VERSION,
                    "role": "automatic",
                    "policy_version": labeler_versions.LABELER_POLICY_VERSION,
                    "rubric_version": labeler_versions.RUBRIC_SCHEMA_VERSION,
                    "classifier_version": labeler_versions.REPLY_CLASSIFIER_VERSION,
                    "evidence_digest": digest,
                    "evidence_digest_scheme": scheme,
                    "semantic_evidence": semantic_evidence,
                    "payload": payload,
                    **({"reply_captures": reply_captures} if reply_captures is not None else {}),
                }
            )
            .committed
        )

    def append_annotation(self, row: HarvestRow, payload: AnnotationPayload) -> bool:
        annotation = asdict(payload)
        rubric = json.loads(payload.rubric_json or "{}")
        captures_by_fingerprint = {
            resolution["fingerprint"]: resolution.pop("reply_captures", [])
            for resolution in rubric.get("per_finding_resolutions") or []
        }
        # Retained text is durable observation content, separate from annotation
        # semantic evidence and the digest that pins human judgments.
        annotation["rubric_json"] = json.dumps(rubric) if payload.rubric_json is not None else None
        inserted = self._append_typed(
            row.session_id,
            {
                "type": "harvest-annotation",
                "annotation": annotation,
                "labeler_policy_version": labeler_versions.LABELER_POLICY_VERSION,
            },
            annotation,
            valid_at=payload.valid_at,
        )
        by_fingerprint = {r["fingerprint"]: r for r in rubric.get("per_finding_resolutions") or []}
        for item in (section_value(self._runs[row.session_id], "findings") or {}).get("items", []):
            resolution = by_fingerprint.get(item["fingerprint"], {})
            evidence = resolution.get("evidence") or []
            disposition = resolution.get("disposition", "unanswered")
            digest = resolution.get("evidence_digest") or labeler_versions.reply_evidence_digest(evidence)
            finding_inserted = self._append_typed(
                row.session_id,
                {
                    "type": "finding-judgment",
                    "disposition": disposition,
                    "rationale": "Automated harvest of recorded finding evidence",
                },
                evidence,
                item_uid=item["item_uid"],
                digest=digest,
                valid_at=payload.valid_at,
                scheme="reply-evidence-v1",
                reply_captures=captures_by_fingerprint.get(item["fingerprint"], []),
            )
            inserted = finding_inserted or inserted
        return inserted

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


def collect_annotation(
    row: HarvestRow,
    *,
    services: HarvestServices,
    console: Console,
) -> tuple[HarvestRow, AnnotationPayload]:
    """Acquire and reduce evidence for a validated captured record.

    Dry-run services keep discovered PR links in memory only.
    """
    repo_resolution = services.resolve_repo(row, console=console)
    services.materialize_base_sha(
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
            if not services.dry_run:
                services.set_pr_link(row, number, slug)
            row = replace(row, pr_number=number, pr_repo=slug)

    evidence = acquire_harvest_evidence(
        row,
        services=services,
        repo_resolution=repo_resolution,
    )
    return row, build_annotation(row, evidence)


async def run_harvest(
    config: HarvestConfig,
    *,
    services: HarvestServices,
) -> dict[str, int]:
    """Validate the frozen run population and append acquired evidence."""
    if (config.store_dir.resolve() != services.store_dir.resolve()
            or config.snapshot_id != services.snapshot_id or config.dry_run != services.dry_run):
        raise ValueError("harvest record-store ownership mismatch")
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
            row, payload = collect_annotation(
                row,
                services=services,
                console=console,
            )
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

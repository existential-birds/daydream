"""Harvest immutable bronze signals into bitemporal label and reward annotations.

HarvestPass owns acquisition and persistence; build_annotation reduces a
validated row and frozen evidence without I/O. Posterior false-positive cost
stays beside the pure intrinsic composite. Qualifying decisive reply time pins
PR outcomes, falling back to merge time; local outcomes have no valid-time pin.

Bronze assembly reads deep/recommendation-verdicts.json and stack-*-records.json.
Absent verdicts preserve the format gate; malformed present JSON closes it.
Review-output character count is the length proxy (root path, then deep path).

Each pass revisits indexed runs. The archive deduplicates unchanged evidence,
policy, labels, and posterior population; changed evidence/policy appends a new
generation for historical as_of queries. Dry-run suppresses all writes. Cache
resume skips completed rows; exhausted rate limits abort without losing progress.
Other row failures are isolated. Missing base_sha is materialized during
acquisition, keeping the later frozen corpus projection free of Git I/O.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.archive.git_safe import _DEFAULT_HOSTS, normalize_remote_url
from daydream.archive.index import (
    append_label_observation,
    query_runs,
    readonly_connection,
    reviewer_set_penalty_prior,
    set_run_pr_link,
)
from daydream.git_ops import GitError, GitHubAuth, RateLimitError
from daydream.training import labeler_versions, reward
from daydream.training.adjudication.snapshot import record_evidence_digest
from daydream.training.backfill_cache import BackfillCache
from daydream.training.base_sha import materialize_base_sha
from daydream.training.harvest_types import BaseShaStatus, HarvestEvidence, HarvestRow
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingResolution,
    PRCommentThreads,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    local_commit_applied_signal,
    per_finding_resolution_signal,
    pr_link_signal,
    pr_merge_signal,
    reviewer_logins_from_comments,
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


def _read_review_output(run_dir: Path) -> str | None:
    """Read review-output.md from the run root, then deep/; return None when absent.

    Propagate filesystem errors other than FileNotFoundError.
    """
    for candidate in (run_dir / _REVIEW_OUTPUT_FILE, run_dir / "deep" / _REVIEW_OUTPUT_FILE):
        try:
            return candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return None


def assemble_scoring_inputs(run_dir: Path) -> ScoringInputs:
    """Read verifier verdicts, artifact validity, and output length from a run.

    Missing verifier verdicts leave correctness absent. Malformed structured
    artifacts fail the format gate; missing evidence never earns credit.
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

    review_text = _read_review_output(run_dir)
    return ScoringInputs(
        verifier_verdicts=verifier_verdicts,
        format_valid=format_valid,
        length=len(review_text) if review_text is not None else None,
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
"""Returned when the fix-applied cascade cannot run (missing recommended.patch
/ diff.patch, empty changed_files, or any subprocess error). The rubric still
carries the field for schema stability; outcome derivation does not depend on
it for the PR-review path."""


# HTTP statuses meaning the PR/commit is genuinely absent (fork/deleted PR 404,
# unpushed-SHA 422) so a row may degrade to its local posterior; every other gh
# failure is transient and must propagate so resume retries, not mislabel (#166).
_BENIGN_PR_ABSENCE_STATUSES = (404, 422)


def _is_benign_pr_absence(exc: GitError) -> bool:
    """Recognize HTTP 404 as absent; unknown statuses propagate as transient failures."""
    match = re.search(r"\bHTTP (\d{3})\b", str(exc))
    return match is not None and int(match.group(1)) in _BENIGN_PR_ABSENCE_STATUSES


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
    """One run's canonical annotation, ready for append_label_observation.

    Unknown labels become []; evidence_sha is the archived head. valid_at is
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


def build_annotation(row: HarvestRow, evidence: HarvestEvidence) -> AnnotationPayload:
    """Purely reduce validated row and immutable evidence into an annotation."""
    rubric = evidence.rubric
    outcome_label = derive_outcome_label(rubric)
    labels = [outcome_label] if outcome_label != "unknown" else []
    valid_at = _rubric_valid_at(rubric)
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
    """Settings for one archive pass.

    Dry-run acquires fresh evidence without mutating bronze, SQLite, clones,
    caches, or completion markers. cache_dir enables response caching/resume;
    repo_clone_root defaults to its repos/ child, or None without a cache.
    session_filter is a session-id prefix; gh_request_spacing_sec separates rows."""

    archive_dir: Path
    dry_run: bool = False
    cache_dir: Path | None = None
    repo_clone_root: Path | None = None
    session_filter: str | None = None
    gh_request_spacing_sec: float = 0.8


class HarvestPass:
    """Own acquisition, reduction, persistence, and resume state for one archive pass."""

    def __init__(self, config: HarvestConfig, *, github_auth: GitHubAuth = git_ops.INHERIT_GITHUB_AUTH) -> None:
        self._config = config
        self._github_auth = github_auth
        self._cache: BackfillCache | None = None
        self._fetched_repos: set[Path] = set()

    @property
    def archive_dir(self) -> Path:
        return self._config.archive_dir

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
        if not self.archive_dir.exists():
            raise FileNotFoundError(f"archive_dir does not exist: {self.archive_dir}")
        if self._config.dry_run:
            with closing(readonly_connection(self.archive_dir)) as conn:
                return [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM runs WHERE session_id LIKE ? || '%'",
                        (session_filter or "",),
                    )
                ]
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
    ) -> BaseShaStatus:
        if self._config.dry_run:
            return "available" if row.base_sha else "unavailable"
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
            readonly=self._config.dry_run,
        )

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
        if not changed_files:
            return _FIX_APPLIED_STUB
        try:
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
        except (FileNotFoundError, OSError, GitError):
            return _FIX_APPLIED_STUB


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
            labeler_version=labeler_versions.LABELER_POLICY_VERSION,
            **asdict(payload),
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

    def _build_rubric_pr(
        self, row: HarvestRow,
        *,
        repo_clone: Path,
        pr_merge: PRMergeSignal,
        captured_comments: list[dict[str, Any]],
        pr_author_logins: frozenset[str] = frozenset(),
        review_author_logins: frozenset[str] = frozenset(),
    ) -> Rubric:
        """Combine a fetched PR state with scoped comment, fix, and finding signals.

        Reuse captured comments; PR/review authors govern decisive reply eligibility."""
        signal_row = row.as_signal_row()
        # Index the captured review comments once; both resolution signals
        # consume the same fingerprint-scoped evidence.
        recorded_fingerprints = self.read_recorded_fingerprints(row)
        comment_threads = PRCommentThreads.from_comments(
            captured_comments, session_fingerprints=list(recorded_fingerprints),
        )
        comments = comment_resolution_signal(signal_row, gh_api=self.github, threads=comment_threads)
        fix = self.fix_applied(
            row,
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
                gh_api=self.github,
                threads=comment_threads,
                pr_author_logins=pr_author_logins,
                review_author_logins=review_author_logins,
            )
            rubric = replace(rubric, per_finding_resolutions=list(per_finding))
        return rubric


    def _build_rubric_local(
        self, row: HarvestRow,
        *,
        repo_clone: Path,
        clone_resolved: bool = False,
    ) -> Rubric:
        """Build local-branch signals, forcing unknown when no clone was resolved.

        An archive-directory placeholder cannot establish that a fix was rejected.
        """
        # Invariant: the local-commit posterior is valid ONLY for PR-less runs. A
        # degraded PR row's merge evidence was merely unavailable, so emit "unknown"
        # rather than risk a "rejected" false negative.
        if row.is_pr or not clone_resolved:
            local = LocalCommitAppliedSignal(verdict="unknown")
        else:
            try:
                local = self.local_commit_applied(
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
                    fingerprint=fingerprint, comment_id=None, disposition="missing",
                    evidence_digest=labeler_versions.reply_evidence_digest([]),
                )
                for fingerprint in self.read_recorded_fingerprints(row)
            ],
        )


    def acquire_harvest_evidence(
        self, row: HarvestRow,
        *,
        repo_resolution: Path | None,
        base_sha_status: BaseShaStatus,
        valid_at_override: str | None = None,
    ) -> HarvestEvidence:
        """Acquire one row's complete external evidence in established order."""
        repo_clone = repo_resolution or self.archive_dir
        pr_merge = None
        if row.is_pr:
            try:
                pr_merge = pr_merge_signal(row.as_signal_row(), gh_api=self.github)
            except RateLimitError:
                raise
            except GitError as exc:
                if not _is_benign_pr_absence(exc):
                    raise

        reviewer_logins: list[str] = []
        pooled_prior = None
        prior_n = 0
        if pr_merge is None:
            rubric = self._build_rubric_local(
                row, repo_clone=repo_clone,
                clone_resolved=repo_resolution is not None,
            )
        else:
            assert row.pr_repo is not None
            captured_comments = None
            endpoint = f"repos/{row.pr_repo}/pulls/{row.pr_number}"
            try:
                reviews = self.github(row.pr_repo, f"{endpoint}/reviews", paginate=True)
                captured_comments = self.github(row.pr_repo, f"{endpoint}/comments", paginate=True)
                reviewer_logins = reviewer_logins_from_comments(reviews, captured_comments)
            except RateLimitError:
                raise
            except GitError:
                pass
            if captured_comments is None:
                captured_comments = self.github(row.pr_repo, f"{endpoint}/comments", paginate=True)
            rubric = self._build_rubric_pr(
                row, repo_clone=repo_clone,
                pr_merge=pr_merge, captured_comments=captured_comments,
                pr_author_logins=frozenset({pr_merge.author_login}) if pr_merge.author_login else frozenset(),
                review_author_logins=frozenset(reviewer_logins),
            )
            pooled_prior, prior_n = self.reviewer_prior(
                tuple(reviewer_logins),
                before_valid_at=_rubric_valid_at(rubric) or self.now_iso(),
                exclude_session=row.session_id,
                repo_slug=row.repo_slug,
            )

        # Preserve the established boundary: bronze reads occur after posterior,
        # reviewer, and prior acquisition.
        scoring_inputs = assemble_scoring_inputs(row.archive_path)
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


    def collect_annotation(
        self, row: HarvestRow, *, console: Console,
    ) -> tuple[HarvestRow, AnnotationPayload]:
        """Collect and reduce the same evidence for preview and canonical harvest.

        Dry-run policy belongs to this pass; preview linking remains in-memory only.
        """
        repo_resolution = self.resolve_repo(row, console=console)
        base_sha_status = self.materialize_base_sha(
            row,
            repo_resolution,
            console=console,
        )
        # Re-link orphan runs before acquiring their posterior. Persistence
        # remains at this exact boundary, so a later row failure keeps the
        # durable link as before.
        if not row.is_pr:
            try:
                link = pr_link_signal(row.as_signal_row(), gh_api=self.github)
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
                if not self._config.dry_run:
                    set_run_pr_link(self.archive_dir, row.session_id, number, slug)
                row = replace(row, pr_number=number, pr_repo=slug)

        evidence = self.acquire_harvest_evidence(
            row,
            repo_resolution=repo_resolution,
            base_sha_status=base_sha_status,
        )
        return row, build_annotation(row, evidence)


    async def run(self) -> dict[str, int]:
        """Validate the archive queue, acquire evidence, and persist annotations."""
        config = self._config
        raw_queue = self.query_rows(config.session_filter)
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
        done = self.completed_sessions()
        queue = [row for row in queue if row.session_id not in done]
        summary["considered"] += len(queue)

        for row in queue:
            try:
                row, payload = self.collect_annotation(
                    row, console=console,
                )
                if config.dry_run:
                    summary["would_annotate"] += 1
                else:
                    if self.append_annotation(row, payload):
                        summary["annotated"] += 1
                    else:
                        summary["skipped"] += 1
                    # A successful append or dedup is complete for resume. A raised
                    # write never advances the marker.
                    self.mark_session_done(row.session_id)
            except RateLimitError:
                summary["aborted"] = 1
                resume_marker = self.progress_path
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
                await self.sleep_between_rows(config.gh_request_spacing_sec)

        return summary

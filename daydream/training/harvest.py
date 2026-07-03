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

Annotation builder:

* :func:`build_annotation` composes the posterior rubric (PR vs local-branch,
  mirroring the former labeler's assembly), derives the outcome label, captures
  the human reviewer set and its pooled prior penalty (PR rows only), scores the
  reward (intrinsic composite plus the calibrated posterior sibling axis fed
  from the outcome label + prior), asserts the breakdown carries the canonical
  :data:`~daydream.training.reward.REWARD_VERSION` before it can be written to
  canonical storage, and returns a frozen :class:`AnnotationPayload`. It is
  *pure of DB writes* — the orchestrator persists the payload. Error policy:
  ``RateLimitError`` propagates (the orchestrator aborts cleanly and preserves
  its resume marker); a *benign* PR-merge-status fetch failure (fork PR 404,
  unpushed-SHA 422) degrades the row to its local-branch posterior; every other
  reviewer-signal, prior-query, posterior-fetch, and git error propagates to the
  caller, which isolates per-row.
* ``valid_at`` is the PR merge timestamp for PR rows and ``None`` for
  non-PR/local rows (the write layer collapses ``None`` → ``observed_at``).

Orchestrator (:func:`run_harvest`):

* **Idempotent and re-runnable:** every indexed run is considered on every
  pass, but the write layer dedups on ``(evidence_sha, reward_version)`` — a
  re-harvest with unchanged evidence is a no-op (counted in ``skipped``). A
  ``REWARD_VERSION`` bump changes the dedup key and so appends a fresh
  generation, letting older ``as_of`` pins still resolve their original
  generation. Only the ``cache``/``dry_run`` paths otherwise suppress writes.
* **Per-row error isolation:** an exception on one row counts in ``errors`` and
  does not derail subsequent rows. Configuration errors (missing
  ``archive_dir``) raise before the loop begins.
* **Capture-time ``base_sha``:** materialized into the manifest when missing
  (the only fallible git I/O of the annotate pass lives here, not in the pure
  build-corpus projection).

The rubric-assembly helpers live here as harvest's own (the legacy
``labeler.py`` is retired in plan Task 13); the git/``gh`` wrappers are
module-level so the orchestrator and tests can monkeypatch them as
injection seams.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.archive.index import (
    append_label_observation,
    query_runs,
    reviewer_set_penalty_prior,
    set_run_pr_link,
)
from daydream.git_ops import GitError, RateLimitError
from daydream.training import reward
from daydream.training.backfill_cache import BackfillCache
from daydream.training.base_sha import materialize_base_sha
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    local_commit_applied_signal,
    per_finding_resolution_signal,
    pr_link_signal,
    pr_merge_signal,
    reviewer_logins_signal,
)
from daydream.training.reward import ScoringInputs, score_trajectory
from daydream.training.rubric import Rubric, derive_outcome_label, derive_per_finding_labels
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
    r"""Return the review-output text, or ``None`` when absent.

    Tries ``review-output.md`` at the run root first (shallow-loop layout),
    then ``deep/review-output.md`` (deep-mode layout), mirroring the former
    exporter's back-compat fallback order.

    Args:
        run_dir: The archived run directory.

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

    Args:
        run_dir: The archived run directory.

    Returns:
        The character count of the first review-output file found, or
        ``None`` when neither location exists.
    """
    text = _read_review_output(run_dir)
    return len(text) if text is not None else None


def assemble_scoring_inputs(run_dir: Path, row: dict[str, Any]) -> ScoringInputs:
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
        grounding_rate=row.get("grounding_rate"),
        format_valid=format_valid,
        length=_read_review_output_length(run_dir),
    )


# git / gh wrappers — module-level so they double as monkeypatch seams.

# Bounded rate-limit backoff for the gh seam (honors parsed Retry-After, capped
# at _MAX_BACKOFF_SEC, falling back to _DEFAULT_BACKOFF_SEC when absent).
_DEFAULT_BACKOFF_SEC = 30.0
_MAX_BACKOFF_SEC = 120.0
_MAX_RATE_LIMIT_RETRIES = 5


def _rate_limit_sleep(seconds: float) -> None:
    """Sleep ``seconds`` between rate-limit retries (seam for tests to stub)."""
    time.sleep(seconds)


async def _row_spacing_sleep(seconds: float) -> None:
    """Pause ``seconds`` between rows to spread ``gh`` calls (seam for tests to stub)."""
    await anyio.sleep(seconds)


def _gh_api(repo: str, endpoint: str, **kwargs: Any) -> Any:
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
            return git_ops.gh_api(Path("."), endpoint, **kwargs)
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
            _rate_limit_sleep(backoff)
    # Unreachable: the loop either returns or raises on the final attempt.
    raise RuntimeError("rate-limit retry loop exited without returning")  # pragma: no cover


def _diff_name_only(repo: Path, base: str, head: str) -> list[str]:
    """Proxy to :func:`daydream.git_ops.diff_name_only`."""
    return git_ops.diff_name_only(repo, base, head)


def _commits_in_window(repo: Path, head: str, base: str) -> list[str]:
    """Return commits on ``base`` since ``head``'s ancestor, oldest → newest.

    Used by the fix-applied cascade to bound the upstream review window.
    The ``head..base`` range already bounds the walk; no date filter is
    needed (see #167).

    :func:`git_ops.log_shas_since` returns newest-first (git log order);
    we reverse to oldest → newest to match the downstream contract in
    :func:`daydream.training.labeler_signals.fix_applied_signal`, where
    ``window[-1]`` is expected to be the latest commit in the window.
    """
    return list(reversed(git_ops.log_shas_since(repo, head, base)))


def _commits_since(repo: Path, branch: str, since: str) -> list[str]:
    """Return commits on ``branch`` after ``since``.

    Used by the local-branch posterior path to walk commits pushed after
    the daydream-recorded ``head_sha``.
    """
    return git_ops.log_shas(repo, branch, since=since)


def _file_at(repo: Path, path: str, sha: str) -> str:
    """Return file content at ``sha``; empty string on missing path."""
    try:
        return git_ops.show(repo, sha, path).decode("utf-8", errors="replace")
    except git_ops.GitError:
        return ""


# Rubric assembly — PR vs local-branch (harvest's own, formerly in labeler.py)


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
    row: dict[str, Any],
    *,
    changed_files: list[str],
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
        return fix_applied_signal(
            row,
            changed_files=changed_files,
            repo_clone=repo_clone,
            diff_fetcher=_diff_name_only,
            commits_in_window_fetcher=_commits_in_window,
            file_at_fetcher=_file_at,
        )
    except (FileNotFoundError, OSError, GitError):
        return _FIX_APPLIED_STUB


def _row_changed_files(row: dict[str, Any]) -> list[str]:
    """Return ``changed_files`` from a sqlite row, decoding the JSON column."""
    raw = row.get("changed_files")
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        decoded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return list(decoded) if isinstance(decoded, list) else []


def _row_is_pr(row: dict[str, Any]) -> bool:
    """Return ``True`` when the row has both ``pr_repo`` and ``pr_number``."""
    return bool(row.get("pr_repo")) and row.get("pr_number") is not None


def _row_recorded_fingerprints(row: dict[str, Any]) -> list[str]:
    """Return the finding fingerprints recorded for a row, in order.

    Prefers a pre-extracted ``findings_fingerprints`` column; otherwise reads
    the archived ``findings.json`` artifact and collects each finding's
    ``fingerprint``. Returns ``[]`` when neither source is available or the
    artifact is missing / malformed — a row with no recorded findings simply
    yields no per-finding join.
    """
    pre_extracted = row.get("findings_fingerprints")
    if isinstance(pre_extracted, list):
        return [str(fp) for fp in pre_extracted]

    archive_path = row.get("archive_path")
    if not archive_path:
        return []
    try:
        data = json.loads((Path(archive_path) / "findings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    findings = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(findings, list):
        return []
    return [str(f["fingerprint"]) for f in findings if isinstance(f, dict) and "fingerprint" in f]


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
    row: dict[str, Any],
    *,
    gh_api: Any,
    repo_clone: Path,
    pr_merge: PRMergeSignal | None = None,
) -> Rubric:
    """Compose all four signals for a row that originated from a PR.

    Args:
        pr_merge: Pre-fetched :class:`PRMergeSignal`. When supplied
            (already resolved by the caller before the catch boundary),
            ``pr_merge_signal`` is not called again. When ``None`` it is
            fetched here as before.
    """
    changed_files = _row_changed_files(row)
    signal_row = {**row, "changed_files": changed_files}
    if pr_merge is None:
        pr_merge = pr_merge_signal(signal_row, gh_api=gh_api)
    comments = comment_resolution_signal(signal_row, gh_api=gh_api)
    fix = _safe_fix_applied(
        signal_row,
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
    recorded_fingerprints = _row_recorded_fingerprints(row)
    if recorded_fingerprints:
        per_finding = per_finding_resolution_signal(
            signal_row, recorded_fingerprints=recorded_fingerprints, gh_api=gh_api
        )
        rubric = replace(rubric, per_finding_labels=list(derive_per_finding_labels(rubric, per_finding)))
    return rubric


def _build_rubric_local(
    row: dict[str, Any],
    *,
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
    if _row_is_pr(row):
        local = LocalCommitAppliedSignal(verdict="unknown")
    elif not clone_resolved:
        local = LocalCommitAppliedSignal(verdict="unknown")
    else:
        try:
            local = local_commit_applied_signal(
                row,
                repo_clone=repo_clone,
                commits_since_fetcher=_commits_since,
                file_at_fetcher=_file_at,
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
    associated".
    """
    if rubric.posterior_source != "pr_review":
        return None
    return "merged" if rubric.pr_merge.merged else "closed"


# Per-run annotation builder


@dataclass(frozen=True)
class AnnotationPayload:
    """One run's bitemporal annotation, ready to persist (no DB writes here).

    Attributes:
        labels: Outcome labels (``[]`` when the derived label is
            ``"unknown"``, else a single-element list).
        pr_state: sqlite ``pr_state`` discriminator (``"merged"``/``"closed"``
            for PR rows, ``None`` for local-branch rows).
        valid_at: The valid-time of the posterior outcome — the PR merge
            timestamp for PR rows, ``None`` for non-PR/local rows (the write
            layer collapses ``None`` → ``observed_at``).
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
            breakdown is a :class:`~daydream.training.reward.PosteriorBreakdown`
            (a mapped maintainer outcome label was supplied).
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


def _degrade_to_local(
    row: dict[str, Any],
    *,
    repo_clone: Path,
    clone_resolved: bool,
) -> tuple[Any, None, list[str], None, int]:
    """Build a local-branch rubric and return the degraded posterior state.

    Used when a PR-path fetch fails benignly (fork PR 404, unpushed-SHA 422)
    or when the row has no PR at all.  Returns a 5-tuple
    ``(rubric, valid_at, reviewer_logins, outcome_prior, prior_n)`` with the
    non-PR defaults so callers can unpack uniformly.
    """
    rubric = _build_rubric_local(row, repo_clone=repo_clone, clone_resolved=clone_resolved)
    return rubric, None, [], None, 0


def build_annotation(
    row: dict[str, Any],
    *,
    run_dir: Path,
    archive_dir: Path,
    gh_api: Any,
    repo_clone: Path,
    clone_resolved: bool = True,
) -> AnnotationPayload:
    """Build one run's bitemporal annotation payload (pure of DB writes).

    Composes the posterior rubric (PR vs local-branch, mirroring the former
    labeler's assembly), derives the outcome label, and assembles the intrinsic
    :class:`ScoringInputs` from bronze.

    For PR rows it additionally captures the human reviewer set
    (:func:`~daydream.training.labeler_signals.reviewer_logins_signal`) and the
    pooled prior penalty over prior runs sharing a reviewer
    (:func:`~daydream.archive.index.reviewer_set_penalty_prior`). The pooled
    mean graduates to the empirical ``outcome_prior`` only when the pooled count
    reaches :data:`_PRIOR_SUFFICIENCY_THRESHOLD`; below threshold the prior is
    left ``None`` (the reducer applies the ``0.5`` default), but
    ``outcome_prior_n`` always records the pooled count for audit. Local/non-PR
    rows have no reviewer set (``reviewer_logins=[]``, ``outcome_prior=None``,
    ``outcome_prior_n=0``).

    Scores the reward via :func:`~daydream.training.reward.score_trajectory` with
    the outcome label + prior; a mapped label yields a
    :class:`~daydream.training.reward.PosteriorBreakdown` whose ``composite`` is
    the pure intrinsic score (C5 — the posterior penalty is a sibling, never
    folded in). Asserts the breakdown carries the canonical
    :data:`~daydream.training.reward.REWARD_VERSION` before returning, so a
    non-canonical (analysis-time override) score can never reach canonical
    storage. Returns a frozen :class:`AnnotationPayload`; the orchestrator
    persists it.

    Args:
        row: The indexed manifest row (carries ``session_id``, the PR
            discriminators, ``head_sha``, ``grounding_rate``, etc.).
        run_dir: The archived run directory (bronze bundle root) feeding the
            intrinsic reward signals.
        archive_dir: The archive root, queried for the pooled reviewer-set prior.
        gh_api: Callable invoked as ``gh_api(repo, endpoint, **kwargs)`` by
            the PR posterior + reviewer signals; unused on the local-branch path.
        repo_clone: Local clone root for the fix-applied / local-commit
            cascades.
        clone_resolved: Whether a real git working tree was obtained for the
            row. When ``False`` the local-branch posterior is forced to
            ``"unknown"`` rather than risking a ``"rejected"`` mislabel from a
            commit check that had no repository to inspect.

    Returns:
        A frozen :class:`AnnotationPayload`. ``valid_at`` is the PR merge
        timestamp for PR rows and ``None`` for non-PR/local rows.

    Raises:
        AssertionError: When the scored breakdown does not carry the canonical
            ``REWARD_VERSION`` (a non-default-weights override leaked in).
        RateLimitError: Propagated unchanged so the orchestrator can abort the
            sweep cleanly and preserve its resume marker.
        Exception: A *benign* PR-merge-status fetch failure (fork PR 404,
            unpushed-SHA 422) is caught and degrades the row to its local-branch
            posterior rather than raising. Every other reviewer-signal,
            prior-query, posterior-fetch (``gh_api``), or git error propagates
            unchanged to the caller (the orchestrator isolates per-row). Once the
            merge status is confirmed, a later ``comment_resolution_signal``
            failure also propagates (the confirmed merge evidence is never
            discarded).
    """
    if _row_is_pr(row):
        try:
            _pr_merge = pr_merge_signal(
                {**row, "changed_files": _row_changed_files(row)},
                gh_api=gh_api,
            )
        except RateLimitError:
            raise
        except GitError as exc:
            # Narrow catch (only pr_merge_signal): a benign 404/422 degrades to
            # local; a transient failure propagates so the run retries, not
            # mislabels. A later comment-fetch GitError stays outside this block.
            if not _is_benign_pr_absence(exc):
                raise
            rubric, valid_at, reviewer_logins, outcome_prior, prior_n = _degrade_to_local(
                row, repo_clone=repo_clone, clone_resolved=clone_resolved
            )
        else:
            # Merge confirmed, so the PR provably exists; a secondary comment-fetch
            # GitError is transient and must propagate, not discard the confirmed
            # PRMergeSignal by degrading to local.
            rubric = _build_rubric_pr(
                row,
                gh_api=gh_api,
                repo_clone=repo_clone,
                pr_merge=_pr_merge,
            )
            valid_at = rubric.pr_merge.merged_at
            try:
                reviewer_logins = reviewer_logins_signal(row, gh_api=gh_api)
            except RateLimitError:
                raise
            except GitError:
                # Reviewer-set prior only refines the decided outcome; an
                # auxiliary-lookup failure degrades to no prior, keeping the label.
                reviewer_logins = []
                outcome_prior = None
                prior_n = 0
            else:
                pooled, prior_n = reviewer_set_penalty_prior(
                    archive_dir,
                    reviewer_logins,
                    before_valid_at=valid_at or datetime.now(timezone.utc).isoformat(),
                    exclude_session=row["session_id"],
                    repo_slug=row.get("repo_slug"),
                )
                outcome_prior = pooled if prior_n >= _PRIOR_SUFFICIENCY_THRESHOLD else None
    else:
        rubric, valid_at, reviewer_logins, outcome_prior, prior_n = _degrade_to_local(
            row, repo_clone=repo_clone, clone_resolved=clone_resolved
        )

    outcome_label = derive_outcome_label(rubric)
    labels = [outcome_label] if outcome_label != "unknown" else []

    inputs = assemble_scoring_inputs(run_dir, row)
    rb = score_trajectory(
        inputs,
        pr_feedback=outcome_label,
        outcome_prior=outcome_prior,
        outcome_prior_n=prior_n,
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
        evidence_sha=row.get("head_sha"),
        rubric_json=json.dumps(rubric.to_dict()),
        reviewer_logins=reviewer_logins,
        has_posterior=isinstance(rb, reward.PosteriorBreakdown),
    )


# Repo resolution — three-tier priority: source_path → clone cache → None


def _resolve_repo_for_row(
    row: dict[str, Any],
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

    Returns:
        Path to a usable working tree, or ``None``.
    """
    source_path = row.get("source_path")
    if source_path and (Path(source_path) / ".git").exists():
        return Path(source_path)

    remote_url = row.get("remote_url")
    repo_slug = row.get("repo_slug")
    if not remote_url or not repo_slug or clone_cache is None:
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
            git_ops.clone(remote_url, cached_repo, blobless=True)
    except (GitError, OSError) as exc:
        print_warning(
            console or create_console(),
            f"harvest: repo resolution failed for {repo_slug}: {type(exc).__name__}: {exc}",
        )
        if not (cached_repo / ".git").exists():
            return None
    return cached_repo


def _materialize_base_sha_if_missing(
    row: dict[str, Any], run_dir: Path, repo_clone: Path | None, *, console: Console | None = None
) -> None:
    """Opportunistically backfill ``code_context.base_sha`` into the manifest.

    Only acts when ``manifest.json`` exists AND ``repo_clone`` is available.
    Any failure is swallowed (opportunistic), leaving ``base_sha`` as ``None``.

    Args:
        row: The indexed manifest row.
        run_dir: The archived run directory (holds ``manifest.json``).
        repo_clone: Resolved repo working tree, or ``None``.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    if repo_clone is None:
        return
    try:
        materialize_base_sha(manifest_path, repo_clone=repo_clone)
    except (OSError, json.JSONDecodeError, GitError) as exc:
        print_warning(
            console or create_console(),
            f"harvest: base_sha backfill failed for {manifest_path}: {type(exc).__name__}: {exc}",
        )


# Orchestrator — idempotent (evidence-hash dedup), re-runnable, per-row isolation


@dataclass(frozen=True)
class HarvestConfig:
    """Configuration for a single :func:`run_harvest` invocation.

    Mirrors the retired labeler's config shape (plan Task 6).

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


async def run_harvest(config: HarvestConfig) -> dict[str, int]:
    """Walk the archive and append one fresh annotation per indexed run.

    The single deferred annotate pass: for every indexed run, materialize the
    capture-time ``base_sha`` (when missing), re-link orphan runs to their PR
    (a run launched before its PR existed has ``pr_number=None`` but carries
    ``repo_slug`` + ``head_sha``; :func:`pr_link_signal` resolves the PR by
    head sha and the linkage is persisted via
    :func:`daydream.archive.index.set_run_pr_link` so it becomes labelable),
    build the bitemporal annotation
    (label + intrinsic reward + posterior sibling axis + captured reviewer set +
    ``valid_at``) via :func:`build_annotation`, and append it through
    :func:`daydream.archive.index.append_label_observation` (which persists
    ``reviewer_logins`` and the ``has_posterior`` population discriminator).

    **Idempotent and re-runnable:** every indexed run is considered, but the
    write layer dedups on ``(evidence_sha, reward_version)`` — a re-harvest with
    unchanged evidence is a no-op counted in ``skipped``. A later
    ``REWARD_VERSION`` bump changes the dedup key and so appends a fresh
    generation, letting older ``as_of`` pins still resolve their original
    generation. Only the ``cache``/``dry_run`` paths otherwise suppress writes.

    Per-row error isolation: an exception on one row counts in ``errors`` and
    does not derail subsequent rows. Configuration errors (missing
    ``archive_dir``) raise before the loop begins.

    Args:
        config: A :class:`HarvestConfig` instance.

    Returns:
        Summary dict with keys ``considered``, ``annotated``,
        ``would_annotate``, ``skipped``, ``errors``, and ``aborted`` (``1`` when
        the sweep stopped early on an exhausted GitHub rate limit, else ``0``).

    Raises:
        FileNotFoundError: When ``config.archive_dir`` does not exist.
            Configuration errors deliberately surface before the loop.
    """
    if not config.archive_dir.exists():
        msg = f"archive_dir does not exist: {config.archive_dir}"
        raise FileNotFoundError(msg)

    # Queue every indexed run (optionally prefix-filtered); the write layer makes
    # re-harvest idempotent by deduping unchanged (evidence_sha, reward_version).
    if config.session_filter:
        queue = query_runs(
            config.archive_dir,
            "session_id LIKE ? || '%'",
            (config.session_filter,),
        )
    else:
        queue = query_runs(config.archive_dir)

    # The cache wraps the gh seam and tracks completed sessions so an interrupted
    # run resumes without re-fetching.
    cache: BackfillCache | None = None
    gh_api_callable: Any = _gh_api
    if config.cache_dir is not None:
        cache = BackfillCache(cache_dir=config.cache_dir, inner=_gh_api)
        gh_api_callable = cache
        done = cache.completed_sessions()
        queue = [row for row in queue if row["session_id"] not in done]

    clone_cache = config.repo_clone_root or (config.cache_dir / "repos" if config.cache_dir else None)
    fetched_repos: set[Path] = set()

    summary = {
        "considered": len(queue),
        "annotated": 0,
        "would_annotate": 0,
        "skipped": 0,
        "errors": 0,
        "aborted": 0,
    }

    console = create_console()

    for row in queue:
        try:
            run_dir = Path(row["archive_path"])
            row_repo_clone = _resolve_repo_for_row(
                row, clone_cache=clone_cache, fetched_repos=fetched_repos, console=console
            )
            _materialize_base_sha_if_missing(row, run_dir, repo_clone=row_repo_clone, console=console)
            # Re-link orphan runs (archived before the PR existed): resolve the
            # now-existing PR by head_sha so the run becomes labelable. Inside the
            # per-row try so a lookup error isolates (RateLimitError still aborts).
            if not _row_is_pr(row):
                try:
                    link = pr_link_signal(row, gh_api=gh_api_callable)
                except RateLimitError:
                    raise
                except GitError as exc:
                    # Benign 404/422 ⇒ PR unresolvable; degrade to local-branch
                    # posterior (row stays pr_number=None). Transient failures
                    # propagate so resume retries, not caches a degraded label.
                    if not _is_benign_pr_absence(exc):
                        raise
                    print_warning(
                        console,
                        f"harvest: PR link lookup failed for session {row['session_id']}; "
                        f"degrading to local-branch posterior: {type(exc).__name__}: {exc}",
                    )
                    link = None
                if link is not None:
                    number, slug = link
                    row["pr_number"] = number
                    row["pr_repo"] = slug
                    if not config.dry_run:
                        set_run_pr_link(config.archive_dir, row["session_id"], number, slug)
            payload = build_annotation(
                row,
                run_dir=run_dir,
                archive_dir=config.archive_dir,
                gh_api=gh_api_callable,
                repo_clone=row_repo_clone or config.archive_dir,
                clone_resolved=row_repo_clone is not None,
            )
            if config.dry_run:
                summary["would_annotate"] += 1
            else:
                appended = append_label_observation(
                    config.archive_dir,
                    row["session_id"],
                    labels=payload.labels,
                    pr_state=payload.pr_state,
                    labeler_version=reward.REWARD_VERSION,
                    evidence_sha=payload.evidence_sha,
                    rubric_json=payload.rubric_json,
                    valid_at=payload.valid_at,
                    reward_version=payload.reward_version,
                    reward_json=payload.reward_json,
                    composite_reward=payload.composite_reward,
                    reviewer_logins=payload.reviewer_logins,
                    has_posterior=payload.has_posterior,
                )
                if appended:
                    summary["annotated"] += 1
                else:
                    # Deduped: unchanged evidence/reward-version is a no-op re-run.
                    summary["skipped"] += 1
                # Either way the row is "done" for resume — re-running must not re-fetch it.
                if cache is not None:
                    cache.mark_session_done(row["session_id"])
        except RateLimitError:
            # Rate limit exhausted: abort cleanly. The failed row is NOT marked
            # done, so its resume marker is preserved for a later re-run.
            summary["aborted"] = 1
            resume_marker = cache.progress_path if cache is not None else None
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
                f"harvest: session {row.get('session_id', '<unknown>')} failed: "
                f"{type(exc).__name__}: {exc}",
            )
            continue

        if not config.dry_run:
            await _row_spacing_sleep(config.gh_request_spacing_sec)

    return summary

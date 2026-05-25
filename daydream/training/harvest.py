"""Harvest pass — assemble immutable bronze signals into reward inputs.

The harvest pass is the single deferred *annotate* step of the corpus
pipeline: it reads an archived run's immutable bronze artifacts, reduces
them to a :class:`~daydream.training.reward.ScoringInputs`, scores an
intrinsic :class:`~daydream.training.reward.RewardBreakdown`, derives the
outcome label, and appends one bitemporal annotation. This module carries
the bronze-signal assembly step (plan Task 4) and the per-run annotation
builder (plan Task 5); the orchestrator lands in a later task.

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

* Absent structured artifacts ⇒ ``verifier_verdicts=None`` (the shallow-run
  path); ``format_valid`` stays ``True`` because nothing failed to parse.
* A *present* verdicts/records file that is malformed JSON ⇒ caught as
  :class:`json.JSONDecodeError` and surfaced as ``format_valid=False``;
  assembly never crashes on bad data.
* ``grounding_rate`` is read from the indexed manifest row
  (``row["grounding_rate"]``), never re-derived here.
* ``length`` is the documented review-output char-count proxy, ``None`` when
  no review output exists.

Annotation builder:

* :func:`build_annotation` composes the posterior rubric (PR vs local-branch,
  mirroring the former labeler's assembly), derives the outcome label, scores
  the intrinsic reward, and returns a frozen :class:`AnnotationPayload`. It is
  *pure of DB writes* — the orchestrator persists the payload. Posterior-fetch
  and git errors propagate to the caller, which isolates per-row.
* ``valid_at`` is the PR merge timestamp for PR rows and ``None`` for
  non-PR/local rows (the write layer collapses ``None`` → ``observed_at``).

The rubric-assembly helpers live here as harvest's own (the legacy
``labeler.py`` is retired in plan Task 13); the git/``gh`` wrappers are
module-level so the orchestrator and tests can monkeypatch them as
injection seams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.training import reward
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    pr_merge_signal,
)
from daydream.training.reward import ScoringInputs, score_trajectory
from daydream.training.rubric import Rubric, derive_outcome_label

_VERDICTS_FILE = "recommendation-verdicts.json"
"""Bronze artifact (under ``deep/``) carrying the ``verdicts`` list."""

_RECORDS_GLOB = "stack-*-records.json"
"""Bronze per-stack finding-record artifacts (under ``deep/``)."""

_REVIEW_OUTPUT_FILE = "review-output.md"
"""Length-proxy artifact; at the run root for shallow runs, under ``deep/`` for deep runs."""


def _read_review_output_length(run_dir: Path) -> int | None:
    """Return the review-output char count, or ``None`` when absent.

    Tries ``review-output.md`` at the run root first (shallow-loop layout),
    then ``deep/review-output.md`` (deep-mode layout), mirroring the former
    exporter's back-compat fallback order.

    Args:
        run_dir: The archived run directory.

    Returns:
        The character count of the first review-output file found, or
        ``None`` when neither location exists.
    """
    for candidate in (run_dir / _REVIEW_OUTPUT_FILE, run_dir / "deep" / _REVIEW_OUTPUT_FILE):
        try:
            return len(candidate.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
    return None


def assemble_scoring_inputs(run_dir: Path, row: dict[str, Any]) -> ScoringInputs:
    """Reduce one run's bronze artifacts to intrinsic :class:`ScoringInputs`.

    Reads the structured bronze artifacts under ``run_dir/deep`` and the
    review-output length proxy, combining them with the indexed
    ``grounding_rate`` into the capture-time signals the reward reducer
    consumes. Absent artifacts yield the shallow-run path
    (``verifier_verdicts=None``); a present-but-malformed structured
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
        # Shallow run — no structured verdicts. Nothing failed to parse.
        pass
    except json.JSONDecodeError:
        # Present but malformed structured artifact ⇒ format gate floors.
        format_valid = False

    # Per-stack records are structural bronze too: a present-but-malformed
    # records file also trips the format gate, even though it doesn't feed a
    # reward axis in the minimal reducer.
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


# ---------------------------------------------------------------------------
# git / gh wrappers — module-level so they double as monkeypatch seams.
# ---------------------------------------------------------------------------


def _diff_name_only(repo: Path, base: str, head: str) -> list[str]:
    """Proxy to :func:`daydream.git_ops.diff_name_only`."""
    return git_ops.diff_name_only(repo, base, head)


def _commits_in_window(repo: Path, head: str, base: str, days: int) -> list[str]:
    """Return commits on ``base`` since ``head``'s ancestor, within ``days``.

    Used by the fix-applied cascade to bound the upstream review window.
    """
    return git_ops.log_shas_since(repo, head, base, since_days=days)


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


# ---------------------------------------------------------------------------
# Rubric assembly — PR vs local-branch (harvest's own, formerly in labeler.py)
# ---------------------------------------------------------------------------


_FIX_APPLIED_STUB = FixAppliedSignal(
    verdict="unknown",
    hunks_applied=0,
    hunks_total=0,
    window_commits=[],
)
"""Returned when the fix-applied cascade cannot run (missing diff.patch,
empty changed_files, or any subprocess error). The rubric still carries
the field for schema stability; outcome derivation does not depend on it
for the PR-review path."""


def _added_lines(diff_text: str) -> list[str]:
    """Extract the content of ``+`` lines from a unified-diff blob.

    Strips the leading ``+`` and any single leading space (the conventional
    unified-diff content marker). Excludes ``+++`` file headers. Whitespace-
    only lines are dropped so they don't poison the substring check.
    """
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++"):
            continue
        if not line.startswith("+"):
            continue
        content = line[1:]
        if content.startswith(" "):
            content = content[1:]
        content = content.strip()
        if content:
            out.append(content)
    return out


def _local_branch_verdict(
    *,
    diff_text: str,
    commits: list[str],
    repo_clone: Path,
    changed_files: list[str],
    file_at_fetcher: Any,
) -> LocalCommitAppliedSignal:
    """Lenient posterior signal for PR-less runs.

    Returns ``"applied"`` if any commit in ``commits`` yields file content
    (via ``file_at_fetcher``) that contains every added line from
    ``diff_text`` as a substring. ``"rejected"`` if commits exist but none
    match; ``"unknown"`` when there are no commits to inspect.
    """
    if not commits:
        return LocalCommitAppliedSignal(verdict="unknown")
    added = _added_lines(diff_text)
    if not added:
        return LocalCommitAppliedSignal(verdict="rejected")
    paths = changed_files if changed_files else [""]
    for commit in commits:
        for path in paths:
            try:
                content = file_at_fetcher(repo_clone, path, commit)
            except Exception:  # noqa: BLE001 - extractor isolation
                continue
            if all(needle in content for needle in added):
                return LocalCommitAppliedSignal(verdict="applied")
    return LocalCommitAppliedSignal(verdict="rejected")


def _safe_fix_applied(
    row: dict[str, Any],
    *,
    changed_files: list[str],
    repo_clone: Path,
    window_days: int,
) -> FixAppliedSignal:
    """Run :func:`fix_applied_signal`, swallowing missing-data errors.

    The cascade needs ``archive_path/diff.patch`` to exist. When the archive
    directory is missing (older runs, dry fixtures), or when ``changed_files``
    is empty, return :data:`_FIX_APPLIED_STUB`.
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
            window_days=window_days,
        )
    except (FileNotFoundError, OSError):
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


def _build_rubric_pr(
    row: dict[str, Any],
    *,
    gh_api: Any,
    repo_clone: Path,
    window_days: int,
) -> Rubric:
    """Compose all four signals for a row that originated from a PR."""
    changed_files = _row_changed_files(row)
    signal_row = {**row, "changed_files": changed_files}
    pr_merge = pr_merge_signal(signal_row, gh_api=gh_api)
    comments = comment_resolution_signal(signal_row, gh_api=gh_api)
    fix = _safe_fix_applied(
        signal_row,
        changed_files=changed_files,
        repo_clone=repo_clone,
        window_days=window_days,
    )
    return Rubric(
        pr_merge=pr_merge,
        fix_applied=fix,
        comment_resolution=comments,
        local_commit_applied=None,
        posterior_source="pr_review",
    )


def _build_rubric_local(
    row: dict[str, Any],
    *,
    repo_clone: Path,
) -> Rubric:
    """Compose signals for a PR-less row (local-branch posterior)."""
    archive_path = Path(row["archive_path"])
    diff_text = ""
    try:
        diff_text = (archive_path / "diff.patch").read_text()
    except (FileNotFoundError, OSError):
        diff_text = ""
    commits = _commits_since(repo_clone, row.get("branch") or "HEAD", row.get("head_sha") or "")
    local = _local_branch_verdict(
        diff_text=diff_text,
        commits=commits,
        repo_clone=repo_clone,
        changed_files=_row_changed_files(row),
        file_at_fetcher=_file_at,
    )
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


# ---------------------------------------------------------------------------
# Per-run annotation builder
# ---------------------------------------------------------------------------


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
            :class:`~daydream.training.reward.RewardBreakdown.to_dict` so
            re-projection has every axis.
        composite_reward: The cached intrinsic composite scalar (or ``None``
            when uncomputable).
        evidence_sha: The run's ``head_sha`` (the evidence anchor for the
            posterior signals), or ``None``.
    """

    labels: list[str]
    pr_state: str | None
    valid_at: str | None
    reward_version: str
    reward_json: str
    composite_reward: float | None
    evidence_sha: str | None


def build_annotation(
    row: dict[str, Any],
    *,
    run_dir: Path,
    gh_api: Any,
    repo_clone: Path,
    window_days: int,
) -> AnnotationPayload:
    """Build one run's bitemporal annotation payload (pure of DB writes).

    Composes the posterior rubric (PR vs local-branch, mirroring the former
    labeler's assembly), derives the outcome label, assembles the intrinsic
    :class:`ScoringInputs` from bronze, and scores a
    :class:`~daydream.training.reward.RewardBreakdown`. Returns a frozen
    :class:`AnnotationPayload`; the orchestrator persists it. Posterior-fetch
    and git errors propagate to the caller (the orchestrator isolates per-row).

    Args:
        row: The indexed manifest row (carries ``session_id``, the PR
            discriminators, ``head_sha``, ``grounding_rate``, etc.).
        run_dir: The archived run directory (bronze bundle root) feeding the
            intrinsic reward signals.
        gh_api: Callable invoked as ``gh_api(repo, endpoint, **kwargs)`` by
            the PR posterior signals; unused on the local-branch path.
        repo_clone: Local clone root for the fix-applied / local-commit
            cascades.
        window_days: Lookback window for the fix-applied cascade.

    Returns:
        A frozen :class:`AnnotationPayload`. ``valid_at`` is the PR merge
        timestamp for PR rows and ``None`` for non-PR/local rows.

    Raises:
        Exception: Any posterior-fetch (``gh_api``) or git error from the
            rubric assembly propagates unchanged to the caller.
    """
    if _row_is_pr(row):
        rubric = _build_rubric_pr(
            row,
            gh_api=gh_api,
            repo_clone=repo_clone,
            window_days=window_days,
        )
        valid_at = rubric.pr_merge.merged_at
    else:
        rubric = _build_rubric_local(row, repo_clone=repo_clone)
        valid_at = None

    outcome_label = derive_outcome_label(rubric)
    labels = [outcome_label] if outcome_label != "unknown" else []

    inputs = assemble_scoring_inputs(run_dir, row)
    rb = score_trajectory(inputs)

    return AnnotationPayload(
        labels=labels,
        pr_state=_pr_state_for_rubric(rubric),
        valid_at=valid_at,
        reward_version=reward.REWARD_VERSION,
        reward_json=json.dumps(rb.to_dict()),
        composite_reward=rb.composite,
        evidence_sha=row.get("head_sha"),
    )

"""Labeler orchestrator with PR vs local-branch dispatch.

Walks the archive index, dispatches each indexed run to either the
PR-review posterior path (``posterior_source = "pr_review"``) or the
local-branch posterior path (``posterior_source = "local_branch"``;
see ``/tmp/research-no-pr.md``), composes the signal extractors from
:mod:`daydream.training.labeler_signals` into a
:class:`~daydream.training.rubric.Rubric`, derives the outcome label
via :func:`~daydream.training.rubric.derive_outcome_label`, and writes
an immutable observation row through
:func:`daydream.archive.index.append_label_observation`.

Operational guarantees:

* Restart-safe: completed sessions are tracked in
  :class:`~daydream.training.backfill_cache.BackfillCache`'s
  ``progress.jsonl``; resumed runs skip them.
* Append-only: every label decision is written as a fresh row in
  ``label_observations``; the denormalized ``runs.outcome_labels``
  cache is refreshed in the same transaction.
* Per-row error isolation: an exception on one row counts in the
  ``errors`` summary and does not derail subsequent rows.
* Configuration errors (missing ``archive_dir``) raise before the
  loop begins.

The version constant ``LABELER_VERSION`` is bumped whenever the rubric
shape, signal extractors, or outcome-derivation logic changes — every
``label_observations`` row records the version that wrote it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from daydream import git_ops
from daydream.archive.index import append_label_observation, latest_label_observation, query_runs
from daydream.training.backfill_cache import BackfillCache
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    pr_merge_signal,
)
from daydream.training.rubric import Rubric, derive_outcome_label
from daydream.ui import create_console, print_warning

LABELER_VERSION = "2026.05.22-1"
"""Bump on any change to rubric shape, signal semantics, or label derivation."""


# ---------------------------------------------------------------------------
# Wrappers — module-level so tests can monkeypatch them as injection seams.
# ---------------------------------------------------------------------------


def _gh_api(repo: str, endpoint: str, **kwargs: Any) -> Any:
    """Proxy to :func:`daydream.git_ops.gh_api` keyed by ``repo`` slug.

    The signal extractors call ``gh_api(repo, endpoint, **kwargs)`` with
    ``repo`` as a slug string (``"owner/name"``). :func:`git_ops.gh_api`
    takes a ``Path`` as its first argument because it uses ``cwd=repo``
    for the shell-out. We adapt by using ``Path(".")`` — ``gh api``
    works from any cwd because it authenticates against the GitHub host
    configured in ``gh auth``, not the local repo.
    """
    return git_ops.gh_api(Path("."), endpoint, **kwargs)


def _diff_name_only(repo: Path, base: str, head: str) -> list[str]:
    """Proxy to :func:`daydream.git_ops.diff_name_only`."""
    return git_ops.diff_name_only(repo, base, head)


def _commits_in_window(repo: Path, head: str, base: str, days: int) -> list[str]:
    """Return commits on ``base`` since ``head``'s ancestor, within ``days``.

    Used by :func:`fix_applied_signal` to bound the upstream review
    window. Implemented inline because ``git_ops`` does not expose a
    direct equivalent; the labeler is the only consumer.
    """
    args = [
        "log",
        f"--since={days} days ago",
        "--pretty=%H",
        f"{head}..{base}",
    ]
    try:
        proc = git_ops._run_git(repo, args, timeout=10)  # noqa: SLF001 - shared helper
    except git_ops.GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _commits_since(repo: Path, branch: str, since: str) -> list[str]:
    """Return commits on ``branch`` after ``since``.

    Used by the local-branch posterior path to walk commits the user
    pushed after the daydream-recorded ``head_sha``.
    """
    args = ["log", "--pretty=%H", f"{since}..{branch}"]
    try:
        proc = git_ops._run_git(repo, args, timeout=10)  # noqa: SLF001 - shared helper
    except git_ops.GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _file_at(repo: Path, path: str, sha: str) -> str:
    """Return file content at ``sha``; empty string on missing path."""
    try:
        return git_ops.show(repo, sha, path).decode("utf-8", errors="replace")
    except git_ops.GitError:
        return ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelerConfig:
    """Configuration for a single ``run_label`` invocation.

    Attributes:
        archive_dir: Path to the daydream archive root (contains
            ``index.db``).
        dry_run: When ``True``, the loop computes labels but does not
            write to ``label_observations`` or the resume log.
        cache_dir: Optional directory backing
            :class:`~daydream.training.backfill_cache.BackfillCache`. When
            ``None``, ``gh_api`` calls hit the network on every row.
        repo_clone_root: Optional root directory under which per-repo
            clones live (used by the fix-applied cascade). Falls back to
            the archive root when unset.
        session_filter: Optional ``session_id`` prefix to restrict the
            queue (mirrors ``daydream export-jsonl --session``).
        fix_applied_window_days: Lookback window for upstream commits
            considered by the fix-applied cascade.
        gh_request_spacing_sec: Sleep duration between rows to spread
            ``gh api`` calls and stay under GitHub's secondary rate
            limits.
    """

    archive_dir: Path
    dry_run: bool = False
    cache_dir: Path | None = None
    repo_clone_root: Path | None = None
    session_filter: str | None = None
    fix_applied_window_days: int = 30
    gh_request_spacing_sec: float = 0.8


# ---------------------------------------------------------------------------
# Helpers — local-branch verdict and rubric assembly
# ---------------------------------------------------------------------------


_FIX_APPLIED_STUB = FixAppliedSignal(
    verdict="unknown",
    hunks_applied=0,
    hunks_total=0,
    window_commits=[],
)
"""Returned when the fix-applied cascade cannot run (missing diff.patch,
empty changed_files, or any subprocess error). The rubric still carries
the field for schema stability; outcome derivation does not depend on
it for the PR-review path."""


def _added_lines(diff_text: str) -> list[str]:
    """Extract content of ``+`` lines from a unified-diff blob.

    Strips the leading ``+`` and any single leading space (which is the
    conventional unified-diff content marker). Excludes ``+++`` file
    headers. Whitespace-only lines are dropped so they don't poison the
    substring check.
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

    Returns ``"applied"`` if any commit in ``commits`` produces a file
    content (via ``file_at_fetcher``) that contains every added line
    from ``diff_text`` as a substring. ``"rejected"`` if commits exist
    but none match; ``"unknown"`` when no commits to inspect.
    """
    if not commits:
        return LocalCommitAppliedSignal(verdict="rejected")
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

    The cascade needs ``archive_path/diff.patch`` to exist. When the
    archive directory is missing (older runs, dry-fixtures), or when
    ``changed_files`` is empty, return :data:`_FIX_APPLIED_STUB`.
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


def _pr_state_for_rubric(rubric: Rubric) -> str | None:
    """Map a PR-review rubric to a sqlite ``pr_state`` discriminator.

    For local-branch rubrics returns ``None`` so the column reflects
    "no PR associated".
    """
    if rubric.posterior_source != "pr_review":
        return None
    return "merged" if rubric.pr_merge.merged else "closed"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_label(config: LabelerConfig) -> dict[str, int]:
    """Walk the archive and label every unlabeled run.

    Args:
        config: A :class:`LabelerConfig` instance.

    Returns:
        Summary dict with keys ``considered``, ``labeled``,
        ``would_label``, ``skipped``, ``errors``.

    Raises:
        FileNotFoundError: When ``config.archive_dir`` does not exist.
            Configuration errors deliberately surface before the loop.
    """
    if not config.archive_dir.exists():
        msg = f"archive_dir does not exist: {config.archive_dir}"
        raise FileNotFoundError(msg)

    # Build the queue: every indexed run, optionally prefix-filtered.
    if config.session_filter:
        queue = query_runs(
            config.archive_dir,
            "session_id LIKE ? || '%'",
            (config.session_filter,),
        )
    else:
        queue = query_runs(config.archive_dir)

    # Skip rows that already carry a label observation; this matches
    # "needs labeling" semantics without an explicit LEFT JOIN.
    queue = [
        row
        for row in queue
        if latest_label_observation(config.archive_dir, row["session_id"]) is None
    ]

    # Cache + resume integration.
    cache: BackfillCache | None = None
    gh_api_callable: Any = _gh_api
    if config.cache_dir is not None:
        cache = BackfillCache(cache_dir=config.cache_dir, inner=_gh_api)
        gh_api_callable = cache
        done = cache.completed_sessions()
        queue = [row for row in queue if row["session_id"] not in done]

    repo_clone = config.repo_clone_root or config.archive_dir

    summary = {
        "considered": len(queue),
        "labeled": 0,
        "would_label": 0,
        "skipped": 0,
        "errors": 0,
    }

    console = create_console()

    for row in queue:
        try:
            if _row_is_pr(row):
                rubric = _build_rubric_pr(
                    row,
                    gh_api=gh_api_callable,
                    repo_clone=repo_clone,
                    window_days=config.fix_applied_window_days,
                )
            else:
                rubric = _build_rubric_local(row, repo_clone=repo_clone)

            outcome_label = derive_outcome_label(rubric)
            labels = [outcome_label] if outcome_label != "unknown" else []

            if config.dry_run:
                summary["would_label"] += 1
            else:
                append_label_observation(
                    config.archive_dir,
                    row["session_id"],
                    labels=labels,
                    pr_state=_pr_state_for_rubric(rubric),
                    labeler_version=LABELER_VERSION,
                    evidence_sha=row.get("head_sha"),
                    rubric_json=json.dumps(rubric.to_dict()),
                )
                summary["labeled"] += 1
                if cache is not None:
                    cache.mark_session_done(row["session_id"])
        except Exception as exc:  # noqa: BLE001 - per-row isolation by design
            summary["errors"] += 1
            print_warning(
                console,
                f"labeler: session {row.get('session_id', '<unknown>')} failed: "
                f"{type(exc).__name__}: {exc}",
            )
            continue

        await anyio.sleep(config.gh_request_spacing_sec)

    return summary



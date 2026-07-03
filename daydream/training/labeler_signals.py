"""Posterior-signal extractors for the labeler.

Each signal is a pure function over ``(manifest_row, fetcher)`` — no LLM,
no I/O beyond the fetcher callables the caller injects. This module
exposes four signals that the labeling pipeline composes into outcome
labels:

* :func:`pr_merge_signal` — was the originating PR merged?
* :func:`fix_applied_signal` — did the recommended diff land in the
  upstream default branch within a configurable window? Implements the
  layered cascade documented in ``docs/signals/fix-applied.md``.
* :func:`comment_resolution_signal` — did bot review comments get a
  reply (proxy for "addressed")?
* :func:`local_commit_applied_signal` — for PR-less runs (see
  ``docs/signals/no-pr.md``), did a later local commit on the same
  branch carry the recommended diff content?

Each signal returns a frozen dataclass so callers can hash / compare
results across re-labels. Fetchers are passed as keyword-only callables
to keep the functions testable without monkeypatching subprocess /
HTTP calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from daydream.pr_review import parse_finding_markers

# Version-stable footer prefix: matching only this prefix (not the full
# version-pinned DAYDREAM_FOOTER) recognises comments from any daydream release.
_DAYDREAM_FOOTER_PREFIX = "<sub>🧙 Posted by [daydream v"


def _is_daydream_comment(comment: dict[str, Any]) -> bool:
    """Return True when ``comment`` was authored by daydream.

    Daydream posts review comments as a normal authenticated user, so
    authorship is identified by the :data:`DAYDREAM_FOOTER` badge in the
    comment body rather than a ``*[bot]`` login.

    The match is made against the version-stable prefix of the footer
    (``_DAYDREAM_FOOTER_PREFIX``) rather than the full version-pinned
    :data:`DAYDREAM_FOOTER` constant, so that comments posted by any release
    of daydream are correctly identified even when their embedded version
    string differs from the currently-installed package.

    Args:
        comment: A parsed PR review comment dict (``body`` field read).

    Returns:
        True iff the comment body contains the daydream footer badge.
    """
    return _DAYDREAM_FOOTER_PREFIX in (comment.get("body") or "")


# Result dataclasses


@dataclass(frozen=True)
class PRMergeSignal:
    """Whether the originating PR was merged.

    Attributes:
        merged: ``True`` if the PR is marked merged on GitHub.
        merged_at: ISO-8601 timestamp of merge, or ``None``.
    """

    merged: bool
    merged_at: str | None


@dataclass(frozen=True)
class FixAppliedSignal:
    """Result of the fix-applied layered cascade.

    Attributes:
        verdict: ``"applied"`` if ≥50% of recommended hunks landed in the
            upstream default branch within the window; ``"not_applied"``
            if the window was non-empty but no hunks landed; ``"unknown"``
            if no window commits exist.
        hunks_applied: Count of hunks whose added lines appear verbatim
            in the post-window file content.
        hunks_total: Total hunks parsed from ``recommended.patch`` (daydream's
            proposed diff); ``diff.patch`` is the legacy-only fallback.
        window_commits: Ordered commit SHAs considered (oldest → newest).
    """

    verdict: Literal["applied", "not_applied", "unknown"]
    hunks_applied: int
    hunks_total: int
    window_commits: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommentResolutionSignal:
    """PR review-comment resolution proxy.

    Attributes:
        total: Top-level bot comments on the PR.
        replied: Top-level bot comments that received at least one reply.
        unresolved: ``total - replied``.
    """

    total: int
    replied: int
    unresolved: int


@dataclass(frozen=True)
class PerFindingResolution:
    """Resolution state for one recorded finding, joined by fingerprint.

    Attributes:
        fingerprint: The 64-hex finding fingerprint recorded at review time.
        resolved: ``True`` when the finding's posted comment received at
            least one reply (proxy for "addressed").
        comment_id: GitHub review-comment ID carrying the finding marker,
            or ``None`` when no surviving comment carries the fingerprint
            (deleted, edited away, or never posted).
    """

    fingerprint: str
    resolved: bool
    comment_id: int | None


@dataclass(frozen=True)
class LocalCommitAppliedSignal:
    """Posterior signal for PR-less runs (local branch only).

    Attributes:
        verdict: ``"applied"`` if a later local commit contains the
            recommended diff content; ``"rejected"`` if no qualifying
            commit exists; ``"unknown"`` if the repo clone is missing.
    """

    verdict: Literal["applied", "rejected", "unknown"]


# Diff parsing helpers


@dataclass(frozen=True)
class _Hunk:
    """One hunk parsed from a unified diff."""

    file: str
    added_lines: tuple[str, ...]


_DIFF_FILE_HEADER = re.compile(r"^diff --git a/(?P<old>.+) b/(?P<new>.+)$")


def _parse_diff_hunks(patch_text: str) -> list[_Hunk]:
    """Parse a unified-diff text into a flat list of hunks.

    Each ``@@`` block becomes one :class:`_Hunk` carrying the added lines
    (without the leading ``+``). Lines that begin with ``+++`` are
    treated as file headers, not additions.
    """
    hunks: list[_Hunk] = []
    current_file: str | None = None
    in_hunk = False
    added: list[str] = []

    def _flush() -> None:
        nonlocal added
        if current_file is not None and added:
            hunks.append(_Hunk(file=current_file, added_lines=tuple(added)))
        added = []

    for line in patch_text.splitlines():
        header = _DIFF_FILE_HEADER.match(line)
        if header:
            _flush()
            current_file = header.group("new")
            in_hunk = False
            continue
        if line.startswith("+++ "):
            # File header inside a diff block — already captured via "diff --git".
            in_hunk = False
            continue
        if line.startswith("--- "):
            in_hunk = False
            continue
        if line.startswith("@@"):
            _flush()
            in_hunk = True
            continue
        if in_hunk and line.startswith("+"):
            added.append(line[1:])
    _flush()
    return hunks


def _hunk_lines_present(added_lines: tuple[str, ...], post_content: str) -> bool:
    """Return ``True`` if every added line appears verbatim in ``post_content``."""
    if not added_lines:
        return False
    haystack_lines = set(post_content.splitlines())
    return all(line in haystack_lines for line in added_lines)


def _archive_is_recommended_patch_aware(archive_path: Path) -> bool:
    """Return True if *archive_path*'s manifest was written by recommended.patch-aware daydream.

    Reads the archived ``manifest.json`` provenance flag
    ``recommended_patch_supported``. When set, a missing ``recommended.patch``
    means the run made no recommendation (review-only, all-declined, reverted,
    or wash) — not a legacy archive — so callers must NOT fall back to
    ``diff.patch`` (the PR-under-review diff). A missing or unparseable manifest
    is treated as legacy, preserving the backward-compatible ``diff.patch``
    fallback for archives created before ``recommended.patch`` existed.
    """
    try:
        data = json.loads((archive_path / "manifest.json").read_text())
    except (OSError, ValueError):
        return False
    return data.get("recommended_patch_supported") is True


def _read_recommended_patch(archive_path: Path) -> str:
    """Read daydream's recommended-change patch for a run.

    Prefers ``recommended.patch`` — daydream's proposed diff, captured post-fix
    — so the applied-signal cascades score the RECOMMENDED changes rather than
    the PR-under-review diff. When ``recommended.patch`` is absent the behaviour
    depends on archive provenance (``manifest.json`` →
    ``recommended_patch_supported``):

    * **New-format archive** (flag set): absence means daydream made no
      recommendation (review-only, all-declined, reverted, or wash). Returns
      ``""`` so the cascade scores no hunks — it must NOT fall back to
      ``diff.patch``, which is the PR-under-review diff and would mislabel such
      runs as "applied".
    * **Legacy archive** (flag absent / no manifest): falls back to
      ``diff.patch`` for backward compatibility with archives created before
      ``recommended.patch`` existed.

    Args:
        archive_path: The archived run directory (bronze bundle root).

    Returns:
        The unified-diff text to parse into recommended hunks (``""`` for a
        new-format run that made no recommendation).

    Raises:
        OSError: For a legacy archive (no ``recommended.patch``) when
            ``diff.patch`` cannot be read. New-format archives with no
            ``recommended.patch`` return ``""`` rather than raising.
    """
    recommended = archive_path / "recommended.patch"
    if recommended.is_file():
        return recommended.read_text()
    if _archive_is_recommended_patch_aware(archive_path):
        return ""
    return (archive_path / "diff.patch").read_text()


# Signal extractors


def pr_merge_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> PRMergeSignal:
    """Return whether the row's originating PR was merged.

    Args:
        row: Manifest row carrying ``pr_repo`` and ``pr_number``.
        gh_api: Callable invoked as ``gh_api(repo, endpoint)`` returning
            the parsed JSON body. Exceptions propagate to the caller.

    Returns:
        :class:`PRMergeSignal`. When ``pr_repo`` or ``pr_number`` is
        ``None`` the signal is ``(False, None)`` without any API call.
    """
    repo = row.get("pr_repo")
    number = row.get("pr_number")
    if repo is None or number is None:
        return PRMergeSignal(merged=False, merged_at=None)
    payload = gh_api(repo, f"repos/{repo}/pulls/{number}")
    return PRMergeSignal(
        merged=bool(payload.get("merged", False)),
        merged_at=payload.get("merged_at"),
    )


def fix_applied_signal(
    row: dict[str, Any],
    *,
    changed_files: list[str],
    repo_clone: Path,
    diff_fetcher: Callable[[Path, str, str], list[str]],
    commits_in_window_fetcher: Callable[[Path, str, str], list[str]],
    file_at_fetcher: Callable[[Path, str, str], str],
) -> FixAppliedSignal:
    """Run the layered cascade for "did the recommended fix land upstream?".

    Cascade (see ``/tmp/research-fix-applied.md``):

    1. Compute the commit window via ``commits_in_window_fetcher``. The
       ``head..base`` range already bounds the walk; no date filter is
       needed (see #167). If empty → ``verdict="unknown"``.
    2. Compute the file overlap between ``changed_files`` and the files
       actually touched by ``diff_fetcher``. If empty → ``not_applied``
       with ``hunks_applied=0`` and ``hunks_total = parsed hunks``.
    3. Parse the recommended-change patch (``archive_path/recommended.patch``,
       falling back to ``diff.patch`` for pre-recommended.patch archives) into
       hunks. For each hunk on a
       file in the overlap, read ``file_at_fetcher(repo, file, window[-1])``
       and check whether every added line appears verbatim.
    4. Verdict: ``applied`` if ``hunks_applied / hunks_total >= 0.5``;
       otherwise ``not_applied``.

    Args:
        row: Manifest row with ``head_sha``, ``base_branch``, ``archive_path``.
        changed_files: Files the agent originally recommended changing.
        repo_clone: Path to a local clone of the target repo.
        diff_fetcher: Returns files touched between ``base`` and ``head``.
        commits_in_window_fetcher: Returns ordered commit SHAs in the
            review window. Called as ``(repo_clone, head_sha,
            base_branch)``.
        file_at_fetcher: Returns file content at a given SHA.

    Returns:
        :class:`FixAppliedSignal` with ``verdict="unknown"`` when the
        commit window is empty, ``"not_applied"`` when no recommended
        files were touched or fewer than half of the parsed hunks
        appear post-window, and ``"applied"`` otherwise.
        ``hunks_applied``, ``hunks_total``, and ``window_commits`` are
        always populated.

    Raises:
        KeyError: If ``row`` is missing ``head_sha``, ``base_branch``,
            or ``archive_path``.
        OSError: For a legacy archive (no ``recommended.patch``) when
            ``diff.patch`` cannot be read from ``archive_path``. New-format
            archives with no ``recommended.patch`` yield an empty patch rather
            than raising (see :func:`_read_recommended_patch`).
        Exception: Any exception raised by ``diff_fetcher``,
            ``commits_in_window_fetcher``, or ``file_at_fetcher`` is
            propagated unchanged.
    """
    head_sha = row["head_sha"]
    base_branch = row["base_branch"]
    archive_path = Path(row["archive_path"])

    diff_patch = _read_recommended_patch(archive_path)
    hunks = _parse_diff_hunks(diff_patch)
    hunks_total = len(hunks)

    window = commits_in_window_fetcher(repo_clone, head_sha, base_branch)
    if not window:
        return FixAppliedSignal(
            verdict="unknown",
            hunks_applied=0,
            hunks_total=hunks_total,
            window_commits=[],
        )

    touched = diff_fetcher(repo_clone, head_sha, base_branch)
    overlap = set(changed_files) & set(touched)
    if not overlap:
        return FixAppliedSignal(
            verdict="not_applied",
            hunks_applied=0,
            hunks_total=hunks_total,
            window_commits=list(window),
        )

    hunks_applied = 0
    post_sha = window[-1]
    file_content_cache: dict[str, str] = {}
    for hunk in hunks:
        if hunk.file not in overlap:
            continue
        post_content = file_content_cache.get(hunk.file)
        if post_content is None:
            post_content = file_at_fetcher(repo_clone, hunk.file, post_sha)
            file_content_cache[hunk.file] = post_content
        if _hunk_lines_present(hunk.added_lines, post_content):
            hunks_applied += 1

    if hunks_total > 0 and (hunks_applied / hunks_total) >= 0.5:
        verdict: Literal["applied", "not_applied", "unknown"] = "applied"
    else:
        verdict = "not_applied"

    return FixAppliedSignal(
        verdict=verdict,
        hunks_applied=hunks_applied,
        hunks_total=hunks_total,
        window_commits=list(window),
    )


@dataclass(frozen=True)
class PRCommentThreads:
    """Indexed view of a PR's review comments shared by the resolution signals.

    Centralises the ``/comments`` fetch and daydream-thread indexing so the
    aggregate (:func:`comment_resolution_signal`) and per-finding
    (:func:`per_finding_resolution_signal`) signals do not refetch or
    re-index the same row's comments. Built by
    :func:`index_pr_review_comments`.

    Attributes:
        top_level_daydream_ids: IDs of footer-marked daydream comments with
            no parent (the "issue" comments).
        replied_ids: Subset of ``top_level_daydream_ids`` that received at
            least one reply.
        comment_id_by_fingerprint: First comment ID carrying each finding
            marker, keyed by 64-hex fingerprint.
    """

    top_level_daydream_ids: set[int]
    replied_ids: set[int]
    comment_id_by_fingerprint: dict[str, int]


def index_pr_review_comments(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> PRCommentThreads | None:
    """Fetch and index a PR's review comments for the resolution signals.

    Single source of truth for the ``repos/{repo}/pulls/{n}/comments`` fetch
    and daydream-thread indexing shared by :func:`comment_resolution_signal`
    and :func:`per_finding_resolution_signal`, so the two signals invoked
    back-to-back on the same row hit the endpoint once (the caller computes
    this and passes it as ``threads=`` to both).

    Args:
        row: Manifest row carrying ``pr_repo`` and ``pr_number``.
        gh_api: Callable returning the parsed comment list. Exceptions
            propagate to the caller.

    Returns:
        :class:`PRCommentThreads`, or ``None`` when the row has no
        associated PR (no fetch performed).
    """
    repo = row.get("pr_repo")
    number = row.get("pr_number")
    if repo is None or number is None:
        return None

    comments = gh_api(repo, f"repos/{repo}/pulls/{number}/comments", paginate=True)

    # Pass 1: top-level daydream comment IDs and the fingerprints they carry.
    top_level_daydream_ids: set[int] = set()
    comment_id_by_fingerprint: dict[str, int] = {}
    for comment in comments:
        if comment.get("in_reply_to_id") is None and _is_daydream_comment(comment):
            cid = comment["id"]
            top_level_daydream_ids.add(cid)
            for fingerprint in parse_finding_markers(comment.get("body") or ""):
                comment_id_by_fingerprint.setdefault(fingerprint, cid)

    # Pass 2: which top-level comments received a reply.
    replied_ids: set[int] = set()
    for comment in comments:
        in_reply_to = comment.get("in_reply_to_id")
        if in_reply_to in top_level_daydream_ids:
            replied_ids.add(in_reply_to)

    return PRCommentThreads(
        top_level_daydream_ids=top_level_daydream_ids,
        replied_ids=replied_ids,
        comment_id_by_fingerprint=comment_id_by_fingerprint,
    )


def comment_resolution_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
    threads: PRCommentThreads | None = None,
) -> CommentResolutionSignal:
    """Return a proxy for "review comments addressed".

    Top-level review comments authored by daydream (identified by the
    :data:`DAYDREAM_FOOTER` badge in the comment body) are treated as
    issues; any reply (regardless of author or body) marks the issue
    resolved.

    Args:
        row: Manifest row carrying ``pr_repo`` and ``pr_number``.
        gh_api: Callable returning the parsed comment list. Ignored when
            ``threads`` is supplied.
        threads: Pre-indexed comment threads from
            :func:`index_pr_review_comments`. When supplied, the
            ``/comments`` fetch is skipped, letting a caller that needs
            both this and :func:`per_finding_resolution_signal` on one row
            fetch the endpoint once.

    Returns:
        :class:`CommentResolutionSignal` with ``(0, 0, 0)`` when the row
        has no associated PR.
    """
    if threads is None:
        threads = index_pr_review_comments(row, gh_api=gh_api)
    if threads is None:
        return CommentResolutionSignal(total=0, replied=0, unresolved=0)
    total = len(threads.top_level_daydream_ids)
    replied = len(threads.replied_ids)
    return CommentResolutionSignal(total=total, replied=replied, unresolved=total - replied)


def per_finding_resolution_signal(
    row: dict[str, Any],
    *,
    recorded_fingerprints: list[str],
    gh_api: Callable[..., Any],
    threads: PRCommentThreads | None = None,
) -> list[PerFindingResolution]:
    """Resolve each recorded finding to its posted comment and reply state.

    Joins the fingerprints recorded at review time (``recorded_fingerprints``)
    against the PR's live review comments by the hidden
    ``<!-- daydream-finding: <fp> -->`` marker embedded in each daydream
    comment body. A finding is ``resolved`` when its comment received at
    least one reply; a fingerprint with no surviving comment yields
    ``comment_id=None`` (deleted / edited away / never posted).

    Args:
        row: Manifest row carrying ``pr_repo`` and ``pr_number``.
        recorded_fingerprints: Fingerprints captured at review time. The
            returned list preserves this order, one entry per fingerprint.
        gh_api: Callable returning the parsed comment list. Ignored when
            ``threads`` is supplied.
        threads: Pre-indexed comment threads from
            :func:`index_pr_review_comments`. When supplied, the
            ``/comments`` fetch is skipped, letting a caller that needs
            both this and :func:`comment_resolution_signal` on one row
            fetch the endpoint once.

    Returns:
        One :class:`PerFindingResolution` per recorded fingerprint, or an
        empty list when the row has no associated PR.
    """
    if threads is None:
        threads = index_pr_review_comments(row, gh_api=gh_api)
    if threads is None:
        return []

    resolutions: list[PerFindingResolution] = []
    for fingerprint in recorded_fingerprints:
        comment_id = threads.comment_id_by_fingerprint.get(fingerprint)
        resolved = comment_id is not None and comment_id in threads.replied_ids
        resolutions.append(
            PerFindingResolution(fingerprint=fingerprint, resolved=resolved, comment_id=comment_id)
        )
    return resolutions


def pr_link_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> tuple[int, str] | None:
    """Resolve an orphan run's PR by its ``head_sha`` (SHA-native lookup).

    A run launched before its PR existed has ``pr_number=None`` but carries
    ``repo_slug`` + ``head_sha``. This looks up the PR(s) whose head commit
    is ``head_sha`` via ``repos/{slug}/commits/{sha}/pulls`` and returns the
    one whose head matches exactly, disambiguating reused branch names.

    Args:
        row: Manifest row carrying ``repo_slug`` and ``head_sha``.
        gh_api: Callable returning the parsed pull list for the commit.

    Returns:
        ``(pr_number, repo_slug)`` for the first pull whose head matches
        ``head_sha``; ``None`` when required fields are missing or no pull
        matches. Exceptions from ``gh_api`` propagate to the caller.
    """
    repo_slug = row.get("repo_slug")
    head_sha = row.get("head_sha")
    if not repo_slug or not head_sha:
        return None

    pulls = gh_api(repo_slug, f"repos/{repo_slug}/commits/{head_sha}/pulls", paginate=True)
    for pr in pulls:
        if pr.get("head", {}).get("sha") == head_sha:
            pr_number = pr.get("number")
            if pr_number is None:
                continue
            pr_repo = pr.get("head", {}).get("repo", {}).get("full_name") or repo_slug
            return int(pr_number), pr_repo
    return None


def local_commit_applied_signal(
    row: dict[str, Any],
    *,
    repo_clone: Path,
    commits_since_fetcher: Callable[[Path, str, str], list[str]],
    file_at_fetcher: Callable[[Path, str, str], str],
) -> LocalCommitAppliedSignal:
    """Posterior signal for PR-less runs.

    See ``/tmp/research-no-pr.md``. Walks the commits on ``row["branch"]``
    that landed after ``row["head_sha"]`` and checks whether any commit
    contains the added lines from the recommended-change patch
    (``recommended.patch``, falling back to ``diff.patch`` for older archives).

    Args:
        row: Manifest row with ``branch``, ``head_sha``, ``archive_path``.
        repo_clone: Path to local clone. ``"unknown"`` if not a directory.
        commits_since_fetcher: Returns ordered commit SHAs on ``branch``
            after ``since_sha``.
        file_at_fetcher: Returns file content at a given SHA.

    Returns:
        :class:`LocalCommitAppliedSignal` with ``verdict="unknown"``
        when ``repo_clone`` is not a directory, ``"rejected"`` when no
        commits follow ``head_sha`` or none contain the recommended
        hunk's added lines, and ``"applied"`` when at least one does.

    Raises:
        KeyError: If ``row`` is missing ``archive_path``, ``branch``,
            or ``head_sha``.
        OSError: For a legacy archive (no ``recommended.patch``) when
            ``diff.patch`` cannot be read from ``archive_path``. New-format
            archives with no ``recommended.patch`` yield an empty patch rather
            than raising (see :func:`_read_recommended_patch`).
        Exception: Any exception raised by ``commits_since_fetcher`` or
            ``file_at_fetcher`` is propagated unchanged.
    """
    if not repo_clone.is_dir():
        return LocalCommitAppliedSignal(verdict="unknown")

    archive_path = Path(row["archive_path"])
    diff_patch = _read_recommended_patch(archive_path)
    hunks = _parse_diff_hunks(diff_patch)

    commits = commits_since_fetcher(repo_clone, row["branch"], row["head_sha"])
    if not commits:
        return LocalCommitAppliedSignal(verdict="rejected")

    for commit in commits:
        file_content_cache: dict[str, str] = {}
        for hunk in hunks:
            content = file_content_cache.get(hunk.file)
            if content is None:
                content = file_at_fetcher(repo_clone, hunk.file, commit)
                file_content_cache[hunk.file] = content
            if _hunk_lines_present(hunk.added_lines, content):
                return LocalCommitAppliedSignal(verdict="applied")

    return LocalCommitAppliedSignal(verdict="rejected")


def reviewer_logins_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> list[str]:
    """Return the human reviewer logins associated with the row's PR.

    A "reviewer" is a human GitHub account that either authored a PR
    review (``/pulls/{n}/reviews``) or replied to one of daydream's
    footer-marked top-level review comments (``/pulls/{n}/comments``).
    The union is taken, then ``[bot]`` logins and any login that authored
    a daydream-footer comment are excluded. ``merged_by`` is **not** used
    (a merge author is not a reviewer).

    Args:
        row: Manifest row carrying ``pr_repo`` and ``pr_number``.
        gh_api: Callable invoked as ``gh_api(repo, endpoint)`` returning
            the parsed JSON body. Exceptions propagate to the caller.

    Returns:
        Sorted, deduped list of human reviewer logins. Empty list when
        the row has no associated PR or no reviewers were found.

    Raises:
        Exception: Any exception raised by ``gh_api`` is propagated
            unchanged; failures are never swallowed into ``[]``.
    """
    repo = row.get("pr_repo")
    number = row.get("pr_number")
    if repo is None or number is None:
        return []

    logins: set[str] = set()
    excluded: set[str] = set()

    # (a) Authors of PR reviews.
    reviews = gh_api(repo, f"repos/{repo}/pulls/{number}/reviews", paginate=True)
    for review in reviews:
        user = review.get("user") or {}
        login = user.get("login", "")
        if login:
            logins.add(login)

    # (b) Authors of replies to daydream's footer-marked top-level comments.
    comments = gh_api(repo, f"repos/{repo}/pulls/{number}/comments", paginate=True)
    daydream_comment_ids: set[int] = set()
    replies_by_parent: dict[int, list[str]] = {}
    for comment in comments:
        in_reply_to = comment.get("in_reply_to_id")
        user = comment.get("user") or {}
        login = user.get("login", "")
        if in_reply_to is None:
            if _is_daydream_comment(comment):
                daydream_comment_ids.add(comment["id"])
                if login:
                    excluded.add(login)
        else:
            if login:
                replies_by_parent.setdefault(in_reply_to, []).append(login)

    for parent_id in daydream_comment_ids:
        logins.update(replies_by_parent.get(parent_id, []))

    # Exclude bots and any login that authored a daydream-footer comment.
    humans = {
        login for login in logins if not login.endswith("[bot]") and login not in excluded
    }
    return sorted(humans)

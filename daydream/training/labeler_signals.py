"""Posterior signals from archived recommendations and injected Git/GitHub fetchers.

PR merge and reply counts provide context; per-finding qualifying replies
provide semantic outcomes. Applied-change signals inspect recommended hunks
on upstream or local commits. Results are frozen dataclasses; fetcher errors
propagate unless a signal explicitly defines an unavailable-data result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast, get_args

from daydream.hunk_index import parse_hunks
from daydream.pr_review import parse_finding_markers
from daydream.training._immutable_json import thaw_json
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.reply_classifier import (
    _user_str,
    classify_reply,
    qualification_reason,
)

# Version-stable footer prefix: matching only this prefix (not the full
# version-pinned DAYDREAM_FOOTER) recognises comments from any daydream release.
_DAYDREAM_FOOTER_PREFIX = "<sub>🧙 Posted by [daydream v"


def _is_daydream_comment(comment: dict[str, Any]) -> bool:
    """Recognize any Daydream release by its version-stable footer prefix.

    Reviews use ordinary user accounts, so a bot-shaped login is insufficient."""
    return _DAYDREAM_FOOTER_PREFIX in (comment.get("body") or "")


# Result dataclasses


@dataclass(frozen=True)
class PRMergeSignal:
    """PR merge/state context. author_login lets fork authors qualify even when
    their GitHub association is not OWNER, MEMBER, or COLLABORATOR."""

    merged: bool
    merged_at: str | None
    state: Literal["open", "closed", "merged", "unknown"] = "unknown"
    draft: bool = False
    author_login: str | None = None


@dataclass(frozen=True)
class FixAppliedSignal:
    """Recommended hunks found at the end of an oldest-to-newest commit window.

    Applied means at least half landed; an empty window is unknown. Hunk counts
    include the legacy diff.patch fallback only for recommendation-unaware archives."""

    verdict: Literal["applied", "not_applied", "unknown"]
    hunks_applied: int
    hunks_total: int
    window_commits: Sequence[str] = field(default_factory=list)


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


PerFindingDisposition = Literal["accepted", "rejected", "ambiguous", "unanswered", "missing"]


def _reply_evidence(
    replies: list[dict[str, Any]],
    pr_author_logins: frozenset[str],
    review_author_logins: frozenset[str],
) -> list[dict[str, Any]]:
    """Persist every reply as a full evidence entry — never reduced to a count (M3)."""
    evidence: list[dict[str, Any]] = []
    for reply in replies:
        if not isinstance(reply, dict):
            continue
        evidence.append(
            {
                "reply_id": reply.get("id"),
                "author": _user_str(reply, "login"),
                "author_association": reply.get("author_association", ""),
                "created_at": reply.get("created_at", ""),
                "body_sha256": hashlib.sha256((reply.get("body") or "").encode("utf-8")).hexdigest(),
                "reason": qualification_reason(reply, pr_author_logins, review_author_logins),
                # Per-reply classifier axis (``classify_reply`` output): the field
                # harvest's ``_decisive_evidence_valid_at`` filters on, so an earlier
                # qualifying-but-ambiguous reply never moves ``valid_at`` ahead of
                # the reply that actually decided the disposition (M12).
                "classifier_label": classify_reply(reply),
            }
        )
    return evidence


def _disposition_from_evidence(evidence: list[dict[str, Any]]) -> PerFindingDisposition:
    """Count only qualifying votes, using the same classification we persist."""
    labels = {entry["classifier_label"] for entry in evidence if not entry["reason"].startswith("excluded:")}
    votes = labels & {"accepted", "rejected"}
    if len(votes) == 1:
        return cast(PerFindingDisposition, votes.pop())
    return "ambiguous" if labels else "unanswered"


@dataclass(frozen=True)
class PerFindingResolution:
    """One fingerprint's disposition and digest-pinned reply evidence.

    A missing comment has comment_id=None and disposition=missing. Surviving
    threads become accepted/rejected from agreeing qualifying votes, ambiguous
    from conflicting or nondirectional votes, or unanswered without qualifying
    authors. Evidence retains every reply's identity, timestamp, body hash,
    qualification reason, and classifier label, including excluded replies."""

    fingerprint: str
    comment_id: int | None
    disposition: PerFindingDisposition
    evidence: Sequence[Mapping[str, Any]] = field(default_factory=list)
    evidence_digest: str = ""


def resolution_to_dict(r: PerFindingResolution) -> dict[str, Any]:
    """Serialize a ``PerFindingResolution`` to the canonical dict shape.

    Emits exactly the canonical keys; ``evidence`` is copied, never aliased.
    """

    return {
        "fingerprint": r.fingerprint,
        "comment_id": r.comment_id,
        "disposition": r.disposition,
        "evidence": [thaw_json(entry) for entry in r.evidence],
        "evidence_digest": r.evidence_digest,
    }


def resolution_from_dict(payload: Mapping[str, Any]) -> PerFindingResolution:
    """Rebuild a ``PerFindingResolution`` from the canonical dict shape.

    Fail-closed: missing required fields raise ``ValueError`` naming the
    field; never ``None``-coerced, no fallback substitution.
    """

    fingerprint = payload.get("fingerprint")
    if not fingerprint:
        raise ValueError("missing or empty fingerprint")
    disposition = payload.get("disposition")
    if disposition not in get_args(PerFindingDisposition):
        raise ValueError(f"invalid disposition: {disposition!r}")
    evidence_digest = payload.get("evidence_digest")
    if not evidence_digest:
        raise ValueError("missing or empty evidence_digest")
    return PerFindingResolution(
        fingerprint=fingerprint,
        comment_id=payload.get("comment_id"),
        disposition=cast(PerFindingDisposition, disposition),
        evidence=list(payload.get("evidence") or []),
        evidence_digest=evidence_digest,
    )


@dataclass(frozen=True)
class LocalCommitAppliedSignal:
    """Posterior signal for PR-less runs (local branch only).

    Attributes:
        verdict: ``"applied"`` if a later local commit contains the
            recommended diff content; ``"rejected"`` if no qualifying
            commit exists; ``"unknown"`` if the repo clone is missing or
            the commit window could not be read.
    """

    verdict: Literal["applied", "rejected", "unknown"]


# Diff parsing helpers


@dataclass(frozen=True)
class _Hunk:
    """One hunk parsed from a unified diff."""

    file: str
    added_lines: tuple[str, ...]


def _parse_diff_hunks(patch_text: str) -> list[_Hunk]:
    """Group the shared diff parser's added lines by file and owning hunk.

    Preserve new-line order and omit hunks without additions."""
    hunks: list[_Hunk] = []
    for path, meta in parse_hunks(patch_text).items():
        grouped: dict[int, list[str]] = {}
        for hunk_index, text in meta["added_text"].values():
            grouped.setdefault(hunk_index, []).append(text)
        for added in grouped.values():
            hunks.append(_Hunk(file=path, added_lines=tuple(added)))
    return hunks


def _hunk_lines_present(added_lines: tuple[str, ...], post_content: str) -> bool:
    """Return ``True`` if every added line appears verbatim in ``post_content``."""
    if not added_lines:
        return False
    haystack_lines = set(post_content.splitlines())
    return all(line in haystack_lines for line in added_lines)


def _count_present_hunks(
    repo_clone: Path,
    hunks: list[_Hunk],
    ref: str,
    file_at_fetcher: Callable[[Path, str, str], str],
    *,
    only_files: set[str] | None = None,
) -> int:
    """Count hunks whose added lines appear verbatim in each file at *ref*.

    Each distinct file is fetched once. ``only_files`` restricts the walk to a
    subset of paths (the fix-applied cascade's changed-file overlap).
    """
    present = 0
    cache: dict[str, str] = {}
    for hunk in hunks:
        if only_files is not None and hunk.file not in only_files:
            continue
        content = cache.get(hunk.file)
        if content is None:
            content = file_at_fetcher(repo_clone, hunk.file, ref)
            cache[hunk.file] = content
        if _hunk_lines_present(hunk.added_lines, content):
            present += 1
    return present


# Signal extractors


def pr_merge_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> PRMergeSignal:
    """Fetch merge/state context; absent PR identity returns an unmerged signal.

    Fetcher errors propagate and no request is made without both identity fields."""
    repo = row.get("pr_repo")
    number = row.get("pr_number")
    if repo is None or number is None:
        return PRMergeSignal(merged=False, merged_at=None)
    payload = gh_api(repo, f"repos/{repo}/pulls/{number}")
    user = payload.get("user") or {}
    return PRMergeSignal(
        merged=bool(payload.get("merged", False)),
        merged_at=payload.get("merged_at"),
        state=(
            "merged"
            if payload.get("merged")
            else payload.get("state")
            if payload.get("state") in ("open", "closed")
            else "unknown"
        ),
        draft=bool(payload.get("draft", False)),
        author_login=user.get("login"),
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
    """Check whether at least half the recommended hunks landed upstream.

    Read the recommended patch, then fetch oldest-to-newest commits in head..base.
    An empty window is unknown. Otherwise inspect the last commit, restricting
    reads to the intersection of changed_files and the fetched diff paths. With
    no overlap or fewer than half the hunks present, return not_applied.
    Every added line in a hunk must appear verbatim. Missing row keys, patch
    read errors, and fetcher failures propagate; counts always describe the patch."""
    head_sha = row["head_sha"]
    base_branch = row["base_branch"]
    diff_patch = row["recommended_patch"]
    hunks = _parse_diff_hunks(diff_patch)
    hunks_total = len(hunks)

    window = commits_in_window_fetcher(repo_clone, head_sha, base_branch)
    hunks_applied = 0
    verdict: Literal["applied", "not_applied", "unknown"] = "unknown"
    if window:
        touched = diff_fetcher(repo_clone, head_sha, base_branch)
        overlap = set(changed_files) & set(touched)
        if overlap:
            hunks_applied = _count_present_hunks(repo_clone, hunks, window[-1], file_at_fetcher, only_files=overlap)
        verdict = "applied" if hunks_total > 0 and hunks_applied / hunks_total >= 0.5 else "not_applied"
    return FixAppliedSignal(
        verdict=verdict,
        hunks_applied=hunks_applied,
        hunks_total=hunks_total,
        window_commits=list(window),
    )


@dataclass(frozen=True)
class PRCommentThreads:
    """One shared PR-comment index for aggregate and per-finding signals.

    IDs cover footer-marked top-level comments; an optional fingerprint scope
    excludes other runs' threads. Each fingerprint keeps its first comment id.
    replies_by_comment retains full reply objects for semantic classification;
    replied_ids is the subset with replies."""

    top_level_daydream_ids: set[int]
    replied_ids: set[int]
    comment_id_by_fingerprint: dict[str, int]
    replies_by_comment: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


def index_pr_review_comments(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
    session_fingerprints: list[str] | None = None,
) -> PRCommentThreads | None:
    """Fetch review comments once and index Daydream threads and their replies.

    A supplied fingerprint list restricts top-level comments to matching markers;
    None indexes all Daydream threads. Missing PR identity returns None without
    fetching. Fetcher errors propagate. Pass the result to both resolution signals."""
    repo = row.get("pr_repo")
    number = row.get("pr_number")
    if repo is None or number is None:
        return None

    comments = gh_api(repo, f"repos/{repo}/pulls/{number}/comments", paginate=True)

    scope = set(session_fingerprints) if session_fingerprints is not None else None

    # Pass 1: top-level daydream comment IDs and the fingerprints they carry.
    top_level_daydream_ids: set[int] = set()
    comment_id_by_fingerprint: dict[str, int] = {}
    for comment in comments:
        if comment.get("in_reply_to_id") is None and _is_daydream_comment(comment):
            markers = parse_finding_markers(comment.get("body") or "")
            if scope is not None and not any(fp in scope for fp in markers):
                continue
            cid = comment["id"]
            top_level_daydream_ids.add(cid)
            for fingerprint in markers:
                comment_id_by_fingerprint.setdefault(fingerprint, cid)

    # Pass 2: which top-level comments received a reply, keeping the full
    # reply objects (author, association, body, timestamps) rather than a
    # count — the reply text is the evidence downstream classifiers need.
    replied_ids: set[int] = set()
    replies_by_comment: dict[int, list[dict[str, Any]]] = {}
    for comment in comments:
        in_reply_to = comment.get("in_reply_to_id")
        if in_reply_to in top_level_daydream_ids:
            replied_ids.add(in_reply_to)
            replies_by_comment.setdefault(in_reply_to, []).append(comment)

    return PRCommentThreads(
        top_level_daydream_ids=top_level_daydream_ids,
        replied_ids=replied_ids,
        comment_id_by_fingerprint=comment_id_by_fingerprint,
        replies_by_comment=replies_by_comment,
    )


def comment_resolution_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
    threads: PRCommentThreads | None = None,
) -> CommentResolutionSignal:
    """Count Daydream threads with any reply, preserving the index's fingerprint scope.

    Reply presence is context, not acceptance. Reuse supplied threads or fetch
    once; a row without a PR returns (0, 0, 0). Harvest supplies a scoped index
    so other runs cannot inflate its counts."""
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
    gh_api: Callable[..., Any] | None,
    threads: PRCommentThreads | None = None,
    pr_author_logins: frozenset[str] = frozenset(),
    review_author_logins: frozenset[str] = frozenset(),
) -> list[PerFindingResolution]:
    """Resolve recorded fingerprints in order against live comment markers.

    Qualifying decisive votes agree to accepted/rejected; conflicting or
    nondirectional votes yield ambiguous; no qualifying reply yields unanswered.
    Deleted, edited-away, or never-posted comments yield missing, never rejected.
    Persist every reply and its qualification reason, including PR/review-author
    gates. Reuse supplied threads or require gh_api; a row without a PR returns []."""
    if threads is None:
        if gh_api is None:
            raise ValueError("per_finding_resolution_signal needs gh_api when threads is not supplied")
        threads = index_pr_review_comments(row, gh_api=gh_api)
    if threads is None:
        return []

    resolutions: list[PerFindingResolution] = []
    for fingerprint in recorded_fingerprints:
        comment_id = threads.comment_id_by_fingerprint.get(fingerprint)
        replies = threads.replies_by_comment.get(comment_id, []) if comment_id is not None else []
        evidence = _reply_evidence(replies, pr_author_logins, review_author_logins)
        resolutions.append(
            PerFindingResolution(
                fingerprint=fingerprint,
                comment_id=comment_id,
                disposition="missing" if comment_id is None else _disposition_from_evidence(evidence),
                evidence=evidence,
                evidence_digest=reply_evidence_digest(evidence),
            )
        )
    return resolutions


def pr_link_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> tuple[int, str] | None:
    """Find the first PR whose head exactly matches the archived head_sha.

    SHA matching avoids reused branch names. Return (number, head repo), falling
    back to the archived repo slug; missing identity or no match returns None.
    Fetcher errors propagate."""
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


def _default_branch_applied(
    row: dict[str, Any],
    *,
    repo_clone: Path,
    hunks: list[_Hunk],
    file_at_fetcher: Callable[[Path, str, str], str],
) -> LocalCommitAppliedSignal:
    """Check the base tip when a deleted/squashed branch has no readable window.

    Try origin/<base> before the possibly stale local ref. Any present hunk is
    applied; absence is unknown because squash edits may defeat verbatim matching.
    An unresolvable ref yields empty content and falls through."""
    base_branch = row.get("base_branch")
    if not base_branch or not hunks:
        return LocalCommitAppliedSignal(verdict="unknown")

    for ref in (f"origin/{base_branch}", base_branch):
        if _count_present_hunks(repo_clone, hunks, ref, file_at_fetcher):
            return LocalCommitAppliedSignal(verdict="applied")

    return LocalCommitAppliedSignal(verdict="unknown")


def local_commit_applied_signal(
    row: dict[str, Any],
    *,
    repo_clone: Path,
    commits_since_fetcher: Callable[[Path, str, str], list[str] | None],
    file_at_fetcher: Callable[[Path, str, str], str],
) -> LocalCommitAppliedSignal:
    """Check local commits after head_sha for any complete recommended hunk.

    A missing clone is unknown; a readable window with no matching commit is
    rejected. An unreadable branch window falls back to the base tip, which can
    establish applied or unknown but never rejection. Patch reads follow the
    recommendation-support rule; missing keys and read/fetch errors propagate."""
    if not repo_clone.is_dir():
        return LocalCommitAppliedSignal(verdict="unknown")

    diff_patch = row["recommended_patch"]
    hunks = _parse_diff_hunks(diff_patch)

    commits = commits_since_fetcher(repo_clone, row["branch"], row["head_sha"])
    if commits is None:
        # Branch ref unreadable — the usual cause is a squash merge, which
        # deletes the branch. The question the posterior actually asks ("did
        # the recommended change land?") survives that, so ask it of the
        # default branch instead of giving up.
        return _default_branch_applied(row, repo_clone=repo_clone, hunks=hunks, file_at_fetcher=file_at_fetcher)
    if not commits:
        return LocalCommitAppliedSignal(verdict="rejected")

    for commit in commits:
        if _count_present_hunks(repo_clone, hunks, commit, file_at_fetcher):
            return LocalCommitAppliedSignal(verdict="applied")

    return LocalCommitAppliedSignal(verdict="rejected")


def reviewer_logins_signal(
    row: dict[str, Any],
    *,
    gh_api: Callable[..., Any],
) -> list[str]:
    """Return sorted human reviewers and authors replying to Daydream threads.

    Union formal-review authors with reply authors; exclude bot logins and
    anyone who authored a Daydream-footer comment. merged_by does not qualify.
    Missing PR identity returns []; GitHub failures propagate."""
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
    humans = {login for login in logins if not login.endswith("[bot]") and login not in excluded}
    return sorted(humans)

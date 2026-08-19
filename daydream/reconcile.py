"""Reconcile current review findings against the bot's prior PR comments.

Stateless cross-run dedup: GitHub is the store. Prior findings are recovered
from the hidden ``daydream-finding`` markers embedded in posted comment bodies
(see `daydream.pr_review.finding_marker`), then partitioned against the
current run's fingerprints into new / matched / stale.

Stale inline findings are minimized via the GraphQL ``minimizeComment``
mutation with classifier ``OUTDATED`` — the Task 0 spike showed
``resolveReviewThread`` is forbidden for the least-privilege App installation
token (``pull_requests: write, contents: read, metadata: read``) while
``minimizeComment`` succeeds with the same token.

This module performs no posting and no artifact I/O; it talks to GitHub only
through `daydream.git_ops.gh_api`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from daydream import git_ops
from daydream.bot_identity import bot_login_matches
from daydream.git_ops import GitError
from daydream.pr_review import parse_finding_markers
from daydream.ui import print_warning

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass
class PriorFinding:
    """One prior daydream finding recovered from the PR.

    Attributes:
        fingerprint: The 64-hex finding fingerprint parsed from the hidden
            comment marker.
        thread_id: GraphQL review-thread node id for inline findings; None
            for body-only findings (review bodies have no thread).
        is_resolved: True when the finding is already closed — the thread was
            resolved (e.g. by a human) or the comment was previously
            minimized by a daydream run.
        comment_node_id: GraphQL node id of the carrying comment; the
            ``minimizeComment`` mutation subject for stale resolution. None
            when unknown.
    """

    fingerprint: str
    thread_id: str | None
    is_resolved: bool
    comment_node_id: str | None = None


@dataclass
class ExternalComment:
    """One inline review comment authored by another review bot.

    Recovered from the PR's review threads so daydream can suppress its own
    findings that duplicate a competitor bot's (e.g. greptile, coderabbit).
    Unlike `PriorFinding` there is no shared fingerprint marker — matching is
    semantic, driven by ``path``/``line`` plus LLM adjudication downstream.

    Attributes:
        path: Repo-relative file the comment is anchored to.
        line: Anchored line in the head commit, or None when GitHub reports
            neither ``line`` nor ``originalLine`` (an outdated/file-level thread).
        body: The comment's markdown body.
        url: Permalink to the comment (for the audit sidecar / ``external_ref``).
        author: The comment author's login, as reported by GraphQL.
    """

    path: str
    line: int | None
    body: str
    url: str
    author: str


@dataclass
class ReconcilePlan:
    """Partition of current fingerprints against prior findings.

    Attributes:
        new: Current fingerprints never posted before, in current order.
        matched: Current fingerprints that already have a prior comment
            (left untouched, even when resolved by a human).
        stale: Prior inline findings absent from the current run and not yet
            resolved — the minimization targets.
    """

    new: list[str]
    matched: set[str]
    stale: list[PriorFinding]


_REVIEW_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          comments(first: 100) {
            nodes { id databaseId body isMinimized author { login } viewerDidAuthor }
          }
        }
      }
    }
  }
}
"""

_EXTERNAL_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          path
          line
          originalLine
          comments(first: 100) {
            nodes { body url author { login } }
          }
        }
      }
    }
  }
}
"""

_MINIMIZE_COMMENT_MUTATION = """
mutation($subjectId: ID!) {
  minimizeComment(input: {subjectId: $subjectId, classifier: OUTDATED}) {
    minimizedComment { isMinimized }
  }
}
"""


def _graphql(repo: Path, query: str, variables: dict[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    """Run a GraphQL operation via ``gh api graphql`` and return the response.

    Args:
        idempotent: Set True for read-only *queries* so the call is retried on
            host-load timeouts; leave False for *mutations*, which must not be
            re-run after a timeout.

    Raises:
        GitError: If the call fails or the response carries GraphQL errors.
    """
    response = git_ops.gh_api(
        repo,
        "graphql",
        method="POST",
        input_data={"query": query, "variables": variables},
        idempotent=idempotent,
    )
    if not isinstance(response, dict) or response.get("errors"):
        raise GitError(f"GraphQL query failed: {response!r}")
    return response


def _authored_by_bot(login: str | None, viewer_did_author: bool, bot_login: str | None) -> bool:
    """True iff GitHub proves the bot authored this node.

    ``viewerDidAuthor`` is decided server-side from the installation token
    (GraphQL only); the ``[bot]``-tolerant login match covers REST reviews
    and acts as defense-in-depth on GraphQL. Either proof suffices. When
    ``bot_login`` is None only ``viewerDidAuthor`` can save a node — REST
    nodes are never trusted in that case (safe degradation).
    """
    if viewer_did_author:
        return True
    return bot_login is not None and bot_login_matches(login, bot_login)


def fetch_prior_findings(
    target_dir: Path, repo_slug: str, pr_number: int, *, bot_login: str | None = None
) -> dict[str, PriorFinding]:
    """Inventory the bot's prior findings on a PR, keyed by fingerprint.

    Combines two sources:

    1. GraphQL ``pullRequest.reviewThreads`` (paginated via ``endCursor``)
       for inline findings — each thread comment whose body carries a
       ``daydream-finding`` marker.
    2. REST ``GET /repos/<owner>/<repo>/pulls/<n>/reviews`` for body-only
       findings embedded in review bodies (``thread_id=None``).

    A marker is trusted **only** when GitHub proves the bot authored it:
    ``viewerDidAuthor == true`` (GraphQL) or ``author.login`` / ``user.login``
    matching *bot_login* via the ``[bot]``-suffix-tolerant comparator. When
    ``bot_login`` is None, GraphQL is still protected by ``viewerDidAuthor``
    and REST harvests nothing (a misconfigured bot double-posts rather than
    ever suppressing a real finding).

    The first occurrence of a fingerprint wins on duplicates. A finding
    reads as resolved when its thread is resolved (human action) or its
    comment was minimized (a prior daydream run marked it stale).

    Returns:
        Mapping of fingerprint to `PriorFinding`, in discovery order.

    Raises:
        GitError: If a GitHub API call fails.
    """
    owner, name = repo_slug.split("/", 1)
    prior: dict[str, PriorFinding] = {}

    cursor: str | None = None
    while True:
        variables = {"owner": owner, "name": name, "number": pr_number, "cursor": cursor}
        response = _graphql(target_dir, _REVIEW_THREADS_QUERY, variables, idempotent=True)
        threads = response["data"]["repository"]["pullRequest"]["reviewThreads"]
        for thread in threads["nodes"]:
            for comment in thread["comments"]["nodes"]:
                if not _authored_by_bot(
                    (comment.get("author") or {}).get("login"),
                    bool(comment.get("viewerDidAuthor")),
                    bot_login,
                ):
                    continue
                for fingerprint in parse_finding_markers(comment.get("body") or ""):
                    if fingerprint in prior:
                        continue
                    prior[fingerprint] = PriorFinding(
                        fingerprint=fingerprint,
                        thread_id=thread["id"],
                        is_resolved=bool(thread["isResolved"]) or bool(comment["isMinimized"]),
                        comment_node_id=comment["id"],
                    )
        page_info = threads["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]

    reviews = git_ops.gh_api(
        target_dir, f"repos/{owner}/{name}/pulls/{pr_number}/reviews", paginate=True, idempotent=True
    )
    for review in reviews:
        if not _authored_by_bot(
            (review.get("user") or {}).get("login"), False, bot_login
        ):
            continue
        for fingerprint in parse_finding_markers(review.get("body") or ""):
            if fingerprint in prior:
                continue
            prior[fingerprint] = PriorFinding(
                fingerprint=fingerprint,
                thread_id=None,
                is_resolved=False,
                comment_node_id=review.get("node_id"),
            )
    return prior


def fetch_external_findings(
    target_dir: Path, repo_slug: str, pr_number: int, *, bot_logins: Sequence[str]
) -> list[ExternalComment]:
    """Inventory inline comments on a PR authored by other review bots.

    Paginates ``pullRequest.reviewThreads`` and keeps every comment whose
    ``author.login`` matches one of *bot_logins* via the ``[bot]``-suffix
    tolerant comparator. Read-only; used by the external-dedup phase to
    suppress daydream findings that a faster competitor bot already posted.

    ``line`` falls back to ``originalLine`` when GitHub reports the thread
    against the original diff (outdated thread); it is None when neither is
    present, in which case the finding can only match at file level.

    Returns:
        Competitor comments in discovery order. Empty when *bot_logins* is
        empty (feature off) — no GitHub call is made in that case.

    Raises:
        GitError: If a GitHub API call fails.
    """
    if not bot_logins:
        return []
    owner, name = repo_slug.split("/", 1)
    found: list[ExternalComment] = []

    cursor: str | None = None
    while True:
        variables = {"owner": owner, "name": name, "number": pr_number, "cursor": cursor}
        response = _graphql(target_dir, _EXTERNAL_THREADS_QUERY, variables, idempotent=True)
        threads = response["data"]["repository"]["pullRequest"]["reviewThreads"]
        for thread in threads["nodes"]:
            path = thread.get("path")
            if not path:
                continue
            line = thread.get("line")
            if not isinstance(line, int):
                original = thread.get("originalLine")
                line = original if isinstance(original, int) else None
            for comment in thread["comments"]["nodes"]:
                login = (comment.get("author") or {}).get("login")
                if not any(bot_login_matches(login, bot) for bot in bot_logins):
                    continue
                found.append(
                    ExternalComment(
                        path=str(path),
                        line=line,
                        body=comment.get("body") or "",
                        url=comment.get("url") or "",
                        author=str(login or ""),
                    )
                )
        page_info = threads["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
    return found


def partition(current: Sequence[str], prior: dict[str, PriorFinding]) -> ReconcilePlan:
    """Partition the current run's fingerprints against prior findings.

    Semantics:
        - new: in ``current`` but never posted before — to be posted.
        - matched: in both, regardless of ``is_resolved`` — a finding a human
          resolved is not re-posted, and its closure is respected.
        - stale: prior inline findings (``thread_id`` set) absent from
          ``current`` and not yet resolved — to be minimized. Body-only
          findings have no thread and simply stop appearing.
    """
    current_set = set(current)
    return ReconcilePlan(
        new=[fp for fp in current if fp not in prior],
        matched={fp for fp in current if fp in prior},
        stale=[
            finding
            for fingerprint, finding in prior.items()
            if fingerprint not in current_set and finding.thread_id is not None and not finding.is_resolved
        ],
    )


def resolve_threads(target_dir: Path, stale: list[PriorFinding]) -> tuple[int, int]:
    """Mark stale findings outdated via GraphQL ``minimizeComment``.

    One mutation per stale finding, keyed on the carrying comment's GraphQL
    node id (``resolveReviewThread`` is forbidden for the least-privilege
    installation token — Task 0 spike). Best-effort: a failure on one
    finding warns and continues, matching the `daydream.pr_review` posture.

    Returns:
        ``(resolved_count, failed_count)``.
    """
    from daydream.agent import console

    resolved = 0
    failed = 0
    for finding in stale:
        if finding.comment_node_id is None:
            print_warning(
                console,
                f"Cannot minimize stale finding {finding.fingerprint[:12]}…: no comment node id",
            )
            failed += 1
            continue
        try:
            response = _graphql(
                target_dir, _MINIMIZE_COMMENT_MUTATION, {"subjectId": finding.comment_node_id}
            )
            minimized = response["data"]["minimizeComment"]["minimizedComment"]["isMinimized"]
        except (GitError, KeyError, TypeError) as exc:
            print_warning(
                console,
                f"Failed to minimize stale finding {finding.fingerprint[:12]}…: {exc}",
            )
            failed += 1
            continue
        if minimized:
            resolved += 1
        else:
            print_warning(
                console,
                f"minimizeComment did not minimize finding {finding.fingerprint[:12]}…",
            )
            failed += 1
    return resolved, failed

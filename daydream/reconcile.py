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
        comment_id: REST database id of the carrying comment (review comment
            for inline findings, review id for body-only findings).
        is_resolved: True when the finding is already closed — the thread was
            resolved (e.g. by a human) or the comment was previously
            minimized by a daydream run.
        comment_node_id: GraphQL node id of the carrying comment; the
            ``minimizeComment`` mutation subject for stale resolution. None
            when unknown.
    """

    fingerprint: str
    thread_id: str | None
    comment_id: int
    is_resolved: bool
    comment_node_id: str | None = None


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
            nodes { id databaseId body isMinimized }
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


def _graphql(repo: Path, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Run a GraphQL query via ``gh api graphql`` and return the response.

    Raises:
        GitError: If the call fails or the response carries GraphQL errors.
    """
    response = git_ops.gh_api(
        repo, "graphql", method="POST", input_data={"query": query, "variables": variables}
    )
    if not isinstance(response, dict) or response.get("errors"):
        raise GitError(f"GraphQL query failed: {response!r}")
    return response


def fetch_prior_findings(target_dir: Path, repo_slug: str, pr_number: int) -> dict[str, PriorFinding]:
    """Inventory the bot's prior findings on a PR, keyed by fingerprint.

    Combines two sources:

    1. GraphQL ``pullRequest.reviewThreads`` (paginated via ``endCursor``)
       for inline findings — each thread comment whose body carries a
       ``daydream-finding`` marker.
    2. REST ``GET /repos/<owner>/<repo>/pulls/<n>/reviews`` for body-only
       findings embedded in review bodies (``thread_id=None``).

    The first occurrence of a fingerprint wins on duplicates. A finding
    reads as resolved when its thread is resolved (human action) or its
    comment was minimized (a prior daydream run marked it stale).

    Args:
        target_dir: Repository working directory (for ``gh`` invocation).
        repo_slug: ``owner/repo`` slug of the PR's base repository.
        pr_number: PR number to inventory.

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
        response = _graphql(target_dir, _REVIEW_THREADS_QUERY, variables)
        threads = response["data"]["repository"]["pullRequest"]["reviewThreads"]
        for thread in threads["nodes"]:
            for comment in thread["comments"]["nodes"]:
                for fingerprint in parse_finding_markers(comment.get("body") or ""):
                    if fingerprint in prior:
                        continue
                    prior[fingerprint] = PriorFinding(
                        fingerprint=fingerprint,
                        thread_id=thread["id"],
                        comment_id=comment["databaseId"],
                        is_resolved=bool(thread["isResolved"]) or bool(comment["isMinimized"]),
                        comment_node_id=comment["id"],
                    )
        page_info = threads["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]

    reviews = git_ops.gh_api(
        target_dir, f"repos/{owner}/{name}/pulls/{pr_number}/reviews", paginate=True
    )
    for review in reviews:
        for fingerprint in parse_finding_markers(review.get("body") or ""):
            if fingerprint in prior:
                continue
            prior[fingerprint] = PriorFinding(
                fingerprint=fingerprint,
                thread_id=None,
                comment_id=review["id"],
                is_resolved=False,
                comment_node_id=review.get("node_id"),
            )
    return prior


def partition(current: Sequence[str], prior: dict[str, PriorFinding]) -> ReconcilePlan:
    """Partition the current run's fingerprints against prior findings.

    Semantics:
        - new: in ``current`` but never posted before — to be posted.
        - matched: in both, regardless of ``is_resolved`` — a finding a human
          resolved is not re-posted, and its closure is respected.
        - stale: prior inline findings (``thread_id`` set) absent from
          ``current`` and not yet resolved — to be minimized. Body-only
          findings have no thread and simply stop appearing.

    Args:
        current: Fingerprints produced by the current run, in post order.
        prior: Prior-finding inventory from `fetch_prior_findings`.

    Returns:
        The `ReconcilePlan` for this run.
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

    Args:
        target_dir: Repository working directory (for ``gh`` invocation).
        stale: Stale findings from `partition`.

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

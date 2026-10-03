"""Reconcile marked findings against bot-authored GitHub history. GitHub is the cross-run
store; current fingerprints partition into new, matched, and stale. Minimize stale
inline comments as OUTDATED using least-privilege App access, which cannot resolve
review threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from daydream import git_ops
from daydream.bot_identity import bot_login_matches
from daydream.git_ops import INHERIT_GITHUB_AUTH, GitError, GitHubAuth
from daydream.reviews.identity import parse_diagram_markers, parse_finding_markers
from daydream.ui import print_warning

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass
class PriorFinding:
    """Recovered fingerprint plus carrying comment/thread identity. A resolved thread or
    minimized comment is already closed; body-only findings have no thread.
    """

    fingerprint: str
    thread_id: str | None
    is_resolved: bool
    comment_node_id: str | None = None


@dataclass(frozen=True)
class PriorDiagramComment:
    """Bot-authored diagram comment with ordered unique marker kinds. REST omits minimized
    state; repeated minimizeComment calls are server-idempotent.
    """

    node_id: str
    kinds: tuple[str, ...]


@dataclass
class ReconcilePlan:
    """New fingerprints retain current order; matched findings stay closed even after human
    resolution. Only unresolved stale inline findings become minimization targets.
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

_MINIMIZE_COMMENT_MUTATION = """
mutation($subjectId: ID!) {
  minimizeComment(input: {subjectId: $subjectId, classifier: OUTDATED}) {
    minimizedComment { isMinimized }
  }
}
"""


def _graphql(
    repo: Path,
    query: str,
    variables: dict[str, Any],
    *,
    idempotent: bool = False,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> dict[str, Any]:
    """Call GraphQL through the shared GitHub boundary and reject response errors. Only
    idempotent reads opt into timeout retries; mutations must not retry.
    """
    response = git_ops.gh_api(
        repo,
        "graphql",
        method="POST",
        input_data={"query": query, "variables": variables},
        idempotent=idempotent,
        auth=auth,
    )
    if not isinstance(response, dict) or response.get("errors"):
        raise GitError(f"GraphQL query failed: {response!r}")
    return response


def _authored_by_bot(login: str | None, viewer_did_author: bool, bot_login: str | None) -> bool:
    """Accept server-proven viewer authorship or a normalized bot-login match. Without bot
    identity, REST cannot prove authorship and no marker is trusted.
    """
    return viewer_did_author or (bot_login is not None and bot_login_matches(login, bot_login))


def fetch_prior_findings(
    target_dir: Path,
    repo_slug: str,
    pr_number: int,
    *,
    bot_login: str | None = None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> dict[str, PriorFinding]:
    """Inventory authenticated markers from paginated GraphQL threads and REST review
    bodies. Viewer authorship or matching bot login is required; unresolved identity
    admits only viewer-proven GraphQL nodes. First fingerprint occurrence wins. Thread
    resolution or comment minimization closes inline findings; API failures propagate.
    """
    owner, name = repo_slug.split("/", 1)
    prior: dict[str, PriorFinding] = {}

    cursor: str | None = None
    while True:
        variables = {"owner": owner, "name": name, "number": pr_number, "cursor": cursor}
        response = _graphql(
            target_dir, _REVIEW_THREADS_QUERY, variables, idempotent=True, auth=auth
        )
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
        target_dir,
        f"repos/{owner}/{name}/pulls/{pr_number}/reviews",
        paginate=True,
        idempotent=True,
        auth=auth,
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


def partition(current: Sequence[str], prior: dict[str, PriorFinding]) -> ReconcilePlan:
    """Keep prior matches closed, including human-resolved findings. Only absent,
    unresolved inline findings become stale; body-only findings simply stop appearing.
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


def minimize_comment(
    target_dir: Path, node_id: str, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> bool:
    """Mark one carrying comment OUTDATED; transport, shape, or unsuccessful-result errors
    return False. The least-privilege token cannot use resolveReviewThread.
    """
    try:
        response = _graphql(
            target_dir, _MINIMIZE_COMMENT_MUTATION, {"subjectId": node_id}, auth=auth
        )
        return bool(response["data"]["minimizeComment"]["minimizedComment"]["isMinimized"])
    except (GitError, KeyError, TypeError):
        return False


def fetch_prior_diagram_comments(
    target_dir: Path,
    repo_slug: str,
    pr_number: int,
    *,
    bot_login: str | None,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> list[PriorDiagramComment]:
    """Read paginated issue comments and retain marked comments with proven bot authorship.
    Unresolved bot identity returns nothing without querying. Preserve API order and
    unique marker-kind order; API failures propagate.
    """
    if bot_login is None:
        return []
    owner, name = repo_slug.split("/", 1)
    comments = git_ops.gh_api(
        target_dir,
        f"repos/{owner}/{name}/issues/{pr_number}/comments",
        paginate=True,
        idempotent=True,
        auth=auth,
    )
    if not isinstance(comments, list):
        return []
    prior: list[PriorDiagramComment] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        if not bot_login_matches((comment.get("user") or {}).get("login"), bot_login):
            continue
        kinds = tuple(dict.fromkeys(kind for kind, _sha in parse_diagram_markers(comment.get("body") or "")))
        node_id = comment.get("node_id")
        if kinds and isinstance(node_id, str) and node_id:
            prior.append(PriorDiagramComment(node_id=node_id, kinds=kinds))
    return prior


def resolve_threads(
    target_dir: Path,
    stale: list[PriorFinding],
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> tuple[int, int]:
    """Minimize stale comments best-effort and return success/failure counts. Missing ids
    or failed mutations warn and do not stop later findings.
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
        if minimize_comment(target_dir, finding.comment_node_id, auth=auth):
            resolved += 1
        else:
            print_warning(
                console,
                f"minimizeComment failed or did not minimize finding {finding.fingerprint[:12]}…",
            )
            failed += 1
    return resolved, failed

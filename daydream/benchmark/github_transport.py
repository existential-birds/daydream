"""Bounded GitHub REST and GraphQL retries with complete nested pagination."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from daydream import git_ops

_RATE_LIMIT_ATTEMPTS = 3
_RATE_LIMIT_MAX_SLEEP_S = 60.0


class _ImportRateLimitError(Exception):
    """A fetch exhausted its rate-limit retries; the PR becomes ``fetch_failed``."""


def _request(
    root: Path, endpoint: str, *, paginate: bool = False,
    input_data: dict[str, Any] | None = None,
) -> Any:
    """Use the GitHub boundary for parsing, auth, and failures for every import read.

    REST and GraphQL share the same bounded rate-limit policy. Only GraphQL
    needs POST; its reads explicitly retain the existing timeout retry policy.
    """
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            return git_ops.gh_api(
                root, endpoint, auth=git_ops.INHERIT_GITHUB_AUTH,
                method="GET" if input_data is None else "POST",
                paginate=paginate, jq=".[]" if paginate else None,
                input_data=input_data, idempotent=input_data is not None,
            )
        except git_ops.RateLimitError as exc:
            if attempt == _RATE_LIMIT_ATTEMPTS - 1:
                raise _ImportRateLimitError(f"gh api {endpoint} rate limited: {exc}") from exc
            wait = exc.retry_after if exc.retry_after is not None else _RATE_LIMIT_MAX_SLEEP_S
            time.sleep(min(wait, _RATE_LIMIT_MAX_SLEEP_S))
    raise AssertionError("rate-limit attempts must be positive")


def _fetch_with_retry(root: Path, owner_repo: str, number: int) -> dict[str, Any]:
    """Fetch a singular PR header, rejecting any non-object response."""
    header = _request(root, f"repos/{owner_repo}/pulls/{number}")
    if not isinstance(header, dict):
        raise git_ops.GitError(f"gh gives no PR header for {owner_repo}#{number}")
    return header


def _rest(root: Path, endpoint: str) -> list[Any]:
    """Fetch every REST page, retaining all rows or propagating the failure."""
    return cast(list[Any], _request(root, endpoint, paginate=True))


_REVIEW_THREADS_QUERY = """
query ReviewThreads($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated subjectType path line originalLine originalStartLine
          side: diffSide startSide: startDiffSide
          comments(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes { id databaseId body author { login type: __typename } createdAt updatedAt url replyTo { id } }
          }
        }
      }
    }
  }
}
"""


_THREAD_COMMENTS_QUERY = """
query ThreadComments($threadId: ID!, $commentsAfter: String) {
  node(id: $threadId) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $commentsAfter) {
        pageInfo { hasNextPage endCursor }
        nodes { id databaseId body author { login type: __typename } createdAt updatedAt url replyTo { id } }
      }
    }
  }
}
"""


def _graphql_with_rate_limit_retry(
    root: Path, variables: dict[str, Any], *, query: str = _REVIEW_THREADS_QUERY
) -> dict[str, Any]:
    """Apply bounded Retry-After retries to GraphQL; exhausted limits retain the rate_limit ledger classification."""
    return cast(dict[str, Any], _request(root, "graphql", input_data={"query": query, "variables": variables}))


def _next_cursor(page_info: dict[str, Any], *, context: str) -> str | None:
    """Return the next cursor; hasNextPage without endCursor raises instead of dropping a page."""
    if not page_info.get("hasNextPage"):
        return None
    after = page_info.get("endCursor")
    if after is None:
        raise git_ops.GitError(f"graphql {context} hasNextPage without an endCursor")
    return str(after)


def _graphql_thread_comments(
    root: Path, thread_id: str, *, after: str | None = None
) -> dict[str, Any]:
    """Fetch a thread comment page; API errors or missing node/comments/nodes/pageInfo raise GitError."""
    variables: dict[str, Any] = {"threadId": thread_id}
    if after is not None:
        variables["commentsAfter"] = after
    resp = _graphql_with_rate_limit_retry(root, variables, query=_THREAD_COMMENTS_QUERY)
    if not isinstance(resp, dict):
        raise git_ops.GitError(
            f"graphql thread comments for {thread_id} returned a non-object response"
        )
    if resp.get("errors"):
        raise git_ops.GitError(f"graphql thread comments for {thread_id} failed: {resp['errors']}")
    try:
        node = resp["data"]["node"]
    except (KeyError, TypeError) as exc:
        raise git_ops.GitError(
            f"graphql response missing node for thread {thread_id}: {exc}"
        ) from exc
    if node is None:
        raise git_ops.GitError(f"graphql node(id:{thread_id}) returned null")
    try:
        comments = node["comments"]
    except (KeyError, TypeError) as exc:
        raise git_ops.GitError(
            f"graphql thread {thread_id} missing comments connection: {exc}"
        ) from exc
    if not (
        isinstance(comments, dict)
        and isinstance(comments.get("nodes"), list)
        and isinstance(comments.get("pageInfo"), dict)
    ):
        raise git_ops.GitError(f"graphql thread {thread_id} comments missing nodes/pageInfo")
    return comments


def _graphql_review_threads(root: Path, owner_repo: str, number: int) -> list[dict[str, Any]]:
    """Collect every thread and every nested reply page in source order."""
    owner, name = owner_repo.split("/", 1)
    all_nodes: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        variables: dict[str, Any] = {"owner": owner, "name": name, "number": number}
        if after is not None:
            variables["after"] = after
        resp = _graphql_with_rate_limit_retry(root, variables)
        if not isinstance(resp, dict):
            raise git_ops.GitError("graphql reviewThreads returned a non-object response")
        if resp.get("errors"):
            raise git_ops.GitError(f"graphql reviewThreads failed: {resp['errors']}")
        try:
            threads = resp["data"]["repository"]["pullRequest"]["reviewThreads"]
        except (KeyError, TypeError) as exc:
            raise git_ops.GitError(f"graphql response missing reviewThreads: {exc}") from exc
        all_nodes.extend(threads.get("nodes") or [])
        page_info = threads.get("pageInfo") or {}
        after = _next_cursor(page_info, context="reviewThreads")
        if after is None:
            break
    for thread in all_nodes:
        comments = thread.get("comments")
        if not isinstance(comments, dict):
            raise git_ops.GitError(
                f"graphql review thread {thread.get('id')} has no comments connection"
            )
        page_info = comments.get("pageInfo")
        if not isinstance(page_info, dict) or "hasNextPage" not in page_info:
            raise git_ops.GitError(
                f"graphql review thread {thread.get('id')} comments missing pageInfo.hasNextPage"
            )
        nested_after = _next_cursor(
            page_info, context=f"review thread {thread.get('id')} comments"
        )
        if nested_after is None:
            continue
        while True:
            page = _graphql_thread_comments(root, thread["id"], after=nested_after)
            comments.setdefault("nodes", []).extend(page["nodes"])
            page_info = page["pageInfo"]
            nested_after = _next_cursor(
                page_info, context=f"thread comments for {thread['id']}"
            )
            if nested_after is None:
                break
    return all_nodes

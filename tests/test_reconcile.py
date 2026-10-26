"""Tests for prior-finding inventory, partition, and stale resolution in `daydream/reconcile.py`."""
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from daydream import git_ops
from daydream.git_ops import GitError
from daydream.pr_review import finding_marker
from daydream.reconcile import (
    PriorFinding,
    fetch_prior_diagram_comments,
    fetch_prior_findings,
    minimize_comment,
    partition,
)


def _thread(thread_id: str, *, comment_node_id: str, database_id: int, body: str, author: str | None = None,
    viewer_did_author: bool | None = None,
) -> dict[str, Any]:
    """One reviewThreads node carrying a single comment."""
    comment: dict[str, Any] = {"id": comment_node_id, "databaseId": database_id, "body": body, "isMinimized": False}
    if author is not None:
        comment["author"] = {"login": author}
    if viewer_did_author is not None:
        comment["viewerDidAuthor"] = bool(viewer_did_author)
    return {"id": thread_id, "isResolved": False, "comments": {"nodes": [comment]}}


def _page(nodes: list[dict[str, Any]], *, next_cursor: str | None = None) -> dict[str, Any]:
    """A reviewThreads GraphQL page; ``next_cursor`` set means another page follows."""
    return {"data": {"repository": {"pullRequest": {"reviewThreads": {
        "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor}, "nodes": nodes,
    }}}}}


def test_known_bot_inventory_partitions_paginated_authorized_and_forged_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Trust and partition the complete fetched population, including human-controlled marker text."""
    a, b, c, d, e, forged_thread, forged_review = (letter * 64 for letter in "abcdef0")
    resolved = _thread("RT_RESOLVED", comment_node_id="C_RESOLVED", database_id=3,
                       body=finding_marker(c), viewer_did_author=True)
    resolved["isResolved"] = True
    minimized = _thread("RT_MINIMIZED", comment_node_id="C_MINIMIZED", database_id=4,
                        body=finding_marker(d), author="daydream")
    minimized["comments"]["nodes"][0]["isMinimized"] = True
    api = Mock(side_effect=[
        _page([
            _thread("RT_VIEWER", comment_node_id="C_VIEWER", database_id=1,
                    body=finding_marker(a), author="viewer-alias", viewer_did_author=True),
            _thread("RT_FORGED", comment_node_id="C_FORGED", database_id=6,
                    body=finding_marker(forged_thread), author="evil-attacker"),
            resolved,
            _thread("RT_PLAIN", comment_node_id="C_PLAIN", database_id=8,
                    body="a bot comment without a finding marker", viewer_did_author=True),
        ], next_cursor="CURSOR_1"),
        _page([
            _thread("RT_BOT", comment_node_id="C_BOT", database_id=2,
                    body=finding_marker(b), author="DAYDREAM[bot]"),
            minimized,
            _thread("RT_DUPLICATE", comment_node_id="C_DUPLICATE", database_id=9,
                    body=finding_marker(a), viewer_did_author=True),
        ]),
        [
            {"id": 901, "node_id": "BODY", "body": finding_marker(e), "user": {"login": "daydream[bot]"}},
            {"id": 902, "node_id": "FORGED", "body": finding_marker(forged_review),
             "user": {"login": "evil-attacker"}},
            {"id": 903, "node_id": "BODY_DUPLICATE", "body": finding_marker(a), "user": {"login": "daydream"}},
        ],
    ])
    monkeypatch.setattr(git_ops, "gh_api", api)

    prior = fetch_prior_findings(tmp_path, "o/r", 7, bot_login="daydream")

    assert prior == {
        a: PriorFinding(a, "RT_VIEWER", False, "C_VIEWER"),
        c: PriorFinding(c, "RT_RESOLVED", True, "C_RESOLVED"),
        b: PriorFinding(b, "RT_BOT", False, "C_BOT"),
        d: PriorFinding(d, "RT_MINIMIZED", True, "C_MINIMIZED"),
        e: PriorFinding(e, None, False, "BODY"),
    }
    assert list(prior) == [a, c, b, d, e]
    plan = partition(current=[a, c, "9" * 64], prior=prior)
    assert plan.new == ["9" * 64] and plan.matched == {a, c}
    assert plan.stale == [PriorFinding(b, "RT_BOT", False, "C_BOT")]
    first, second, reviews = api.call_args_list
    assert first.args == second.args == (tmp_path, "graphql")
    assert [call.kwargs["input_data"]["variables"] for call in (first, second)] == [
        {"owner": "o", "name": "r", "number": 7, "cursor": None},
        {"owner": "o", "name": "r", "number": 7, "cursor": "CURSOR_1"},
    ]
    assert all(call.kwargs["method"] == "POST" and call.kwargs["idempotent"] for call in (first, second))
    assert reviews.args == (tmp_path, "repos/o/r/pulls/7/reviews")
    assert reviews.kwargs["paginate"] is True and reviews.kwargs["idempotent"] is True


def test_unresolved_bot_inventory_accepts_only_viewer_proof(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    viewer, forged = "a" * 64, "b" * 64
    api = Mock(side_effect=[_page([
        _thread("RT_VIEWER", comment_node_id="C_VIEWER", database_id=1,
                body=finding_marker(viewer), viewer_did_author=True),
        _thread("RT_FORGED", comment_node_id="C_FORGED", database_id=2,
                body=finding_marker(forged), author="daydream[bot]"),
    ]), [{"node_id": "REST_FORGED", "body": finding_marker(forged), "user": {"login": "daydream[bot]"}}]])
    monkeypatch.setattr(git_ops, "gh_api", api)
    assert fetch_prior_findings(tmp_path, "o/r", 7, bot_login=None) == {
        viewer: PriorFinding(viewer, "RT_VIEWER", False, "C_VIEWER"),
    }
    assert api.call_count == 2


# --- Prior diagram comments (issue #1113) ------------------------------------


def test_fetch_prior_diagram_comments_harvests_nothing_without_a_bot_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REST has no ``viewerDidAuthor``, so an unresolved login must not query at all."""
    def _forbidden(*_args: Any, **_kw: Any) -> Any:
        raise AssertionError("must not call GitHub with an unresolved bot login")
    monkeypatch.setattr(git_ops, "gh_api", _forbidden)
    assert fetch_prior_diagram_comments(tmp_path, "o/r", 7, bot_login=None) == []

def test_minimize_comment_reports_failure_instead_of_raising(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every transport/shape failure is False; only a proven fold is True."""
    monkeypatch.setattr(
        git_ops, "gh_api", lambda *_a, **_k: {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}},
    )
    assert minimize_comment(tmp_path, "IC_1") is True

    monkeypatch.setattr(git_ops, "gh_api",
        lambda *_a, **_k: {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": False}}}},
    )
    assert minimize_comment(tmp_path, "IC_1") is False

    monkeypatch.setattr(git_ops, "gh_api", lambda *_a, **_k: {"data": {}})
    assert minimize_comment(tmp_path, "IC_1") is False

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise GitError("gh exploded")
    monkeypatch.setattr(git_ops, "gh_api", _boom)
    assert minimize_comment(tmp_path, "IC_1") is False

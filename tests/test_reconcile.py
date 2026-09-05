"""Tests for prior-finding inventory, partition, and stale resolution in `daydream/reconcile.py`."""
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.pr_review import finding_marker
from daydream.reconcile import PriorFinding, fetch_prior_findings, partition

# --- Canned gh_api responses for fetch_prior_findings ----------------------


def _thread(
    thread_id: str,
    *,
    comment_node_id: str,
    database_id: int,
    body: str,
    author: str | None = None,
    viewer_did_author: bool | None = None,
) -> dict[str, Any]:
    """One reviewThreads node carrying a single comment."""
    comment: dict[str, Any] = {
        "id": comment_node_id,
        "databaseId": database_id,
        "body": body,
        "isMinimized": False,
    }
    if author is not None:
        comment["author"] = {"login": author}
    if viewer_did_author is not None:
        comment["viewerDidAuthor"] = bool(viewer_did_author)
    return {
        "id": thread_id,
        "isResolved": False,
        "comments": {"nodes": [comment]},
    }


def _page(nodes: list[dict[str, Any]], *, next_cursor: str | None = None) -> dict[str, Any]:
    """A reviewThreads GraphQL page; ``next_cursor`` set means another page follows."""
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor},
                        "nodes": nodes,
                    }
                }
            }
        }
    }


_GRAPHQL_PAGE_1: dict[str, Any] = _page(
    [_thread("RT_0", comment_node_id="PRRC_0", database_id=100, body="a human comment with no marker")],
    next_cursor="CURSOR_1",
)

_GRAPHQL_PAGE_2: dict[str, Any] = _page(
    [_thread("RT_1", comment_node_id="PRRC_1", database_id=101,
             body="Race in cache\n\n" + finding_marker("a" * 64),
             viewer_did_author=True)],
)

_REST_REVIEWS: list[dict[str, Any]] = [
    {"id": 900, "node_id": "PRR_900", "body": "review summary, no marker"},
    {"id": 901, "node_id": "PRR_901", "body": "File-level note\n\n" + finding_marker("b" * 64),
     "user": {"login": "daydream-bot"}},
]


def _fake_gh_api_two_pages(repo: Any, endpoint: str, **kwargs: Any) -> Any:
    """Canned gh_api: a two-page GraphQL thread inventory + one REST reviews page."""
    if endpoint == "graphql":
        cursor = kwargs["input_data"]["variables"].get("cursor")
        return _GRAPHQL_PAGE_2 if cursor == "CURSOR_1" else _GRAPHQL_PAGE_1
    if endpoint.endswith("/pulls/7/reviews"):
        return _REST_REVIEWS
    raise AssertionError(f"unexpected gh_api endpoint: {endpoint}")


# --- Tests ------------------------------------------------------------------


def test_partition_new_matched_stale_and_respects_human_resolution() -> None:
    prior = {
        "f1": PriorFinding("f1", thread_id="T1", is_resolved=False),
        "f2": PriorFinding("f2", thread_id="T2", is_resolved=False),
        "f3": PriorFinding("f3", thread_id="T3", is_resolved=True),
        "f4": PriorFinding("f4", thread_id=None, is_resolved=False),  # body-only
    }
    plan = partition(current=["f1", "f3", "f9"], prior=prior)
    assert plan.new == ["f9"]                       # never posted -> post
    assert [p.fingerprint for p in plan.stale] == ["f2"]  # unresolved inline, gone -> resolve
    assert plan.matched == {"f1", "f3"}             # f3 resolved by a human: stays closed
    # body-only f4 is stale but has no thread; it must NOT appear in plan.stale


def test_fetch_prior_findings_parses_markers_across_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(git_ops, "gh_api", _fake_gh_api_two_pages)  # canned GraphQL + REST pages
    prior = fetch_prior_findings(tmp_path, "o/r", 7, bot_login="daydream-bot")
    assert prior["a" * 64].thread_id == "RT_1" and prior["b" * 64].thread_id is None


def test_fetch_prior_findings_ignores_marker_from_non_bot_author(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A thread whose comment carries the marker but is authored by a human.
    fp = "a" * 64
    pages = [_page([
        _thread("RT_1", comment_node_id="PRRC_1", database_id=101,
                body=finding_marker(fp), author="evil-attacker"),  # no viewerDidAuthor
    ])]
    def _gh(repo: Any, endpoint: str, **kw: Any) -> Any:
        if endpoint == "graphql":
            return pages.pop(0) if pages else _page([])
        if endpoint.endswith("/pulls/7/reviews"):
            return []
        raise AssertionError(endpoint)
    monkeypatch.setattr(git_ops, "gh_api", _gh)
    prior = fetch_prior_findings(tmp_path, "o/r", 7, bot_login="daydream")
    assert prior == {}   # forged marker ignored -> not trusted


def test_fetch_prior_findings_ignores_review_marker_from_non_bot_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fp = "b" * 64
    def _gh(repo: Any, endpoint: str, **kw: Any) -> Any:
        if endpoint == "graphql":
            return _page([])
        if endpoint.endswith("/pulls/7/reviews"):
            return [{"id": 9, "node_id": "PRR_9", "body": finding_marker(fp),
                     "user": {"login": "evil-attacker"}}]
        raise AssertionError(endpoint)
    monkeypatch.setattr(git_ops, "gh_api", _gh)
    prior = fetch_prior_findings(tmp_path, "o/r", 7, bot_login="daydream")
    assert prior == {}


def test_fetch_prior_findings_trusts_bot_login_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fp = "c" * 64
    def _gh(repo: Any, endpoint: str, **kw: Any) -> Any:
        if endpoint == "graphql":
            return _page([_thread("RT_2", comment_node_id="PRRC_2", database_id=102,
                                  body=finding_marker(fp), author="daydream[bot]")])
        if endpoint.endswith("/pulls/7/reviews"):
            return []
        raise AssertionError(endpoint)
    monkeypatch.setattr(git_ops, "gh_api", _gh)
    prior = fetch_prior_findings(tmp_path, "o/r", 7, bot_login="daydream")
    assert fp in prior and prior[fp].thread_id == "RT_2"


def test_fetch_prior_findings_trusts_viewerDidAuthor_without_bot_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fp = "d" * 64
    def _gh(repo: Any, endpoint: str, **kw: Any) -> Any:
        if endpoint == "graphql":
            return _page([_thread("RT_3", comment_node_id="PRRC_3", database_id=103,
                                  body=finding_marker(fp), viewer_did_author=True)])
        if endpoint.endswith("/pulls/7/reviews"):
            return []
        raise AssertionError(endpoint)
    monkeypatch.setattr(git_ops, "gh_api", _gh)
    prior = fetch_prior_findings(tmp_path, "o/r", 7, bot_login=None)  # misconfigured
    assert fp in prior   # viewerDidAuthor still proves authorship


# --- Prior diagram comments (issue #1113) ------------------------------------


def _issue_comment(
    node_id: str, *, body: str, login: str | None = "daydream[bot]"
) -> dict[str, Any]:
    """One REST issue-comment object."""
    comment: dict[str, Any] = {"id": 1, "node_id": node_id, "body": body}
    if login is not None:
        comment["user"] = {"login": login}
    return comment


def test_fetch_prior_diagram_comments_trusts_only_the_bot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Marker present AND author proven; kinds come back de-duplicated, in order."""
    from daydream.pr_review import diagram_marker
    from daydream.reconcile import fetch_prior_diagram_comments

    sha = "a" * 40
    comments = [
        _issue_comment("IC_1", body=f"{diagram_marker('flowchart', sha)}\n{diagram_marker('sequence', sha)}"),
        _issue_comment("IC_2", body=diagram_marker("sequence", sha), login="evil-attacker"),
        _issue_comment("IC_3", body="a plain human comment"),
        _issue_comment("IC_4", body=f"{diagram_marker('sequence', sha)} {diagram_marker('sequence', sha)}"),
    ]

    def _gh(_repo: Any, endpoint: str, **_kw: Any) -> Any:
        assert endpoint == "repos/o/r/issues/7/comments"
        return comments

    monkeypatch.setattr(git_ops, "gh_api", _gh)
    prior = fetch_prior_diagram_comments(tmp_path, "o/r", 7, bot_login="daydream")
    assert [(c.node_id, c.kinds) for c in prior] == [
        ("IC_1", ("flowchart", "sequence")),
        ("IC_4", ("sequence",)),
    ]


def test_fetch_prior_diagram_comments_harvests_nothing_without_a_bot_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REST has no ``viewerDidAuthor``, so an unresolved login must not query at all."""
    from daydream.reconcile import fetch_prior_diagram_comments

    def _forbidden(*_args: Any, **_kw: Any) -> Any:
        raise AssertionError("must not call GitHub with an unresolved bot login")

    monkeypatch.setattr(git_ops, "gh_api", _forbidden)
    assert fetch_prior_diagram_comments(tmp_path, "o/r", 7, bot_login=None) == []


def test_minimize_comment_reports_failure_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every transport/shape failure is False; only a proven fold is True."""
    from daydream.git_ops import GitError
    from daydream.reconcile import minimize_comment

    monkeypatch.setattr(
        git_ops,
        "gh_api",
        lambda *_a, **_k: {
            "data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}
        },
    )
    assert minimize_comment(tmp_path, "IC_1") is True

    monkeypatch.setattr(
        git_ops,
        "gh_api",
        lambda *_a, **_k: {
            "data": {"minimizeComment": {"minimizedComment": {"isMinimized": False}}}
        },
    )
    assert minimize_comment(tmp_path, "IC_1") is False

    monkeypatch.setattr(git_ops, "gh_api", lambda *_a, **_k: {"data": {}})
    assert minimize_comment(tmp_path, "IC_1") is False

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise GitError("gh exploded")

    monkeypatch.setattr(git_ops, "gh_api", _boom)
    assert minimize_comment(tmp_path, "IC_1") is False

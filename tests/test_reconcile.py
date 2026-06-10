"""Tests for prior-finding inventory, partition, and stale resolution in `daydream/reconcile.py`."""

from typing import Any

from daydream import git_ops
from daydream.pr_review import finding_marker
from daydream.reconcile import PriorFinding, fetch_prior_findings, partition

# --- Canned gh_api responses for fetch_prior_findings ----------------------

_GRAPHQL_PAGE_1: dict[str, Any] = {
    "data": {
        "repository": {
            "pullRequest": {
                "reviewThreads": {
                    "pageInfo": {"hasNextPage": True, "endCursor": "CURSOR_1"},
                    "nodes": [
                        {
                            "id": "RT_0",
                            "isResolved": False,
                            "comments": {
                                "nodes": [
                                    {
                                        "id": "PRRC_0",
                                        "databaseId": 100,
                                        "body": "a human comment with no marker",
                                        "isMinimized": False,
                                    }
                                ]
                            },
                        }
                    ],
                }
            }
        }
    }
}

_GRAPHQL_PAGE_2: dict[str, Any] = {
    "data": {
        "repository": {
            "pullRequest": {
                "reviewThreads": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [
                        {
                            "id": "RT_1",
                            "isResolved": False,
                            "comments": {
                                "nodes": [
                                    {
                                        "id": "PRRC_1",
                                        "databaseId": 101,
                                        "body": "Race in cache\n\n" + finding_marker("a" * 64),
                                        "isMinimized": False,
                                    }
                                ]
                            },
                        }
                    ],
                }
            }
        }
    }
}

_REST_REVIEWS: list[dict[str, Any]] = [
    {"id": 900, "node_id": "PRR_900", "body": "review summary, no marker"},
    {"id": 901, "node_id": "PRR_901", "body": "File-level note\n\n" + finding_marker("b" * 64)},
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
        "f1": PriorFinding("f1", thread_id="T1", comment_id=1, is_resolved=False),
        "f2": PriorFinding("f2", thread_id="T2", comment_id=2, is_resolved=False),
        "f3": PriorFinding("f3", thread_id="T3", comment_id=3, is_resolved=True),
        "f4": PriorFinding("f4", thread_id=None, comment_id=4, is_resolved=False),  # body-only
    }
    plan = partition(current=["f1", "f3", "f9"], prior=prior)
    assert plan.new == ["f9"]                       # never posted -> post
    assert [p.fingerprint for p in plan.stale] == ["f2"]  # unresolved inline, gone -> resolve
    assert plan.matched == {"f1", "f3"}             # f3 resolved by a human: stays closed
    # body-only f4 is stale but has no thread; it must NOT appear in plan.stale


def test_fetch_prior_findings_parses_markers_across_pages(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(git_ops, "gh_api", _fake_gh_api_two_pages)  # canned GraphQL + REST pages
    prior = fetch_prior_findings(tmp_path, "o/r", 7)
    assert prior["a" * 64].thread_id == "RT_1" and prior["b" * 64].thread_id is None

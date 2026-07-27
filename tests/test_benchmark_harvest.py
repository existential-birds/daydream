"""Tests for ``daydream bench harvest``.

Covers the ``[bot]``-suffix login trap (GitHub's REST/GraphQL mismatch), the
pure records-to-corpus projection, and a real-path harvest through
``_handle_bench_command`` with the ``gh`` network boundary faked.
"""

import json

import pytest

from daydream.benchmark.cli import _handle_bench_command
from daydream.benchmark.harvest import bot_login_matches, build_harvested_corpus


@pytest.mark.parametrize(
    ("observed_login", "configured_login", "expected"),
    [
        pytest.param("coderabbitai[bot]", "coderabbitai[bot]", True, id="rest-both-suffixed"),
        pytest.param("coderabbitai[bot]", "coderabbitai", True, id="rest-observed-suffixed"),
        pytest.param("greptileai[bot]", "coderabbitai[bot]", False, id="different-bot"),
        pytest.param("coderabbitai", "coderabbitai[bot]", True, id="graphql-stripped"),
        pytest.param(None, "coderabbitai[bot]", False, id="missing-observed-login"),
        pytest.param("CodeRabbitAI[bot]", "coderabbitai", True, id="observed-case-insensitive"),
        pytest.param("coderabbitai", "CodeRabbitAI[bot]", True, id="configured-case-insensitive"),
        pytest.param("coderabbitai", "coderabbit", False, id="different-stem"),
    ],
)
def test_bot_login_matches_cases(observed_login, configured_login, expected):
    """Match bot logins across case and GitHub's optional app suffix."""
    assert bot_login_matches(observed_login, configured_login) is expected


def test_build_harvested_corpus_emits_golden_with_comment_key_and_resolved_flag():
    records = [
        {
            "pr_number": 5,
            "comments": [
                {"path": "a.py", "line": 12, "body": "Null deref here", "created_at": "2026-01-01T00:00:00Z"},
                {"path": "b.py", "line": 3, "body": "Unused import", "created_at": "2026-01-01T00:01:00Z"},
            ],
            "threads": [
                {"path": "a.py", "line": 12, "is_resolved": True, "author": "cr"},
                {"path": "b.py", "line": 3, "is_resolved": False, "author": "cr"},
            ],
        }
    ]
    corpus = build_harvested_corpus(records, repo="acme/widgets", bot="cr[bot]")

    entry = corpus["https://github.com/acme/widgets/pull/5"]
    golden = entry["golden_comments"]
    # "comment" is the key the judge reads; resolved preserves the acted-upon signal.
    assert [g["comment"] for g in golden] == ["Null deref here", "Unused import"]
    assert [g["resolved"] for g in golden] == [True, False]
    assert [g["path"] for g in golden] == ["a.py", "b.py"]
    assert all(g["severity"] is None for g in golden)

    # The bot's own review is injected under the stripped stem as a scorable arm.
    review = entry["reviews"][0]
    assert review["tool"] == "cr"
    assert review["pr_url"] == "https://github.com/acme/widgets/pull/5"
    assert [c["body"] for c in review["review_comments"]] == ["Null deref here", "Unused import"]


def test_harvest_keeps_only_snapshot_commit_comments(tmp_path, monkeypatch, fake_gh):
    # A review at commit A plus a later inline comment at commit B: the replay
    # runs at A, so only the A-era finding may enter the golden set.
    monkeypatch.chdir(tmp_path)
    commit_a, commit_b = "a" * 40, "b" * 40
    fake_gh.set_response(
        "GET",
        "repos/acme/widgets/pulls",
        [
            {
                "number": 5,
                "title": "Add widget cache",
                "state": "open",
                "created_at": "2026-01-01T00:00:00Z",
                "base": {"ref": "main"},
                "head": {"ref": "feature/cache"},
            }
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/acme/widgets/pulls/5/reviews",
        [
            {
                "id": 1,
                "user": {"login": "cr[bot]"},
                "body": "Found one issue.",
                "commit_id": commit_a,
                "submitted_at": "2026-01-02T00:00:00Z",
                "state": "COMMENTED",
            }
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/acme/widgets/pulls/5/comments",
        [
            {
                "id": 10,
                "user": {"login": "cr[bot]"},
                "path": "a.py",
                "line": 12,
                "body": "Null deref here",
                "created_at": "2026-01-02T00:00:00Z",
                "commit_id": commit_a,
                "original_commit_id": commit_a,
            },
            {
                "id": 11,
                "user": {"login": "cr[bot]"},
                "path": "c.py",
                "line": 7,
                "body": "Race on the new cache write",
                "created_at": "2026-01-05T00:00:00Z",
                "commit_id": commit_b,
                "original_commit_id": commit_b,
            },
        ],
    )
    fake_gh.set_response(
        "graphql_threads",
        value={
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [],
                        }
                    }
                }
            }
        },
    )

    out = tmp_path / "corpus"
    assert _handle_bench_command(["harvest", "--repo", "acme/widgets", "--bot", "cr[bot]", "--out", str(out)]) == 0

    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["prs"][0]["review_commit_id"] == commit_a
    assert index["prs"][0]["n_inline_comments"] == 1

    record = json.loads((out / "harvest" / "pr-5.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in record["comments"]] == [10]

    corpus = json.loads((out / "results" / "benchmark_data.json").read_text(encoding="utf-8"))
    entry = corpus["https://github.com/acme/widgets/pull/5"]
    assert [g["comment"] for g in entry["golden_comments"]] == ["Null deref here"]
    assert [c["body"] for c in entry["reviews"][0]["review_comments"]] == ["Null deref here"]


def test_harvest_command_writes_corpus_files(tmp_path, monkeypatch, fake_gh):
    monkeypatch.chdir(tmp_path)
    fake_gh.set_response(
        "GET",
        "repos/acme/widgets/pulls",
        [
            {
                "number": 5,
                "title": "Add widget cache",
                "state": "closed",
                "merged_at": "2026-01-03T00:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "base": {"ref": "develop", "sha": "d" * 40},
                "head": {"ref": "feature/cache"},
            }
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/acme/widgets/pulls/5/reviews",
        [
            {
                "id": 1,
                "user": {"login": "cr[bot]"},
                "body": "Found one issue.",
                "commit_id": "a" * 40,
                "submitted_at": "2026-01-02T00:00:00Z",
                "state": "COMMENTED",
            },
            {"id": 2, "user": {"login": "carol"}, "body": "lgtm", "commit_id": "f" * 40},
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/acme/widgets/pulls/5/comments",
        [
            {
                "id": 10,
                "user": {"login": "cr[bot]"},
                "path": "a.py",
                "line": 12,
                "body": "Null deref here",
                "created_at": "2026-01-02T00:00:00Z",
                "commit_id": "a" * 40,
            },
            {
                "id": 11,
                "user": {"login": "cr[bot]"},
                "path": "a.py",
                "line": 12,
                "body": "thanks for fixing",
                "in_reply_to_id": 10,
                "created_at": "2026-01-04T00:00:00Z",
            },
            {"id": 12, "user": {"login": "carol"}, "path": "b.py", "line": 3, "body": "nit"},
        ],
    )
    # GraphQL returns the *stripped* login: the suffix tolerance is what makes
    # this thread attach to the --bot "cr[bot]" run.
    fake_gh.set_response(
        "graphql_threads",
        value={
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "isResolved": True,
                                    "isOutdated": False,
                                    "path": "a.py",
                                    "line": 12,
                                    "comments": {"nodes": [{"author": {"login": "cr"}}]},
                                }
                            ],
                        }
                    }
                }
            }
        },
    )

    out = tmp_path / "corpus"
    rc = _handle_bench_command(["harvest", "--repo", "acme/widgets", "--bot", "cr[bot]", "--out", str(out)])
    assert rc == 0

    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["repo"] == "acme/widgets" and index["bot"] == "cr[bot]"
    assert index["n_prs_with_bot_activity"] == 1
    entry = index["prs"][0]
    assert entry["pr_number"] == 5
    assert entry["review_commit_id"] == "a" * 40
    assert entry["base_ref"] == "develop"
    assert entry["base_sha"] == "d" * 40  # immutable historic base, not re-derived at replay time
    assert entry["n_inline_comments"] == 1  # reply and non-bot comment excluded
    assert entry["n_review_summaries"] == 1  # carol's review excluded
    assert entry["n_resolved_threads"] == 1
    assert entry["threads_complete"] is True

    record = json.loads((out / "harvest" / "pr-5.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in record["comments"]] == [10]
    assert record["base_ref"] == "develop"
    assert record["base_sha"] == "d" * 40

    corpus = json.loads((out / "results" / "benchmark_data.json").read_text(encoding="utf-8"))
    golden = corpus["https://github.com/acme/widgets/pull/5"]["golden_comments"]
    assert [g["comment"] for g in golden] == ["Null deref here"]
    assert golden[0]["resolved"] is True
    assert corpus["https://github.com/acme/widgets/pull/5"]["reviews"][0]["tool"] == "cr"

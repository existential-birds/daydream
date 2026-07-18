"""Tests for ``daydream bench harvest``.

Covers the ``[bot]``-suffix login trap (GitHub's REST/GraphQL mismatch), the
pure records-to-corpus projection, and a real-path harvest through
``_handle_bench_command`` with the ``gh`` network boundary faked.
"""

import json

from daydream.benchmark.cli import _handle_bench_command
from daydream.benchmark.harvest import bot_login_matches, build_harvested_corpus


def test_bot_login_matches_rest_suffix_form():
    # REST user.login keeps the suffix; --bot may be given either way.
    assert bot_login_matches("coderabbitai[bot]", "coderabbitai[bot]")
    assert bot_login_matches("coderabbitai[bot]", "coderabbitai")
    assert not bot_login_matches("greptileai[bot]", "coderabbitai[bot]")


def test_bot_login_matches_graphql_stripped_form():
    # GraphQL author.login drops the suffix; it must still match --bot "x[bot]".
    assert bot_login_matches("coderabbitai", "coderabbitai[bot]")
    assert not bot_login_matches(None, "coderabbitai[bot]")


def test_bot_login_matches_is_case_insensitive():
    # The stem comparison is case-insensitive. The literal "[bot]" suffix GitHub
    # appends is always lowercase, so it is stripped before casefolding.
    assert bot_login_matches("CodeRabbitAI[bot]", "coderabbitai")
    assert bot_login_matches("coderabbitai", "CodeRabbitAI[bot]")
    assert not bot_login_matches("coderabbitai", "coderabbit")


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
                "base": {"ref": "develop"},
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
    rc = _handle_bench_command(
        ["harvest", "--repo", "acme/widgets", "--bot", "cr[bot]", "--out", str(out)]
    )
    assert rc == 0

    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["repo"] == "acme/widgets" and index["bot"] == "cr[bot]"
    assert index["n_prs_with_bot_activity"] == 1
    entry = index["prs"][0]
    assert entry["pr_number"] == 5
    assert entry["review_commit_id"] == "a" * 40
    assert entry["base_ref"] == "develop"
    assert entry["n_inline_comments"] == 1  # reply and non-bot comment excluded
    assert entry["n_review_summaries"] == 1  # carol's review excluded
    assert entry["n_resolved_threads"] == 1
    assert entry["threads_complete"] is True

    record = json.loads((out / "harvest" / "pr-5.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in record["comments"]] == [10]
    assert record["base_ref"] == "develop"

    corpus = json.loads((out / "results" / "benchmark_data.json").read_text(encoding="utf-8"))
    golden = corpus["https://github.com/acme/widgets/pull/5"]["golden_comments"]
    assert [g["comment"] for g in golden] == ["Null deref here"]
    assert golden[0]["resolved"] is True
    assert corpus["https://github.com/acme/widgets/pull/5"]["reviews"][0]["tool"] == "cr"

"""Harvest a review bot's historic PR reviews into a benchmark corpus.

``daydream bench harvest --repo OWNER/REPO --bot LOGIN --out DIR`` scans a
repository's pull requests for one review bot's activity and writes a corpus
that ``daydream bench --harvest-dir DIR`` can run against. For every PR the bot
touched it captures:

- the bot's review summaries (with the ``commit_id`` each was made against),
- the bot's standalone inline comments anchored to the snapshot commit (thread
  replies and off-snapshot comments excluded — they are not findings about the
  replayed tree and would inflate the golden set),
- review-thread resolution status (the "acted upon" signal, stored as metadata),
- the snapshot commit to replay daydream against (the latest bot review's
  ``commit_id``), plus the PR's immutable ``base.sha`` so the historic diff
  stays reproducible after the base branch has moved on.

Output layout — one harvest dir is one corpus, so nothing is nested per repo::

    DIR/index.json                    # PR inventory (drives corpus.harvested_corpus)
    DIR/harvest/pr-<N>.json           # full per-PR record
    DIR/results/benchmark_data.json   # the corpus the harness reads/injects into

All GitHub access goes through :func:`daydream.git_ops.gh_api`, so rate-limit
and timeout discipline is shared with the rest of daydream.

Exports:
    bot_login_matches: ``[bot]``-suffix-tolerant login comparison.
    fetch_review_threads: Review-thread resolution state via GraphQL.
    harvest_pr: Collect one PR's bot activity.
    build_harvested_corpus: Project harvest records into a benchmark corpus dict.
    run_harvest: Drive the whole harvest and write the corpus to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.agent import console
from daydream.benchmark.benchmark_data import save_benchmark_data
from daydream.ui import print_dim, print_info, print_warning

#: GraphQL page size for review threads (GitHub's per-connection maximum).
_THREAD_PAGE_SIZE = 100

_REVIEW_THREADS_QUERY = """
query($owner:String!,$repo:String!,$num:Int!,$cursor:String){
  repository(owner:$owner,name:$repo){
    pullRequest(number:$num){
      reviewThreads(first:%d,after:$cursor){
        nodes{
          isResolved isOutdated path line
          comments(first:1){ nodes{ author{login} } }
        }
        pageInfo{ hasNextPage endCursor }
      }
    }
  }
}
""" % _THREAD_PAGE_SIZE


def bot_login_matches(login: str | None, bot: str) -> bool:
    """Match a bot login tolerant of GitHub's REST/GraphQL ``[bot]`` mismatch.

    REST ``user.login`` keeps the ``[bot]`` suffix (``coderabbitai[bot]``);
    GraphQL ``author.login`` drops it (``coderabbitai``). Compare on the
    stripped, lowercased stem so both forms match one ``--bot`` value.
    """
    return bot_stem(login) == bot_stem(bot)


def bot_stem(login: str | None) -> str:
    """Return a login's comparison stem: ``[bot]`` suffix dropped, lowercased."""
    return (login or "").removesuffix("[bot]").lower()


def fetch_review_threads(repo_slug: str, pr_number: int) -> tuple[list[dict[str, Any]], bool]:
    """Fetch a PR's review threads (resolution + first-comment author) via GraphQL.

    Returns:
        ``(threads, ok)`` with one dict per thread
        (``is_resolved``/``is_outdated``/``path``/``line``/``author``). ``ok`` is
        ``False`` when a transport or shape error truncated the set, so the
        caller can mark that PR's resolved count unreliable rather than reading a
        partial fetch as a real value. Best-effort: never raises.
    """
    owner, _, name = repo_slug.partition("/")
    threads: list[dict[str, Any]] = []
    cursor: str | None = None
    cwd = Path.cwd()
    try:
        while True:
            variables: dict[str, Any] = {"owner": owner, "repo": name, "num": pr_number}
            if cursor:
                variables["cursor"] = cursor
            payload = git_ops.gh_api(
                cwd,
                "graphql",
                method="POST",
                input_data={"query": _REVIEW_THREADS_QUERY, "variables": variables},
                idempotent=True,
            )
            node = payload["data"]["repository"]["pullRequest"]["reviewThreads"]
            for thread in node["nodes"]:
                comments = (thread.get("comments") or {}).get("nodes") or []
                author = ""
                if comments and comments[0].get("author"):
                    author = comments[0]["author"].get("login", "")
                threads.append(
                    {
                        "is_resolved": thread.get("isResolved", False),
                        "is_outdated": thread.get("isOutdated", False),
                        "path": thread.get("path"),
                        "line": thread.get("line"),
                        "author": author,
                    }
                )
            page = node["pageInfo"]
            if not page["hasNextPage"]:
                break
            cursor = page["endCursor"]
    except (git_ops.GitError, KeyError, TypeError, json.JSONDecodeError):
        return threads, False
    return threads, True


def _paginated(endpoint: str) -> list[Any]:
    """Fetch every page of a REST list endpoint as a flat list.

    ``--paginate`` concatenates each page's raw JSON, which is not itself valid
    JSON for array endpoints; the ``.[]`` filter flattens every page to one
    value per line instead (see :func:`daydream.git_ops.gh_api`).
    """
    result = git_ops.gh_api(Path.cwd(), endpoint, paginate=True, jq=".[]", idempotent=True)
    return list(result) if isinstance(result, list) else []


def harvest_pr(repo_slug: str, pr: dict[str, Any], bot: str) -> dict[str, Any] | None:
    """Collect one PR's bot activity.

    Args:
        pr: A REST pull-request object (``repos/{repo}/pulls``).

    Returns:
        The harvest record, or ``None`` when the bot left nothing on this PR.
        ``comments`` is narrowed to the ``review_commit_id`` snapshot so the
        golden findings and the replayed commit describe the same tree.
    """
    number = pr["number"]

    reviews = _paginated(f"repos/{repo_slug}/pulls/{number}/reviews")
    # Keep reviews carrying a body; a pure approval is not a finding.
    bot_reviews = [
        review
        for review in reviews
        if bot_login_matches((review.get("user") or {}).get("login"), bot)
        and (review.get("body") or "").strip()
    ]

    comments = _paginated(f"repos/{repo_slug}/pulls/{number}/comments")
    # Drop thread replies (in_reply_to_id set): they are not standalone findings,
    # and counting them inflates the golden set and can taint review_commit_id.
    bot_comments = [
        comment
        for comment in comments
        if not comment.get("in_reply_to_id")
        and bot_login_matches((comment.get("user") or {}).get("login"), bot)
    ]

    if not bot_reviews and not bot_comments:
        return None

    # Snapshot to replay against: the latest bot review's commit, falling back to
    # the latest inline comment's commit.
    review_commit_id = None
    if bot_reviews:
        latest = max(bot_reviews, key=lambda r: r.get("submitted_at") or "")
        review_commit_id = latest.get("commit_id")
    if not review_commit_id and bot_comments:
        latest_comment = max(bot_comments, key=lambda c: c.get("created_at") or "")
        review_commit_id = latest_comment.get("commit_id") or latest_comment.get("original_commit_id")

    # The golden findings must describe the commit that gets replayed, so drop
    # comments anchored to any other one.
    if review_commit_id:
        snapshot_comments = [
            comment
            for comment in bot_comments
            if review_commit_id in (comment.get("commit_id"), comment.get("original_commit_id"))
        ]
        dropped = len(bot_comments) - len(snapshot_comments)
        if dropped:
            print_dim(
                console,
                f"PR #{number}: dropped {dropped} inline comment(s) outside snapshot {review_commit_id[:8]}",
            )
        bot_comments = snapshot_comments

    threads, threads_complete = fetch_review_threads(repo_slug, number)
    bot_threads = [t for t in threads if bot_login_matches(t.get("author"), bot)]

    return {
        "repo": repo_slug,
        "pr_number": number,
        "title": pr.get("title"),
        "state": pr.get("state"),
        "merged_at": pr.get("merged_at"),
        "created_at": pr.get("created_at"),
        "base_ref": (pr.get("base") or {}).get("ref"),
        "base_sha": (pr.get("base") or {}).get("sha"),
        "head_ref": (pr.get("head") or {}).get("ref"),
        "bot": bot,
        "review_commit_id": review_commit_id,
        "n_review_summaries": len(bot_reviews),
        "n_inline_comments": len(bot_comments),
        "reviews": [
            {
                "id": r.get("id"),
                "commit_id": r.get("commit_id"),
                "submitted_at": r.get("submitted_at"),
                "state": r.get("state"),
                "body": r.get("body"),
            }
            for r in bot_reviews
        ],
        "comments": [
            {
                "id": c.get("id"),
                "path": c.get("path"),
                "line": c.get("line"),
                "original_line": c.get("original_line"),
                "start_line": c.get("start_line"),
                "commit_id": c.get("commit_id"),
                "original_commit_id": c.get("original_commit_id"),
                "in_reply_to_id": c.get("in_reply_to_id"),
                "subject_type": c.get("subject_type"),
                "created_at": c.get("created_at"),
                "body": c.get("body"),
            }
            for c in bot_comments
        ],
        "threads": bot_threads,
        "threads_complete": threads_complete,
    }


def _resolved_lookup(record: dict[str, Any]) -> dict[tuple[Any, Any], bool]:
    """Map ``(path, line)`` to whether that review thread was resolved."""
    lookup: dict[tuple[Any, Any], bool] = {}
    for thread in record.get("threads", []):
        key = (thread.get("path"), thread.get("line"))
        lookup[key] = bool(thread.get("is_resolved")) or lookup.get(key, False)
    return lookup


def build_harvested_corpus(
    records: list[dict[str, Any]], *, repo: str, bot: str
) -> dict[str, Any]:
    """Project harvest records into a ``benchmark_data.json`` corpus dict.

    The bot's standalone inline comments become the ``golden_comments`` (the
    recall denominator), each carrying the ``"comment"`` key the judge reads plus
    a ``resolved`` flag preserving the acted-upon signal as metadata. The same
    comments are additionally injected as the bot's own review entry under the
    stripped bot stem, so the bot is scorable as just another tool label.
    """
    corpus: dict[str, Any] = {}
    tool = bot_stem(bot)
    for record in records:
        golden_url = f"https://github.com/{repo}/pull/{record['pr_number']}"
        resolved_by_anchor = _resolved_lookup(record)
        golden_comments = [
            {
                "comment": comment.get("body") or "",
                "path": comment.get("path"),
                "line": comment.get("line"),
                "resolved": resolved_by_anchor.get((comment.get("path"), comment.get("line")), False),
                "severity": None,
            }
            for comment in record.get("comments", [])
        ]
        corpus[golden_url] = {
            "pr_url": golden_url,
            "repo_name": repo,
            "golden_comments": golden_comments,
            "reviews": [
                {
                    "tool": tool,
                    "repo_name": repo,
                    "pr_url": golden_url,
                    "review_comments": [
                        {
                            "path": comment.get("path"),
                            "line": comment.get("line"),
                            "body": comment.get("body") or "",
                            "created_at": comment.get("created_at"),
                        }
                        for comment in record.get("comments", [])
                    ],
                }
            ],
        }
    return corpus


def _list_pull_requests(repo_slug: str, *, limit: int, state: str) -> list[dict[str, Any]]:
    """List up to *limit* pull requests in *repo_slug* matching *state*.

    ``merged`` is not a REST state: it is requested as ``closed`` and filtered on
    a non-null ``merged_at``.
    """
    rest_state = "closed" if state == "merged" else state
    prs = _paginated(f"repos/{repo_slug}/pulls?state={rest_state}&per_page=100&sort=created&direction=desc")
    if state == "merged":
        prs = [pr for pr in prs if pr.get("merged_at")]
    return prs[:limit]


def run_harvest(repo: str, bot: str, out_dir: Path, *, limit: int = 200, state: str = "all") -> int:
    """Harvest *bot*'s reviews in *repo* into a corpus rooted at *out_dir*.

    Returns:
        ``0`` on success; ``2`` when *repo* is not an ``owner/repo`` slug.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        print_warning(console, "--repo must be owner/repo")
        return 2

    print_info(console, f"Listing up to {limit} {state} PR(s) in {repo}…")
    prs = _list_pull_requests(repo, limit=limit, state=state)

    index: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for pr in prs:
        number = pr["number"]
        try:
            record = harvest_pr(repo, pr, bot)
        except (git_ops.GitError, json.JSONDecodeError) as exc:
            print_warning(console, f"PR #{number}: {type(exc).__name__}: {exc}; skipping")
            continue
        if record is None:
            continue
        records.append(record)
        _write_json(out_dir / "harvest" / f"pr-{number}.json", record)
        resolved = sum(1 for t in record["threads"] if t.get("is_resolved"))
        index.append(
            {
                "pr_number": number,
                "title": record["title"],
                "state": record["state"],
                "merged": bool(record["merged_at"]),
                "base_ref": record["base_ref"],
                "base_sha": record["base_sha"],
                "review_commit_id": record["review_commit_id"],
                "n_inline_comments": record["n_inline_comments"],
                "n_review_summaries": record["n_review_summaries"],
                "n_resolved_threads": resolved,
                "threads_complete": record["threads_complete"],
            }
        )
        print_dim(
            console,
            f"PR #{number}: {record['n_inline_comments']} inline, "
            f"{record['n_review_summaries']} summaries, {resolved} resolved "
            f"@ {str(record['review_commit_id'])[:8]}",
        )

    index.sort(key=lambda r: r["pr_number"], reverse=True)
    _write_json(
        out_dir / "index.json",
        {"repo": repo, "bot": bot, "n_prs_with_bot_activity": len(index), "prs": index},
    )
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_benchmark_data(results_dir / "benchmark_data.json", build_harvested_corpus(records, repo=repo, bot=bot))
    print_info(console, f"Harvested {len(index)} PR(s) with {bot} activity -> {out_dir}")
    return 0


def _write_json(path: Path, obj: Any) -> None:
    """Write *obj* as indented JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

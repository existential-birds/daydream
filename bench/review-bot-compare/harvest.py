#!/usr/bin/env python3
"""Harvest a review bot's historic PR reviews from GitHub.

Generalized over any bot (`--bot coderabbitai[bot]`, a greptile handle, ...)
and any repo (`--repo owner/repo`). Pure `gh api` — no local checkout needed.

For every PR the bot reviewed, captures:
  - the bot's review summaries (with the commit_id each was made against)
  - the bot's inline review comments (path, line, body, commit_id)
  - review-thread resolution status (the "acted upon" ground-truth signal)
  - the snapshot SHA to replay against (latest bot review's commit_id)

Output: <out>/<owner__repo>/pr-<N>.json plus index.json.

Usage:
  python harvest.py --repo owner/repo --bot "coderabbitai[bot]" \
      --out ./out --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    bot_login_matches,
    gh_json,
    gh_paginate,
    graphql_review_threads,
    repo_slug,
    write_json,
)


def harvest_pr(repo: str, owner: str, name: str, pr: dict, bot: str) -> dict | None:
    """Collect one PR's bot activity, or None if the bot left nothing."""
    n = pr["number"]

    reviews = gh_paginate(f"repos/{repo}/pulls/{n}/reviews")
    bot_reviews = [r for r in reviews if bot_login_matches((r.get("user") or {}).get("login"), bot)]

    comments = gh_paginate(f"repos/{repo}/pulls/{n}/comments")
    bot_comments = [c for c in comments if bot_login_matches((c.get("user") or {}).get("login"), bot)]

    # Keep reviews carrying a body or any inline comment; skip pure approvals.
    bot_reviews = [r for r in bot_reviews if (r.get("body") or "").strip()]

    if not bot_reviews and not bot_comments:
        return None

    # Snapshot to replay against: latest bot review's commit_id, falling back to
    # the latest inline comment's commit_id.
    review_commit_id = None
    if bot_reviews:
        latest = max(bot_reviews, key=lambda r: r.get("submitted_at") or "")
        review_commit_id = latest.get("commit_id")
    if not review_commit_id and bot_comments:
        latest_c = max(bot_comments, key=lambda c: c.get("created_at") or "")
        review_commit_id = latest_c.get("commit_id") or latest_c.get("original_commit_id")

    threads = graphql_review_threads(owner, name, n)
    bot_threads = [t for t in threads if bot_login_matches(t.get("author"), bot)]

    return {
        "repo": repo,
        "pr_number": n,
        "title": pr.get("title"),
        "state": pr.get("state"),
        "merged_at": pr.get("mergedAt"),
        "created_at": pr.get("createdAt"),
        "base_ref": pr.get("baseRefName"),
        "head_ref": pr.get("headRefName"),
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
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--bot", required=True, help="bot user.login, e.g. coderabbitai[bot]")
    ap.add_argument("--out", default="./out", type=Path, help="output root dir")
    ap.add_argument("--limit", type=int, default=200, help="max PRs to scan")
    ap.add_argument("--state", default="all", choices=["all", "open", "closed", "merged"])
    args = ap.parse_args()

    owner, _, name = args.repo.partition("/")
    if not name:
        print("--repo must be owner/repo", file=sys.stderr)
        return 2

    out_dir = args.out / repo_slug(args.repo)

    print(f"Listing up to {args.limit} {args.state} PRs in {args.repo} ...", file=sys.stderr)
    prs = gh_json([
        "pr", "list", "--repo", args.repo,
        "--state", args.state, "--limit", str(args.limit),
        "--json", "number,title,state,baseRefName,headRefName,mergedAt,createdAt",
    ]) or []

    index = []
    for pr in prs:
        n = pr["number"]
        try:
            record = harvest_pr(args.repo, owner, name, pr, args.bot)
        except RuntimeError as e:
            print(f"  PR #{n}: error ({e}); skipping", file=sys.stderr)
            continue
        if record is None:
            continue
        write_json(out_dir / f"pr-{n}.json", record)
        resolved = sum(1 for t in record["threads"] if t.get("is_resolved"))
        index.append({
            "pr_number": n,
            "title": record["title"],
            "state": record["state"],
            "merged": bool(record["merged_at"]),
            "review_commit_id": record["review_commit_id"],
            "n_inline_comments": record["n_inline_comments"],
            "n_review_summaries": record["n_review_summaries"],
            "n_resolved_threads": resolved,
        })
        print(
            f"  PR #{n}: {record['n_inline_comments']} inline, "
            f"{record['n_review_summaries']} summaries, "
            f"{resolved} resolved @ {str(record['review_commit_id'])[:8]}",
            file=sys.stderr,
        )

    index.sort(key=lambda r: r["pr_number"], reverse=True)
    write_json(out_dir / "index.json", {
        "repo": args.repo,
        "bot": args.bot,
        "n_prs_with_bot_activity": len(index),
        "prs": index,
    })
    print(
        f"\nHarvested {len(index)} PRs with {args.bot} activity -> {out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

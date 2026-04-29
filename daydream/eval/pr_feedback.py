"""Fetch and correlate PR review feedback from GitHub.

Uses ``gh api`` to retrieve inline review comments daydream posted on a PR,
their reply threads, and the PR's overall disposition.  Matches comments to
trajectory findings by file path and line number.

Usage::

    from daydream.eval.pr_feedback import fetch_pr_feedback
    feedback = fetch_pr_feedback("HealthByRo/mono", 37979)
"""

import json
import subprocess
from typing import Any


def _gh_api(endpoint: str) -> Any:
    """Call ``gh api`` with pagination and return parsed JSON."""
    result = subprocess.run(  # noqa: S603 - endpoint is not user-controlled
        ["gh", "api", "--paginate", endpoint],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _is_daydream_comment(comment: dict) -> bool:
    """Heuristic: identify comments posted by daydream."""
    body = comment.get("body", "")
    user = comment.get("user", {}).get("login", "")
    # daydream posts via a bot or the user's account with distinctive markers
    markers = [
        "daydream",
        "Dependency Impact",
        "confidence:",
        "Confidence:",
        "QUAL-04",
        "review-output",
    ]
    return any(m.lower() in body.lower() for m in markers) or "daydream" in user.lower()


def _extract_file_line(comment: dict) -> tuple[str, int | None]:
    """Extract the file path and line from an inline review comment."""
    path = comment.get("path", "")
    line = comment.get("line") or comment.get("original_line")
    return path, line


def _classify_response(replies: list[dict]) -> str:
    """Classify the aggregate response to a review comment.

    Returns one of: accepted, rejected, discussed, no_response.
    """
    if not replies:
        return "no_response"

    texts = " ".join(r.get("body", "") for r in replies).lower()

    rejection_signals = [
        "disagree", "won't fix", "wontfix", "not a bug", "intentional",
        "by design", "false positive", "incorrect", "wrong", "nit",
        "this is fine", "already handled", "not applicable", "n/a",
    ]
    acceptance_signals = [
        "fixed", "done", "good catch", "thanks", "will fix",
        "addressed", "resolved", "agreed", "updated", "applied",
    ]

    rejection_hits = sum(1 for s in rejection_signals if s in texts)
    acceptance_hits = sum(1 for s in acceptance_signals if s in texts)

    if acceptance_hits > rejection_hits:
        return "accepted"
    if rejection_hits > acceptance_hits:
        return "rejected"
    return "discussed"


def _match_comment_to_finding(
    comment: dict,
    findings: list[dict],
) -> dict | None:
    """Match a PR comment to a finding by file path and line proximity."""
    c_path, c_line = _extract_file_line(comment)
    if not c_path:
        return None

    best = None
    best_distance = float("inf")

    for f in findings:
        f_file = f.get("file", "")
        f_line = f.get("line")
        if not f_file or not c_path.endswith(f_file):
            continue
        if f_line is not None and c_line is not None:
            distance = abs(f_line - c_line)
            if distance < best_distance:
                best = f
                best_distance = distance
        elif best is None:
            best = f

    if best is not None and best_distance <= 10:
        return best
    return best if best is not None else None


def fetch_pr_feedback(owner_repo: str, pr_number: int, findings: list[dict] | None = None) -> dict:
    """Fetch PR feedback and correlate with findings.

    Args:
        owner_repo: GitHub repo in ``owner/repo`` format.
        pr_number: Pull request number.
        findings: Optional list of finding dicts from ``analyze_findings()``
                  to match against PR comments.

    Returns:
        Dict with PR metadata, daydream comments, response classification,
        and finding correlations.
    """
    findings = findings or []

    # Fetch PR metadata
    pr = _gh_api(f"/repos/{owner_repo}/pulls/{pr_number}")
    pr_meta = {
        "number": pr_number,
        "repo": owner_repo,
        "state": pr.get("state"),
        "merged": pr.get("merged", False),
        "title": pr.get("title", ""),
        "created_at": pr.get("created_at"),
        "merged_at": pr.get("merged_at"),
    }

    # Fetch review objects (approve / request changes)
    reviews = _gh_api(f"/repos/{owner_repo}/pulls/{pr_number}/reviews")
    review_dispositions = [
        {"user": r.get("user", {}).get("login"), "state": r.get("state")}
        for r in reviews
    ]

    # Fetch inline review comments (includes reply threads)
    comments = _gh_api(f"/repos/{owner_repo}/pulls/{pr_number}/comments")

    # Separate daydream comments from replies
    daydream_comments: list[dict] = []
    reply_map: dict[int, list[dict]] = {}  # comment_id -> replies

    for c in comments:
        reply_to = c.get("in_reply_to_id")
        if reply_to:
            reply_map.setdefault(reply_to, []).append(c)
        elif _is_daydream_comment(c):
            daydream_comments.append(c)

    # Build per-comment analysis
    comment_analyses: list[dict] = []
    for dc in daydream_comments:
        cid = dc["id"]
        replies = reply_map.get(cid, [])
        path, line = _extract_file_line(dc)
        matched_finding = _match_comment_to_finding(dc, findings)
        response = _classify_response(replies)

        comment_analyses.append({
            "comment_id": cid,
            "file": path,
            "line": line,
            "body_preview": dc.get("body", "")[:200],
            "reply_count": len(replies),
            "response_classification": response,
            "reply_authors": list({r.get("user", {}).get("login") for r in replies}),
            "matched_finding_id": matched_finding.get("id") if matched_finding else None,
            "matched_finding_confidence": (
                matched_finding.get("confidence") if matched_finding else None
            ),
        })

    # Aggregate stats
    classifications = [ca["response_classification"] for ca in comment_analyses]
    acceptance_rate = (
        round(classifications.count("accepted") / len(classifications), 4)
        if classifications
        else None
    )
    rejection_rate = (
        round(classifications.count("rejected") / len(classifications), 4)
        if classifications
        else None
    )

    # Check for commits after review (proxy for "changes applied")
    commits = _gh_api(f"/repos/{owner_repo}/pulls/{pr_number}/commits")
    review_timestamps = [
        dc.get("created_at", "")
        for dc in daydream_comments
    ]
    latest_review_ts = max(review_timestamps) if review_timestamps else None
    commits_after_review = 0
    if latest_review_ts:
        for commit in commits:
            commit_date = (commit.get("commit", {}).get("committer", {}).get("date", ""))
            if commit_date > latest_review_ts:
                commits_after_review += 1

    return {
        "pr": pr_meta,
        "review_dispositions": review_dispositions,
        "daydream_comments": {
            "total": len(daydream_comments),
            "with_replies": sum(1 for ca in comment_analyses if ca["reply_count"] > 0),
            "no_response": sum(1 for ca in comment_analyses if ca["response_classification"] == "no_response"),
        },
        "response_summary": {
            "accepted": classifications.count("accepted"),
            "rejected": classifications.count("rejected"),
            "discussed": classifications.count("discussed"),
            "no_response": classifications.count("no_response"),
            "acceptance_rate": acceptance_rate,
            "rejection_rate": rejection_rate,
        },
        "commits_after_review": commits_after_review,
        "comment_details": comment_analyses,
    }

"""Normalize GitHub evidence and project candidates from immutable authoring anchors."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from daydream import git_ops
from daydream.benchmark import schema
from daydream.pr_review import FINDING_MARKER_RE


def _as_author(raw: dict[str, Any]) -> dict[str, Any]:
    author = raw.get("user") or {}
    return {"login": author.get("login", ""), "type": author.get("type", "User")}


def _record_common(author: dict[str, Any], body: str) -> dict[str, Any]:
    """The author + body-hash + bot block shared by every evidence record builder."""
    return {
        "author": {"login": author.get("login", ""), "type": author.get("type", "User")},
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "is_bot": author.get("type") == "Bot",
    }


def _rest_evidence(raw: dict[str, Any], kind: str, node_prefix: str) -> dict[str, Any]:
    """Normalize the shared REST identity, timestamps, author, and body contract."""
    db_id = int(raw["id"])
    return {
        "source_id": f"github:{kind}:{db_id}",
        "kind": kind,
        "database_id": db_id,
        "node_id": raw.get("node_id") or f"{node_prefix}_{db_id}",
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "url": raw.get("html_url") or "",
        **_record_common(raw.get("user") or {}, raw.get("body") or ""),
    }


def _evidence_from_review(raw: dict[str, Any]) -> dict[str, Any]:
    submitted = raw.get("submitted_at")
    return {
        **_rest_evidence(raw, "review", "PRR"),
        "created_at": raw.get("created_at") or submitted,
        "updated_at": raw.get("updated_at") or submitted,
        "submitted_at": submitted,
        **{key: raw.get(key) for key in ("commit_id", "original_commit_id", "state")},
    }


def _evidence_from_inline(raw: dict[str, Any]) -> dict[str, Any]:
    subject_type = raw.get("subject_type")
    if subject_type is None:
        subject_type = "file" if raw.get("path") is None else "line"
    return {
        **_rest_evidence(raw, "inline_comment", "DIFF"),
        **{key: raw.get(key) for key in (
            "commit_id", "original_commit_id", "path", "original_path", "line", "start_line",
            "original_line", "original_start_line", "side", "start_side",
        )},
        "thread_id": None,
        "review_id": str(raw["pull_request_review_id"]) if raw.get("pull_request_review_id") is not None else None,
        "reply_to_id": str(raw["in_reply_to_id"]) if raw.get("in_reply_to_id") is not None else None,
        "subject_type": subject_type,
    }


def _evidence_from_thread(thread: dict[str, Any], comment: dict[str, Any]) -> dict[str, Any]:
    db_id = int(comment["databaseId"])
    body = comment.get("body") or ""
    author = comment.get("author") or {}
    subject = str(thread.get("subjectType") or "").lower()
    subject_type = subject if subject in ("line", "file") else None
    fields = {
        "source_id": f"github:thread_comment:{db_id}",
        "kind": "thread_comment",
        "database_id": db_id,
        "node_id": comment.get("id") or f"TH_{db_id}",
        "created_at": comment.get("createdAt"),
        "updated_at": comment.get("updatedAt") or comment.get("createdAt"),
        "subject_type": subject_type,
        "side": thread.get("side"),
        "start_side": thread.get("startSide"),
        "path": thread.get("path"),
        "line": thread.get("line"),
        "original_line": thread.get("originalLine"),
        "original_start_line": thread.get("originalStartLine"),
        "resolved": bool(thread.get("isResolved", False)),
        "outdated": bool(thread.get("isOutdated", False)),
        "thread_id": thread.get("id"),
        "reply_to_id": (comment.get("replyTo") or {}).get("id"),
        "url": comment.get("url") or "",
    }
    fields.update(_record_common(author, body))
    return fields


def _canonical_comment_from_thread(
    thread: dict[str, Any], comment: dict[str, Any]
) -> dict[str, Any]:
    """Normalize a GraphQL-only comment as inline evidence; REST-only commit anchors stay absent."""
    rec = _evidence_from_thread(thread, comment)
    db_id = rec["database_id"]
    rec["source_id"] = f"github:inline_comment:{db_id}"
    rec["kind"] = "inline_comment"
    rec["commit_id"] = None
    rec["original_commit_id"] = None
    rec["review_id"] = None
    rec["dismissed"] = False
    return rec


def _reconcile_inline_evidence(
    inline_records: list[dict[str, Any]],
    thread_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay thread state onto REST comments by database ID, falling back to node ID."""
    inline_by_db = {rec["database_id"]: rec for rec in inline_records}
    inline_by_node = {rec["node_id"]: rec for rec in inline_records if rec.get("node_id")}
    canonical: list[dict[str, Any]] = [dict(rec) for rec in inline_records]
    canonical_by_db = {rec["database_id"]: rec for rec in canonical}
    for thread in thread_nodes:
        nodes = thread.get("comments", {}).get("nodes")
        if nodes is None:
            continue  # a thread with no comments contributes nothing
        for comment in nodes:
            if not isinstance(comment, dict) or "databaseId" not in comment:
                raise git_ops.GitError(
                    f"graphql review thread {thread.get('id')} comment node missing databaseId"
                )
            db_id = int(comment["databaseId"])
            base = inline_by_db.get(db_id)
            if base is None and comment.get("id"):
                base = inline_by_node.get(comment["id"])
            if base is not None:
                rec = canonical_by_db[base["database_id"]]
            else:
                rec = _canonical_comment_from_thread(thread, comment)
                canonical.append(rec)
            rec["thread_id"] = thread.get("id")
            rec["resolved"] = bool(thread.get("isResolved", False))
            rec["outdated"] = bool(thread.get("isOutdated", False))
            if rec.get("reply_to_id") is None:
                rec["reply_to_id"] = (comment.get("replyTo") or {}).get("id")
    return canonical


def _join_dismissal(
    canonical: list[dict[str, Any]], review_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join DISMISSED review state onto its inline comments by review ID."""
    states = {str(int(raw["id"])): raw.get("state") for raw in review_records}
    for rec in canonical:
        review_id = rec.get("review_id")
        if rec.get("kind") == "inline_comment" and review_id and states.get(review_id) == "DISMISSED":
            rec["dismissed"] = True
    return canonical


def _normalize_body(body: str) -> str:
    """Normalize line endings and strip trailing whitespace at the document end."""
    return body.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def _projected_body(body: str) -> str:
    """Normalize candidate text after removing canonical Daydream markers."""
    return _normalize_body(FINDING_MARKER_RE.sub("", body))


_MARKDOWN_PREFIX = re.compile(r"^(#{1,6}\s+|[-*]\s+)")


def _derive_title(body: str) -> str:
    """The bounded title from the first nonblank line of a body."""
    for line in body.split("\n"):
        if not line.strip():
            continue
        title = " ".join(line.split())
        return _MARKDOWN_PREFIX.sub("", title, count=1)
    return ""


def _title_ok(title: str) -> bool:
    """True when *title* is non-empty and within the 500-character bound."""
    return 0 < len(title) <= 500


def _anchor_location(
    evidence: schema.EvidenceRecord,
) -> tuple[schema.Location | None, str | None]:
    """Project location solely from a strict authoring anchor."""
    if evidence.subject_type != "file" and (evidence.side == "LEFT" or evidence.start_side == "LEFT"):
        return None, "side"
    anchor = evidence.authoring_anchor
    if anchor is None:
        return None, "history-unavailable"
    if anchor.status != "derived":
        return None, anchor.status
    if (
        evidence.subject_type == "file"
        or anchor.start_line is None
        or anchor.end_line is None
        or anchor.start_line < 1
        or anchor.end_line < anchor.start_line
    ):
        return None, "range-unavailable"
    if anchor.path is None:
        return None, "path-unavailable"
    return (
        schema.Location(path=anchor.path, start_line=anchor.start_line, end_line=anchor.end_line),
        None,
    )


def _project_one(evidence: schema.EvidenceRecord, head_sha: str) -> schema.Candidate:
    body = _projected_body(evidence.body)
    title = _derive_title(body)
    location: schema.Location | None = None
    reason: str | None = None
    if evidence.kind == "inline_comment":
        location, reason = _anchor_location(evidence)
        anchor = evidence.authoring_anchor
        if reason is None and anchor is not None and anchor.commit_id != head_sha:
            reason = "re-anchored"
    elif evidence.commit_id != head_sha:
        reason = "commit"

    # Later exclusions take precedence over anchor/commit failures.
    for excluded, code in (
        (not _title_ok(title), "title"),
        (evidence.outdated, "outdated"),
        (evidence.dismissed, "dismissed"),
    ):
        if excluded:
            reason = code
    return schema.Candidate(
        source_id=evidence.source_id,
        title=title,
        body=body,
        severity=None,
        location=location,
        exact_acceptable=reason is None,
        not_exact_reason=reason,
    )


def project_candidates(
    doc: schema.ImportDocument, head_sha: str
) -> list[schema.Candidate]:
    """Project nonempty root inline comments and COMMENTED/CHANGES_REQUESTED reviews."""
    cands: list[schema.Candidate] = []
    for evidence in doc.evidence:
        if not evidence.body:
            continue
        if evidence.kind == "inline_comment":
            if evidence.reply_to_id is not None:
                continue  # replies are evidence, not candidates
        elif evidence.kind == "review":
            if evidence.state not in ("COMMENTED", "CHANGES_REQUESTED"):
                continue
        else:
            continue
        cands.append(_project_one(evidence, head_sha))
    return cands


def _evidence_projection_hash(rec: dict[str, Any]) -> str:
    """Hash projection content with canonical defaults; metadata changes do not stale gold."""
    author_raw = rec.get("author")
    author = author_raw if isinstance(author_raw, dict) else {}
    anchor_raw = rec.get("authoring_anchor")
    anchor = {
        key: anchor_raw.get(key)
        for key in ("version", "status", "commit_id", "path", "start_line", "end_line")
    } if isinstance(anchor_raw, dict) else None
    values = {
        key: rec.get(key)
        for key in (
            "commit_id", "original_commit_id", "path", "original_path", "line", "start_line",
            "original_line", "original_start_line", "side", "start_side", "subject_type",
            "reply_to_id", "state",
        )
    }
    values.update({
        "body_sha256": str(rec.get("body_sha256") or ""),
        "author.login": str(author.get("login") or ""),
        "author.type": str(author.get("type") or ""),
        "authoring_anchor": anchor,
        **{key: bool(rec.get(key, False)) for key in ("resolved", "outdated", "dismissed")},
    })
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_signature_from_doc(
    doc: schema.ImportDocument,
    *,
    downgrade_start_line: set[int] | None = None,
) -> frozenset[tuple[int, str]]:
    """Return immutable ``(database_id, projection_hash)`` pairs for typed evidence."""
    return frozenset(
        (e.database_id, _evidence_projection_hash(_signature_dict(e, downgrade_start_line=downgrade_start_line)))
        for e in doc.evidence
    )


def _signature_dict(
    e: schema.EvidenceRecord, *, downgrade_start_line: set[int] | None = None
) -> dict[str, Any]:
    """Dump evidence for hashing, omitting only explicitly identified legacy range fields."""
    rd = e.model_dump(mode="json")
    if downgrade_start_line is not None and e.database_id in downgrade_start_line:
        rd.pop("original_start_line", None)
    return rd


def _evidence_signature_from_raw(raw: dict[str, Any]) -> frozenset[tuple[int, str]]:
    """Hash raw evidence with the same defaults, retaining distinct legacy projections per id."""
    return frozenset(
        (int(e["database_id"]), _evidence_projection_hash(e))
        for e in raw.get("evidence", [])
    )

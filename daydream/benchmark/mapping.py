"""Deterministic mapping from canonical merged items to benchmark review comments.

Mirrors the transforms in :func:`daydream.pr_review.parsed_issues_from_items`,
but folds ``description`` into the comment ``body`` and emits the benchmark
``review_comments`` shape (``{path, line, body, created_at, confidence, severity}``) instead of a
``ParsedIssue``. Pure and deterministic — no I/O, no LLM.
"""

from __future__ import annotations

from typing import Any

from daydream.pr_review import extract_item_fields


def merged_items_to_review_comments(
    doc: dict[str, Any],
    *,
    created_at: str,
) -> list[dict[str, Any]]:
    """Convert a merged-items document into benchmark review comments.

    Args:
        doc: Parsed ``merged-items.json`` with a top-level ``items`` list.
        created_at: Timestamp passed verbatim onto every emitted comment.

    Returns:
        A list of ``{path, line, body, created_at, confidence, severity}``
        dicts, one per item with a non-empty ``file``. Items with an empty
        ``file`` are skipped, as are items whose assembled ``body`` is empty
        (no ``description``, ``severity``, ``confidence``, or ``rationale``
        text) — an empty-body comment is never emitted. A null
        or non-integer ``line`` is preserved as ``None``. The ``body`` leads
        with the finding ``description``, followed by ``**Severity:**`` and
        ``**Confidence:**`` badges, and the ``rationale`` only when it differs
        from the description; sections are joined with ``"\\n\\n"``. The
        structured ``confidence`` (uppercased) and ``severity`` (lowercased)
        fields carry the same values, ``None`` when absent; the body badges
        are kept for human/judge readability.
    """
    out: list[dict[str, Any]] = []
    for raw in doc.get("items", []):
        fields = extract_item_fields(raw)
        if fields is None:
            continue
        body_parts: list[str] = []
        if fields.description:
            body_parts.append(fields.description)
        if fields.severity:
            body_parts.append(f"**Severity:** {fields.severity}")
        if fields.confidence:
            body_parts.append(f"**Confidence:** {fields.confidence}")
        if fields.rationale and fields.rationale != fields.description:
            body_parts.append(fields.rationale)
        body = "\n\n".join(body_parts)
        if not body.strip():
            continue
        out.append(
            {
                "path": fields.path,
                "line": fields.line_int,
                "body": body,
                "created_at": created_at,
                "confidence": fields.confidence,
                "severity": fields.severity,
            }
        )
    return out

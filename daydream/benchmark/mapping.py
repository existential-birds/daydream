"""Deterministic mapping from canonical merged items to benchmark review comments.

Mirrors the transforms in :func:`daydream.pr_review.parsed_issues_from_items`,
but folds ``description`` into the comment ``body`` and emits the benchmark
``review_comments`` shape (``{path, line, body, created_at}``) instead of a
``ParsedIssue``. Pure and deterministic — no I/O, no LLM.
"""

from __future__ import annotations

from typing import Any

from daydream.pr_review import _extract_item_fields


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
        A list of ``{path, line, body, created_at}`` dicts, one per item with
        a non-empty ``file``. Items with an empty ``file`` are skipped; a null
        or non-integer ``line`` is preserved as ``None``. The ``body`` leads
        with the finding ``description``, followed by ``**Severity:**`` and
        ``**Confidence:**`` badges, and the ``rationale`` only when it differs
        from the description; sections are joined with ``"\\n\\n"``.
    """
    out: list[dict[str, Any]] = []
    for raw in doc.get("items", []):
        fields = _extract_item_fields(raw)
        if fields is None:
            continue
        path, line_int, description, rationale, severity, confidence, _is_cross_stack = fields
        body_parts: list[str] = []
        if description:
            body_parts.append(description)
        if severity:
            body_parts.append(f"**Severity:** {severity}")
        if confidence:
            body_parts.append(f"**Confidence:** {confidence}")
        if rationale and rationale != description:
            body_parts.append(rationale)
        body = "\n\n".join(body_parts)
        out.append(
            {
                "path": path,
                "line": line_int,
                "body": body,
                "created_at": created_at,
            }
        )
    return out

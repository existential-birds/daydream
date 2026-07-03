"""Deterministic mapping from canonical merged items to benchmark review comments.

Mirrors the transforms in :func:`daydream.pr_review.parsed_issues_from_items`,
but folds ``description`` into the comment ``body`` and emits the benchmark
``review_comments`` shape (``{path, line, body, created_at, confidence, severity}``) instead of a
``ParsedIssue``. Pure and deterministic — no I/O, no LLM.
"""

from __future__ import annotations

from typing import Any

from daydream.pr_review import extract_item_fields

_RANKS: dict[str, int] = {"low": 1, "medium": 2, "high": 3}


def _below_threshold(value: str | None, threshold: str | None) -> bool:
    """Return True when threshold is set and value is missing or ranks below it."""
    if threshold is None:
        return False
    if value is None:
        return True  # missing field + threshold set → drop (conservative)
    return _RANKS.get(value.lower(), 0) < _RANKS.get(threshold.lower(), 0)


def merged_items_to_review_comments(
    doc: dict[str, Any],
    *,
    created_at: str,
    min_confidence: str | None = None,
    min_severity: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a merged-items document into benchmark review comments.

    Args:
        doc: Parsed ``merged-items.json`` with a top-level ``items`` list.
        created_at: Timestamp passed verbatim onto every emitted comment.
        min_confidence: When set, drop any item whose confidence ranks below
            this threshold (or is missing). Compared case-insensitively on the
            low/medium/high scale.
        min_severity: When set, drop any item whose severity ranks below this
            threshold (or is missing). Compared case-insensitively on the
            low/medium/high scale.

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
        if _below_threshold(fields.confidence, min_confidence):
            continue
        if _below_threshold(fields.severity, min_severity):
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

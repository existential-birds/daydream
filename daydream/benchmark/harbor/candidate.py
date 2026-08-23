"""Candidate artifact builder for the privacy-safe Harbor review agent (issue #780).

Pure, stdlib-only, host-testable: converts the deep pipeline's canonical
``merged-items.json`` entries into the strict §9 candidate artifact. No Harbor
import; the verifier's own ``verifier_core`` derivation and caps are reused so
the agent produces exactly what the verifier re-derives (a drift would fail
every trial). Every failure carries a ``kind`` so a failed run reports which
failure class occurred (missing/corrupt merged output, over-limit, invalid
candidate, write failure) instead of presenting silence.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import verifier_core as vc
from daydream.pr_review import extract_item_fields

_ARTIFACT_KEYS = ("schema_version", "case_id", "base_ref", "head_ref", "findings")


class CandidateError(Exception):
    """Typed agent-failure carrier for candidate artifact production.

    Attributes:
        kind: The failure class -- ``"over_limit"`` or ``"write_failure"``
            (``"missing_merged"`` / ``"corrupt_merged"`` are raised by the
            entrypoint's publish step with the same carrier).
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _assemble_body(fields: Any) -> str:
    """Assemble the candidate body exactly like ``benchmark.mapping``.

    The body leads with the finding description, followed by the
    ``**Severity:**`` / ``**Confidence:**`` badges, and the rationale only when
    it differs from the description; sections are joined with ``"\\n\\n"``. This
    guarantees a non-blank body whenever the title is non-blank (the verifier
    rejects blank titles and bodies) and mirrors the canonical consumers.
    """
    parts: list[str] = []
    if fields.description:
        parts.append(fields.description)
    if fields.severity:
        parts.append(f"**Severity:** {fields.severity}")
    if fields.confidence:
        parts.append(f"**Confidence:** {fields.confidence}")
    if fields.rationale and fields.rationale != fields.description:
        parts.append(fields.rationale)
    return "\n\n".join(parts)


def build_candidate_findings(items: list[dict], *, case_id: str) -> list[dict]:
    """Convert canonical merged items into candidate findings.

    Each item is mapped through ``pr_review.extract_item_fields`` (the shared
    item contract). Skipped items: an empty ``file`` (``extract_item_fields``
    returns ``None``), a ``line`` that is not a positive int (a null/invalid
    line would emit a partially-populated location the verifier rejects), and
    an item whose mapped title or body is blank (the verifier rejects blanks).
    Never fabricates a path or line.

    ``candidate_id`` mirrors ``verifier_core.derive_candidate_id`` byte-for-byte:
    the canonical six-field tuple (each ``or ""``) with a zero-based ordinal per
    identical tuple in merged-item order, hashed with the opaque ``case_id``
    salt, so the hidden verifier re-derives identical ids.
    """
    findings: list[dict[str, Any]] = []
    groups: dict[tuple[str, ...], int] = {}
    for raw in items:
        fields = extract_item_fields(raw)
        if fields is None:
            continue
        if fields.line_int is None or fields.line_int < 1:
            continue
        title = fields.description
        body = _assemble_body(fields)
        if not title.strip() or not body.strip():
            continue
        entry: dict[str, Any] = {
            "title": title,
            "body": body,
            "severity": fields.severity,
            "path": fields.path,
            "start_line": fields.line_int,
            "end_line": fields.line_int,
        }
        canonical = (
            title or "",
            body or "",
            fields.severity or "",
            fields.path or "",
            str(fields.line_int),
            str(fields.line_int),
        )
        ordinal = groups.get(canonical, 0)
        groups[canonical] = ordinal + 1
        entry["candidate_id"] = vc.derive_candidate_id(case_id, entry, ordinal)
        findings.append(entry)
    return findings

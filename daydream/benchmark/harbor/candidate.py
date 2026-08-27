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


class CandidateError(Exception):
    """Typed agent-failure carrier for candidate artifact production.

    Attributes:
        kind: The failure class -- ``"over_limit"``, ``"invalid_finding"`` or
            ``"write_failure"`` (``"missing_merged"`` / ``"corrupt_merged"``
            are raised by the entrypoint's publish step with the same carrier).
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _candidate_title(description: str) -> str:
    """Return a verifier-bounded display title without dropping review content.

    Canonical Daydream descriptions can be paragraph-length. The full text is
    retained in the candidate body; the title uses its first non-empty line and
    is shortened deterministically only when that line exceeds the verifier's
    500-character display bound.
    """
    first_line = next(
        (line.strip() for line in description.splitlines() if line.strip()),
        description.strip(),
    )
    if len(first_line) <= 500:
        return first_line
    return first_line[:497].rstrip() + "..."


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


def build_candidate_findings(items: list[dict[str, Any]], *, case_id: str) -> list[dict[str, Any]]:
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
        title = _candidate_title(fields.description)
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
        # Enforce the verifier's per-finding bounds fail-closed, reusing the
        # verifier's own validators so no drift surfaces as a verifier-rejected
        # artifact after the builder already declared success: an over-long
        # body (>8 KiB), a non-enum severity, a
        # rooted/'..'-containing/NUL path, or a non-positive/non-ascending
        # line range are each a typed failure, never an artifact the verifier
        # would reject (``_validate_location`` re-checks path + lines on parse).
        try:
            vc._validate_title(title)
            vc._validate_body(body)
            vc._validate_severity(entry["severity"])
            vc._validate_path(entry["path"])
            vc._validate_lines(entry["start_line"], entry["end_line"])
        except vc.VerifierError as exc:
            raise CandidateError(
                f"cannot build candidate finding: {exc}", kind="invalid_finding"
            ) from exc
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


def build_candidate_artifact(
    case_id: str,
    findings: list[dict[str, Any]],
    *,
    base_ref: str = "base",
    head_ref: str = "head",
) -> dict[str, Any]:
    """Assemble the strict §9 candidate artifact, enforcing the caps fail-closed.

    Returns ``{"schema_version": 1, "case_id": case_id, "base_ref": base_ref,
    "head_ref": head_ref, "findings": findings}`` -- an empty ``findings`` list
    is a clean review. ``base_ref``/``head_ref`` default to ``"base"``/`"head"`
    per the bound-task contract and are threaded from the container env key on
    the entrypoint path. Exceeding the 100-finding cap or the 1 MiB
    serialized-artifact cap raises ``CandidateError(kind="over_limit")`` --
    never silently truncates to fit.
    """
    artifact = {
        "schema_version": 1,
        "case_id": case_id,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "findings": findings,
    }
    if len(findings) > vc.MAX_CANDIDATE_FINDINGS:
        raise CandidateError(
            f"candidate artifact exceeds {vc.MAX_CANDIDATE_FINDINGS} findings "
            f"({len(findings)})",
            kind="over_limit",
        )
    if len(json.dumps(artifact).encode("utf-8")) > vc.MAX_ARTIFACT_BYTES:
        raise CandidateError(
            f"candidate artifact exceeds {vc.MAX_ARTIFACT_BYTES} bytes",
            kind="over_limit",
        )
    return artifact


def write_candidate_artifact_atomic(dest: str | Path, artifact: dict[str, Any]) -> None:
    """Write *artifact* to *dest* atomically (temp + rename).

    Writes to a sibling ``.tmp-<uuid>`` file in the destination directory then
    ``os.replace``s it into place, so a reader sees either the prior complete
    artifact or the complete new artifact -- never a torn write. Any ``OSError``
    raises ``CandidateError(kind="write_failure")``; a failure is never
    silently discarded.
    """
    dest = Path(dest)
    payload = json.dumps(artifact).encode("utf-8")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / (dest.name + f".tmp-{uuid.uuid4().hex}")
        tmp.write_bytes(payload)
        os.replace(tmp, dest)
    except OSError as exc:
        raise CandidateError(
            f"cannot write candidate artifact {dest}: {exc}", kind="write_failure"
        ) from exc

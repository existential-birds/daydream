"""Build privacy-safe Harbor candidates from canonical merged findings. Reuse verifier
identity/cap rules. Typed failures distinguish missing/corrupt input, invalid findings,
size limits, and write errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import verifier_core as vc
from daydream.json_utils import atomic_write_bytes, umask_derived_mode
from daydream.pr_review import extract_item_fields


class CandidateError(Exception):
    """Candidate failure with a stable kind, also used for missing/corrupt merged output."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _candidate_title(description: str) -> str:
    """Bound the first nonempty description line to 500 characters; retain full content in
    the body.
    """
    first_line = next(
        (line.strip() for line in description.splitlines() if line.strip()),
        description.strip(),
    )
    if len(first_line) <= 500:
        return first_line
    return first_line[:497].rstrip() + "..."


def _assemble_body(fields: Any) -> str:
    """Join description, severity/confidence badges, and distinct rationale into a nonblank
    body.
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
    """Project canonical findings without inventing locations. Skip missing files,
    nonpositive/noninteger lines, and blank title/body. Derive ids with opaque case salt
    and per-identical-content ordinals in merged order, normalizing nullable tuple
    values like the verifier.
    """
    findings: list[dict[str, Any]] = []
    groups: dict[tuple[object, ...], int] = {}
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
        entry_input = {
            "title": title,
            "body": body,
            "severity": fields.severity,
            "path": fields.path,
            "start_line": fields.line_int,
            "end_line": fields.line_int,
        }
        try:
            entry = vc.parse_finding_content(entry_input)
        except vc.VerifierError as exc:
            raise CandidateError(
                f"cannot build candidate finding: {exc}", kind="invalid_finding"
            ) from exc
        wire = entry.to_dict()
        wire["candidate_id"] = vc.assign_candidate_id(case_id, entry, groups)
        findings.append(wire)
    return findings


def build_candidate_artifact(
    case_id: str,
    findings: list[dict[str, Any]],
    *,
    base_ref: str = "base",
    head_ref: str = "head",
) -> dict[str, Any]:
    """Build schema-1 candidates with opaque case and bound base/head refs. Empty findings
    mean a clean review. Enforce 100 findings and 1 MiB serialized size by raising
    over_limit; never truncate evidence to fit.
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
    try:
        vc.validate_candidate_artifact(artifact)
    except vc.VerifierError as exc:
        raise CandidateError(
            f"cannot build candidate artifact: {exc}", kind="invalid_finding"
        ) from exc
    return artifact


def write_candidate_artifact_atomic(dest: str | Path, artifact: dict[str, Any]) -> None:
    """Atomically replace complete candidate bytes; OSError becomes a write_failure
    CandidateError.
    """
    dest = Path(dest)
    payload = json.dumps(artifact).encode("utf-8")
    try:
        atomic_write_bytes(dest, payload, fsync=False, dir_fsync=False, mode=umask_derived_mode())
    except OSError as exc:
        raise CandidateError(
            f"cannot write candidate artifact {dest}: {exc}", kind="write_failure"
        ) from exc

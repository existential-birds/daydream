"""Build privacy-safe Harbor candidates from sealed committed-review results.
Reuse verifier identity/cap rules. Typed failures distinguish invalid or incomplete
results, unsupported locations, size limits, and write errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import verifier_core as vc
from daydream.committed_review_model import CommittedReviewResult, validate_committed_result
from daydream.json_utils import atomic_write_bytes, umask_derived_mode


class CandidateError(Exception):
    """Candidate failure with a stable kind."""

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


def build_committed_candidate(
    result: CommittedReviewResult, *, immutable_input: dict[str, Any], case_id: str,
    base_ref: str = "base", head_ref: str = "head",
) -> dict[str, Any]:
    """Project only complete installed current evidence, preserving local source identity."""
    value = result.to_dict()
    try:
        validate_committed_result(value)
    except (ValueError, TypeError) as exc:
        raise CandidateError(f"invalid committed review result: {exc}", kind="invalid_result") from exc
    if (not result.complete or result.capture_state != "complete" or result.output_error is not None
            or result.delivery_state not in {"not_requested", "complete"}
            or value.get("immutable_input") != immutable_input):
        raise CandidateError("committed review is incomplete or belongs to another immutable input", kind="incomplete")
    findings: list[dict[str, Any]] = []
    groups: dict[tuple[object, ...], int] = {}
    for claim in value["findings"]:
        payload = claim["payload"]
        if payload["line"] is None:
            raise CandidateError("Harbor candidates require a concrete finding line", kind="unsupported_location")
        title = _candidate_title(claim["title"]) or _candidate_title(payload["claim"])
        body = [payload[key] for key in ("claim", "trigger", "consequence", "grounds")]
        if claim["title"].strip() and title != claim["title"].strip():
            body.insert(0, claim["title"])
        finding = {
            "title": title,
            "body": "\n\n".join(body),
            "severity": None, "path": payload["file"], "start_line": payload["line"], "end_line": payload["line"],
        }
        try:
            entry = vc.parse_finding_content(finding)
        except vc.VerifierError as exc:
            raise CandidateError(f"cannot project committed finding: {exc}", kind="invalid_finding") from exc
        entry["candidate_id"] = vc.assign_candidate_id(case_id, entry, groups)
        findings.append(entry)
    return build_candidate_artifact(case_id, findings, base_ref=base_ref, head_ref=head_ref)


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

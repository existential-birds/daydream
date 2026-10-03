"""Derive authoring locations and comparison facts from authenticated Git snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from daydream import git_ops
from daydream.benchmark import schema, snapshot
from daydream.benchmark.schema import EXTRACTION_VERSION


def _anchor_fail_closed(
    status: Literal["history-unavailable", "path-unavailable", "range-unavailable"],
) -> schema.AuthoringAnchor:
    """Construct a closed anchor status with every data field unset."""
    return schema.AuthoringAnchor(
        version=1, status=status, commit_id=None, path=None, start_line=None, end_line=None,
    )


def _derive_one_anchor(
    record: schema.EvidenceRecord,
    mirror_repo: Path,
    head_sha: str,
) -> schema.AuthoringAnchor:
    """Derive one root inline authoring anchor through the pinned mirror."""
    original_commit_id = record.original_commit_id
    if original_commit_id is None:
        return _anchor_fail_closed("history-unavailable")
    start = record.original_start_line or record.original_line
    end = record.original_line
    if start is None or end is None:
        return _anchor_fail_closed("range-unavailable")
    if start > end:
        # GitHub can supply inverted ranges; close this anchor before model validation.
        return _anchor_fail_closed("range-unavailable")
    path = record.path or record.original_path
    if path is None:
        return _anchor_fail_closed("path-unavailable")
    try:
        authoring_path = snapshot.derive_authoring_path(
            mirror_repo, original_commit_id, path, head_sha
        )
    except snapshot.AnchorDerivationError as exc:
        # The closed reason rides on the exception; anything else is history.
        if exc.reason == "path-unavailable":
            return _anchor_fail_closed("path-unavailable")
        return _anchor_fail_closed("history-unavailable")
    except git_ops.GitError:
        # A hard git failure (subprocess/OS-level, e.g. a rename-trace diff
        # timeout) fails the anchor closed rather than aborting the import.
        return _anchor_fail_closed("history-unavailable")
    try:
        return schema.AuthoringAnchor(
            version=1, status="derived", commit_id=original_commit_id,
            path=authoring_path, start_line=start, end_line=end,
        )
    except ValidationError:
        # Range fields already validated. The derived Git path can still violate
        # schema rules (for example ':'); close only this anchor.
        return _anchor_fail_closed("path-unavailable")


def _extract_prioritization_facts(
    doc: schema.ImportDocument,
    mirror_repo: Path,
    head_sha: str,
    candidate_ids: set[str],
) -> schema.PrioritizationFacts:
    """Compare strict authoring anchors with the pinned head, in authoring coordinates."""
    candidates: dict[str, schema.PrioritizationCandidate] = {}
    non_candidates: dict[str, schema.PrioritizationCandidate] = {}
    # Whole-tree diffs are shared by every record at the same authoring commit.
    diff_cache: dict[tuple[str, str], snapshot.AnchorDiff] = {}
    for record in doc.evidence:
        anchor = (
            record.authoring_anchor.model_dump(mode="json")
            if record.authoring_anchor
            else None
        )
        relation: str = "unavailable"
        delta: str = "locationless"
        if anchor is not None and anchor.get("status") == "derived":
            # A derived anchor carries the authoring commit (schema-enforced);
            # both probes run against that commit, never a re-anchored field.
            delta = "unavailable"
            try:
                relation = snapshot.commit_relation(
                    mirror_repo, head_sha, anchor["commit_id"]
                )
                delta = snapshot.anchor_delta(
                    mirror_repo, anchor["commit_id"], head_sha, anchor,
                    diff_cache=diff_cache,
                )
            except git_ops.GitError:
                pass
        entry = schema.PrioritizationCandidate(
            commit_relation=relation,  # type: ignore[arg-type]
            anchor_delta=delta,  # type: ignore[arg-type]
        )
        (candidates if record.source_id in candidate_ids else non_candidates)[
            record.source_id
        ] = entry
    return schema.PrioritizationFacts(
        extraction_version=EXTRACTION_VERSION,
        head_sha=head_sha,
        candidates=candidates,
        non_candidates=non_candidates,
    )


def _reuse_prior_facts(
    prior: dict[str, Any] | None,
    head_sha: str,
    changed_ids: set[int] | None,
    candidate_ids: set[str],
    evidence_ids: set[str],
) -> schema.PrioritizationFacts | None:
    """Reuse valid facts only when extraction version, head, evidence, and candidate split match."""
    if prior is None:
        return None
    if prior.get("extraction_version") != EXTRACTION_VERSION:
        return None
    if prior.get("head_sha") != head_sha:
        return None
    if changed_ids:
        return None
    if not isinstance(prior.get("candidates"), dict) or not isinstance(
        prior.get("non_candidates"), dict
    ):
        return None
    # Projection changes can alter membership without changing raw evidence.
    if set(prior["candidates"]) != candidate_ids:
        return None
    if set(prior["non_candidates"]) != evidence_ids - candidate_ids:
        return None
    try:
        return schema.PrioritizationFacts.model_validate(prior)
    except ValidationError:
        return None


def _derive_authoring_anchors(
    doc: schema.ImportDocument,
    mirror_repo: Path,
    head_sha: str,
    changed_ids: set[int] | None = None,
) -> None:
    """Derive root inline anchors before projection when the freeze mirror is available."""
    for record in doc.evidence:
        if record.kind != "inline_comment" or record.reply_to_id is not None:
            continue
        if record.authoring_anchor is None or (
            changed_ids is not None and record.database_id in changed_ids
        ):
            record.authoring_anchor = _derive_one_anchor(record, mirror_repo, head_sha)

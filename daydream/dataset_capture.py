"""Serialize exposed run evidence directly from the joined immutable boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from daydream import git_ops
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.provenance import capture_executable_provenance
from daydream.deep.diff import _diff_changed_files
from daydream.deep.records import item_source_uids
from daydream.observability.runtime import current_run_trace_id
from daydream.pr_review import compute_fingerprint, extract_item_fields, parsed_issues_from_items
from daydream.timeutil import now_iso_utc

if TYPE_CHECKING:
    from daydream.artifact_visibility import ArtifactSession, ArtifactTreeSnapshot
    from daydream.run_artifacts import _RunArtifacts
    from daydream.run_config import RunConfig
    from daydream.run_snapshot import ManifestRunIdentity
    from daydream.trajectory import RunWriteSnapshot
    from daydream.workspace import WorkContext


def retain_original_task(
    run_artifacts: _RunArtifacts, *, work: WorkContext, config: RunConfig,
    analyzed_revision: dict[str, Any], diff: str,
) -> None:
    """Retain producer endpoints and complete input before any model work."""
    try:
        remote = git_ops.remote_url(work.source)
    except git_ops.GitError:
        remote = None
    slug, canonical = normalize_remote_url(remote) if remote else (None, None)
    captured = work.captured_source
    local = captured is not None and captured.source_id is not None
    dirty_tracked: bool | None = None
    if captured is None:
        try:
            dirty_tracked = git_ops.has_worktree_edits(work.repo)
        except git_ops.GitError:
            pass
    # An accepted capture has its own index. Live checkout status after that
    # boundary cannot attest whether accepted working bytes were dirty.
    run_artifacts.capture.original_task = {
        "analyzed_revision": analyzed_revision,
        "diff": diff,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "repository": {"repo_slug": slug, "remote_url": canonical,
                       "host": urlparse(canonical).hostname if canonical else None},
        "pr": {"number": config.pr_number if work.review_pr is None else work.review_pr.number,
               "repo": config.pr_repo if work.review_pr is None else f'{work.review_pr.owner}/{work.review_pr.repo}'},
        "changed_files": _diff_changed_files(diff),
        "input_scope": ("captured_working" if local else "committed" if captured is not None
                        or config.findings_out is not None and config.output_mode != "diagram"
                        else "committed_and_tracked_worktree"),
        "dirty_tracked": dirty_tracked,
        "untracked_files_included": local,
    }


def collect_verification(deep: Path, read_json: Callable[[Path], Any]) -> dict[str, Any]:
    """Read existing verification artifacts; callers own acquisition diagnostics.

    This never executes checks or refreshes remote CI. Model verdicts remain raw
    claims, distinct from the host execution attempts and their bound identities.
    """
    verification_names = {"recommendation-verdicts.json", "fix-outcomes.json", "fix-failures.json",
                          "fix-quality-gate.json", "test-verdict.json", "recommended-capture.json",
                          "fix-footprint.json", "push-verdict.json", "remote-ci-verdict.json"}
    return {path.name: payload for path in sorted(deep.glob("*.json"))
            if path.name in verification_names and (payload := read_json(path)) is not None}


def capture_run_record(
    *, artifacts: ArtifactTreeSnapshot, artifact_session: ArtifactSession, selected: RunWriteSnapshot | None,
    original_task: dict[str, Any] | None, identity: ManifestRunIdentity | None,
    config: RunConfig, work: WorkContext, outcome: Literal["success", "failed", "interrupted"],
    forbidden_store_roots: tuple[Path, ...] = (),
) -> None:
    """Commit one raw run, without archive reconstruction or trace readback.

    Evidence documents stream into the store one checked read at a time, as their exact OTLP/JSON lines."""
    from daydream.dataset import RUN_SCHEMA_VERSION, LocalRecordStore, portable_evidence_document

    store_path = config.dataset_store_path or Path.home() / ".daydream" / "dataset"
    resolved_store = store_path.resolve()
    for owned_root in (work.source, work.repo, artifacts.root.parent, *forbidden_store_roots):
        resolved_owned = owned_root.resolve()
        if resolved_store.is_relative_to(resolved_owned) or resolved_owned.is_relative_to(resolved_store):
            raise ValueError("dataset store overlaps a run-owned namespace")

    root = artifacts.root / ".daydream"
    deep = root / "deep"
    review_sources = selected is not None and identity is not None and identity.phases.merge
    acquisition_failures: dict[str, Any] = {}

    def read_json(path: Path) -> Any:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            acquisition_failures[path.name] = {"status": "failed", "reason": type(exc).__name__}
            return None
        try:
            return json.loads(text)
        except ValueError:
            acquisition_failures[path.name] = {"status": "failed", "reason": "malformed_json", "raw_text": text}
            return None

    merged_path = deep / "merged-items.json"
    merged = read_json(merged_path) if review_sources and merged_path.is_file() else None
    items = merged.get("items") if isinstance(merged, dict) else None
    if merged is not None and not isinstance(items, list):
        acquisition_failures[merged_path.name] = {"status": "failed", "reason": "malformed_shape", "raw_json": merged}
        merged = None
    items = items if isinstance(items, list) else []
    canonical_items = []
    for item in items:
        fields = extract_item_fields(item) if isinstance(item, dict) else None
        if fields is None or not isinstance(item.get("item_uid"), str) or not item["item_uid"]:
            acquisition_failures[merged_path.name] = {
                "status": "failed", "reason": "malformed_shape", "raw_json": merged,
            }
            continue
        parsed_finding = parsed_issues_from_items([item])[0]
        canonical_items.append({**item, "body": parsed_finding.body,
                                "source_uids": item_source_uids(item), "fingerprint": compute_fingerprint(
            fields.path, fields.description, fields.rationale,
        )})
    coverage_path = deep / "review-coverage.json"
    coverage = read_json(coverage_path) if review_sources and coverage_path.is_file() else None
    verification = collect_verification(deep, read_json) if review_sources else {}
    patch_path = root / "recommended.patch"
    patch = patch_path.read_text(encoding="utf-8") if review_sources and patch_path.is_file() else None
    from daydream.artifact_visibility import ArtifactVisibilityError

    try:
        code_workspace = artifact_session.code_workspace
        final_head = (work.head_sha if work.captured_source is not None else
                      code_workspace.run(partial(git_ops.head_sha, code_workspace.path)))
    except (ArtifactVisibilityError, git_ops.GitError):
        final_head = None
    provenance = {"producer": capture_executable_provenance().to_dict(),
                  "repository_context": {"branch": work.head_branch, "base_branch": work.base_branch,
                                         "source_path": str(work.source)},
                  "effective_configuration": None if identity is None else asdict(identity),
                  "flow": config.flow_name or ("diagram" if config.output_mode == "diagram" else "deep")}
    if identity is not None and identity.profile is not None:
        profile = identity.profile
        provenance["profile"] = {"profile_schema_version": profile.schema_version,
                                 "profile_name": profile.name, "profile_source_kind": profile.source_kind,
                                 "profile_digest": profile.digest}
    from daydream.run_config import resolved_phase_settings

    provenance["resolved_phases"] = resolved_phase_settings(config, ("review", "arbiter", "verify", "fix", "test"))
    provenance["file_configuration"] = None if config.file_config is None else asdict(config.file_config)
    provenance["collection_diagnostics"] = acquisition_failures
    def evidence(value: Any, *, failed: bool = False, absent: str = "unproduced") -> dict[str, Any]:
        return {"status": "available" if value is not None else "failed" if failed else absent,
                "value": value}

    documents = None if selected is None else (
        portable_evidence_document(document.validated(artifacts.session_id)[0],
                                   trajectory_id=document.trajectory_id, session_id=artifacts.session_id)
        for document in selected.documents
    )
    LocalRecordStore(store_path).commit_run({"schema_version": RUN_SCHEMA_VERSION,
        "run_id": artifacts.session_id, "captured_at": selected.cutoff_at if selected else now_iso_utc(),
        "trace_id": current_run_trace_id(),
        "outcome": outcome, "original_task": evidence(original_task, absent="unavailable"),
        "final_state": evidence({"head_sha": final_head, **(
            {'source_id': work.captured_source.source_id}
            if work.captured_source is not None and work.captured_source.source_id is not None else {})}),
        "recommended_patch": evidence({
            "patch": patch, "sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "capture": verification.get("recommended-capture.json"),
        } if patch is not None else None),
        "trajectories": evidence({"root_trajectory_id": selected.root_trajectory_id,
            "status": selected.status, "cutoff_at": selected.cutoff_at} if selected is not None else None),
        "findings": evidence({"items": canonical_items, "terminal_coverage": coverage}
            if merged is not None or coverage is not None else None,
            failed="merged-items.json" in acquisition_failures),
        "verification": evidence(verification or None, failed="recommendation-verdicts.json" in acquisition_failures),
        "provenance": provenance,
        "completeness": {"artifact_acquisition": "failed" if acquisition_failures else "complete",
            "trajectory": "unproduced" if selected is None else selected.status,
            "terminal_review": "unproduced" if coverage is None else "available"},
    }, documents)


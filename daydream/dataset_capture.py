"""Serialize exposed run evidence directly from the joined immutable boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from daydream import git_ops
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.provenance import capture_executable_provenance
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep.diff import _diff_changed_files
from daydream.deep.records import item_source_uids, record_issues, record_uid
from daydream.pr_review import compute_fingerprint, extract_item_fields
from daydream.timeutil import now_iso_utc
from daydream.training.reward import DEFAULT_WEIGHTS, REWARD_VERSION, ScoringInputs, score_trajectory

if TYPE_CHECKING:
    from daydream.artifact_visibility import ArtifactTreeSnapshot
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
    try:
        status = git_ops.status_porcelain(work.repo)
        dirty_tracked: bool | None = any(
            not line.startswith("??") and not line[3:].startswith(".daydream/")
            and line[3:] not in (".daydream", REVIEW_OUTPUT_FILE) for line in status.splitlines()
        )
    except git_ops.GitError:
        dirty_tracked = None
    run_artifacts.capture.original_task = {
        "analyzed_revision": analyzed_revision,
        "diff": diff,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "repository": {"repo_slug": slug, "remote_url": canonical,
                       "host": urlparse(canonical).hostname if canonical else None},
        "pr": {"number": config.pr_number, "repo": config.pr_repo},
        "changed_files": _diff_changed_files(diff),
        "input_scope": "committed" if config.findings_out is not None and config.output_mode != "diagram"
                       else "committed_and_tracked_worktree",
        "dirty_tracked": dirty_tracked,
        "untracked_files_included": False,
    }


def capture_run_record(
    *, artifacts: ArtifactTreeSnapshot, selected: RunWriteSnapshot | None,
    original_task: dict[str, Any] | None, identity: ManifestRunIdentity | None,
    config: RunConfig, work: WorkContext, outcome: Literal["success", "failed", "interrupted"],
    forbidden_store_roots: tuple[Path, ...] = (),
) -> None:
    """Commit one raw run, without archive reconstruction or trace readback."""
    from daydream.dataset import LocalRecordStore
    from daydream.training.harvest import assemble_scoring_inputs

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

    documents = [] if selected is None else [
        document.validated_payload(artifacts.session_id) for document in selected.documents
    ]
    claims = []
    for path in sorted(deep.glob("stack-*-records.json")) if review_sources else []:
        payload = read_json(path)
        if payload is None:
            continue
        records = record_issues(payload)
        if records is None or any(not isinstance(record, dict) or not record_uid(record) for record in records):
            acquisition_failures[path.name] = {"status": "failed", "reason": "malformed_shape", "raw_json": payload}
            continue
        claims.append({"stack": path.name[len("stack-"):-len("-records.json")], "records": payload})
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
        canonical_items.append({**item, "source_uids": item_source_uids(item), "fingerprint": compute_fingerprint(
            fields.path, fields.description, fields.rationale,
        )})
    derivation_names = {
        "arbiter-input.json", "suppression-input.json", "dedup-candidates.json",
        "adjudication-provenance.json", "dropped-speculative.json", "folded-structural.json",
        "alternatives.json",
    }
    derivation = {path.name: payload for path in sorted(deep.glob("*.json"))
                  if review_sources and (path.name in derivation_names or path.name.startswith("arbiter-group-"))
                  and (payload := read_json(path)) is not None}
    coverage_path = deep / "review-coverage.json"
    coverage = read_json(coverage_path) if review_sources and coverage_path.is_file() else None
    verification_names = {"recommendation-verdicts.json", "fix-outcomes.json", "fix-failures.json",
                          "fix-quality-gate.json", "test-verdict.json", "recommended-capture.json",
                          "fix-footprint.json", "push-verdict.json", "remote-ci-verdict.json"}
    verification = {path.name: payload for path in sorted(deep.glob("*.json"))
                    if review_sources and path.name in verification_names and (payload := read_json(path)) is not None}
    if verification:
        verification["item_associations"] = [
            {key: item[key] for key in ("id", "item_uid", "uid", "source_uids", "verify_decision",
                                       "verifier_verdict", "verifier_reason", "unverified_assumptions") if key in item}
            for item in canonical_items
        ]
    patch_path = root / "recommended.patch"
    patch = patch_path.read_text(encoding="utf-8") if review_sources and patch_path.is_file() else None
    scoring_inputs = assemble_scoring_inputs(root) if review_sources else ScoringInputs(None, True, None)
    review_path = next((path for path in (root / "review-output.md", deep / "review-output.md")
                        if review_sources and path.is_file()), None)
    review_text = review_path.read_text(encoding="utf-8") if review_path else None
    has_scoring = bool(claims or merged is not None or review_text is not None or scoring_inputs.verifier_verdicts
                       or not scoring_inputs.format_valid)
    scoring = capture_scoring(scoring_inputs, review_text)
    try:
        final_head = git_ops.head_sha(work.repo)
    except git_ops.GitError:
        final_head = None
    provenance = {"producer": capture_executable_provenance().to_dict(),
                  "effective_configuration": None if identity is None else asdict(identity),
                  "flow": config.flow_name or ("diagram" if config.output_mode == "diagram" else "deep"),
                  "license": {"status": "unavailable", "reason": "license evidence not acquired"}}
    from daydream.run_config import (
        _resolved_backend_name,
        _resolved_latency_profile,
        _resolved_model,
        _resolved_reasoning_effort,
    )

    provenance["resolved_phases"] = {phase: {
        "backend": _resolved_backend_name(config, phase), "model": _resolved_model(config, phase),
        "reasoning_effort": _resolved_reasoning_effort(config, phase),
    } for phase in ("intent", "wonder", "per_stack_review", "parse", "arbiter", "merge", "verify", "fix", "test")}
    provenance["latency_profile"] = asdict(_resolved_latency_profile(config))
    provenance["latency_route"] = None if config.latency_route is None else asdict(config.latency_route)
    provenance["file_configuration"] = None if config.file_config is None else asdict(config.file_config)
    routing_path = deep / "latency-routing.json"
    provenance["latency_routing"] = read_json(routing_path) if review_sources and routing_path.is_file() else None
    provenance["collection_diagnostics"] = acquisition_failures
    def evidence(value: Any, *, failed: bool = False, absent: str = "unproduced") -> dict[str, Any]:
        return {"status": "available" if value is not None else "failed" if failed else absent,
                "value": value}

    LocalRecordStore(store_path).commit_run({"schema_version": "daydream.run.v1",
        "run_id": artifacts.session_id, "captured_at": selected.cutoff_at if selected else now_iso_utc(),
        "outcome": outcome, "original_task": evidence(original_task, absent="unavailable"),
        "final_state": evidence({"head_sha": final_head}),
        "recommended_patch": evidence({
            "patch": patch, "sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "capture": verification.get("recommended-capture.json"),
        } if patch is not None else None),
        "trajectories": evidence({"documents": documents, "root_trajectory_id": selected.root_trajectory_id,
            "status": selected.status, "cutoff_at": selected.cutoff_at} if selected is not None else None),
        "findings": evidence({"claims": claims, "items": canonical_items, "derivation": derivation,
            "terminal_coverage": coverage} if claims or merged is not None or coverage is not None else None,
            failed="merged-items.json" in acquisition_failures or any(
                name.startswith("stack-") and name.endswith("-records.json") for name in acquisition_failures)),
        "verification": evidence(verification or None, failed="recommendation-verdicts.json" in acquisition_failures),
        "scoring": evidence(scoring if has_scoring else None), "provenance": provenance,
        "completeness": {"artifact_acquisition": "failed" if acquisition_failures else "complete",
            "trajectory": "unproduced" if selected is None else selected.status,
            "terminal_review": "unproduced" if coverage is None else "available"},
    })


def capture_scoring(inputs: ScoringInputs, review_text: str | None) -> dict[str, Any]:
    """Keep the producer's exact scoring length and existing intrinsic reducer."""
    return {
        "verifier_verdicts": inputs.verifier_verdicts,
        "format_valid": inputs.format_valid,
        "review_text": review_text,
        "length": inputs.length,
        "reward_policy": {"version": REWARD_VERSION, "configuration": {
            **vars(DEFAULT_WEIGHTS),
            "verdict_map": dict(DEFAULT_WEIGHTS.verdict_map),
            "fp_penalty_map": dict(DEFAULT_WEIGHTS.fp_penalty_map),
        }},
        "persisted_breakdown": score_trajectory(inputs).to_dict(),
        "posterior_cost": None,
    }

"""Typed record evidence for offline projection tests; no legacy artifact trees."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore, semantic_evidence_digest
from daydream.dataset_capture import capture_scoring
from daydream.training.corpus_projection.projector import BuildFrozenCorpusConfig
from daydream.training.reward import ScoringInputs
from tests.harness.dataset import observation, run_record

CAPTURED_AT = "2026-10-04T10:00:00Z"
VALID_AT = "2026-10-04T11:00:00Z"
OBSERVED_AT = "2026-10-04T12:00:00Z"
AS_OF = "2026-10-05T00:00:00+00:00"


def projection_run(
    run_id: str = "sess-a", *, dispositions: tuple[str, ...] = ("accepted", "rejected", "ambiguous"),
    repo_slug: str | None = "owner/repo-a", stack: str = "python", profile: str = "deep-review",
    siblings: int = 0, **overrides: Any,
) -> dict[str, Any]:
    items = [{"item_uid": f"item:{index}", "source_uids": [],
              "fingerprint": hashlib.sha256(f"{run_id}:{index}".encode()).hexdigest(),
              "body": f"{disposition} finding body"} for index, disposition in enumerate(dispositions)]
    diff = f"diff --git a/{run_id}.py b/{run_id}.py\n-pass\n+fixed\n"
    root = {"schema_version": "ATIF-v1.7", "session_id": run_id, "trajectory_id": run_id,
            "agent": {"name": "daydream", "version": "1"},
            "steps": [{"step_id": 1, "source": "agent", "message": "review"}]}
    documents = [root]
    if siblings:
        children = [{**root, "trajectory_id": f"{run_id}:fix-{index}"} for index in range(siblings)]
        # Actual capture holds standalone documents plus ordered root invocation summaries.
        root["extra"] = {"subtrajectories": [{"trajectory_id": child["trajectory_id"]} for child in children]}
        documents += children
    provenance = {"profile": {"profile_schema_version": 2, "profile_name": profile,
                              "profile_source_kind": "builtin", "profile_digest": "d" * 64}, "stack": stack}
    scoring = capture_scoring(ScoringInputs(verifier_verdicts=None, format_valid=True, length=6), "review")
    sections = dict(provenance=provenance, original_task={"status": "available", "value": {
        "analyzed_revision": {"head_sha": "2" * 40, "merge_base_sha": "1" * 40,
                              "diff_key": f"diff-{run_id}", "pr_base_sha": None},
        "diff": diff, "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "repository": {"repo_slug": repo_slug, "remote_url": None, "host": "github.com"},
        "pr": {"number": 1, "repo": repo_slug}, "changed_files": [f"{run_id}.py"], "input_scope": "committed",
        "dirty_tracked": False, "untracked_files_included": False}},
        trajectories={"status": "available", "value": {"documents": documents, "root_trajectory_id": run_id,
                                                        "status": "complete", "cutoff_at": CAPTURED_AT}},
        findings={"status": "available", "value": {"claims": [], "items": items,
                                                   "derivation": {}, "terminal_coverage": None}},
        scoring={"status": "available", "value": scoring})
    return run_record(run_id, **{**sections, **overrides})


def append_projection_evidence(
    store: LocalRecordStore, run: dict[str, Any], *,
    dispositions: tuple[str, ...] = ("accepted", "rejected", "ambiguous"),
    spdx_id: str | None = "MIT", role: str = "rater", evidence_valid_at: str = VALID_AT,
) -> None:
    """Append license and per-host-item judgments to an already committed run."""
    run_id = run["run_id"]
    if spdx_id is not None:
        license_evidence = {"spdx_id": spdx_id, "source": "fixture"}
        store.append_observation(observation(f"{run_id}:license", run_id=run_id, item_uid=None,
            semantic_evidence=license_evidence, evidence_digest=semantic_evidence_digest(license_evidence),
            payload={"type": "enrichment", "kind": "license", "evidence": {
                "status": "available", "value": license_evidence}}))
    for item, disposition in zip(run["findings"]["value"]["items"], dispositions):
        evidence = [{"reply_id": item["item_uid"], "classifier_label": disposition,
                     "valid_at": evidence_valid_at, "created_at": evidence_valid_at,
                     "body": "confirmed"}] if disposition in {"accepted", "rejected"} else []
        store.append_observation(observation(f"{run_id}:{item['item_uid']}:judgment", run_id=run_id,
            item_uid=item["item_uid"], role=role, valid_at=evidence_valid_at,
            semantic_evidence=evidence, evidence_digest=semantic_evidence_digest(evidence),
            payload={"type": "finding-judgment", "disposition": disposition, "rationale": "fixture judgment"}))


def seed_projection_store(root: Path, **kwargs: Any) -> LocalRecordStore:
    store = LocalRecordStore(root / "records")
    add_projection_run(store, **kwargs)
    return store


def add_projection_run(store: LocalRecordStore, *, spdx_id: str | None = "MIT", **kwargs: Any) -> dict[str, Any]:
    run = projection_run(**kwargs)
    store.commit_run(run)
    append_projection_evidence(store, run, spdx_id=spdx_id,
                               dispositions=kwargs.get("dispositions", ("accepted", "rejected", "ambiguous")))
    return run


def policy_file(root: Path, decisions: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "license-policy.json"
    path.write_text(json.dumps({"policy_version": "1", "spdx_decisions": decisions or {"MIT": "accepted"}}))
    return path


def projection_config(store: LocalRecordStore, root: Path, **kwargs: Any) -> BuildFrozenCorpusConfig:
    snapshot = store.select_snapshot(observed_before=AS_OF, valid_before=AS_OF)
    policy = kwargs.pop("license_policy_path", None)
    if policy is None:
        policy = policy_file(root)
    return BuildFrozenCorpusConfig(store_dir=store.root, snapshot_id=snapshot["snapshot_id"],
        out_dir=kwargs.pop("out_dir", root / "out"), license_policy_path=policy, **kwargs)

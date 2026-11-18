"""Typed record evidence for offline projection tests; no legacy artifact trees."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore, semantic_evidence_digest
from daydream.training.corpus_projection.projector import BuildFrozenCorpusConfig
from tests.harness.dataset import observation, run_record
from tests.harness.trajectory import portable_document

CAPTURED_AT = "2026-10-04T10:00:00Z"
VALID_AT = "2026-10-04T11:00:00Z"
AS_OF = "2026-10-05T00:00:00+00:00"


def projection_run(
    run_id: str = "sess-a", *, dispositions: tuple[str, ...] = ("accepted", "rejected", "ambiguous"),
    repo_slug: str | None = "owner/repo-a", profile: str = "deep-review",
    siblings: int = 0, **overrides: Any,
) -> dict[str, Any]:
    items = [{"item_uid": f"item:{index}", "source_uids": [],
              "fingerprint": hashlib.sha256(f"{run_id}:{index}".encode()).hexdigest(),
              "body": f"{disposition} finding body"} for index, disposition in enumerate(dispositions)]
    diff = f"diff --git a/{run_id}.py b/{run_id}.py\n-pass\n+fixed\n"
    steps = ({"step_id": 1, "source": "agent", "message": "review"},)
    children = [f"{run_id}:fix-{index}" for index in range(siblings)]
    # Actual capture holds standalone documents plus ordered root fork summaries.
    documents = [portable_document(
        trajectory_id, session_id=run_id, steps=steps, parent=None if trajectory_id == run_id else run_id,
        subtrajectories=tuple({"trajectory_id": child} for child in children) if trajectory_id == run_id else (),
    ) for trajectory_id in (run_id, *children)]
    provenance = {"profile": {"profile_schema_version": 2, "profile_name": profile,
                              "profile_source_kind": "builtin", "profile_digest": "d" * 64}}
    sections = dict(provenance=provenance, original_task={"status": "available", "value": {
        "analyzed_revision": {"head_sha": "2" * 40, "merge_base_sha": "1" * 40,
                              "diff_key": f"diff-{run_id}", "pr_base_sha": None},
        "diff": diff, "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "repository": {"repo_slug": repo_slug, "remote_url": None, "host": "github.com"},
        "pr": {"number": 1, "repo": repo_slug}, "changed_files": [f"{run_id}.py"], "input_scope": "committed",
        "dirty_tracked": False, "untracked_files_included": False}},
        trajectories={"status": "available", "value": {"documents": documents, "root_trajectory_id": run_id,
                                                        "status": "complete", "cutoff_at": CAPTURED_AT}},
        findings={"status": "available", "value": {"items": items, "terminal_coverage": None}})
    return run_record(run_id, **{**sections, **overrides})


def append_projection_evidence(
    store: LocalRecordStore, run: dict[str, Any], *,
    dispositions: tuple[str, ...] = ("accepted", "rejected", "ambiguous"),
) -> None:
    """Append per-host-item judgments to an already committed run."""
    run_id = run["run_id"]
    for item, disposition in zip(run["findings"]["value"]["items"], dispositions):
        evidence = [{"reply_id": item["item_uid"], "classifier_label": disposition,
                     "valid_at": VALID_AT, "created_at": VALID_AT,
                     "body": "confirmed"}] if disposition in {"accepted", "rejected"} else []
        store.append_observation(observation(f"{run_id}:{item['item_uid']}:judgment", run_id=run_id,
            item_uid=item["item_uid"], role="rater", valid_at=VALID_AT,
            semantic_evidence=evidence, evidence_digest=semantic_evidence_digest(evidence),
            payload={"type": "finding-judgment", "disposition": disposition, "rationale": "fixture judgment"}))


def seed_projection_store(root: Path, **kwargs: Any) -> LocalRecordStore:
    store = LocalRecordStore(root / "records")
    add_projection_run(store, **kwargs)
    return store


def add_projection_run(store: LocalRecordStore, **kwargs: Any) -> dict[str, Any]:
    run = projection_run(**kwargs)
    store.commit_run(run)
    append_projection_evidence(store, run,
                               dispositions=kwargs.get("dispositions", ("accepted", "rejected", "ambiguous")))
    return run


def projection_config(store: LocalRecordStore, root: Path, **kwargs: Any) -> BuildFrozenCorpusConfig:
    snapshot = store.select_snapshot(observed_before=AS_OF, valid_before=AS_OF)
    return BuildFrozenCorpusConfig(store_dir=store.root, snapshot_id=snapshot["snapshot_id"],
        out_dir=kwargs.pop("out_dir", root / "out"), **kwargs)

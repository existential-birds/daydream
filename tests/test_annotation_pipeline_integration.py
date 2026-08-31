"""AC 7 end-to-end: hydrate -> preview -> adjudication -> VM-loss resume ->
canonical harvest -> publication -> clean download -> build-v2 dry run.

Fake-Hub only (K6, mirrors test_archive_hydrate_integration.py) — no network.
"""
from __future__ import annotations

import json
from pathlib import Path

from daydream.archive import hydrate
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.cli import handle_adjudicate
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.publish import (
    publish_annotation_state,
    publish_final_annotation_bundle,
    resume_annotation_state,
)
from daydream.training.labeler_versions import ANNOTATION_SNAPSHOT_SCHEMA_VERSION
from tests.fixtures.training.build_hub_snapshot import SNAPSHOT_REVISION, build_snapshot


def test_full_annotation_pipeline_survives_vm_loss(tmp_path: Path) -> None:
    hub = build_snapshot()

    # 1. hydrate: VM-local SQLite index over the fake Hub snapshot
    stage = tmp_path / "stage"
    hydrate.run_hydrate_hub(hydrate.HydrateHubConfig(
        source_repo="org/private-ds", source_revision=SNAPSHOT_REVISION,
        destination_repo="org/private-ds", stage_dir=stage), client=hub)
    curation_id = next((stage / "curated").iterdir()).name

    # 2. semantic preview -> sessions.jsonl + preview manifest (snapshot id),
    #    read directly off the hydrated staging archive (no sessions.jsonl there)
    pin = {
        "curation_id": curation_id, "sanitized_hub_commit": SNAPSHOT_REVISION,
        "source_hub_commit": SNAPSHOT_REVISION,
        "archive_index_digest": "c" * 64,
        "evidence_observed_at": "2026-01-01T00:00:00+00:00",
        "as_of": "2026-02-01T00:00:00+00:00",
        "labeler_version": "1055-human-r1", "rubric_version": "984-adjudicate-r1",
        "classifier_version": "980-classifier-r1",
    }
    mat = tmp_path / "mat"
    result = run_materialize(stage, mat, pin=pin)
    snapshot_id = result["snapshot_id"]
    assert result["record_count"] == 3  # one unanswered finding per fixture session

    # 3. adjudication: build the queue over the materialized snapshot, label one
    #    batch, then publish the durable state (queue + ledger + observations).
    state = tmp_path / "state"
    assert handle_adjudicate([
        "build", "--index-root", str(mat), "--state-dir", str(state)]) == 0
    assert handle_adjudicate([
        "label", "--state-dir", str(state), "--batch", "1",
        "--disposition", "accepted", "--rationale", "clear maintainer approval",
        "--labeler", "alice"]) == 0
    run_preview(mat, state / "preview-ledger.json")
    publish_annotation_state(
        hub, state, manifest=mat / "preview-manifest.json", batch_complete=True)

    # 4. VM loss: fresh disk, resume must restore byte-identical state
    fresh = tmp_path / "fresh-vm"
    resumed = resume_annotation_state(
        hub, manifest=mat / "preview-manifest.json", stage_dir=fresh)
    assert resumed["observation_count"] == 1
    assert (fresh / "observations.jsonl").read_bytes() == \
        (state / "observations.jsonl").read_bytes()

    # 5. canonical harvest: drift-checked, appends label_observations exactly
    #    once per session into the hydrated stage's SQLite index
    harvest = run_canonical_harvest(
        index_root=mat, materialize_dir=mat, archive_dir=stage,
        observations_path=fresh / "observations.jsonl")
    assert harvest["appended_sessions"] == 3
    assert harvest["human_adjudicated"] == 1
    from daydream.archive.index import label_observation_history

    for session_id in ("sess-a", "sess-b", "sess-c"):
        assert len(label_observation_history(stage, session_id)) == 1

    # 6. publish the final annotation bundle (SHA256SUMS + _SUCCESS last)
    bundle_root = mat / "final-bundle"
    bundle_root.mkdir()
    (bundle_root / "annotations.jsonl").write_bytes(
        (mat / "annotations.jsonl").read_bytes())
    (bundle_root / "sessions.jsonl").write_bytes(
        (mat / "sessions.jsonl").read_bytes())
    (bundle_root / "lineage.json").write_text(json.dumps({
        "curation_id": curation_id, "sanitized_hub_commit": SNAPSHOT_REVISION,
        "snapshot_id": snapshot_id,
        "schema_version": f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}",
        "batch_fileset_digest": "b" * 64,
        "labeler_version": pin["labeler_version"],
        "rubric_version": pin["rubric_version"],
        "classifier_version": pin["classifier_version"],
        "as_of": pin["as_of"],
    }, sort_keys=True) + "\n", encoding="utf-8")
    final = publish_final_annotation_bundle(
        hub, bundle_root, manifest=mat / "preview-manifest.json", verify_download=True)
    assert final["hub_commit_sha"]

    # 7. clean download into a fresh dir verifies checksums
    clean = tmp_path / "clean-download"
    clean.mkdir()
    prefix = final["prefix"]
    for key, data in hub.files.items():
        if key.startswith(prefix):
            rel = Path(key[len(prefix):])
            rel.parent.mkdir(parents=True, exist_ok=True)
            (clean / rel).write_bytes(data)
    from daydream.training.corpus_v2.bundle import _verify_sha256sums

    _verify_sha256sums(clean, "")  # raises on any corruption

    # 8. corpus-v2 build over the curation bundle + the downloaded annotation
    #    bundle (the hydrated stage's curated/ dir is the curation bundle)
    from daydream.training.corpus_v2.projector import (
        BuildCorpusV2Config,
        run_build_corpus_v2,
    )

    summary = run_build_corpus_v2(BuildCorpusV2Config(
        out_dir=tmp_path / "corpus-out",
        bundle_dir=stage / "curated" / curation_id,
        annotation_bundle_dir=clean,
    ))
    assert (tmp_path / "corpus-out" / "_SUCCESS").is_file()
    assert summary["total"] >= 0

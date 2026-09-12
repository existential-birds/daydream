"""Published annotation compatibility with the CPU-only corpus projector.

External Hub/license services are fake. The separate publication integration
test exercises the actual root CLI and deletes all first-VM state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from daydream.archive import hydrate, license_enrich
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.cli import handle_adjudicate
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.publish import (
    publish_annotation_state,
    resume_annotation_state,
)
from tests.fixtures.training.build_hub_snapshot import build_publication_hubs


def test_full_annotation_pipeline_survives_vm_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.training.adjudication import cli as adjudication_cli

    hubs = build_publication_hubs()
    source = hubs.source
    annotations = hubs.annotations
    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: annotations)

    class ExternalLicenseResolver:
        def resolve(self, repo_slug: str, repo_commit: str | None) -> license_enrich.EnrichedEvidence:
            commit = repo_commit or "a" * 40
            return license_enrich.EnrichedEvidence("MIT", f"github:{repo_slug}@{commit}", commit)

    monkeypatch.setattr(license_enrich, "_make_license_resolver", ExternalLicenseResolver)
    monkeypatch.setenv("HF_TOKEN", "offline-fixture-token")
    monkeypatch.setenv("GITHUB_TOKEN", "offline-fixture-token")
    policy_path = hubs.policy_path

    # 1. hydrate: VM-local SQLite index over the fake Hub snapshot
    stage = tmp_path / "stage"
    hydrated = hydrate.run_hydrate_hub(hydrate.HydrateHubConfig(
        source_repo=source.repo_id, source_revision=hubs.source_revision,
        destination_repo=source.repo_id, stage_dir=stage,
        license_policy_path=str(policy_path)), client=source)
    curation_id = hydrated.curation_id

    # 2. semantic preview -> sessions.jsonl + preview manifest (snapshot id),
    #    read directly off the hydrated staging archive (no sessions.jsonl there)
    pin = {
        "curation_id": curation_id, "sanitized_hub_commit": hubs.source_revision,
        "source_hub_commit": hubs.source_revision,
        "archive_index_digest": hashlib.sha256((stage / "index.db").read_bytes()).hexdigest(),
        "evidence_observed_at": "2026-01-01T00:00:00+00:00",
        "as_of": "2026-02-01T00:00:00+00:00",
        "labeler_version": "1055-human-r1", "rubric_version": "984-adjudicate-r1",
        "classifier_version": "980-classifier-r1",
    }
    mat = tmp_path / "mat"
    result = run_materialize(stage, mat, pin=pin)
    snapshot_id = result["snapshot_id"]
    assert result["record_count"] == 3  # one finding per fixture session

    # 3. adjudication: build the queue over the materialized snapshot (only
    #    the unresolved item remains — the automatic decisive findings are
    #    already adjudicated), label it, then publish the durable state.
    state = tmp_path / "state"
    assert handle_adjudicate([
        "build", "--index-root", str(mat), "--state-dir", str(state)]) == 0
    assert handle_adjudicate([
        "label", "--state-dir", str(state), "--batch", "1",
        "--disposition", "accepted", "--rationale", "clear maintainer approval",
        "--labeler", "alice"]) == 0
    run_preview(mat, state / "preview-ledger.json")
    publish_annotation_state(annotations, state, manifest=mat / "preview-manifest.json")

    # 4. VM loss: fresh disk, resume must restore byte-identical state
    fresh = tmp_path / "fresh-vm"
    resumed = resume_annotation_state(
        annotations, curation_id=curation_id, expected_snapshot_id=snapshot_id, destination=fresh)
    assert resumed["observation_count"] == 1
    assert (fresh / "observations.jsonl").read_bytes() == \
        (state / "observations.jsonl").read_bytes()

    # 5. canonical harvest: drift-checked, appends label_observations exactly
    #    once per session into the hydrated stage's SQLite index
    harvest = run_canonical_harvest(
        index_root=stage, materialize_dir=mat, archive_dir=stage,
        observations_path=fresh / "observations.jsonl")
    assert harvest["appended_sessions"] == 3
    assert harvest["human_adjudicated"] == 1
    from daydream.archive.index import label_observation_history

    for session_id in ("sess-a", "sess-b", "sess-c"):
        assert len(label_observation_history(stage, session_id)) == 1

    # 6-7. final bundle: CLI only — build + dry-run + publish (the resumed
    # state dir is the observations source the coverage report's gate reads).
    assert handle_adjudicate([
        "publish-final", "--index-root", str(stage), "--materialize-dir", str(mat),
        "--archive-dir", str(stage), "--curation-bundle-dir", str(stage / "curated" / curation_id),
        "--state-dir", str(fresh),
        "--hub-repo", annotations.repo_id, "--dry-run"]) == 0
    assert handle_adjudicate([
        "publish-final", "--index-root", str(stage), "--materialize-dir", str(mat),
        "--archive-dir", str(stage),
        "--curation-bundle-dir", str(stage / "curated" / curation_id),
        "--state-dir", str(fresh),
        "--hub-repo", annotations.repo_id]) == 0
    success_commit = annotations.commit_order[-1]
    assert len(success_commit["contains"]) == 1
    success_path = success_commit["contains"][0]
    assert success_path.endswith("/_SUCCESS")
    success = json.loads(annotations.download_file(success_path, success_commit["sha"]))

    # 7. A supported pinned download independently verifies the published tree.
    clean = tmp_path / "clean-download"
    assert handle_adjudicate([
        "download-final", "--hub-repo", annotations.repo_id,
        "--curation-id", curation_id, "--snapshot-id", success["final_snapshot_id"],
        "--revision", success_commit["sha"], "--destination", str(clean),
    ]) == 0
    from daydream.training.corpus_projection.bundle import _verify_sha256sums

    _verify_sha256sums(clean, "")  # raises on any corruption

    # 8. projection: both automatic gold classes + the human-adjudicated record.
    # The human rater's decisive label is merged into the annotation row
    # before publication, so the human-adjudicated finding is gold too
    # (decisive + evidence); task-only findings never reach corpus.jsonl —
    # the projector routes them to adjudication-report.json (D8) and
    # summary["total"] counts emitted records only.
    from daydream.training.corpus_projection.projector import (
        BuildFrozenCorpusConfig,
        build_frozen_corpus,
    )

    summary = build_frozen_corpus(BuildFrozenCorpusConfig(
        out_dir=tmp_path / "corpus-out", bundle_dir=stage / "curated" / curation_id,
        annotation_bundle_dir=clean, license_policy_path=policy_path))
    assert (tmp_path / "corpus-out" / "_SUCCESS").is_file()
    records = [json.loads(line) for line in
               (tmp_path / "corpus-out" / "corpus.jsonl").read_text().splitlines() if line]
    assert summary["total"] == 3
    assert sorted(r["tier"] for r in records) == ["gold", "gold", "gold"]
    assert {r["session_id"] for r in records} == {"sess-a", "sess-b", "sess-c"}
    # The canonical records must carry the nested profile block verbatim
    # (Req 8: profile values never dropped at the projection boundary) —
    # annotation rows are build_canonical_record output with the profile
    # nested under "profile", not flat profile_* keys.
    for record in records:
        assert record["profile"] == {
            "profile_schema_version": 2, "profile_name": "pr_review",
            "profile_source_kind": "builtin", "profile_digest": "d" * 64,
        }
        assert record["stack"] == "python"

"""AC 7 end-to-end: hydrate -> preview -> adjudication -> VM-loss resume ->
canonical harvest -> CLI-only final publish -> clean download -> build-v2.

Fake-Hub only (K6, mirrors test_archive_hydrate_integration.py) — no network.
Every post-hydration publication step goes through the supported CLI
(``handle_adjudicate``); no hand-authored bundle files anywhere (M7).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.archive import hydrate
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.cli import handle_adjudicate
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.publish import (
    publish_annotation_state,
    resume_annotation_state,
)
from tests.fixtures.training.build_snapshot_decisive import (
    SNAPSHOT_REVISION,
    build_snapshot_decisive,
)


def test_full_annotation_pipeline_survives_vm_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daydream.training.adjudication import cli as adjudication_cli

    hub = build_snapshot_decisive()
    # Route the CLI's Hub client factory at the in-memory FakeHub (the
    # documented _make_client monkeypatch seam) — the test itself stays
    # CLI-only.
    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: hub)

    # Since #1094, any non-dry publication requires a pinned license policy
    # (fail-closed). The snapshot sessions carry declared MIT evidence, so a
    # policy accepting MIT admits them all.
    policy_path = tmp_path / "license-policy.json"
    policy_path.write_text(json.dumps(
        {"policy_version": "1", "spdx_decisions": {"MIT": "accepted"}}) + "\n")

    # 1. hydrate: VM-local SQLite index over the fake Hub snapshot
    stage = tmp_path / "stage"
    hydrated = hydrate.run_hydrate_hub(hydrate.HydrateHubConfig(
        source_repo="org/private-ds", source_revision=SNAPSHOT_REVISION,
        destination_repo="org/private-ds", stage_dir=stage,
        license_policy_path=str(policy_path)), client=hub)
    curation_id = hydrated.curation_id

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
        "--hub-repo", "org/private-ds", "--dry-run"]) == 0
    assert handle_adjudicate([
        "publish-final", "--index-root", str(stage), "--materialize-dir", str(mat),
        "--archive-dir", str(stage),
        "--curation-bundle-dir", str(stage / "curated" / curation_id),
        "--state-dir", str(fresh),
        "--hub-repo", "org/private-ds"]) == 0
    final_prefix = f"annotations/{curation_id}/{snapshot_id}/final/"
    assert hub.files[f"{final_prefix}_SUCCESS"] == b""

    # 7. clean download into a fresh dir verifies checksums. Copying the
    # published files off the (fake) Hub is fixture transport — the bundle
    # itself was constructed and uploaded exclusively by the CLI above; this
    # loop only materializes the download side of the runbook's verify step.
    clean = tmp_path / "clean-download"
    clean.mkdir()
    for key, data in hub.files.items():
        if key.startswith(final_prefix):
            rel = Path(key[len(final_prefix):])
            rel.parent.mkdir(parents=True, exist_ok=True)
            (clean / rel).write_bytes(data)
    from daydream.training.corpus_v2.bundle import _verify_sha256sums

    _verify_sha256sums(clean, "")  # raises on any corruption

    # 8. corpus-v2: both automatic gold classes + the human-adjudicated record.
    # The human rater's decisive label is merged into the annotation row
    # before publication, so the human-adjudicated finding is gold too
    # (decisive + evidence); task-only findings never reach corpus.jsonl —
    # the projector routes them to adjudication-report.json (D8) and
    # summary["total"] counts emitted records only.
    from daydream.training.corpus_v2.projector import (
        BuildCorpusV2Config,
        run_build_corpus_v2,
    )

    policy_path = tmp_path / "license-policy.json"
    policy_path.write_text(json.dumps(
        {"policy_version": "1", "spdx_decisions": {"MIT": "accepted"}}) + "\n")
    summary = run_build_corpus_v2(BuildCorpusV2Config(
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


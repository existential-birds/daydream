"""Construct the complete final bundle from pipeline state, including generated lineage, without Hub access
or hand-authored lineage.
"""

import hashlib
import json
from pathlib import Path

import pytest

from daydream.archive.index import _get_connection
from daydream.archive.sanitize import _derivative_digest
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.final_bundle import (
    FINAL_IDENTITY_FILES,
    build_final_bundle,
    final_snapshot_id,
)
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.publish import FinalAnnotationBundle, publish_final_annotation_bundle
from daydream.training.corpus_projection.bundle import load_curated_bundle
from daydream.training.corpus_projection.projector import _verify_annotation_bundle
from daydream.training.labeler_versions import ANNOTATION_SNAPSHOT_SCHEMA_VERSION
from tests.fixtures.training.build_hub_snapshot import AnnotationsHub
from tests.harness.adjudication import accepted_observation, policy_binding, seed_index_dispositions
from tests.test_training_adjudication_canonical import _PIN as _CANONICAL_PIN

_SOURCE = "b" * 40
_POLICY_BINDING, _CURATION_ID = policy_binding(_SOURCE)
_PIN = {**_CANONICAL_PIN, "curation_id": _CURATION_ID, "sanitized_hub_commit": _SOURCE}


def _refresh_curation_envelope(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "_SUCCESS"}:
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  " f"{path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def seed_final_bundle_state(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """Seed one accepted, one rejected, and one unanswered finding across index, archive, and
    materialization.
    """

    root = tmp_path / "index"
    seed_index_dispositions(root)
    (root / "policy-binding.json").write_text(json.dumps(_POLICY_BINDING, sort_keys=True) + "\n", encoding="utf-8")
    (root / "curation-manifest.json").write_text(json.dumps({
                "schema_version": "1", "source_hub_commit": _SOURCE, "curation_id": _CURATION_ID,
                "sanitizer_version": "v1", "hydration_index_schema_version": "v1", "admission_policy_version": "v1",
                "publication_prefix": f"curated/{_CURATION_ID}/", "batches": [],
            }, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _refresh_curation_envelope(root)
    archive = tmp_path / "archive"
    conn = _get_connection(archive)
    for n in (1, 2, 3):
        conn.execute("INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
            f"VALUES ('s{n}', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s{n}')"
        )
    conn.commit()
    conn.close()
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    return root, mat, archive, _PIN


def test_build_final_bundle_constructs_complete_staging_dir(tmp_path: Path) -> None:
    index_root, mat, archive_dir, pin = seed_final_bundle_state(tmp_path)
    # The final coverage report requires matching human observations.

    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(accepted_observation()) + "\n", encoding="utf-8")
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=obs_path)
    out = tmp_path / "final-bundle"
    summary = build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out, observations_path=obs_path,
    )
    for name in FINAL_IDENTITY_FILES:
        assert (out / name).is_file(), name
    lineage = json.loads((out / "lineage.json").read_text())
    assert lineage["curation_id"] == pin["curation_id"]
    assert lineage["sanitized_hub_commit"] == pin["sanitized_hub_commit"]
    assert lineage["snapshot_id"]
    assert lineage["batch_fileset_digest"] == _derivative_digest(index_root)
    assert lineage["schema_version"] == f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}"
    assert lineage["as_of"] == pin["as_of"]
    assert lineage["labeler_version"] == pin["labeler_version"]
    assert lineage["rubric_version"] == pin["rubric_version"]
    assert lineage["classifier_version"] == pin["classifier_version"]
    report = json.loads((out / "coverage-report.json").read_text())
    # Demote automatic decisive labels that lack matching human observations. Only human outcomes
    # contribute to the coverage numerator and denominator.
    assert report["outcome_coverage"] == {"adjudicated": 1, "total": 1}
    assert report["unresolved"] == 0
    assert report["admission_gate"]["passes_80pct"] is True
    counts = summary["disposition_counts"]
    assert set(counts) == {"accepted", "rejected", "ambiguous", "unanswered", "missing"}


def test_final_bundle_copies_both_consumer_views_from_one_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir)
    annotation_path = mat / "annotations.jsonl"
    original = annotation_path.read_bytes()
    replaced = False
    read_bytes = Path.read_bytes

    def read_and_replace(path: Path) -> bytes:
        nonlocal replaced
        data = read_bytes(path)
        if path == annotation_path and not replaced:
            path.write_bytes(original + b"\n")
            replaced = True
        return data

    out = tmp_path / "final-bundle"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_and_replace)
        build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)

    assert replaced
    assert (out / "annotations.jsonl").read_bytes() == original
    assert (out / "sessions.jsonl").read_bytes() == original

def test_build_final_bundle_gate_fails_without_human_adjudication(tmp_path: Path) -> None:
    """Automatic decisive labels must not count as human coverage toward the 80% gate."""
    _index_root, _mat, _archive_dir, out = _built_final_bundle(tmp_path)
    report = json.loads((out / "coverage-report.json").read_text())
    assert report["outcome_coverage"] == {"adjudicated": 0, "total": 0}
    assert report["unresolved"] == 0
    assert report["admission_gate"]["passes_80pct"] is False

def test_build_final_bundle_tolerates_publish_stage_leftover(tmp_path: Path) -> None:
    index_root, mat, archive_dir, out = _built_final_bundle(tmp_path)
    original_identity = final_snapshot_id(out)
    stage = out / ".publish-stage"
    stage.mkdir()
    (stage / "annotations.jsonl").write_text("stale-stage", encoding="utf-8")
    (stage / "_SUCCESS").write_text("", encoding="utf-8")
    summary = build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)
    assert ".publish-stage" not in summary["files"]
    assert final_snapshot_id(out) == original_identity
    assert (stage / "annotations.jsonl").read_text() == "stale-stage"
    for name in FINAL_IDENTITY_FILES:
        assert (out / name).is_file(), name

@pytest.mark.parametrize("kind", ["file", "directory-symlink", "file-symlink"])
def test_legacy_publish_stage_must_be_a_real_directory(kind: str, tmp_path: Path) -> None:
    index_root, mat, archive_dir, out = _built_final_bundle(tmp_path)
    stage = out / ".publish-stage"
    outside = tmp_path / "outside"
    if kind == "file":
        stage.write_bytes(b"not a scratch directory")
    elif kind == "directory-symlink":
        outside.mkdir()
        stage.symlink_to(outside, target_is_directory=True)
    else:
        outside.write_bytes(b"private outside bytes")
        stage.symlink_to(outside)

    with pytest.raises(ValueError, match="foreign"):
        final_snapshot_id(out)
    with pytest.raises(ValueError, match="foreign content"):
        build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)

    hub = AnnotationsHub(repo_id="org/private-annotations")
    with pytest.raises(ValueError, match="exactly the seven semantic files"):
        publish_final_annotation_bundle(hub, FinalAnnotationBundle.read(out))
    assert hub.commit_order == []

def test_build_final_bundle_unpinned_as_of_emits_empty_not_none(tmp_path: Path) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    manifest_path = mat / "preview-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["as_of"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "final-bundle"
    build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)
    lineage = json.loads((out / "lineage.json").read_text())
    assert lineage["as_of"] == ""

def test_build_final_bundle_is_byte_identical_on_re_run(tmp_path: Path) -> None:
    index_root, mat, archive_dir, out_one = _built_final_bundle(tmp_path)
    out_two = tmp_path / "bundle-two"
    build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out_two)
    for name in FINAL_IDENTITY_FILES:
        assert (out_one / name).read_bytes() == (out_two / name).read_bytes(), name

def test_build_final_bundle_refuses_non_empty_out_dir(tmp_path: Path) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    out = tmp_path / "final-bundle"
    out.mkdir()
    (out / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="final-bundle"):
        build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)

def test_build_final_bundle_fails_closed_on_missing_materialized_outputs(tmp_path: Path,) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    empty = tmp_path / "empty-mat"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="annotations.jsonl"):
        build_final_bundle(
            index_root=index_root, materialize_dir=empty, archive_dir=archive_dir, out_dir=tmp_path / "out",
        )


def _built_final_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    out = tmp_path / "final-bundle"
    build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)
    return index_root, mat, archive_dir, out


def test_build_final_bundle_copies_semantically_bound_policy_and_preview(tmp_path: Path) -> None:
    index_root, _mat, _archive, out = _built_final_bundle(tmp_path)

    assert tuple(sorted(path.name for path in out.iterdir())) == tuple(sorted(FINAL_IDENTITY_FILES))
    assert (out / "preview-manifest.json").read_bytes() == (tmp_path / "mat" / "preview-manifest.json").read_bytes()
    assert (out / "policy-binding.json").read_bytes() == (index_root / "policy-binding.json").read_bytes()

@pytest.mark.parametrize(("mutation", "message"),
    [({"policy_version": "rival-v2"}, "derives curation_id"), ({"policy_digest": True}, "invalid policy_digest"),
        ({"allow_copyleft": ["owner/repo", "owner/repo"]}, "invalid allow_copyleft"),
        ({"foreign": "value"}, "exact v2 field set"),
    ],
)
def test_policy_binding_semantics_fail_even_with_regenerated_envelope(
    mutation: dict[str, object], message: str, tmp_path: Path
) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    binding = dict(_POLICY_BINDING)
    binding.update(mutation)
    (index_root / "policy-binding.json").write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_curation_envelope(index_root)
    out = tmp_path / "final-bundle"

    with pytest.raises(ValueError, match=message):
        build_final_bundle(index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out)
    assert not out.exists()

def test_policy_binding_requires_producer_canonical_bytes(tmp_path: Path) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    (index_root / "policy-binding.json").write_text(
        json.dumps(_POLICY_BINDING, sort_keys=True, separators=(",", ":")), encoding="utf-8",
    )
    _refresh_curation_envelope(index_root)

    with pytest.raises(ValueError, match="canonically encoded"):
        build_final_bundle(
            index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=tmp_path / "final-bundle",
        )

@pytest.mark.parametrize("name", FINAL_IDENTITY_FILES)
def test_complete_identity_changes_for_every_semantic_file(name: str, tmp_path: Path) -> None:
    _index_root, _mat, _archive, out = _built_final_bundle(tmp_path)
    before, before_digests = final_snapshot_id(out)

    (out / name).write_bytes((out / name).read_bytes() + b" ")
    after, after_digests = final_snapshot_id(out)

    assert after != before
    assert after_digests[name] != before_digests[name]

def test_complete_seven_file_bundle_passes_existing_public_consumer(tmp_path: Path) -> None:
    index_root, _mat, _archive, out = _built_final_bundle(tmp_path)
    sums = "".join(
        f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(FINAL_IDENTITY_FILES)
    )
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    (out / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    lineage, annotations = _verify_annotation_bundle(out, load_curated_bundle(index_root), index_root)
    assert lineage["curation_id"] == _CURATION_ID
    assert annotations == (out / "annotations.jsonl").read_bytes()

"""Final-bundle staging constructor tests (issue #1078, Task 5 / M4 core).

``build_final_bundle`` must assemble the complete publish-ready staging
directory (annotations, sessions, observation history, coverage report,
generated lineage) from pipeline state alone — no hand-authored lineage file —
without touching the Hub.
"""

import json
from pathlib import Path

from daydream.archive.sanitize import _derivative_digest
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.final_bundle import build_final_bundle
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.labeler_versions import ANNOTATION_SNAPSHOT_SCHEMA_VERSION

_PIN = {
    "curation_id": "cur-1", "sanitized_hub_commit": "a" * 40,
    "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
    "evidence_observed_at": "2026-01-01T00:00:00+00:00",
    "as_of": "2026-02-01T00:00:00+00:00",
    "labeler_version": "v1", "rubric_version": "v1", "classifier_version": "v1",
}


def seed_final_bundle_state(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """Index + archive + materialize-dir fixture whose index carries one
    ``accepted``, one ``rejected``, and one ``unanswered`` finding (the same
    seed shape as the canonical-harvest decisive fixture)."""
    from daydream.archive.index import _get_connection

    root = tmp_path / "index"
    root.mkdir(exist_ok=True)
    resolutions = [
        {
            "fingerprint": f"fp-{n}", "disposition": disposition,
            "evidence": [{"reply_id": n, "body_sha256": "abc",
                          "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }
        for n, disposition in enumerate(("accepted", "rejected", "unanswered"), start=1)
    ]
    sessions = [
        {
            "session_id": f"s{n}", "trajectory_id": f"s{n}-t", "segment_id": f"s{n}-seg",
            "resolutions": [resolution],
        }
        for n, resolution in enumerate(resolutions, start=1)
    ]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
    archive = tmp_path / "archive"
    conn = _get_connection(archive)
    for n in (1, 2, 3):
        conn.execute(
            "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
            f"VALUES ('s{n}', '2026-01-01T00:00:00+00:00', 'deep', 'archive/s{n}')"
        )
    conn.commit()
    conn.close()
    mat = tmp_path / "mat"
    run_materialize(root, mat, pin=_PIN)
    return root, mat, archive, _PIN


def test_build_final_bundle_constructs_complete_staging_dir(tmp_path: Path) -> None:
    index_root, mat, archive_dir, pin = seed_final_bundle_state(tmp_path)
    # The human adjudication state the final bundle's report must see: alice
    # accepts the s1 finding (the same shape the CLI `label` verb records).
    from daydream.training.corpus_v2.identity import record_id

    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps({
        "record_id": record_id("s1", "s1-t", "s1-seg", "fp-1"),
        "disposition": "accepted",
        "evidence_digest": "d" * 32,
        "evidence": [{"reply_id": 1, "body_sha256": "abc",
                       "created_at": "2026-01-01T00:00:00+00:00"}],
        "labeler": "alice", "role": "rater",
        "rationale": "clear maintainer approval",
        "valid_at": "2026-02-02T00:00:00+00:00",
        "observed_at": "2026-02-02T00:00:00+00:00",
        "rubric_version": "v1",
    }) + "\n", encoding="utf-8")
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=obs_path)
    out = tmp_path / "final-bundle"
    summary = build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out,
        observations_path=obs_path,
    )
    for name in ("annotations.jsonl", "sessions.jsonl", "label-observations.jsonl",
                 "coverage-report.json", "lineage.json"):
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
    # The admission gate counts human-adjudicated outcome-bearing records only:
    # alice's accepted finding is adjudicated; the automatic decisive records
    # carry no human observation, so gold-eligibility demotion keeps them out
    # of the outcome-bearing numerator/denominator (issue #336 findings 1-2).
    assert report["outcome_coverage"] == {"adjudicated": 1, "total": 1}
    assert report["unresolved"] == 0
    assert report["admission_gate"]["passes_80pct"] is True
    counts = summary["disposition_counts"]
    assert set(counts) == {"accepted", "rejected", "ambiguous", "unanswered", "missing"}


def test_build_final_bundle_gate_fails_without_human_adjudication(tmp_path: Path) -> None:
    """With no human observations the 80% admission gate must FAIL, not pass
    trivially on every automatic decisive record (issue #336 finding 1)."""
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    out = tmp_path / "final-bundle"
    build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out
    )
    report = json.loads((out / "coverage-report.json").read_text())
    assert report["outcome_coverage"] == {"adjudicated": 0, "total": 0}
    assert report["unresolved"] == 0
    assert report["admission_gate"]["passes_80pct"] is False


def test_build_final_bundle_tolerates_publish_stage_leftover(tmp_path: Path) -> None:
    """A real publish leaves ``.publish-stage/`` in the bundle dir; the next
    construction/dry-run over the same dir must treat it as publish scratch,
    never foreign content (issue #336 finding 5)."""
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    out = tmp_path / "final-bundle"
    build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out
    )
    stage = out / ".publish-stage"
    stage.mkdir()
    (stage / "annotations.jsonl").write_text("stale-stage", encoding="utf-8")
    (stage / "_SUCCESS").write_text("", encoding="utf-8")
    summary = build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out
    )
    assert ".publish-stage" not in summary["files"]
    for name in ("annotations.jsonl", "sessions.jsonl", "label-observations.jsonl",
                 "coverage-report.json", "lineage.json"):
        assert (out / name).is_file(), name


def test_build_final_bundle_unpinned_as_of_emits_empty_not_none(tmp_path: Path) -> None:
    """A null (unpinned) manifest ``as_of`` must serialize into lineage.json as
    the empty unpinned-edge string, never the fabricated "None" (issue #336
    finding 4)."""
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    manifest_path = mat / "preview-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["as_of"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "final-bundle"
    build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out
    )
    lineage = json.loads((out / "lineage.json").read_text())
    assert lineage["as_of"] == ""


def test_build_final_bundle_is_byte_identical_on_re_run(tmp_path: Path) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
    out_one = tmp_path / "bundle-one"
    out_two = tmp_path / "bundle-two"
    build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out_one
    )
    build_final_bundle(
        index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out_two
    )
    for name in ("annotations.jsonl", "sessions.jsonl", "label-observations.jsonl",
                 "coverage-report.json", "lineage.json"):
        assert (out_one / name).read_bytes() == (out_two / name).read_bytes(), name


def test_build_final_bundle_refuses_non_empty_out_dir(tmp_path: Path) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    out = tmp_path / "final-bundle"
    out.mkdir()
    (out / "stale.txt").write_text("stale", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="final-bundle"):
        build_final_bundle(
            index_root=index_root, materialize_dir=mat, archive_dir=archive_dir, out_dir=out
        )


def test_build_final_bundle_fails_closed_on_missing_materialized_outputs(
    tmp_path: Path,
) -> None:
    index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
    empty = tmp_path / "empty-mat"
    empty.mkdir()
    import pytest

    with pytest.raises(FileNotFoundError, match="annotations.jsonl"):
        build_final_bundle(
            index_root=index_root, materialize_dir=empty, archive_dir=archive_dir,
            out_dir=tmp_path / "out",
        )

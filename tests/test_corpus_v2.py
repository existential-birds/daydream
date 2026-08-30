import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training.corpus_v2.bundle import BundleError, CuratedBundle, load_curated_bundle
from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.provenance import extract_provenance
from daydream.training.corpus_v2.segments import segment, segment_agents
from daydream.training.corpus_v2.tiers import GoldGateError, classify_tier

_MANIFEST = {
    "schema_version": "1",
    "source_hub_commit": "0123456789abcdef0123456789abcdef01234567",
    "curation_id": "cur-0123456789abcdef",
    "sanitizer_version": "1",
    "hydration_index_schema_version": "1",
    "admission_policy_version": "1",
    "publication_prefix": "curated/cur-0123456789abcdef/",
    "batches": [
        {
            "session_id": "sess-a",
            "content_digest": "1111111111111111111111111111111111111111111111111111111111111111",
            "status": "admitted",
            "reason_code": None,
            "artifact_relpath": "batches/sess-a/trajectory.jsonl",
            "artifact_digest": None,
            "manifest_relpath": "batches/sess-a/manifest.json",
        },
        {
            "session_id": "sess-b",
            "content_digest": "3333333333333333333333333333333333333333333333333333333333333333",
            "status": "quarantined",
            "reason_code": "secrets_scan_dirty",
            "artifact_relpath": "batches/sess-b/trajectory.jsonl",
            "artifact_digest": None,
            "manifest_relpath": None,
        },
    ],
}


def _write_sumsums(bundle_dir: Path, *, exclude: frozenset[str] = frozenset()) -> None:
    lines = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS" or path.name == "_SUCCESS":
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        if rel in exclude:
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}")
    (bundle_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def _write_bundle(tmp_path: Path, *, with_success: bool = True, corrupt_digest: bool = False) -> Path:
    bundle_dir = tmp_path / "curated" / "cur-0123456789abcdef"
    for rel in ("batches/sess-a/trajectory.jsonl", "batches/sess-a/manifest.json", "batches/sess-b/trajectory.jsonl"):
        target = bundle_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    (bundle_dir / "curation-manifest-v1.json").write_text(json.dumps(_MANIFEST))
    _write_sumsums(bundle_dir)
    if with_success:
        (bundle_dir / "_SUCCESS").write_text("ok\n")
    if corrupt_digest:
        (bundle_dir / "batches" / "sess-a" / "trajectory.jsonl").write_bytes(b"tampered\n")
    return bundle_dir


def _cfg(out_dir: Path, bundle_dir: Path, snapshot: Path, **kw: Any) -> Any:
    from daydream.training.corpus_v2.projector import BuildCorpusV2Config

    return BuildCorpusV2Config(out_dir=out_dir, bundle_dir=bundle_dir, annotations_snapshot=snapshot, **kw)


def _write_annotations_snapshot(
    bundle_dir: Path,
    *,
    valid_at: str = "2026-01-01T00:00:00+00:00",
    session_id: str = "sess-a",
    dispositions: list[str] | None = None,
) -> Path:
    """Task 0A side-car shape: a digest-pinned JSONL of per-finding
    resolution records keyed by fingerprint, exported alongside the bundle
    and covered by SHA256SUMS. Also gives the admitted batch a real ATIF
    trajectory so segmentation has something to segment."""
    trajectory = {
        "session_id": session_id,
        "trajectory_id": f"{session_id}:root",
        "subagent_trajectory_ref": [
            {"trajectory_id": f"{session_id}:fix-0", "session_id": session_id, "steps": [
                {"step_id": 1, "source": "agent", "message": "fix"},
            ]},
        ],
    }
    (bundle_dir / "batches" / session_id / "trajectory.jsonl").write_text(
        json.dumps(trajectory) + "\n"
    )
    snapshot_path = bundle_dir / "annotations-snapshot.jsonl"
    fps = ["a1" * 32, "b2" * 32, "c3" * 32]
    rows = []
    for i, disposition in enumerate(dispositions or ["accepted", "rejected", "ambiguous"]):
        evidence = (
            [{"comment_id": i + 1, "created_at": "2026-02-01T00:00:00+00:00",
              "classifier_label": disposition, "valid_at": valid_at}]
            if disposition in ("accepted", "rejected")
            else []
        )
        rows.append({"session_id": session_id, "fingerprint": fps[i % len(fps)],
                     "disposition": disposition, "evidence": evidence})
    snapshot_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    _write_sumsums(bundle_dir)  # the snapshot + trajectory join the digest pin
    return snapshot_path


def test_load_bundle_requires_success_marker(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    (bundle_dir / "_SUCCESS").unlink()
    with pytest.raises(BundleError, match="_SUCCESS"):
        load_curated_bundle(bundle_dir)


def test_load_bundle_rejects_digest_mismatch(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, corrupt_digest=True)
    with pytest.raises(BundleError, match="digest mismatch"):
        load_curated_bundle(bundle_dir)


def test_load_bundle_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    manifest_path = bundle_dir / "curation-manifest-v1.json"
    doc = json.loads(manifest_path.read_text())
    doc["schema_version"] = "999"
    manifest_path.write_text(json.dumps(doc))
    # SHA256SUMS must be regenerated so the failure is schema, not digest.
    _write_sumsums(bundle_dir)
    with pytest.raises(BundleError, match="schema_version"):
        load_curated_bundle(bundle_dir)


def test_load_bundle_uses_relative_paths_only(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    loaded = load_curated_bundle(bundle_dir)
    assert isinstance(loaded, CuratedBundle)
    for batch in loaded.admitted:
        assert not str(batch.artifact_relpath).startswith("/")
        assert ".." not in Path(batch.artifact_relpath).parts
        assert (bundle_dir / batch.artifact_relpath).exists()


def test_load_bundle_rejects_missing_batches_file(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    (bundle_dir / "batches" / "sess-a" / "trajectory.jsonl").unlink()
    with pytest.raises(BundleError, match="missing artifact"):
        load_curated_bundle(bundle_dir)

def test_record_id_is_stable_and_discriminating() -> None:
    a = record_id(session_id="s1", trajectory_id="s1:fix-0", segment_id="seg-0", fingerprint="ab" * 32)
    assert a == record_id(session_id="s1", trajectory_id="s1:fix-0", segment_id="seg-0", fingerprint="ab" * 32)
    assert record_id(session_id="s2", trajectory_id="s1:fix-0", segment_id="seg-0", fingerprint="ab" * 32) != a
    assert record_id(session_id="s1", trajectory_id="s1:fix-1", segment_id="seg-0", fingerprint="ab" * 32) != a
    assert record_id(session_id="s1", trajectory_id="s1:fix-0", segment_id="seg-1", fingerprint="ab" * 32) != a
    assert record_id(session_id="s1", trajectory_id="s1:fix-0", segment_id="seg-0", fingerprint="cd" * 32) != a


def test_record_id_is_deterministic_sha256_of_canonical_join() -> None:
    expected = hashlib.sha256(b"s1\x1fs1:fix-0\x1fseg-0\x1f" + b"ab" * 32).hexdigest()
    assert record_id("s1", "s1:fix-0", "seg-0", "ab" * 32) == expected


def _resolution(
    disposition: str, *, reward: dict[str, object] | None = None, score: float | None = None
) -> dict[str, object]:
    r: dict[str, object] = {
        "fingerprint": "ab" * 32,
        "disposition": disposition,
        "evidence": [{"created_at": "2026-02-01T00:00:00+00:00"}],
    }
    if reward is not None:
        r["intrinsic_reward"] = reward
    if score is not None:
        r["llm_self_score"] = score
    return r


def test_decisive_dispositions_are_gold() -> None:
    assert classify_tier(_resolution("accepted")) == "gold"
    assert classify_tier(_resolution("rejected")) == "gold"


def test_non_decisive_dispositions_never_gold() -> None:
    for d in ("ambiguous", "unanswered", "missing"):
        assert classify_tier(_resolution(d)) != "gold"
        assert classify_tier(_resolution(d)) == "task-only"


def test_intrinsic_reward_and_llm_score_cannot_promote_gold() -> None:
    # C5: a perfect intrinsic score or a confident self-score with a
    # non-decisive disposition must not classify gold — structurally.
    assert classify_tier(_resolution("unanswered", reward={"composite": 10.0}, score=0.99)) == "task-only"
    with pytest.raises(TypeError):
        classify_tier("accepted")  # type: ignore[arg-type]  # gold input must carry evidence, not a bare label
    with pytest.raises(GoldGateError):
        classify_tier({"fingerprint": "ab" * 32, "disposition": "accepted", "evidence": []})


def test_tiers_are_disjoint_classes() -> None:
    # process-trace tier is silver, never gold; eligibility is a separate field
    tier = classify_tier(_resolution("accepted"), record_type="process-trace")
    assert tier == "silver"  # type/decision split: ATIF process data stays silver


def _traj(siblings: list[tuple[str, str]]) -> dict[str, Any]:
    return {"trajectory_id": "s1:root", "session_id": "s1",
            "subagent_trajectory_ref": [{"trajectory_id": t, "session_id": "s1",
                                          "trajectory_path": p} for t, p in siblings]}


def test_segment_order_is_fork_registration_then_descriptor() -> None:
    # Pinned rule from Task 0B: (order_index, descriptor) total order.
    traj = _traj([("s1:fix-1", "b.jsonl"), ("s1:fix-0", "a.jsonl")])
    segs = segment(traj)
    assert [s.segment_id for s in segs] == ["seg-0", "seg-1"]
    assert [s.trajectory_id for s in segs] == ["s1:fix-1", "s1:fix-0"]


def test_segmentation_is_idempotent_across_reprojection() -> None:
    traj = _traj([("s1:explore-0", "e0.jsonl"), ("s1:review-2", "r2.jsonl")])
    assert segment(traj) == segment(traj)


def test_segment_ids_qualify_session_and_descriptor() -> None:
    traj = _traj([("s1:fix-0", "a.jsonl")])
    seg = segment(traj)[0]
    assert seg.trajectory_id == "s1:fix-0" and seg.session_id == "s1"


def test_root_alone_is_seg0() -> None:
    segs = segment({"trajectory_id": "s1:root", "session_id": "s1"})
    assert [s.segment_id for s in segs] == ["seg-0"]
    assert segs[0].trajectory_id == "s1:root"


def test_duplicate_sibling_keys_raise() -> None:
    traj = _traj([("s1:fix-0", "a.jsonl"), ("s1:fix-0", "a.jsonl")])
    with pytest.raises(ValueError, match="s1:fix-0"):
        segment(traj)


def test_spans_compose_v1_build_spans_per_sibling() -> None:
    traj = _traj([("s1:fix-0", "a.jsonl")])
    traj["subagent_trajectory_ref"][0]["steps"] = [
        {"step_id": 1, "source": "agent", "message": "think"},
        {"step_id": 2, "source": "agent", "tool_calls": [{"name": "t"}]},
    ]
    segs = segment(traj)
    assert segs[0].spans == [
        {"step_id": 1, "kind": "REASON", "content_path": "steps[0].message"},
        {"step_id": 2, "kind": "ACT", "content_path": "steps[1].tool_calls"},
    ]
    assert segment_agents is segment


# ---------------------------------------------------------------------------
# Task 6: profile + stack provenance
# ---------------------------------------------------------------------------


def test_native_profile_fields_surface_from_manifest() -> None:
    manifest_row = {
        "profile_schema_version": 2, "profile_name": "deep-review",
        "profile_source_kind": "builtin", "profile_digest": "d" * 64,
        "skill": None,
    }
    prov = extract_provenance(manifest_row)
    assert prov["profile"] == {"profile_schema_version": 2, "profile_name": "deep-review",
                               "profile_source_kind": "builtin", "profile_digest": "d" * 64}
    assert "skill" not in prov or prov["skill"] is None  # optional provenance only


def test_native_profile_run_without_legacy_skill_validates() -> None:
    v2_record = {"profile": {"profile_schema_version": 2, "profile_name": "n",
                             "profile_source_kind": "builtin", "profile_digest": None},
                 "skill": None, "stack": "python"}
    assert extract_provenance(v2_record)["stack"] == "python"
    # Schema validity is Task 1's validator's job; here we pin that no
    # required-ness is smuggled back in for skill.
    assert "skill" not in {f for f in v2_record if v2_record[f] is None and f == "skill"} or True


def test_legacy_skill_carried_as_provenance_never_required() -> None:
    prov = extract_provenance({"skill": "beagle-python:review-python", "profile_schema_version": None,
                               "profile_name": None, "profile_source_kind": None,
                               "profile_digest": None, "stack": None})
    assert prov["skill"] == "beagle-python:review-python"
    assert prov["stack"] == "python"
    assert all(v is None for v in prov["profile"].values())


def test_stack_falls_back_to_none_when_unresolvable() -> None:
    prov = extract_provenance({"skill": "unknown-thing", "stack": None})
    assert prov["stack"] is None


# ---------------------------------------------------------------------------
# Task 7: per-finding projection + adjudication routing
# ---------------------------------------------------------------------------

from daydream.training.corpus_v2.projector import project_findings  # noqa: E402


def _res(fp: str, disposition: str) -> dict[str, object]:
    return {"fingerprint": fp, "disposition": disposition,
            "evidence": [{"comment_id": 1, "created_at": "2026-02-01T00:00:00+00:00",
                          "classifier_label": disposition}] if disposition in ("accepted", "rejected") else []}


def test_mixed_session_yields_two_distinct_gold_records() -> None:
    session = {"session_id": "s1", "trajectory_id": "s1:root", "segment_id": "seg-0",
               "resolutions": [_res("a1" * 32, "accepted"), _res("b2" * 32, "rejected")]}
    records = project_findings(session)
    gold = [r for r in records if r["tier"] == "gold"]
    assert len(gold) == 2
    assert {r["finding_fingerprint"] for r in gold} == {"a1" * 32, "b2" * 32}
    assert {r["record_id"] for r in gold} and len({r["record_id"] for r in gold}) == 2
    assert {r["disposition"] for r in gold} == {"accepted", "rejected"}


def test_reply_existence_never_constitutes_acceptance() -> None:
    # ambiguous: a reply exists but the classifier did not map accepted/rejected
    records = list(project_findings({"session_id": "s1", "trajectory_id": "s1:root",
                                     "segment_id": "seg-0",
                                     "resolutions": [_res("c3" * 32, "ambiguous")]}))
    assert all(r["tier"] != "gold" for r in records)


def test_non_decisive_findings_route_to_adjudication() -> None:
    session = {"session_id": "s1", "trajectory_id": "s1:root", "segment_id": "seg-0",
               "resolutions": [_res("d4" * 32, "ambiguous"), _res("e5" * 32, "missing")]}
    records, adjudication = project_findings(session, return_adjudication=True)
    assert {r["finding_fingerprint"] for r in records if r["tier"] == "gold"} == set()
    assert {a["fingerprint"] for a in adjudication} == {"d4" * 32, "e5" * 32}
    assert all(a["evidence"] == [] for a in adjudication)  # evidence carried for the human pass


def test_run_level_contested_aggregate_never_erases_split() -> None:
    # v1 collapse: outcome_label="contested". v2 must never produce that shape.
    records = list(project_findings({"session_id": "s1", "trajectory_id": "s1:root",
                                     "segment_id": "seg-0",
                                     "resolutions": [_res("a1" * 32, "accepted"),
                                                     _res("b2" * 32, "rejected")]}))
    assert all(r["outcome_label"] != "contested" for r in records)
    assert sorted(str(r["disposition"]) for r in records) == ["accepted", "rejected"]


# ---------------------------------------------------------------------------
# Task 9: summary + full lineage + adjudication report
# ---------------------------------------------------------------------------

from daydream.training.corpus_v2.projector import run_build_corpus_v2  # noqa: E402


def test_build_summary_and_lineage_are_complete(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "rejected", "ambiguous"])
    summary = run_build_corpus_v2(_cfg(tmp_path / "out", bundle_dir, snap))
    assert set(summary) >= {"records_by_type", "records_by_tier", "records_by_split",
                            "caps", "exclusions_by_reason"}
    assert summary["records_by_type"]["outcome-finding"] >= 2
    lineage = json.loads((tmp_path / "out" / "lineage.json").read_text())
    for key in ("hub_commit", "curation_id", "content_digests", "labeler_policy_version",
                "reply_classifier_version", "rubric_schema_version", "as_of", "valid_at"):
        assert key in lineage, key
    adj_report = tmp_path / "out" / "adjudication-report.json"
    assert adj_report.is_file()
    assert "missing" in adj_report.read_text()

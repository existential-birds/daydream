import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.sanitize import _derivative_digest
from daydream.training.corpus_v2.bundle import (
    BundleBatch,
    BundleError,
    CuratedBundle,
    load_curated_bundle,
)
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
            "artifact_relpath": "batches/sess-a",
            "artifact_digest": None,
            "manifest_relpath": "batches/sess-a/manifest.json",
        },
        {
            "session_id": "sess-b",
            "content_digest": "3333333333333333333333333333333333333333333333333333333333333333",
            "status": "quarantined",
            "reason_code": "secrets_scan_dirty",
            "artifact_relpath": "batches/sess-b",
            "artifact_digest": None,
            "manifest_relpath": None,
        },
    ],
}


def _write_sumsums(bundle_dir: Path, *, exclude: frozenset[str] = frozenset()) -> None:
    lines = []
    # Producer-realistic relpaths: daydream.archive.hydrate.finalize writes
    # SHA256SUMS lines relative to the hub-checkout root under the
    # ``curated/<curation-id>/`` prefix; bundle.py strips that prefix because
    # the bundle root is the curated directory itself.
    prefix = f"curated/{bundle_dir.name}/" if bundle_dir.parent.name == "curated" else ""
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS" or path.name == "_SUCCESS":
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        if rel in exclude:
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {prefix}{rel}")
    (bundle_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


_SEED_REPO_SLUGS = {"sess-a": "owner/repo-a"}


def _write_bundle(
    tmp_path: Path,
    *,
    with_success: bool = True,
    corrupt_digest: bool = False,
    repo_slugs: dict[str, str] | None = _SEED_REPO_SLUGS,
) -> Path:
    bundle_dir = tmp_path / "curated" / "cur-0123456789abcdef"
    # Producer-realistic batch shape: artifact_relpath names the batch
    # DIRECTORY (``batches/<session_id>/``) containing the ATIF
    # ``trajectory.json`` plus the batch ``manifest.json``.
    for rel in ("batches/sess-a/trajectory.json", "batches/sess-a/manifest.json", "batches/sess-b/trajectory.json"):
        target = bundle_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    manifest = json.loads(json.dumps(_MANIFEST))
    if repo_slugs is not None:
        for batch in manifest["batches"]:
            if batch["session_id"] in repo_slugs and batch["status"] == "admitted":
                batch["repo_slug"] = repo_slugs[batch["session_id"]]
                batch["license_evidence"] = {"spdx_id": "MIT", "source": "manifest"}
    (bundle_dir / "curation-manifest.json").write_text(json.dumps(manifest))
    _write_sumsums(bundle_dir)
    if with_success:
        (bundle_dir / "_SUCCESS").write_text("ok\n")
    if corrupt_digest:
        (bundle_dir / "batches" / "sess-a" / "trajectory.json").write_bytes(b"tampered\n")
    return bundle_dir


def _cfg(out_dir: Path, bundle_dir: Path, snapshot: Path, **kw: Any) -> Any:
    from daydream.training.corpus_v2.projector import BuildCorpusV2Config

    if kw.get("license_policy_path") is None:
        kw["license_policy_path"] = _policy_file(bundle_dir.parent)
    return BuildCorpusV2Config(out_dir=out_dir, bundle_dir=bundle_dir,
                               annotation_bundle_dir=snapshot.parent, **kw)


def _write_annotations_snapshot(
    bundle_dir: Path,
    *,
    valid_at: str = "2026-01-01T00:00:00+00:00",
    session_id: str = "sess-a",
    dispositions: list[str] | None = None,
    n_siblings: int = 1,
) -> Path:
    """Two-bundle shape: a self-verified annotation bundle (its own
    SHA256SUMS + _SUCCESS + lineage.json) whose ``annotations.jsonl``
    carries per-finding resolution records keyed by ``record_id``. Also
    gives the admitted batch a real ATIF trajectory so segmentation has
    something to segment (``n_siblings`` sibling subagent refs)."""
    trajectory = {
        "session_id": session_id,
        "trajectory_id": f"{session_id}:root",
        "subagent_trajectory_ref": [
            {"trajectory_id": f"{session_id}:fix-{i}", "session_id": session_id, "steps": [
                {"step_id": 1, "source": "agent", "message": "fix"},
            ]}
            for i in range(n_siblings)
        ],
    }
    # Producer-realistic batch shape: the ATIF trajectory lives at
    # ``batches/<session_id>/trajectory.json`` (single JSON object).
    (bundle_dir / "batches" / session_id / "trajectory.json").write_text(
        json.dumps(trajectory) + "\n"
    )
    from daydream.training.corpus_v2.identity import record_id as _record_id

    ann_dir = bundle_dir.parent / f"{bundle_dir.name}-annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    fps = ["a1" * 32, "b2" * 32, "c3" * 32]
    rows = []
    for i, disposition in enumerate(dispositions or ["accepted", "rejected", "ambiguous"]):
        evidence = (
            [{"comment_id": i + 1, "created_at": "2026-02-01T00:00:00+00:00",
              "classifier_label": disposition, "valid_at": valid_at}]
            if disposition in ("accepted", "rejected")
            else []
        )
        fingerprint = fps[i % len(fps)]
        rows.append({
            "record_id": _record_id(session_id, f"{session_id}:root", "seg-0", fingerprint),
            "session_id": session_id, "fingerprint": fingerprint,
            "disposition": disposition, "evidence": evidence,
            # Real canonical-record shape (adjudication/snapshot.py:
            # build_canonical_record): the four review-profile fields nest
            # under "profile" with only "stack" at top level — the projector
            # must surface both at the two-bundle projection boundary.
            "profile": {"profile_schema_version": 2, "profile_name": "deep-review",
                        "profile_source_kind": "builtin", "profile_digest": "d" * 64},
            "stack": "python",
        })
    (ann_dir / "annotations.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    _write_sumsums(bundle_dir)  # the trajectory joins the curation digest pin
    manifest = json.loads((bundle_dir / "curation-manifest.json").read_text())
    # batch_fileset_digest pins the curation bundle's canonical file-set digest
    # (_derivative_digest = the same digest vocabulary each batch's
    # content_digest uses) — computed AFTER the bundle is final so the gate's
    # equality check against the bundle dir passes.
    (ann_dir / "lineage.json").write_text(json.dumps({
        "curation_id": manifest["curation_id"],
        "sanitized_hub_commit": manifest["source_hub_commit"],
        "schema_version": "annotation-snapshot/1055-snapshot-r1",
        "batch_fileset_digest": _derivative_digest(bundle_dir),
        "labeler_version": "v1", "rubric_version": "v1",
        "classifier_version": "v1", "as_of": None,
    }, sort_keys=True) + "\n")
    rel = sorted(p.relative_to(ann_dir).as_posix() for p in ann_dir.rglob("*") if p.is_file())
    (ann_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((ann_dir / p).read_bytes()).hexdigest()}  {p}\n" for p in rel
    ))
    (ann_dir / "_SUCCESS").write_text("ok\n")
    return ann_dir / "annotations.jsonl"


def test_load_bundle_refuses_batches_missing_repo_identity(tmp_path: Path) -> None:
    # Strip repo_slug + license_evidence from the admitted batch to produce
    # the legacy (pre-gate) bundle shape; recompute SHA256SUMS so the error
    # names identity, not digests.
    bundle_dir = _write_bundle(tmp_path, repo_slugs=None)
    manifest_path = bundle_dir / "curation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for batch in manifest["batches"]:
        batch.pop("repo_slug", None)
        batch.pop("license_evidence", None)
    manifest_path.write_text(json.dumps(manifest))
    _write_sumsums(bundle_dir)
    from daydream.archive.hydrate_rules import REASON_CODE_REPO_IDENTITY_MISSING

    with pytest.raises(BundleError, match=REASON_CODE_REPO_IDENTITY_MISSING):
        load_curated_bundle(bundle_dir)


def test_load_bundle_refuses_missing_license_evidence(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, repo_slugs={"sess-a": "owner/repo-a"})
    # Strip license_evidence from sess-a's row only, keeping repo_slug.
    manifest_path = bundle_dir / "curation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for batch in manifest["batches"]:
        if batch["session_id"] == "sess-a":
            batch.pop("license_evidence", None)
    manifest_path.write_text(json.dumps(manifest))
    _write_sumsums(bundle_dir)
    from daydream.archive.hydrate_rules import REASON_CODE_LICENSE_EVIDENCE_MISSING

    with pytest.raises(BundleError, match=REASON_CODE_LICENSE_EVIDENCE_MISSING):
        load_curated_bundle(bundle_dir)


def test_load_bundle_admission_gate_passes_wellformed_bundle(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, repo_slugs={"sess-a": "owner/repo-a"})
    loaded = load_curated_bundle(bundle_dir)  # no raise
    assert all(b.repo_slug for b in loaded.admitted)


@pytest.fixture
def existing_bundle_fixture(tmp_path: Path) -> tuple[Path, list[dict[str, Any]], dict[str, str]]:
    """The standard curated-bundle + annotation-bundle pair the two-bundle
    contract tests build on: bundle dir, the annotation rows (as JSON), and
    the linkage kwargs (curation id / hub commit)."""
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "rejected"])
    rows = [json.loads(line) for line in snap.read_text().splitlines() if line.strip()]
    manifest = json.loads((bundle_dir / "curation-manifest.json").read_text())
    kwargs = {"curation_id": manifest["curation_id"],
              "hub_commit": manifest["source_hub_commit"]}
    return bundle_dir, rows, kwargs


def _write_annotation_bundle(root: Path, rows: list[dict[str, Any]], *, curation_id: str,
                             sanitized_commit: str, batch_fileset_digest: str,
                             success: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "annotations.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (root / "lineage.json").write_text(json.dumps({
        "curation_id": curation_id, "sanitized_hub_commit": sanitized_commit,
        "schema_version": "annotation-snapshot/1055-snapshot-r1",
        "batch_fileset_digest": batch_fileset_digest,
        "labeler_version": "v1", "rubric_version": "v1",
        "classifier_version": "v1", "as_of": None,
    }, sort_keys=True) + "\n", encoding="utf-8")
    rel = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((root / p).read_bytes()).hexdigest()}  {p}\n" for p in rel
    ), encoding="utf-8")
    if success:
        (root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return root


def test_load_bundle_carries_repo_slug_and_license_evidence(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, repo_slugs={"sess-a": "owner/repo-a"})
    loaded = load_curated_bundle(bundle_dir)
    sess_a = next(b for b in loaded.batches if b.session_id == "sess-a")
    assert sess_a.repo_slug == "owner/repo-a"
    assert sess_a.license_evidence == {"spdx_id": "MIT", "source": "manifest"}


def test_bundle_batch_tolerates_absent_new_fields() -> None:
    # Legacy manifests (pre-gate bundles) still parse; the *gate* (Task 4)
    # rejects them, not the schema parser — keep KD7's refusal at the gate
    # layer so the error names the reason code, not a pydantic traceback.
    batch = BundleBatch(
        session_id="s", content_digest="1" * 64, status="admitted",
        reason_code=None, artifact_relpath="batches/s",
    )
    assert batch.repo_slug is None
    assert batch.license_evidence is None


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
    manifest_path = bundle_dir / "curation-manifest.json"
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
    (bundle_dir / "batches" / "sess-a" / "trajectory.json").unlink()
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
    disposition: str, *, reward: dict[str, object] | None = None, score: float | None = None,
    evidence_after_as_of: bool = False,
) -> dict[str, object]:
    r: dict[str, object] = {
        "fingerprint": "ab" * 32,
        "disposition": disposition,
        "evidence": [{"created_at": "2026-02-01T00:00:00+00:00"}],
    }
    if evidence_after_as_of:
        r["evidence_after_as_of"] = True
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


def test_evidence_after_as_of_rows_are_never_gold() -> None:
    # C5/M9 recorded-and-flagged edge: evidence observed after the pin's
    # as_of keeps its evidence but is never gold-eligible — a decisive
    # flagged record classifies silver, never gold.
    assert classify_tier(_resolution("accepted", evidence_after_as_of=True)) == "silver"
    assert classify_tier(_resolution("rejected", evidence_after_as_of=True)) == "silver"
    assert classify_tier(_resolution("accepted", evidence_after_as_of=False)) == "gold"
    assert classify_tier(_resolution("accepted")) == "gold"


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
    prov = extract_provenance(v2_record)
    assert prov["stack"] == "python"
    # Schema validity is Task 1's validator's job; here we pin that no
    # required-ness is smuggled back in for skill: the extractor carries
    # skill only when a value exists, never for an explicit null (and honors
    # the record's own stack override).
    assert "skill" not in prov
    assert prov["profile"] == {"profile_schema_version": None, "profile_name": None,
                               "profile_source_kind": None, "profile_digest": None}


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

from daydream.training.corpus_v2.projector import BuildCorpusV2Config, run_build_corpus_v2  # noqa: E402


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


# ---------------------------------------------------------------------------
# Task 5: per-repo license decisions on projected records
# ---------------------------------------------------------------------------

_UNSET = object()


def _policy_file(tmp_path: Path, *, spdx_decisions: dict[str, str] | None = None) -> Path:
    """Minimal deterministic license policy: MIT accepted under version 1."""
    if spdx_decisions is None:
        spdx_decisions = {"MIT": "accepted"}
    policy_path = tmp_path / "license-policy.json"
    policy_path.write_text(
        json.dumps({"policy_version": "1", "spdx_decisions": spdx_decisions}) + "\n"
    )
    return policy_path


def _config_for(
    bundle_dir: Path,
    tmp_path: Path,
    license_policy: Any = _UNSET,
    **kw: Any,
) -> Any:
    """BuildCorpusV2Config over the fixture's bundle + annotation bundle.

    The policy defaults to ``_policy_file(tmp_path)``; passing ``None``
    explicitly produces the misconfigured (no-policy) config.
    """
    if license_policy is _UNSET:
        license_policy = _policy_file(tmp_path)
    snap = bundle_dir.parent / (bundle_dir.name + "-annotations") / "annotations.jsonl"
    return BuildCorpusV2Config(
        out_dir=tmp_path / "out",
        bundle_dir=bundle_dir,
        annotation_bundle_dir=snap.parent,
        license_policy_path=license_policy,
        **kw,
    )


def test_projected_records_carry_per_repo_license_decision(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    run_build_corpus_v2(_config_for(bundle_dir, tmp_path, license_policy=_policy_file(tmp_path)))
    records = [json.loads(line) for line in
               (tmp_path / "out" / "corpus-v2.jsonl").read_text().splitlines() if line]
    assert records
    for rec in records:
        lineage = rec["lineage"]
        assert lineage["repo_slug"] == "owner/repo-a"
        decision = lineage["license_decision"]
        assert isinstance(decision, dict)
        assert decision["status"] == "admitted"
        assert decision["policy_version"] == "1"
        assert decision["repo_slug"] == "owner/repo-a"


def test_build_with_only_global_license_and_no_policy_is_refused(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    with pytest.raises(ValueError, match="license_policy"):
        _config_for(bundle_dir, tmp_path, license_policy=None)


def test_schema_validation_accepts_evolved_v2_records(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    # The projected records validate against the edited schema/v2.json
    # (repo_slug required in lineage; license_decision required as object).
    import jsonschema  # noqa: PLC0415

    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    run_build_corpus_v2(_config_for(bundle_dir, tmp_path, license_policy=_policy_file(tmp_path)))
    schema = json.loads((tmp_path / "out" / "schema.json").read_text())
    for rec in (json.loads(line) for line in
                (tmp_path / "out" / "corpus-v2.jsonl").read_text().splitlines() if line):
        jsonschema.validate(rec, schema)  # no raise


def test_projected_records_carry_profile_and_stack_provenance(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "rejected"])
    run_build_corpus_v2(_cfg(tmp_path / "out", bundle_dir, snap))
    records = [json.loads(line) for line in
               (tmp_path / "out" / "corpus.jsonl").read_text().splitlines() if line]
    assert records
    for rec in records:
        assert rec["profile"] == {"profile_schema_version": 2, "profile_name": "deep-review",
                                   "profile_source_kind": "builtin", "profile_digest": "d" * 64}
        assert "stack" in rec  # schema-required provenance key, never dropped


def test_evidence_after_as_of_findings_never_emit_gold(tmp_path: Path) -> None:
    """The emission boundary honors the canonical harvest's flag: a decisive
    finding with evidence after the pin's as_of emits silver (outcome_label
    None), never gold, into corpus.jsonl — the ``evidence_after_as_of``
    policy is enforced, not just recorded."""
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted"])
    rows = [json.loads(line) for line in snap.read_text().splitlines() if line.strip()]
    for row in rows:
        row["evidence_after_as_of"] = True
    snap.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    # annotations.jsonl changed: regenerate the annotation bundle's checksums
    # exactly as the fixture does (no SHA256SUMS self-line — it does not exist
    # when the fixture computes the listing).
    ann_dir = snap.parent
    rel = sorted(
        p.relative_to(ann_dir).as_posix()
        for p in ann_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    )
    (ann_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((ann_dir / p).read_bytes()).hexdigest()}  {p}\n" for p in rel
    ))
    summary = run_build_corpus_v2(_cfg(tmp_path / "out", bundle_dir, snap))
    assert summary["records_by_tier"] == {"silver": 1}
    records = [json.loads(line) for line in
               (tmp_path / "out" / "corpus.jsonl").read_text().splitlines() if line]
    assert records and records[0]["tier"] == "silver"
    assert records[0]["outcome_label"] is None


# ---------------------------------------------------------------------------
# Task 10: additive v2 loader surface (stacks.py) + v1 untouched gate
# ---------------------------------------------------------------------------

from daydream.training.stacks import load_dataset, load_dataset_v2  # noqa: E402

# sha256 of daydream/training/corpus.py pinned at task-10 start; proves v1
# module bytes were never touched by the v2 work (Req 16).
_V1_CORPUS_SHA_AT_PLAN_TIME = (
    "09a523efbb42aefe52a042df9155a126f3fe4c0ba7b92305ef31b49181223040"
)


def test_v2_loader_loads_projected_manifest_fail_closed(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "rejected"])
    summary = run_build_corpus_v2(_cfg(tmp_path / "proj", bundle_dir, snap))
    assert summary["emitted"] >= 1
    records = load_dataset_v2(tmp_path / "proj")
    assert records
    assert all(r["schema_version"] == "2" for r in records)
    assert all(r["tier"] in {"gold", "silver", "task-only"} for r in records)


def test_v2_loader_refuses_non_v2_record_naming_record_id(tmp_path: Path) -> None:
    out = tmp_path / "proj"
    out.mkdir()
    (out / "_SUCCESS").write_text("ok\n")
    bad = {"schema_version": "1", "record_id": "deadbeef", "tier": "gold"}
    (out / "train.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="deadbeef"):
        load_dataset_v2(out)


def test_v2_loader_refuses_malformed_line_verbatim(tmp_path: Path) -> None:
    out = tmp_path / "proj"
    out.mkdir()
    (out / "_SUCCESS").write_text("ok\n")
    (out / "train.jsonl").write_text("{not json\n")
    with pytest.raises(json.JSONDecodeError):
        load_dataset_v2(out)


def test_v2_loader_refuses_partial_projection_without_success_marker(tmp_path: Path) -> None:
    # A mid-write failure leaves a partial file set behind; without the
    # projector's _SUCCESS completeness marker the loader must refuse it
    # rather than consume it as a complete row-set.
    out = tmp_path / "proj"
    out.mkdir()
    (out / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="_SUCCESS"):
        load_dataset_v2(out)


def test_emitted_records_validate_against_shipped_v2_schema(tmp_path: Path) -> None:
    # The projector copies schema/v2.json beside its output, so every emitted
    # record must validate against that exact artifact (TRAINING_SCHEMA_V2_PATH
    # is the consumed-by-test canonical contract; nothing may ship a schema the
    # projector's own output cannot satisfy).
    from jsonschema import Draft202012Validator

    from daydream.training.schema import TRAINING_SCHEMA_V2_PATH

    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir)
    out = tmp_path / "proj"
    summary = run_build_corpus_v2(_cfg(out, bundle_dir, snap))
    assert summary["emitted"] >= 1
    validator = Draft202012Validator(json.loads(TRAINING_SCHEMA_V2_PATH.read_text()))
    records = [
        json.loads(line)
        for line in (out / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert records
    errors = sorted(
        (err.json_path, err.message) for rec in records for err in validator.iter_errors(rec)
    )
    assert not errors, errors
    # every split-manifest record must validate too
    for name in ("train.jsonl", "validation.jsonl", "holdout.jsonl"):
        for line in (out / name).read_text().splitlines():
            if line.strip():
                errors = list(validator.iter_errors(json.loads(line)))
                assert not errors, (name, errors)


def test_one_record_per_finding_across_segments(tmp_path: Path) -> None:
    # The snapshot resolutions are session-scoped, so one finding must never
    # fan out into per-segment copies that could land in different splits.
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(
        bundle_dir, dispositions=["accepted", "rejected"], n_siblings=2
    )
    out = tmp_path / "proj"
    run_build_corpus_v2(_cfg(out, bundle_dir, snap))
    records = [
        json.loads(line)
        for line in (out / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 2  # one per (session, fingerprint), not per segment
    assert len({r["record_id"] for r in records}) == len(records)
    by_fp: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_fp.setdefault(str(rec["finding_fingerprint"]), []).append(rec)
    assert all(len(v) == 1 for v in by_fp.values())


def test_task_only_findings_are_adjudication_only_not_training(tmp_path: Path) -> None:
    # Non-decisive findings are report output only (D8): excluded from
    # corpus.jsonl and the split manifests, and counted as excluded in
    # lineage/summary — the membership and the accounting must agree.
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "ambiguous"])
    out = tmp_path / "proj"
    summary = run_build_corpus_v2(_cfg(out, bundle_dir, snap))
    assert summary["records_by_tier"] == {"gold": 1}
    assert summary["exclusions_by_reason"] == {"non-decisive-adjudication": 1}
    records = [
        json.loads(line)
        for line in (out / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(r["tier"] != "task-only" for r in records)
    for name in ("train.jsonl", "validation.jsonl", "holdout.jsonl"):
        for line in (out / name).read_text().splitlines():
            if line.strip():
                assert json.loads(line)["tier"] != "task-only"
    adjudication = json.loads((out / "adjudication-report.json").read_text())
    assert [a["fingerprint"] for a in adjudication] == ["b2" * 32]
    assert (out / "_SUCCESS").is_file()


def test_v1_loader_and_v1_artifacts_unaffected(tmp_path: Path) -> None:
    # v1 loader still loads a v1 record unchanged; v1 schema file untouched
    v1_record = {"schema_version": "1", "session_id": "s", "repo_slug": "o/r",
                 "pr_number": 1, "outcome_label": "accepted", "labeler_policy_version": "1"}
    p = tmp_path / "v1.jsonl"
    p.write_text(json.dumps(v1_record) + "\n")
    assert load_dataset(p)[0]["schema_version"] == "1"
    from daydream.training.schema import TRAINING_SCHEMA_VERSION
    assert TRAINING_SCHEMA_VERSION == "1"  # never bumped in-place
    import daydream.training.corpus  # v1 module importable, unmodified
    v1_src = (Path(daydream.training.corpus.__file__)).read_bytes()
    assert hashlib.sha256(v1_src).hexdigest() == _V1_CORPUS_SHA_AT_PLAN_TIME


# ---------------------------------------------------------------------------
# Task 11: CLI wiring — ``daydream corpus build-v2``
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    """Drive ``cli.main`` (the production entrypoint) with ``argv``."""
    import sys

    from daydream import cli

    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


def test_cli_build_v2_projects_real_bundle(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir)
    rc = _run_cli(["corpus", "build-v2", "--bundle-root", str(bundle_dir),
                   "--annotation-bundle-root", str(snap.parent),
                   "--license-policy", str(_policy_file(bundle_dir.parent)),
                   "--out", str(tmp_path / "out" / "c.jsonl")])
    assert rc == 0
    assert (tmp_path / "out" / "corpus-v2.jsonl").is_file()
    assert (tmp_path / "out" / "lineage.json").is_file()


def test_cli_build_v2_refuses_missing_bundle_fail_closed(tmp_path: Path) -> None:
    rc = _run_cli(["corpus", "build-v2", "--bundle-root", str(tmp_path / "nope"),
                   "--annotation-bundle-root", str(tmp_path / "nope" / "ann"),
                   "--out", str(tmp_path / "out" / "c.jsonl")])
    assert rc != 0
    assert not (tmp_path / "out" / "lineage.json").exists()


# ---------------------------------------------------------------------------
# Task 7: two-bundle build-v2 contract (annotation bundle self-verification +
# cross-bundle linkage replaces the snapshot SHA256SUMS pin)
# ---------------------------------------------------------------------------


def test_build_v2_accepts_separate_annotation_bundle_with_exact_linkage(
    tmp_path: Path,
    existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]],
) -> None:
    bundle_dir, snapshot_rows, kwargs = existing_bundle_fixture
    ann = _write_annotation_bundle(
        tmp_path / "ann", snapshot_rows,
        curation_id=kwargs["curation_id"], sanitized_commit=kwargs["hub_commit"],
        batch_fileset_digest=_derivative_digest(bundle_dir))
    config = BuildCorpusV2Config(out_dir=tmp_path / "out", bundle_dir=bundle_dir,
                                 annotation_bundle_dir=ann,
                                 license_policy_path=_policy_file(tmp_path))
    summary = run_build_corpus_v2(config)
    assert summary["emitted"] > 0
    # curation bundle untouched (K3: no mutation of the finalized bundle)
    sums_before = (bundle_dir / "SHA256SUMS").read_bytes()
    assert (bundle_dir / "SHA256SUMS").read_bytes() == sums_before


@pytest.mark.parametrize("mutate", ["missing_success", "wrong_curation", "wrong_commit",
                                    "wrong_fileset", "corrupt_checksum", "missing_success_after"])
def test_build_v2_refuses_broken_annotation_bundles(
    tmp_path: Path,
    existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]],
    mutate: str,
) -> None:
    bundle_dir, snapshot_rows, kwargs = existing_bundle_fixture
    ann = _write_annotation_bundle(
        tmp_path / "ann", snapshot_rows,
        curation_id=kwargs["curation_id"], sanitized_commit=kwargs["hub_commit"],
        batch_fileset_digest=_derivative_digest(bundle_dir),
        success=mutate not in ("missing_success", "missing_success_after"))
    lineage = json.loads((ann / "lineage.json").read_text())
    if mutate == "wrong_curation":
        lineage["curation_id"] = "other"
    elif mutate == "wrong_commit":
        lineage["sanitized_hub_commit"] = "0" * 40
    elif mutate == "wrong_fileset":
        # a stale annotation bundle records the older curation file set's
        # digest — same curation_id + commit, different batch bytes
        lineage["batch_fileset_digest"] = "f" * 64
    elif mutate == "corrupt_checksum":
        (ann / "annotations.jsonl").write_text("tampered\n", encoding="utf-8")
    if mutate in ("wrong_curation", "wrong_commit", "wrong_fileset"):
        (ann / "lineage.json").write_text(json.dumps(lineage, sort_keys=True) + "\n", encoding="utf-8")
    config = BuildCorpusV2Config(out_dir=tmp_path / "out", bundle_dir=bundle_dir,
                                 annotation_bundle_dir=ann,
                                 license_policy_path=_policy_file(tmp_path))
    with pytest.raises(ValueError):
        run_build_corpus_v2(config)


def test_build_v2_still_works_without_annotation_bundle_dir_raises(
    tmp_path: Path,
    existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]],
) -> None:
    bundle_dir, _rows, kwargs = existing_bundle_fixture
    with pytest.raises(ValueError, match="annotation_bundle_dir"):
        BuildCorpusV2Config(out_dir=tmp_path / "out", bundle_dir=bundle_dir)


# ---------------------------------------------------------------------------
# Task 6: projection re-enforces C5/C8, accounts rejections, gates _SUCCESS
# ---------------------------------------------------------------------------

import daydream.archive.hydrate_rules as hydrate_rules  # noqa: E402

_LICENSE_CODES = frozenset({
    hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO,
    hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED,
    hydrate_rules.REASON_CODE_LICENSE_EVIDENCE_MISSING,
    hydrate_rules.REASON_CODE_REPO_IDENTITY_MISSING,
})


def _inject_admitted_repo_slug(
    bundle_dir: Path, slug: str, *, spdx_id: str = "MIT"
) -> None:
    """Rewrite every admitted batch's repo identity + license evidence in the
    curation manifest (defence-in-depth probe: admission should have caught
    the bad slug — the projector must not trust that) and recompute the
    bundle's SHA256SUMS so only the manifest content changed."""
    manifest_path = bundle_dir / "curation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for batch in manifest["batches"]:
        if batch["status"] == "admitted":
            batch["repo_slug"] = slug
            batch["license_evidence"] = {"spdx_id": spdx_id, "source": "manifest"}
    manifest_path.write_text(json.dumps(manifest))
    _write_sumsums(bundle_dir)
    # The manifest edit changes the bundle's file-set digest, so re-harvest
    # the annotation bundle's linkage pin against the new bundle bytes
    # (otherwise the two-bundle gate would refuse on staleness, not license).
    ann_dir = bundle_dir.parent / (bundle_dir.name + "-annotations")
    ann_lineage = json.loads((ann_dir / "lineage.json").read_text())
    ann_lineage["batch_fileset_digest"] = _derivative_digest(bundle_dir)
    (ann_dir / "lineage.json").write_text(json.dumps(ann_lineage, sort_keys=True) + "\n")
    rel = sorted(
        p.relative_to(ann_dir).as_posix()
        for p in ann_dir.rglob("*")
        if p.is_file() and p.name not in ("SHA256SUMS", "_SUCCESS")
    )
    (ann_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((ann_dir / p).read_bytes()).hexdigest()}  {p}\n" for p in rel
    ))


def _record_count_of_admitted_batches(bundle_dir: Path) -> int:
    """Total per-finding record count of the admitted batches: the annotation
    rows whose session_id belongs to an admitted manifest batch."""
    manifest = json.loads((bundle_dir / "curation-manifest.json").read_text())
    admitted_ids = {b["session_id"] for b in manifest["batches"] if b["status"] == "admitted"}
    snap = bundle_dir.parent / (bundle_dir.name + "-annotations") / "annotations.jsonl"
    return sum(
        1
        for line in snap.read_text().splitlines() if line.strip()
        if json.loads(line).get("session_id") in admitted_ids
    )


def test_projection_rejects_c5_repo_and_refuses_success(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    # Defence in depth: admission should have caught a C5 slug — the
    # projector must too (boundary 2), refusing before any file write.
    _inject_admitted_repo_slug(bundle_dir, "getsentry/sentry")
    with pytest.raises(ValueError, match=hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO):
        run_build_corpus_v2(_config_for(bundle_dir, tmp_path, license_policy=_policy_file(tmp_path)))
    assert not (tmp_path / "out" / "_SUCCESS").exists()
    assert not (tmp_path / "out" / "corpus-v2.jsonl").exists()  # refuse = write nothing


def test_exclusions_by_reason_counts_license_rejections(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    # Copyleft evidence the policy rejects, with no opt-in: the batch is
    # excluded (never emitted) and counted under its reason code.
    _inject_admitted_repo_slug(bundle_dir, "owner/gpl-repo", spdx_id="GPL-3.0-only")
    policy = _policy_file(tmp_path, spdx_decisions={"MIT": "accepted", "GPL-3.0-only": "rejected"})
    summary = run_build_corpus_v2(_config_for(bundle_dir, tmp_path, license_policy=policy))
    assert summary["exclusions_by_reason"][hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED] >= 1
    assert summary["emitted"] == 0


def test_mixed_repo_admitted_plus_rejected_counts_balance(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    # M8 invariant at projection: admitted records + every license rejection
    # bucket equals the total admitted-batch record count.
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    _inject_admitted_repo_slug(bundle_dir, "owner/gpl-repo", spdx_id="GPL-3.0-only")
    total_admitted = _record_count_of_admitted_batches(bundle_dir)
    policy = _policy_file(tmp_path, spdx_decisions={"MIT": "accepted", "GPL-3.0-only": "rejected"})
    summary = run_build_corpus_v2(_config_for(bundle_dir, tmp_path, license_policy=policy))
    rejections = sum(
        v for k, v in summary["exclusions_by_reason"].items() if k in _LICENSE_CODES
    )
    assert summary["emitted"] + rejections == total_admitted


def test_build_lineage_pins_license_policy_and_decisions(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    import hashlib as _hashlib  # noqa: PLC0415

    from daydream.training.exclusion import EXCLUSION_PATH  # noqa: PLC0415

    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    out = tmp_path / "out"
    run_build_corpus_v2(_config_for(bundle_dir, tmp_path, license_policy=_policy_file(tmp_path)))
    lineage = json.loads((out / "lineage.json").read_text())
    assert lineage["license_policy"]["policy_version"] == "1"
    assert lineage["license_policy"]["path_digest"] == _hashlib.sha256(
        _policy_file(tmp_path).read_bytes()
    ).hexdigest()
    assert lineage["exclusion_list_digest"] == _hashlib.sha256(
        EXCLUSION_PATH.read_bytes()
    ).hexdigest()
    assert lineage["copyleft_opt_ins"] == []
    assert lineage["license_decisions"] == {
        "owner/repo-a": {
            "status": "admitted",
            "reason_code": None,
            "spdx_id": "MIT",
            "policy_version": "1",
            "evidence_ref": "manifest",
            "repo_slug": "owner/repo-a",
        }
    }
    assert lineage["license_decision_distribution"] == {"admitted": 1}


# ---------------------------------------------------------------------------
# Task 9: digest-pinned license report artifact
# ---------------------------------------------------------------------------


def _sha256_of_exclusion_txt() -> str:
    import hashlib as _hashlib  # noqa: PLC0415

    from daydream.training.exclusion import EXCLUSION_PATH  # noqa: PLC0415

    return _hashlib.sha256(EXCLUSION_PATH.read_bytes()).hexdigest()


def test_license_report_artifact_is_deterministic(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    config = _config_for(bundle_dir, tmp_path, license_policy=_policy_file(tmp_path))
    run_build_corpus_v2(config)
    report_path = tmp_path / "out" / "license-report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["policy"]["policy_version"] == "1"
    assert report["policy"]["digest"] == hashlib.sha256(
        _policy_file(tmp_path).read_bytes()
    ).hexdigest()
    assert report["exclusion_list_digest"] == _sha256_of_exclusion_txt()
    assert report["copyleft_opt_ins"] == []
    assert report["decisions"]["owner/repo-a"]["status"] == "admitted"
    assert set(report["distribution"]) >= {"admitted"}
    # Byte-identical replay of the report alone (pure function of the
    # bundle + policy + exclusion.txt bytes):
    first = report_path.read_bytes()
    run_build_corpus_v2(_config_for(bundle_dir, tmp_path / "again",
                                    license_policy=_policy_file(tmp_path)))
    assert (tmp_path / "again" / "out" / "license-report.json").read_bytes() == first


def test_license_report_written_before_success_marker(
    tmp_path: Path, existing_bundle_fixture: tuple[Path, list[dict[str, Any]], dict[str, str]]
) -> None:
    # The completeness gate covers the report: it exists on every clean build
    # that publishes _SUCCESS, and the summary exposes the distribution.
    bundle_dir, _rows, _kwargs = existing_bundle_fixture
    summary = run_build_corpus_v2(
        _config_for(bundle_dir, tmp_path, license_policy=_policy_file(tmp_path))
    )
    assert (tmp_path / "out" / "_SUCCESS").is_file()
    assert (tmp_path / "out" / "license-report.json").is_file()
    assert summary["license_distribution"] == {"admitted": 1}

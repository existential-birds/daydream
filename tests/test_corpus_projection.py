"""Pure reductions and real CLI projection over pinned record-store evidence."""
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from daydream.dataset import LocalRecordStore, semantic_evidence_digest
from daydream.training.corpus_projection.projector import BuildFrozenCorpusConfig, build_frozen_corpus
from daydream.training.corpus_projection.provenance import extract_provenance
from daydream.training.corpus_projection.segments import segment
from daydream.training.corpus_projection.tiers import GoldGateError, classify_tier
from daydream.training.exclusion import EXCLUSION_PATH
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.record_identity import record_finding_id
from daydream.training.stacks import load_dataset_v2
from tests.harness.adjudication import reply_evidence
from tests.harness.dataset import observation
from tests.harness.record_projection import (
    add_projection_run,
    append_projection_evidence,
    policy_file,
    projection_config,
    projection_run,
    seed_projection_store,
)
from tests.harness.scripts import cli_main


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_record_identity_is_stable_and_discriminating() -> None:
    identity = record_finding_id("s1", "s1:fix-0", "seg-0", "item:1")
    assert record_finding_id("s2", "s1:fix-0", "seg-0", "item:1") != identity
    assert record_finding_id("s1", "s1:fix-1", "seg-0", "item:1") != identity
    assert record_finding_id("s1", "s1:fix-0", "seg-1", "item:1") != identity
    assert record_finding_id("s1", "s1:fix-0", "seg-0", "item:2") != identity


def test_record_identity_has_explicit_versioned_host_uid_hash() -> None:
    payload = b'["record-snapshot-v1","s1","s1:fix-0","seg-0","item:1"]'
    assert record_finding_id("s1", "s1:fix-0", "seg-0", "item:1") == hashlib.sha256(payload).hexdigest()


def _resolution(disposition: str, *, reward: dict[str, object] | None = None, score: float | None = None,
    evidence_after_as_of: bool = False,
) -> dict[str, object]:
    r: dict[str, object] = {
        "fingerprint": "ab" * 32, "disposition": disposition, "evidence": [{"created_at": "2026-02-01T00:00:00+00:00"}],
    }
    if evidence_after_as_of:
        r["evidence_after_as_of"] = True
    if reward is not None:
        r["intrinsic_reward"] = reward
    if score is not None:
        r["llm_self_score"] = score
    return r


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

def test_duplicate_sibling_keys_raise() -> None:
    traj = _traj([("s1:fix-0", "a.jsonl"), ("s1:fix-0", "a.jsonl")])
    with pytest.raises(ValueError, match="s1:fix-0"):
        segment(traj)

# Task 6: profile + stack provenance


def test_absent_explicit_stack_stays_unknown() -> None:
    prov = extract_provenance({"profile_name": "deep-review"})
    assert prov["stack"] is None

def test_cli_reply_existence_never_constitutes_acceptance(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    run = projection_run(dispositions=("ambiguous",))
    store.commit_run(run)
    append_projection_evidence(store, run, dispositions=())
    semantic, capture = reply_evidence("9", "Needs investigation.\n")
    assert semantic["classifier_label"] == "ambiguous"
    store.append_observation(observation(
        "reply", run_id="sess-a", item_uid="item:0", role="automatic",
        semantic_evidence=[semantic], evidence_digest=reply_evidence_digest([semantic]),
        evidence_digest_scheme="reply-evidence-v1", reply_captures=[capture],
        payload={"type": "finding-judgment", "disposition": "ambiguous", "rationale": "nondirectional reply"},
    ))
    config = projection_config(store, tmp_path)
    assert cli_main(_cli_args(config)) == 0
    assert _read_jsonl(config.out_dir / "corpus.jsonl") == []
    report = json.loads((config.out_dir / "adjudication-report.json").read_text())
    assert len(report) == 1
    assert report[0]["disposition"] == "ambiguous"
    assert report[0]["evidence"] == [semantic]
    assert report[0]["reply_captures"] == [capture]


def _cli_args(config: BuildFrozenCorpusConfig) -> list[str]:
    return ["corpus", "build", "--store", str(config.store_dir), "--snapshot-id", config.snapshot_id,
            "--license-policy", str(config.license_policy_path), "--out", str(config.out_dir / "corpus.jsonl")]


def test_cli_build_records_keeps_finding_population_task_reward_and_lineage(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path, siblings=3)
    config = projection_config(store, tmp_path)
    assert cli_main(_cli_args(config)) == 0
    rows = _read_jsonl(config.out_dir / "corpus.jsonl")
    assert len(rows) == 2
    assert {row["outcome_label"] for row in rows} == {"accepted", "rejected"}
    assert len({row["record_id"] for row in rows}) == 2
    assert {row["trajectory_id"] for row in rows} == {"sess-a:fix-0"}
    assert {row["task_segment"] for row in rows} == {"seg-0"}
    run = store.read_snapshot(config.snapshot_id).runs[0]
    for row in rows:
        jsonschema.validate(row, json.loads((config.out_dir / "schema.json").read_text()))
        assert row["finding_text"] == f"{row['disposition']} finding body"
        assert row["finding_text_sha256"] == hashlib.sha256(row["finding_text"].encode()).hexdigest()
        assert row["diff"] == run["original_task"]["value"]["diff"]
        assert row["task_identity"]["base_sha"] == "1" * 40
        assert row["task_identity"]["head_sha"] == "2" * 40
        assert row["task_identity"]["diff_ref"]["run_id"] == "sess-a"
        assert row["lineage"]["snapshot_id"] == config.snapshot_id
        assert row["lineage"]["source_identity_version"] == "record-snapshot-v1"
        assert row["profile"]["profile_name"] == "deep-review"
        assert row["stack"] == "python"
        assert row["intrinsic_reward"] == run["scoring"]["value"]["persisted_breakdown"]
    report = json.loads((config.out_dir / "adjudication-report.json").read_text())
    assert len(report) == 1 and report[0]["disposition"] == "ambiguous"
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert lineage["adjudication_count"] == 1
    assert lineage["exclusions_by_reason"] == {"non-decisive-adjudication": 1}
    assert config.license_policy_path is not None
    assert lineage["license_policy"]["path_digest"] == hashlib.sha256(
        config.license_policy_path.read_bytes()).hexdigest()
    assert lineage["exclusion_list_digest"] == hashlib.sha256(EXCLUSION_PATH.read_bytes()).hexdigest()
    assert lineage["snapshot"]["snapshot_id"] == config.snapshot_id
    assert len(lineage["content_digests"]) == 5
    assert (config.out_dir / "SHA256SUMS").is_file()
    assert (config.out_dir / "_SUCCESS").is_file()
    assert len(load_dataset_v2(config.out_dir)) == 2


@pytest.mark.parametrize(("slug", "spdx", "allowed", "reason"), [
    ("getsentry/sentry", "MIT", frozenset(), "c5_excluded_repo"),
    (None, "MIT", frozenset(), "repo_identity_missing"),
    ("owner/repo", None, frozenset(), "license_evidence_missing"),
    ("owner/repo", "NOASSERTION", frozenset(), "license_evidence_missing"),
    ("owner/gpl-repo", "GPL-3.0-only", frozenset(), "c8_copyleft_unopted"),
    ("owner/gpl-repo", "GPL-3.0-only", frozenset({"owner/other"}), "c8_copyleft_unopted"),
])
def test_cli_license_refusal_writes_no_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], slug: str | None, spdx: str | None,
    allowed: frozenset[str], reason: str,
) -> None:
    store = seed_projection_store(tmp_path, repo_slug=slug, spdx_id=spdx)
    policy = policy_file(tmp_path, {"MIT": "accepted", "GPL-3.0-only": "rejected"})
    config = projection_config(store, tmp_path, license_policy_path=policy, allow_copyleft=allowed)
    args = _cli_args(config)
    if allowed:
        args += ["--allow-copyleft", *allowed]
    assert cli_main(args) == 1
    captured = capsys.readouterr()
    assert reason in captured.out + captured.err
    assert not config.out_dir.exists()


def test_copyleft_opt_in_and_mixed_repo_license_decisions_are_pinned(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path)
    add_projection_run(store, run_id="sess-b", repo_slug="owner/gpl-repo", spdx_id="GPL-3.0-only")
    policy = policy_file(tmp_path, {"MIT": "accepted", "GPL-3.0-only": "rejected"})
    config = projection_config(store, tmp_path, license_policy_path=policy,
                               allow_copyleft=frozenset({"OWNER/GPL-REPO"}))
    result = build_frozen_corpus(config)
    assert result["license_distribution"] == {"admitted": 2}
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert set(lineage["license_decisions"]) == {"sess-a", "sess-b"}
    assert lineage["copyleft_opt_ins"] == ["OWNER/GPL-REPO"]
    report = json.loads((config.out_dir / "license-report.json").read_text())
    assert report["decisions"] == lineage["license_decisions"]
    assert report["distribution"] == result["license_distribution"]


def test_missing_judgments_remain_in_complete_adjudication_population(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    run = projection_run()
    store.commit_run(run)
    append_projection_evidence(store, run, dispositions=())
    result = build_frozen_corpus(projection_config(store, tmp_path))
    assert result["total"] == 0
    assert result["adjudication"] == 3
    report = json.loads((tmp_path / "out" / "adjudication-report.json").read_text())
    assert {entry["disposition"] for entry in report} == {"unanswered"}


@pytest.mark.parametrize("failure", ["snapshot", "license-policy", "trajectory"])
def test_required_frozen_inputs_fail_before_output(tmp_path: Path, failure: str) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path)
    if failure == "snapshot":
        (store.root / "snapshots" / f"{config.snapshot_id}.jsonl").unlink()
    elif failure == "license-policy":
        assert config.license_policy_path is not None
        config.license_policy_path.unlink()
    else:
        fresh = LocalRecordStore(tmp_path / "absent-trace")
        run = projection_run(trajectories={"status": "unproduced"})
        fresh.commit_run(run)
        append_projection_evidence(fresh, run)
        config = projection_config(fresh, tmp_path)
    assert cli_main(_cli_args(config)) == 1
    assert not config.out_dir.exists()


def test_v2_loader_rejects_incomplete_wrong_version_and_malformed_outputs(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path)
    build_frozen_corpus(config)
    (config.out_dir / "_SUCCESS").unlink()
    with pytest.raises(ValueError, match="_SUCCESS"):
        load_dataset_v2(config.out_dir)
    (config.out_dir / "_SUCCESS").write_text("ok\n")
    # Force a nonempty split regardless of deterministic membership.
    row = _read_jsonl(config.out_dir / "corpus.jsonl")[0]
    row["schema_version"] = "1"
    (config.out_dir / "train.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        load_dataset_v2(config.out_dir)
    (config.out_dir / "train.jsonl").write_text("not-json\n")
    with pytest.raises(json.JSONDecodeError):
        load_dataset_v2(config.out_dir)


def test_configuration_requires_policy_and_matching_temporal_pin(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path)
    with pytest.raises(ValueError, match="license_policy_path"):
        BuildFrozenCorpusConfig(out_dir=tmp_path, store_dir=store.root, snapshot_id="f" * 64)
    config = projection_config(store, tmp_path, as_of="2026-10-04T00:00:00Z")
    with pytest.raises(ValueError, match="valid_before"):
        build_frozen_corpus(config)
    assert not config.out_dir.exists()


def test_same_fingerprint_keeps_host_identity_labels_and_deterministic_membership(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    run = projection_run(dispositions=("accepted", "rejected"))
    items = run["findings"]["value"]["items"]
    items[1]["fingerprint"] = items[0]["fingerprint"]
    store.commit_run(run)
    append_projection_evidence(store, run, dispositions=("accepted", "rejected"))
    config = projection_config(store, tmp_path)
    build_frozen_corpus(config)
    rows = _read_jsonl(config.out_dir / "corpus.jsonl")
    assert len(rows) == 2
    assert len({row["finding_fingerprint"] for row in rows}) == 1
    assert {row["item_uid"] for row in rows} == {"item:0", "item:1"}
    assert {row["disposition"] for row in rows} == {"accepted", "rejected"}
    for row in rows:
        payload = json.dumps(["record-snapshot-v1", "sess-a", "sess-a", "seg-0", row["item_uid"]],
                             separators=(",", ":")).encode()
        assert row["record_id"] == hashlib.sha256(payload).hexdigest()
    first = (config.out_dir / "corpus.jsonl").read_bytes()
    build_frozen_corpus(config)
    assert (config.out_dir / "corpus.jsonl").read_bytes() == first


def test_cli_posterior_annotation_preserves_captured_native_profile_with_claim_stack(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    run = projection_run(dispositions=("accepted",))
    run["provenance"].pop("stack")
    run["findings"]["value"]["items"][0]["source_uids"] = ["claim:1"]
    run["findings"]["value"]["claims"] = [{"stack": "python", "records": [{"uid": "claim:1"}]}]
    store.commit_run(run)
    append_projection_evidence(store, run, dispositions=("accepted",))
    breakdown = run["scoring"]["value"]["persisted_breakdown"]
    annotation = {"labels": ["accepted"], "pr_state": "closed", "valid_at": "2026-10-04T11:00:00Z",
                  "reward_version": breakdown["reward_version"], "reward_json": json.dumps(breakdown),
                  "composite_reward": breakdown["composite"], "evidence_sha": "1" * 40,
                  "rubric_json": json.dumps({"posterior_source": "pr_review"}), "reviewer_logins": ["alice"],
                  "has_posterior": True, "reply_classifier_version": "980-classifier-r1", "reply_evidence_digest": None}
    semantic, capture = reply_evidence("9", "good catch\ncafé ☕\n")
    annotation["rubric_json"] = json.dumps({"posterior_source": "pr_review", "per_finding_resolutions": [{
        "fingerprint": run["findings"]["value"]["items"][0]["fingerprint"], "disposition": "accepted",
        "evidence": [semantic], "evidence_digest": reply_evidence_digest([semantic]), "reply_captures": [capture],
    }]})
    store.append_observation(observation("harvest", schema_version="daydream.observation.v2", run_id="sess-a",
        item_uid=None, role="automatic", semantic_evidence=[], evidence_digest=semantic_evidence_digest([]),
        payload={"type": "harvest-annotation", "annotation": annotation, "labeler_policy_version": "980-policy-r1"}))
    config = projection_config(store, tmp_path)
    assert cli_main(_cli_args(config)) == 0
    row = _read_jsonl(config.out_dir / "corpus.jsonl")[0]
    assert row["profile"] == run["provenance"]["profile"]
    assert row["stack"] == "python"
    assert row["annotation"] == annotation
    assert row["intrinsic_reward"] == breakdown
    assert row["reply_captures"] == [capture]
    assert row["evidence"] == [semantic]
    jsonschema.validate(row, json.loads((config.out_dir / "schema.json").read_text()))


def test_equivalent_records_preserve_normalized_pre_cutover_training_examples(tmp_path: Path) -> None:
    """Compare retained fields to frozen baseline output; source identities evolve explicitly."""
    store = LocalRecordStore(tmp_path / "records")
    run = projection_run(dispositions=("accepted", "rejected"), repo_slug="owner/repo-e808f6", siblings=1)
    task = run["original_task"]["value"]
    task["diff"] = ("diff --git a/sess-a.py b/sess-a.py\n--- a/sess-a.py\n+++ b/sess-a.py\n"
                    "@@ -1 +1 @@\n-pass\n+fixed-sess-a\n")
    task["diff_sha256"] = hashlib.sha256(task["diff"].encode()).hexdigest()
    task["analyzed_revision"].update(merge_base_sha="66cefd1a2110dbd56f08f258798f9c8ab0a4d377",
                                    head_sha="62dae747aff23cdad1bd9e0aa26b423e8b64754b")
    items = run["findings"]["value"]["items"]
    for item, fingerprint, text in zip(items, ("a1" * 32, "b2" * 32),
                                       ("exact localized finding body", "rejected finding body")):
        item.update(fingerprint=fingerprint, body=text)
    store.commit_run(run)
    license_evidence = {"spdx_id": "MIT", "source": "manifest"}
    store.append_observation(observation("license", run_id="sess-a", item_uid=None,
        semantic_evidence=license_evidence, evidence_digest=semantic_evidence_digest(license_evidence),
        payload={"type": "enrichment", "kind": "license", "evidence": {
            "status": "available", "value": license_evidence}}))
    for index, item in enumerate(items):
        disposition = ("accepted", "rejected")[index]
        evidence = [{"comment_id": index + 1, "classifier_label": disposition,
                     "created_at": "2026-02-01T00:00:00+00:00", "valid_at": "2026-01-01T00:00:00+00:00"}]
        store.append_observation(observation(f"judgment-{index}", run_id="sess-a", item_uid=item["item_uid"],
            valid_at="2026-01-01T00:00:00+00:00", semantic_evidence=evidence,
            evidence_digest=semantic_evidence_digest(evidence),
            payload={"type": "finding-judgment", "disposition": disposition, "rationale": "baseline equivalent"}))
    snapshot = store.select_snapshot(observed_before="2026-10-05T00:00:00+00:00")
    config = BuildFrozenCorpusConfig(out_dir=tmp_path / "out", store_dir=store.root,
                                    snapshot_id=snapshot["snapshot_id"], license_policy_path=policy_file(tmp_path))
    build_frozen_corpus(config)
    rows = _read_jsonl(config.out_dir / "corpus.jsonl")
    fields = ("schema_version", "record_type", "tier", "session_id", "trajectory_id", "task_segment",
              "finding_fingerprint", "disposition", "outcome_label", "evidence", "profile", "stack",
              "finding_text", "finding_text_sha256", "diff")
    normalized = [{**{name: row[name] for name in fields},
                   "lineage": {name: row["lineage"][name] for name in (
                       "as_of", "valid_at", "exclusion_reason", "repo_slug", "license_decision", "diff_digest")},
                   "task_identity": {name: row["task_identity"][name] for name in (
                       "repo_slug", "base_sha", "head_sha", "diff_digest")}} for row in rows]
    expected = json.loads((Path(__file__).parent / "fixtures/training/projection-preservation.json").read_text())
    assert sorted(normalized, key=lambda row: row["finding_fingerprint"]) == expected["records"]


@pytest.mark.parametrize("target", ["store-root", "shards", "ancestor", "output-alias", "store-alias"])
def test_cli_output_refuses_record_store_overlap_without_mutating_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], target: str,
) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path)
    input_root = store.root
    if target == "store-root":
        output_root = store.root
    elif target == "shards":
        output_root = store.root / "runs"
    elif target == "ancestor":
        output_root = store.root.parent
    elif target == "output-alias":
        alias = tmp_path / "output-alias"
        alias.symlink_to(store.root, target_is_directory=True)
        output_root = alias / "runs"
    else:
        input_root = tmp_path / "store-alias"
        input_root.symlink_to(store.root, target_is_directory=True)
        output_root = store.root
    (tmp_path / "operator-notes.txt").write_text("existing local evidence and notes must survive")
    before_files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    before_paths = set(tmp_path.rglob("*"))
    args = ["corpus", "build", "--store", str(input_root), "--snapshot-id", config.snapshot_id,
            "--license-policy", str(config.license_policy_path), "--out", str(output_root / "corpus.jsonl")]
    assert cli_main(args) == 1
    assert "overlaps the record store namespace" in capsys.readouterr().out
    after_files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after_files == before_files
    assert set(tmp_path.rglob("*")) == before_paths

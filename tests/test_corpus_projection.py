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
from daydream.training.stacks import load_dataset_v2
from tests.harness.adjudication import reply_evidence
from tests.harness.dataset import observation
from tests.harness.record_projection import (
    add_projection_run,
    append_projection_evidence,
    projection_config,
    projection_run,
    seed_projection_store,
)
from tests.harness.scripts import cli_main


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def test_gold_requires_decisive_timely_evidence() -> None:
    # C5: a perfect intrinsic score or a confident self-score with a
    # non-decisive disposition must not classify gold — structurally.
    assert classify_tier(_resolution("unanswered", reward={"composite": 10.0}, score=0.99)) == "task-only"
    with pytest.raises(TypeError):
        classify_tier("accepted")  # type: ignore[arg-type]  # gold input must carry evidence, not a bare label
    with pytest.raises(GoldGateError):
        classify_tier({"fingerprint": "ab" * 32, "disposition": "accepted", "evidence": []})
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


def test_duplicate_sibling_keys_raise() -> None:
    traj = _traj([("s1:fix-0", "a.jsonl"), ("s1:fix-0", "a.jsonl")])
    with pytest.raises(ValueError, match="s1:fix-0"):
        segment(traj)


# Task 6: profile + stack provenance


def _cli_args(config: BuildFrozenCorpusConfig) -> list[str]:
    return ["corpus", "build", "--store", str(config.store_dir), "--snapshot-id", config.snapshot_id,
            "--out", str(config.out_dir / "corpus.jsonl")]


def test_cli_build_records_keeps_finding_population_task_reward_and_lineage(tmp_path: Path) -> None:
    """One frozen mixed-source population owns emission, complete adjudication and provenance.

    Per-row helper tests miss interactions between canonical source identities, nonvoting captured
    replies and findings without judgments when the projector joins the complete store.
    """
    store = seed_projection_store(tmp_path, siblings=3)
    add_projection_run(store, run_id="mixed-case", dispositions=("accepted",), repo_slug="OWNER/Repo")
    add_projection_run(
        store, run_id="https-source", dispositions=("accepted",), repo_slug=" https://github.com/OWNER/Repo.git "
    )
    add_projection_run(
        store, run_id="ssh-source", dispositions=("accepted",), repo_slug="git@github.com:OWNER/Repo.git"
    )
    add_projection_run(store, run_id="git-suffix", dispositions=("accepted",), repo_slug="OWNER/Repo.git")
    unclaimed = projection_run("unknown-stack", dispositions=("accepted",))
    unclaimed["provenance"].pop("stack")
    store.commit_run(unclaimed)
    append_projection_evidence(store, unclaimed, dispositions=("accepted",))
    store.commit_run(projection_run("missing-judgments"))
    semantic, capture = reply_evidence("9", "Needs investigation.\n")
    store.append_observation(
        observation(
            "reply",
            run_id="sess-a",
            item_uid="item:2",
            role="automatic",
            semantic_evidence=[semantic],
            evidence_digest=reply_evidence_digest([semantic]),
            evidence_digest_scheme="reply-evidence-v1",
            reply_captures=[capture],
            payload={"type": "finding-judgment", "disposition": "ambiguous", "rationale": "nondirectional reply"},
        )
    )
    config = projection_config(store, tmp_path)
    assert cli_main(_cli_args(config)) == 0
    rows = _read_jsonl(config.out_dir / "corpus.jsonl")
    assert len(rows) == 7 and len({row["record_id"] for row in rows}) == 7
    assert {row["session_id"]: row["lineage"]["repo_slug"] for row in rows} == {
        "sess-a": "owner/repo-a",
        "mixed-case": "OWNER/Repo",
        "https-source": "OWNER/Repo",
        "ssh-source": "OWNER/Repo",
        "git-suffix": "OWNER/Repo",
        "unknown-stack": "owner/repo-a",
    }
    assert all(row["task_identity"]["repo_slug"] == row["lineage"]["repo_slug"] for row in rows)
    base_rows = [row for row in rows if row["session_id"] == "sess-a"]
    assert {row["outcome_label"] for row in base_rows} == {"accepted", "rejected"}
    assert {row["trajectory_id"] for row in base_rows} == {"sess-a:fix-0"}
    assert {row["task_segment"] for row in base_rows} == {"seg-0"}
    frozen = store.read_snapshot(config.snapshot_id)
    runs = {run["run_id"]: run for run in frozen.runs}
    schema = json.loads((config.out_dir / "schema.json").read_text())
    for row in rows:
        run = runs[row["session_id"]]
        jsonschema.validate(row, schema)
        assert row["finding_text"] == f"{row['disposition']} finding body"
        assert row["finding_text_sha256"] == hashlib.sha256(row["finding_text"].encode()).hexdigest()
        assert row["diff"] == run["original_task"]["value"]["diff"]
        assert row["task_identity"]["base_sha"] == "1" * 40
        assert row["task_identity"]["head_sha"] == "2" * 40
        assert row["task_identity"]["diff_ref"]["run_id"] == row["session_id"]
        assert row["lineage"]["snapshot_id"] == config.snapshot_id
        assert row["lineage"]["source_identity_version"] == "record-snapshot-v2"
        assert row["profile"]["profile_name"] == "deep-review"
        assert row["stack"] == (None if row["session_id"] == "unknown-stack" else "python")
        assert row["intrinsic_reward"] == run["scoring"]["value"]["persisted_breakdown"]
    report = json.loads((config.out_dir / "adjudication-report.json").read_text())
    assert {(entry["session_id"], entry["item_uid"]): entry["disposition"] for entry in report} == {
        ("sess-a", "item:2"): "ambiguous",
        ("missing-judgments", "item:0"): "unanswered",
        ("missing-judgments", "item:1"): "unanswered",
        ("missing-judgments", "item:2"): "unanswered",
    }
    undecided = next(entry for entry in report if entry["session_id"] == "sess-a")
    assert undecided["evidence"] == [semantic] and undecided["reply_captures"] == [capture]
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert lineage["adjudication_count"] == 4
    assert lineage["exclusions_by_reason"] == {"non-decisive-adjudication": 4}
    assert lineage["exclusion_list_digest"] == hashlib.sha256(EXCLUSION_PATH.read_bytes()).hexdigest()
    assert lineage["snapshot"]["snapshot_id"] == config.snapshot_id
    assert len(lineage["content_digests"]) == 16  # seven captured runs and nine observations
    assert (config.out_dir / "SHA256SUMS").is_file() and (config.out_dir / "_SUCCESS").is_file()
    assert len(load_dataset_v2(config.out_dir)) == 7


def test_cli_identity_and_benchmark_refusal_writes_no_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    for slug, reason in (("GETSENTRY/SENTRY", "c5_excluded_repo"), (None, "repo_identity_missing")):
        root = tmp_path / reason
        store = seed_projection_store(root, repo_slug=slug)
        config = projection_config(store, root)
        assert cli_main(_cli_args(config)) == 1
        captured = capsys.readouterr()
        assert reason in captured.out + captured.err
        assert not config.out_dir.exists()
    fork = projection_run(repo_slug="contributor/sentry")
    fork["original_task"]["value"]["pr"]["repo"] = "https://github.com/GetSentry/Sentry.git"
    (store := LocalRecordStore(tmp_path / "fork")).commit_run(fork)
    assert cli_main(_cli_args(config := projection_config(store, tmp_path / "fork-out"))) == 1
    assert "c5_excluded_repo" in "".join(capsys.readouterr()) and not config.out_dir.exists()


def test_required_frozen_inputs_fail_before_output(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path)
    (store.root / "snapshots" / f"{config.snapshot_id}.jsonl").unlink()
    assert cli_main(_cli_args(config)) == 1
    assert not config.out_dir.exists()
    fresh = LocalRecordStore(tmp_path / "absent-trace")
    run = projection_run(trajectories={"status": "unproduced"})
    fresh.commit_run(run)
    append_projection_evidence(fresh, run)
    config = projection_config(fresh, tmp_path / "absent-trace-output")
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


def test_configuration_requires_matching_temporal_pin(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path, as_of="2026-10-04T00:00:00Z")
    with pytest.raises(ValueError, match="valid_before"):
        build_frozen_corpus(config)
    assert not config.out_dir.exists()


def test_cli_posterior_annotation_preserves_captured_native_profile_with_claim_stack(tmp_path: Path) -> None:
    # A record without legacy profile fields carries its native effective-configuration profile.
    native = {"schema_version": "1", "name": "strict", "source_kind": "builtin", "digest": "d" * 64}
    assert extract_provenance({"effective_configuration": {"profile": native}})["profile"] == {
        f"profile_{name}": value for name, value in native.items()}
    explicit = extract_provenance({"profile_name": "explicit", "effective_configuration": {"profile": native}})
    assert explicit["profile"]["profile_name"] == "explicit"  # Recorded profile fields win over native fallback.
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
    store.append_observation(observation("harvest", schema_version="daydream.observation.v3", run_id="sess-a",
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


def test_cli_output_refuses_record_store_overlap_without_mutating_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path)
    input_root = tmp_path / "store-alias"
    input_root.symlink_to(store.root, target_is_directory=True)
    output_root = store.root
    (tmp_path / "operator-notes.txt").write_text("existing local evidence and notes must survive")
    before_files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    before_paths = set(tmp_path.rglob("*"))
    args = ["corpus", "build", "--store", str(input_root), "--snapshot-id", config.snapshot_id,
            "--out", str(output_root / "corpus.jsonl")]
    assert cli_main(args) == 1
    assert "overlaps the record store namespace" in capsys.readouterr().out
    after_files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after_files == before_files
    assert set(tmp_path.rglob("*")) == before_paths

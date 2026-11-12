"""Run the CPU pipeline on real projected files without mocks. Preserve frozen splits and projection
digests, and reject a gold record without its finding text.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.dataset import LocalRecordStore
from daydream.training.coordinator import PipelineConfig, run_pipeline
from daydream.training.corpus_projection.projector import build_frozen_corpus
from daydream.training.corpus_projection.splits import assign_split
from daydream.training.reward_model import OutcomeModel, score_comment
from daydream.training.stacks import _directory_digest, load_v2_projection
from tests.harness.record_projection import add_projection_run, projection_config

SALT = "issue-1081-coordinator-salt"
HOLDOUT_RATE = 0.2
VAL_RATE = 0.2
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
ACCEPTED_TEXT = "exact localized finding body"
REJECTED_TEXT = "rejected finding body"
DIFF_BODY = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n"


def _record_id(session_id: str, fingerprint: str) -> str:
    return hashlib.sha256(f"{session_id}\x1f{fingerprint}".encode("utf-8")).hexdigest()


def _v2_record(
    *, session_id: str, split: str, label: str | None, fingerprint: str, record_type: str = "outcome-finding",
    tier: str = "gold", finding_text: str | None = None,
) -> dict[str, Any]:
    """A full v2 record the projector would emit for one finding."""
    record: dict[str, Any] = {
        "schema_version": "3", "record_id": _record_id(session_id, fingerprint), "record_type": record_type,
        "tier": tier, "session_id": session_id, "trajectory_id": f"traj-{session_id}", "task_segment": "segment-0",
        "finding_fingerprint": fingerprint, "disposition": label if label is not None else "ambiguous", "evidence": [],
        "profile": {"profile_schema_version": 1, "profile_name": "decisive-only", "profile_source_kind": "curation",
            "profile_digest": hashlib.sha256(b"profile").hexdigest(),
        }, "stack": "python", "outcome_label": label, "lineage": {
            "hub_commit": None, "snapshot_id": "c" * 64, "source_identity_version": "record-snapshot-v2",
            "content_digests": [], "labeler_policy_version": "labeler-v3",
            "reply_classifier_version": "rc-1", "rubric_schema_version": "rubric-v2", "as_of": "2026-01-01T00:00:00Z",
            "valid_at": "2026-01-01T00:00:00Z", "split": split, "exclusion_reason": None, "repo_slug": "owner/repo",
        }, "task_identity": {
            "repo_slug": "owner/repo", "base_sha": BASE_SHA, "head_sha": HEAD_SHA,
            "diff_digest": hashlib.sha256(DIFF_BODY.encode("utf-8")).hexdigest(),
            "diff_ref": {"content_digest": hashlib.sha256(DIFF_BODY.encode("utf-8")).hexdigest(),
                "relpath": f"batches/{session_id}/diff.patch",
            },
        },
        "diff": DIFF_BODY,
    }
    if finding_text is not None:
        record["finding_text"] = finding_text
        record["finding_text_sha256"] = hashlib.sha256(finding_text.encode("utf-8")).hexdigest()
    return record


def _build_projection(
    tmp_path: Path, *, omit_finding_text: bool = False,
) -> Path:
    """Build gold outcomes plus silver/task-only records in content-derived splits; seed labels so the
    holdout contains both gold classes.
    """
    session_ids = [f"sess-{i:04d}" for i in range(80)]
    split_of = {sid: assign_split(_record_id(sid, "fp"), salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE)
        for sid in session_ids
    }
    holdout_sessions = [sid for sid in session_ids if split_of[sid] == "holdout"]
    assert len(holdout_sessions) >= 2
    # Seed both labels on both sides of the holdout boundary.
    label_of: dict[str, str] = {holdout_sessions[0]: "accepted", holdout_sessions[1]: "rejected"}
    others = [sid for sid in session_ids if sid not in label_of]
    for idx, sid in enumerate(others):
        label_of[sid] = "accepted" if idx % 2 == 0 else "rejected"

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "holdout": []}

    def _place(record: dict[str, Any]) -> None:
        split = str(record["lineage"]["split"])
        records_by_split[split].append(record)

    for sid in session_ids:
        label = label_of[sid]
        _place(_v2_record(session_id=sid, split=split_of[sid], label=label, fingerprint="fp",
                finding_text=None if (omit_finding_text and label == "accepted") else (
                    ACCEPTED_TEXT if label == "accepted" else REJECTED_TEXT
                ),
            )
        )
    # Non-gold derived record types use their own identity split.
    for sid, fingerprint, rtype, tier, text in (
        ("sess-silver", "fp-silver", "process-trace", "silver", "process commentary"),
        ("sess-task", "fp-task", "task-only", "task-only", None),
    ):
        record = _v2_record(
            session_id=sid, split="train", label=None, fingerprint=fingerprint, record_type=rtype, tier=tier,
            finding_text=text,
        )
        record["lineage"]["split"] = assign_split(
            str(record["record_id"]), salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE,
        )
        _place(record)

    out = tmp_path / "proj"
    out.mkdir()
    for split, records in records_by_split.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8"
        )
    (out / "lineage.json").write_text(
        json.dumps({"schema_version": "lineage", "salt": SALT, "holdout_rate": HOLDOUT_RATE, "val_rate": VAL_RATE})
        + "\n",
        encoding="utf-8",
    )
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                                            for path in sorted(out.iterdir())))
    (out / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return out


def test_v2_projection_without_finding_text_fails_closed(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path, omit_finding_text=True)
    with pytest.raises(RuntimeError, match="finding_text"):
        run_pipeline(PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out"), dry_run=False)


def test_cli_projection_wiring(tmp_path: Path, cli_runner: Any) -> None:
    proj_dir = _build_projection(tmp_path)
    out = tmp_path / "cli-out"
    res = cli_runner.invoke(["train", "--projection", str(proj_dir), "--out", str(out), "--dry-run"])
    assert res.exit_code == 0, res
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_identity"]["corpus_digest"] == load_v2_projection(proj_dir).digest


def _build_projection_50(tmp_path: Path) -> Path:
    """Build 50 schema-3 records from observation-v3 evidence through the offline projector."""
    store = LocalRecordStore(tmp_path / "projection-fixture" / "records")
    for index in range(25):
        add_projection_run(store, run_id=f"sess-gold-{index:02d}", dispositions=("accepted", "rejected"),
                           repo_slug=f"owner/repo-{index % 3}")
    for index in range(2):
        add_projection_run(store, run_id=f"sess-amb-{index}", dispositions=("ambiguous",))
    config = projection_config(store, tmp_path / "projection-fixture", salt="issue-1081-fixture-salt",
                               holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE)
    assert build_frozen_corpus(config)["emitted"] == 50
    return config.out_dir


def test_integration_50_real_projection_full_pipeline(tmp_path: Path) -> None:
    """Feed the real 50-record projector fixture through all pipeline stages. Require stable corpus digests
    and SFT rows without fixture post-processing; both gold classes are present.
    """

    proj_dir = _build_projection_50(tmp_path)
    projection = load_v2_projection(proj_dir)
    assert len(projection.records) == 50
    assert projection.by_split == {
        split: [json.loads(line) for line in (proj_dir / f"{split}.jsonl").read_text().splitlines() if line]
        for split in ("train", "validation", "holdout")
    }
    assert sum(len(records) for records in projection.by_split.values()) == 50
    assert projection.lineage["salt"] == "issue-1081-fixture-salt"
    copied = tmp_path / "copied-projection"
    shutil.copytree(proj_dir, copied)
    assert load_v2_projection(copied).digest == projection.digest
    (copied / "train.jsonl").write_bytes((copied / "train.jsonl").read_bytes() + b"\n")
    assert _directory_digest(copied) != projection.digest
    with pytest.raises(ValueError, match="digest mismatch"):
        load_v2_projection(copied)
    (copied / "SHA256SUMS").write_text("")
    pytest.raises(ValueError, load_v2_projection, copied).match("leaves out")
    (copied / "SHA256SUMS").unlink()
    pytest.raises(ValueError, load_v2_projection, copied).match("missing SHA256SUMS")
    gold_labels = {cast(str, r["outcome_label"]) for r in projection.records if r.get("tier") == "gold"}
    assert {"accepted", "rejected"} <= gold_labels

    manifest = run_pipeline(PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out"), dry_run=False)
    assert manifest["stages"]["stage0"]["status"] == "complete"
    sft_rows = [json.loads(line)
        for line in (tmp_path / "out/stage1/sft-dataset.jsonl").read_text().splitlines()
        if line
    ]
    assert sft_rows

    first = manifest["run_identity"]["corpus_digest"]
    assert first == projection.digest
    manifest2 = run_pipeline(PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out2"), dry_run=False)
    assert manifest2["run_identity"]["corpus_digest"] == first
    stage0 = tmp_path / "out/stage0"
    model = OutcomeModel(**json.loads((stage0 / "model-state.json").read_text()))
    assert model.label_ratio_reported == 0.5 and model.train_rows + model.held_out_rows == 50
    assert model.split_digest == manifest["stages"]["stage0"]["split"]["digest"]
    assert score_comment(model, "accepted finding body") > score_comment(model, "rejected finding body")
    gate = json.loads((stage0 / "gate-report.json").read_text())
    assert gate == manifest["stages"]["stage0"]["gate"] and gate["passed"] is True
    assert gate["separation"] > 0 and gate["evidence_digest"]
    assert 0 <= gate["accepted_ratio"] <= 1
    assert {name: (stage0 / name).read_bytes()
            for name in ("labels.jsonl", "split.json", "model-state.json", "gate-report.json")} == {
        name: (tmp_path / "out2/stage0" / name).read_bytes()
        for name in ("labels.jsonl", "split.json", "model-state.json", "gate-report.json")}

    # Resume consumes the persisted identity before any stage can overwrite evidence.
    from dataclasses import replace

    from daydream.training.lineage import ResumeAborted

    cfg = PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out")
    prior = {str(path.relative_to(cfg.out_dir)): path.read_bytes() for path in cfg.out_dir.rglob("*") if path.is_file()}
    resumed = run_pipeline(cfg, dry_run=False)
    assert resumed["run_identity"] == manifest["run_identity"]
    assert {
        str(path.relative_to(cfg.out_dir)): path.read_bytes() for path in cfg.out_dir.rglob("*") if path.is_file()
    } == prior
    with pytest.raises(ValueError) as failure:
        run_pipeline(replace(cfg, learning_rate=2e-5, max_seq_len=16384), dry_run=False)
    assert isinstance(failure.value, ResumeAborted)
    assert "learning_rate" in str(failure.value) and "max_seq_len" in str(failure.value)
    assert {
        str(path.relative_to(cfg.out_dir)): path.read_bytes() for path in cfg.out_dir.rglob("*") if path.is_file()
    } == prior


def test_projection_is_the_only_input(tmp_path: Path) -> None:
    legacy_ctor = cast(Any, PipelineConfig)
    with pytest.raises(TypeError):
        legacy_ctor(corpus=tmp_path / "corpus.jsonl", out_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="projection"):
        PipelineConfig(out_dir=tmp_path / "out")

"""Run the CPU pipeline on real projected files without mocks. Preserve frozen splits and projection
digests, and reject missing Stage-1/2 fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.training.coordinator import PipelineConfig, run_pipeline
from daydream.training.corpus_projection.splits import assign_split
from daydream.training.gate import _build_frozen_split, _split_digest
from daydream.training.reward_model import train_outcome_model
from daydream.training.rft import RftConfig, run_rft
from daydream.training.stacks import load_v2_projection
from tests.fixtures.training.build_projection_50 import build_projection_50

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
    tier: str = "gold", finding_text: str | None = None, base_sha: str = BASE_SHA,
) -> dict[str, Any]:
    """A full v2 record the projector would emit for one finding."""
    record: dict[str, Any] = {
        "schema_version": "2", "record_id": _record_id(session_id, fingerprint), "record_type": record_type,
        "tier": tier, "session_id": session_id, "trajectory_id": f"traj-{session_id}", "task_segment": "segment-0",
        "finding_fingerprint": fingerprint, "disposition": label if label is not None else "ambiguous", "evidence": [],
        "profile": {"profile_schema_version": 1, "profile_name": "decisive-only", "profile_source_kind": "curation",
            "profile_digest": hashlib.sha256(b"profile").hexdigest(),
        }, "stack": "python", "outcome_label": label, "lineage": {
            "hub_commit": None, "curation_id": "cur-1", "content_digests": [], "labeler_policy_version": "labeler-v3",
            "reply_classifier_version": "rc-1", "rubric_schema_version": "rubric-v2", "as_of": "2026-01-01T00:00:00Z",
            "valid_at": "2026-01-01T00:00:00Z", "split": split, "exclusion_reason": None, "repo_slug": "owner/repo",
            "license_decision": {"status": "admitted", "repo_slug": "owner/repo", "reason_code": None},
        }, "task_identity": {
            "repo_slug": "owner/repo", "source": "curation-bundle", "base_sha": base_sha, "head_sha": HEAD_SHA,
            "diff_digest": hashlib.sha256(DIFF_BODY.encode("utf-8")).hexdigest(),
            "diff_ref": {"content_digest": hashlib.sha256(DIFF_BODY.encode("utf-8")).hexdigest(),
                "relpath": f"batches/{session_id}/diff.patch",
            }, "replay_verification": None,
        },
        "diff": DIFF_BODY,
    }
    if finding_text is not None:
        record["finding_text"] = finding_text
        record["finding_text_sha256"] = hashlib.sha256(finding_text.encode("utf-8")).hexdigest()
    return record


def _build_projection(
    tmp_path: Path, *, omit_finding_text: bool = False, base_sha: str = BASE_SHA, n_sessions: int = 80,
) -> Path:
    """Build gold outcomes plus silver/task-only records in content-derived splits; seed labels so the
    holdout contains both gold classes.
    """
    session_ids = [f"sess-{i:04d}" for i in range(n_sessions)]
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
                ), base_sha=base_sha,
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
    (out / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return out


def _holdout_gold_comment_ids(proj_dir: Path) -> list[str]:
    proj = load_v2_projection(proj_dir)
    ids = []
    for record in proj.records:
        if (
            cast(dict[str, object], record["lineage"])["split"] == "holdout"
            and record.get("tier") == "gold"
            and record.get("outcome_label") in ("accepted", "rejected")
        ):
            ids.append(str(record["session_id"]))
    return ids


def test_stage0_v2_frozen_split_sft_and_rft_rows(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path)
    cfg = PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out")
    manifest = run_pipeline(cfg, dry_run=False)

    assert manifest["stages"]["stage0"]["status"] == "complete"
    split = json.loads((tmp_path / "out/stage0/split.json").read_text())
    held_out_ids = sorted(_holdout_gold_comment_ids(proj_dir))
    assert split["held_out_rows"] == len(held_out_ids)
    assert split["digest"] == _split_digest(held_out_ids, cfg.seed)

    assert manifest["run_identity"]["corpus_digest"] == load_v2_projection(proj_dir).digest

    # SFT retains accepted findings as nonempty completions.
    sft_lines = (tmp_path / "out/stage1/sft-dataset.jsonl").read_text().splitlines()
    assert sft_lines
    sft_rows = [json.loads(line) for line in sft_lines]
    assert sft_rows[0]["completion"] == ACCEPTED_TEXT
    assert all(row["completion"] != REJECTED_TEXT for row in sft_rows)

    assert manifest["stages"]["stage1"]["tier_counts"]["silver"] == 1

    # RFT retains validated full SHAs and the diff body.
    rft_lines = (tmp_path / "out/stage2/rft-inputs.jsonl").read_text().splitlines()
    assert rft_lines
    rft_rows = [json.loads(line) for line in rft_lines]
    for row in rft_rows:
        assert len(row["base_sha"]) == 40 and all(c in "0123456789abcdef" for c in row["base_sha"])
        assert len(row["head_sha"]) == 40 and all(c in "0123456789abcdef" for c in row["head_sha"])
        assert row["diff"] == DIFF_BODY
        # Frozen V2 records are format-valid; replay scores their adjudicated outcome through the
        # shared verdict vocabulary.
        assert row["format_valid"] is True
        assert row["verifier_verdicts"]

def test_stage0_v2_gold_record_without_finding_text_fails_closed(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path, omit_finding_text=True)
    cfg = PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out")
    with pytest.raises(RuntimeError, match="finding_text"):
        run_pipeline(cfg, dry_run=False)


def test_stage0_uses_pinned_splits_even_when_records_are_in_other_files(tmp_path: Path) -> None:
    """File order determines labels; pinned partition order determines seeded SGD."""
    proj_dir = _build_projection(tmp_path)
    records: list[dict[str, Any]] = []
    for split in ("validation", "holdout", "train"):
        records.extend(json.loads(line) for line in (proj_dir / f"{split}.jsonl").read_text().splitlines())
    (proj_dir / "train.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (proj_dir / "validation.jsonl").write_text("")
    (proj_dir / "holdout.jsonl").write_text("")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "holdout": []}
    labels = []
    for record in records:
        if record["outcome_label"] not in ("accepted", "rejected"):
            continue
        row = {
            "comment_id": record["session_id"],
            "text": record["finding_text"],
            "label": record["outcome_label"],
            "labeler_policy_version": record["lineage"]["labeler_policy_version"],
        }
        labels.append(row)
        rows_by_split[record["lineage"]["split"]].append(row)
    expected_labels = tmp_path / "expected-labels.jsonl"
    expected_labels.write_text("\n".join(json.dumps(row, sort_keys=True) for row in labels))
    expected_split = _build_frozen_split(
        expected_labels,
        train_rows=[*rows_by_split["train"], *rows_by_split["validation"]],
        held_out_rows=rows_by_split["holdout"],
        seed=0,
        held_out_fraction=HOLDOUT_RATE,
    )
    expected_model = train_outcome_model(expected_split, seed=0)

    out = tmp_path / "out"
    run_pipeline(PipelineConfig(projection=proj_dir, out_dir=out, stages=("stage0",)), dry_run=False)
    assert (out / "stage0/labels.jsonl").read_bytes() == expected_labels.read_bytes()
    assert json.loads((out / "stage0/model-state.json").read_text()) == expected_model.state_dict()

def test_stage2_v2_truncated_sha_fails_closed(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path, base_sha="abc123")
    cfg = PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out")
    with pytest.raises(RuntimeError, match="base_sha"):
        run_pipeline(cfg, dry_run=False)

def test_cli_projection_wiring(tmp_path: Path, cli_runner: Any) -> None:
    proj_dir = _build_projection(tmp_path)
    out = tmp_path / "cli-out"
    res = cli_runner.invoke(["train", "--projection", str(proj_dir), "--out", str(out), "--dry-run"])
    assert res.exit_code == 0, res
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_identity"]["corpus_digest"] == load_v2_projection(proj_dir).digest

def test_integration_50_real_projection_full_pipeline(tmp_path: Path) -> None:
    """Feed the real 50-record projector fixture through all pipeline stages. Require stable corpus digests
    and replayable RFT rows without fixture post-processing; both gold classes and non-gold record types
    are present.
    """

    proj_dir = build_projection_50(tmp_path)
    projection = load_v2_projection(proj_dir)
    assert len(projection.records) == 50
    gold_labels = {cast(str, r["outcome_label"]) for r in projection.records if r.get("tier") == "gold"}
    assert {"accepted", "rejected"} <= gold_labels
    record_types = {cast(str, r["record_type"]) for r in projection.records}
    assert {"process-trace", "task-only"} <= record_types

    manifest = run_pipeline(PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out"), dry_run=False)
    assert manifest["stages"]["stage0"]["status"] == "complete"
    sft_rows = [json.loads(line)
        for line in (tmp_path / "out/stage1/sft-dataset.jsonl").read_text().splitlines()
        if line
    ]
    rft_rows = [json.loads(line)
        for line in (tmp_path / "out/stage2/rft-inputs.jsonl").read_text().splitlines()
        if line
    ]
    assert sft_rows and rft_rows
    for row in rft_rows:
        assert len(row["base_sha"]) == 40
        assert len(row["head_sha"]) == 40
        assert row["repo_slug"]
        assert row["diff"]

    # Replay rebuilds tasks from the projection's frozen repo/base/head/diff identity without
    # fixture-side diff materialization.
    replay = run_rft(RftConfig(inputs=tmp_path / "out/stage2/rft-inputs.jsonl", seed=7, rubric_version="2026.08.29-1",
            output_dir=tmp_path / "out/replay",
        )
    )
    assert replay.winners_path.is_file()
    # Accepted gold must score through correctness so winner filtering discriminates between
    # candidates.
    winners = json.loads(replay.winners_path.read_text())["winners"]
    assert winners
    assert any(float(w["breakdown"]["composite"]) > 0 for w in winners)

    first = manifest["run_identity"]["corpus_digest"]
    assert first == projection.digest
    manifest2 = run_pipeline(PipelineConfig(projection=proj_dir, out_dir=tmp_path / "out2"), dry_run=False)
    assert manifest2["run_identity"]["corpus_digest"] == first

def test_projection_is_the_only_input(tmp_path: Path) -> None:
    legacy_ctor = cast(Any, PipelineConfig)
    with pytest.raises(TypeError):
        legacy_ctor(corpus=tmp_path / "corpus.jsonl", out_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="projection"):
        PipelineConfig(out_dir=tmp_path / "out")

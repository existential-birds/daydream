"""Corpus-v2 integration tests for the four-stage training coordinator.

Enters from the production entrypoint (``run_pipeline``) with a real corpus-v2
projection directory on the real filesystem (mocking nothing — every stage
here is CPU-bound) and asserts observable outcomes: Stage 0 consumes the
projector's frozen split rather than re-freezing at runtime, the manifest
carries the projection's directory-level digest, and the Stage-1/2 row
builders read the v2 fields (``finding_text``, ``task_identity``) fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training.coordinator import PipelineConfig, run_pipeline
from daydream.training.corpus_v2.splits import assign_split
from daydream.training.gate import _split_digest
from daydream.training.stacks_v2 import load_v2_projection

SALT = "issue-1081-coordinator-salt"
HOLDOUT_RATE = 0.2
VAL_RATE = 0.2
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
ACCEPTED_TEXT = "exact localized finding body"
REJECTED_TEXT = "rejected finding body"
DIFF_BODY = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n"


def _record_id(session_id: str, fingerprint: str) -> str:
    """The v2 record id: sha256 over the canonical (session, fingerprint) join."""
    return hashlib.sha256(f"{session_id}\x1f{fingerprint}".encode("utf-8")).hexdigest()


def _v2_record(
    *,
    session_id: str,
    split: str,
    label: str | None,
    fingerprint: str,
    record_type: str = "outcome-finding",
    tier: str = "gold",
    finding_text: str | None = None,
    base_sha: str = BASE_SHA,
) -> dict[str, Any]:
    """A full v2 record the projector would emit for one finding."""
    record: dict[str, Any] = {
        "schema_version": "2",
        "record_id": _record_id(session_id, fingerprint),
        "record_type": record_type,
        "tier": tier,
        "session_id": session_id,
        "trajectory_id": f"traj-{session_id}",
        "task_segment": "segment-0",
        "finding_fingerprint": fingerprint,
        "disposition": label if label is not None else "ambiguous",
        "evidence": [],
        "profile": {
            "profile_schema_version": 1,
            "profile_name": "decisive-only",
            "profile_source_kind": "curation",
            "profile_digest": hashlib.sha256(b"profile").hexdigest(),
        },
        "stack": "python",
        "outcome_label": label,
        "lineage": {
            "hub_commit": None,
            "curation_id": "cur-1",
            "content_digests": [],
            "labeler_policy_version": "labeler-v3",
            "reply_classifier_version": "rc-1",
            "rubric_schema_version": "rubric-v2",
            "as_of": "2026-01-01T00:00:00Z",
            "valid_at": "2026-01-01T00:00:00Z",
            "split": split,
            "exclusion_reason": None,
            "repo_slug": "owner/repo",
            "license_decision": {
                "status": "admitted",
                "repo_slug": "owner/repo",
                "reason_code": None,
            },
        },
        "task_identity": {
            "repo_slug": "owner/repo",
            "source": "curation-bundle",
            "base_sha": base_sha,
            "head_sha": HEAD_SHA,
            "diff_digest": hashlib.sha256(DIFF_BODY.encode("utf-8")).hexdigest(),
            "diff_ref": {
                "content_digest": hashlib.sha256(DIFF_BODY.encode("utf-8")).hexdigest(),
                "relpath": f"batches/{session_id}/diff.patch",
            },
            "replay_verification": None,
        },
        # The materialized diff body the RFT replay rebuilds the task from.
        "diff": DIFF_BODY,
    }
    if finding_text is not None:
        record["finding_text"] = finding_text
        record["finding_text_sha256"] = hashlib.sha256(finding_text.encode("utf-8")).hexdigest()
    return record


def _build_projection(
    tmp_path: Path,
    *,
    omit_finding_text: bool = False,
    base_sha: str = BASE_SHA,
    n_sessions: int = 80,
) -> Path:
    """Build a real corpus-v2 projection directory with accepted + rejected
    gold outcome findings (plus one silver process-trace and one task-only
    record), placed in the split file each record id deterministically
    assigns. Label assignment is seeded off the deterministic holdout
    membership so the holdout side always carries both gold classes."""
    session_ids = [f"sess-{i:04d}" for i in range(n_sessions)]
    split_of = {
        sid: assign_split(
            _record_id(sid, "fp"),
            salt=SALT,
            holdout_rate=HOLDOUT_RATE,
            val_rate=VAL_RATE,
        )
        for sid in session_ids
    }
    holdout_sessions = [sid for sid in session_ids if split_of[sid] == "holdout"]
    assert len(holdout_sessions) >= 2
    # Deterministic labels: first holdout session accepted, second rejected,
    # everything else alternating — both classes on both sides of the boundary.
    label_of: dict[str, str] = {
        holdout_sessions[0]: "accepted",
        holdout_sessions[1]: "rejected",
    }
    others = [sid for sid in session_ids if sid not in label_of]
    for idx, sid in enumerate(others):
        label_of[sid] = "accepted" if idx % 2 == 0 else "rejected"

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "holdout": [],
    }

    def _place(record: dict[str, Any]) -> None:
        split = str(record["lineage"]["split"])
        records_by_split[split].append(record)

    for sid in session_ids:
        label = label_of[sid]
        _place(
            _v2_record(
                session_id=sid,
                split=split_of[sid],
                label=label,
                fingerprint="fp",
                finding_text=None if (omit_finding_text and label == "accepted") else (
                    ACCEPTED_TEXT if label == "accepted" else REJECTED_TEXT
                ),
                base_sha=base_sha,
            )
        )
    # A silver process-trace and a task-only record — schema-distinct
    # non-gold types that must not become gold outcome rows. Each lands in
    # the split its own record id deterministically assigns.
    for sid, fingerprint, rtype, tier, text in (
        ("sess-silver", "fp-silver", "process-trace", "silver", "process commentary"),
        ("sess-task", "fp-task", "task-only", "task-only", None),
    ):
        record = _v2_record(
            session_id=sid,
            split="train",
            label=None,
            fingerprint=fingerprint,
            record_type=rtype,
            tier=tier,
            finding_text=text,
        )
        record["lineage"]["split"] = assign_split(
            str(record["record_id"]),
            salt=SALT,
            holdout_rate=HOLDOUT_RATE,
            val_rate=VAL_RATE,
        )
        _place(record)

    out = tmp_path / "proj"
    out.mkdir()
    for split, records in records_by_split.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8"
        )
    (out / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": "corpus-v2",
                "salt": SALT,
                "holdout_rate": HOLDOUT_RATE,
                "val_rate": VAL_RATE,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return out


def _holdout_gold_comment_ids(proj_dir: Path) -> list[str]:
    proj = load_v2_projection(proj_dir)
    ids = []
    for record in proj.by_split["holdout"]:
        if record.get("tier") == "gold" and record.get("outcome_label") in ("accepted", "rejected"):
            ids.append(str(record["session_id"]))
    return ids


def test_stage0_v2_frozen_split_sft_and_rft_rows(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path)
    cfg = PipelineConfig(corpus_v2=proj_dir, out_dir=tmp_path / "out")
    manifest = run_pipeline(cfg, dry_run=False)

    # Stage 0 runs to completion on the FROZEN split (not re-frozen at runtime).
    assert manifest["stages"]["stage0"]["status"] == "complete"
    split = json.loads((tmp_path / "out/stage0/split.json").read_text())
    held_out_ids = sorted(_holdout_gold_comment_ids(proj_dir))
    assert split["held_out_rows"] == len(held_out_ids)
    # The split digest is exactly the frozen projector boundary: the content
    # digest of the holdout gold comment ids under the run seed.
    assert split["digest"] == _split_digest(held_out_ids, cfg.seed)

    # The manifest's corpus digest is the projection's directory-level digest.
    assert manifest["run_identity"]["corpus_digest"] == load_v2_projection(proj_dir).digest

    # SFT: non-empty accepted-only completions carrying the real finding text.
    sft_lines = (tmp_path / "out/stage1/sft-dataset.jsonl").read_text().splitlines()
    assert sft_lines
    sft_rows = [json.loads(line) for line in sft_lines]
    assert sft_rows[0]["completion"] == ACCEPTED_TEXT
    assert all(row["completion"] != REJECTED_TEXT for row in sft_rows)

    # SFT tier counts separate silver process traces from gold rows.
    assert manifest["stages"]["stage1"]["tier_counts"]["silver"] == 1

    # RFT: non-empty inputs with full-length validated SHAs and a diff body.
    rft_lines = (tmp_path / "out/stage2/rft-inputs.jsonl").read_text().splitlines()
    assert rft_lines
    rft_rows = [json.loads(line) for line in rft_lines]
    for row in rft_rows:
        assert len(row["base_sha"]) == 40 and all(c in "0123456789abcdef" for c in row["base_sha"])
        assert len(row["head_sha"]) == 40 and all(c in "0123456789abcdef" for c in row["head_sha"])
        assert row["diff"] == DIFF_BODY


def test_stage0_v2_gold_record_without_finding_text_fails_closed(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path, omit_finding_text=True)
    cfg = PipelineConfig(corpus_v2=proj_dir, out_dir=tmp_path / "out")
    with pytest.raises(RuntimeError, match="finding_text"):
        run_pipeline(cfg, dry_run=False)


def test_stage2_v2_truncated_sha_fails_closed(tmp_path: Path) -> None:
    proj_dir = _build_projection(tmp_path, base_sha="abc123")
    cfg = PipelineConfig(corpus_v2=proj_dir, out_dir=tmp_path / "out")
    with pytest.raises(RuntimeError, match="base_sha"):
        run_pipeline(cfg, dry_run=False)


def test_corpus_and_corpus_v2_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        PipelineConfig(
            corpus=tmp_path / "corpus.jsonl",
            corpus_v2=tmp_path / "proj",
            out_dir=tmp_path / "out",
        )
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig(out_dir=tmp_path / "out")

"""Stage-0 gate refusal (M4): a Stage-3 run may not start without gate evidence.

Every path here is fail-closed: a missing, unreadable, or failed gate report
refuses the run and names the reason. There is no default-to-allowed branch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from daydream_review.gate_refusal import (
    Stage0GateRefused,
    _evidence_digest,
    require_outcome_model_bound,
    require_stage0_gate,
)

MODEL_STATE = {
    "weights": {"bug": 1.0, "race": 0.5, "regression": 0.75},
    "bias": -0.25,
    "split_digest": "split-digest",
    "label_ratio_reported": 0.5,
    "train_rows": 10,
    "held_out_rows": 4,
    "held_out_accuracy": 0.75,
    "model_fingerprint": "abc12345",
}


def _bound_gate_report() -> dict[str, object]:
    evidence = {
        "split_digest": MODEL_STATE["split_digest"],
        "model_fingerprint": MODEL_STATE["model_fingerprint"],
        "thresholds": {"min_separation": 0.1, "min_calibration": 0.5},
        "held_out_rows": MODEL_STATE["held_out_rows"],
        "separation": 0.2,
        "calibration": 0.75,
        "accepted_ratio": 0.5,
    }
    return {
        "passed": True,
        "separation": evidence["separation"],
        "calibration": evidence["calibration"],
        "accepted_ratio": evidence["accepted_ratio"],
        "evidence_digest": _evidence_digest(evidence),
        "thresholds": evidence["thresholds"],
        "held_out_rows": evidence["held_out_rows"],
    }


def _write_checkpoint(tmp_path: Path, *, split_digest: str = "split-digest") -> Path:
    p = tmp_path / "outcome-model.json"
    p.write_text(json.dumps({**MODEL_STATE, "split_digest": split_digest}), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "content, match",
    [
        (None, "missing"),
        ("{not json", "unreadable"),
        ('{"passed": false, "separation": 0.01}', "failed"),
    ],
    ids=["missing", "unparseable", "failed"],
)
def test_start_refused(tmp_path: Path, content: str | None, match: str) -> None:
    gate_report_path = tmp_path / "gate.json"
    if content is not None:
        gate_report_path.write_text(content)
    with pytest.raises(Stage0GateRefused, match=match) as exc_info:
        require_stage0_gate(gate_report_path=gate_report_path)
    assert str(gate_report_path) in str(exc_info.value)


def test_start_allowed_on_passed_gate(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text('{"passed": true, "evidence_digest": "abc"}')
    report = require_stage0_gate(gate_report_path=tmp_path / "gate.json")  # no raise
    assert report["evidence_digest"] == "abc"


def test_checkpoint_bound_to_passed_report(tmp_path: Path) -> None:
    """The fixture pair (report + matching checkpoint) passes the binding."""
    report = _bound_gate_report()
    require_outcome_model_bound(report, _write_checkpoint(tmp_path))  # no raise


@pytest.mark.parametrize(
    "state, report, match",
    [
        (None, _bound_gate_report(), "missing"),
        ("{not json", _bound_gate_report(), "unreadable"),
        (json.dumps({**MODEL_STATE, "split_digest": "some-other-split"}), _bound_gate_report(), "does not bind"),
        (
            json.dumps(MODEL_STATE),
            {"passed": True, "evidence_digest": "fixture-digest"},
            "lacks the recomputable evidence",
        ),
    ],
    ids=["missing", "unreadable", "digest-mismatch", "no-measurements"],
)
def test_checkpoint_refused(
    tmp_path: Path, state: str | None, report: dict[str, object], match: str
) -> None:
    checkpoint = tmp_path / "outcome-model.json"
    if state is not None:
        checkpoint.write_text(state)
    with pytest.raises(Stage0GateRefused, match=match):
        require_outcome_model_bound(report, checkpoint)


def _projection_record(
    *, session_id: str, split: str, label: str | None,
    finding_text: str | None, diff_body: str, base_sha: str, head_sha: str,
) -> dict[str, object]:
    """A projected-corpus record the projector would emit for one finding, kept
    minimal but schema-faithful so the coordinator's fail-closed gates admit it."""
    diff_digest = hashlib.sha256(diff_body.encode("utf-8")).hexdigest()
    record: dict[str, object] = {
        "schema_version": "2",
        "record_id": hashlib.sha256(f"{session_id}\x1ffp".encode("utf-8")).hexdigest(),
        "record_type": "outcome-finding",
        "tier": "gold",
        "session_id": session_id,
        "trajectory_id": f"traj-{session_id}",
        "task_segment": "segment-0",
        "finding_fingerprint": "fp",
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
            "license_decision": {"status": "admitted", "repo_slug": "owner/repo", "reason_code": None},
        },
        "task_identity": {
            "repo_slug": "owner/repo",
            "source": "curation-bundle",
            "base_sha": base_sha,
            "head_sha": head_sha,
            "diff_digest": diff_digest,
            "diff_ref": {
                "content_digest": diff_digest,
                "relpath": f"batches/{session_id}/diff.patch",
            },
            "replay_verification": None,
        },
        "diff": diff_body,
    }
    if finding_text is not None:
        record["finding_text"] = finding_text
        record["finding_text_sha256"] = hashlib.sha256(finding_text.encode("utf-8")).hexdigest()
    return record


def _build_projection(tmp_path: Path) -> Path:
    """Build a real projected-corpus projection directory with accepted + rejected
    gold outcome rows on both sides of the frozen split."""
    from daydream.training.corpus_projection.splits import assign_split

    salt = "gate-refusal-projection-salt"
    holdout_rate = val_rate = 0.2
    diff_body = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n"
    base_sha, head_sha = "a" * 40, "b" * 40
    sessions = [f"sess-{i:04d}" for i in range(80)]
    split_of = {
        sid: assign_split(
            hashlib.sha256(f"{sid}\x1ffp".encode("utf-8")).hexdigest(),
            salt=salt,
            holdout_rate=holdout_rate,
            val_rate=val_rate,
        )
        for sid in sessions
    }
    holdout = [sid for sid in sessions if split_of[sid] == "holdout"]
    assert len(holdout) >= 2
    label_of = {holdout[0]: "accepted", holdout[1]: "rejected"}
    idx = 0
    for sid in sessions:
        if sid in label_of:
            continue
        label_of[sid] = "accepted" if idx % 2 == 0 else "rejected"
        idx += 1

    by_split: dict[str, list[str]] = {"train": [], "validation": [], "holdout": []}
    for sid in sessions:
        label = label_of[sid]
        text = (
            "exact localized accepted finding body"
            if label == "accepted"
            else "exact localized rejected finding body"
        )
        record = _projection_record(
            session_id=sid, split=split_of[sid], label=label,
            finding_text=text, diff_body=diff_body, base_sha=base_sha, head_sha=head_sha,
        )
        by_split[str(record["lineage"]["split"])].append(  # type: ignore[index]
            json.dumps(record, sort_keys=True)
        )

    proj = tmp_path / "proj"
    proj.mkdir()
    for split_name, lines in by_split.items():
        (proj / f"{split_name}.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
    (proj / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": "lineage",
                "salt": salt,
                "holdout_rate": holdout_rate,
                "val_rate": val_rate,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (proj / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return proj


def test_coordinator_gate_report_is_consumed_unmodified(tmp_path: Path) -> None:
    """Stage-boundary contract audit: the gate-report.json the coordinator's
    Stage-0 stage writes must satisfy require_stage0_gate verbatim — no
    reshaping at the handoff, no nested wrapper that reads as passed=None."""
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    projection = _build_projection(tmp_path)
    run_pipeline(
        PipelineConfig(projection=projection, out_dir=tmp_path, stages=("stage0",)),
        dry_run=True,
    )
    report_path = tmp_path / "stage0" / "gate-report.json"
    # The behavioral contract: the on-disk gate-report.json (manifest "gate" and the
    # file both derive from the same report.to_dict(), so a manifest-vs-file digest
    # comparison is tautological) must satisfy the Stage-3 boundary consumer verbatim.
    report = require_stage0_gate(gate_report_path=report_path)  # no raise
    assert report["passed"] is True

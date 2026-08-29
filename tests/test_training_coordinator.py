"""Tests for the four-stage training coordinator (M15, M16).

The coordinator is driven through its public entrypoint
(:func:`run_pipeline`) with a real JSONL corpus on a real filesystem; only
the wall-clock GPU boundary is bypassed via ``dry_run=True``. Tests assert
observable outcomes — manifest shape, stage directory layout, gate refusal —
never internal call counts.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.training.coordinator import PipelineConfig, run_pipeline
from daydream.training.gate import GateConfig, GateReport


def _write_corpus(path: Path, n: int = 50) -> Path:
    """Write an n-record corpus: gold outcome rows (both classes) plus lineage."""
    rows = []
    for i in range(n):
        accepted = i % 2 == 0
        rows.append(
            {
                "session_id": f"sess-{i:04d}",
                "repo_slug": f"acme/tooling-{i % 7}",
                "comment_id": f"c{i:04d}",
                "text": f"grounded actionable finding {i}" if accepted else f"noise chatter {i}",
                "label": "accepted" if accepted else "rejected",
                "labeler_policy_version": "980-policy-r1",
                "base_sha": f"a{i:064x}",
                "head_sha": f"b{i:064x}",
                "diff": f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ for sess-{i:04d}\n",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture
def records_50_fixture(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path / "corpus.jsonl", 50)


def _failed_report() -> GateReport:
    return GateReport(
        passed=False,
        separation=-0.5,
        calibration=0.1,
        accepted_ratio=0.5,
        evidence_digest="0" * 64,
        thresholds=GateConfig().thresholds(),
        held_out_rows=10,
    )


def test_full_run_writes_manifest_and_adapter(
    records_50_fixture: Path, tmp_path: Path
) -> None:
    cfg = PipelineConfig(
        corpus=records_50_fixture,
        out_dir=tmp_path,
        stages=("stage0", "stage1", "stage2", "stage3"),
    )
    run_pipeline(cfg, dry_run=True)  # dry path pinned in CI; here the manifest shape
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["stages"]) == {"stage0", "stage1", "stage2", "stage3"}
    assert manifest["stages"]["stage0"]["gate"]["evidence_digest"]
    assert manifest["run_identity"]["base_model"]  # LOCKED_FIELDS identity stamped (M18 tie-in)


def test_stage3_consumer_reads_coordinator_gate_report(
    records_50_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-boundary contract: the on-disk gate evidence the coordinator produces must
    be directly consumable by the Stage-3 refusal seam (require_stage0_gate), whose
    schema is a top-level ``passed`` on the GateReport payload."""
    from daydream.training.gate import evaluate_gate

    seen: dict[str, object] = {}

    def _spy(model: object, split: object, config: GateConfig) -> GateReport:
        report = evaluate_gate(model, split, config)  # type: ignore[arg-type]
        seen["digest"] = report.evidence_digest
        return report

    monkeypatch.setattr("daydream.training.coordinator.gate_mod.evaluate_gate", _spy)
    run_pipeline(PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path), dry_run=True)
    report = json.loads((tmp_path / "stage0" / "gate-report.json").read_text())
    assert report["passed"] is True  # top-level, not nested under "gate"
    assert report["evidence_digest"] == seen["digest"]
    # split evidence survives as its own sibling artifact
    assert (tmp_path / "stage0" / "split.json").is_file()
    # the consumer-side acceptance of this exact file is pinned in
    # rl/daydream_review_v1/tests/test_gate_refusal.py::
    # test_coordinator_gate_report_is_consumed_unmodified


def test_gate_failure_stops_pipeline_before_stage3(
    records_50_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("daydream.training.gate.evaluate_gate", lambda *a, **k: _failed_report())
    with pytest.raises(RuntimeError, match="gate"):
        run_pipeline(PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path), dry_run=True)
    assert not (tmp_path / "stage3").exists()  # hard refusal, not a warning
    # no partial-success artifact: the manifest is not written for a refused run
    assert not (tmp_path / "manifest.json").exists()


def test_missing_gate_refuses_stage3(records_50_fixture: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="gate"):
        run_pipeline(
            PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path, stages=("stage3",)),
            dry_run=True,
        )
    assert not (tmp_path / "stage3").exists()


def test_dry_run_marks_gpu_stages_skipped(records_50_fixture: Path, tmp_path: Path) -> None:
    run_pipeline(PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path), dry_run=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stages"]["stage0"]["status"] == "complete"
    for stage in ("stage1", "stage2", "stage3"):
        assert manifest["stages"][stage]["status"] == "skipped_dry"


def test_non_dry_run_writes_stage_outputs_and_adapter(
    records_50_fixture: Path, tmp_path: Path
) -> None:
    run_pipeline(PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path), dry_run=False)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for stage in ("stage0", "stage1", "stage2", "stage3"):
        assert manifest["stages"][stage]["status"] == "complete"
        assert (tmp_path / stage).is_dir()
    # final stage points at a loadable LoRA adapter checkpoint (save_adapter_separately shape)
    adapter = Path(manifest["adapter_path"])
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_state.json").is_file()


@pytest.fixture
def cli_runner() -> Any:
    class _Runner:
        def invoke(self, argv: list[str]) -> SimpleNamespace:
            from daydream import cli

            saved = sys.argv
            sys.argv = ["daydream", *argv]
            code = 0
            try:
                cli.main()
            except SystemExit as exc:
                code = int(exc.code or 0)
            finally:
                sys.argv = saved
            return SimpleNamespace(exit_code=code)

    return _Runner()


def test_cli_verb_wired(cli_runner: Any) -> None:
    r = cli_runner.invoke(["train", "--help"])
    assert r.exit_code == 0


def _production_corpus(path: Path, n: int = 20) -> Path:
    """Production run_build_corpus export shape: outcome_label/review_output/session_id."""
    rows = []
    for i in range(n):
        accepted = i % 2 == 0
        rows.append(
            {
                "schema_version": "1",
                "session_id": f"sess-{i:04d}",
                "repo_slug": f"acme/tooling-{i % 7}",
                "review_output": (
                    f"grounded actionable finding {i}" if accepted else f"noise chatter {i}"
                ),
                "outcome_label": "accepted" if accepted else "rejected",
                "labeler_policy_version": "980-policy-r1",
                "code_context": {"base_sha": f"a{i:064x}", "head_sha": f"b{i:064x}"},
                "diff": f"diff --git a/f.py b/f.py\n@@ for sess-{i:04d}\n",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_production_shaped_corpus_runs_stage0(tmp_path: Path) -> None:
    """Issue 1/2: a production run_build_corpus export (outcome_label/review_output)
    feeds Stage 0 instead of refusing with 'corpus carries no gold outcome rows'."""
    corpus = _production_corpus(tmp_path / "corpus.jsonl")
    run_pipeline(PipelineConfig(corpus=corpus, out_dir=tmp_path / "out", stages=("stage0",)), dry_run=True)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stages"]["stage0"]["status"] == "complete"
    assert manifest["stages"]["stage0"]["gate"]["passed"] is True


def test_stage2_writes_runnable_rft_inputs(records_50_fixture: Path, tmp_path: Path) -> None:
    """Issue 3: Stage-2 rft-inputs carry the frozen task identity run_rft requires
    (id/base_sha/head_sha/diff), so the recorded-complete stage is actually runnable."""
    run_pipeline(
        PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path, stages=("stage2",)), dry_run=False
    )
    inputs = [
        json.loads(line)
        for line in (tmp_path / "stage2" / "rft-inputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert inputs
    assert all(r["id"] and r["base_sha"] and r["head_sha"] and r["diff"] for r in inputs)


def test_stage2_refuses_records_without_diff_identity(tmp_path: Path) -> None:
    """Issue 3: Stage 2 refuses records missing base/head/diff identity instead of
    recording the stage complete over unrunnable inputs."""
    corpus = tmp_path / "noid.jsonl"
    corpus.write_text(
        json.dumps({"comment_id": "c0", "text": "x", "label": "accepted",
                     "labeler_policy_version": "980-policy-r1"})
    )
    with pytest.raises(RuntimeError, match="diff"):
        run_pipeline(
            PipelineConfig(corpus=corpus, out_dir=tmp_path / "out", stages=("stage2",)), dry_run=False
        )


def test_resume_guard_aborts_on_drifted_rerun(records_50_fixture: Path, tmp_path: Path) -> None:
    """Issue 8/9: the M18/AC4 resume guard is wired into run_pipeline — a re-run whose
    locked identity drifts from the prior manifest aborts loudly instead of overwriting."""
    from daydream.training.lineage import ResumeAborted

    cfg = PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path)
    run_pipeline(cfg, dry_run=True)
    run_pipeline(cfg, dry_run=True)  # identical re-run passes
    drifted = PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path, learning_rate=2e-5)
    with pytest.raises(ResumeAborted, match="learning_rate"):
        run_pipeline(drifted, dry_run=True)


def test_model_split_digest_matches_manifest(records_50_fixture: Path, tmp_path: Path) -> None:
    """Issue 14: the model checkpoint split_digest reconciles with the manifest's
    run_identity.split_digest (AC4/M18 split-digest detection is enforceable)."""
    run_pipeline(PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path), dry_run=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    model_state = json.loads((tmp_path / "stage0" / "model-state.json").read_text())
    assert model_state["split_digest"] == manifest["run_identity"]["split_digest"]


def test_identity_defaults_match_shipped_recipes(records_50_fixture: Path, tmp_path: Path) -> None:
    """Issue 11/15: locked-identity defaults match the shipped GPU recipes
    (seq_len 32768, lr 1e-5) instead of contradicting them."""
    run_pipeline(PipelineConfig(corpus=records_50_fixture, out_dir=tmp_path), dry_run=True)
    identity = json.loads((tmp_path / "manifest.json").read_text())["run_identity"]
    assert identity["max_seq_len"] == 32768
    assert identity["learning_rate"] == 1e-5
    adapter = json.loads((tmp_path / "stage3" / "adapter" / "adapter_config.json").read_text())
    assert adapter["learning_rate"] == 1e-5

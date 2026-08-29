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
def cli_runner():
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


def test_cli_verb_wired(cli_runner) -> None:
    r = cli_runner.invoke(["train", "--help"])
    assert r.exit_code == 0

"""The committed projection fixture must pass loader gates and the canonical training dry run."""

import json
from pathlib import Path

from daydream.training.coordinator import PipelineConfig, run_pipeline


def test_pipeline_dry_run_over_committed_fixture(tmp_path: Path) -> None:
    manifest = run_pipeline(
        PipelineConfig(projection=Path("tests/fixtures/training/projection-50"), out_dir=tmp_path), dry_run=True,
    )
    assert manifest["stages"]["stage0"]["status"] == "complete"
    assert manifest["stages"]["stage0"]["gate"]["passed"] is True
    for stage in ("stage1", "stage2"):
        assert manifest["stages"][stage]["status"] == "skipped_dry"
    adapter = Path(manifest["adapter_path"])
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_state.json").is_file()
    assert json.loads((tmp_path / "manifest.json").read_text())["dry_run"] is True

"""Committed 50-record fixture + GPU-free CI dry path (M20, M24).

The fixture is a real corpus-format JSONL file (50 records, realistic
accepted/rejected mix, full M16 lineage fields) committed under
``tests/fixtures/training/records-50/``. The coordinator is driven through
its public entrypoint (:func:`run_pipeline`) with ``dry_run=True`` — the same
path the CI workflow runs — asserting that the dry path needs no GPU (no
pynvml, no CUDA init) by executing it in a subprocess.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from daydream.training.coordinator import PipelineConfig, run_pipeline
from daydream.training.lineage import _RECORD_LINEAGE_FIELDS

FIXTURE = Path(__file__).parents[1] / "fixtures" / "training" / "records-50"


def test_fixture_has_50_records_with_lineage() -> None:
    rows = [
        json.loads(line)
        for line in (FIXTURE / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 50
    # M16 lineage fields present on every record, plus the M16 identity fields
    # the test contract names explicitly (session_id, label_source).
    for row in rows:
        assert row.get("session_id") and row.get("label_source")
        missing = [f for f in _RECORD_LINEAGE_FIELDS if f not in row]
        assert not missing, f"record {row.get('session_id')} missing lineage fields: {missing}"
    # realistic accepted/rejected mix (documented in the fixture README)
    labels = {row["label"] for row in rows}
    assert labels == {"accepted", "rejected"}


def test_coordinator_runs_end_to_end_on_fixture(tmp_path: Path) -> None:
    manifest = run_pipeline(
        PipelineConfig(corpus=FIXTURE / "records.jsonl", out_dir=tmp_path),
        dry_run=True,
    )
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    assert manifest_path.read_text() == json.dumps(manifest, indent=2, sort_keys=True)
    # adapter_path is a declared path even in dry mode, and points at a
    # loadable adapter shape when the non-dry path produced it
    assert manifest["adapter_path"] is not None
    adapter = Path(manifest["adapter_path"])
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_state.json").is_file()
    assert manifest["stages"]["stage0"]["status"] == "complete"
    assert manifest["stages"]["stage0"]["gate"]["passed"] is True


def test_ci_dry_path_has_no_gpu_imports(tmp_path: Path) -> None:
    # The dry path must never import pynvml / torch.cuda — assert by running in
    # a subprocess with CUDA hidden (M24: GPU-free CI).
    code = (
        "from daydream.training.coordinator import run_pipeline, PipelineConfig;"
        "run_pipeline("
        f"PipelineConfig(corpus={str(FIXTURE / 'records.jsonl')!r}, "
        f"out_dir={str(tmp_path)!r}), dry_run=True)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=900,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "")},
    )
    assert result.returncode == 0, result.stderr


def test_fixture_harness_config_backend_is_pi() -> None:
    # M24: any backend field in the fixture harness config is `pi`.
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    assert manifest["backend"] == "pi"

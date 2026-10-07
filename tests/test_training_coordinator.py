"""PipelineConfig accepts only frozen projection input. Stage integration lives in
test_training_coordinator_projection.py.
"""

from pathlib import Path
from typing import Any

from daydream.training.coordinator import PipelineConfig


def test_cli_verb_wired(cli_runner: Any) -> None:
    r = cli_runner.invoke(["train", "--help"])
    assert r.exit_code == 0

def test_pipeline_config_projection_only() -> None:

    cfg = PipelineConfig(out_dir=Path("/tmp/x"), projection=Path("/tmp/proj"))
    assert cfg.projection == Path("/tmp/proj")

def test_train_cli_rejects_legacy_corpus_flag(cli_runner: Any) -> None:
    r = cli_runner.invoke(["train", "--corpus", "x.jsonl", "--out", str(Path("/tmp/train-out"))])
    assert r.exit_code != 0

"""PipelineConfig accepts only frozen projection input. Stage integration lives in
test_training_coordinator_projection.py.
"""

from pathlib import Path
from typing import Any


def test_train_cli_rejects_legacy_corpus_flag(cli_runner: Any) -> None:
    r = cli_runner.invoke(["train", "--corpus", "x.jsonl", "--out", str(Path("/tmp/train-out"))])
    assert r.exit_code != 0

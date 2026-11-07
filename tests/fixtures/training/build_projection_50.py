"""Build 50 schema-3 records from observation-v3 evidence through the offline projector."""
from __future__ import annotations

from pathlib import Path

from daydream.dataset import LocalRecordStore
from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.harness.record_projection import add_projection_run, projection_config

SALT = "issue-1081-fixture-salt"
HOLDOUT_RATE = 0.2
VAL_RATE = 0.2


def build_projection_50(tmp_path: Path) -> Path:
    store = LocalRecordStore(tmp_path / "projection-fixture" / "records")
    for index in range(23):
        add_projection_run(store, run_id=f"sess-gold-{index:02d}", dispositions=("accepted", "rejected"),
                           repo_slug=f"owner/repo-{index % 3}")
    for index in range(2):
        add_projection_run(store, run_id=f"sess-amb-{index}", dispositions=("ambiguous",))
    config = projection_config(store, tmp_path / "projection-fixture", salt=SALT, holdout_rate=HOLDOUT_RATE,
                               val_rate=VAL_RATE, emit_process_traces=True)
    result = build_frozen_corpus(config)
    assert result["total"] == 50
    return config.out_dir

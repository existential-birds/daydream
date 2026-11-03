"""M8/M10 gates: real-path CLI smoke over the committed fixture, replay determinism, no-default-mutation."""

import json
from pathlib import Path

from tests.harness.scripts import cli_main


def test_calibration_consumes_canonical_record_projection_and_preserves_reward_policy(tmp_path: Path) -> None:
    from daydream.training.corpus_projection.projector import build_frozen_corpus
    from daydream.training.reward import DEFAULT_WEIGHTS
    from tests.harness.record_projection import projection_config, seed_projection_store

    store = seed_projection_store(tmp_path, dispositions=("accepted", "rejected"))
    config = projection_config(store, tmp_path)
    build_frozen_corpus(config)
    rows = [json.loads(line) for line in (config.out_dir / "corpus.jsonl").read_text().splitlines()]
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({row["record_id"]: {"accepted": row["disposition"] == "accepted"} for row in rows}))
    breakdowns = tmp_path / "breakdowns.json"
    breakdowns.write_text(json.dumps({row["record_id"]: {"w_fp": 0.55 if row["disposition"] == "accepted" else 0.45}
                                     for row in rows}))
    before = repr(DEFAULT_WEIGHTS)
    out = tmp_path / "calibration"
    assert cli_main(["corpus", "calibrate-reward", "--corpus-dir", str(config.out_dir), "--gold-labels", str(gold),
                     "--breakdowns", str(breakdowns), "--run-id", "canonical-records", "--seed", "42",
                     "--candidate", "w_fp=0.1,0.2,0.3", "--bootstrap-resamples", "20", "--out", str(out)]) == 0
    artifact = json.loads((out / "calibration.json").read_text())
    assert artifact["record_count"] == 2
    assert artifact["metrics"]["class_balance"] == {"accepted": 1, "rejected": 1}
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert artifact["lineage"]["content_digests"] == lineage["content_digests"]
    assert repr(DEFAULT_WEIGHTS) == before

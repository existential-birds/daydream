"""Byte-parity + metric-equivalence + separate-filesystem isolation for the Harbor verifier assets.

Golden gate: the ``templates/tests/verifier_core.py`` copy must stay
byte-identical (SHA-256) to the in-repo source so future edits to the source
fail loudly. ``templates/metric.py``'s inlined aggregation must equal
``verifier_core.aggregate_metrics`` field-for-field on the same rows.
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verifier_core_template_is_byte_identical_to_source() -> None:
    source = REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py"
    copy = (
        REPO
        / "daydream"
        / "benchmark"
        / "harbor"
        / "templates"
        / "tests"
        / "verifier_core.py"
    )
    assert copy.exists()
    assert _sha256(copy) == _sha256(source)


def test_metric_entry_aggregates_identically_to_verifier_core(sr_metric, tmp_path, monkeypatch) -> None:
    rows = [
        {
            "reward": 0.8,
            "tp": 2,
            "fp": 0,
            "fn": 1,
            "precision": 1.0,
            "recall": 0.6667,
            "f1": 0.8,
            "gold_count": 3,
            "candidate_count": 2,
            "clean_task": 0,
            "clean_pass": 0,
            "verifier_error": 0,
        },
        None,  # failed task (null row)
        {
            "reward": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "gold_count": 0,
            "candidate_count": 0,
            "clean_task": 1,
            "clean_pass": 1,
            "verifier_error": 0,
        },
    ]
    lp = tmp_path / "rewards.jsonl"
    lp.write_text(
        "\n".join(json.dumps(r) if r is not None else "null" for r in rows) + "\n"
    )

    from daydream.benchmark.harbor import verifier_core as vc

    expected = vc.aggregate_metrics(rows)

    result = sr_metric.aggregate_rewards_file(str(lp))
    assert result == expected  # same aggregation contract on the same rows
    assert result["failed_task_count"] == 1 and result["task_count"] == 3
    assert result["mean_task_score"] == (0.8 + 0.0 + 1.0) / 3

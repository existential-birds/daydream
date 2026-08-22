"""Byte-parity + metric-equivalence + separate-filesystem isolation for the Harbor verifier assets.

Golden gate: the ``templates/tests/verifier_core.py`` copy must stay
byte-identical (SHA-256) to the in-repo source so future edits to the source
fail loudly. ``templates/metric.py``'s inlined aggregation must equal
``verifier_core.aggregate_metrics`` field-for-field on the same rows.
"""

import hashlib
import json
import subprocess
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


def test_metric_entry_aggregates_as_identically_to_verifier_core(sr_metric, tmp_path, monkeypatch) -> None:
    rows: list[dict[str, object] | None] = [
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
    assert result == expected                          # same aggregation contract
    assert result["task_count"] == 3 and result["scored_task_count"] == 2
    assert result["infra_error_task_count"] == 1 and "failed_task_count" not in result
    assert result["mean_task_score"] == (0.8 + 1.0) / 2


def test_metric_subprocess_runs_with_harbor_args_and_writes_output(tmp_path) -> None:
    from daydream.benchmark.harbor import build

    metric_path = tmp_path / "metric.py"
    metric_path.write_bytes(build.render_metric())
    inp = tmp_path / "rewards.jsonl"
    inp.write_text(
        '{"reward":0.8,"tp":2,"fp":0,"fn":1}\n'
        'null\n'
        '{"reward":1.0,"tp":0,"fp":0,"fn":0,"clean_task":1}\n'
    )
    out = tmp_path / "out" / "metric.json"
    proc = subprocess.run(
        ["uv", "run", "--script", str(metric_path), "-i", str(inp), "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["task_count"] == 3  # attempted = all rows (stable across old/new aggregation)
    assert not (out.parent / ".metric.json.tmp").exists()  # atomic write leaves no temp leftover


def test_metric_subprocess_unscored_rows_not_turned_into_zeros(tmp_path) -> None:
    from daydream.benchmark.harbor import build

    metric_path = tmp_path / "metric.py"
    metric_path.write_bytes(build.render_metric())
    inp = tmp_path / "rewards.jsonl"
    inp.write_text(
        '{"reward":0.8,"tp":2,"fp":0,"fn":1}\n'
        'null\n'
        '{"reward":1.0,"tp":0,"fp":0,"fn":0,"clean_task":1}\n'
        '{"reward":0.0,"tp":0,"fp":5,"fn":5,"verifier_error":1}\n')
    out = tmp_path / "metric.json"
    proc = subprocess.run(
        ["uv", "run", "--script", str(metric_path), "-i", str(inp), "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    m = json.loads(out.read_text())
    assert m["task_count"] == 4
    assert m["scored_task_count"] == 2 and m["infra_error_task_count"] == 2
    assert (m["total_tp"], m["total_fp"], m["total_fn"]) == (2, 0, 1)  # unscored rows contribute nothing
    assert abs(m["mean_task_score"] - 0.9) < 1e-9                       # (0.8+1.0)/2, never over 4
    assert m["micro_precision"] == 1.0 and m["micro_recall"] == 2.0 / 3.0

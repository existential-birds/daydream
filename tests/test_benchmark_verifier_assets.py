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
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_metric_subprocess(tmp_path: Path, rows: str, out: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Compile the metric entrypoint, feed it ``rows`` JSONL, and run it via ``uv run --script``.

    Keeps the subprocess contract/flag surface (``-i``/``-o``, timeout, returncode
    assertion) in one place; returns ``(out, parsed_result)``.
    """
    from daydream.benchmark.harbor import build

    metric_path = tmp_path / "metric.py"
    metric_path.write_bytes(build.render_metric())
    inp = tmp_path / "rewards.jsonl"
    inp.write_text(rows)
    if out is None:
        out = tmp_path / "metric.json"
    proc = subprocess.run(
        ["uv", "run", "--script", str(metric_path), "-i", str(inp), "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return out, json.loads(out.read_text())


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


def test_metric_entry_aggregates_as_identically_to_verifier_core(
    sr_metric: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "location_exact": 0,
            "location_near": 0,
            "location_file": 1,
            "location_miss": 0,
            "location_credit": 0.0,
            "location_present": 1,
            "severity_exact": 1,
            "severity_within_1": 1,
            "severity_mean_distance": 0.0,
            "severity_credit": 1.0,
            "severity_present": 1,
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
            "location_exact": 0,
            "location_near": 0,
            "location_file": 0,
            "location_miss": 0,
            "location_credit": 0.0,
            "location_present": 0,
            "severity_exact": 0,
            "severity_within_1": 0,
            "severity_mean_distance": 0.0,
            "severity_credit": 0.0,
            "severity_present": 0,
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


def test_metric_subprocess_runs_with_harbor_args_and_writes_output(tmp_path: Path) -> None:
    out, result = _run_metric_subprocess(
        tmp_path,
        '{"reward":0.8,"tp":2,"fp":0,"fn":1}\n'
        'null\n'
        '{"reward":1.0,"tp":0,"fp":0,"fn":0,"clean_task":1}\n',
        out=tmp_path / "out" / "metric.json",
    )
    assert result["task_count"] == 3  # attempted = all rows (stable across old/new aggregation)
    # atomic write leaves no temp leftover: metric.py names its temp
    # ".{out.name}.{pid}.tmp", so glob the pid-suffixed pattern the write
    # actually uses (the old ".metric.json.tmp" literal could never match).
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))


def test_metric_subprocess_unscored_rows_not_turned_into_zeros(tmp_path: Path) -> None:
    _, m = _run_metric_subprocess(
        tmp_path,
        '{"reward":0.8,"tp":2,"fp":0,"fn":1}\n'
        'null\n'
        '{"reward":1.0,"tp":0,"fp":0,"fn":0,"clean_task":1}\n'
        '{"reward":0.0,"tp":0,"fp":5,"fn":5,"verifier_error":1}\n',
    )
    assert m["task_count"] == 4
    assert m["scored_task_count"] == 2 and m["infra_error_task_count"] == 2
    assert (m["total_tp"], m["total_fp"], m["total_fn"]) == (2, 0, 1)  # unscored rows contribute nothing
    assert abs(m["mean_task_score"] - 0.9) < 1e-9                       # (0.8+1.0)/2, never over 4
    assert m["micro_precision"] == 1.0 and m["micro_recall"] == 2.0 / 3.0


def test_range_distance_cannot_drift_from_hunk_index() -> None:
    """The isolated verifier must preserve the shared range-distance contract."""
    import ast
    import sys

    from daydream import hunk_index
    from daydream.benchmark.harbor import verifier_core as vc

    # These cases cover the inclusive interior, both sides of a range, and a
    # one-line range without using either implementation as the oracle.
    for line, start, end, expected in (
        (15, 10, 20, 0),
        (9, 10, 20, 1),
        (21, 10, 20, 1),
        (10, 10, 10, 0),
        (9, 10, 10, 1),
        (11, 10, 10, 1),
    ):
        assert hunk_index.range_distance(line, start, end) == expected
        assert vc._range_distance(line, start, end) == expected

    source = (
        REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_roots <= sys.stdlib_module_names | {"__future__"}


def test_location_tolerance_meets_floor() -> None:
    from daydream.benchmark.harbor import verifier_core as vc
    assert vc.LOCATION_TOLERANCE >= 3  # below 3 measures the snapper, not the reviewer (R2)


def test_metric_helper_functions_cannot_drift_from_verifier_core() -> None:
    """The metric.py template's helper functions must stay byte-identical to verifier_core.

    ``render_metric()`` splices only ``aggregate_metrics`` into the compiled
    metric; the ``_as_int/_as_float/_f1/_axis_aggregates`` helpers it
    resolves at runtime are the template-local copies. They must not drift
    from the in-repo verifier_core copies (which the corpus pool uses), or
    the compiled metric and the pool would disagree on row coercion / f1 /
    axis pooling. This gate makes any drift fail loudly instead of silently
    diverging.
    """
    import inspect

    from daydream.benchmark.harbor import verifier_core as vc

    tmpl = (REPO / "daydream" / "benchmark" / "harbor" / "templates" / "metric.py").read_text(encoding="utf-8")
    for name in ("_as_int", "_as_float", "_f1", "_axis_aggregates"):
        src = inspect.getsource(getattr(vc, name))
        assert src in tmpl, f"metric.py template's {name} drifted from verifier_core.py"

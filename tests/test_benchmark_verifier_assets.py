"""Behavioral contract tests for the Harbor verifier scoring core and metric.

``verifier_core.aggregate_metrics`` is the canonical scorer: these tests pin
its observable aggregation contract (pooling, zero-denominator semantics,
absent-axis handling) plus the compiled metric's subprocess behavior, so
later canonicalization work must preserve behavior rather than source text.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import verifier_core

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
    shutil.copy(REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py", tmp_path / "verifier_core.py")
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


def test_copy_assets_emits_canonical_verifier_core_bytes(tmp_path: Path) -> None:
    from daydream.benchmark.harbor.build import _copy_assets

    out = dict(_copy_assets(tmp_path))
    source = REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py"
    deployed = tmp_path / "tests" / "verifier_core.py"
    assert deployed.exists()
    assert out["tests/verifier_core.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not (REPO / "daydream" / "benchmark" / "harbor" / "templates" / "tests"
                / "verifier_core.py").exists()



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
    """The verifier's private range_distance copy must stay semantically
    byte-locked to daydream.hunk_index.range_distance (issue #971 R8)."""
    import inspect

    from daydream import hunk_index
    from daydream.benchmark.harbor import verifier_core as vc
    host = inspect.getsource(hunk_index.range_distance)
    # the copy is private and stdlib-only; assert the arithmetic body matches
    for line in (
        "if start <= line <= end:",
        "return 0",
        "if line < start:",
        "return start - line",
        "return line - end",
    ):
        assert line in host and line in inspect.getsource(vc._range_distance)
    assert "import daydream" not in inspect.getsource(vc)


def test_location_tolerance_meets_floor() -> None:
    from daydream.benchmark.harbor import verifier_core as vc
    assert vc.LOCATION_TOLERANCE >= 3  # below 3 measures the snapper, not the reviewer (R2)


def test_render_metric_loads_colocated_canonical_and_matches_host(
    tmp_path: Path,
) -> None:
    """The rendered metric must obtain aggregate_metrics by loading the colocated
    canonical verifier_core.py from the stage root — not from a spliced body."""
    from daydream.benchmark.harbor import build, verifier_core

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "metric.py").write_bytes(build.render_metric())
    shutil.copy(
        Path(verifier_core.__file__),
        stage / "verifier_core.py",
    )
    import types

    mod = types.ModuleType("metric")
    mod.__dict__["__file__"] = str(stage / "metric.py")  # uv run --script provides this
    exec(compile((stage / "metric.py").read_text(), "metric.py", "exec"), mod.__dict__)
    rows: list[dict[str, object] | None] = [{"reward": 0.5, "tp": 1, "fp": 0, "fn": 1, "verifier_error": 0}, None]
    assert mod.aggregate_metrics(rows) == verifier_core.aggregate_metrics(rows)


def test_metric_template_has_no_helper_duplicates_or_markers() -> None:
    from daydream.benchmark.harbor.package import template_text

    text = template_text("metric.py")
    for banned in (
        "__AGGREGATION_BODY",
        "_axis_aggregates",
        "def _f1",
        "def _as_int",
        "def _as_float",
    ):
        assert banned not in text


def test_aggregate_metrics_pools_tp_fp_fn_and_zero_denominators_are_one() -> None:
    rows: list[dict[str, object] | None] = [
        {"reward": 0.8, "tp": 2, "fp": 1, "fn": 0, "precision": 2 / 3, "recall": 1.0,
         "f1": 0.8, "gold_count": 2, "candidate_count": 3, "clean_task": 0, "clean_pass": 0,
         "verifier_error": 0},
        {"reward": 1.0, "tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
         "f1": 1.0, "gold_count": 0, "candidate_count": 0, "clean_task": 1, "clean_pass": 1,
         "verifier_error": 0},
        None,
    ]
    m = verifier_core.aggregate_metrics(rows)
    assert m["total_tp"] == 2 and m["total_fp"] == 1 and m["total_fn"] == 0
    assert m["task_count"] == 3 and m["scored_task_count"] == 2
    assert m["infra_error_task_count"] == 1


def test_aggregate_metrics_empty_rows_score_one() -> None:
    m = verifier_core.aggregate_metrics([])
    assert m["task_count"] == 0
    assert m["micro_f1"] == 1.0 and m["mean_task_score"] == 1.0 and m["clean_accuracy"] == 1.0


def test_aggregate_metrics_zero_scored_rows_headline_rates_are_one() -> None:
    rows = [{"verifier_error": 1, "reward": 0.0, "tp": 0, "fp": 0, "fn": 0}]
    m = verifier_core.aggregate_metrics(rows)  # type: ignore[arg-type]
    assert m["micro_precision"] == 1.0 and m["micro_recall"] == 1.0 and m["micro_f1"] == 1.0


def test_aggregate_metrics_axis_absent_is_zero_pairs_not_raise() -> None:
    rows = [{"reward": 1.0, "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
             "f1": 1.0, "clean_task": 1, "clean_pass": 1, "verifier_error": 0}]
    m = verifier_core.aggregate_metrics(rows)  # type: ignore[arg-type]
    assert m["location_pairs_scored"] == 0 and m["severity_pairs_scored"] == 0
    assert m["location_exact_rate"] == 0.0  # absent axis = missing signal, not 1.0

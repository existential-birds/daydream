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


@pytest.fixture()
def sr_metric(tmp_path: Path) -> Any:
    """Stage the rendered metric next to a colocated canonical ``verifier_core.py``."""
    from daydream.benchmark.harbor import build, verifier_core

    (tmp_path / "metric.py").write_bytes(build.render_metric())
    shutil.copy(Path(verifier_core.__file__), tmp_path / "verifier_core.py")
    return tmp_path / "metric.py"


def _reward_row(*, tp: int, fp: int, fn: int, reward: float, clean: bool) -> dict[str, object]:
    """A representative scored row; shape mirrors ``_reward`` in tests/test_benchmark_objective.py."""
    return {
        "reward": reward,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": (tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 1.0,
        "f1": 0.0,
        "clean_task": 1 if clean else 0,
        "clean_pass": 1 if clean else 0,
        "verifier_error": 0,
    }


def test_rendered_metric_matches_host_on_representative_rows(
    sr_metric: Any, tmp_path: Path
) -> None:
    """The rendered metric, run via ``uv run --script`` over JSONL that mixes
    clean scored rows with malformed/null/shape-wrong rows, must produce exactly
    what the canonical ``verifier_core.aggregate_metrics`` computes in-process."""
    from daydream.benchmark.harbor import verifier_core

    clean = _reward_row(tp=2, fp=0, fn=1, reward=0.8, clean=False)
    rows = "\n".join([json.dumps(clean), "null", "not-json", "[]", json.dumps(clean)])
    _, result = _run_metric_subprocess(tmp_path, rows)
    flat: list[dict[str, object] | None] = [clean, None, None, None, clean]
    expected = verifier_core.aggregate_metrics(flat)
    assert result == expected
    assert result["infra_error_task_count"] == 3


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


def test_deployed_scoring_surfaces_are_stdlib_only_and_daydream_free() -> None:
    """Both deployed scoring surfaces — the rendered metric and the colocated
    canonical copy — must stay stdlib-only and free of any ``daydream`` import."""
    from daydream.benchmark.harbor import build

    metric_text = build.render_metric().decode("utf-8")
    canonical = (REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py").read_text()
    for surface_name, text in (("metric", metric_text), ("canonical", canonical)):
        assert "import daydream" not in text, surface_name
        assert "from daydream" not in text, surface_name
        assert "pydantic" not in text, surface_name


def test_deployed_canonical_copy_exposes_score_review_surface(tmp_path: Path) -> None:
    """``score_review.py`` consumes (``VerifierError``, ``Verdict``, ``validate_exact_keys``,
    ``validate_candidate_artifact``, ``score_review``, ``derive_candidate_id``) from the
    deployed canonical file; the copy emitted by ``_copy_assets`` must define each."""
    from daydream.benchmark.harbor.build import _copy_assets

    out = dict(_copy_assets(tmp_path))
    deployed = tmp_path / "tests" / "verifier_core.py"
    assert deployed.exists()
    assert out["tests/verifier_core.py"] == _sha256(
        REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py"
    )

    import sys
    import types

    # dataclasses resolves cls.__module__ via sys.modules at decoration time,
    # so register the namespace as a module before exec (P1 spike pitfall).
    mod = types.ModuleType("deployed_verifier_core")
    ns = mod.__dict__
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(deployed.read_text(), "verifier_core.py", "exec"), ns)  # noqa: S102
    finally:
        del sys.modules[mod.__name__]
    for name in (
        "VerifierError",
        "Verdict",
        "validate_exact_keys",
        "validate_candidate_artifact",
        "score_review",
        "derive_candidate_id",
    ):
        assert name in ns and ns[name] is not None, name


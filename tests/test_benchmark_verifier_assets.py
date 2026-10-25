"""Behavioral contract tests for the Harbor verifier scoring core and metric.

``verifier_core.aggregate_metrics`` is the canonical scorer: these tests pin
its observable aggregation contract (pooling, zero-denominator semantics,
absent-axis handling) plus the compiled metric's subprocess behavior, so
later canonicalization work must preserve behavior rather than source text.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from daydream.benchmark.harbor import build, candidate, verifier_core, verifier_core as vc
from daydream.benchmark.harbor.build import _copy_assets
from daydream.hunk_index import range_distance as hunk_range_distance

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _copy_assets(tmp_path)
    path = tmp_path / "tests" / "verifier_core.py"
    spec = importlib.util.spec_from_file_location("copied_validation_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _finding_content() -> dict[str, Any]:
    return {"title": "Cache key omits tenant", "body": "Two tenants can receive the same cached response.",
        "severity": "high", "path": "src/cache.py", "start_line": 3, "end_line": 5,
    }


@pytest.mark.parametrize("locationless", [False, True])
def test_public_content_parser_preserves_valid_values(locationless: bool) -> None:
    raw = _finding_content()
    raw.update(title=" t " + "x" * 497, body="é" * 4096, severity=None)
    if locationless:
        raw.update(path=None, start_line=None, end_line=None)
    original = dict(raw)
    parsed = verifier_core.parse_finding_content(raw)
    assert parsed == original and parsed is not raw
    assert raw == original


@pytest.mark.parametrize(
    "changes",
    [
        {"title": " "},
        {"title": "x" * 501},
        {"title": "a\x00b"},
        {"body": ""},
        {"body": "é" * 4096 + "x"},
        {"body": "a\x00b"},
        {"severity": "critical"},
        {"path": ""},
        {"path": "/src/cache.py"},
        {"path": "src/../cache.py"},
        {"path": "src/ca\x00che.py"},
        {"start_line": True},
        {"start_line": 0},
        {"start_line": "3"},
        {"end_line": "5"},
        {"end_line": 2},
        {"path": None},
    ],
)
def test_public_content_parser_rejects_invalid_content(changes: dict[str, Any]) -> None:
    with pytest.raises(verifier_core.VerifierError):
        verifier_core.parse_finding_content(_finding_content() | changes)


@pytest.mark.parametrize("shape", ["non-object", "missing-severity", "unknown-key"])
def test_public_content_parser_requires_exact_keys(shape: str) -> None:
    raw: Any = _finding_content()
    if shape == "non-object":
        raw = []
    elif shape == "missing-severity":
        del raw["severity"]
    else:
        raw["finding_id"] = "a" * 64
    with pytest.raises(verifier_core.VerifierError):
        verifier_core.parse_finding_content(raw)


def test_host_built_artifacts_match_copied_verifier(copied_verifier: ModuleType) -> None:
    key = "case-validation-parity"
    merged = {"file": "src/cache.py", "line": 3, "description": "Tenant omitted",
              "rationale": "Responses cross tenants", "severity": "high"}
    candidate_artifact = candidate.build_candidate_artifact(
        key, candidate.build_candidate_findings([merged, merged], case_id=key)
    )
    findings: list[dict[str, Any]] = [
        {"finding_id": "a" * 64, "title": "Cache key omits tenant", "body": "Wrong tenant",
         "severity": "high", "location": {"path": "src/cache.py", "start_line": 3, "end_line": 5}},
        {"finding_id": "b" * 64, "title": "Missing access check", "body": "No authorization",
         "severity": None, "location": None},
    ]
    gold = build.build_gold_list(findings, key=key)
    oracle = build.build_oracle_artifact(key, findings)
    for artifact in (candidate_artifact, oracle, candidate.build_candidate_artifact(key, [])):
        expected = [asdict(f) for f in verifier_core.validate_candidate_artifact(artifact)]
        assert [asdict(f) for f in copied_verifier.validate_candidate_artifact(artifact)] == expected
        assert [f["candidate_id"] for f in artifact["findings"]] == [f["candidate_id"] for f in expected]
    assert candidate_artifact["findings"][0]["candidate_id"] != candidate_artifact["findings"][1]["candidate_id"]
    other_key = "case-another-review"
    second_artifact = candidate.build_candidate_artifact(
        other_key, candidate.build_candidate_findings([merged, merged], case_id=other_key)
    )
    copied_verifier.validate_candidate_artifact(second_artifact)
    second_gold = build.build_gold_list(findings, key=other_key)
    copied_verifier.validate_gold_set(second_gold, case_id=other_key)
    assert {finding["finding_id"] for finding in gold}.isdisjoint(finding["finding_id"] for finding in second_gold)
    assert {f["candidate_id"] for f in candidate_artifact["findings"]}.isdisjoint(
        f["candidate_id"] for f in second_artifact["findings"]
    )
    for entries in (gold, []):
        expected = [asdict(f) for f in verifier_core.validate_gold_set(entries, case_id=key)]
        assert [asdict(f) for f in copied_verifier.validate_gold_set(entries, case_id=key)] == expected

    # Mutate host-built artifacts at the input boundary: both deployed and host
    # validators must reject them with their own public exception category.
    located_gold = next(i for i, finding in enumerate(gold) if finding["path"] is not None)
    for changes in ({"body": "é" * 4096 + "x"}, {"path": None}, {"severity": "critical"}):
        bad_candidate = deepcopy(candidate_artifact)
        bad_candidate["findings"][0].update(changes)
        bad_gold = deepcopy(gold)
        bad_gold[located_gold].update(changes)
        for module in (verifier_core, copied_verifier):
            with pytest.raises(module.VerifierError):
                module.validate_candidate_artifact(bad_candidate)
            with pytest.raises(module.VerifierError):
                module.validate_gold_set(bad_gold, case_id=key)

    for module in (verifier_core, copied_verifier):
        wrong_id = deepcopy(candidate_artifact)
        wrong_id["findings"][0]["candidate_id"] = "0" * 64
        with pytest.raises(module.VerifierError, match="derived id"):
            module.validate_candidate_artifact(wrong_id)
        wrong_gold_id = deepcopy(gold)
        wrong_gold_id[0]["finding_id"] = "0" * 64
        with pytest.raises(module.VerifierError, match="canonical digest"):
            module.validate_gold_set(wrong_gold_id, case_id=key)
        too_many = deepcopy(candidate_artifact)
        too_many["findings"] = candidate_artifact["findings"] * 51
        with pytest.raises(module.VerifierError, match="100 findings"):
            module.validate_candidate_artifact(too_many)
        with pytest.raises(module.VerifierError, match="50 findings"):
            module.validate_gold_set(gold * 26, case_id=key)
        too_large = deepcopy(candidate_artifact)
        too_large["findings"][0]["body"] = "x" * module.MAX_ARTIFACT_BYTES
        with pytest.raises(module.VerifierError, match="1 MiB"):
            module.validate_candidate_artifact(too_large)


_MISSING = object()  # sentinel: bare "verifier_core" absent from sys.modules


def _run_metric_subprocess(tmp_path: Path, rows: str, out: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Compile the metric entrypoint, feed it ``rows`` JSONL, and run it via ``uv run --script``.

    Keeps the subprocess contract/flag surface (``-i``/``-o``, timeout, returncode
    assertion) in one place; returns ``(out, parsed_result)``.
    """

    metric_path = tmp_path / "metric.py"
    metric_path.write_bytes(build.render_metric())
    shutil.copy(REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py", tmp_path / "verifier_core.py")
    inp = tmp_path / "rewards.jsonl"
    inp.write_text(rows)
    if out is None:
        out = tmp_path / "metric.json"
    proc = subprocess.run(["uv", "run", "--script", str(metric_path), "-i", str(inp), "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return out, json.loads(out.read_text())


def test_copy_assets_emits_canonical_verifier_core_bytes(tmp_path: Path) -> None:
    out = dict(_copy_assets(tmp_path))
    source = REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py"
    deployed = tmp_path / "tests" / "verifier_core.py"
    assert deployed.read_bytes() == source.read_bytes()
    assert out["tests/verifier_core.py"] == hashlib.sha256(deployed.read_bytes()).hexdigest()
    # The deployed scorer must import and execute with site-packages disabled.
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import runpy,sys; m=runpy.run_path(sys.argv[1]); "
            "artifact={'schema_version':1,'case_id':'x','base_ref':'b','head_ref':'h','findings':[]}; "
            "print(m['score_review']([], artifact, []).reward)",
            str(deployed),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1.0"


def test_location_tolerance_meets_floor() -> None:
    assert vc.LOCATION_TOLERANCE >= 3  # below 3 measures the snapper, not the reviewer (R2)


def test_range_distance_cannot_drift_from_hunk_index() -> None:
    cases: list[tuple[int, int, int]] = [
        (1, 5, 10),   # below the range -> distance to the start boundary
        (4, 5, 10),   # one line below start
        (5, 5, 10),   # on the start boundary -> inside
        (7, 5, 10),   # inside the range
        (10, 5, 10),  # on the end boundary -> inside
        (11, 5, 10),  # one line above end
        (12, 5, 10),  # above the range -> distance to the end boundary
        (0, 0, 0),    # degenerate single-line range, on the line
        (2, 0, 0),    # degenerate range, below
        (0, 3, 3),    # degenerate range, above
    ]
    for line, start, end in cases:
        assert vc._range_distance(line, start, end) == hunk_range_distance(line, start, end)


def test_render_metric_loads_colocated_canonical_and_matches_host(tmp_path: Path,) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "metric.py").write_bytes(build.render_metric())
    shutil.copy(Path(verifier_core.__file__), stage / "verifier_core.py")

    prior_verifier_core = sys.modules.get("verifier_core", _MISSING)
    mod = types.ModuleType("metric")
    mod.__dict__["__file__"] = str(stage / "metric.py")  # uv run --script provides this
    exec(compile((stage / "metric.py").read_text(), "metric.py", "exec"), mod.__dict__)
    # Loading the canonical scorer must not leave the exec'd module registered
    # under the bare name: a later `import verifier_core` in the same process
    # would otherwise silently resolve to this compiled copy (import hygiene).
    assert sys.modules.get("verifier_core", _MISSING) is prior_verifier_core
    rows: list[dict[str, object] | None] = [{"reward": 0.5, "tp": 1, "fp": 0, "fn": 1, "verifier_error": 0}, None]
    assert mod.aggregate_metrics(rows) == verifier_core.aggregate_metrics(rows)


def test_aggregate_metrics_empty_rows_score_one() -> None:
    m = verifier_core.aggregate_metrics([])
    assert m["task_count"] == 0
    assert (m["micro_precision"], m["micro_recall"], m["micro_f1"]) == (1.0, 1.0, 1.0)
    assert m["mean_task_score"] == 1.0 and m["clean_accuracy"] == 1.0
    for tier in ("exact", "near", "file", "miss"):
        assert m[f"location_{tier}_rate"] == 0.0
    assert m["location_pairs_scored"] == 0 and m["severity_pairs_scored"] == 0
    assert m["severity_exact_rate"] == 0.0
    assert m["severity_within_1_rate"] == 0.0
    assert m["severity_mean_distance"] == 0.0 and m["severity_credit"] == 0.0
    assert m["total_location_exact"] == 0 and m["total_severity_exact"] == 0


def test_aggregate_metrics_zero_scored_rows_headline_rates_are_one() -> None:
    rows: list[dict[str, object] | None] = [None, {"verifier_error": 1, "reward": 0.0, "tp": 0, "fp": 0, "fn": 0}]
    m = verifier_core.aggregate_metrics(rows)
    assert (m["scored_task_count"], m["infra_error_task_count"], m["mean_task_score"]) == (0, 2, 1.0)
    assert m["micro_precision"] == 1.0 and m["micro_recall"] == 1.0 and m["micro_f1"] == 1.0


def test_metric_subprocess_pools_scored_rows_and_retains_infrastructure_failures(tmp_path: Path) -> None:
    scored: dict[str, object] = {"reward": 0.8, "tp": 2, "fp": 0, "fn": 1, "verifier_error": 0}
    clean: dict[str, object] = {"reward": 1.0, "tp": 0, "fp": 0, "fn": 0, "clean_task": 1, "clean_pass": 1}
    failed: dict[str, object] = {"reward": 0.0, "tp": 50, "fp": 50, "fn": 50, "verifier_error": 1}
    dirty_clean: dict[str, object] = {
        "reward": 0.0, "tp": 0, "fp": 2, "fn": 0, "clean_task": 1, "clean_pass": 0, "verifier_error": 0,
    }
    rows = "\n".join([
        json.dumps(scored), "null", "not-json", "[]", json.dumps(clean), json.dumps(dirty_clean), json.dumps(failed),
    ])
    out, result = _run_metric_subprocess(tmp_path, rows, out=tmp_path / "out" / "metric.json")
    assert result == verifier_core.aggregate_metrics([scored, None, None, None, clean, dirty_clean, failed])
    assert (result["task_count"], result["scored_task_count"], result["infra_error_task_count"]) == (7, 3, 4)
    assert (result["total_tp"], result["total_fp"], result["total_fn"]) == (2, 2, 1)
    assert result["mean_task_score"] == 0.6 and result["micro_f1"] == pytest.approx(4 / 7)
    assert result["micro_precision"] == 0.5 and result["micro_recall"] == 2 / 3
    assert result["clean_task_count"] == 2 and result["clean_accuracy"] == 0.5
    assert result["location_pairs_scored"] == 0 and result["severity_pairs_scored"] == 0
    assert result["location_exact_rate"] == 0.0 and result["severity_credit"] == 0.0
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))

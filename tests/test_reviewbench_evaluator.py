"""Supplied evaluator integration, publication safety and independent-round reports."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.reviewbench import evaluator
from daydream.benchmark.reviewbench.contract import Case, load_cases, verify_source


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _finding() -> dict[str, Any]:
    return {
        "producer": "synthetic-native", "file": "main.py", "start_line": 1, "end_line": 1,
        "message": "The response omits the tenant key, allowing cached data to cross tenant boundaries.",
    }


def _metric_set() -> dict[str, Any]:
    return {
        "grounded_precision": 0.5, "grounded_recall": 1.0,
        "augmented_precision": 0.5, "augmented_recall": 1.0,
        "novel_tp_count": 1, "novel_tp_yield": 0.25,
        "candidate_count": 4, "golden_tp_count": 1, "matched_count": 2, "matched_tp_count": 1,
        "unmatched_count": 2, "unmatched_tp_count": 1, "unmatched_fp_count": 1, "golden_covered_count": 1,
    }


def _metrics(case: Case, *, provider: str = "chosen-provider", model: str = "chosen-model") -> dict[str, Any]:
    strata = {
        "overall": _metric_set(), "by_severity": {"high": _metric_set()}, "by_category": {"security": _metric_set()},
    }
    return {
        "macro": strata, "micro": copy.deepcopy(strata), "pr_count": 1,
        "per_pr": [{"pr_key": case.key, "metrics": copy.deepcopy(strata)}],
        "eval_config": {
            "golden_hash": "a" * 64, "classifier_model": f"{provider}/{model}",
            "classifier_prompt_hash": "b" * 64, "matcher_model": f"{provider}/{model}",
            "matcher_prompt_hash": "c" * 64,
            "evaluated_prs_hash": hashlib.sha256(case.key.encode()).hexdigest(),
        },
        "candidate_usage": {"total": {"time_in_ms": 12, "completion_tokens": None}},
    }


@pytest.fixture
def external_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Case]:
    """Replace npm with an offline process; retain real files and publication."""
    source = tmp_path / "external-supplied"
    source.mkdir()
    (source / "golden").mkdir()
    identity = {"revision": "synthetic-external-process", "hashes": {}}
    monkeypatch.setattr(evaluator, "verify_source", lambda _source: identity)
    case = Case("https://github.com/example/repo", 7, "1" * 40, "2" * 40)
    _write_json(source / "corpus" / "manifest.json", [asdict(case)])
    _write_json(source / "golden" / f"{case.key}.json", {"pr": asdict(case), "findings": []})
    candidate_dir = tmp_path / "candidate" / "round-1" / "findings"
    _write_json(candidate_dir / f"{case.key}.json", {"pr": asdict(case), "agent": "native", "findings": [_finding()]})
    _write_json(source / "payload.json", _metrics(case))
    shim = tmp_path / "bin" / "npm"
    shim.parent.mkdir()
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "root = pathlib.Path.cwd()\n"
        "(root / 'args.json').write_text(json.dumps(args))\n"
        "mode = os.environ.get('DAYDREAM_TEST_EVAL_MODE', 'success')\n"
        "out = pathlib.Path(args[args.index('--output') + 1])\n"
        "if mode == 'failure':\n"
        "    out.write_text('{\"partial\": true}\\n')\n"
        "    out.with_name('results.checkpoint.json').write_text('{\"completed\": []}\\n')\n"
        "    print('synthetic evaluator process failed', file=sys.stderr)\n"
        "    raise SystemExit(9)\n"
        "if mode == 'missing': raise SystemExit(0)\n"
        "payload = json.loads((root / 'payload.json').read_text())\n"
        "out.write_text('{bad' if mode == 'malformed' else json.dumps(payload))\n"
        "details = [{'pr_key': p['pr_key'], 'candidate_findings': []} for p in payload['per_pr']]\n"
        "if mode != 'missing-details': out.with_name('results.details.json').write_text(json.dumps(details))\n"
        "if mode == 'checkpoint': out.with_name('results.checkpoint.json').write_text('{}')\n"
        "if mode == 'mutated-candidate':\n"
        "    candidates = pathlib.Path(args[args.index('--candidate') + 1])\n"
        "    next(candidates.glob('*.json')).write_text('{}')\n"
        "if mode == 'mutated-reference':\n"
        "    golden = pathlib.Path(args[args.index('--golden') + 1])\n"
        "    next(golden.glob('*.json')).write_text('{}')\n"
        "if mode == 'mutated-manifest':\n"
        "    pathlib.Path(args[args.index('--manifest') + 1]).write_text('[]')\n"
        "if mode == 'concurrent-output':\n"
        "    target = pathlib.Path(os.environ['DAYDREAM_TEST_OUTPUT_CONFLICT'])\n"
        "    target.mkdir()\n"
        "    (target / 'retained.txt').write_text('concurrent bytes')\n"
        "print('synthetic supplied summary')\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim.parent) + os.pathsep + os.environ["PATH"])
    return source, candidate_dir, case


def test_standalone_evaluator_preserves_selected_model_and_raw_metrics(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path,
) -> None:
    """A Harbor provider/default leaking into standalone changes its experiment identity."""
    source, candidates, case = external_evaluator
    output = tmp_path / "scores" / "candidate" / "round-1"
    result = evaluator.evaluate_round(
        source, candidates, output, provider="chosen-provider", model="chosen-model",
    )
    assert evaluator.load_evaluation(source, candidates, output) == result
    assert result["metrics"] == _metrics(case)
    assert result["evaluator"]["workflow"] == "supplied-standalone"
    assert result["evaluator"]["snapshot"]["cases"] == {case.key: asdict(case)}
    assert result["evaluator"]["snapshot"]["candidate_files"] == {
        f"{case.key}.json": hashlib.sha256((candidates / f"{case.key}.json").read_bytes()).hexdigest(),
    }
    args = json.loads((source / "args.json").read_text())
    assert args[args.index("--provider") + 1] == "chosen-provider"
    assert args[args.index("--model") + 1] == "chosen-model"
    assert args[args.index("--candidate") + 1] == str(candidates)
    assert "--strict" in args and "--limit" not in args
    assert not list(output.parent.glob(".round-1-*"))


@pytest.mark.parametrize("fault", [
    "results", "details", "candidate", "manifest", "reference", "missing-provenance",
    "source", "membership", "model", "swapped-output", "swapped-candidates", "checkpoint", "old-schema",
])
def test_saved_evaluation_must_remain_bound_to_current_round_evidence(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, fault: str,
) -> None:
    """Reporting rechecks evidence independently of its validity when first published."""
    source, candidates, case = external_evaluator
    output = tmp_path / "scores" / "round-1"
    evaluator.evaluate_round(source, candidates, output, provider="chosen-provider", model="chosen-model")
    provenance = json.loads((output / "evaluator.json").read_text())
    artifacts = {
        "results": output / "results.json", "details": output / "results.details.json",
        "candidate": candidates / f"{case.key}.json", "manifest": source / "corpus" / "manifest.json",
        "reference": source / "golden" / f"{case.key}.json",
    }
    if fault in artifacts:
        path = artifacts[fault]
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "missing-provenance":
        (output / "evaluator.json").unlink()
    elif fault == "checkpoint":
        _write_json(output / "results.checkpoint.json", {})
    elif fault == "swapped-candidates":
        other_candidates = tmp_path / "other-candidate" / "round-1" / "findings"
        shutil.copytree(candidates, other_candidates)
        candidates = other_candidates
    elif fault == "swapped-output":
        other_output = tmp_path / "scores" / "other-round"
        evaluator.evaluate_round(
            source, candidates, other_output, provider="chosen-provider", model="chosen-model",
        )
        for filename in ("evaluator.json", "results.json", "results.details.json"):
            shutil.copyfile(other_output / filename, output / filename)
    else:
        target, key, replacement = {
            "source": (provenance["snapshot"]["source"], "revision", "different-source"),
            "membership": (provenance["snapshot"], "cases", {}),
            "model": (provenance, "provider", "another-provider"),
            "old-schema": (provenance, "schema_version", 1),
        }[fault]
        target[key] = replacement
        _write_json(output / "evaluator.json", provenance)
    with pytest.raises(evaluator.EvaluationError):
        evaluator.load_evaluation(source, candidates, output)


@pytest.mark.parametrize("mode", [
    "failure", "missing", "malformed", "missing-details", "checkpoint", "mutated-candidate",
    "mutated-reference", "mutated-manifest", "interrupted",
])
def test_failed_or_incomplete_evaluator_cannot_publish(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failed/interrupted judging retains its actual partial files without admitting them as a completed round."""
    source, candidates, _ = external_evaluator
    output = tmp_path / "scores" / "new-round"
    monkeypatch.setenv("DAYDREAM_TEST_EVAL_MODE", "checkpoint" if mode == "interrupted" else mode)
    if mode == "interrupted":
        read_text = Path.read_text

        def interrupt_after_judging(path: Path, *args: Any, **kwargs: Any) -> str:
            contents = read_text(path, *args, **kwargs)
            if path.name == "results.json":
                raise KeyboardInterrupt
            return contents

        monkeypatch.setattr(Path, "read_text", interrupt_after_judging)
    with pytest.raises(KeyboardInterrupt if mode == "interrupted" else evaluator.EvaluationError):
        evaluator.evaluate_round(source, candidates, output, provider="chosen-provider", model="chosen-model")
    assert not output.exists()
    retained, = output.parent.glob(".new-round-*")
    assert str(retained) in capsys.readouterr().err
    assert not (retained / "evaluator.json").exists()
    if mode == "failure":
        assert (retained / "results.json").read_bytes() == b'{"partial": true}\n'
        assert (retained / "results.checkpoint.json").read_bytes() == b'{"completed": []}\n'
        with pytest.raises(evaluator.EvaluationError):
            evaluator.evaluate_round(source, candidates, output, provider="chosen-provider", model="chosen-model")
        assert len(list(output.parent.glob(".new-round-*"))) == 2
        assert (retained / "results.json").read_bytes() == b'{"partial": true}\n'
    elif mode in {"checkpoint", "interrupted"}:
        assert (retained / "results.checkpoint.json").read_bytes() == b'{}'
        assert (retained / "results.json").read_bytes() == (source / "payload.json").read_bytes()
        assert (retained / "results.details.json").is_file()
    elif mode == "malformed":
        assert (retained / "results.json").read_bytes() == b'{bad'
    with pytest.raises(evaluator.EvaluationError, match="evaluator.json"):
        evaluator.load_evaluation(source, candidates, retained)


def test_evaluator_preserves_existing_and_concurrently_published_output(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun or competing evaluator cannot silently overwrite previously valid artifacts."""
    source, candidates, _ = external_evaluator
    output = tmp_path / "scores" / "round-1"
    evaluator.evaluate_round(source, candidates, output, provider="chosen-provider", model="chosen-model")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(evaluator.EvaluationError, match="already exists"):
        evaluator.evaluate_round(source, candidates, output, provider="chosen-provider", model="chosen-model")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    conflict = tmp_path / "scores" / "round-2"
    monkeypatch.setenv("DAYDREAM_TEST_EVAL_MODE", "concurrent-output")
    monkeypatch.setenv("DAYDREAM_TEST_OUTPUT_CONFLICT", str(conflict))
    with pytest.raises(evaluator.EvaluationError, match="appeared"):
        evaluator.evaluate_round(source, candidates, conflict, provider="chosen-provider", model="chosen-model")
    assert (conflict / "retained.txt").read_text() == "concurrent bytes"
    assert not (conflict / "results.json").exists()
    # Collide at the real install boundary after other files have been linked.
    partial = tmp_path / "scores" / "round-3"
    monkeypatch.setenv("DAYDREAM_TEST_EVAL_MODE", "success")
    real_link = os.link

    def competing_install(source: Any, destination: Any, **kwargs: Any) -> None:
        if Path(destination) == partial / "results.json":
            Path(destination).write_bytes(b"competing result bytes")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", competing_install)
    with pytest.raises(evaluator.EvaluationError, match="publication failed"):
        evaluator.evaluate_round(source, candidates, partial, provider="chosen-provider", model="chosen-model")
    assert (partial / "results.json").read_bytes() == b"competing result bytes"
    assert (partial / "results.details.json").is_file()
    assert not (partial / "evaluator.json").exists()
    with pytest.raises(evaluator.EvaluationError, match="evaluator.json"):
        evaluator.load_evaluation(source, candidates, partial)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    retained, = partial.parent.glob(".round-3-*")
    assert (retained / "results.json").read_bytes() == (source / "payload.json").read_bytes()
    assert (retained / "results.details.json").read_bytes() == (partial / "results.details.json").read_bytes()
    assert not (retained / "evaluator.json").exists()
    with pytest.raises(evaluator.EvaluationError, match="evaluator.json"):
        evaluator.load_evaluation(source, candidates, retained)


@pytest.mark.parametrize("change", [
    "duplicate", "nested-rounds", "wrong-scored-key", "wrong-judge", "wrong-hash",
    "wrong-base", "head-prefix-collision", "missing-metric", "invalid-count", "invalid-ratio",
    "nonfinite",
])
def test_evaluator_refuses_merged_rounds_or_mismatched_scoring_identities(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, change: str,
) -> None:
    """Upstream recursive merging and overlap-only scoring cannot label an incomplete round complete."""
    source, candidates, case = external_evaluator
    payload = _metrics(case)
    if change == "duplicate":
        (candidates / "duplicate.json").write_bytes((candidates / f"{case.key}.json").read_bytes())
    elif change == "nested-rounds":
        _write_json(candidates / "another-round" / "findings.json", {"pr": asdict(case)})
    elif change == "missing-metric":
        del payload["micro"]["overall"]["grounded_precision"]
    elif change in {"wrong-base", "head-prefix-collision"}:
        path = candidates / f"{case.key}.json"
        candidate = json.loads(path.read_text())
        candidate["pr"]["base" if change == "wrong-base" else "head"] = (
            "3" * 40 if change == "wrong-base" else case.head[:8] + "3" * 32
        )
        _write_json(path, candidate)
    else:
        target, key, replacement = {
            "wrong-scored-key": (payload["per_pr"][0], "pr_key", "other-pr"),
            "wrong-judge": (payload["eval_config"], "matcher_model", "other-provider/other-model"),
            "wrong-hash": (payload["eval_config"], "evaluated_prs_hash", "0" * 64),
            "invalid-count": (payload["micro"]["overall"], "candidate_count", -1),
            "invalid-ratio": (payload["micro"]["overall"], "grounded_precision", "0.5"),
            "nonfinite": (payload["macro"]["overall"], "grounded_precision", float("nan")),
        }[change]
        target[key] = replacement
    _write_json(source / "payload.json", payload)
    output = tmp_path / "new-scores"
    with pytest.raises(evaluator.EvaluationError):
        evaluator.evaluate_round(
            source, candidates, output, provider="chosen-provider", model="chosen-model",
        )
    assert not output.exists()


def test_three_round_report_averages_only_corresponding_supplied_ratios(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path,
) -> None:
    """Independent repeats retain nulls and raw counts instead of pooling or averaging metric denominators."""
    source, candidates, case = external_evaluator
    other = Case(case.repo, 8, case.base, case.head)
    _write_json(source / "corpus" / "manifest.json", [asdict(case), asdict(other)])
    _write_json(source / "golden" / f"{other.key}.json", {"pr": asdict(other), "findings": []})
    _write_json(candidates / f"{other.key}.json", {"pr": asdict(other), "findings": []})
    evaluations = []
    for number, ratio in enumerate([0.25, 0.5, 0.75]):
        result = _metrics(case)
        result["per_pr"].append(_metrics(other)["per_pr"][0])
        result["per_pr"][1]["metrics"]["overall"]["grounded_precision"] = 1.0
        result["per_pr"][0]["metrics"]["overall"]["grounded_precision"] = ratio
        if number == 1:
            result["per_pr"].reverse()
            result["macro"]["overall"]["augmented_precision"] = None
        result["pr_count"] = 2
        result["eval_config"]["evaluated_prs_hash"] = hashlib.sha256(
            "\n".join(sorted([case.key, other.key])).encode(),
        ).hexdigest()
        result["macro"]["overall"]["grounded_precision"] = ratio
        result["micro"]["overall"]["grounded_precision"] = None
        _write_json(source / "payload.json", result)
        evaluations.append(evaluator.evaluate_round(
            source, candidates, tmp_path / f"scores-{number}", provider="chosen-provider", model="chosen-model",
        ))
    averaged = evaluator.average_rounds(evaluations)
    assert averaged["round_count"] == 3 and averaged["case_keys"] == sorted([case.key, other.key])
    assert averaged["macro"]["overall"]["grounded_precision"] == {"mean": 0.5, "available_rounds": 3}
    assert averaged["macro"]["overall"]["augmented_precision"] == {"mean": 0.5, "available_rounds": 2}
    assert averaged["micro"]["overall"]["grounded_precision"] == {"mean": None, "available_rounds": 0}
    assert "candidate_count" not in averaged["macro"]["overall"]
    assert averaged["rounds"] == [evaluation["metrics"] for evaluation in evaluations]
    assert averaged["per_pr"][0]["metrics"]["overall"]["grounded_precision"] == {
        "mean": 0.5, "available_rounds": 3,
    }
    assert averaged["per_pr"][1]["metrics"]["overall"]["grounded_precision"] == {
        "mean": 1.0, "available_rounds": 3,
    }
    assert averaged["per_pr"][0]["metrics"]["by_category"]["security"]["augmented_recall"]["mean"] == 1.0


@pytest.mark.parametrize("change", ["round-count", "membership", "evaluator", "reference", "prompt"])
def test_repeatability_report_refuses_incompatible_rounds(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, change: str,
) -> None:
    """Individually admitted rounds still cannot average incompatible membership or judge configuration."""
    source, candidates, case = external_evaluator
    evaluations = []
    for number in range(2 if change == "round-count" else 3):
        model = "different" if change == "evaluator" and number == 1 else "chosen-model"
        selected = Case(case.repo, 8, case.base, case.head) if change == "membership" and number == 1 else case
        _write_json(source / "corpus" / "manifest.json", [asdict(selected)])
        _write_json(source / "golden" / f"{selected.key}.json", {"pr": asdict(selected), "findings": []})
        _write_json(candidates / f"{case.key}.json", {"pr": asdict(selected), "findings": []})
        result = _metrics(selected, model=model)
        if number == 1 and change in {"reference", "prompt"}:
            result["eval_config"]["golden_hash" if change == "reference" else "matcher_prompt_hash"] = "different"
        _write_json(source / "payload.json", result)
        evaluations.append(evaluator.evaluate_round(
            source, candidates, tmp_path / f"scores-{number}", provider="chosen-provider", model=model,
        ))
    with pytest.raises(evaluator.EvaluationError):
        evaluator.average_rounds(evaluations)


@pytest.fixture
def supplied_source() -> Path:
    configured = os.environ.get("DAYDREAM_REVIEWBENCH_SOURCE")
    if not configured:
        pytest.skip("set DAYDREAM_REVIEWBENCH_SOURCE to the pinned checkout with npm ci installed")
    source = Path(configured).resolve()
    verify_source(source)
    if not (source / "node_modules" / ".bin" / "tsx").is_file():
        pytest.fail("Run npm ci in the explicitly selected ReviewBench source checkout")
    return source


def test_supplied_validator_and_empty_real_judge_work_offline(supplied_source: Path, tmp_path: Path) -> None:
    """A supplied case completes empty validation/judging before model creation, without a checkout."""
    case = load_cases(supplied_source, "test")[0]
    pr_metadata = _write_json(tmp_path / "pr.json", asdict(case))
    candidates = tmp_path / "candidate" / "round-1" / "findings"
    candidate = _write_json(candidates / f"{case.key}.json", {
        "pr": asdict(case), "agent": "synthetic-completed-empty", "findings": [], "usage": {"time_in_ms": 12},
    })
    validation_command = ["npm", "run", "harbor:validate-findings", "--",
                          "--candidate", str(candidate), "--pr", str(pr_metadata)]
    validation = subprocess.run(validation_command, cwd=supplied_source, check=True, capture_output=True, text=True)
    assert "No judging or inference performed" in validation.stdout
    result = evaluator.evaluate_round(
        supplied_source, candidates, tmp_path / "scores", provider="not-a-provider", model="not-a-model",
    )
    metrics = result["metrics"]
    assert metrics["pr_count"] == 1
    assert metrics["micro"]["overall"]["candidate_count"] == 0
    assert metrics["micro"]["overall"]["grounded_precision"] is None
    assert metrics["micro"]["overall"]["grounded_recall"] == 0
    assert metrics["candidate_usage"]["total"]["time_in_ms"] == 12
    assert metrics["candidate_usage"]["total"]["completion_tokens"] is None
    assert metrics["eval_config"]["matcher_model"] == "not-a-provider/not-a-model"
    golden_count = metrics["micro"]["overall"]["golden_tp_count"]
    assert golden_count > 0
    assert f"0c, {golden_count}g_tp" in evaluator.render_report(supplied_source, tmp_path / "scores" / "results.json")
    nonempty = _write_json(tmp_path / "nonempty-metrics.json", _metrics(case))
    summary = evaluator.render_report(supplied_source, nonempty)
    assert "Grounded:  P=50.0%  R=100.0%" in summary
    assert "By Severity (macro)" in summary and "By Category (macro)" in summary
    candidate.write_text("{bad", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(validation_command, cwd=supplied_source, check=True, capture_output=True, text=True)

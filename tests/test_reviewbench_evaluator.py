"""Supplied evaluator integration, publication safety and independent-round reports."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.reviewbench import evaluator
from daydream.benchmark.reviewbench.contract import Case, verify_source
from tests.harness.git_helpers import git, init_repo, seeded_commit, write_and_stage


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
        "if mode == 'failure':\n"
        "    print('synthetic evaluator process failed', file=sys.stderr)\n"
        "    raise SystemExit(9)\n"
        "out = pathlib.Path(args[args.index('--output') + 1])\n"
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
        source, candidates, output, provider="chosen-provider", model="chosen-model", expected_keys=[case.key],
    )
    assert evaluator.load_evaluation(source, candidates, output) == result
    assert result["metrics"] == _metrics(case)
    assert result["evaluator"]["workflow"] == "supplied-standalone"
    assert result["evaluator"]["case_keys"] == [case.key]
    assert result["evaluator"]["candidate_files"] == {
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
    "source", "membership", "model", "fingerprints", "swapped-round", "checkpoint",
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
    elif fault == "swapped-round":
        other_candidates = tmp_path / "other-candidate" / "round-1" / "findings"
        shutil.copytree(candidates, other_candidates)
        other_output = tmp_path / "scores" / "other-round"
        evaluator.evaluate_round(
            source, other_candidates, other_output, provider="chosen-provider", model="chosen-model",
        )
        for filename in ("evaluator.json", "results.json", "results.details.json"):
            shutil.copyfile(other_output / filename, output / filename)
    else:
        target, key, replacement = {
            "source": (provenance["source"], "revision", "different-source"),
            "membership": (provenance, "case_keys", []),
            "model": (provenance, "provider", "another-provider"),
            "fingerprints": (provenance["eval_config"], "golden_hash", "different-reference"),
        }[fault]
        target[key] = replacement
        _write_json(output / "evaluator.json", provenance)
    with pytest.raises(evaluator.EvaluationError):
        evaluator.load_evaluation(source, candidates, output)


@pytest.mark.parametrize("mode", [
    "failure", "missing", "malformed", "missing-details", "checkpoint", "mutated-candidate",
    "mutated-reference", "mutated-manifest",
])
def test_failed_or_incomplete_evaluator_cannot_publish(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """Native reports must not consume missing/malformed/partial output as a successful round."""
    source, candidates, _ = external_evaluator
    output = tmp_path / "scores" / "new-round"
    monkeypatch.setenv("DAYDREAM_TEST_EVAL_MODE", mode)
    with pytest.raises(evaluator.EvaluationError):
        evaluator.evaluate_round(source, candidates, output, provider="chosen-provider", model="chosen-model")
    assert not output.exists()
    assert not list(output.parent.glob(".new-round-*"))


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
    assert not list(partial.parent.glob(".round-3-*"))


@pytest.mark.parametrize("change", [
    "missing-case", "duplicate", "nested-rounds", "wrong-scored-key", "wrong-judge", "wrong-hash",
    "wrong-base", "head-prefix-collision", "missing-metric", "invalid-count", "invalid-ratio",
])
def test_evaluator_refuses_merged_rounds_or_mismatched_scoring_identities(
    external_evaluator: tuple[Path, Path, Case], tmp_path: Path, change: str,
) -> None:
    """Upstream recursive merging and overlap-only scoring cannot label an incomplete round complete."""
    source, candidates, case = external_evaluator
    expected = [case.key]
    payload = _metrics(case)
    if change == "missing-case":
        expected.append("missing")
    elif change == "duplicate":
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
        }[change]
        target[key] = replacement
    _write_json(source / "payload.json", payload)
    output = tmp_path / "new-scores"
    with pytest.raises(evaluator.EvaluationError):
        evaluator.evaluate_round(
            source, candidates, output, provider="chosen-provider", model="chosen-model", expected_keys=expected,
        )
    assert not output.exists()


def test_three_round_report_averages_only_corresponding_supplied_ratios() -> None:
    """Independent repeats retain nulls and raw counts instead of pooling or averaging metric denominators."""
    case = Case("https://github.com/example/repo", 7, "1" * 40, "2" * 40)
    rounds = [_metrics(case) for _ in range(3)]
    for result, ratio in zip(rounds, [0.25, 0.5, 0.75], strict=True):
        result["macro"]["overall"]["grounded_precision"] = ratio
        result["micro"]["overall"]["grounded_precision"] = None
    rounds[1]["macro"]["overall"]["augmented_precision"] = None
    averaged = evaluator.average_rounds(rounds)
    assert averaged["round_count"] == 3 and averaged["case_keys"] == [case.key]
    assert averaged["macro"]["overall"]["grounded_precision"] == {"mean": 0.5, "available_rounds": 3}
    assert averaged["macro"]["overall"]["augmented_precision"] == {"mean": 0.5, "available_rounds": 2}
    assert averaged["micro"]["overall"]["grounded_precision"] == {"mean": None, "available_rounds": 0}
    assert "candidate_count" not in averaged["macro"]["overall"]
    assert averaged["rounds"] == rounds
    assert averaged["per_pr"][0]["metrics"]["by_category"]["security"]["augmented_recall"]["mean"] == 1.0


@pytest.mark.parametrize("change", ["round-count", "membership", "evaluator", "reference", "prompt", "nonfinite"])
def test_repeatability_report_refuses_incompatible_rounds(change: str) -> None:
    """Averages must not hide a changed judge/reference/prompt, missing case or invalid score."""
    case = Case("https://github.com/example/repo", 7, "1" * 40, "2" * 40)
    rounds = [_metrics(case) for _ in range(3)]
    if change == "round-count":
        rounds.pop()
    elif change == "membership":
        rounds[1]["per_pr"][0]["pr_key"] = "missing-original-pr"
    elif change == "nonfinite":
        rounds[1]["macro"]["overall"]["grounded_precision"] = float("nan")
    else:
        key = {"evaluator": "matcher_model", "reference": "golden_hash", "prompt": "matcher_prompt_hash"}[change]
        rounds[1]["eval_config"][key] = "different"
    with pytest.raises(evaluator.EvaluationError):
        evaluator.average_rounds(rounds)


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
    """A real clean Git case proves empty success through the supplied parser/judge without any valid model."""
    repo = tmp_path / "snapshot"
    init_repo(repo)
    write_and_stage(repo, "main.py", "VALUE = 1\n")
    base = seeded_commit(repo, "base")
    write_and_stage(repo, "main.py", "VALUE = 2\n")
    head = seeded_commit(repo, "head")
    case = Case("https://github.com/example/repo", 7, base, head)
    pr_metadata = _write_json(tmp_path / "pr.json", asdict(case))
    candidates = tmp_path / "candidate" / "round-1" / "findings"
    candidate = _write_json(candidates / f"{case.key}.json", {
        "pr": asdict(case), "agent": "synthetic-completed-empty", "findings": [], "usage": {"time_in_ms": 12},
    })
    manifest = _write_json(tmp_path / "manifest.json", [asdict(case)])
    golden = tmp_path / "golden"
    _write_json(golden / f"{case.key}.json", {
        "pr_key": case.key, "pr": asdict(case),
        "findings": [{**_finding(), "tp_fp": "tp", "severity": "high", "category": "security"}],
    })
    assert "No judging or inference performed" in evaluator.validate_findings(supplied_source, candidate, pr_metadata)
    result = evaluator.evaluate_round(
        supplied_source, candidates, tmp_path / "scores", provider="not-a-provider", model="not-a-model",
        expected_keys=[case.key], golden_dir=golden, manifest=manifest, snapshot_dir=repo,
    )
    metrics = result["metrics"]
    assert metrics["pr_count"] == 1
    assert metrics["micro"]["overall"]["candidate_count"] == 0
    assert metrics["micro"]["overall"]["grounded_precision"] is None
    assert metrics["micro"]["overall"]["grounded_recall"] == 0
    assert metrics["candidate_usage"]["total"]["time_in_ms"] == 12
    assert metrics["candidate_usage"]["total"]["completion_tokens"] is None
    assert metrics["eval_config"]["matcher_model"] == "not-a-provider/not-a-model"
    assert "0c, 1g_tp" in evaluator.render_report(supplied_source, tmp_path / "scores" / "results.json")
    assert not git(repo, "remote") and not git(repo, "status", "--porcelain", "--ignored")
    nonempty = _write_json(tmp_path / "nonempty-metrics.json", _metrics(case))
    summary = evaluator.render_report(supplied_source, nonempty)
    assert "Grounded:  P=50.0%  R=100.0%" in summary
    assert "By Severity (macro)" in summary and "By Category (macro)" in summary
    candidate.write_text("{bad", encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError):
        evaluator.validate_findings(supplied_source, candidate, pr_metadata)

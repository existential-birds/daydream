"""Invoke the pinned supplied evaluator without owning its metrics or judge policy.

Each invocation consumes one candidate round and publishes a new output directory.
The standalone judge is an explicit, potentially chargeable operation. Validation
and summary rendering do not invoke a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from daydream.json_utils import publish_staged_directory

from .contract import Case, file_digest, verify_source

_RATIOS = (
    "grounded_precision", "grounded_recall", "augmented_precision", "augmented_recall", "novel_tp_yield",
)
_COUNTS = (
    "novel_tp_count", "candidate_count", "golden_tp_count", "matched_count", "matched_tp_count",
    "unmatched_count", "unmatched_tp_count", "unmatched_fp_count", "golden_covered_count",
)


class EvaluationError(ValueError):
    """The supplied evaluator did not produce a complete, verified result."""


def _run(command: list[str], source: Path, *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    # This destructive upstream switch belongs to its disposable Harbor verifier,
    # never the operator's standalone reference checkout.
    environment.pop("RB_DELETE_GOLDEN_AFTER_LOAD", None)
    try:
        result = subprocess.run(
            command, cwd=source, capture_output=True, text=True, timeout=timeout, check=False, env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluationError(f"Unable to run supplied ReviewBench process: {exc}") from exc
    if result.returncode:
        evidence = (result.stderr or result.stdout).strip()[-4000:]
        raise EvaluationError(f"Supplied ReviewBench process exited {result.returncode}: {evidence}")
    return result


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Missing or malformed ReviewBench artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"ReviewBench artifact must be an object: {path.name}")
    return value


def _candidate_membership(candidate_dir: Path) -> tuple[dict[str, Case], dict[str, str]]:
    if not candidate_dir.is_dir():
        raise EvaluationError("Candidate input must be one round's findings directory")
    paths = sorted(candidate_dir.rglob("*.json"))
    if not paths or any(path.parent != candidate_dir for path in paths):
        raise EvaluationError("Candidate input must contain findings directly, without nested rounds")
    cases: dict[str, Case] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        raw = _object(path)
        try:
            case = Case.from_dict(raw["pr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"Invalid frozen PR identity in {path.name}") from exc
        if case.key in cases:
            raise EvaluationError(f"Duplicate candidate PR identity: {case.key}")
        cases[case.key] = case
        hashes[path.name] = file_digest(path)
    return cases, hashes


def _reference_inputs(manifest: Path, golden_dir: Path, cases: dict[str, Case]) -> dict[str, str]:
    try:
        raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, list):
            raise ValueError("manifest must be an array")
        supplied = [Case.from_dict(row) for row in raw_manifest]
        by_key = {case.key: case for case in supplied}
        if len(by_key) != len(supplied):
            raise ValueError("manifest has duplicate or colliding identities")
        for key, case in cases.items():
            if by_key.get(key) != case:
                raise ValueError(f"candidate {key} differs from the full manifest identity")
            golden = _object(golden_dir / f"{key}.json")
            if Case.from_dict(golden["pr"]) != case:
                raise ValueError(f"candidate {key} differs from the full golden identity")
        return {key: file_digest(golden_dir / f"{key}.json") for key in sorted(cases)}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"Invalid supplied evaluator reference inputs: {exc}") from exc


def _snapshot(source: Path, candidate_dir: Path) -> dict[str, Any]:
    """Validate and bind every input to the fixed supplied standalone evaluator."""
    source_identity = verify_source(source)
    cases, hashes = _candidate_membership(candidate_dir)
    manifest = source / "corpus" / "manifest.json"
    return {
        "source": source_identity, "cases": {key: asdict(case) for key, case in sorted(cases.items())},
        "candidate_files": hashes, "candidate_directory": str(candidate_dir),
        "reference_files": _reference_inputs(manifest, source / "golden", cases),
        "manifest_sha256": file_digest(manifest),
    }


def _result_keys(metrics: dict[str, Any]) -> list[str]:
    per_pr = metrics.get("per_pr")
    if not isinstance(per_pr, list) or not all(isinstance(row, dict) for row in per_pr):
        raise EvaluationError("Supplied evaluator omitted per-PR results")
    keys = [row.get("pr_key") for row in per_pr]
    if not all(isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
        raise EvaluationError("Supplied evaluator returned invalid or duplicate PR keys")
    if type(metrics.get("pr_count")) is not int or metrics["pr_count"] != len(keys):
        raise EvaluationError("Supplied evaluator PR count does not match its results")
    return sorted(keys)


def _validate_stratified(strata: Any) -> None:
    if not isinstance(strata, dict) or not all(
        isinstance(strata.get(field), dict) for field in ("overall", "by_severity", "by_category")
    ):
        raise EvaluationError("Supplied evaluator omitted stratified metrics")
    for metric in [strata["overall"], *strata["by_severity"].values(), *strata["by_category"].values()]:
        if not isinstance(metric, dict) or any(key not in metric for key in (*_RATIOS, *_COUNTS)):
            raise EvaluationError("Supplied evaluator omitted metric fields")
        if any(type(metric[key]) is not int or metric[key] < 0 for key in _COUNTS):
            raise EvaluationError("Supplied evaluator returned invalid metric counts")
        if any(metric[key] is not None and (
            type(metric[key]) not in (int, float) or not math.isfinite(metric[key]) or not 0 <= metric[key] <= 1
        ) for key in _RATIOS):
            raise EvaluationError("Supplied evaluator returned invalid metric ratios")


def _validate_metrics(metrics: dict[str, Any], keys: Sequence[str], *, provider: str, model: str) -> None:
    if _result_keys(metrics) != sorted(keys):
        raise EvaluationError("Supplied evaluator did not score exactly the candidate round membership")
    for scope in ("macro", "micro"):
        _validate_stratified(metrics.get(scope))
    for row in metrics["per_pr"]:
        _validate_stratified(row.get("metrics"))
    config = metrics.get("eval_config")
    required = ("golden_hash", "classifier_prompt_hash", "matcher_prompt_hash", "evaluated_prs_hash")
    if not isinstance(config, dict) or any(not isinstance(config.get(key), str) or not config[key] for key in required):
        raise EvaluationError("Supplied evaluator omitted reference or prompt fingerprints")
    if any(config.get(key) != f"{provider}/{model}" for key in ("classifier_model", "matcher_model")):
        raise EvaluationError("Supplied evaluator model identity differs from the selected standalone judge")
    expected_hash = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    if config["evaluated_prs_hash"] != expected_hash:
        raise EvaluationError("Supplied evaluator membership fingerprint differs from the candidate round")


def _validate_details(path: Path, keys: Sequence[str]) -> None:
    try:
        details = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError("Supplied evaluator omitted valid finding details") from exc
    if not isinstance(details, list) or not all(isinstance(row, dict) for row in details):
        raise EvaluationError("Supplied evaluator finding details must be an array of PR objects")
    if sorted(row.get("pr_key", "") for row in details) != sorted(keys):
        raise EvaluationError("Supplied evaluator details differ from the candidate round membership")


def load_evaluation(source: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Read a round only when its evidence still binds these candidates and source.

    A standalone metrics file is not evidence that the current round was scored.
    Stored hashes protect results, finding details, candidates and references;
    directory bindings prevent swapping equal-looking rounds or candidates.
    """
    source, candidate_dir, output_dir = source.resolve(), candidate_dir.resolve(), output_dir.resolve()
    provenance = _object(output_dir / "evaluator.json")
    if provenance.get("schema_version") != 2:
        raise EvaluationError("Unsupported ReviewBench evaluation; create a fresh evaluation directory")
    snapshot = _snapshot(source, candidate_dir)
    keys = sorted(snapshot["cases"])
    if (provenance.get("workflow") != "supplied-standalone" or provenance.get("snapshot") != snapshot
            or provenance.get("output_directory") != str(output_dir)):
        raise EvaluationError("Stored evaluation provenance differs from the current candidate round or source")
    provider, model = provenance.get("provider"), provenance.get("model")
    if not isinstance(provider, str) or not provider.strip() or not isinstance(model, str) or not model.strip():
        raise EvaluationError("Stored evaluation omitted its selected provider or model")
    output, details = output_dir / "results.json", output_dir / "results.details.json"
    try:
        if (provenance.get("results_sha256") != file_digest(output)
                or provenance.get("details_sha256") != file_digest(details)):
            raise EvaluationError("Stored evaluator results or details differ from their recorded hashes")
    except OSError as exc:
        raise EvaluationError("Stored evaluation evidence is missing or malformed") from exc
    metrics = _object(output)
    _validate_metrics(metrics, keys, provider=provider, model=model)
    _validate_details(details, keys)
    if (output_dir / "results.checkpoint.json").exists():
        raise EvaluationError("Stored evaluator has an incomplete checkpoint")
    return {"metrics": metrics, "evaluator": provenance, "output_dir": str(output_dir)}


def evaluate_round(
    source: Path,
    candidate_dir: Path,
    output_dir: Path,
    *,
    provider: str,
    model: str,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run the explicitly selected supplied standalone judge for one isolated round.

    Caller-owned generation accounting remains independent: this evaluator scores
    only produced candidates.
    Existing output directories, including previous successes, are never reused.
    An install failure can leave an incomplete directory without evaluator.json;
    preserve it and select a fresh output directory before retrying.
    """
    source, candidate_dir, output_dir = source.resolve(), candidate_dir.resolve(), output_dir.resolve()
    if not provider.strip() or not model.strip():
        raise EvaluationError("An explicit standalone evaluator provider and model are required")
    if type(concurrency) is not int or concurrency < 1:
        raise EvaluationError("Evaluator concurrency must be a positive integer")
    if output_dir.exists():
        raise EvaluationError("Evaluator output directory already exists; select a new candidate/round output")
    snapshot = _snapshot(source, candidate_dir)
    keys = sorted(snapshot["cases"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        output = staged / "results.json"
        command = [
            "npm", "run", "judge", "--", "--candidate", str(candidate_dir), "--golden", str(source / "golden"),
            "--manifest", str(source / "corpus" / "manifest.json"), "--provider", provider, "--model", model,
            "--output", str(output), "--concurrency", str(concurrency), "--strict", "--allow-empty",
        ]
        result = _run(command, source)
        metrics = _object(output)
        _validate_metrics(metrics, keys, provider=provider, model=model)
        details_path = staged / "results.details.json"
        _validate_details(details_path, keys)
        if (staged / "results.checkpoint.json").exists():
            raise EvaluationError("Supplied evaluator retained an incomplete checkpoint")
        if _snapshot(source, candidate_dir) != snapshot:
            raise EvaluationError("Candidate findings, supplied source or reference inputs changed during evaluation")
        provenance = {
            "schema_version": 2, "workflow": "supplied-standalone", "provider": provider, "model": model,
            "snapshot": snapshot, "output_directory": str(output_dir), "concurrency": concurrency,
            "results_sha256": file_digest(output), "details_sha256": file_digest(details_path),
        }
        (staged / "evaluator.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        (staged / "evaluator.stdout.log").write_text(result.stdout, encoding="utf-8")
        (staged / "evaluator.stderr.log").write_text(result.stderr, encoding="utf-8")
        if output_dir.exists():
            raise EvaluationError("Evaluator output appeared during evaluation; preserving it")
        try:
            publish_staged_directory(staged, output_dir, commit_last="evaluator.json")
        except OSError as exc:
            raise EvaluationError(
                "Evaluation publication failed; preserve incomplete artifacts and select a fresh output directory"
            ) from exc
        return {"metrics": metrics, "evaluator": provenance, "output_dir": str(output_dir)}
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def render_report(source: Path, metrics_file: Path) -> str:
    """Render existing supplied metrics with its own printSummary, without judging."""
    source = source.resolve()
    verify_source(source)
    _object(metrics_file)
    return _run([
        str(source / "node_modules" / ".bin" / "tsx"), str(Path(__file__).with_name("report.ts")),
        str(source), str(metrics_file.resolve()),
    ], source, timeout=120).stdout


def _ratio_means(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    means: dict[str, Any] = {}
    for key in _RATIOS:
        values = [row.get(key) for row in rows]
        available = [value for value in values if value is not None]
        means[key] = {
            "mean": sum(available) / len(available) if available else None, "available_rounds": len(available),
        }
    return means


def _stratified_means(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": _ratio_means([row["overall"] for row in rows])}
    for group in ("by_severity", "by_category"):
        labels = sorted(set().union(*(row[group] for row in rows)))
        result[group] = {label: _ratio_means([row[group].get(label, {}) for row in rows]) for label in labels}
    return result


def average_rounds(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average metrics admitted by evaluate_round/load_evaluation for three complete rounds.

    Counts and usage remain in each raw round. Null scores are unavailable, never
    zero; each mean records how many rounds measured that ratio. The caller owns
    proof that these three isolated artifacts cover the entire selected set.
    """
    if len(evaluations) != 3:
        raise EvaluationError("Repeatability reporting requires exactly three independent rounds")
    rounds = [evaluation["metrics"] for evaluation in evaluations]
    keys = sorted(row["pr_key"] for row in rounds[0]["per_pr"])
    config = rounds[0]["eval_config"]
    if any(
        sorted(row["pr_key"] for row in result["per_pr"]) != keys or result["eval_config"] != config
        for result in rounds
    ):
        raise EvaluationError("Round membership, reference, prompts and evaluator identities must agree")
    per_round = [{row["pr_key"]: row["metrics"] for row in result["per_pr"]} for result in rounds]
    return {
        "round_count": 3, "case_keys": keys, "eval_config": config,
        "macro": _stratified_means([row["macro"] for row in rounds]),
        "micro": _stratified_means([row["micro"] for row in rounds]),
        "per_pr": [
            {"pr_key": key, "metrics": _stratified_means([row[key] for row in per_round])} for key in keys
        ],
        "rounds": rounds,
    }

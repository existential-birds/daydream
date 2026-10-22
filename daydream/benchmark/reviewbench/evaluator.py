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
    source_identity = verify_source(source)
    provenance = _object(output_dir / "evaluator.json")
    cases, candidate_hashes = _candidate_membership(candidate_dir)
    keys = sorted(cases)
    if (provenance.get("workflow") != "supplied-standalone" or provenance.get("source") != source_identity
            or provenance.get("case_keys") != keys or provenance.get("candidate_files") != candidate_hashes
            or provenance.get("candidate_directory") != str(candidate_dir)
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
        manifest, golden_dir = Path(provenance["manifest"]), Path(provenance["golden_directory"])
        if (provenance.get("manifest_sha256") != file_digest(manifest)
                or provenance.get("reference_files") != _reference_inputs(manifest, golden_dir, cases)):
            raise EvaluationError("Stored evaluation reference inputs differ from their recorded hashes")
    except (OSError, KeyError, TypeError) as exc:
        raise EvaluationError("Stored evaluation evidence is missing or malformed") from exc
    metrics = _object(output)
    _validate_metrics(metrics, keys, provider=provider, model=model)
    _validate_details(details, keys)
    if provenance.get("eval_config") != metrics["eval_config"]:
        raise EvaluationError("Stored evaluator fingerprints differ from their paired metrics")
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
    expected_keys: Sequence[str] | None = None,
    golden_dir: Path | None = None,
    manifest: Path | None = None,
    snapshot_dir: Path | None = None,
    repo_dir: Path | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run the explicitly selected supplied standalone judge for one isolated round.

    Caller-owned generation accounting remains independent: this evaluator scores
    only produced candidates. ``expected_keys`` can require complete membership.
    Existing output directories, including previous successes, are never reused.
    An install failure can leave an incomplete directory without evaluator.json;
    preserve it and select a fresh output directory before retrying.
    """
    source, candidate_dir, output_dir = source.resolve(), candidate_dir.resolve(), output_dir.resolve()
    source_identity = verify_source(source)
    if not provider.strip() or not model.strip():
        raise EvaluationError("An explicit standalone evaluator provider and model are required")
    if type(concurrency) is not int or concurrency < 1:
        raise EvaluationError("Evaluator concurrency must be a positive integer")
    if output_dir.exists():
        raise EvaluationError("Evaluator output directory already exists; select a new candidate/round output")
    cases, candidate_hashes = _candidate_membership(candidate_dir)
    keys = sorted(cases)
    if expected_keys is not None and (len(set(expected_keys)) != len(expected_keys) or sorted(expected_keys) != keys):
        raise EvaluationError("Produced candidates differ from expected round membership; refusing a partial score")
    golden_dir = (golden_dir or source / "golden").resolve()
    manifest = (manifest or source / "corpus" / "manifest.json").resolve()
    if not golden_dir.is_dir() or not manifest.is_file():
        raise EvaluationError("The selected reference directory and corpus manifest must exist")
    reference_hashes = _reference_inputs(manifest, golden_dir, cases)
    manifest_hash = file_digest(manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        output = staged / "results.json"
        command = [
            "npm", "run", "judge", "--", "--candidate", str(candidate_dir), "--golden", str(golden_dir),
            "--manifest", str(manifest), "--provider", provider, "--model", model,
            "--output", str(output), "--concurrency", str(concurrency), "--strict", "--allow-empty",
        ]
        if snapshot_dir is not None:
            command.extend(["--snapshot-dir", str(snapshot_dir.resolve())])
        if repo_dir is not None:
            command.extend(["--repo-dir", str(repo_dir.resolve())])
        result = _run(command, source)
        metrics = _object(output)
        _validate_metrics(metrics, keys, provider=provider, model=model)
        details_path = staged / "results.details.json"
        _validate_details(details_path, keys)
        if (staged / "results.checkpoint.json").exists():
            raise EvaluationError("Supplied evaluator retained an incomplete checkpoint")
        if _candidate_membership(candidate_dir) != (cases, candidate_hashes):
            raise EvaluationError("Candidate findings changed while the evaluator was running")
        if (file_digest(manifest) != manifest_hash or _reference_inputs(manifest, golden_dir, cases) != reference_hashes
                or verify_source(source) != source_identity):
            raise EvaluationError("Supplied source or reference inputs changed during evaluation")
        provenance = {
            "workflow": "supplied-standalone", "provider": provider, "model": model,
            "source": source_identity, "candidate_files": candidate_hashes, "case_keys": keys,
            "candidate_directory": str(candidate_dir), "output_directory": str(output_dir),
            "manifest": str(manifest), "golden_directory": str(golden_dir),
            "manifest_sha256": manifest_hash, "reference_files": reference_hashes,
            "eval_config": metrics["eval_config"],
            "results_sha256": file_digest(output), "details_sha256": file_digest(details_path), "command": command,
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
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in values):
            raise EvaluationError(f"Invalid supplied score: {key}")
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


def average_rounds(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average corresponding supplied ratios across three independent complete rounds.

    Counts and usage remain in each raw round. Null scores are unavailable, never
    zero; each mean records how many rounds measured that ratio. The caller owns
    proof that these three isolated artifacts cover the entire selected set.
    """
    if len(rounds) != 3:
        raise EvaluationError("Repeatability reporting requires exactly three independent rounds")
    keys = _result_keys(rounds[0])
    config = rounds[0].get("eval_config")
    if not isinstance(config, dict) or any(
        _result_keys(row) != keys or row.get("eval_config") != config for row in rounds
    ):
        raise EvaluationError("Round membership, reference, prompts and evaluator identities must agree")
    try:
        for result in rounds:
            _validate_stratified(result["macro"])
            _validate_stratified(result["micro"])
            for row in result["per_pr"]:
                _validate_stratified(row["metrics"])
        per_round = [{row["pr_key"]: row["metrics"] for row in result["per_pr"]} for result in rounds]
        return {
            "round_count": 3, "case_keys": keys, "eval_config": config,
            "macro": _stratified_means([row["macro"] for row in rounds]),
            "micro": _stratified_means([row["micro"] for row in rounds]),
            "per_pr": [
                {"pr_key": key, "metrics": _stratified_means([row[key] for row in per_round])} for key in keys
            ],
            "rounds": list(rounds),
        }
    except (KeyError, TypeError) as exc:
        raise EvaluationError("Malformed supplied round metrics") from exc
